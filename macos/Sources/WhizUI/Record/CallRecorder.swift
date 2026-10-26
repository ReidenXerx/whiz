import AVFoundation
import CoreMedia
import os
import ScreenCaptureKit

/// What a recording takes in.
struct RecordingSources: Equatable, Sendable {
    /// The microphone's `AVCaptureDevice.uniqueID`; nil records no microphone.
    var microphone: String?
    /// Everything the Mac plays except Whiz itself: the other side of a call.
    var systemAudio: Bool
    /// The display to record as video; nil records no picture.
    var display: CGDirectDisplayID?

    /// ScreenCaptureKit (and the Screen Recording permission) is needed for
    /// the call's audio and for the screen, not for the microphone alone.
    var needsScreenCapture: Bool { systemAudio || display != nil }
    var hasAudio: Bool { microphone != nil || systemAudio }
}

/// One recording, from `start` to `stop`.
///
/// The microphone comes from AVCaptureSession, the call's audio and the
/// screen from ScreenCaptureKit (whose own microphone capture needs macOS
/// 15; Whiz runs on 13). Every buffer from both lands on one serial queue,
/// which is the only place the writer is touched.
final class CallRecorder: NSObject, @unchecked Sendable {
    let url: URL
    let sources: RecordingSources

    /// Recording stopped without being asked: the permission was taken away,
    /// the display went, the disk filled. Called once, on the capture queue;
    /// the file still has to be finished with `stop()`.
    var onInterruption: (@Sendable (String) -> Void)?

    struct Meter: Sendable, Equatable {
        var microphone: Float = 0
        var system: Float = 0
        /// Something worth telling the person without stopping: the
        /// microphone was unplugged, a source sends audio Whiz cannot use.
        var warning: String?
    }

    private let queue = DispatchQueue(label: "dev.reidenxerx.whiz.recorder", qos: .userInitiated)
    private let meter = OSAllocatedUnfairLock(initialState: Meter())
    // Confined to `queue` once recording starts.
    private var writer: RecordingWriter?
    private var stopped = false
    private var interrupted = false
    private var stream: SCStream?
    private var session: AVCaptureSession?
    private var microphoneClock: CMClock?
    private var observers: [NSObjectProtocol] = []
    /// Host time of the recording's first frame.
    private(set) var origin: CMTime = .invalid

    init(url: URL, sources: RecordingSources) {
        self.url = url
        self.sources = sources
    }

    /// The latest peak levels (reset on read, so a meter falls back when the
    /// room goes quiet) and any warning.
    func readMeter() -> Meter {
        meter.withLock { state in
            let reading = state
            state.microphone = 0
            state.system = 0
            return reading
        }
    }

    // MARK: - Start and stop

    /// Start every source. Throws, leaving nothing behind, when one cannot.
    func start() async throws {
        guard sources.hasAudio else { throw RecordingError("choose a microphone or the call's audio to record") }

        var screen: (stream: SCStream, video: RecordingWriter.Video?)?
        if sources.needsScreenCapture {
            screen = try await makeStream()
        }
        let capture = try sources.microphone.map(makeMicrophoneSession)

        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let writer = try RecordingWriter(url: url, video: screen?.video)
        let origin = CMClockGetTime(CMClockGetHostTimeClock())
        self.origin = origin
        try queue.sync {
            try writer.start(at: origin)
            self.writer = writer
        }

        if let capture {
            session = capture
            microphoneClock = capture.synchronizationClock
            observeMicrophone(capture)
            capture.startRunning()
            guard capture.isRunning else {
                await discard()
                throw RecordingError("the microphone did not start; another app may be using it exclusively")
            }
        }
        if let screen {
            stream = screen.stream
            do {
                try await screen.stream.startCapture()
            } catch {
                await discard()
                throw RecordingError(Self.describe(error))
            }
        }
    }

    /// Stop every source and finish the file.
    func stop() async throws {
        if let stream { try? await stream.stopCapture() }
        session?.stopRunning()
        removeObservers()
        let writer: RecordingWriter? = queue.sync {
            stopped = true
            return self.writer
        }
        guard let writer else { return }
        // Nothing reaches the writer once `stopped` is set.
        try await writer.finish()
    }

    private func discard() async {
        if let stream { try? await stream.stopCapture() }
        session?.stopRunning()
        removeObservers()
        let writer: RecordingWriter? = queue.sync {
            stopped = true
            return self.writer
        }
        await writer?.abandon()
        try? FileManager.default.removeItem(at: url)
    }

    // MARK: - Sources

    private func makeStream() async throws -> (SCStream, RecordingWriter.Video?) {
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        } catch {
            throw RecordingError(Self.describe(error))
        }
        let wanted = sources.display ?? CGMainDisplayID()
        guard let display = content.displays.first(where: { $0.displayID == wanted }) ?? content.displays.first else {
            throw RecordingError("no display to record from")
        }
        // Whiz's own windows and sounds stay out of the recording.
        let ownApp = content.applications.filter { $0.processID == ProcessInfo.processInfo.processIdentifier }
        let filter = SCContentFilter(display: display, excludingApplications: ownApp, exceptingWindows: [])

        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = sources.systemAudio
        configuration.excludesCurrentProcessAudio = true
        configuration.sampleRate = AudioMixer.sampleRate
        configuration.channelCount = 1

        var video: RecordingWriter.Video?
        if sources.display != nil {
            let size = Self.videoSize(pixelWidth: Self.pixelWidth(of: display), pixelHeight: Self.pixelHeight(of: display))
            video = RecordingWriter.Video(width: size.width, height: size.height, frameRate: 15)
            configuration.width = size.width
            configuration.height = size.height
            configuration.minimumFrameInterval = CMTime(value: 1, timescale: 15)
            configuration.pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            configuration.showsCursor = true
            configuration.queueDepth = 6
        } else {
            // Audio only. ScreenCaptureKit still captures a picture; keep it
            // tiny and rare, and throw it away.
            configuration.width = 16
            configuration.height = 16
            configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        }

        let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
        do {
            try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
            if sources.systemAudio {
                try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
            }
        } catch {
            throw RecordingError(Self.describe(error))
        }
        return (stream, video)
    }

    private func makeMicrophoneSession(_ id: String) throws -> AVCaptureSession {
        guard let device = AVCaptureDevice(uniqueID: id) else {
            throw RecordingError("the microphone is not connected; choose another one")
        }
        let session = AVCaptureSession()
        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: device)
        } catch {
            throw RecordingError("cannot use \(device.localizedName): \(error.localizedDescription)")
        }
        let output = AVCaptureAudioDataOutput()
        output.audioSettings = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: AudioMixer.sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        output.setSampleBufferDelegate(self, queue: queue)
        session.beginConfiguration()
        guard session.canAddInput(input), session.canAddOutput(output) else {
            session.commitConfiguration()
            throw RecordingError("cannot record from \(device.localizedName)")
        }
        session.addInput(input)
        session.addOutput(output)
        session.commitConfiguration()
        return session
    }

    private func observeMicrophone(_ session: AVCaptureSession) {
        let center = NotificationCenter.default
        let warn: @Sendable (Notification) -> Void = { [weak self] _ in
            self?.meter.withLock {
                $0.warning = "The microphone stopped. Recording goes on with the call's audio only."
            }
        }
        observers.append(center.addObserver(forName: .AVCaptureSessionRuntimeError, object: session, queue: nil, using: warn))
        observers.append(center.addObserver(forName: .AVCaptureSessionWasInterrupted, object: session, queue: nil, using: warn))
    }

    private func removeObservers() {
        observers.forEach(NotificationCenter.default.removeObserver)
        observers = []
    }

    // MARK: - Sizes

    /// The display's own pixels, at most 1920 wide, in even numbers (H.264
    /// needs them even).
    static func videoSize(pixelWidth: Int, pixelHeight: Int) -> (width: Int, height: Int) {
        let width = max(min(pixelWidth, 1920), 2)
        let height = max(Int((Double(pixelHeight) * Double(width) / Double(max(pixelWidth, 1))).rounded()), 2)
        return (width & ~1, height & ~1)
    }

    private static func pixelWidth(of display: SCDisplay) -> Int {
        CGDisplayCopyDisplayMode(display.displayID)?.pixelWidth ?? display.width
    }

    private static func pixelHeight(of display: SCDisplay) -> Int {
        CGDisplayCopyDisplayMode(display.displayID)?.pixelHeight ?? display.height
    }

    /// ScreenCaptureKit's errors, worded for the person recording.
    static func describe(_ error: Error) -> String {
        let nsError = error as NSError
        if nsError.domain == SCStreamErrorDomain {
            switch SCStreamError.Code(rawValue: nsError.code) {
            case .userDeclined:
                return "Whiz is not allowed to record the screen and system audio. Allow it in System Settings → Privacy & Security → Screen & System Audio Recording, then quit and reopen Whiz."
            case .userStopped:
                return "Recording was stopped from the menu bar."
            case .noDisplayList, .missingEntitlements:
                return "Whiz is not allowed to record the screen and system audio. Allow it in System Settings → Privacy & Security → Screen & System Audio Recording, then quit and reopen Whiz."
            default:
                break
            }
        }
        return nsError.localizedDescription
    }

    // MARK: - Samples (capture queue)

    private func now() -> CMTime { CMClockGetTime(CMClockGetHostTimeClock()) }

    private func audio(_ sample: CMSampleBuffer, from source: AudioMixer.Source, at time: CMTime) {
        guard !stopped, let writer, CMSampleBufferDataIsReady(sample) else { return }
        let samples: [Float]
        do {
            samples = try AudioSamples.mono(from: sample)
        } catch {
            let name = source == .microphone ? "The microphone" : "The call's audio"
            meter.withLock { $0.warning = "\(name) sends audio Whiz cannot record (\(error))." }
            return
        }
        writer.addAudio(samples, from: source, at: time, now: now())
        let peak = AudioSamples.peak(samples)
        meter.withLock { state in
            switch source {
            case .microphone: state.microphone = max(state.microphone, peak)
            case .system: state.system = max(state.system, peak)
            }
        }
        checkWriter(writer)
    }

    private func checkWriter(_ writer: RecordingWriter) {
        guard let failure = writer.failure, !interrupted else { return }
        interrupted = true
        onInterruption?("Recording stopped: \(failure.localizedDescription)")
    }
}

extension CallRecorder: AVCaptureAudioDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        var time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if let microphoneClock {
            time = CMSyncConvertTime(time, from: microphoneClock, to: CMClockGetHostTimeClock())
        }
        audio(sampleBuffer, from: .microphone, at: time)
    }
}

extension CallRecorder: SCStreamOutput, SCStreamDelegate {
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        switch type {
        case .audio:
            audio(sampleBuffer, from: .system, at: CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        case .screen:
            guard !stopped, sources.display != nil, let writer, Self.isCompleteFrame(sampleBuffer) else { return }
            writer.addVideo(sampleBuffer)
            checkWriter(writer)
        default:
            break
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        queue.async { [self] in
            guard !stopped, !interrupted else { return }
            interrupted = true
            onInterruption?(Self.describe(error))
        }
    }

    /// ScreenCaptureKit also sends "nothing changed" and "display asleep"
    /// buffers with no picture in them.
    static func isCompleteFrame(_ sample: CMSampleBuffer) -> Bool {
        guard let attachments = CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: false)
                as? [[SCStreamFrameInfo: Any]],
              let raw = attachments.first?[.status] as? Int,
              let status = SCFrameStatus(rawValue: raw)
        else { return false }
        return status == .complete
    }
}
