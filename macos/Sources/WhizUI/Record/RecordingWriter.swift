import AVFoundation
import CoreMedia

/// Writes a recording: the mixed audio as one AAC track, and the screen as
/// H.264 when it is recorded too.
///
/// Not thread-safe: `CallRecorder` calls it from its one capture queue only
/// (hence `@unchecked Sendable`, to hand it there).
///
/// The file is written in fragments every few seconds, so a crash or a
/// power cut mid-call leaves a playable recording up to the last fragment,
/// not a file with no index.
final class RecordingWriter: @unchecked Sendable {
    struct Video: Equatable {
        var width: Int
        var height: Int
        var frameRate: Int
    }

    let url: URL
    private let writer: AVAssetWriter
    private let audioInput: AVAssetWriterInput
    private let videoInput: AVAssetWriterInput?
    private let audioFormat: CMAudioFormatDescription
    private var mixer = AudioMixer()
    /// Host time of frame 0 on the timeline.
    private var origin: CMTime = .invalid
    private(set) var audioFrames = 0
    private(set) var videoFrames = 0
    private var lastVideoTime: CMTime = .invalid

    /// .mov with video, .m4a without.
    static func fileType(video: Bool) -> AVFileType { video ? .mov : .m4a }

    init(url: URL, video: Video?) throws {
        self.url = url
        writer = try AVAssetWriter(outputURL: url, fileType: Self.fileType(video: video != nil))
        writer.movieFragmentInterval = CMTime(seconds: 5, preferredTimescale: 600)
        writer.shouldOptimizeForNetworkUse = false

        audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: AudioMixer.sampleRate,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 96_000,
        ])
        audioInput.expectsMediaDataInRealTime = true
        guard writer.canAdd(audioInput) else { throw RecordingError("cannot write audio to \(url.lastPathComponent)") }
        writer.add(audioInput)

        if let video {
            // Screen content: text and slides, mostly still. A keyframe every
            // two seconds keeps frame grabs for the transcript fast.
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: video.width,
                AVVideoHeightKey: video.height,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: Self.videoBitRate(video),
                    AVVideoExpectedSourceFrameRateKey: video.frameRate,
                    AVVideoMaxKeyFrameIntervalDurationKey: 2,
                    AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
                ] as [String: Any],
            ])
            input.expectsMediaDataInRealTime = true
            guard writer.canAdd(input) else { throw RecordingError("cannot write video to \(url.lastPathComponent)") }
            writer.add(input)
            videoInput = input
        } else {
            videoInput = nil
        }

        var description = AudioStreamBasicDescription(
            mSampleRate: Float64(AudioMixer.sampleRate), mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
            mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
        var format: CMAudioFormatDescription?
        let status = CMAudioFormatDescriptionCreate(
            allocator: kCFAllocatorDefault, asbd: &description, layoutSize: 0, layout: nil,
            magicCookieSize: 0, magicCookie: nil, extensions: nil, formatDescriptionOut: &format)
        guard status == noErr, let format else { throw RecordingError("cannot describe the audio format (\(status))") }
        audioFormat = format
    }

    /// About 0.08 bits a pixel at the frame rate: sharp text on a still
    /// screen, roughly 700 MB an hour at 1920 × 1200.
    static func videoBitRate(_ video: Video) -> Int {
        max(Int(Double(video.width * video.height * video.frameRate) * 0.08), 500_000)
    }

    /// Begin the file; samples are placed relative to `origin` (host time).
    func start(at origin: CMTime) throws {
        guard writer.startWriting() else {
            throw RecordingError("cannot write \(url.lastPathComponent): \(writer.error?.localizedDescription ?? "unknown error")")
        }
        writer.startSession(atSourceTime: origin)
        self.origin = origin
    }

    var failure: Error? { writer.status == .failed ? writer.error : nil }

    /// The timeline frame a host time falls on.
    func frame(at time: CMTime) -> Int {
        Int((CMTimeGetSeconds(CMTimeSubtract(time, origin)) * Double(AudioMixer.sampleRate)).rounded())
    }

    /// A source's samples, starting at host time `time`. Written once the
    /// other source has caught up or `now` says it is late.
    func addAudio(_ samples: [Float], from source: AudioMixer.Source, at time: CMTime, now: CMTime) {
        guard origin.isValid else { return }
        mixer.add(samples, from: source, at: frame(at: time))
        // Half a second of slack for the slower source before its share of
        // the timeline is written as silence.
        if let chunk = mixer.drain(deadline: frame(at: now) - AudioMixer.sampleRate / 2) {
            appendMixed(chunk)
        }
    }

    /// A screen frame, as ScreenCaptureKit delivered it.
    func addVideo(_ sample: CMSampleBuffer) {
        guard let videoInput, origin.isValid else { return }
        let time = CMSampleBufferGetPresentationTimeStamp(sample)
        guard CMTimeCompare(time, origin) >= 0,
              !lastVideoTime.isValid || CMTimeCompare(time, lastVideoTime) > 0,
              videoInput.isReadyForMoreMediaData   // a frame dropped under load is fine
        else { return }
        if videoInput.append(sample) {
            videoFrames += 1
            lastVideoTime = time
        }
    }

    /// Write what is left and close the file. Throws when nothing at all
    /// was recorded — an empty file is not a recording.
    func finish() async throws {
        if let chunk = mixer.drainAll() { appendMixed(chunk) }
        guard audioFrames > 0 || videoFrames > 0 else {
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: url)
            throw RecordingError("nothing was recorded: no sound and no picture arrived")
        }
        audioInput.markAsFinished()
        videoInput?.markAsFinished()
        // The file ends with the audio: a screen that sat still for the
        // last minute is still a minute of recording.
        let end = CMTimeAdd(origin, CMTime(value: CMTimeValue(audioFrames), timescale: CMTimeScale(AudioMixer.sampleRate)))
        writer.endSession(atSourceTime: lastVideoTime.isValid ? CMTimeMaximum(end, lastVideoTime) : end)
        await writer.finishWriting()
        if writer.status != .completed {
            throw RecordingError("could not finish \(url.lastPathComponent): \(writer.error?.localizedDescription ?? "unknown error")")
        }
    }

    /// Recording stopped by a failure: keep the fragments written so far.
    func abandon() async {
        if writer.status == .writing { try? await finish() }
    }

    private func appendMixed(_ chunk: (start: Int, samples: [Float])) {
        guard !chunk.samples.isEmpty, writer.status == .writing else { return }
        // Behind the encoder this means the Mac is badly overloaded; losing
        // the chunk would shift everything after it, so wait the moment out.
        var waited = 0
        while !audioInput.isReadyForMoreMediaData, waited < 200 {
            usleep(1_000)
            waited += 1
        }
        let time = CMTimeAdd(origin, CMTime(value: CMTimeValue(chunk.start), timescale: CMTimeScale(AudioMixer.sampleRate)))
        guard let sample = Self.sampleBuffer(chunk.samples, at: time, format: audioFormat),
              audioInput.append(sample) else { return }
        audioFrames = chunk.start + chunk.samples.count
    }

    /// Mono float samples as a CMSampleBuffer the encoder takes.
    static func sampleBuffer(_ samples: [Float], at time: CMTime, format: CMAudioFormatDescription) -> CMSampleBuffer? {
        let length = samples.count * MemoryLayout<Float>.size
        var block: CMBlockBuffer?
        guard CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault, memoryBlock: nil, blockLength: length,
            blockAllocator: kCFAllocatorDefault, customBlockSource: nil, offsetToData: 0,
            dataLength: length, flags: kCMBlockBufferAssureMemoryNowFlag, blockBufferOut: &block) == noErr,
              let block else { return nil }
        let copied = samples.withUnsafeBytes { bytes in
            CMBlockBufferReplaceDataBytes(with: bytes.baseAddress!, blockBuffer: block,
                                          offsetIntoDestination: 0, dataLength: length)
        }
        guard copied == noErr else { return nil }
        var sample: CMSampleBuffer?
        guard CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            allocator: kCFAllocatorDefault, dataBuffer: block, formatDescription: format,
            sampleCount: samples.count, presentationTimeStamp: time,
            packetDescriptions: nil, sampleBufferOut: &sample) == noErr else { return nil }
        return sample
    }
}

/// A recording failure, worded for the person recording.
struct RecordingError: LocalizedError, Equatable {
    var message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
