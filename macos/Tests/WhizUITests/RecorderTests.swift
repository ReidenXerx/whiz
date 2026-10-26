// AVAssetTrack is Sendable only in newer SDKs than CI's.
@preconcurrency import AVFoundation
import Foundation
import Testing
@testable import WhizUI

@Suite("Recording: mixing the two sides")
struct AudioMixerTests {
    @Test("buffers land at their timestamps and the two sources add up")
    func placesAndMixes() {
        var mixer = AudioMixer()
        mixer.add([0.25, 0.25, 0.25, 0.25], from: .microphone, at: 0)
        mixer.add([0.5, 0.5], from: .system, at: 2)
        let chunk = mixer.drain(deadline: 0)
        #expect(chunk?.start == 0)
        // The system has reached frame 4 and the microphone frame 4.
        #expect(chunk?.samples == [0.25, 0.25, 0.75, 0.75])
        #expect(mixer.released == 4)
    }

    @Test("nothing is released past the slower source until the deadline says it is late")
    func waitsForTheSlowerSource() {
        var mixer = AudioMixer()
        mixer.add([Float](repeating: 0.1, count: 100), from: .microphone, at: 0)
        mixer.add([Float](repeating: 0.1, count: 10), from: .system, at: 0)
        #expect(mixer.drain(deadline: 0)?.samples.count == 10)
        // The call went quiet: by the deadline its share is silence.
        let late = mixer.drain(deadline: 150)
        #expect(late?.start == 10)
        #expect(late?.samples.count == 140)
        #expect(late?.samples[0] == Float(0.1))
        #expect(late?.samples[139] == 0)
    }

    @Test("jitter within 50 ms continues the source; a real gap stays a gap")
    func jitterAndGaps() {
        var mixer = AudioMixer()
        mixer.add([1, 1], from: .microphone, at: 0)
        mixer.add([1, 1], from: .microphone, at: 3)          // 1 frame late: jitter
        #expect(mixer.drainAll()?.samples == [1, 1, 1, 1])
        mixer.add([1], from: .microphone, at: 4 + 4_800)     // 100 ms on: a gap
        let gap = mixer.drainAll()
        #expect(gap?.samples.count == 4_801)
        #expect(gap?.samples.first == 0 && gap?.samples.last == 1)
    }

    @Test("a source that repeats what it already sent is not mixed twice")
    func overlapIsDropped() {
        var mixer = AudioMixer()
        mixer.add([Float](repeating: 0.5, count: 6_000), from: .system, at: 0)
        // The source's clock fell 100 ms behind: its next buffer claims to
        // start 4800 frames earlier than where the last one ended.
        mixer.add([Float](repeating: 0.5, count: 6_000), from: .system, at: 1_200)
        let mixed = mixer.drainAll()
        #expect(mixed?.samples.count == 7_200)
        #expect(mixed?.samples.allSatisfy { $0 == 0.5 } == true)
    }

    @Test("audio arriving after its frames were written is dropped")
    func lateAudio() {
        var mixer = AudioMixer()
        mixer.add([0.8, 0.8], from: .microphone, at: 0)
        #expect(mixer.drain(deadline: 4)?.samples == [0.8, 0.8, 0, 0])
        mixer.add([Float](repeating: 0.5, count: 6), from: .system, at: 0)
        #expect(mixer.drainAll()?.samples == [0.5, 0.5])
    }

    @Test("the mix is clipped to ±1, as the encoder would")
    func clipping() {
        var mixer = AudioMixer()
        mixer.add([0.8, -0.8], from: .microphone, at: 0)
        mixer.add([0.8, -0.8], from: .system, at: 0)
        #expect(mixer.drainAll()?.samples == [1, -1])
    }
}

@Suite("Recording: sources and files")
@MainActor
struct RecordingFilesTests {
    @Test("the screen is recorded at its own size up to 1920 wide, in even pixels")
    func videoSize() {
        #expect(CallRecorder.videoSize(pixelWidth: 3024, pixelHeight: 1964) == (1920, 1246))
        #expect(CallRecorder.videoSize(pixelWidth: 1440, pixelHeight: 900) == (1440, 900))
        #expect(CallRecorder.videoSize(pixelWidth: 1367, pixelHeight: 769) == (1366, 768))
    }

    @Test("Screen Recording permission is needed for call audio or the screen, not for the microphone")
    func permissions() {
        #expect(!RecordingSources(microphone: "mic", systemAudio: false, display: nil).needsScreenCapture)
        #expect(RecordingSources(microphone: nil, systemAudio: true, display: nil).needsScreenCapture)
        #expect(RecordingSources(microphone: "mic", systemAudio: false, display: 1).needsScreenCapture)
        #expect(!RecordingSources(microphone: nil, systemAudio: false, display: 1).hasAudio)
    }

    @Test("audio-only recordings are .m4a, with the screen .mov")
    func fileTypes() {
        #expect(RecordingWriter.fileType(video: false) == .m4a)
        #expect(RecordingWriter.fileType(video: true) == .mov)
    }

    @Test("the writer produces a file AVFoundation reads back, of the length it was given")
    func writesReadableAudio() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("call.m4a")

        let writer = try RecordingWriter(url: url, video: nil)
        let origin = CMTime(seconds: 1_000, preferredTimescale: 48_000)
        try writer.start(at: origin)
        // Two seconds of a tone from the "microphone", the "call" joining
        // half a second in: written 100 ms at a time, as capture would.
        let tone = (0..<4_800).map { Float(sin(Double($0) * 2 * .pi * 440 / 48_000) * 0.3) }
        for step in 0..<20 {
            let at = CMTimeAdd(origin, CMTime(value: CMTimeValue(step * 4_800), timescale: 48_000))
            writer.addAudio(tone, from: .microphone, at: at, now: at)
            if step >= 5 { writer.addAudio(tone, from: .system, at: at, now: at) }
        }
        try await writer.finish()

        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        #expect(abs(duration - 2) < 0.05, "duration \(duration)")
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        #expect(tracks.count == 1)
    }

    @Test("a recording that captured nothing leaves no file behind")
    func emptyRecording() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).m4a")
        let writer = try RecordingWriter(url: url, video: nil)
        try writer.start(at: CMTime(seconds: 1, preferredTimescale: 600))
        await #expect(throws: RecordingError.self) { try await writer.finish() }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }
}

/// Real capture, off by default: it needs Screen Recording and Microphone
/// permission for the process running the tests, and it records this Mac.
/// `WHIZ_LIVE_CAPTURE=1 swift test --filter LiveCapture`. Plays a quiet tone
/// so the call side has something to hear; one test at a time, or each
/// would hear the other's.
@Suite("Recording: live capture", .serialized, .enabled(if: ProcessInfo.processInfo.environment["WHIZ_LIVE_CAPTURE"] == "1"))
struct LiveCaptureTests {
    @Test("call audio + microphone + screen for three seconds")
    func records() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("live.mov")
        let microphone = AVCaptureDevice.default(for: .audio)?.uniqueID
        let recorder = CallRecorder(url: url, sources: RecordingSources(
            microphone: microphone, systemAudio: true, display: CGMainDisplayID()))
        try await recorder.start()
        let tone = Process()
        tone.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
        tone.arguments = ["-v", "0.15", "/System/Library/Sounds/Glass.aiff"]
        try await Task.sleep(for: .milliseconds(500))
        try tone.run()
        var heardSystem: Float = 0
        var heardMicrophone: Float = 0
        for _ in 0..<25 {
            try await Task.sleep(for: .milliseconds(100))
            let meter = recorder.readMeter()
            heardSystem = max(heardSystem, meter.system)
            heardMicrophone = max(heardMicrophone, meter.microphone)
        }
        try await recorder.stop()

        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        let audio = try await asset.loadTracks(withMediaType: .audio)
        let video = try await asset.loadTracks(withMediaType: .video)
        print("live capture: \(duration) s, system peak \(heardSystem), microphone peak \(heardMicrophone), video \(video.count)")
        #expect(duration > 2.5 && duration < 4)
        #expect(audio.count == 1 && video.count == 1)
        #expect(heardSystem > 0.001, "the tone reached the recording")
        if microphone != nil { #expect(heardMicrophone > 0) }
        let size = try await video.first?.load(.naturalSize)
        #expect((size?.width ?? 0) > 0)
    }

    @Test("the call's audio lands where it happened, not where the file started")
    func placesSoundInTime() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("timing.m4a")
        let recorder = CallRecorder(url: url, sources: RecordingSources(microphone: nil, systemAudio: true, display: nil))
        try await recorder.start()
        try await Task.sleep(for: .milliseconds(1_500))
        let tone = Process()
        tone.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
        tone.arguments = ["-v", "0.15", "/System/Library/Sounds/Glass.aiff"]
        let played = CMClockGetTime(CMClockGetHostTimeClock())
        try tone.run()
        try await Task.sleep(for: .milliseconds(2_000))
        try await recorder.stop()

        let expected = CMTimeGetSeconds(CMTimeSubtract(played, recorder.origin))
        let onset = try await Self.onset(of: url)
        print("tone played at \(expected) s, heard at \(onset.map(String.init(describing:)) ?? "never") s")
        let heard = try #require(onset)
        // afplay takes a moment to start the sound; never earlier than played.
        #expect(heard >= expected - 0.03 && heard < expected + 0.35)
    }

    /// Seconds into the file where the sound first rises above silence.
    static func onset(of url: URL) async throws -> Double? {
        let asset = AVURLAsset(url: url)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderAudioMixOutput(audioTracks: try await asset.loadTracks(withMediaType: .audio), audioSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 48_000, AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true, AVLinearPCMIsNonInterleaved: false,
            AVLinearPCMIsBigEndianKey: false,
        ])
        reader.add(output)
        reader.startReading()
        var frame = 0
        while let sample = output.copyNextSampleBuffer() {
            let samples = try AudioSamples.mono(from: sample)
            if let index = samples.firstIndex(where: { abs($0) > 0.005 }) {
                return Double(frame + index) / 48_000
            }
            frame += samples.count
        }
        return nil
    }
}

@Suite("Recording: preferences and hand-off")
@MainActor
struct RecordingPreferencesTests {
    private func makeSuite() -> UserDefaults {
        UserDefaults(suiteName: "test.recording.\(UUID().uuidString)")!
    }

    @Test("preferences survive a round trip; a fresh Mac gets the defaults")
    func roundTrip() {
        let defaults = makeSuite()
        #expect(RecordingPreferences.load(from: defaults) == RecordingPreferences())
        var preferences = RecordingPreferences()
        preferences.microphone = "off"
        preferences.screen = true
        preferences.display = 7
        preferences.folderPath = "/tmp/calls"
        preferences.save(to: defaults)
        #expect(RecordingPreferences.load(from: defaults) == preferences)
    }

    @Test("recordings go to ~/Movies/Whiz unless the person chose a folder")
    func folder() {
        var preferences = RecordingPreferences()
        #expect(preferences.folder.path.hasSuffix("/Movies/Whiz"))
        preferences.folderPath = "/tmp/calls"
        #expect(preferences.folder.path == "/tmp/calls")
    }

    @Test("a Whiz recording is an .m4a in the recordings folder; anything else is not")
    func recognizesRecordings() {
        var preferences = RecordingPreferences()
        preferences.folderPath = "/tmp/calls"
        #expect(preferences.isRecording(URL(fileURLWithPath: "/tmp/calls/Recording 2026-10-26 at 14.03.m4a")))
        #expect(!preferences.isRecording(URL(fileURLWithPath: "/tmp/calls/Recording.mov")), "video is diarized anyway")
        #expect(!preferences.isRecording(URL(fileURLWithPath: "/tmp/other/memo.m4a")))
        #expect(!preferences.isRecording(URL(fileURLWithPath: "/tmp/calls/sub/memo.m4a")))
    }

    @Test("files are named like macOS screenshots and never overwrite one another")
    func naming() throws {
        var components = DateComponents()
        (components.year, components.month, components.day, components.hour, components.minute) = (2026, 10, 26, 14, 3)
        let date = try #require(Calendar.current.date(from: components))
        #expect(RecorderModel.fileName(at: date) == "Recording 2026-10-26 at 14.03")

        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let first = RecorderModel.uniqueURL(in: folder, name: "Recording", extension: "m4a")
        #expect(first.lastPathComponent == "Recording.m4a")
        FileManager.default.createFile(atPath: first.path, contents: Data())
        #expect(RecorderModel.uniqueURL(in: folder, name: "Recording", extension: "m4a").lastPathComponent == "Recording 2.m4a")
    }

    @Test("transcribing a recorded call asks the CLI to tell speakers apart")
    func recordedCallsAreDiarized() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let arguments = directory.appendingPathComponent("argv")
        let script = directory.appendingPathComponent("whiz")
        try """
        #!/bin/sh
        echo "$@" > '\(arguments.path)'
        exit 0
        """.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)

        let call = URL(fileURLWithPath: "/tmp/calls/Recording.m4a")
        let model = TranscriptionViewModel(
            lastRunStore: LastRunStore(defaults: makeSuite()), optionsStore: OptionsStore(defaults: makeSuite()),
            isRecording: { $0 == call },
            makeBackend: { CLIBackend(executable: script) })
        for (input, diarized) in [(call, true), (URL(fileURLWithPath: "/tmp/memo.m4a"), false)] {
            try? FileManager.default.removeItem(at: arguments)
            model.inputURL = input
            model.start()
            for _ in 0..<200 where model.isRunning { try await Task.sleep(for: .milliseconds(25)) }
            let argv = try String(contentsOf: arguments, encoding: .utf8)
            #expect(argv.contains("--speakers") == diarized, "\(input.lastPathComponent): \(argv)")
        }
    }
}
