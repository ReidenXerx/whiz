import Testing
import Foundation
@testable import WhizKit

// swift-testing (see ConfigTests for the toolchain rationale).
//
// The window and view themselves have no test host (the repo has no UI test
// infrastructure, consistent with the other UI files); what IS pinned here is
// the flow's contract: output location, the real unsupported-container
// rejection, the pipeline keys the native backend reads, and the model
// preference order it searches — plus the fixture backend that keeps the
// view model's state machine covered without a Whisper model on disk.

/// The UI-era stand-in pipeline, kept as a fixture: real shape, fake work.
// The unsupported-container check is the real one from `AudioFileDecoder`,
// phases and log lines mirror the native pipeline's, and the run writes its
// log as the output artifact. It lives in the test target now — nothing
// simulated ships in the app binary.
struct SimulatedTranscriptionBackend: TranscriptionBackend {

    /// Length of each simulated work step.
    var tick: Duration = .milliseconds(450)

    func transcribe(
        input: URL,
        outputDirectory: URL,
        onEvent: @escaping @Sendable (TranscriptionEvent) -> Void
    ) async throws -> URL {
        // Not simulated: the real error path — AVFoundation cannot demux
        // these containers, and the native pipeline refuses the same way.
        if AudioFileDecoder.isUnsupportedContainer(input) {
            throw AudioDecodeError.unsupportedContainer(input.pathExtension.lowercased())
        }

        var lines: [String] = []
        func log(_ text: String) {
            lines.append(text)
            onEvent(.log(text))
        }

        log("(simulation fixture — not the native pipeline)")

        onEvent(.phase("Decoding audio"))
        log("input: \(input.lastPathComponent)")
        try Task.checkCancellation()
        try await Task.sleep(for: tick)
        log("audio: 16 kHz mono · 4,000,000 samples (04:10.0)")
        onEvent(.progress(0.2))

        onEvent(.phase("Transcribing"))
        let segments = 9
        for segment in 1...segments {
            try Task.checkCancellation()
            try await Task.sleep(for: tick)
            let start = Double(segment) * 15.0
            log(String(format: "segment %d/%d  [%06.1f → %06.1f]", segment, segments, start, start + 12.0))
            onEvent(.progress(0.2 + 0.7 * Double(segment) / Double(segments)))
        }

        onEvent(.phase("Writing outputs"))
        try Task.checkCancellation()
        try await Task.sleep(for: tick)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let logText = """
            whiz transcription (simulated run)
            input: \(input.lastPathComponent)

            """ + lines.joined(separator: "\n") + "\n"
        try logText.write(
            to: outputDirectory.appendingPathComponent("transcription-log.txt"),
            atomically: true, encoding: .utf8)
        log("output: \(outputDirectory.path)")
        onEvent(.progress(1))
        onEvent(.phase("Finished"))
        return outputDirectory
    }
}

@Suite("Transcription flow")
struct TranscriptionFlowTests {

    @Test("outputs land in a sibling <stem>.transcript directory")
    func outputDirectoryNaming() {
        let input = URL(fileURLWithPath: "/Movies/clip.mp4")
        #expect(TranscriptionOutputs.directory(for: input)
                == URL(fileURLWithPath: "/Movies/clip.transcript"))

        // Extensionless file: deletingPathExtension is a no-op there.
        let bare = URL(fileURLWithPath: "/Movies/interview")
        #expect(TranscriptionOutputs.directory(for: bare)
                == URL(fileURLWithPath: "/Movies/interview.transcript"))
    }

    @Test("the flow rejects containers AVFoundation cannot demux")
    func rejectsUnsupportedContainers() async throws {
        let url = tempURL("mkv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("not a matroska".utf8).write(to: url)

        // The real rejection, not the simulation's: the same check
        // AudioFileDecoder will make when the native pipeline lands.
        let backend = SimulatedTranscriptionBackend(tick: .milliseconds(1))
        await #expect(throws: AudioDecodeError.unsupportedContainer("mkv")) {
            _ = try await backend.transcribe(
                input: url, outputDirectory: tempURL("transcript")) { _ in }
        }
    }

    @Test("a simulated run reports monotonic progress and writes its log to the output directory")
    func simulatedRunWritesLog() async throws {
        let input = tempURL("mp4")
        defer { try? FileManager.default.removeItem(at: input) }
        try Data("stub".utf8).write(to: input)

        let out = tempURL("transcript")
        defer { try? FileManager.default.removeItem(at: out) }

        let collected = EventCollector()
        let backend = SimulatedTranscriptionBackend(tick: .milliseconds(1))
        let returned = try await backend.transcribe(
            input: input, outputDirectory: out, onEvent: collected.record)

        #expect(returned == out)

        // A status bar must never move backwards — the value drives a
        // determinate ProgressView.
        let progressValues = collected.progressValues
        #expect(progressValues == progressValues.sorted())
        #expect(progressValues.last == 1.0)
        #expect(progressValues.count > 2)

        let phases = collected.phases
        #expect(phases.first == "Decoding audio")
        #expect(phases.contains("Transcribing"))
        #expect(phases.last == "Finished")

        let logURL = out.appendingPathComponent("transcription-log.txt")
        let written = try String(contentsOf: logURL, encoding: .utf8)
        #expect(written.contains("input: \(input.lastPathComponent)"))
        #expect(written.contains("segment 9/9"))
        #expect(written.contains("simulation"))
    }

    @Test("batch pipeline keys parse from the shared config, defaults mirror config.py")
    func batchSettingsParseAndDefaults() {
        // Defaults are config.py's pipeline defaults: language "auto",
        // vad on, threshold 0.5, no explicit model.
        let defaults = BatchSettings.from([:])
        #expect(defaults.language == "auto")
        #expect(defaults.model.isEmpty)
        #expect(defaults.vad == true)
        #expect(defaults.vadModel.isEmpty)
        #expect(defaults.vadThreshold == 0.5)

        let parsed = BatchSettings.from(FlatTOML.parse("""
            language = "de"
            vad = false
            vad_threshold = 0.35
            vad_model = "~/models/silero.bin"
            """))
        #expect(parsed.language == "de")
        #expect(parsed.vad == false)
        #expect(parsed.vadThreshold == 0.35)
        #expect(parsed.vadModel == "~/models/silero.bin")
    }

    @Test("batch model preference mirrors models.py:PREFERENCE")
    func batchPreferencePinned() {
        // models.py:47-56, verbatim order — q5_0 turbo first. NS-15 leaves
        // "q5_0 batch vs unquantized turbo" open, so parity with the Python
        // pipeline is the only defensible default; drift here changes which
        // model a fresh Mac transcribes with.
        #expect(WhisperModel.batchPreference == [
            "ggml-large-v3-turbo-q5_0.bin",
            "ggml-large-v3-turbo.bin",
            "ggml-large-v3-turbo-q8_0.bin",
            "ggml-large-v3-q5_0.bin",
            "ggml-large-v3.bin",
            "ggml-medium-q5_0.bin",
            "ggml-medium.bin",
            "ggml-small-q5_0.bin",
            "ggml-small.bin",
        ])
        // Deliberately NOT the dictation order — that divergence is documented
        // in WhisperModel.swift and must not silently unify.
        #expect(WhisperModel.batchPreference != WhisperModel.preference)

        // An explicit configured path wins, but only if it exists.
        #expect(WhisperModel.resolveBatch(configured: "/definitely/not/a/model.bin") == nil)
    }

    // MARK: - Helpers

    private func tempURL(_ ext: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-transcribe-\(UUID().uuidString).\(ext)")
    }
}

/// End-to-end through the REAL pipeline — decode → beam-search whisper with
/// VAD → SRT/JSON written — using a golden-corpus WAV and the same local
/// model discovery the app uses. Gated: without a downloaded model there is
/// nothing to transcribe with, and an empty-output "success" would test
/// nothing, so the test disables itself rather than faking a run.
@Suite("Native pipeline end to end")
struct NativePipelineTests {

    /// …/macos/Tests/WhizKitTests/<this>.swift → repo root (tuning/golden).
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // WhizKitTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // macos
        .deletingLastPathComponent()   // repo root

    @Test(.disabled(if: WhisperModel.resolveBatch(configured: "") == nil))
    func goldenFixtureThroughTheRealPipeline() async throws {
        let input = Self.repoRoot.appendingPathComponent("tuning/golden/quiet_two_utterances.wav")
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-e2e-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: output) }

        let collector = EventCollector()
        let backend = NativeTranscriptionBackend(settings: BatchSettings.from([:]))
        let returned = try await backend.transcribe(
            input: input, outputDirectory: output, onEvent: collector.record)

        #expect(returned == output)
        #expect(collector.progressValues.last == 1.0)
        #expect(collector.phases.contains("Transcribing"))
        #expect(collector.phases.last == "Finished")

        let stem = input.deletingPathExtension().lastPathComponent
        let srt = try String(contentsOf: output.appendingPathComponent("\(stem).srt"), encoding: .utf8)
        let json = try String(contentsOf: output.appendingPathComponent("\(stem).json"), encoding: .utf8)

        // The corpus WAV is synthetic: whisper may produce any number of
        // segments including zero, so what is pinned is that every stage ran
        // and both outputs are real, parseable artifacts.
        let object = try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        #expect((object?["transcription"] as? [[String: Any]]) != nil)
        print("E2E: \(srt.count) SRT bytes, \(String(describing: (object?["transcription"] as? [[String: Any]])?.count)) segments")
    }
}

/// Lock-guarded collection point for `@Sendable` backend events in tests.
private final class EventCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [TranscriptionEvent] = []

    func record(_ event: TranscriptionEvent) {
        lock.lock()
        defer { lock.unlock() }
        events.append(event)
    }

    var progressValues: [Double] {
        lock.lock()
        defer { lock.unlock() }
        return events.compactMap {
            if case .progress(let fraction) = $0 { return fraction }
            return nil
        }
    }

    var phases: [String] {
        lock.lock()
        defer { lock.unlock() }
        return events.compactMap {
            if case .phase(let text) = $0 { return text }
            return nil
        }
    }
}