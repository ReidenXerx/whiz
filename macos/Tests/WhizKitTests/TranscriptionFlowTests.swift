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

    @Test("an explicit configured model path wins, but only if it exists")
    func resolveConfigured() {
        #expect(WhisperModel.resolve(configured: "/definitely/not/a/model.bin") == nil)
    }

    // MARK: - Degraded-run contract (PR #4: d8ab1b9 + a02a70a)

    @Test("degraded detection: the note marks an HTML page; letterized lines mark a TXT")
    func degradedDetection() {
        // HTML: the provenance note is the marker (it appears verbatim — no
        // escapable characters).
        #expect(DegradedArtifacts.looksDegradedHTML(
            "<html>\(DegradedArtifacts.genericLabelNote)</html>"))
        #expect(!DegradedArtifacts.looksDegradedHTML(
            "<span class=\"speaker\" style=\"color:#e74c3c\">Speaker A</span>"))

        // TXT: every content line must be the bare generic form.
        #expect(DegradedArtifacts.looksDegradedTXT(
            "Speaker (00:00:01): one\nSpeaker (00:00:05): two\n"))
        #expect(DegradedArtifacts.looksDegradedTXT(""))
        #expect(!DegradedArtifacts.looksDegradedTXT(
            "Speaker A (00:00:01): real label"))
        #expect(!DegradedArtifacts.looksDegradedTXT(
            "Speaker (00:00:01): one\nAlice (00:00:05): named"))
        // The generic-line shape itself is precise.
        #expect(DegradedArtifacts.isGenericDialogueLine("Speaker (01:23:45): hi"))
        #expect(!DegradedArtifacts.isGenericDialogueLine("Speaker (1:23:45): hi"))
        #expect(!DegradedArtifacts.isGenericDialogueLine("Speaker (01:23:45)hi"))
    }

    @Test("the no-clobber guard: missing/degraded may be written, labeled is kept")
    func noClobberGuard() throws {
        let dir = tempURL("dir")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let missing = dir.appendingPathComponent("missing.html")
        #expect(DegradedArtifacts.mayOverwriteHTML(at: missing))

        let degradedHTML = dir.appendingPathComponent("degraded.html")
        try "<p>\(DegradedArtifacts.genericLabelNote)</p>"
            .write(to: degradedHTML, atomically: true, encoding: .utf8)
        #expect(DegradedArtifacts.mayOverwriteHTML(at: degradedHTML))

        let labeledHTML = dir.appendingPathComponent("labeled.html")
        try "Speaker A everywhere".write(to: labeledHTML, atomically: true, encoding: .utf8)
        #expect(!DegradedArtifacts.mayOverwriteHTML(at: labeledHTML))

        let labeledTXT = dir.appendingPathComponent("labeled.txt")
        try "Speaker A (00:00:01): real".write(to: labeledTXT, atomically: true, encoding: .utf8)
        #expect(!DegradedArtifacts.mayOverwriteTXT(at: labeledTXT))

        // Unreadable-but-existing: "when in doubt, keep" (cli.py parity).
        let unreadable = dir.appendingPathComponent("unreadable.txt")
        try Data([0xFF, 0xFE, 0x00, 0xD8]).write(to: unreadable)
        #expect(!DegradedArtifacts.mayOverwriteTXT(at: unreadable))
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

    @Test(.disabled(if: WhisperModel.resolve(configured: "") == nil))
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

    /// Real speech through the real pipeline — the degraded-run contract
    /// (PR #4 d8ab1b9/a02a70a) exercised end to end: an unlabeled run writes
    /// bare-Speaker artifacts with the HTML provenance note, keeps an existing
    /// labeled page, and overwrites its own degraded one.
    @Test(.disabled(if: WhisperModel.resolve(configured: "") == nil))
    func degradedArtifactsThroughTheRealPipeline() async throws {
        let speech = try Self.generateSpeechWav()
        defer { try? FileManager.default.removeItem(at: speech) }
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-e2e-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: output) }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

        let stem = speech.deletingPathExtension().lastPathComponent
        let htmlURL = output.appendingPathComponent("\(stem).speakers.html")
        // Pre-seed a LABELED transcript from an earlier diarized run of the
        // same input: this run has no diarization (audio input), so the
        // no-clobber guard must keep it.
        let labeledHTML = "<html>Speaker A everywhere: a diarized run's real labels</html>"
        try labeledHTML.write(to: htmlURL, atomically: true, encoding: .utf8)

        let collector = EventCollector()
        let backend = NativeTranscriptionBackend(settings: BatchSettings.from([:]))
        _ = try await backend.transcribe(
            input: speech, outputDirectory: output, onEvent: collector.record)

        // Segments exist this time, so the TXT is written — degraded form.
        let txt = try String(contentsOf: output.appendingPathComponent("\(stem).speakers.txt"), encoding: .utf8)
        #expect(!txt.isEmpty)
        #expect(DegradedArtifacts.looksDegradedTXT(txt))
        // The labeled HTML was kept, not overwritten with generic labels.
        #expect(try String(contentsOf: htmlURL, encoding: .utf8) == labeledHTML)

        // Idempotence: seed a degraded HTML (this run's own shape) and run
        // again — it IS overwritten, and the fresh page carries the note.
        try DegradedArtifacts.genericLabelNote.write(to: htmlURL, atomically: true, encoding: .utf8)
        _ = try await backend.transcribe(
            input: speech, outputDirectory: output, onEvent: collector.record)
        let fresh = try String(contentsOf: htmlURL, encoding: .utf8)
        #expect(fresh != labeledHTML)
        #expect(fresh.contains(DegradedArtifacts.genericLabelNote))
        print("E2E degraded: \(txt.count) TXT bytes, fresh HTML carries the note")
    }

    /// A few seconds of real speech via macOS TTS — `say` + `afconvert`, the
    /// same recipe the PR #4 analysis probe used. Both ship with macOS.
    private static func generateSpeechWav() throws -> URL {
        let aiff = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-e2e-speech-\(UUID().uuidString).aiff")
        let wav = aiff.deletingPathExtension().appendingPathExtension("wav")

        let say = Process()
        say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        say.arguments = ["-o", aiff.path,
                         "Hello, this is a test of the transcription pipeline.",
                         "The weather is nice today and the quick brown fox jumps over the lazy dog."]
        try say.run()
        say.waitUntilExit()

        let convert = Process()
        convert.executableURL = URL(fileURLWithPath: "/usr/bin/afconvert")
        convert.arguments = ["-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                             aiff.path, wav.path]
        try convert.run()
        convert.waitUntilExit()

        try? FileManager.default.removeItem(at: aiff)
        guard say.terminationStatus == 0, convert.terminationStatus == 0,
              FileManager.default.fileExists(atPath: wav.path)
        else { throw AudioDecodeError.unreadableMedia(wav.lastPathComponent) }
        return wav
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
/// Does a language chosen in the dialog actually reach the transcriber?
///
/// Written because the observable symptom — "English comes out Russian
/// whatever I pick" — has two very different causes: the picker not being
/// wired, or a stale `language` value being forced. These pin the wiring, so
/// the next occurrence points at the value rather than the plumbing.
@Suite("Language selection reaches the run")
@MainActor
struct LanguageSelectionTests {

    @Test("the dialog's language overrides the config default")
    func dialogOverridesConfig() {
        var base = BatchSettings.from(["language": .string("ru")])
        base.aiModel = ""
        let model = TranscriptionSetupModel(settings: base)
        #expect(model.language == "ru", "seeded from config")

        model.language = "en"
        #expect(model.resolvedSettings().language == "en",
                "picker choice must win over the config value")
    }

    @Test("each of the offered languages survives to the settings")
    func everyOfferedLanguageSurvives() {
        let model = TranscriptionSetupModel(settings: BatchSettings.from([:]))
        for code in ["auto"] + WhisperLanguages.offeredCodes {
            model.language = code
            #expect(model.resolvedSettings().language == code)
        }
    }

    @Test("a fresh dialog re-reads the config rather than caching")
    func restartRereadsConfig() {
        // The window is reused across runs, so a stale setup model would pin
        // whatever language was chosen the first time it was ever opened.
        let flow = TranscriptionFlowModel()
        flow.setupLoader = { _ in }
        flow.restart()
        let first = flow.setup
        first.language = "uk"
        flow.restart()
        #expect(flow.setup !== first, "restart must build a new setup model")
        #expect(flow.setup.language != "uk" || BatchSettings.load().language == "uk",
                "a new dialog must not inherit the previous run's choice")
    }
}
