import Foundation
import Testing
@testable import WhizUI

/// The parser that turns CLI output into events.
///
/// This is the whole interface between the app and the pipeline, so it is
/// tested against the exact line shapes `whiz/ui.py` emits when piped. The
/// Python side pins those shapes in `tests/test_ui_machine_contract.py`; these
/// tests pin this side's reading of them. Either file failing means the two
/// halves have drifted.
@Suite("CLI output parsing")
struct CLIOutputParsingTests {

    @Test("a phase line becomes a phase event")
    func phaseLine() {
        #expect(CLIBackend.classify("▸ diarizing") == .phase("diarizing"))
        #expect(CLIBackend.classify("▸ capturing frames") == .phase("capturing frames"))
    }

    @Test("an artifact line yields label and path")
    func artifactLine() {
        let event = CLIBackend.classify("✓ Wrote labeled SRT: /tmp/a.speakers.srt")
        #expect(event == .artifact(Artifact(
            label: "Wrote labeled SRT",
            url: URL(fileURLWithPath: "/tmp/a.speakers.srt"))))
    }

    @Test("paths containing spaces and colons survive")
    func awkwardPaths() {
        // Splitting on the LAST ": " would truncate both of these. macOS
        // filenames routinely contain spaces, and a colon is legal too.
        let spaced = CLIBackend.classify("✓ Wrote analysis: /Users/a b/My Talk.analysis.md")
        guard case .artifact(let a)? = spaced else { Issue.record("not parsed"); return }
        #expect(a.url.path == "/Users/a b/My Talk.analysis.md")

        let colon = CLIBackend.classify("✓ Wrote dialogue TXT: /tmp/10:30 standup.txt")
        guard case .artifact(let b)? = colon else { Issue.record("not parsed"); return }
        #expect(b.url.path == "/tmp/10:30 standup.txt")
    }

    @Test("ordinary output is not mistaken for a marker")
    func plainLinesAreNotEvents() {
        for line in [
            "Model   ggml-large-v3-turbo.bin",
            "whisper_print_timings: total time = 1234.56 ms",
            "degraded: diarization unavailable, using generic Speaker labels",
            "",
            "✓",                       // marker with no body
            "▸",                       // phase with no label
            "✓ Wrote something",       // artifact with no separator
        ] {
            #expect(CLIBackend.classify(line) == nil, "false positive on \(line.debugDescription)")
        }
    }

    @Test("every artifact label the CLI emits maps to a usable kind")
    func artifactKinds() {
        // The labels come from cli.py; the kind is derived from the extension,
        // because the label is prose and the extension is a fact.
        let cases: [(String, Artifact.Kind)] = [
            ("/a.speakers.srt", .subtitles),
            ("/a.speakers.txt", .transcript),
            ("/a.speakers.html", .webPage),
            ("/a.analysis.md", .analysis),
            ("/a.frames.json", .data),
        ]
        for (path, expected) in cases {
            let artifact = Artifact(label: "x", url: URL(fileURLWithPath: path))
            #expect(artifact.kind == expected, "wrong kind for \(path)")
        }
    }
}

@Suite("CLI arguments")
struct CLIArgumentTests {

    @Test("a bare request passes only the input")
    func minimalRequest() {
        // Everything unset must stay unset: whiz turns speakers, screenshots
        // and speaker-naming on by itself for video, and passing explicit
        // values here would override a default the CLI is better placed to
        // choose.
        let argv = CLIBackend.arguments(
            for: TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4")))
        #expect(argv == ["transcribe", "/tmp/a.mp4"])
    }

    @Test("options map to the flags they claim to")
    func fullRequest() {
        var request = TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4"))
        request.language = "ru"
        request.speakers = 3
        request.speakerNames = "Alice,Bob,Carol"
        request.screenshots = false
        request.analyze = true
        request.aiModel = "qwen3.5:9b"
        let argv = CLIBackend.arguments(for: request)
        #expect(argv.firstIndex(of: "--language").map { argv[$0 + 1] } == "ru")
        #expect(argv.firstIndex(of: "--speakers").map { argv[$0 + 1] } == "3")
        #expect(argv.firstIndex(of: "--speakers-names").map { argv[$0 + 1] } == "Alice,Bob,Carol")
        #expect(argv.contains("--no-screenshots"))
        #expect(argv.contains("--analyze"))
        #expect(argv.firstIndex(of: "--ai-model").map { argv[$0 + 1] } == "qwen3.5:9b")
    }

    @Test("a zero speaker count is omitted, not sent as zero")
    func zeroSpeakersMeansAuto() {
        // 0 is the CLI's "auto-detect", and it is also what an empty stepper
        // reads as — sending it explicitly would be harmless today but pins
        // the wrong intent.
        var request = TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4"))
        request.speakers = 0
        #expect(!CLIBackend.arguments(for: request).contains("--speakers"))
    }

    @Test("an empty speaker-names field sends no flag, not an empty name")
    func emptyNamesMeansAuto() {
        // Empty means "CLI decides" for every field; a literal empty-string
        // name would clobber a real speaker's label.
        var request = TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4"))
        request.speakerNames = ""
        #expect(!CLIBackend.arguments(for: request).contains("--speakers-names"))
    }
}

@Suite("Locating whiz")
struct WhizLocatorTests {

    @Test("an installed location wins over PATH")
    func installedLocationWins() {
        // pipx's ~/.local/bin should beat whatever PATH happens to hold.
        let installed = URL(fileURLWithPath: "/bin/sh")
        let found = WhizLocator.find(environment: ["PATH": "/usr/bin"], candidates: [installed])
        #expect(found?.path == "/bin/sh")
    }

    @Test("an explicit override wins over discovery")
    func overrideWins() {
        // A GUI app does not inherit the shell PATH, so the override is how a
        // developer points the app at a checkout.
        let url = WhizLocator.find(environment: ["WHIZ_EXECUTABLE": "/bin/sh"])
        #expect(url?.path == "/bin/sh")
    }

    @Test("a non-existent override resolves to nothing rather than a bad path")
    func badOverrideFails() {
        #expect(WhizLocator.find(environment: ["WHIZ_EXECUTABLE": "/nope/whiz"]) == nil)
    }

    @Test("PATH is searched when nothing is installed in the usual places")
    func pathFallback() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let fake = directory.appendingPathComponent("whiz")
        FileManager.default.createFile(atPath: fake.path, contents: Data("#!/bin/sh\n".utf8))
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fake.path)

        // No candidates, so PATH is the only source — otherwise a real whiz in
        // the developer's ~/.local/bin answers instead and the test passes for
        // the wrong reason.
        let found = WhizLocator.find(environment: ["PATH": directory.path], candidates: [])
        #expect(found?.path == fake.path)
    }
}


/// The whole path: spawn a process, stream its output, parse it, honour the
/// exit code.
///
/// Driven by a stub that prints the exact line shapes `whiz/ui.py` emits when
/// piped, so it exercises real `Process` plumbing — chunk boundaries, the
/// unterminated final line, the termination handler — without needing models,
/// media or a whiz install.
@Suite("Running a backend")
struct CLIBackendRunTests {

    /// Writes an executable stub that replays `output` and exits with `code`.
    private func stub(output: String, code: Int32 = 0) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let script = directory.appendingPathComponent("whiz")
        let body = "#!/bin/sh\ncat <<'WHIZEOF'\n\(output)\nWHIZEOF\nexit \(code)\n"
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: script.path)
        return script
    }

    private func collect(_ script: URL) async throws -> [TranscriptionEvent] {
        let box = EventBox()
        try await CLIBackend(executable: script).run(
            TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4")),
            onEvent: { box.append($0) })
        return box.events
    }

    @Test("phases and artifacts are recovered from a real run")
    func parsesARealRun() async throws {
        let script = try stub(output: """
        \u{25B8} transcribing
        whisper_print_timings: total time = 1234.56 ms
        \u{25B8} diarizing
        \u{2713} Wrote labeled SRT: /tmp/a.speakers.srt
        \u{2713} Wrote HTML transcript: /tmp/a.speakers.html
        """)
        defer { try? FileManager.default.removeItem(at: script.deletingLastPathComponent()) }

        let events = try await collect(script)
        let phases = events.compactMap { if case .phase(let p) = $0 { return p } else { return nil } }
        #expect(phases == ["transcribing", "diarizing"])

        let artifacts = events.compactMap { if case .artifact(let a) = $0 { return a } else { return nil } }
        #expect(artifacts.map(\.label) == ["Wrote labeled SRT", "Wrote HTML transcript"])
        #expect(artifacts.map(\.kind) == [.subtitles, .webPage])

        // Every line reaches the log pane, including the ones that also became
        // events — the pane shows the run verbatim.
        let logs = events.compactMap { if case .log(let l) = $0 { return l } else { return nil } }
        #expect(logs.count == 5)
        #expect(logs.contains { $0.hasPrefix("whisper_print_timings") })
    }

    @Test("a nonzero exit throws and carries the reason")
    func nonzeroExitThrows() async throws {
        // NS-4 makes a degraded run exit nonzero deliberately; a UI that
        // ignored the code would present it as success.
        let script = try stub(output: """
        \u{25B8} transcribing
        error: no diarization models and --speakers was requested
        """, code: 2)
        defer { try? FileManager.default.removeItem(at: script.deletingLastPathComponent()) }

        await #expect(throws: TranscriptionFailure.self) { try await collect(script) }
        do { _ = try await collect(script) } catch let failure as TranscriptionFailure {
            guard case .failed(let code, let summary) = failure else {
                Issue.record("wrong case"); return
            }
            #expect(code == 2)
            #expect(summary.contains("no diarization models"),
                    "the last meaningful line should be the reason, got \(summary.debugDescription)")
        }
    }

    @Test("a final line without a trailing newline is not dropped")
    func unterminatedFinalLine() async throws {
        // printf rather than a heredoc: the last line has no newline, which is
        // exactly when a run fails and the reason is in that line.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let script = directory.appendingPathComponent("whiz")
        try "#!/bin/sh\nprintf '\u{2713} Wrote analysis: /tmp/a.analysis.md'\n"
            .write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: script.path)

        let events = try await collect(script)
        let artifacts = events.compactMap { if case .artifact(let a) = $0 { return a } else { return nil } }
        #expect(artifacts.count == 1, "the flush after exit must emit the partial line")
        #expect(artifacts.first?.kind == .analysis)
    }

    @Test("cancelling terminates the process instead of waiting it out")
    func cancellationTerminates() async throws {
        // A 30-second script must end promptly after cancel() — and surface
        // as CancellationError, not as a bogus success (the script exits 0 if
        // left alone) or as a failed(code:summary) the user would read as a
        // pipeline problem. NS-4 makes degraded runs exit nonzero on
        // purpose; a user stop is neither degraded nor a run.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let script = directory.appendingPathComponent("whiz")
        try "#!/bin/sh\nsleep 30\nexit 0\n".write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: script.path)

        let backend = CLIBackend(executable: script)
        let box = EventBox()
        let started = Date()
        let task = Task {
            try await backend.run(
                TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4")),
                onEvent: { box.append($0) })
        }
        try await Task.sleep(for: .milliseconds(300))
        backend.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(Date().timeIntervalSince(started) < 10,
                "cancel() must terminate the script, not wait out its sleep")
    }
}

/// Events arrive on the reader queue, so collection needs a lock.
private final class EventBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [TranscriptionEvent] = []
    func append(_ event: TranscriptionEvent) {
        lock.lock(); defer { lock.unlock() }
        storage.append(event)
    }
    var events: [TranscriptionEvent] {
        lock.lock(); defer { lock.unlock() }
        return storage
    }
}
