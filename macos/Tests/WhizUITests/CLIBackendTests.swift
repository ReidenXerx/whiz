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
            "✓ Decision: ship Friday", // LLM prose shaped like the marker
            "✓ Wrote labeled SRT: a.speakers.srt",  // relative: not a file the UI can trust
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


@Suite("Child environment")
struct ChildEnvironmentTests {

    private let home = "/Users/someone"

    @Test("a Finder-launched PATH gains the tool directories, prepended")
    func finderPathIsRepaired() {
        // launchd hands a Finder-launched app this PATH; whiz shells out to
        // whisper-cli and ffmpeg by name and would find neither.
        let env = CLIBackend.childEnvironment(
            from: ["PATH": "/usr/bin:/bin:/usr/sbin:/sbin"], home: home)
        #expect(env["PATH"] == "/opt/homebrew/bin:/usr/local/bin:/Users/someone/.local/bin"
                + ":/usr/bin:/bin:/usr/sbin:/sbin")
    }

    @Test("entries already present keep their place and are not duplicated")
    func existingEntriesKeepOrder() {
        // A shell-launched app already has a rich PATH; its order is the
        // user's choice and must survive, minus duplicates and empties.
        let env = CLIBackend.childEnvironment(
            from: ["PATH": "/usr/bin:/usr/local/bin::/usr/bin:/opt/homebrew/bin"], home: home)
        #expect(env["PATH"] == "/Users/someone/.local/bin:/usr/bin:/usr/local/bin:/opt/homebrew/bin")
    }

    @Test("a missing PATH starts from the system default")
    func missingPath() {
        let env = CLIBackend.childEnvironment(from: [:], home: home)
        #expect(env["PATH"] == "/opt/homebrew/bin:/usr/local/bin:/Users/someone/.local/bin"
                + ":/usr/bin:/bin:/usr/sbin:/sbin")
    }

    @Test("the GUI-pipe opt-in is set and nothing else is lost")
    func pipeOptInIsSet() {
        // WHIZ_UI_PIPE=1 is the contract that lets the CLI ask speaker names
        // over a non-TTY stdin; without it the naming bar never appears.
        let env = CLIBackend.childEnvironment(
            from: ["PATH": "/usr/bin", "WHIZ_CONFIG_DIR": "/tmp/cfg", "LANG": "ru_RU.UTF-8"], home: home)
        #expect(env["WHIZ_UI_PIPE"] == "1")
        #expect(env["WHIZ_CONFIG_DIR"] == "/tmp/cfg")
        #expect(env["LANG"] == "ru_RU.UTF-8")
    }
}

@Suite("Line splitting")
struct LineCollectorTests {

    private func split(_ chunks: [Data], flush: Bool = true) -> [TranscriptionEvent] {
        let box = EventBox()
        let collector = LineCollector(onEvent: { box.append($0) })
        for chunk in chunks { collector.ingest(chunk) }
        if flush { collector.flush() }
        return box.events
    }

    @Test("a chunk boundary inside a multi-byte character loses nothing")
    func utf8SplitAcrossChunks() {
        // Every marker glyph is multi-byte, and so is Cyrillic speech. Cut
        // the bytes at every possible offset: decoding chunk-by-chunk failed
        // on a chunk ending mid-character and dropped it whole.
        let bytes = Data("▸ транскрипция\n✓ Wrote analysis: /tmp/Отчёт.analysis.md\n".utf8)
        for cut in 1..<bytes.count {
            let events = split([bytes.prefix(cut), bytes.suffix(from: cut)])
            #expect(events.contains(.phase("транскрипция")), "lost the phase at cut \(cut)")
            #expect(events.contains(.artifact(Artifact(
                label: "Wrote analysis",
                url: URL(fileURLWithPath: "/tmp/Отчёт.analysis.md")))),
                "lost the artifact at cut \(cut)")
        }
    }

    @Test("an unterminated multi-byte tail is decoded by flush")
    func flushDecodesTail() {
        let bytes = Data("▸ диаризация".utf8)
        let events = split([bytes.prefix(5), bytes.suffix(from: 5)])
        #expect(events.contains(.phase("диаризация")))
    }

    @Test("invalid bytes cost a replacement character, not the line")
    func invalidBytesKeepTheLine() {
        var bytes = Data("error: bad ".utf8)
        bytes.append(contentsOf: [0xFF, 0xFE])
        bytes.append(Data(" byte\n".utf8))
        let logs = split([bytes]).compactMap { if case .log(let l) = $0 { return l } else { return nil } }
        #expect(logs.count == 1)
        #expect(logs.first?.hasPrefix("error: bad ") == true)
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

    @Test("a speaker-name prompt line yields a prompt event")
    func speakerPromptLine() {
        let event = CLIBackend.classify("? speaker-name: Speaker A | hello world | Alice")
        #expect(event == .speakerName(SpeakerNamePrompt(
            label: "Speaker A", quote: "hello world", suggestion: "Alice")))

        let unknown = CLIBackend.classify("? speaker-name: Speaker B | some words")
        #expect(unknown == .speakerName(SpeakerNamePrompt(
            label: "Speaker B", quote: "some words", suggestion: "")))

        // The emitter substitutes the lookalike U+01C0 stroke for " | "
        // inside fields, so a separator-shaped quote arrives intact and the
        // first/last split stays exact.
        let tricky = CLIBackend.classify("? speaker-name: Speaker A | he said ' ǀ ' loud | Bob")
        let parts: SpeakerNamePrompt? = {
            if case .speakerName(let p) = tricky { return p }
            return nil
        }()
        #expect(parts?.quote == "he said ' ǀ ' loud")
        #expect(parts?.suggestion == "Bob")

        // Malformed shapes are not events — a log line is not a question.
        #expect(CLIBackend.classify("? speaker-name: ") == nil)
        #expect(CLIBackend.classify("? speaker-name: Speaker A") == nil)
    }

    @Test("a progress line becomes a percent event")
    func progressLine() {
        #expect(CLIBackend.classify("~ progress: 42%") == .progress(42))
        #expect(CLIBackend.classify("~ progress: 0%") == .progress(0))
        #expect(CLIBackend.classify("~ progress: 100%") == .progress(100))

        // Malformed shapes are not events — a log line is not a percent.
        #expect(CLIBackend.classify("~ progress: ") == nil)
        #expect(CLIBackend.classify("~ progress: 4x%") == nil)
        #expect(CLIBackend.classify("~ progress: 101%") == nil)
        #expect(CLIBackend.classify("~ progress: 42") == nil)
        #expect(CLIBackend.classify("~progress: 42%") == nil)
    }

    @Test("a naming answer reaches the run's stdin")
    func answersReachStdin() async throws {
        // A stub that replays a machine prompt and then reads one stdin line:
        // answering prompts through the process's stdin pipe is exactly how
        // the naming bar talks to a live run. `read -t` bounds the wait so a
        // backend that never writes fails the assertion instead of hanging
        // the suite; `cat` and /usr/bin/printf are external processes so
        // their output reaches the pipe immediately (sh's builtin printf
        // would buffer until exit — after the read that needs it).
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let script = directory.appendingPathComponent("whiz")
        // A local first: method-call syntax binds to the last operand of a
        // chained +, so `.write` directly on the concatenation does not even
        // compile (String + Void is not String).
        let body = "#!/bin/sh\n"
            + "cat <<'WHIZEOF'\n"
            + "? speaker-name: Speaker A | hello there | Alice\n"
            + "WHIZEOF\n"
            + "read -t 10 name\n"
            + "/usr/bin/printf 'named: %s\\n' \"$name\"\n"
            + "exit 0\n"
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: script.path)

        let box = EventBox()
        let backend = CLIBackend(executable: script)
        let runTask = Task {
            try await backend.run(
                TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4")),
                onEvent: { box.append($0) })
        }

        // The prompt must arrive before there is anything to answer.
        var prompt: SpeakerNamePrompt?
        for _ in 0..<100 {
            prompt = box.events.compactMap {
                if case .speakerName(let p) = $0 { return p } else { return nil }
            }.first
            if prompt != nil { break }
            try await Task.sleep(for: .milliseconds(50))
        }
        guard let prompt else {
            backend.cancel()
            _ = try? await runTask.value
            Issue.record("prompt never arrived")
            return
        }

        backend.answerSpeakerName("Vadim", for: prompt)

        // The script only finishes once stdin delivered the answer, so a
        // clean completion is itself part of the claim.
        try await runTask.value
        let logs = box.events.compactMap { if case .log(let l) = $0 { return l } else { return nil } }
        #expect(logs.contains { $0.contains("named: Vadim") },
                "the answer must reach the run's stdin, got \(logs)")
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

extension CLIBackendRunTests {

    /// An executable stub from a raw shell body.
    fileprivate func script(_ body: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("whiz")
        try ("#!/bin/sh\n" + body).write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    fileprivate func logs(_ events: [TranscriptionEvent]) -> [String] {
        events.compactMap { if case .log(let l) = $0 { return l } else { return nil } }
    }

    @Test("every line is delivered before run returns, and none after")
    func drainsBeforeReturning() async throws {
        // The drain race: the termination handler fired while the tail was
        // still in the pipe, so the last lines were dropped — and a reader
        // still mid-chunk could emit after run returned (after the view
        // model finished its stream). 20000 lines make the tail large.
        let url = try script("/usr/bin/seq 1 20000\n/usr/bin/printf 'tail-no-newline'\n")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let box = EventBox()
        try await CLIBackend(executable: url).run(
            TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4")),
            onEvent: { box.append($0) })
        let atReturn = box.events.count
        let lines = logs(box.events)
        #expect(lines.count == 20001)
        #expect(lines.first == "1")
        #expect(lines.dropLast().last == "20000")
        #expect(lines.last == "tail-no-newline")

        try await Task.sleep(for: .milliseconds(300))
        #expect(box.events.count == atReturn, "an event arrived after run returned")
    }

    @Test("the child sees the GUI-pipe opt-in and a repaired PATH")
    func childEnvironmentReachesTheProcess() async throws {
        let url = try script("echo \"pipe=$WHIZ_UI_PIPE\"\necho \"path=$PATH\"\n")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let box = EventBox()
        try await CLIBackend(executable: url).run(
            TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4")),
            onEvent: { box.append($0) })
        let lines = logs(box.events)
        #expect(lines.contains("pipe=1"))
        #expect(lines.contains { $0.hasPrefix("path=") && $0.contains("/opt/homebrew/bin") })
    }

    @Test("a grandchild holding the pipe open does not wedge the run")
    func grandchildDoesNotWedge() async throws {
        // whisper-cli or pip can outlive whiz (after a terminate() only the
        // direct child is signalled) while still holding the write end. The
        // EOF wait after exit is bounded; buffered output is still drained.
        let url = try script("echo first\nsleep 8 &\necho last\nexit 0\n")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let box = EventBox()
        let started = Date()
        try await CLIBackend(executable: url).run(
            TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4")),
            onEvent: { box.append($0) })
        #expect(Date().timeIntervalSince(started) < 6,
                "run must not wait for the grandchild's EOF")
        #expect(logs(box.events) == ["first", "last"])
    }

    @Test("Stop pressed after a clean exit does not relabel the run as stopped")
    func cancelAfterExitIsNotAStop() async throws {
        // The process exits 0 at once; a grandchild keeps run() in its
        // post-exit EOF wait, which is exactly the window a late Stop lands
        // in. The grandchild announces once `kill -0` fails on its parent —
        // i.e. the shell has been reaped — so the Stop below provably lands
        // after the exit, however loaded the machine is.
        let url = try script("""
            echo done
            ( while kill -0 $$ 2>/dev/null; do sleep 0.05; done; echo parent-gone; sleep 3 ) &
            exit 0

            """)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let backend = CLIBackend(executable: url)
        let box = EventBox()
        let task = Task {
            try await backend.run(
                TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4")),
                onEvent: { box.append($0) })
        }
        for _ in 0..<200 where !logs(box.events).contains("parent-gone") {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(logs(box.events).contains("parent-gone"))
        backend.cancel()
        // No throw: a run that exited 0 before Stop reached it finished.
        try await task.value
    }

    @Test("answering a prompt after the run's process died does not crash")
    func answerToDeadChildIsIgnored() async throws {
        // The shell exits right after asking; its stdin read end closes with
        // it, so the answer hits a dead pipe (EPIPE / SIGPIPE territory).
        let url = try script("""
            cat <<'WHIZEOF'
            ? speaker-name: Speaker A | hello there
            WHIZEOF
            sleep 2 &
            exit 0

            """)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let box = EventBox()
        let backend = CLIBackend(executable: url)
        let task = Task {
            try await backend.run(
                TranscriptionRequest(input: URL(fileURLWithPath: "/tmp/a.mp4")),
                onEvent: { box.append($0) })
        }
        var prompt: SpeakerNamePrompt?
        for _ in 0..<100 where prompt == nil {
            prompt = box.events.compactMap { if case .speakerName(let p) = $0 { return p } else { return nil } }.first
            if prompt == nil { try await Task.sleep(for: .milliseconds(20)) }
        }
        let asked = try #require(prompt)
        try await Task.sleep(for: .milliseconds(300))
        backend.answerSpeakerName("Vadim", for: asked)
        backend.answerSpeakerName("again", for: asked)
        try await task.value
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
