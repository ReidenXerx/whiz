import Foundation

/// Runs the `whiz` CLI and turns its output into `TranscriptionEvent`s.
///
/// The CLI is the contract. `whiz/ui.py` degrades to escape-free plain text
/// when stderr is not a TTY, and four of those line shapes are
/// machine-readable:
///
///     ▸ <phase label>
///     ✓ <artifact label>: <absolute path>
///     ? speaker-name: <label> | <quote>[ | <suggested name>]
///     ~ progress: <N>%
///
/// Speaker prompts arrive over a pipe only because the child's environment
/// carries `WHIZ_UI_PIPE=1` (see `childEnvironment`).
///
/// Those shapes are pinned by `tests/test_ui_machine_contract.py` on the Python
/// side, so an edit to `ui.wrote` that would break this parser fails a test
/// there rather than silently producing a UI with no artifacts.
///
/// Everything else is passed through verbatim as `.log`, which is the point:
/// the pane shows the real run, including the warnings a degraded run prints,
/// rather than a sanitised version of it.
///
/// A class rather than a struct because it is mutable at runtime: the user can
/// cancel a run, which terminates a process that is still attached to a live
/// reader queue. All mutable state sits behind a lock (`Process` is neither
/// Sendable nor safe to touch from two tasks at once).
final class CLIBackend: TranscriptionBackend, @unchecked Sendable {

    /// How to invoke whiz. Resolved once at construction so a missing install
    /// is reported before a run starts rather than mid-pipeline.
    let executable: URL

    /// Marker characters, kept here rather than inline so the contract is
    /// stated in exactly one place on this side too.
    private static let phaseMarker = "▸ "
    private static let artifactMarker = "✓ "
    private static let speakerNameMarker = "? speaker-name: "
    private static let progressMarker = "~ progress: "

    /// The live process and the cancelled flag — see `ProcessState`.
    private let state = ProcessState()

    init(executable: URL) {
        self.executable = executable
    }

    /// Locate whiz, preferring an explicit configuration over discovery.
    init() throws {
        guard let found = WhizLocator.find() else { throw TranscriptionFailure.whizNotFound }
        self.executable = found
    }

    /// How long `run` keeps reading after the process exits before it stops
    /// waiting for EOF. A grandchild (whisper-cli, ffmpeg, pip) inherits the
    /// pipe's write end and can hold it open after its parent died — most
    /// often after a `terminate()`, which only signals the direct child. The
    /// bound keeps such a run from wedging; whatever is buffered by then is
    /// still drained.
    static let eofGraceAfterExit: TimeInterval = 2

    func run(
        _ request: TranscriptionRequest,
        onEvent: @escaping @Sendable (TranscriptionEvent) -> Void
    ) async throws {
        try await run(arguments: Self.arguments(for: request), onEvent: onEvent)
    }

    /// Run `executable` with any argv under the same pipe discipline as a
    /// transcription: merged output parsed into events, EOF-and-exit before
    /// done, nonzero exit thrown as `TranscriptionFailure`, cancel → stop.
    /// The setup assistant drives `whiz models download`, `whiz setup
    /// speakers` — and `brew` / `pipx`, whose lines simply arrive as `.log`.
    func run(
        arguments: [String],
        onEvent: @escaping @Sendable (TranscriptionEvent) -> Void
    ) async throws {
        try Task.checkCancellation()

        let process = Process()
        process.executableURL = executable
        process.arguments = arguments

        // whiz writes progress to stderr and leaves stdout for data. Both are
        // merged: a user reading a log pane does not care which stream a line
        // came from, and interleaving them preserves the real ordering.
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        // The CLI names speakers over stdin. A Pipe never EOFs until closed,
        // so a piped run that asks and receives no answer blocks forever
        // inside `input()` — the stdin side stays OPEN while the run is live
        // and answers arrive the moment the user gives one. A run that never
        // asks is unaffected: it never reads stdin.
        let stdinPipe = Pipe()
        process.standardInput = stdinPipe
        // No TTY on a pipe, which is exactly what selects ui.py's plain-text
        // branch — the parser below depends on that. See `childEnvironment`
        // for PATH and the WHIZ_UI_PIPE opt-in.
        process.environment = Self.childEnvironment(
            from: ProcessInfo.processInfo.environment,
            home: FileManager.default.homeDirectoryForCurrentUser.path)

        let collector = LineCollector(onEvent: onEvent)
        let reader = PipeReader(handle: pipe.fileHandleForReading, collector: collector)
        let exited = OneShotSignal()
        // Before `run()`: a process that exits instantly must still be seen.
        process.terminationHandler = { _ in exited.fire() }

        reader.start()
        do {
            try process.run()
        } catch {
            reader.stop()
            process.terminationHandler = nil
            throw error
        }
        state.setProcess(process)
        state.setStdin(stdinPipe.fileHandleForWriting)
        // A cancel() that landed between entry and the spawn has no process to
        // terminate; honour it now rather than letting the run finish anyway.
        state.terminateIfCancelRequested()

        // The run is over when the process has exited AND the pipe hit EOF —
        // in either order. Exit alone is not enough: the last lines (the
        // failure reason, the final artifact) can still sit in the pipe when
        // the termination handler fires. EOF alone is not enough either: a
        // grandchild can keep the write end open, so after exit the wait is
        // bounded and the remainder is drained without blocking.
        await exited.wait()
        let sawEOF = await reader.eof.wait(timeout: Self.eofGraceAfterExit)
        if !sawEOF { reader.drainAvailable() }
        reader.stop()
        collector.flush()
        // Nothing is emitted past this point: the handler is gone, the drain
        // is done, and the reader refuses late callbacks (see PipeReader).
        state.setProcess(nil)
        state.setStdin(nil)

        // A cancelled run is a stop, not a failure: NS-4 makes a degraded run
        // exit nonzero on purpose, and conflating the two would report "the
        // user pressed Stop" as a pipeline problem (or worse, a cancelled
        // failure as success). Only a terminate() actually delivered to a
        // live process counts, and a clean exit 0 is a success even then —
        // Stop pressed as a run finishes must not relabel it "Stopped".
        let cleanExit = process.terminationReason == .exit && process.terminationStatus == 0
        if state.didTerminate, !cleanExit { throw CancellationError() }

        guard process.terminationStatus == 0 else {
            // NS-4: a degraded run exits nonzero deliberately. Surfacing the
            // last meaningful line gives the user the reason rather than a
            // bare code.
            throw TranscriptionFailure.failed(
                code: process.terminationStatus,
                summary: collector.lastMeaningfulLine)
        }
    }

    /// Directories a Finder-launched app's PATH lacks but whiz's tools live
    /// in: Homebrew (Apple silicon, then Intel) for whisper-cli and ffmpeg,
    /// pipx/uv's `~/.local/bin` for whiz itself.
    static func toolDirectories(home: String) -> [String] {
        ["/opt/homebrew/bin", "/usr/local/bin", (home as NSString).appendingPathComponent(".local/bin")]
    }

    /// The child's environment: the app's own, with PATH repaired and the
    /// GUI-pipe opt-in set.
    ///
    /// An app launched from Finder or the Dock gets launchd's minimal PATH
    /// (`/usr/bin:/bin:/usr/sbin:/sbin`), so whiz — which shells out to
    /// whisper-cli and ffmpeg by name — would fail to find them although
    /// everything works from Terminal. Missing tool directories are
    /// PREPENDED (a Homebrew ffmpeg should win over nothing at all, and an
    /// already-present entry keeps its position); duplicates and empty
    /// entries are dropped, otherwise the existing order is untouched.
    ///
    /// `WHIZ_UI_PIPE=1` tells the CLI its non-TTY stdin is this app's answer
    /// channel. Without it, the CLI does not issue `? speaker-name:` prompts
    /// over a pipe — so a shell loop piping into whiz never has its stdin
    /// consumed as speaker names.
    static func childEnvironment(from base: [String: String], home: String) -> [String: String] {
        var environment = base
        let existing = (base["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin")
            .split(separator: ":", omittingEmptySubsequences: true)
            .map(String.init)
        var seen = Set<String>()
        var entries: [String] = []
        for directory in toolDirectories(home: home) where !existing.contains(directory) {
            if seen.insert(directory).inserted { entries.append(directory) }
        }
        for directory in existing where seen.insert(directory).inserted {
            entries.append(directory)
        }
        environment["PATH"] = entries.joined(separator: ":")
        environment["WHIZ_UI_PIPE"] = "1"
        return environment
    }

    /// Terminate the current run at the next safe point; `run` then throws
    /// `CancellationError`. Safe to call before a run starts or after it ends.
    func cancel() {
        state.cancel()
    }

    /// Answer a pending speaker-naming prompt by writing one line to the CLI's
    /// stdin. The prompt's own answer semantics live in the CLI: a name names
    /// the speaker, an empty line confirms the suggestion, and '-' declines
    /// it, keeping the default label.
    func answerSpeakerName(_ answer: String, for prompt: SpeakerNamePrompt) {
        state.writeLine(answer + "\n")
    }

    /// Build the argv for a request.
    ///
    /// Only flags the UI actually exposes. Anything omitted keeps the CLI's own
    /// default, which matters for video: `whiz` turns speakers, screenshots and
    /// speaker-naming on by itself for video input, and passing explicit values
    /// would override a default the CLI is better placed to choose.
    static func arguments(for request: TranscriptionRequest) -> [String] {
        var argv = ["transcribe", request.input.path]
        if let language = request.language, !language.isEmpty {
            argv += ["--language", language]
        }
        if let speakers = request.speakers, speakers > 0 {
            argv += ["--speakers", String(speakers)]
        }
        // One comma-joined token: the CLI flattens either form, and a single
        // token keeps argv readable in the log.
        if let names = request.speakerNames, !names.isEmpty {
            argv += ["--speakers-names", names]
        }
        if let screenshots = request.screenshots {
            argv.append(screenshots ? "--screenshots" : "--no-screenshots")
        }
        if request.analyze {
            argv.append("--analyze")
            if let model = request.aiModel, !model.isEmpty {
                argv += ["--ai-model", model]
            }
        }
        return argv
    }

    /// Classify one output line.
    ///
    /// `nil` for a line that is only log output. Split on the FIRST ": " —
    /// macOS paths contain spaces and colons, and taking the last separator
    /// would truncate them.
    static func classify(_ line: String) -> TranscriptionEvent? {
        if line.hasPrefix(phaseMarker) {
            let label = String(line.dropFirst(phaseMarker.count))
                .trimmingCharacters(in: .whitespaces)
            return label.isEmpty ? nil : .phase(label)
        }
        if line.hasPrefix(artifactMarker) {
            let body = String(line.dropFirst(artifactMarker.count))
            guard let separator = body.range(of: ": ") else { return nil }
            let label = String(body[..<separator.lowerBound])
            let path = String(body[separator.upperBound...])
                .trimmingCharacters(in: .whitespaces)
            // Absolute paths only. The GUI always passes an absolute input,
            // so every real artifact line carries one; a relative "path" is
            // prose that happens to look like the marker — an LLM summary
            // echoed into the log ("✓ Decision: ship Friday") must not
            // become a clickable file.
            guard path.hasPrefix("/") else { return nil }
            return .artifact(Artifact(label: label, url: URL(fileURLWithPath: path)))
        }
        if line.hasPrefix(speakerNameMarker) {
            let body = String(line.dropFirst(speakerNameMarker.count))
            // Segments are joined on " | ": label | quote [| suggestion].
            // The emitter substitutes a lookalike glyph (U+01C0) for " | "
            // inside fields, so no field can contain the separator; taking
            // the label before the FIRST and the suggestion after the LAST
            // is exact, and a separator-shaped quote cannot tear the line.
            guard let labelRange = body.range(of: " | ") else { return nil }
            let label = String(body[..<labelRange.lowerBound])
            var quote = String(body[labelRange.upperBound...])
            guard !label.isEmpty, !quote.isEmpty else { return nil }
            var suggestion = ""
            if let suggestionRange = quote.range(of: " | ", options: .backwards) {
                suggestion = String(quote[suggestionRange.upperBound...])
                quote = String(quote[..<suggestionRange.lowerBound])
            }
            return .speakerName(SpeakerNamePrompt(
                label: label, quote: quote, suggestion: suggestion))
        }
        if line.hasPrefix(progressMarker) {
            // `~ progress: N%` — whole, unpadded, always within 0-100 on the
            // emit side; the bounds here are defence against a malformed
            // hand-written line.
            let body = line.dropFirst(progressMarker.count)
            guard body.hasSuffix("%"),
                  let percent = Int(body.dropLast()) else { return nil }
            guard (0...100).contains(percent) else { return nil }
            return .progress(percent)
        }
        return nil
    }
}

/// The mutable part of a run: the live process, its stdin, and what Stop did.
///
/// `cancel()` may arrive from the UI thread while `run` is awaiting the
/// termination handler, so all of it is lock-guarded. `terminate()` is
/// skipped for a process that never launched or already exited — it throws
/// an Obj-C exception in both cases — and `didTerminate` records only a
/// terminate() that was actually delivered, so a Stop that lands after the
/// process exited cannot relabel a finished run as stopped.
private final class ProcessState: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var stdin: FileHandle?
    private var cancelRequested = false
    private var terminated = false

    func setProcess(_ process: Process?) {
        lock.lock()
        defer { lock.unlock() }
        self.process = process
    }

    func setStdin(_ handle: FileHandle?) {
        lock.lock()
        defer { lock.unlock() }
        // Per-descriptor SIGPIPE suppression: a write to a child that already
        // died then fails with EPIPE instead of killing the app. main.swift
        // also ignores SIGPIPE process-wide; this keeps the write path safe
        // on its own (tests, or any host that did not).
        if let handle { _ = fcntl(handle.fileDescriptor, F_SETNOSIGPIPE, 1) }
        self.stdin = handle
    }

    /// Write one answer line. A closed or dead pipe (the run ended between
    /// the prompt arriving and the user answering) is ignored — the CLI's
    /// own EOF path keeps the default label, and there is nothing to
    /// recover on this side.
    func writeLine(_ text: String) {
        lock.lock()
        defer { lock.unlock() }
        guard let stdin else { return }
        try? stdin.write(contentsOf: Data(text.utf8))
    }

    /// True only when a terminate() reached a running process.
    var didTerminate: Bool {
        lock.lock()
        defer { lock.unlock() }
        return terminated
    }

    func cancel() {
        lock.lock()
        defer { lock.unlock() }
        cancelRequested = true
        terminateLocked()
    }

    /// Honour a cancel() that arrived before the process existed.
    func terminateIfCancelRequested() {
        lock.lock()
        defer { lock.unlock() }
        guard cancelRequested else { return }
        terminateLocked()
    }

    private func terminateLocked() {
        guard let process, process.isRunning else { return }
        process.terminate()
        terminated = true
    }
}

/// A one-shot event that async code can wait for, optionally with a bound.
///
/// Firing is idempotent; every waiter — past or future — sees it.
private final class OneShotSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false
    private var nextID = 0
    private var waiters: [Int: CheckedContinuation<Bool, Never>] = [:]

    func fire() {
        lock.lock()
        guard !fired else { lock.unlock(); return }
        fired = true
        let pending = Array(waiters.values)
        waiters = [:]
        lock.unlock()
        for waiter in pending { waiter.resume(returning: true) }
    }

    func wait() async {
        _ = await wait(timeout: nil)
    }

    /// `true` if the signal fired, `false` if `timeout` elapsed first.
    func wait(timeout: TimeInterval?) async -> Bool {
        await withCheckedContinuation { continuation in
            lock.lock()
            if fired {
                lock.unlock()
                continuation.resume(returning: true)
                return
            }
            let id = nextID
            nextID += 1
            waiters[id] = continuation
            lock.unlock()
            guard let timeout else { return }
            DispatchQueue.global().asyncAfter(deadline: .now() + timeout) { [self] in
                lock.lock()
                let expired = waiters.removeValue(forKey: id)
                lock.unlock()
                expired?.resume(returning: false)
            }
        }
    }
}

/// Reads the merged output pipe into a `LineCollector` until EOF.
///
/// Every read and every hand-off to the collector happens under one lock, and
/// once `stop()` (or a drain) has run, late readability callbacks return
/// without reading — so after `run` has stopped the reader, no event can
/// reach the caller. An empty read is EOF: the handler removes itself and
/// fires `eof`.
private final class PipeReader: @unchecked Sendable {
    private let handle: FileHandle
    private let collector: LineCollector
    private let lock = NSLock()
    private var stopped = false
    let eof = OneShotSignal()

    init(handle: FileHandle, collector: LineCollector) {
        self.handle = handle
        self.collector = collector
    }

    func start() {
        handle.readabilityHandler = { [self] readable in
            lock.lock()
            defer { lock.unlock() }
            guard !stopped else { return }
            let data = readable.availableData
            if data.isEmpty {
                stopped = true
                readable.readabilityHandler = nil
                eof.fire()
                return
            }
            collector.ingest(data)
        }
    }

    /// Read whatever is already buffered without waiting for EOF — the
    /// write end may be held open by a grandchild that outlived the run.
    func drainAvailable() {
        lock.lock()
        defer { lock.unlock() }
        guard !stopped else { return }
        stopped = true
        handle.readabilityHandler = nil
        let fd = handle.fileDescriptor
        _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK)
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            let count = buffer.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress, $0.count) }
            guard count > 0 else { break }  // 0 = EOF, -1 = EAGAIN or error
            collector.ingest(Data(buffer[0..<count]))
        }
    }

    func stop() {
        lock.lock()
        defer { lock.unlock() }
        stopped = true
        handle.readabilityHandler = nil
    }
}

/// Splits a byte stream into lines and classifies them.
///
/// A class with a lock rather than a struct: the readability handler fires on
/// an arbitrary queue, and a partial line at a chunk boundary has to survive
/// between callbacks. Splitting per-chunk instead would tear a marker in half
/// and drop the artifact.
///
/// Buffers raw BYTES and splits on 0x0A before decoding. A pipe chunk can end
/// in the middle of a multi-byte character (every marker glyph is one, and so
/// is Cyrillic speech); decoding chunk-by-chunk fails on such a chunk and
/// would lose it whole. Each complete line is decoded leniently, so even
/// invalid bytes cost one replacement character, never a line.
final class LineCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()
    private var lastLine = ""
    private let onEvent: @Sendable (TranscriptionEvent) -> Void

    init(onEvent: @escaping @Sendable (TranscriptionEvent) -> Void) {
        self.onEvent = onEvent
    }

    func ingest(_ data: Data) {
        lock.lock()
        pending.append(data)
        var lines: [String] = []
        while let newline = pending.firstIndex(of: 0x0A) {
            lines.append(String(decoding: pending[pending.startIndex..<newline], as: UTF8.self))
            pending.removeSubrange(pending.startIndex...newline)
        }
        lock.unlock()
        for line in lines { emit(line) }
    }

    /// Emit whatever is left when the process ends — the final line often has
    /// no trailing newline, and it is frequently the error message.
    func flush() {
        lock.lock()
        let remainder = pending
        pending = Data()
        lock.unlock()
        if !remainder.isEmpty { emit(String(decoding: remainder, as: UTF8.self)) }
    }

    var lastMeaningfulLine: String {
        lock.lock()
        defer { lock.unlock() }
        return lastLine
    }

    private func emit(_ raw: String) {
        let line = raw.trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty else { return }
        lock.lock()
        lastLine = line
        lock.unlock()
        onEvent(.log(line))
        if let event = CLIBackend.classify(line) { onEvent(event) }
    }
}
