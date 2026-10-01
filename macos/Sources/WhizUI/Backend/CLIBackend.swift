import Foundation

/// Runs the `whiz` CLI and turns its output into `TranscriptionEvent`s.
///
/// The CLI is the contract. `whiz/ui.py` degrades to escape-free plain text
/// when stderr is not a TTY, and three of those line shapes are
/// machine-readable:
///
///     ▸ <phase label>
///     ✓ <artifact label>: <path>
///     ? speaker-name: <label> | <quote>[ | <suggested name>]
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

    func run(
        _ request: TranscriptionRequest,
        onEvent: @escaping @Sendable (TranscriptionEvent) -> Void
    ) async throws {
        try Task.checkCancellation()

        let process = Process()
        process.executableURL = executable
        process.arguments = Self.arguments(for: request)

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
        // branch — the parser below depends on that.
        process.environment = ProcessInfo.processInfo.environment

        let collector = LineCollector(onEvent: onEvent)
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            collector.ingest(data)
        }

        try process.run()
        state.setProcess(process)
        state.setStdin(stdinPipe.fileHandleForWriting)
        // A cancel() that landed between entry and the spawn has no process to
        // terminate; honour it now rather than letting the run finish anyway.
        if state.isCancelled, process.isRunning {
            process.terminate()
        }

        // `waitUntilExit` blocks, so keep it off the caller's thread.
        await withCheckedContinuation { continuation in
            process.terminationHandler = { _ in continuation.resume() }
        }
        pipe.fileHandleForReading.readabilityHandler = nil
        collector.flush()
        state.setStdin(nil)

        // A cancelled run is a stop, not a failure: NS-4 makes a degraded run
        // exit nonzero on purpose, and conflating the two would report "the
        // user pressed Stop" as a pipeline problem (or worse, a cancelled
        // failure as success).
        if state.isCancelled { throw CancellationError() }

        guard process.terminationStatus == 0 else {
            // NS-4: a degraded run exits nonzero deliberately. Surfacing the
            // last meaningful line gives the user the reason rather than a
            // bare code.
            throw TranscriptionFailure.failed(
                code: process.terminationStatus,
                summary: collector.lastMeaningfulLine)
        }
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
            guard !path.isEmpty else { return nil }
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
        return nil
    }
}

/// The mutable part of a run: the live process plus the cancelled flag.
///
/// `cancel()` may arrive from the UI thread while `run` is awaiting the
/// termination handler, so both the flag and the process are lock-guarded.
/// `terminate()` is skipped for a process that never launched or already
/// exited — it throws an Obj-C exception in both cases.
private final class ProcessState: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var stdin: FileHandle?
    private var cancelled = false

    func setProcess(_ process: Process?) {
        lock.lock()
        defer { lock.unlock() }
        self.process = process
    }

    func setStdin(_ handle: FileHandle?) {
        lock.lock()
        defer { lock.unlock() }
        self.stdin = handle
    }

    /// Write one answer line. A closed handle (the run ended between the
    /// prompt arriving and the user answering) silently no-ops — the CLI's
    /// own EOF path keeps the default label. `write(_:)` is the non-throwing
    /// legacy API on this SDK; a throwing call would need `try`, but there
    /// is nothing to recover from on a dead pipe.
    func writeLine(_ text: String) {
        lock.lock()
        defer { lock.unlock() }
        guard let stdin else { return }
        stdin.write(text.data(using: .utf8) ?? Data())
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func cancel() {
        lock.lock()
        defer { lock.unlock() }
        cancelled = true
        guard let process, process.isRunning else { return }
        process.terminate()
    }
}

/// Splits a byte stream into lines and classifies them.
///
/// A class with a lock rather than a struct: the readability handler fires on
/// an arbitrary queue, and a partial line at a chunk boundary has to survive
/// between callbacks. Splitting per-chunk instead would tear a marker in half
/// and drop the artifact.
private final class LineCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = ""
    private var lastLine = ""
    private let onEvent: @Sendable (TranscriptionEvent) -> Void

    init(onEvent: @escaping @Sendable (TranscriptionEvent) -> Void) {
        self.onEvent = onEvent
    }

    func ingest(_ data: Data) {
        guard let text = String(data: data, encoding: .utf8) else { return }
        lock.lock()
        pending += text
        var lines = pending.components(separatedBy: "\n")
        pending = lines.removeLast()  // trailing partial line
        lock.unlock()
        for line in lines { emit(line) }
    }

    /// Emit whatever is left when the process ends — the final line often has
    /// no trailing newline, and it is frequently the error message.
    func flush() {
        lock.lock()
        let remainder = pending
        pending = ""
        lock.unlock()
        if !remainder.isEmpty { emit(remainder) }
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
