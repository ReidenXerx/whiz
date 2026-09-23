import Foundation

/// Runs the `whiz` CLI and turns its output into `TranscriptionEvent`s.
///
/// The CLI is the contract. `whiz/ui.py` degrades to escape-free plain text
/// when stderr is not a TTY, and two of those line shapes are machine-readable:
///
///     ▸ <phase label>
///     ✓ <artifact label>: <path>
///
/// Those shapes are pinned by `tests/test_ui_machine_contract.py` on the Python
/// side, so an edit to `ui.wrote` that would break this parser fails a test
/// there rather than silently producing a UI with no artifacts.
///
/// Everything else is passed through verbatim as `.log`, which is the point:
/// the pane shows the real run, including the warnings a degraded run prints,
/// rather than a sanitised version of it.
struct CLIBackend: TranscriptionBackend {

    /// How to invoke whiz. Resolved once at construction so a missing install
    /// is reported before a run starts rather than mid-pipeline.
    let executable: URL

    /// Marker characters, kept here rather than inline so the contract is
    /// stated in exactly one place on this side too.
    private static let phaseMarker = "▸ "
    private static let artifactMarker = "✓ "

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
        let process = Process()
        process.executableURL = executable
        process.arguments = Self.arguments(for: request)

        // whiz writes progress to stderr and leaves stdout for data. Both are
        // merged: a user reading a log pane does not care which stream a line
        // came from, and interleaving them preserves the real ordering.
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
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

        // `waitUntilExit` blocks, so keep it off the caller's thread.
        await withCheckedContinuation { continuation in
            process.terminationHandler = { _ in continuation.resume() }
        }
        pipe.fileHandleForReading.readabilityHandler = nil
        collector.flush()

        guard process.terminationStatus == 0 else {
            // NS-4: a degraded run exits nonzero deliberately. Surfacing the
            // last meaningful line gives the user the reason rather than a
            // bare code.
            throw TranscriptionFailure.failed(
                code: process.terminationStatus,
                summary: collector.lastMeaningfulLine)
        }
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
        return nil
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
