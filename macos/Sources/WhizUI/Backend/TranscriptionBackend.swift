import Foundation

/// One transcription run, reported as it happens.
///
/// Deliberately describes *what the user sees*, not how the work is done, so
/// the same events can come from the CLI today and from an in-process C++ core
/// later without the UI noticing.
enum TranscriptionEvent: Equatable, Sendable {
    /// A pipeline stage started — `whiz` calls these phases ("transcribing",
    /// "diarizing", "capturing frames").
    case phase(String)
    /// A line of output, for the log pane. Includes lines that also produced a
    /// `phase` or `artifact`, so the pane shows the run verbatim.
    case log(String)
    /// An artifact landed on disk.
    case artifact(Artifact)
}

/// A file a run produced, named the way the CLI announced it.
struct Artifact: Equatable, Sendable, Identifiable {
    var id: URL { url }
    var label: String
    var url: URL

    /// What the UI offers to do with it. Derived from the extension rather
    /// than the label, because the label is prose and the extension is a fact.
    var kind: Kind {
        switch url.pathExtension.lowercased() {
        case "srt": return .subtitles
        case "txt": return .transcript
        case "html": return .webPage
        case "md": return .analysis
        case "json": return .data
        default: return .other
        }
    }

    enum Kind: Sendable { case subtitles, transcript, webPage, analysis, data, other }
}

/// Options for a run. Mirrors the CLI flags the UI exposes — not all of them,
/// only the ones worth a control.
struct TranscriptionRequest: Equatable, Sendable {
    var input: URL
    var language: String?
    var speakers: Int?
    var screenshots: Bool?
    var analyze: Bool = false
    var aiModel: String?
}

/// Anything that can run a transcription.
///
/// The seam that keeps this package from caring whether whiz is a subprocess.
/// When the shared core becomes C++ a second conformance replaces
/// `CLIBackend` and the UI does not change.
protocol TranscriptionBackend: Sendable {
    /// Run to completion, reporting progress through `onEvent`.
    ///
    /// Throws `TranscriptionFailure.failed` on a nonzero exit — which is
    /// load-bearing rather than incidental: NS-4 makes a degraded run exit
    /// nonzero *on purpose*, so a UI that ignored the exit code would present
    /// a silently-degraded run as a success.
    func run(
        _ request: TranscriptionRequest,
        onEvent: @escaping @Sendable (TranscriptionEvent) -> Void
    ) async throws

    /// Ask the run to stop at the next safe point.
    ///
    /// `run` then throws `CancellationError` rather than
    /// `TranscriptionFailure`: a stop is a stop, not a degraded run (NS-4).
    /// Not throwing at all — finishing normally where it was — is acceptable;
    /// the caller treats cancellation and completion alike for cleanup.
    func cancel()
}

extension TranscriptionBackend {
    /// Backends that cannot force a stop (an in-process core stops through
    /// Swift task cancellation instead) get a no-op.
    func cancel() {}
}

enum TranscriptionFailure: LocalizedError, Equatable {
    /// The CLI exited nonzero. `summary` is the last meaningful line it
    /// printed, which is usually the actual reason.
    case failed(code: Int32, summary: String)
    case whizNotFound

    var errorDescription: String? {
        switch self {
        case .failed(let code, let summary):
            return summary.isEmpty ? "whiz exited with code \(code)." : summary
        case .whizNotFound:
            return "Could not find the whiz command. Install it with:\n"
                 + "  pipx install whiz"
        }
    }
}
