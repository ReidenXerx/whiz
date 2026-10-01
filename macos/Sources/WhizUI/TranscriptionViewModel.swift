import Foundation
import SwiftUI

/// Owns the run and everything the views bind to.
///
/// The model is deliberately thin: it turns a form into a
/// `TranscriptionRequest`, hands it to the backend, and folds the resulting
/// events into published state. Every bit of pipeline behaviour stays in the
/// CLI — this layer only observes.
@MainActor
final class TranscriptionViewModel: ObservableObject {

    // --- The request, as the form edits it ---

    @Published var inputURL: URL?
    @Published var language = ""
    /// 0 = auto-detect; the CLI is better placed to decide for video input,
    /// which auto-enables diarization on its own.
    @Published var speakers = 0
    @Published var screenshots: ScreenshotsMode = .auto
    @Published var analyze = false
    /// Only sent when `analyze` is on and the field is non-empty — an empty
    /// field must mean "config decides", not "send an empty string".
    @Published var aiModel = ""

    // --- Run state ---

    @Published private(set) var isRunning = false
    @Published private(set) var phase: String?
    @Published private(set) var logLines: [String] = []
    @Published private(set) var artifacts: [Artifact] = []
    /// The failure the run ended on, if any. Views turn this into an alert.
    @Published var errorMessage: String?
    /// True when whiz could not be located — the views then show install
    /// guidance instead of letting the user press Transcribe into a wall.
    @Published private(set) var whizMissing = false
    /// The last finished run's artifacts, loaded from `LastRunStore` so a
    /// fresh window can offer the previous run's results instead of looking
    /// empty. A window that has run something shows its own artifacts; this
    /// is the fresh-window state (see MainView's empty state).
    @Published private(set) var lastRun: LastRunRecord?
    /// What the control bar's status chip says about the last run in THIS
    /// window: running, or how it ended. A window shows its own run; the
    /// persisted record above is what fresh windows see.
    @Published private(set) var lastOutcome: RunOutcome?
    /// When the current run started — drives the running clock.
    @Published private(set) var runStartedAt: Date?
    /// How long the last run took, in seconds. Recorded for every ending,
    /// including stops and failures.
    @Published private(set) var lastDuration: Double?

    /// The log pane is a transcript, not a ledger: keep the tail.
    static let logLineCap = 400

    /// Human clock for the status chip: `3:12`, `1:47:30`. Zero-pads seconds,
    /// shows hours only when hours exist, and clamps a clock-skewed negative
    /// to zero rather than printing nonsense.
    static func formatDuration(_ seconds: Double) -> String {
        let total = max(Int(seconds.rounded()), 0)
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, secs)
        }
        return String(format: "%d:%02d", minutes, secs)
    }

    private var backend: CLIBackend?
    private var runTask: Task<Void, Never>?
    /// Injectable for tests: a suite-backed store rather than the real
    /// domain (see LastRunStore for why the real domain must stay untouched).
    private let lastRunStore: LastRunStore

    init(lastRunStore: LastRunStore = LastRunStore()) {
        self.lastRunStore = lastRunStore
        lastRun = lastRunStore.load()
    }

    var hasError: Bool { errorMessage != nil }

    func start() {
        guard !isRunning, let input = inputURL else { return }

        let resolved: CLIBackend
        do {
            resolved = try CLIBackend()
        } catch {
            whizMissing = true
            return
        }
        whizMissing = false

        var request = TranscriptionRequest(input: input)
        request.language = language.isEmpty ? nil : language
        request.speakers = speakers > 0 ? speakers : nil
        request.screenshots = screenshots.requestValue
        request.analyze = analyze
        request.aiModel = (analyze && !aiModel.isEmpty) ? aiModel : nil

        isRunning = true
        phase = nil
        logLines = []
        artifacts = []
        errorMessage = nil
        backend = resolved
        runStartedAt = Date()
        lastDuration = nil
        lastOutcome = .running

        // The backend reports events from the pipe's reader queue; a stream
        // moves them onto the main actor without polling. `onEvent` is only
        // `yield`, which is safe from any thread.
        let (events, continuation) = AsyncStream<TranscriptionEvent>.makeStream()
        runTask = Task {
            // Consume concurrently with the run: `run` suspends on the
            // termination handler while this loop applies events as they land.
            let consumer = Task {
                for await event in events { apply(event) }
            }
            var outcome: RunOutcome = .finished
            do {
                try await resolved.run(request) { continuation.yield($0) }
            } catch is CancellationError {
                outcome = .stopped
                appendLog("Stopped.")
            } catch let failure as TranscriptionFailure {
                outcome = .failed
                errorMessage = failure.errorDescription
            } catch {
                outcome = .failed
                errorMessage = error.localizedDescription
            }
            continuation.finish()
            await consumer.value
            isRunning = false
            phase = nil
            lastOutcome = outcome
            lastDuration = Date().timeIntervalSince(runStartedAt ?? Date())
            // Persist the results only when the run actually produced
            // something: a failed or stopped run keeps the previous record
            // (saveReturning returns nil for empty lists), so a fresh
            // window never advertises a run that wrote nothing.
            if !artifacts.isEmpty {
                lastRun = lastRunStore.saveReturning(artifacts: artifacts, input: input)
            }
        }
    }

    func stop() {
        backend?.cancel()
        runTask?.cancel()
    }

    /// Clear the "whiz not found" guidance — the next `start` re-checks.
    func dismissMissingWarning() {
        whizMissing = false
    }

    // MARK: - Events

    private func apply(_ event: TranscriptionEvent) {
        switch event {
        case .phase(let label):
            phase = label
        case .log(let line):
            appendLog(line)
        case .artifact(let artifact):
            artifacts.append(artifact)
        }
    }

    private func appendLog(_ line: String) {
        logLines.append(line)
        if logLines.count > Self.logLineCap {
            logLines.removeFirst(logLines.count - Self.logLineCap)
        }
    }
}

/// How the last run in this window ended — the control bar's status chip.
enum RunOutcome: Equatable, Sendable {
    case running
    case finished
    case failed
    case stopped
}

/// The tri-state behind the screenshots control: the CLI's own default for
/// video input, forced on, or forced off.
enum ScreenshotsMode: String, CaseIterable, Identifiable {
    case auto = "Auto (video only)"
    case on = "On"
    case off = "Off"

    var id: String { rawValue }

    /// What the request should carry. `nil` means "pass no flag" — the CLI
    /// decides, which is the whole point of the Auto case.
    var requestValue: Bool? {
        switch self {
        case .auto: return nil
        case .on: return true
        case .off: return false
        }
    }

    /// Compact label for the segmented control — the raw values are whole
    /// sentences meant for the old form's pickers.
    var shortLabel: String {
        switch self {
        case .auto: return "Auto"
        case .on: return "On"
        case .off: return "Off"
        }
    }
}
