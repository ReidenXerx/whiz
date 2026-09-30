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

    /// The log pane is a transcript, not a ledger: keep the tail.
    static let logLineCap = 400

    private var backend: CLIBackend?
    private var runTask: Task<Void, Never>?

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
            do {
                try await resolved.run(request) { continuation.yield($0) }
            } catch is CancellationError {
                appendLog("Stopped.")
            } catch let failure as TranscriptionFailure {
                errorMessage = failure.errorDescription
            } catch {
                errorMessage = error.localizedDescription
            }
            continuation.finish()
            await consumer.value
            isRunning = false
            phase = nil
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
}