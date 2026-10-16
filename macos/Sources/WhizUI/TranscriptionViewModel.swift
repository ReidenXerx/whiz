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
    /// Names for the CLI's `--speakers-names`: comma-separated, assigned by
    /// total talk time (most talkative first). Empty = keep default labels —
    /// the CLI still auto-names from voice profiles when it can.
    @Published var speakerNames = ""
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
    /// The recent finished runs, newest first, loaded from `LastRunStore`.
    /// A fresh window offers the whole recent set, not just the newest one —
    /// the artifacts are on disk either way.
    @Published private(set) var recentRuns: [LastRunRecord] = []
    /// What the control bar's status chip says about the last run in THIS
    /// window: running, or how it ended. A window shows its own run; the
    /// persisted record above is what fresh windows see.
    @Published private(set) var lastOutcome: RunOutcome?
    /// When the current run started — drives the running clock.
    @Published private(set) var runStartedAt: Date?
    /// How long the last run took, in seconds. Recorded for every ending,
    /// including stops and failures.
    @Published private(set) var lastDuration: Double?
    /// The speaker-naming prompt the run is currently waiting on, if any.
    /// Prompts arrive one at a time (the CLI asks sequentially); answering
    /// releases the run to continue.
    @Published private(set) var pendingSpeakerPrompt: SpeakerNamePrompt?
    /// How far along the current phase is, 0-100 — `nil` when the phase
    /// reports nothing measurable. Reset on every phase change so a new
    /// stage starts at "no percent yet", not stuck at the last one's 100.
    @Published private(set) var progressPercent: Int?
    /// The phases this run has been through, in order — the run view's
    /// timeline. Each step closes when the next phase starts (or the run
    /// ends), so a finished step carries its own duration.
    @Published private(set) var steps: [PhaseStep] = []
    /// The input of the run the sidebar's "now" row stands for: set when a
    /// run starts, cleared when it finishes with results (it then lives in
    /// history) or when the user dismisses a failed/stopped one.
    @Published private(set) var liveRunInput: URL?

    // --- Navigation ---

    /// What the window shows. Persisted, so a relaunch lands where the user
    /// left off; a stale `.live` (no run survives a relaunch) or a run that
    /// is no longer in history reads as the new-run form.
    @Published var selection: SidebarItem = .compose {
        didSet { if selection != oldValue { optionsStore.saveSelection(selection) } }
    }
    /// Set by ⌘O from anywhere; the window owns the file importer.
    @Published var isPickingFile = false

    /// The log pane is a transcript, not a ledger: keep the tail.
    static let logLineCap = 400

    /// Human clock for the status chip: `3:12`, `1:47:30`. Zero-pads seconds,
    /// shows hours only when hours exist, and clamps a clock-skewed negative
    /// to zero rather than printing nonsense.
    nonisolated static func formatDuration(_ seconds: Double) -> String {
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
    private let optionsStore: OptionsStore
    /// Injectable for tests: a stub executable instead of discovering the
    /// developer's real whiz (which would run a real pipeline mid-suite, and
    /// fail on a machine without one).
    private let makeBackend: () throws -> CLIBackend

    init(lastRunStore: LastRunStore = LastRunStore(),
         optionsStore: OptionsStore = OptionsStore(),
         makeBackend: @escaping () throws -> CLIBackend = { try CLIBackend() }) {
        self.lastRunStore = lastRunStore
        self.optionsStore = optionsStore
        self.makeBackend = makeBackend
        lastRun = lastRunStore.load()
        recentRuns = lastRunStore.loadHistory()
        restoreOptions()
        selection = Self.restorable(optionsStore.loadSelection(), history: recentRuns)
    }

    /// A persisted selection only survives if it still points at something.
    nonisolated static func restorable(_ item: SidebarItem?, history: [LastRunRecord]) -> SidebarItem {
        switch item {
        case .run(let path) where history.contains(where: { $0.inputPath == path }):
            return .run(path)
        default:
            return .compose
        }
    }

    /// The history record a `.run` selection names, if it still exists.
    func record(for path: String) -> LastRunRecord? {
        recentRuns.first { $0.inputPath == path }
    }

    /// Choose the next run's input and show the form with it.
    func choose(_ url: URL) {
        guard !isRunning else { return }
        inputURL = url
        selection = .compose
    }

    /// "Transcribe Again": the form, pre-filled with that run's input. The
    /// options stay the persisted ones — the record does not carry options.
    func prepareRerun(_ record: LastRunRecord) {
        choose(record.inputURL)
    }

    /// Forget a run (its files stay on disk).
    func removeFromHistory(_ record: LastRunRecord) {
        lastRunStore.remove(inputPath: record.inputPath)
        recentRuns = lastRunStore.loadHistory()
        lastRun = recentRuns.first
        if selection == .run(record.inputPath) { selection = .compose }
    }

    /// Put away a failed or stopped run's live view.
    func dismissLiveRun() {
        guard !isRunning else { return }
        liveRunInput = nil
        steps = []
        logLines = []
        artifacts = []
        lastOutcome = nil
        if selection == .live { selection = .compose }
    }

    /// Seconds left in a phase from its percent and how long it has run —
    /// `nil` until the estimate means something (a few percent in, a few
    /// seconds in), so the UI never flashes "2 hours left" at 1%.
    nonisolated static func estimatedRemaining(percent: Int?, elapsed: TimeInterval) -> TimeInterval? {
        guard let percent, percent >= 3, percent < 100, elapsed >= 5 else { return nil }
        return elapsed * Double(100 - percent) / Double(percent)
    }

    /// Bring the persisted options into the form once, at birth — the form
    /// then owns them until `start()` writes them back.
    private func restoreOptions() {
        let record = optionsStore.load()
        language = record.language
        speakers = record.speakers
        speakerNames = record.speakerNames
        if let mode = ScreenshotsMode(rawValue: record.screenshotsRaw) {
            screenshots = mode
        }
        analyze = record.analyze
        aiModel = record.aiModel
    }

    var hasError: Bool { errorMessage != nil }

    func start() {
        guard !isRunning, let input = inputURL else { return }

        let resolved: CLIBackend
        do {
            resolved = try makeBackend()
        } catch {
            whizMissing = true
            return
        }
        whizMissing = false

        var request = TranscriptionRequest(input: input)
        request.language = language.isEmpty ? nil : language
        request.speakers = speakers > 0 ? speakers : nil
        request.speakerNames = speakerNames.trimmingCharacters(in: .whitespaces).isEmpty
            ? nil : speakerNames
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
        pendingSpeakerPrompt = nil
        progressPercent = nil
        steps = []
        liveRunInput = input
        selection = .live
        optionsStore.save(OptionsRecord(
            language: language,
            speakers: speakers,
            speakerNames: speakerNames,
            screenshotsRaw: screenshots.rawValue,
            analyze: analyze,
            aiModel: aiModel))

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
            closeCurrentStep(outcome: outcome)
            isRunning = false
            phase = nil
            progressPercent = nil
            // A prompt the run never got an answer to dies with the run: left
            // up, the naming bar would offer a Submit that writes to nothing.
            pendingSpeakerPrompt = nil
            lastOutcome = outcome
            lastDuration = Date().timeIntervalSince(runStartedAt ?? Date())
            // Persist the results only when the run actually produced
            // something: a failed or stopped run keeps the previous record
            // (saveReturning returns nil for empty lists), so a fresh
            // window never advertises a run that wrote nothing.
            if !artifacts.isEmpty {
                lastRun = lastRunStore.saveReturning(
                    artifacts: artifacts, input: input, duration: lastDuration)
                recentRuns = lastRunStore.loadHistory()
            }
            // A clean run with results now lives in history: show it there.
            // Failures and stops stay on the live view, with the reason.
            if outcome == .finished, !artifacts.isEmpty {
                liveRunInput = nil
                if selection == .live { selection = .run(input.path) }
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

    /// Answer the pending speaker-naming prompt, unblocking the run. The
    /// CLI's answer semantics apply: a name names the speaker, and '-' is
    /// the decline sentinel — the default label is kept even when a
    /// suggestion exists, which is how the UI rejects a wrong voice-profile
    /// auto-match instead of confirming it.
    func answerSpeakerName(_ name: String) {
        guard let prompt = pendingSpeakerPrompt else { return }
        backend?.answerSpeakerName(name, for: prompt)
        pendingSpeakerPrompt = nil
    }

    // MARK: - Events

    private func apply(_ event: TranscriptionEvent) {
        switch event {
        case .phase(let label):
            phase = label
            progressPercent = nil
            closeCurrentStep(outcome: nil)
            steps.append(PhaseStep(id: steps.count, label: label, startedAt: Date()))
        case .log(let line):
            appendLog(line)
        case .artifact(let artifact):
            artifacts.append(artifact)
        case .speakerName(let prompt):
            pendingSpeakerPrompt = prompt
        case .progress(let percent):
            progressPercent = percent
        }
    }

    /// End the open step: when the next phase starts (`outcome` nil, it
    /// finished), or when the run ends, which also says HOW it ended.
    private func closeCurrentStep(outcome: RunOutcome?) {
        guard let last = steps.indices.last, steps[last].endedAt == nil else { return }
        steps[last].endedAt = Date()
        switch outcome {
        case .failed: steps[last].state = .failed
        case .stopped: steps[last].state = .stopped
        default: steps[last].state = .done
        }
    }

    private func appendLog(_ line: String) {
        logLines.append(line)
        if logLines.count > Self.logLineCap {
            logLines.removeFirst(logLines.count - Self.logLineCap)
        }
    }
}

/// One phase of a run, as the timeline shows it.
struct PhaseStep: Identifiable, Equatable, Sendable {
    enum State: Equatable, Sendable { case active, done, failed, stopped }

    let id: Int
    /// The CLI's own phase label ("transcribing", "analyzing (chained)").
    var label: String
    var startedAt: Date
    var endedAt: Date?
    var state: State = .active

    /// The CLI's phase names are engineer's words ("diarizing"); the step
    /// list speaks the user's. Unknown phases fall back to the label,
    /// capitalized — a new CLI phase still shows, just less polished.
    static let friendlyTitles: [String: String] = [
        "extracting audio": "Extracting audio",
        "diarizing": "Telling speakers apart",
        "transcribing": "Transcribing speech",
        "merging speakers": "Matching words to speakers",
        "capturing frames": "Capturing video frames",
        "writing HTML transcript": "Writing the HTML transcript",
        "choosing AI model": "Choosing an AI model",
        "analyzing": "Writing the AI summary",
    ]

    /// "transcribing" → "Transcribing speech"; the chained-analysis suffix is
    /// CLI bookkeeping, not something a reader needs.
    var title: String {
        let trimmed = label.replacingOccurrences(of: " (chained)", with: "")
        if let friendly = Self.friendlyTitles[trimmed] { return friendly }
        guard let first = trimmed.first else { return trimmed }
        return first.uppercased() + trimmed.dropFirst()
    }

    var duration: TimeInterval? { endedAt.map { $0.timeIntervalSince(startedAt) } }
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
