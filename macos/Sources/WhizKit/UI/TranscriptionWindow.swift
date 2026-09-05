import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Batch transcription flow: menu → setup window → progress → Finder
/// reveal, all inside ONE window whose content is a conditional render
/// (TranscriptionFlowView: setup phase until Run, progress phase after).
/// The window manager only creates the window, installs that view once, and
/// routes the Browse panel — phase changes are SwiftUI view updates, not
/// contentView swaps.
///
/// It mirrors `SettingsWindow`'s activation dance — whiz is an accessory
/// app, so a bare `makeKeyAndOrderFront` would leave this behind whatever
/// the user was doing.
@MainActor
final class TranscriptionWindow: NSObject, NSWindowDelegate {

    private var window: NSWindow?
    private let flow = TranscriptionFlowModel()
    private var hasShown = false

    /// Menu entry point. A run in flight means "show me my window", not
    /// "restart and kill it"; anything else gets a fresh setup snapshot,
    /// picking up config edits made since the last open.
    func start() {
        if flow.hasActiveRun {
            show()
            return
        }
        flow.restart()
        if window == nil { build() }
        show()
    }

    // MARK: - Content

    private func installContent() {
        window?.contentView = NSHostingView(rootView: TranscriptionFlowView(
            model: flow,
            browse: { [weak self] setup in
                self?.browse { url in
                    if let url { setup.pathText = url.path }
                }
            },
            onCancel: { [weak self] in
                self?.window?.performClose(nil)
            }))
    }

    // MARK: - File picking (the setup form's Browse button)

    private func browse(onPick: @escaping (URL?) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.movie, .video, .audio]
        panel.title = "Transcribe"
        panel.message = "Choose a video or audio file to transcribe."
        panel.prompt = "Open"

        // Same Space handling the Settings window needed (commit 2cad76b):
        // follow the user's current Space, including full-screen ones. Without
        // these behaviors the panel is anchored to a fixed Space and macOS
        // switches the user to wherever that is.
        panel.collectionBehavior = [.moveToActiveSpace, .fullScreenAuxiliary]

        // Presented as a sheet on the transcribe window, not as a free-floating
        // panel. `panel.begin` is modeless: from an accessory app it could end up
        // ordered behind the window that opened it, so clicking Browse appeared
        // to do nothing at all.
        //
        // A sheet also keeps the Space fix that `begin` was chosen for. It is
        // attached to a window that is already on the user's current Space, so
        // there is nothing for macOS to go hunting for and no jump to the
        // Desktop (the SettingsWindow bug, commit 2cad76b).
        if let window {
            NSApp.activate(ignoringOtherApps: true)
            panel.beginSheetModal(for: window) { response in
                onPick(response == .OK ? panel.url : nil)
            }
        } else {
            // No window to attach to — should not happen, since Browse is only
            // reachable from the form, but fall back rather than silently
            // dropping the click.
            panel.begin { response in
                onPick(response == .OK ? panel.url : nil)
            }
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    // MARK: - Window plumbing (mirrors SettingsWindow)

    private func build() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 460, height: 360),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.title = "whiz Transcribe"
        window.collectionBehavior = [.moveToActiveSpace, .fullScreenAuxiliary]
        window.isReleasedWhenClosed = false  // reused across runs
        window.delegate = self
        self.window = window

        // Install the content here, while building, exactly as SettingsWindow
        // does. It used to be guarded by `if window?.contentView == nil` from
        // start(), which never fired: NSWindow supplies a default empty NSView
        // on init, so contentView is never nil. The window opened showing that
        // blank default view instead of the setup form.
        //
        // Safe to do once: the view observes `flow`, so a fresh run re-renders
        // through the model rather than needing new content.
        installContent()
    }

    private func show() {
        guard let window else { return }
        // Centre on first open only, so a window the user has moved stays put —
        // the same convention SettingsWindow uses. Before ordering front, so
        // the window never appears in the default spot and visibly jumps.
        if !hasShown {
            window.center()
        }
        NSApp.setActivationPolicy(.regular)
        // Order front *before* activating — see SettingsWindow.show() for why.
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        hasShown = true
    }

    func windowWillClose(_ notification: Notification) {
        // Closing the window mid-run stops the transcription, not just the UI.
        flow.cancelActiveRun()
        NSApp.setActivationPolicy(.accessory)
    }
}

// MARK: - Output location

/// Where a transcription's outputs land: a sibling directory named
/// `<stem>.transcript`, created by the backend when it writes. The Python
/// pipeline writes plain `<stem>.srt`/`.txt` next to the input; a folder keeps
/// a run's outputs (and, later, frame manifests) groupable instead of
/// scattered, and gives the "open a folder with output" step one place.
enum TranscriptionOutputs {

    static func directory(for input: URL) -> URL {
        input.deletingLastPathComponent()
            .appendingPathComponent(input.deletingPathExtension().lastPathComponent + ".transcript")
    }
}

// MARK: - Backend seam

/// The pipeline contract the UI drives. `NativeTranscriptionBackend` is the
/// real implementation — `AudioFileDecoder` → `WhisperBatchTranscriber`
/// (beam-search profile, built-in VAD) → `TranscriptFormatter` outputs — and
/// `SimulatedTranscriptionBackend` (now a fixture in the test target) kept
/// the window honest before it landed.
///
/// `Sendable` because the backend runs off the MainActor by design — the
/// view model hands the whole run over and observes events back.
protocol TranscriptionBackend: Sendable {
    /// Transcribe `input`, writing outputs into `outputDirectory` (which the
    /// backend creates) and reporting progress/log lines as they happen.
    /// Returns the populated output directory. Cancellation is cooperative via
    /// the surrounding `Task` — throw `CancellationError` promptly.
    func transcribe(
        input: URL,
        outputDirectory: URL,
        onEvent: @escaping @Sendable (TranscriptionEvent) -> Void
    ) async throws -> URL
}

/// UI progress update. `Sendable` because it crosses from the backend's
/// context into the MainActor view model.
enum TranscriptionEvent: Sendable {
    case phase(String)
    case progress(Double)
    case log(String)
}

// MARK: - View model

@MainActor
final class TranscriptionViewModel: ObservableObject {

    enum Stage: Equatable {
        case running
        case finished
        case cancelled
        case failed(String)
    }

    @Published var phase = ""
    @Published var progress: Double = 0
    @Published var log: [String] = []
    @Published private(set) var stage: Stage = .running

    let inputURL: URL
    let outputDirectory: URL

    private let backend: any TranscriptionBackend
    private var run: Task<Void, Never>?

    init(input: URL, output: URL, backend: any TranscriptionBackend) {
        self.inputURL = input
        self.outputDirectory = output
        self.backend = backend
    }

    func start() {
        run = Task {
            do {
                _ = try await backend.transcribe(input: inputURL, outputDirectory: outputDirectory) {
                    [weak self] event in
                    Task { @MainActor in self?.handle(event) }
                }
                stage = .finished
            } catch is CancellationError {
                stage = .cancelled
            } catch {
                log.append("error: \(error.localizedDescription)")
                stage = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        run?.cancel()
    }

    /// The "on finish it will open a folder with output" step; also bound to a
    /// button in the finished state so it can be re-done.
    func revealOutput() {
        NSWorkspace.shared.open(outputDirectory)
    }

    private func handle(_ event: TranscriptionEvent) {
        switch event {
        case .phase(let text): phase = text
        case .progress(let fraction): progress = fraction
        case .log(let line): log.append(line)
        }
    }
}

// MARK: - View

/// The running-transcription window: status bar and phase on top, the log
/// window below, and outcome actions at the bottom.
struct TranscriptionView: View {
    @ObservedObject var viewModel: TranscriptionViewModel

    var body: some View {
        VStack(spacing: 12) {
            header
            statusBar
            logWindow
            footer
        }
        .padding(16)
        .frame(minWidth: 420, minHeight: 320)
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.inputURL.lastPathComponent)
                    .font(.headline)
                Text(viewModel.phase)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            switch viewModel.stage {
            case .running:
                Button("Cancel") { viewModel.cancel() }
            case .finished:
                Button("Open Output Folder") { viewModel.revealOutput() }
            case .cancelled:
                Text("Cancelled").foregroundStyle(.secondary)
            case .failed(let message):
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.yellow)
                    .help(message)
            }
        }
    }

    /// The status bar: determinate progress while the backend reports
    /// fractions, with the run's stage under it.
    private var statusBar: some View {
        VStack(alignment: .leading, spacing: 4) {
            ProgressView(value: viewModel.progress)
            HStack {
                Text(statusText)
                    .font(.caption)
                    .foregroundStyle(statusColor)
                Spacer()
                Text("\(Int((viewModel.progress * 100).rounded()))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var logWindow: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    // Append-only log, so the offset is a stable identity —
                    // scrolling by line *text* would jump to the first of two
                    // identical lines.
                    ForEach(Array(viewModel.log.enumerated()), id: \.offset) { index, line in
                        Text(line)
                            .font(.system(size: 11, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(index)
                    }
                }
                .padding(8)
            }
            .background(Color(nsColor: .controlBackgroundColor))
            .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color(nsColor: .separatorColor)))
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .onChange(of: viewModel.log.count) { _ in
                guard !viewModel.log.isEmpty else { return }
                withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo(viewModel.log.count - 1) }
            }
        }
        .frame(maxHeight: .infinity)
    }

    private var footer: some View {
        VStack(spacing: 6) {
            switch viewModel.stage {
            case .failed(let message):
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(2)
            case .cancelled:
                Text("Stopped — no output was written.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            default:
                EmptyView()
            }
        }
    }

    private var statusText: String {
        switch viewModel.stage {
        case .running: return viewModel.phase.isEmpty ? "Working…" : viewModel.phase
        case .finished: return "Finished — output folder opened"
        case .cancelled: return "Cancelled"
        case .failed: return "Failed"
        }
    }

    private var statusColor: Color {
        switch viewModel.stage {
        case .running: return .secondary
        case .finished: return .green
        case .cancelled: return .secondary
        case .failed: return .red
        }
    }
}