import SwiftUI
import Combine
import UniformTypeIdentifiers

/// The single window: one control bar up top, the run below it.
///
/// The control bar answers the three questions a user actually asks —
/// what file (the pill: click, ⌘O, or drop), what state (the status chip:
/// live phase + elapsed while running, the outcome after), what to do
/// (Transcribe / Stop) — and the options sit in one compact row under it.
///
/// The area below shows only what exists: log and artifacts during and after
/// a run, the last run's card when idle. Never an empty pane pretending to
/// be content. The log pane stays the main character, because a run is
/// minutes of waiting and the verbatim output is the most honest progress
/// indicator there is.
struct MainView: View {
    @StateObject private var model = TranscriptionViewModel()
    @State private var showingPicker = false
    /// Ticks once a second so the running chip's clock moves.
    @State private var now = Date()

    var body: some View {
        VStack(spacing: 0) {
            controlBar
            Divider()
            contentArea
        }
        .navigationTitle("whiz — Transcribe")
        .fileImporter(
            isPresented: $showingPicker,
            allowedContentTypes: [.movie, .audio],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                model.inputURL = url
            }
        }
        .alert(
            "Transcription failed",
            isPresented: Binding(
                get: { model.hasError },
                set: { if !$0 { model.errorMessage = nil } }
            )
        ) {
            Button("OK") {}
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    // MARK: - Control bar

    private var controlBar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 10) {
                inputPill
                if model.lastOutcome != nil {
                    statusChip
                }
                Spacer(minLength: 0)
                runControls
            }
            optionsRow
        }
        .padding(12)
        .onReceive(Timer.publish(every: 1, on: .main, in: .common).autoconnect()) { now = $0 }
        // A keyboard shortcut needs a button to live on; this one is invisible.
        .background(
            Button("Choose File…") { showingPicker = true }
                .keyboardShortcut("o", modifiers: .command)
                .frame(width: 0)
                .opacity(0)
                .disabled(model.isRunning)
        )
    }

    /// The input file, as a pill. Click (or ⌘O) to choose.
    private var inputPill: some View {
        HStack(spacing: 8) {
            Image(systemName: model.inputURL == nil ? "film" : "waveform")
                .foregroundStyle(.secondary)
            Text(model.inputURL?.lastPathComponent ?? "Choose a file…")
                .foregroundStyle(model.inputURL == nil ? .secondary : .primary)
                .lineLimit(1)
                .truncationMode(.middle)
                .help(model.inputURL?.path ?? "Click to choose — or drop a file when idle")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .frame(maxWidth: 280)
        .background(.quinary, in: RoundedRectangle(cornerRadius: 6))
        .contentShape(Rectangle())
        .onTapGesture { if !model.isRunning { showingPicker = true } }
    }

    /// What the last run in this window is doing / how it ended.
    private var statusChip: some View {
        Group {
            switch model.lastOutcome {
            case .running:
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text(model.phase ?? "Running…")
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(model.phase ?? "")
                    Text(TranscriptionViewModel.formatDuration(
                        now.timeIntervalSince(model.runStartedAt ?? now)))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            case .finished:
                Group {
                    if let duration = model.lastDuration {
                        Label("Finished · \(TranscriptionViewModel.formatDuration(duration))",
                              systemImage: "checkmark.circle.fill")
                    } else {
                        Label("Finished", systemImage: "checkmark.circle.fill")
                    }
                }
                .foregroundStyle(.green)
            case .failed:
                Label("Failed", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .help(model.errorMessage ?? "See the alert for details")
            case .stopped:
                Label("Stopped", systemImage: "stop.circle.fill")
                    .foregroundStyle(.secondary)
            case nil:
                EmptyView()
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.quinary, in: Capsule())
        .lineLimit(1)
    }

    @ViewBuilder
    private var runControls: some View {
        if model.isRunning {
            Button("Stop") { model.stop() }
                .keyboardShortcut(".", modifiers: .command)
                .tint(.red)
        } else {
            Button("Transcribe") { model.start() }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut("r", modifiers: .command)
                .disabled(model.inputURL == nil)
        }
    }

    /// Every option on one row: the old grouped form spent the top half of
    /// the window on controls that are set once and never touched again.
    private var optionsRow: some View {
        HStack(spacing: 14) {
            HStack(spacing: 4) {
                Text("Language").foregroundStyle(.secondary)
                TextField("auto", text: $model.language)
                    .frame(width: 50)
                    .help("BCP-47 tag, e.g. ru or en. Empty = auto-detect.")
            }
            HStack(spacing: 4) {
                Text("Speakers").foregroundStyle(.secondary)
                Stepper(model.speakers == 0 ? "auto" : String(model.speakers),
                        value: $model.speakers, in: 0...10)
                    .help("0 = the CLI decides (auto-detect for video).")
            }
            HStack(spacing: 4) {
                Text("Frames").foregroundStyle(.secondary)
                Picker("", selection: $model.screenshots) {
                    ForEach(ScreenshotsMode.allCases) { mode in
                        Text(mode.shortLabel).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 170)
                .help("Auto = frames only for video input. On/Off force it.")
            }
            HStack(spacing: 4) {
                Toggle("Analyze", isOn: $model.analyze)
                    .help("Runs an AI pass over the transcript (--analyze).")
                if model.analyze {
                    TextField("AI model (config default)", text: $model.aiModel)
                        .frame(width: 130)
                        .help("Sent as --ai-model when non-empty. Default: whiz config set ai_model=…")
                }
            }
            Spacer(minLength: 0)
        }
        .disabled(model.isRunning)
    }

    // MARK: - Content area

    /// Only what exists: install guidance, the run (live or finished), or the
    /// idle area. Never an empty pane.
    @ViewBuilder
    private var contentArea: some View {
        if model.whizMissing {
            missingWhizBanner
        } else if model.isRunning || !model.logLines.isEmpty || !model.artifacts.isEmpty {
            runArea
        } else {
            idleArea
        }
    }

    private var runArea: some View {
        HSplitView {
            logPane
                .frame(minWidth: 300)
            if !model.artifacts.isEmpty {
                Divider()
                artifactPane
                    .frame(minWidth: 210, idealWidth: 270)
            }
        }
    }

    private var logPane: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Log").font(.headline)
                Spacer()
                if model.logLines.count >= TranscriptionViewModel.logLineCap {
                    Text("showing the last \(TranscriptionViewModel.logLineCap) lines")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                openFolderButton
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(model.logLines.indices, id: \.self) { index in
                            Text(model.logLines[index])
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .id(index)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 12)
                }
                .onChange(of: model.logLines.count) { count in
                    guard count > 0 else { return }
                    proxy.scrollTo(count - 1, anchor: .bottom)
                }
            }
        }
    }

    private var artifactPane: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Artifacts").font(.headline)
                Spacer()
                openFolderButton
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            List(model.artifacts) { artifact in
                artifactRow(label: artifact.label, url: artifact.url)
            }
        }
    }

    /// The artifacts live next to the input file; this is the one click that
    /// gets the user to them in Finder.
    private var openFolderButton: some View {
        Button {
            let dir = model.artifacts.first?.url.deletingLastPathComponent()
                ?? model.inputURL?.deletingLastPathComponent()
            if let dir { NSWorkspace.shared.open(dir) }
        } label: {
            Image(systemName: "folder")
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
        .help("Open the folder holding this run's output")
        .disabled(model.artifacts.isEmpty && model.inputURL == nil)
    }

    private func artifactRow(label: String, url: URL) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                Text(url.lastPathComponent)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(url.path)
            }
            Spacer()
            Text(Artifact(label: label, url: url).kind.badge)
                .font(.caption2)
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(.quaternary, in: Capsule())
            Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            Button("Open") { NSWorkspace.shared.open(url) }
        }
        .padding(.vertical, 1)
    }

    // MARK: - Idle area

    private var idleArea: some View {
        ZStack {
            if let record = model.lastRun {
                lastRunCard(record)
            } else {
                firstRunHint
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .onTapGesture { if !model.isRunning { showingPicker = true } }
        .onDrop(of: [UTType.fileURL], isTargeted: nil) { providers in
            acceptDrop(providers)
        }
    }

    private var firstRunHint: some View {
        VStack(spacing: 12) {
            Image(systemName: "square.and.arrow.down")
                .font(.system(size: 34))
                .foregroundStyle(.secondary)
            Text("Drop a media file here, or click to choose one")
                .foregroundStyle(.secondary)
            Text("whiz transcribes, labels speakers, grabs frames and can analyze — the log shows the run verbatim. ⌘R starts, ⌘O browses, ⌘. stops.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 460)
        }
        .padding(24)
    }

    /// The previous run's results, offered in a fresh window. A window that
    /// never ran anything used to look exactly like "no run ever happened",
    /// including seconds after a finished run's window was closed. The
    /// artifacts are on disk; this card says so.
    private func lastRunCard(_ record: LastRunRecord) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Text("Last run").font(.headline)
                Text("·").foregroundStyle(.tertiary)
                Text(URL(fileURLWithPath: record.inputPath).lastPathComponent)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(record.inputPath)
                Spacer()
                Text(record.finishedAt.formatted(date: .abbreviated, time: .shortened))
                    .foregroundStyle(.secondary)
            }
            List(record.artifacts) { artifact in
                artifactRow(label: artifact.label, url: artifact.url)
            }
            .frame(height: min(120 + CGFloat(record.artifacts.count) * 32, 220))
            HStack {
                Spacer()
                Button("Open Folder") {
                    let dir = record.artifacts.first?.url.deletingLastPathComponent()
                        ?? URL(fileURLWithPath: record.inputPath).deletingLastPathComponent()
                    NSWorkspace.shared.open(dir)
                }
                .controlSize(.small)
            }
        }
        .padding(16)
        .frame(maxWidth: 520)
        .background(.quinary, in: RoundedRectangle(cornerRadius: 10))
    }

    /// A dropped file chooses the input and starts the run — the fastest path
    /// from "I have a recording" to "it's running". Folders are ignored; the
    /// CLI is the contract on which media it can read, not this view.
    private func acceptDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        _ = provider.loadObject(ofClass: URL.self) { url, _ in
            guard let url = url else { return }
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue else { return }
            Task { @MainActor in
                guard !model.isRunning else { return }
                model.inputURL = url
                model.start()
            }
        }
        return true
    }

    private var missingWhizBanner: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("The whiz command was not found", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text("The app looks in ~/.local/bin (pipx), Homebrew and ~/.cargo/bin, and in this app's PATH. Install the CLI, or point at a checkout with the WHIZ_EXECUTABLE environment variable.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("Install with:  pipx install whiz")
                .font(.system(size: 12, design: .monospaced))
            Button("Check Again") { model.dismissMissingWarning() }
                .padding(.top, 2)
        }
        .padding(16)
        .frame(maxWidth: 620)
        .frame(maxWidth: .infinity)
    }
}

extension Artifact.Kind {
    /// Short tag for the artifact list — an extension is a fact, prose is not.
    var badge: String {
        switch self {
        case .subtitles: return "SRT"
        case .transcript: return "TXT"
        case .webPage: return "HTML"
        case .analysis: return "MD"
        case .data: return "JSON"
        case .other: return "FILE"
        }
    }
}
