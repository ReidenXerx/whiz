import SwiftUI
import UniformTypeIdentifiers

/// The single window: request form, run controls, log pane, artifacts.
///
/// Layout stays plain on purpose — the log pane is the main character, because
/// a transcription run is minutes of waiting and the verbatim output is the
/// most honest progress indicator there is.
struct MainView: View {
    @StateObject private var model = TranscriptionViewModel()
    @State private var showingPicker = false

    var body: some View {
        VStack(spacing: 0) {
            Form {
                inputSection
                optionsSection
                runSection
            }
            .formStyle(.grouped)

            Divider()

            if model.whizMissing {
                missingWhizBanner
            } else if model.artifacts.isEmpty && model.logLines.isEmpty {
                emptyPane
            } else {
                outputSection
            }
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

    // MARK: - Form sections

    private var inputSection: some View {
        Section("Input") {
            HStack {
                Button("Choose File…") { showingPicker = true }
                    .disabled(model.isRunning)
                Text(model.inputURL?.lastPathComponent ?? "No file selected")
                    .foregroundStyle(model.inputURL == nil ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(model.inputURL?.path ?? "")
                Spacer()
            }
        }
    }

    private var optionsSection: some View {
        Section("Options") {
            TextField("Language (empty = auto)", text: $model.language)
                .disabled(model.isRunning)
            Stepper(
                "Speakers \(model.speakers == 0 ? "(auto-detect)" : String(model.speakers))",
                value: $model.speakers, in: 0...10
            )
            .disabled(model.isRunning)
            Picker("Screenshots", selection: $model.screenshots) {
                ForEach(ScreenshotsMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .disabled(model.isRunning)
            Toggle("Analyze after transcription (--analyze)", isOn: $model.analyze)
                .disabled(model.isRunning)
            if model.analyze {
                TextField("AI model (empty = config)", text: $model.aiModel)
                    .disabled(model.isRunning)
                    .help("Sent as --ai-model. Configure the default with: whiz config set ai_model=…")
            }
        }
    }

    private var runSection: some View {
        Section {
            HStack {
                if model.isRunning {
                    ProgressView()
                        .controlSize(.small)
                    Text(model.phase.map { "Phase: \($0)" } ?? "Running…")
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Spacer()
                    Button("Stop") { model.stop() }
                        .tint(.red)
                } else {
                    Button("Transcribe") { model.start() }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.inputURL == nil)
                    Spacer()
                }
            }
        } footer: {
            Text("The CLI is the contract: this app runs `whiz transcribe` and reads its output. Everything it prints appears below.")
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Output

    private var outputSection: some View {
        HSplitView {
            logPane
                .frame(minWidth: 320)
            if !model.artifacts.isEmpty {
                Divider()
                artifactPane
                    .frame(minWidth: 220, idealWidth: 280)
            }
        }
    }

    private var logPane: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Log")
                .font(.headline)
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
            Text("Artifacts")
                .font(.headline)
                .padding(.horizontal, 12)
                .padding(.top, 8)
            List(model.artifacts) { artifact in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(artifact.label)
                        Text(artifact.url.lastPathComponent)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .help(artifact.url.path)
                    }
                    Spacer()
                    Text(artifact.kind.badge)
                        .font(.caption2)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(.quaternary, in: Capsule())
                    Button("Reveal") {
                        NSWorkspace.shared.activateFileViewerSelecting([artifact.url])
                    }
                    Button("Open") {
                        NSWorkspace.shared.open(artifact.url)
                    }
                }
                .padding(.vertical, 1)
            }
        }
    }

    private var emptyPane: some View {
        VStack {
            Spacer()
            Text("Choose a file and press Transcribe.\nOutput appears here while the run is live.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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