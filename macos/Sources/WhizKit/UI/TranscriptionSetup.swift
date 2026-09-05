import SwiftUI
import UniformTypeIdentifiers

/// Per-run settings shown before transcription starts — the Transcribe… menu
/// opens this instead of a file picker, so the knobs that make sense per
/// input (speaker count, OCR, AI analysis) live next to the file they apply
/// to. Global preferences stay in the shared config; this view only *reads*
/// them to seed defaults and produces a per-run `BatchSettings` overlay —
/// the on-disk config is never modified by a run.
///
/// The analyze toggle defaults to whatever `ai_model` is configured (the
/// pipeline's opt-in), the OCR toggle to the config's `ocr`, and speakers
/// to auto-detect unless a count is configured.
@MainActor
final class TranscriptionSetupModel: ObservableObject {

    /// The config snapshot this run's overrides apply to (injectable for tests).
    private let settings: BatchSettings

    @Published var pathText = ""
    @Published var speakersAuto: Bool
    @Published var speakerCount: Int
    @Published var ocrEnabled: Bool
    @Published var analyzeEnabled: Bool
    @Published var aiModel: String
    @Published var availableModels: [String] = []
    @Published var modelsLoading = true

    init(settings: BatchSettings = .load()) {
        self.settings = settings
        self.speakersAuto = settings.numSpeakers == 0
        self.speakerCount = max(2, settings.numSpeakers)
        self.ocrEnabled = settings.ocr
        self.analyzeEnabled = !settings.aiModel.isEmpty
        self.aiModel = settings.aiModel
    }

    /// Fetch the model list from the configured server. The configured
    /// model is kept in the list even when the server is down, so a saved
    /// choice stays selectable.
    func loadModels() async {
        var models = await AnalysisEngine.listModels(baseURL: settings.aiBaseURL)
        if !settings.aiModel.isEmpty, !models.contains(settings.aiModel) {
            models.insert(settings.aiModel, at: 0)
        }
        availableModels = models
        if aiModel.isEmpty || !models.contains(aiModel) {
            aiModel = models.first ?? ""
        }
        modelsLoading = false
    }

    /// The chosen file: what the Browse button set, or a pasted path that
    /// exists on disk.
    var selectedInput: URL? {
        let trimmed = pathText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              FileManager.default.fileExists(atPath: trimmed)
        else { return nil }
        return URL(fileURLWithPath: trimmed)
    }

    var canRun: Bool { selectedInput != nil }

    /// The per-run settings overlay: the config snapshot with this dialog's
    /// choices applied. Analyze off clears aiModel for THIS run only — the
    /// pipeline treats an empty model as "skip analysis" — and a garbled
    /// speaker count falls back to auto (0).
    func resolvedSettings() -> BatchSettings {
        var s = settings
        s.numSpeakers = speakersAuto ? 0 : max(0, speakerCount)
        s.ocr = ocrEnabled
        if analyzeEnabled, !aiModel.isEmpty {
            s.aiModel = aiModel
        } else {
            s.aiModel = ""
        }
        return s
    }
}

/// The whole transcription flow's state machine: setup phase (fresh config
/// snapshot) → running phase (one live run). The window renders this with
/// a conditional `if let` — a single SwiftUI hosting view for the window's
/// lifetime, so phase changes are ordinary view updates, not contentView
/// swaps, and the window keeps one fixed size.
@MainActor
final class TranscriptionFlowModel: ObservableObject {

    /// Non-nil = the progress phase is showing (running, finished or failed).
    @Published private(set) var run: TranscriptionViewModel?

    /// The setup form's state, re-seeded from the config on every restart.
    @Published private(set) var setup: TranscriptionSetupModel

    init() {
        self.setup = TranscriptionSetupModel()
    }

    /// A run is in flight — a mid-run menu click must NOT restart the flow
    /// and kill the user's transcription; it just re-shows the window. A
    /// finished/failed/cancelled run is not "active": starting over is
    /// exactly what the user wants then.
    var hasActiveRun: Bool {
        if let run, case .running = run.stage { return true }
        return false
    }

    /// Run button: apply the dialog's overlay and swap to the progress phase.
    /// The backend is injectable so tests can drive the phases without a
    /// whisper model; production always uses the real one.
    func startRun(backend: (any TranscriptionBackend)? = nil) {
        guard let input = setup.selectedInput else { return }
        let model = TranscriptionViewModel(
            input: input,
            output: TranscriptionOutputs.directory(for: input),
            backend: backend ?? NativeTranscriptionBackend(settings: setup.resolvedSettings()))
        run = model
        model.start()
    }

    /// Menu entry (when nothing is active): cancel anything lingering, fresh
    /// setup snapshot — picks up config edits made since the last open.
    func restart() {
        run?.cancel()
        run = nil
        setup = TranscriptionSetupModel()
    }

    /// Window close: stop a live run, not just the UI.
    func cancelActiveRun() {
        run?.cancel()
    }
}

/// One window, one hosting view, two phases: the conditional render the flow
/// is built around. The fixed 460×360 frame fits both layouts — no window
/// resizing between phases at all.
struct TranscriptionFlowView: View {
    @ObservedObject var model: TranscriptionFlowModel
    var browse: (TranscriptionSetupModel) -> Void
    var onCancel: () -> Void

    var body: some View {
        Group {
            if let run = model.run {
                TranscriptionView(viewModel: run)
            } else {
                TranscriptionSetupView(
                    model: model.setup,
                    onBrowse: { browse(model.setup) },
                    onCancel: onCancel,
                    onRun: { model.startRun() })
            }
        }
        .frame(width: 460, height: 360)
    }
}

/// The setup form: path + browse, speakers (auto toggle hides the count),
/// OCR, analyze + model picker, then Cancel / Run.
struct TranscriptionSetupView: View {
    @ObservedObject var model: TranscriptionSetupModel
    var onBrowse: () -> Void
    var onCancel: () -> Void
    var onRun: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            pathRow
            Divider()
            speakersRow
            ocrRow
            analyzeRow
            Spacer(minLength: 0)
            footer
        }
        .padding(16)
        .frame(width: 460, height: 360)
        .task { await model.loadModels() }
    }

    private var pathRow: some View {
        HStack(spacing: 8) {
            TextField("Path to video or audio", text: $model.pathText)
            Button("Browse…") { onBrowse() }
        }
    }

    /// Auto-detect hides the number input entirely, per the flow's design:
    /// the count only means anything when the user explicitly knows it.
    private var speakersRow: some View {
        HStack(spacing: 8) {
            Toggle("Auto-detect speakers", isOn: $model.speakersAuto)
            Spacer()
            if !model.speakersAuto {
                Stepper(
                    "Speakers: \(model.speakerCount)",
                    value: $model.speakerCount,
                    in: 2...32)
                    .fixedSize()
            }
        }
    }

    private var ocrRow: some View {
        Toggle("Read on-screen text (OCR)", isOn: $model.ocrEnabled)
    }

    private var analyzeRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle("Analyze with AI", isOn: $model.analyzeEnabled)
            if model.analyzeEnabled {
                HStack(spacing: 8) {
                    if model.modelsLoading {
                        ProgressView()
                            .controlSize(.small)
                        Text("Loading models…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else if model.availableModels.isEmpty {
                        Text("No models found — check ai_base_url in config")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        Picker("Model", selection: $model.aiModel) {
                            ForEach(model.availableModels, id: \.self) { model in
                                Text(model).tag(model)
                            }
                        }
                        .pickerStyle(.menu)
                        .frame(maxWidth: 260)
                    }
                }
                .padding(.leading, 18)
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("Cancel") { onCancel() }
            Spacer()
            Button("Run") { onRun() }
                .keyboardShortcut(.defaultAction)
                .disabled(!model.canRun)
        }
    }
}