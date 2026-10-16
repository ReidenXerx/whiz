import AppKit
import SwiftUI

/// The languages offered by name. whisper accepts many more; a code that is
/// not in this list (set from the terminal) is still shown and kept.
enum Languages {
    static let common: [(code: String, name: String)] = [
        ("en", "English"), ("uk", "Ukrainian"), ("ru", "Russian"), ("de", "German"),
        ("fr", "French"), ("es", "Spanish"), ("it", "Italian"), ("pt", "Portuguese"),
        ("pl", "Polish"), ("nl", "Dutch"), ("tr", "Turkish"), ("cs", "Czech"),
        ("sv", "Swedish"), ("zh", "Chinese"), ("ja", "Japanese"), ("ko", "Korean"),
        ("ar", "Arabic"), ("hi", "Hindi"),
    ]

    static func name(for code: String) -> String {
        common.first { $0.code == code }?.name ?? code
    }
}

/// ⌘, — whiz's own config, through the CLI. Per-run choices (language for
/// this file, speaker names) stay on the new-transcription form; this is
/// what is true for every run, in the terminal too.
struct SettingsView: View {
    @EnvironmentObject private var settings: SettingsModel

    var body: some View {
        Group {
            switch settings.state {
            case .idle, .loading:
                ProgressView("Reading whiz settings…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .unavailable(let reason):
                EmptyState(symbol: "exclamationmark.triangle", title: "Settings unavailable",
                           message: reason) {
                    Button("Try Again") { Task { await settings.reload() } }
                }
            case .loaded:
                TabView {
                    TranscriptionSettings()
                        .tabItem { Label("Transcription", systemImage: "waveform") }
                    SpeakerSettings()
                        .tabItem { Label("Speakers", systemImage: "person.2") }
                    AISettings()
                        .tabItem { Label("AI", systemImage: "sparkles") }
                    AboutSettings()
                        .tabItem { Label("whiz", systemImage: "terminal") }
                }
            }
        }
        .frame(width: 580, height: 600)
        .task { await settings.reload() }
    }
}

/// The CLI's verdict on the last write, under every tab.
private struct SaveErrorFooter: View {
    @EnvironmentObject private var settings: SettingsModel

    var body: some View {
        if let error = settings.lastError {
            Label(error, systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.red)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 20)
                .padding(.bottom, 12)
        }
    }
}

private struct TranscriptionSettings: View {
    @EnvironmentObject private var settings: SettingsModel

    private static let formats: [(key: String, title: String, note: String)] = [
        ("srt", "SRT subtitles", "Timed captions"),
        ("txt", "Plain text", "whisper's raw text"),
        ("vtt", "WebVTT", "Captions for the web"),
        ("html", "HTML transcript", "Speaker-colored page with frames"),
        ("json", "JSON", "Added automatically when speakers or HTML need it"),
    ]

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    Picker("Whisper model", selection: modelBinding) {
                        Text("Automatic (best installed)").tag("")
                        ForEach(settings.models) { model in
                            Text("\(model.alias)  ·  \(Format.fileSize(Int64(model.sizeMb * 1_048_576)))")
                                .tag(model.alias)
                        }
                        let current = settings.string("model")
                        if !current.isEmpty, !settings.models.contains(where: { $0.alias == current }) {
                            Text(current).tag(current)
                        }
                    }
                    if settings.models.isEmpty {
                        Text("No models installed yet. Download one in Terminal: whiz models download turbo")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Picker("Default language", selection: languageBinding) {
                        Text("Detect automatically").tag("auto")
                        Divider()
                        ForEach(Languages.common, id: \.code) { language in
                            Text(language.name).tag(language.code)
                        }
                        let current = settings.string("language")
                        if !current.isEmpty, current != "auto",
                           !Languages.common.contains(where: { $0.code == current }) {
                            Text(current).tag(current)
                        }
                    }
                }
                Section {
                    Toggle(isOn: boolBinding("vad")) {
                        Text("Skip silence")
                        Text("Voice activity detection: faster runs, fewer invented words in quiet parts.")
                    }
                    Toggle(isOn: boolBinding("verbose")) {
                        Text("Detailed whisper log")
                        Text("Include whisper.cpp's full output in a run's Details.")
                    }
                }
                Section("Files to write") {
                    ForEach(Self.formats, id: \.key) { format in
                        Toggle(isOn: formatBinding(format.key)) {
                            Text(format.title)
                            Text(format.note)
                        }
                    }
                }
            }
            .formStyle(.grouped)
            SaveErrorFooter()
        }
    }

    private var modelBinding: Binding<String> {
        Binding(get: { settings.string("model") },
                set: { settings.set("model", .string($0)) })
    }

    private var languageBinding: Binding<String> {
        Binding(get: { settings.string("language").isEmpty ? "auto" : settings.string("language") },
                set: { settings.set("language", .string($0)) })
    }

    private func boolBinding(_ key: String) -> Binding<Bool> {
        Binding(get: { settings.bool(key) }, set: { settings.set(key, .bool($0)) })
    }

    private func formatBinding(_ format: String) -> Binding<Bool> {
        Binding(
            get: { settings.list("outputs").contains(format) },
            set: { on in
                var outputs = settings.list("outputs").filter { $0 != format }
                if on { outputs.append(format) }
                settings.set("outputs", .list(outputs))
            })
    }
}

private struct SpeakerSettings: View {
    @EnvironmentObject private var settings: SettingsModel
    @State private var matchThreshold: Double = 0.8

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    Toggle(isOn: boolBinding("diarize")) {
                        Text("Detect speakers in audio files")
                        Text("Video always detects speakers; this turns it on for audio files too.")
                    }
                    Toggle(isOn: boolBinding("save_voice_profiles")) {
                        Text("Remember voices")
                        Text("Names you give are kept as voice profiles, so the same people are recognized in later recordings.")
                    }
                }
                Section {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Recognition strictness")
                            Spacer()
                            Text(matchThreshold, format: .number.precision(.fractionLength(2)))
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $matchThreshold, in: 0.5...0.95, step: 0.05) {
                            EmptyView()
                        } minimumValueLabel: {
                            Text("Loose").font(.caption)
                        } maximumValueLabel: {
                            Text("Strict").font(.caption)
                        } onEditingChanged: { editing in
                            if !editing { settings.set("speaker_match_threshold", .double(matchThreshold)) }
                        }
                        Text("How closely a voice must match a profile before it is named automatically. You confirm every automatic name at the end of a run.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .formStyle(.grouped)
            SaveErrorFooter()
        }
        .onAppear { matchThreshold = settings.double("speaker_match_threshold") }
    }

    private func boolBinding(_ key: String) -> Binding<Bool> {
        Binding(get: { settings.bool(key) }, set: { settings.set(key, .bool($0)) })
    }
}

private struct AISettings: View {
    @EnvironmentObject private var settings: SettingsModel
    @State private var baseURL = ""
    @State private var model = ""
    @State private var apiKey = ""
    @State private var maxFrames = 50

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    TextField("Server", text: $baseURL, prompt: Text("http://localhost:11434/v1"))
                        .onSubmit { settings.set("ai_base_url", .string(baseURL)) }
                    TextField("Model", text: $model, prompt: Text("e.g. qwen3:8b"))
                        .onSubmit { settings.set("ai_model", .string(model)) }
                } footer: {
                    Text("Ollama or any OpenAI-compatible server. Press Return to save a field.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section {
                    HStack {
                        SecureField("API key", text: $apiKey,
                                    prompt: Text(settings.secretIsSet("ai_api_key") ? "Saved — type to replace" : "Not needed for local Ollama"))
                            .onSubmit(saveKey)
                        if !apiKey.isEmpty {
                            Button("Save", action: saveKey)
                        } else if settings.secretIsSet("ai_api_key") {
                            Button("Remove") { settings.setSecret("ai_api_key", "") }
                        }
                    }
                } footer: {
                    Text("Stored in whiz's config file (readable only by you). Never shown again after saving.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section {
                    Stepper(value: $maxFrames, in: 0...200, step: 5) {
                        HStack {
                            Text("Frames sent to vision models")
                            Spacer()
                            Text("\(maxFrames)").monospacedDigit().foregroundStyle(.secondary)
                        }
                    }
                    .onChange(of: maxFrames) { value in
                        settings.set("ai_max_frames", .int(value))
                    }
                }
            }
            .formStyle(.grouped)
            SaveErrorFooter()
        }
        .onAppear {
            baseURL = settings.string("ai_base_url")
            model = settings.string("ai_model")
            maxFrames = settings.int("ai_max_frames")
        }
        .onDisappear {
            // A typed-but-not-submitted field still counts when leaving the tab.
            settings.set("ai_base_url", .string(baseURL))
            settings.set("ai_model", .string(model))
        }
    }

    private func saveKey() {
        guard !apiKey.isEmpty else { return }
        settings.setSecret("ai_api_key", apiKey)
        apiKey = ""
    }
}

private struct AboutSettings: View {
    @EnvironmentObject private var settings: SettingsModel

    var body: some View {
        Form {
            Section("Command-line tool") {
                LabeledContent("Version", value: settings.version ?? "unknown")
                if let executable = settings.executable {
                    LabeledContent("Location") {
                        Text(executable.path)
                            .textSelection(.enabled)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                Text("The app runs this whiz for every transcription. Update it with `whiz upgrade` in Terminal.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let path = settings.snapshot?.path {
                Section("Config file") {
                    LabeledContent("Path") {
                        Text(path)
                            .textSelection(.enabled)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    HStack {
                        Button("Open in Text Editor") {
                            let url = URL(fileURLWithPath: path)
                            let editor = URL(fileURLWithPath: "/System/Applications/TextEdit.app")
                            NSWorkspace.shared.open([url], withApplicationAt: editor,
                                                    configuration: NSWorkspace.OpenConfiguration())
                        }
                        Button("Show in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                        }
                        Spacer()
                        Button("Reload") { Task { await settings.reload() } }
                    }
                }
            }
        }
        .formStyle(.grouped)
    }
}
