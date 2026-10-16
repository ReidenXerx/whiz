import SwiftUI
import UniformTypeIdentifiers

/// The new-transcription form: a file, the few choices that differ per
/// recording, and one button.
///
/// Everything that is the same for every run (model, output formats, AI
/// server) lives in Settings; the persisted values here are the per-run
/// ones the user tends to repeat (`OptionsStore`).
struct ComposeView: View {
    @EnvironmentObject private var model: TranscriptionViewModel
    @EnvironmentObject private var settings: SettingsModel
    @EnvironmentObject private var setup: SetupModel
    @State private var dropTargeted = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("New Transcription")
                        .font(.largeTitle.weight(.bold))
                    Text("Transcribe a recording, tell speakers apart, capture video frames and summarize it.")
                        .foregroundStyle(.secondary)
                }

                if model.whizMissing || (!setup.isChecking && setup.firstMissing != nil) {
                    SetupBanner { setup.isPresented = true }
                }

                fileSection
                optionsSection
                startRow
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 28)
            .frame(maxWidth: 680)
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: - File

    @ViewBuilder
    private var fileSection: some View {
        if let url = model.inputURL {
            SelectedFileCard(url: url,
                             onChange: { model.isPickingFile = true },
                             onClear: { model.inputURL = nil })
                .onDrop(of: [UTType.fileURL], isTargeted: $dropTargeted, perform: acceptDrop)
        } else {
            dropZone
        }
    }

    private var dropZone: some View {
        Button {
            model.isPickingFile = true
        } label: {
            VStack(spacing: 12) {
                Image(systemName: "arrow.down.doc")
                    .font(.system(size: 34, weight: .light))
                    .foregroundStyle(dropTargeted ? Color.accentColor : Color.secondary)
                VStack(spacing: 4) {
                    Text("Drop a recording or video here")
                        .font(.title3.weight(.medium))
                    Text("or click to choose a file  ·  ⌘O")
                        .foregroundStyle(.secondary)
                }
                Text("MOV, MP4, M4A, MP3, WAV and anything else ffmpeg reads")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 190)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(dropTargeted ? Color.accentColor.opacity(0.08) : Color.primary.opacity(0.025))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(dropTargeted ? Color.accentColor : Color.secondary.opacity(0.35),
                                  style: StrokeStyle(lineWidth: 1.5, dash: [6, 5]))
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onDrop(of: [UTType.fileURL], isTargeted: $dropTargeted, perform: acceptDrop)
        .animation(.easeOut(duration: 0.12), value: dropTargeted)
    }

    private func acceptDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        _ = provider.loadObject(ofClass: URL.self) { url, _ in
            guard let url else { return }
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue else { return }
            Task { @MainActor in model.choose(url) }
        }
        return true
    }

    // MARK: - Options

    private var optionsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("Options")
            VStack(spacing: 0) {
                OptionRow(title: "Language",
                          detail: "Leave on automatic unless detection picks the wrong one.") {
                    Picker("", selection: $model.language) {
                        Text(defaultLanguageTitle).tag("")
                        Divider()
                        ForEach(Languages.common, id: \.code) { language in
                            Text(language.name).tag(language.code)
                        }
                        if !model.language.isEmpty,
                           !Languages.common.contains(where: { $0.code == model.language }) {
                            Text(model.language).tag(model.language)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 200)
                }
                Divider().padding(.leading, 16)
                OptionRow(title: "Speakers",
                          detail: "How many people talk. Automatic works for most recordings.") {
                    Picker("", selection: $model.speakers) {
                        Text("Automatic").tag(0)
                        Divider()
                        ForEach(1...10, id: \.self) { count in
                            Text(count == 1 ? "1 person" : "\(count) people").tag(count)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 200)
                }
                Divider().padding(.leading, 16)
                OptionRow(title: "Names",
                          detail: "Optional. Given by talk time, most talkative first. Known voices are recognized on their own, and you can name the rest at the end.") {
                    TextField("", text: $model.speakerNames, prompt: Text("Alice, Bob"))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 200)
                }
                Divider().padding(.leading, 16)
                OptionRow(title: "Video frames",
                          detail: "A still for each line of the transcript. Automatic captures them for video only.") {
                    Picker("", selection: $model.screenshots) {
                        ForEach(ScreenshotsMode.allCases) { mode in
                            Text(mode.shortLabel).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(width: 200)
                }
                Divider().padding(.leading, 16)
                OptionRow(title: "AI summary",
                          detail: analysisDetail) {
                    Toggle("", isOn: $model.analyze)
                        .toggleStyle(.switch)
                        .labelsHidden()
                        .frame(width: 200, alignment: .trailing)
                }
                if model.analyze {
                    Divider().padding(.leading, 16)
                    OptionRow(title: "AI model", detail: "Only for this run. The default is set in Settings.") {
                        TextField("", text: $model.aiModel,
                                  prompt: Text(settings.string("ai_model").isEmpty ? "Default" : settings.string("ai_model")))
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 200)
                    }
                }
            }
            .card(padding: 0)
        }
        .disabled(model.isRunning)
    }

    private var defaultLanguageTitle: String {
        let configured = settings.string("language")
        if configured.isEmpty || configured == "auto" { return "Detect automatically" }
        return "Default (\(Languages.name(for: configured)))"
    }

    private var analysisDetail: String {
        let server = settings.string("ai_base_url")
        if server.contains("localhost") || server.contains("127.0.0.1") {
            return "Summary and action items from your local model, after the transcript is written."
        }
        return "Summary and action items from the AI server in Settings, after the transcript is written."
    }

    // MARK: - Start

    private var startRow: some View {
        HStack(spacing: 12) {
            if model.isRunning {
                Label("A transcription is already running.", systemImage: "hourglass")
                    .foregroundStyle(.secondary)
                Button("Show Progress") { model.selection = .live }
                    .buttonStyle(.link)
            }
            Spacer()
            Button {
                model.start()
            } label: {
                Text("Transcribe")
                    .frame(minWidth: 110)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .keyboardShortcut(.defaultAction)
            .disabled(model.inputURL == nil || model.isRunning || setup.firstMissing != nil)
        }
    }
}

/// One labelled row of the options card: title + explanation on the left,
/// the control on the right — the System Settings shape.
private struct OptionRow<Control: View>: View {
    let title: String
    let detail: String
    @ViewBuilder var control: () -> Control

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 12)
            control()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 11)
    }
}

/// The chosen input: what it is, how long, how big.
private struct SelectedFileCard: View {
    let url: URL
    let onChange: () -> Void
    let onClear: () -> Void

    @State private var duration: Double?

    var body: some View {
        HStack(spacing: 14) {
            ThumbnailView(source: .file(url), size: CGSize(width: 96, height: 60))
            VStack(alignment: .leading, spacing: 4) {
                Text(url.lastPathComponent)
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(url.path)
                Text(summary)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Text(url.deletingLastPathComponent().path.replacingOccurrences(
                    of: NSHomeDirectory(), with: "~"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            Button("Change…", action: onChange)
            Button(action: onClear) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)
            .help("Clear")
        }
        .card(padding: 14)
        .task(id: url) { duration = await MediaInfo.duration(of: url) }
    }

    private var summary: String {
        var parts = [MediaKind(url: url).label]
        if let duration { parts.append(Format.clock(duration)) }
        if let size = MediaInfo.fileSize(of: url) { parts.append(Format.fileSize(size)) }
        return parts.joined(separator: "  ·  ")
    }
}
