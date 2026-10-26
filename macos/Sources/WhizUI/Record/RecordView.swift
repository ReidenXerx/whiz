import AppKit
import SwiftUI

/// Record a call: the microphone, the call's audio, the screen if wanted,
/// then (by default) straight into a transcription.
struct RecordView: View {
    @EnvironmentObject private var recorder: RecorderModel
    @EnvironmentObject private var model: TranscriptionViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Record a Call")
                        .font(.largeTitle.weight(.bold))
                    Text("Your microphone and the call’s sound — Zoom, Meet, Teams or anything else your Mac plays — and the screen if you want it. The recording stays on this Mac. Let the others on the call know you are recording.")
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if let error = recorder.errorMessage {
                    Notice(symbol: "exclamationmark.triangle.fill", tint: .orange, text: error) {
                        if error.contains("Screen & System Audio Recording") {
                            Button("Open System Settings") { NSWorkspace.shared.open(RecorderModel.screenSettingsURL) }
                        } else if error.contains("Privacy & Security → Microphone") {
                            Button("Open System Settings") { NSWorkspace.shared.open(RecorderModel.microphoneSettingsURL) }
                        }
                    }
                }

                if recorder.isBusy {
                    RecordingPanel()
                } else {
                    if let url = recorder.lastRecording, FileManager.default.fileExists(atPath: url.path) {
                        SavedRecordingCard(url: url)
                    }
                    sourcesSection
                    afterSection
                }
                controls
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 28)
            .frame(maxWidth: 680)
            .frame(maxWidth: .infinity)
        }
        .onAppear {
            recorder.refreshDevices()
            recorder.refreshPermissions()
        }
        // Back from System Settings with the switch turned on.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            recorder.refreshPermissions()
        }
    }

    // MARK: - Sources

    private var sourcesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("Record")
            VStack(spacing: 0) {
                Row(title: "Microphone", detail: "Your side of the call.") {
                    Picker("", selection: $recorder.preferences.microphone) {
                        Text(recorder.defaultMicrophoneName.map { "Default (\($0))" } ?? "Default").tag("default")
                        if !recorder.microphones.isEmpty { Divider() }
                        ForEach(recorder.microphones) { device in
                            Text(device.name).tag(device.id)
                        }
                        if !["default", "off"].contains(recorder.preferences.microphone),
                           !recorder.microphones.contains(where: { $0.id == recorder.preferences.microphone }) {
                            Text("Not connected — uses the default").tag(recorder.preferences.microphone)
                        }
                        Divider()
                        Text("Off").tag("off")
                    }
                    .labelsHidden()
                    .frame(width: 230)
                }
                Divider().padding(.leading, 16)
                Row(title: "Call audio", detail: "Everything your Mac plays, except Whiz: the other side of the call.") {
                    Toggle("", isOn: $recorder.preferences.systemAudio)
                        .toggleStyle(.switch)
                        .labelsHidden()
                        .frame(width: 230, alignment: .trailing)
                }
                Divider().padding(.leading, 16)
                Row(title: "Screen", detail: "Video of the screen: the transcript gets a picture for each line, slides and shared screens included.") {
                    Picker("", selection: screenChoice) {
                        Text("Off").tag(UInt32?.none)
                        Divider()
                        ForEach(recorder.displays) { display in
                            Text(display.name).tag(UInt32?.some(display.id))
                        }
                    }
                    .labelsHidden()
                    .frame(width: 230)
                }
                if needsScreenPermission {
                    Divider().padding(.leading, 16)
                    HStack(spacing: 10) {
                        Image(systemName: "lock.fill").foregroundStyle(.orange)
                        Text("Recording the call’s audio\(recorder.preferences.screen ? " or the screen" : "") needs the Screen & System Audio Recording permission.")
                            .font(.callout)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer()
                        Button("Allow…") { recorder.requestScreenAccess() }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 11)
                }
            }
            .card(padding: 0)
        }
    }

    /// One picker for "screen off" and "which screen".
    private var screenChoice: Binding<UInt32?> {
        Binding(
            get: {
                guard recorder.preferences.screen else { return nil }
                let id = recorder.preferences.display
                return recorder.displays.contains { $0.id == id } ? id : recorder.displays.first?.id
            },
            set: { choice in
                recorder.preferences.screen = choice != nil
                if let choice { recorder.preferences.display = choice }
            })
    }

    private var needsScreenPermission: Bool {
        !recorder.screenAllowed && (recorder.preferences.systemAudio || recorder.preferences.screen)
    }

    private var afterSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("Afterwards")
            VStack(spacing: 0) {
                Row(title: "Transcribe when I stop",
                    detail: "With the options on New Transcription; speakers are told apart.") {
                    Toggle("", isOn: $recorder.preferences.transcribeWhenDone)
                        .toggleStyle(.switch)
                        .labelsHidden()
                        .frame(width: 230, alignment: .trailing)
                }
                Divider().padding(.leading, 16)
                Row(title: "Save to", detail: recorder.preferences.folder.path
                        .replacingOccurrences(of: NSHomeDirectory(), with: "~")) {
                    HStack {
                        Button("Show") { showFolder() }
                        Button("Change…") { chooseFolder() }
                    }
                    .frame(width: 230, alignment: .trailing)
                }
            }
            .card(padding: 0)
        }
    }

    // MARK: - Controls

    private var controls: some View {
        HStack(spacing: 12) {
            if !recorder.isBusy, recorder.preferences.systemAudio, recorder.preferences.microphone != "off" {
                Label("Wear headphones: through speakers, the microphone hears the call a second time.",
                      systemImage: "headphones")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                recorder.toggle()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: recorder.isRecording ? "stop.fill" : "record.circle")
                    Text(buttonTitle)
                }
                .frame(minWidth: 110)
            }
            .buttonStyle(.borderedProminent)
            .tint(recorder.isRecording ? .red : .accentColor)
            .controlSize(.large)
            .keyboardShortcut(.defaultAction)
            .disabled(recorder.state == .starting || recorder.state == .stopping)
        }
    }

    private var buttonTitle: String {
        switch recorder.state {
        case .idle: return "Record"
        case .starting: return "Starting…"
        case .recording: return "Stop"
        case .stopping: return "Saving…"
        }
    }

    private func showFolder() {
        let folder = recorder.preferences.folder
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        NSWorkspace.shared.open(folder)
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Choose"
        panel.directoryURL = recorder.preferences.folder
        if panel.runModal() == .OK, let url = panel.url {
            recorder.preferences.folderPath = url.path
        }
    }
}

/// While recording: the clock, what is being heard, and any warning.
private struct RecordingPanel: View {
    @EnvironmentObject private var recorder: RecorderModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Circle()
                    .fill(Color.red)
                    .frame(width: 12, height: 12)
                    .opacity(recorder.isRecording ? 1 : 0.4)
                Text(recorder.isRecording ? "Recording" : (recorder.state == .stopping ? "Saving" : "Starting"))
                    .font(.title2.weight(.semibold))
                Spacer()
                Text(RecorderModel.clock(recorder.elapsed))
                    .font(.system(size: 34, weight: .medium, design: .rounded).monospacedDigit())
            }
            let sources = recorder.sources
            if sources.microphone != nil {
                MeterRow(title: "Microphone", level: \.microphone)
            }
            if sources.systemAudio {
                MeterRow(title: "Call audio", level: \.system)
            }
            if sources.display != nil {
                Label(recorder.displays.first { $0.id == sources.display }?.name ?? "Screen",
                      systemImage: "display")
                    .foregroundStyle(.secondary)
            }
            if let warning = recorder.warning {
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            }
            Text("You can close this window: recording goes on, and the menu bar shows it with a Stop button.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .card(padding: 20)
    }
}

/// A level meter, on a decibel scale: speech sits in the middle, not at
/// the bottom edge.
private struct MeterRow: View {
    @EnvironmentObject private var recorder: RecorderModel
    let title: String
    let level: KeyPath<RecordingMeter, Float>

    var body: some View {
        HStack(spacing: 12) {
            Text(title)
                .frame(width: 90, alignment: .leading)
            MeterBar(meter: recorder.meter, level: level)
        }
    }
}

private struct MeterBar: View {
    @ObservedObject var meter: RecordingMeter
    let level: KeyPath<RecordingMeter, Float>

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.08))
                Capsule()
                    .fill(fraction > 0.92 ? Color.orange : Color.green)
                    .frame(width: geometry.size.width * fraction)
                    .animation(.linear(duration: 0.1), value: fraction)
            }
        }
        .frame(height: 8)
    }

    /// -60 dBFS (silence) to 0 dBFS (full scale).
    private var fraction: CGFloat {
        let peak = meter[keyPath: level]
        guard peak > 0 else { return 0 }
        return CGFloat(min(max((20 * log10(peak) + 60) / 60, 0), 1))
    }
}

/// The recording just made.
private struct SavedRecordingCard: View {
    @EnvironmentObject private var recorder: RecorderModel
    @EnvironmentObject private var model: TranscriptionViewModel
    let url: URL
    @State private var duration: Double?

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: MediaKind(url: url).symbol)
                .font(.title2)
                .foregroundStyle(.secondary)
                .frame(width: 32)
            VStack(alignment: .leading, spacing: 3) {
                Text("Saved \(url.lastPathComponent)")
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text([duration.map(Format.clock), MediaInfo.fileSize(of: url).map(Format.fileSize)]
                        .compactMap { $0 }.joined(separator: "  ·  "))
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            Button("Transcribe") {
                model.choose(url)
                model.start()
            }
            .disabled(model.isRunning)
            .help(model.isRunning ? "A transcription is already running." : "")
        }
        .card(padding: 14)
        .task(id: url) { duration = await MediaInfo.duration(of: url) }
    }
}

private struct Row<Control: View>: View {
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

private struct Notice<Actions: View>: View {
    let symbol: String
    let tint: Color
    let text: String
    @ViewBuilder var actions: () -> Actions

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: symbol)
                .foregroundStyle(tint)
                .font(.title3)
            Text(text)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 8)
            actions()
        }
        .card(padding: 14)
    }
}
