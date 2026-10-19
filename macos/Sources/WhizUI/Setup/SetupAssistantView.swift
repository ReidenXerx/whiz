import AppKit
import SwiftUI

/// First launch: what Whiz is, then everything it needs — each part checked,
/// and fixable with one button.
struct SetupAssistantView: View {
    @EnvironmentObject private var setup: SetupModel
    @EnvironmentObject private var settings: SettingsModel
    @Environment(\.dismiss) private var dismiss
    /// Called when the user finishes or skips, so the window can remember.
    var onFinish: () -> Void

    @State private var page: Page = .welcome

    enum Page { case welcome, checklist }

    var body: some View {
        Group {
            switch page {
            case .welcome:
                WelcomePage { withAnimation(.easeInOut(duration: 0.25)) { page = .checklist } }
                    .transition(.move(edge: .leading).combined(with: .opacity))
            case .checklist:
                ChecklistPage(onDone: finish)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .frame(width: 720, height: 640)
        .background(Color(nsColor: .windowBackgroundColor))
        .task { await setup.refresh() }
        // Coming back from Terminal (the Homebrew installer, or a command
        // typed by hand) checks again, so the list catches up on its own.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            guard !setup.isChecking, setup.task == nil else { return }
            Task { await setup.refresh() }
        }
    }

    private func finish() {
        onFinish()
        dismiss()
    }
}

// MARK: - Welcome

private struct WelcomePage: View {
    var onContinue: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 24)
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .frame(width: 128, height: 128)
                .shadow(color: .black.opacity(0.18), radius: 12, y: 6)
            Text("Welcome to Whiz")
                .font(.system(size: 34, weight: .bold))
                .padding(.top, 18)
            Text("Turn recordings into transcripts you can read, search and act on.")
                .font(.title3)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.top, 6)

            VStack(alignment: .leading, spacing: 18) {
                Feature(symbol: "lock.shield", tint: .blue,
                        title: "Runs on this Mac",
                        detail: "Speech recognition happens on your computer. Your recordings never leave it.")
                Feature(symbol: "person.2.wave.2", tint: .orange,
                        title: "Knows who said what",
                        detail: "Tells speakers apart and recognizes voices you have named before.")
                Feature(symbol: "sparkles", tint: .purple,
                        title: "Summaries if you want them",
                        detail: "Your own AI model writes a summary and action items. Optional.")
            }
            .frame(maxWidth: 460, alignment: .leading)
            .padding(.top, 34)

            Spacer(minLength: 24)
            Button(action: onContinue) {
                Text("Set Up Whiz")
                    .frame(minWidth: 180)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .keyboardShortcut(.defaultAction)
            Text("Takes a few minutes, mostly downloading.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .padding(.top, 8)
                .padding(.bottom, 28)
        }
        .padding(.horizontal, 40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct Feature: View {
    let symbol: String
    let tint: Color
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: symbol)
                .font(.system(size: 22, weight: .medium))
                .foregroundStyle(tint)
                .frame(width: 34)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                Text(detail)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

// MARK: - Checklist

private struct ChecklistPage: View {
    @EnvironmentObject private var setup: SetupModel
    var onDone: () -> Void

    private var subtitle: String {
        guard setup.isReady else {
            return setup.isBundled
                ? "Whiz needs a speech model: a one-time download."
                : "Whiz needs the first three. Each one installs with a click."
        }
        if setup.isDone(.speakers) == true, setup.isDone(.ai) == true { return "Everything is set up." }
        return "Everything needed is in place. The last two are optional extras."
    }

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Get Whiz Ready")
                    .font(.title.weight(.bold))
                Text(subtitle)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 32)
            .padding(.top, 28)
            .padding(.bottom, 16)

            ScrollView {
                VStack(spacing: 10) {
                    CLIItem()
                    if setup.visibleItems.contains(.tools) { ToolsItem() }
                    ModelItem()
                    SpeakersItem()
                    AIItem()
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 16)
            }

            Divider()
            HStack {
                if setup.isChecking {
                    ProgressView().controlSize(.small)
                    Text("Checking…").foregroundStyle(.secondary)
                } else {
                    Button("Check Again") { Task { await setup.refresh() } }
                        .disabled(setup.task != nil)
                }
                Spacer()
                if !setup.isReady {
                    Button("Skip for Now", action: onDone)
                }
                if setup.isReady {
                    Button(action: onDone) {
                        Text("Start Transcribing")
                            .frame(minWidth: 130)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(setup.task != nil)
                    .keyboardShortcut(.defaultAction)
                }
            }
            .controlSize(.large)
            .padding(.horizontal, 32)
            .padding(.vertical, 16)
        }
    }
}

/// One checklist row: number/status, what it is, and what to do.
private struct ChecklistItem<Content: View>: View {
    @EnvironmentObject private var setup: SetupModel
    let item: SetupItem
    let number: Int
    let title: String
    let optional: Bool
    var summary: String
    @ViewBuilder var content: () -> Content

    var body: some View {
        let done = setup.isDone(item)
        let running = setup.task?.item == item
        HStack(alignment: .top, spacing: 14) {
            status(done: done, running: running)
                .frame(width: 26, height: 26)
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Text(title).font(.headline)
                    if optional {
                        Text("Optional")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.primary.opacity(0.07)))
                    }
                }
                Text(summary)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if running, let task = setup.task {
                    TaskProgress(task: task)
                } else {
                    content()
                }
                if let failure = setup.failures[item], !running {
                    Label(failure, systemImage: "exclamationmark.triangle.fill")
                        .font(.callout)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
        }
        .card(padding: 16)
        .opacity(done == true && !running ? 0.92 : 1)
    }

    @ViewBuilder
    private func status(done: Bool?, running: Bool) -> some View {
        if running {
            ProgressView().controlSize(.small)
        } else if done == true {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 22))
                .foregroundStyle(.green)
        } else {
            ZStack {
                Circle()
                    .strokeBorder(optional ? Color.secondary.opacity(0.5) : Color.accentColor, lineWidth: 1.5)
                Text("\(number)")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(optional ? Color.secondary : Color.accentColor)
            }
        }
    }
}

private struct TaskProgress: View {
    @EnvironmentObject private var setup: SetupModel
    let task: SetupModel.RunningTask

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(task.title).font(.callout.weight(.medium))
                Spacer()
                if let percent = task.percent {
                    Text("\(percent)%").monospacedDigit().foregroundStyle(.secondary)
                }
                Button("Stop") { setup.cancelTask() }
                    .controlSize(.small)
            }
            if let percent = task.percent {
                ProgressView(value: Double(percent), total: 100)
            } else {
                ProgressView().progressViewStyle(.linear)
            }
            if !task.detail.isEmpty {
                Text(task.detail)
                    .font(.caption.monospaced())
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
    }
}

/// A command shown as something the user could also type.
private struct CommandHint: View {
    let command: String

    var body: some View {
        HStack(spacing: 6) {
            Text(command)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(command, forType: .string)
            } label: {
                Image(systemName: "doc.on.doc")
            }
            .buttonStyle(.borderless)
            .help("Copy")
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(RoundedRectangle(cornerRadius: 5).fill(Color.primary.opacity(0.06)))
    }
}

private struct CLIItem: View {
    @EnvironmentObject private var setup: SetupModel

    var body: some View {
        ChecklistItem(item: .cli, number: setup.number(of: .cli), title: "The whiz engine", optional: false,
                      summary: summary) {
            if setup.isDone(.cli) != true {
                HStack(spacing: 10) {
                    Button(setup.whizExecutable == nil ? "Install" : "Update") { setup.installCLI() }
                        .buttonStyle(.borderedProminent)
                        .disabled(setup.task != nil || (!setup.hasPipx && !setup.hasBrew))
                    CommandHint(command: "pipx install \(SetupModel.installSource)")
                }
                if !setup.hasPipx && !setup.hasBrew {
                    HomebrewNote(item: .cli)
                }
            }
        }
    }

    private var summary: String {
        if let report = setup.report, setup.isBundled {
            return "Built into Whiz (version \(report.version)), with whisper.cpp and the audio tools. Nothing to install."
        }
        if let report = setup.report, let path = setup.whizExecutable?.path {
            return "Version \(report.version), at \(path.replacingOccurrences(of: NSHomeDirectory(), with: "~"))."
        }
        if setup.whizExecutable != nil {
            return "Installed, but it could not report its status."
        }
        return "The app does its work with the whiz command-line tool. It installs with pipx."
    }
}

private struct ToolsItem: View {
    @EnvironmentObject private var setup: SetupModel

    var body: some View {
        ChecklistItem(item: .tools, number: setup.number(of: .tools), title: "whisper.cpp and ffmpeg", optional: false,
                      summary: summary) {
            if setup.isDone(.tools) == false, setup.report != nil {
                HStack(spacing: 10) {
                    Button("Install with Homebrew") { setup.installTools() }
                        .buttonStyle(.borderedProminent)
                        .disabled(setup.task != nil || !setup.hasBrew)
                    CommandHint(command: "brew install \(missing.joined(separator: " "))")
                }
                if !setup.hasBrew { HomebrewNote(item: .tools) }
            }
        }
    }

    private var missing: [String] {
        var names: [String] = []
        if setup.report?.check("whisper_cli")?.ok != true { names.append("whisper-cpp") }
        if setup.report?.check("ffmpeg")?.ok != true { names.append("ffmpeg") }
        return names
    }

    private var summary: String {
        switch setup.isDone(.tools) {
        case true?: return "The speech engine and the audio converter are installed."
        case false? where setup.report != nil:
            let names = missing.map { $0 == "whisper-cpp" ? "whisper.cpp" : $0 }
            return "Missing: \(names.joined(separator: " and ")). whisper.cpp recognizes speech; ffmpeg reads audio from any file."
        default: return "The speech engine and the audio converter. Checked once whiz is installed."
        }
    }
}

private struct ModelItem: View {
    @EnvironmentObject private var setup: SetupModel
    @State private var choice = SpeechModelChoice.all[0]

    var body: some View {
        ChecklistItem(item: .model, number: setup.number(of: .model), title: "A speech model", optional: false,
                      summary: summary) {
            if setup.isDone(.model) == false || setup.hasQuantizedFallback {
                if setup.hasQuantizedFallback {
                    Label("This model is quantized (compressed), which costs accuracy. Download a full-quality one; Whiz switches to it on its own.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.callout)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(SpeechModelChoice.all) { option in
                        Button {
                            choice = option
                        } label: {
                            HStack(spacing: 10) {
                                Image(systemName: choice == option ? "largecircle.fill.circle" : "circle")
                                    .foregroundStyle(choice == option ? Color.accentColor : Color.secondary)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(option.title).fontWeight(.medium)
                                    Text(option.detail).font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Text(option.size).font(.callout).monospacedDigit().foregroundStyle(.secondary)
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(10)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color.primary.opacity(0.04)))
                Button("Download \(choice.title)") { setup.downloadModel(choice) }
                    .buttonStyle(.borderedProminent)
                    .disabled(setup.task != nil || setup.whizExecutable == nil)
            }
        }
    }

    private var summary: String {
        if let model = setup.report?.check("model"), model.ok {
            let size = model.sizeMb.map { Format.fileSize(Int64($0 * 1_048_576)) } ?? ""
            let using = "Using \(model.alias ?? "a whisper model")\(size.isEmpty ? "" : " (\(size))")"
            return setup.hasQuantizedFallback ? "\(using), the only model on this Mac." : "\(using). Pick another in Settings."
        }
        return "Downloaded once, then everything runs offline."
    }
}

private struct SpeakersItem: View {
    @EnvironmentObject private var setup: SetupModel

    var body: some View {
        ChecklistItem(item: .speakers, number: setup.number(of: .speakers), title: "Speaker detection", optional: true,
                      summary: setup.isDone(.speakers) == true
                        ? "Ready. Recordings are split by who is talking."
                        : "Tells voices apart and labels who said what. About 90 MB.") {
            if setup.isDone(.speakers) == false {
                Button("Set Up Speaker Detection") { setup.setupSpeakers() }
                    .disabled(setup.task != nil || setup.whizExecutable == nil)
            }
        }
    }
}

private struct AIItem: View {
    @EnvironmentObject private var setup: SetupModel
    @EnvironmentObject private var settings: SettingsModel
    @State private var selected = ""

    var body: some View {
        ChecklistItem(item: .ai, number: setup.number(of: .ai), title: "AI summaries", optional: true, summary: summary) {
            if setup.isCheckingAI {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Looking for your AI server…").foregroundStyle(.secondary)
                }
            } else if let ai = setup.report?.ai, ai.reachable == true, let models = ai.models, !models.isEmpty {
                HStack(spacing: 10) {
                    Picker("Model", selection: $selected) {
                        ForEach(models, id: \.self) { Text($0).tag($0) }
                    }
                    .frame(maxWidth: 300)
                    Button(ai.modelAvailable == true && selected == ai.model ? "Saved" : "Use This Model") {
                        settings.set("ai_model", .string(selected))
                        Task {
                            try? await Task.sleep(for: .milliseconds(400))
                            await setup.checkAI()
                        }
                    }
                    .disabled(selected.isEmpty || (ai.modelAvailable == true && selected == ai.model))
                }
                .onAppear { selected = ai.modelAvailable == true ? ai.model : (models.first ?? "") }
            } else if setup.report?.ai.reachable == false {
                HStack(spacing: 10) {
                    Link("Get Ollama", destination: URL(string: "https://ollama.com/download")!)
                    Button("Check Again") { Task { await setup.checkAI() } }
                    Text("or set another server in Settings ▸ AI")
                        .font(.caption).foregroundStyle(.secondary)
                }
            } else {
                Button("Look for an AI Server") { Task { await setup.checkAI() } }
                    .disabled(setup.whizExecutable == nil)
            }
        }
        .task(id: setup.whizExecutable) {
            if setup.whizExecutable != nil, setup.report?.ai.reachable == nil { await setup.checkAI() }
        }
    }

    private var summary: String {
        guard let ai = setup.report?.ai else {
            return "A summary and action items after each transcript, from a model you run (such as Ollama)."
        }
        if ai.modelAvailable == true { return "Using \(ai.model) at \(ai.baseUrl)." }
        if ai.reachable == true { return "Found a server at \(ai.baseUrl). Choose the model to summarize with." }
        if ai.reachable == false { return "No AI server answered at \(ai.baseUrl). Install Ollama and pull a model, or skip this." }
        return "A summary and action items after each transcript, from a model you run (such as Ollama)."
    }
}

private struct HomebrewNote: View {
    @EnvironmentObject private var setup: SetupModel
    var item: SetupItem

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Button("Install Homebrew…") { setup.installHomebrew(for: item) }
                    .disabled(setup.task != nil)
                Link("What is Homebrew?", destination: URL(string: "https://brew.sh")!)
                    .font(.callout)
            }
            Label(setup.homebrewStarted
                  ? "Finish the installer in Terminal, then come back here. Whiz checks again on its own."
                  : "Homebrew is needed first. Its installer opens in Terminal and asks for your Mac password.",
                  systemImage: setup.homebrewStarted ? "terminal" : "info.circle")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }
}

/// The compose view's reminder when an essential is still missing.
struct SetupBanner: View {
    @EnvironmentObject private var setup: SetupModel
    var open: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "wrench.and.screwdriver.fill")
                .font(.title2)
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 2) {
                Text("Whiz isn’t ready to transcribe yet")
                    .font(.headline)
                Text(reason)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Finish Setup…", action: open)
                .buttonStyle(.borderedProminent)
        }
        .card()
    }

    private var reason: String {
        switch setup.firstMissing {
        case .cli: return "The whiz command-line tool is not installed."
        case .tools: return "whisper.cpp or ffmpeg is missing."
        case .model: return "No speech model has been downloaded."
        default: return "Something it needs is missing."
        }
    }
}
