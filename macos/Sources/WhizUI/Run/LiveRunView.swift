import AppKit
import Combine
import SwiftUI

/// The run in progress — or the one that just failed or was stopped.
///
/// The steps are the story: what whiz has done, what it is doing, how far
/// along, how long is left. The verbatim log is still one click away under
/// Details, because when something goes wrong it is the most honest account
/// there is — it just is not the main character any more.
struct LiveRunView: View {
    @EnvironmentObject private var model: TranscriptionViewModel
    @AppStorage("dev.reidenxerx.whiz.showRunDetails") private var showDetails = false
    @State private var now = Date()

    private let ticker = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                if let prompt = model.pendingSpeakerPrompt {
                    SpeakerNamingCard(prompt: prompt)
                        .id(prompt.id)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
                if model.lastOutcome == .failed {
                    failureCard
                }
                stepsCard
                if !model.artifacts.isEmpty {
                    writtenSoFar
                }
                detailsSection
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 28)
            .frame(maxWidth: Theme.readingWidth)
            .frame(maxWidth: .infinity)
            .animation(.easeOut(duration: 0.2), value: model.pendingSpeakerPrompt)
        }
        .onReceive(ticker) { now = $0 }
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .center, spacing: 14) {
            if let input = model.liveRunInput {
                ThumbnailView(source: .file(input), size: CGSize(width: 72, height: 46))
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(model.liveRunInput?.lastPathComponent ?? "Transcription")
                    .font(.title2.weight(.semibold))
                    .lineLimit(1)
                    .truncationMode(.middle)
                statusLine
            }
            Spacer()
            if model.isRunning {
                Button(role: .destructive) {
                    model.stop()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .controlSize(.large)
                .keyboardShortcut(".", modifiers: .command)
            } else {
                Button("Edit & Retry") {
                    if let input = model.liveRunInput { model.choose(input) }
                }
                .controlSize(.large)
                Button("Done") { model.dismissLiveRun() }
                    .controlSize(.large)
            }
        }
    }

    @ViewBuilder
    private var statusLine: some View {
        switch model.lastOutcome {
        case .running:
            Text("Running for \(Format.clock(now.timeIntervalSince(model.runStartedAt ?? now)))")
                .monospacedDigit()
                .foregroundStyle(.secondary)
        case .failed:
            Label("Failed after \(Format.clock(model.lastDuration ?? 0))",
                  systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
        case .stopped:
            Label("Stopped after \(Format.clock(model.lastDuration ?? 0))", systemImage: "stop.circle")
                .foregroundStyle(.secondary)
        case .finished:
            Label("Finished in \(Format.clock(model.lastDuration ?? 0)) — nothing new was written",
                  systemImage: "checkmark.circle")
                .foregroundStyle(.secondary)
        case nil:
            EmptyView()
        }
    }

    // MARK: - Steps

    private var stepsCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            if model.steps.isEmpty {
                StepRow(symbol: .spinner, title: model.isRunning ? "Starting whiz…" : "No steps ran",
                        trailing: nil, progress: nil, note: nil)
            }
            ForEach(Array(model.steps.enumerated()), id: \.element.id) { index, step in
                if index > 0 { Divider().padding(.leading, 44) }
                stepRow(step)
            }
        }
        .card(padding: 4)
    }

    private func stepRow(_ step: PhaseStep) -> some View {
        switch step.state {
        case .done:
            return StepRow(symbol: .done, title: step.title,
                           trailing: step.duration.map(Format.clock), progress: nil, note: nil)
        case .failed:
            return StepRow(symbol: .failed, title: step.title,
                           trailing: step.duration.map(Format.clock), progress: nil, note: nil)
        case .stopped:
            return StepRow(symbol: .stopped, title: step.title,
                           trailing: step.duration.map(Format.clock), progress: nil, note: "Stopped here")
        case .active:
            let elapsed = now.timeIntervalSince(step.startedAt)
            let percent = model.progressPercent
            let remaining = TranscriptionViewModel.estimatedRemaining(percent: percent, elapsed: elapsed)
            var trailing = Format.clock(max(elapsed, 0))
            if let percent { trailing = "\(percent)%  ·  " + trailing }
            return StepRow(symbol: .spinner, title: step.title, trailing: trailing,
                           progress: percent.map { Double($0) / 100 },
                           note: remaining.map(Format.remaining)
                               ?? (model.pendingSpeakerPrompt != nil ? "Waiting for you to name a speaker" : nil))
        }
    }

    // MARK: - Failure / output

    private var failureCard: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.octagon.fill")
                .font(.title2)
                .foregroundStyle(.red)
            VStack(alignment: .leading, spacing: 6) {
                Text("whiz stopped with an error")
                    .font(.headline)
                Text(model.errorMessage ?? "See Details below for the full log.")
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                HStack {
                    Button("Copy Error") { copy(model.errorMessage ?? "") }
                    Button("Show Details") { showDetails = true }
                }
                .controlSize(.small)
                .padding(.top, 2)
            }
        }
        .card()
        .overlay(RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
            .strokeBorder(Color.red.opacity(0.35), lineWidth: 1))
    }

    private var writtenSoFar: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("Written so far")
            FlowChips(items: model.artifacts.map { ($0.label.replacingOccurrences(of: "Wrote ", with: ""), $0.url) })
        }
    }

    private var detailsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button {
                    withAnimation(.easeOut(duration: 0.15)) { showDetails.toggle() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "chevron.right")
                            .rotationEffect(.degrees(showDetails ? 90 : 0))
                            .font(.caption.weight(.semibold))
                        Text("Details")
                            .font(.subheadline.weight(.semibold))
                    }
                    .foregroundStyle(.secondary)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                Spacer()
                if showDetails {
                    if model.logLines.count >= TranscriptionViewModel.logLineCap {
                        Text("last \(TranscriptionViewModel.logLineCap) lines")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                    Button("Copy Log") { copy(model.logLines.joined(separator: "\n")) }
                        .controlSize(.small)
                }
            }
            if showDetails {
                LogView(lines: model.logLines)
                    .frame(height: 280)
            }
        }
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// One step of the timeline.
private struct StepRow: View {
    enum Symbol { case spinner, done, failed, stopped }

    let symbol: Symbol
    let title: String
    let trailing: String?
    let progress: Double?
    let note: String?

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            icon
                .frame(width: 20, height: 20)
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(title)
                        .fontWeight(symbol == .spinner ? .semibold : .regular)
                    Spacer()
                    if let trailing {
                        Text(trailing)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
                if let progress {
                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                }
                if let note {
                    Text(note)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 11)
    }

    @ViewBuilder
    private var icon: some View {
        switch symbol {
        case .spinner:
            ProgressView().controlSize(.small)
        case .done:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed:
            Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
        case .stopped:
            Image(systemName: "stop.circle.fill").foregroundStyle(.secondary)
        }
    }
}

/// "Who is this?" — asked while the run waits. The field starts from the
/// suggestion (a voice-profile match: Return confirms it, which the CLI
/// counts as a human confirmation); an emptied field or "Keep" sends the
/// CLI's '-' decline, keeping the default label even over a suggestion.
private struct SpeakerNamingCard: View {
    @EnvironmentObject private var model: TranscriptionViewModel
    let prompt: SpeakerNamePrompt
    @State private var draft = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: "person.crop.circle.badge.questionmark")
                    .font(.title)
                    .foregroundStyle(Color.accentColor)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Who is \(prompt.label)?")
                        .font(.headline)
                    Text(prompt.suggestion.isEmpty
                         ? "whiz found a voice it does not know yet."
                         : "This sounds like \(display(prompt.suggestion)). Press Return to confirm.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            if !prompt.quote.isEmpty {
                HStack(spacing: 10) {
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(Color.accentColor.opacity(0.5))
                        .frame(width: 3)
                    Text(display(prompt.quote))
                        .italic()
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            HStack(spacing: 10) {
                TextField("", text: $draft, prompt: Text("Name"))
                    .textFieldStyle(.roundedBorder)
                    .controlSize(.large)
                    .focused($focused)
                    .onSubmit(submit)
                Button("Save Name", action: submit)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(draft.trimmingCharacters(in: .whitespaces).isEmpty)
                Button("Keep “\(prompt.label)”") {
                    draft = ""
                    submit()
                }
                .controlSize(.large)
            }
            Text("Names are remembered as voice profiles, so this person is recognized next time.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .card(padding: 18)
        .overlay(RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
            .strokeBorder(Color.accentColor.opacity(0.55), lineWidth: 1.5))
        .onAppear {
            draft = display(prompt.suggestion)
            focused = true
        }
    }

    /// The machine line swaps " | " inside fields for U+01C0 so the split
    /// stays exact; show and send back the real character.
    private func display(_ text: String) -> String {
        text.replacingOccurrences(of: "\u{01C0}", with: "|")
    }

    private func submit() {
        let name = draft.trimmingCharacters(in: .whitespaces)
        model.answerSpeakerName(name.isEmpty ? "-" : name)
        draft = ""
    }
}

/// The verbatim run log: monospaced, selectable, pinned to the bottom.
struct LogView: View {
    let lines: [String]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(lines.indices, id: \.self) { index in
                        Text(lines[index])
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Self.color(for: lines[index]))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(index)
                    }
                }
                .textSelection(.enabled)
                .padding(10)
            }
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color(nsColor: .textBackgroundColor)))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Color(nsColor: .separatorColor), lineWidth: 0.5))
            .onAppear { proxy.scrollTo(lines.count - 1, anchor: .bottom) }
            .onChange(of: lines.count) { count in
                guard count > 0 else { return }
                proxy.scrollTo(count - 1, anchor: .bottom)
            }
        }
    }

    /// The machine lines stand out; everything else is quiet.
    private static func color(for line: String) -> Color {
        if line.hasPrefix("▸ ") { return .accentColor }
        if line.hasPrefix("✓ ") { return .green }
        let lower = line.lowercased()
        if lower.contains("error") || lower.contains("failed") || lower.hasPrefix("warning") { return .orange }
        return .secondary
    }
}

/// Artifact chips that open their file.
struct FlowChips: View {
    let items: [(String, URL)]

    var body: some View {
        // A simple wrap: LazyVGrid with adaptive columns reads as a flow.
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 8, alignment: .leading)],
                  alignment: .leading, spacing: 8) {
            ForEach(items, id: \.1) { label, url in
                Button {
                    NSWorkspace.shared.open(url)
                } label: {
                    HStack(spacing: 6) {
                        Image(nsImage: NSWorkspace.shared.icon(forFile: url.path))
                            .resizable()
                            .frame(width: 16, height: 16)
                        Text(label.prefix(1).uppercased() + label.dropFirst())
                            .lineLimit(1)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Capsule().fill(Color.primary.opacity(0.06)))
                }
                .buttonStyle(.plain)
                .help(url.path)
            }
        }
    }
}
