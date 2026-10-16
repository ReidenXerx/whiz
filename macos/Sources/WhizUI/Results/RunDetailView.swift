import AppKit
import SwiftUI

/// The tabs of a finished run.
enum DetailTab: String, CaseIterable, Identifiable {
    case transcript = "Transcript"
    case analysis = "Summary"
    case files = "Files"

    var id: String { rawValue }
}

/// What a finished run left on disk, read once per run.
struct RunResults: Sendable {
    var transcript: TranscriptDocument?
    var analysis: AnalysisDocument?
    var html: URL?

    static func load(_ record: LastRunRecord) async -> RunResults {
        await Task.detached(priority: .userInitiated) {
            let urls = record.artifacts.map(\.url)
            let transcript = TranscriptDocument.load(input: record.inputURL, artifacts: urls)
            var analysis: AnalysisDocument?
            let base = record.inputURL.deletingPathExtension().path
            let analysisURL = urls.first { $0.lastPathComponent.hasSuffix(".analysis.md") }
                ?? URL(fileURLWithPath: base + ".analysis.md")
            if let text = try? String(contentsOf: analysisURL, encoding: .utf8) {
                analysis = AnalysisDocument.parse(text)
            }
            let htmlURL = urls.first { $0.pathExtension.lowercased() == "html" }
                ?? URL(fileURLWithPath: base + ".speakers.html")
            let html = FileManager.default.fileExists(atPath: htmlURL.path) ? htmlURL : nil
            return RunResults(transcript: transcript, analysis: analysis, html: html)
        }.value
    }
}

/// A finished run: what it was, and what it found.
struct RunDetailView: View {
    @EnvironmentObject private var model: TranscriptionViewModel
    let record: LastRunRecord

    @AppStorage("dev.reidenxerx.whiz.detailTab") private var tabRaw = DetailTab.transcript.rawValue
    @StateObject private var playback: PlaybackController
    @State private var results: RunResults?
    @State private var query = ""
    /// Bumped when something new was written (a summary made from here),
    /// so the files are read again.
    @State private var reloadToken = 0

    init(record: LastRunRecord) {
        self.record = record
        _playback = StateObject(wrappedValue: PlaybackController(url: record.inputURL))
    }

    private var tab: DetailTab { DetailTab(rawValue: tabRaw) ?? .transcript }

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, 24)
                .padding(.top, 18)
                .padding(.bottom, 12)
            Divider()
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .searchable(text: $query, placement: .toolbar, prompt: "Search transcript")
        .onChange(of: query) { value in
            if !value.isEmpty, tab != .transcript { tabRaw = DetailTab.transcript.rawValue }
        }
        .task(id: "\(record.finishedAt.timeIntervalSince1970)#\(reloadToken)") {
            let loaded = await RunResults.load(record)
            // The remembered tab is a preference, not a promise: a run with
            // no summary opens on its transcript, not on an empty page.
            if tab == .analysis, loaded.analysis == nil, loaded.transcript != nil {
                tabRaw = DetailTab.transcript.rawValue
            }
            results = loaded
        }
        .onDisappear { playback.shutDown() }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: 14) {
                ThumbnailView(source: .file(record.inputURL), size: CGSize(width: 76, height: 48))
                VStack(alignment: .leading, spacing: 3) {
                    Text(record.inputURL.lastPathComponent)
                        .font(.title2.weight(.semibold))
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(record.inputPath)
                    Text(subtitle)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer(minLength: 12)
                if let html = results?.html {
                    Button {
                        NSWorkspace.shared.open(html)
                    } label: {
                        Label("Open in Browser", systemImage: "safari")
                    }
                    .help("The HTML transcript the CLI wrote, with frames and search")
                }
                Button {
                    revealOutputs()
                } label: {
                    Label("Show in Finder", systemImage: "folder")
                }
                Menu {
                    Button("Transcribe Again…") { model.prepareRerun(record) }
                    Button("Copy Transcript") { copyTranscript() }
                        .disabled(results?.transcript == nil)
                    Button("Copy Input Path") { copy(record.inputPath) }
                    Divider()
                    Button("Remove from History") { model.removeFromHistory(record) }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .help("More")
            }
            Picker("", selection: $tabRaw) {
                ForEach(DetailTab.allCases) { tab in
                    Text(tab.rawValue).tag(tab.rawValue)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 300)
        }
    }

    private var subtitle: String {
        var parts = [MediaKind(url: record.inputURL).label]
        if let duration = results?.transcript?.duration, duration > 0 {
            parts.append(Format.spokenDuration(duration))
        }
        if let speakers = results?.transcript?.speakers.count, speakers > 0 {
            parts.append(speakers == 1 ? "1 speaker" : "\(speakers) speakers")
        }
        parts.append("Transcribed \(Format.when(record.finishedAt).lowercasedFirst)")
        return parts.joined(separator: "  ·  ")
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        if let results {
            switch tab {
            case .transcript:
                if let transcript = results.transcript {
                    TranscriptView(document: transcript, playback: playback, query: query)
                } else {
                    EmptyState(symbol: "text.page.slash", title: "No transcript to show",
                               message: "None of this run's files could be read as a transcript. They may have been moved or deleted.") {
                        Button("Show Files") { tabRaw = DetailTab.files.rawValue }
                    }
                }
            case .analysis:
                AnalysisView(analysis: results.analysis, record: record,
                             hasTranscript: results.transcript != nil) { reloadToken += 1 }
            case .files:
                FilesView(record: record)
            }
        } else {
            ProgressView()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    // MARK: - Actions

    private func revealOutputs() {
        let existing = record.artifacts.map(\.url).filter { FileManager.default.fileExists(atPath: $0.path) }
        if existing.isEmpty {
            NSWorkspace.shared.activateFileViewerSelecting([record.inputURL])
        } else {
            NSWorkspace.shared.activateFileViewerSelecting(existing)
        }
    }

    private func copyTranscript() {
        guard let transcript = results?.transcript else { return }
        var lines: [String] = []
        for (index, segment) in transcript.segments.enumerated() {
            if transcript.startsTurn(at: index) {
                if !lines.isEmpty { lines.append("") }
                let who = segment.speaker.map { "\($0) " } ?? ""
                lines.append("\(who)(\(Format.clock(segment.start)))")
            }
            lines.append(segment.text)
        }
        copy(lines.joined(separator: "\n"))
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

private extension String {
    var lowercasedFirst: String {
        guard let first else { return self }
        return first.lowercased() + dropFirst()
    }
}

// MARK: - Transcript

/// The transcript, readable: speaker turns, timestamps that play the moment,
/// frames beside the lines they belong to.
struct TranscriptView: View {
    let document: TranscriptDocument
    @ObservedObject var playback: PlaybackController
    let query: String

    @AppStorage("dev.reidenxerx.whiz.followPlayback") private var follow = true
    @AppStorage("dev.reidenxerx.whiz.showPlayer") private var showPlayer = true
    @AppStorage("dev.reidenxerx.whiz.showFrames") private var showFrames = true
    @State private var enlargedFrame: URL?

    var body: some View {
        let visible = document.matchingPositions(query)
        let nowPlaying = playback.isPlaying ? document.position(at: playback.currentTime) : nil

        VStack(spacing: 0) {
            if playback.isAvailable, let player = playback.player, showPlayer {
                PlayerSurface(player: player)
                    .frame(height: playback.kind == .video ? 260 : 56)
                    .background(playback.kind == .video ? Color.black : Color.clear)
                Divider()
            }
            toolbarRow(visibleCount: visible.count)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        if visible.isEmpty {
                            Text("No lines match “\(query)”.")
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity)
                                .padding(40)
                        }
                        ForEach(visible, id: \.self) { position in
                            row(at: position, playing: position == nowPlaying)
                                .id(position)
                        }
                    }
                    .padding(.vertical, 10)
                    .frame(maxWidth: Theme.readingWidth + 160)
                    .frame(maxWidth: .infinity)
                }
                .onChange(of: nowPlaying) { id in
                    guard follow, query.isEmpty, let id else { return }
                    withAnimation(.easeInOut(duration: 0.25)) { proxy.scrollTo(id, anchor: .center) }
                }
            }
        }
        .sheet(item: Binding(get: { enlargedFrame.map(IdentifiedURL.init) },
                             set: { enlargedFrame = $0?.url })) { item in
            FrameSheet(url: item.url)
        }
    }

    private func toolbarRow(visibleCount: Int) -> some View {
        HStack(spacing: 14) {
            if document.hasSpeakers {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 14) {
                        ForEach(document.speakers) { speaker in
                            HStack(spacing: 6) {
                                SpeakerBadge(name: speaker.name, colorIndex: speaker.colorIndex)
                                Text(speaker.share, format: .percent.precision(.fractionLength(0)))
                                    .font(.caption)
                                    .monospacedDigit()
                                    .foregroundStyle(.secondary)
                            }
                            .help("\(speaker.name) talks for \(Format.spokenDuration(speaker.seconds))")
                        }
                    }
                }
            } else {
                Text("No speaker labels in this transcript")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            if !query.isEmpty {
                Text(visibleCount == 1 ? "1 match" : "\(visibleCount) matches")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            if document.hasFrames {
                iconToggle("photo.on.rectangle", isOn: $showFrames,
                           help: showFrames ? "Hide frames" : "Show frames")
            }
            if playback.isAvailable {
                iconToggle("text.line.first.and.arrowtriangle.forward", isOn: $follow,
                           help: follow ? "Stop following playback" : "Follow playback")
                iconToggle(playback.kind == .video ? "play.rectangle" : "play.circle", isOn: $showPlayer,
                           help: showPlayer ? "Hide player" : "Show player")
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
    }

    /// A quiet toolbar toggle: tinted when on, secondary when off.
    private func iconToggle(_ symbol: String, isOn: Binding<Bool>, help: String) -> some View {
        Button {
            isOn.wrappedValue.toggle()
        } label: {
            Image(systemName: symbol)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(isOn.wrappedValue ? Color.accentColor : Color.secondary)
                .frame(width: 26, height: 22)
                .background(RoundedRectangle(cornerRadius: 5)
                    .fill(isOn.wrappedValue ? Color.accentColor.opacity(0.14) : Color.clear))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(help)
    }

    private func row(at position: Int, playing: Bool) -> some View {
        let segment = document.segments[position]
        let newTurn = !query.isEmpty || document.startsTurn(at: position)
        let color = document.colorIndex(for: segment.speaker)

        return VStack(alignment: .leading, spacing: 4) {
            if newTurn, let speaker = segment.speaker {
                SpeakerBadge(name: speaker, colorIndex: color)
                    .font(.callout)
                    .padding(.leading, 58 + 14)
            }
            // Baseline-aligned: the timestamp reads as the label of the
            // line beside it, not as something floating above or below.
            HStack(alignment: .firstTextBaseline, spacing: 14) {
                Button {
                    playback.seek(to: segment.start)
                } label: {
                    Text(Format.clock(segment.start))
                        .font(.system(.callout, design: .monospaced))
                        .foregroundStyle(playing ? Color.accentColor : Color.secondary)
                        .frame(width: 58, alignment: .trailing)
                }
                .buttonStyle(.plain)
                .disabled(!playback.isAvailable)
                .help(playback.isAvailable ? "Play from here" : "The recording is no longer at its original location")

                Text(highlighted(segment.text))
                    .font(.body)
                    .lineSpacing(3)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)

                if showFrames, let frame = segment.frame {
                    Button {
                        enlargedFrame = frame
                    } label: {
                        ThumbnailView(source: .frame(frame), size: CGSize(width: 96, height: 54))
                    }
                    .buttonStyle(.plain)
                    .help("Enlarge frame")
                    // Top of the thumbnail level with the top of the text.
                    .alignmentGuide(.firstTextBaseline) { d in d[.top] + 13 }
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, newTurn ? 14 : 3)
        .padding(.bottom, 3)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(playing ? Color.accentColor.opacity(0.09) : Color.clear)
                .padding(.horizontal, 8)
        )
    }

    private func highlighted(_ text: String) -> AttributedString {
        var attributed = AttributedString(text)
        let needle = query.trimmingCharacters(in: .whitespaces)
        guard !needle.isEmpty else { return attributed }
        var searchStart = attributed.startIndex
        while searchStart < attributed.endIndex,
              let range = attributed[searchStart...].range(of: needle, options: [.caseInsensitive, .diacriticInsensitive]) {
            attributed[range].backgroundColor = Color.yellow.opacity(0.45)
            searchStart = range.upperBound
        }
        return attributed
    }
}

private struct IdentifiedURL: Identifiable {
    var id: URL { url }
    let url: URL
}

/// A frame at full size.
private struct FrameSheet: View {
    let url: URL
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            if let image = NSImage(contentsOf: url) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(minWidth: 640, maxWidth: 1100, minHeight: 360, maxHeight: 700)
            }
            HStack {
                Text(url.lastPathComponent).foregroundStyle(.secondary)
                Spacer()
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(12)
        }
    }
}

// MARK: - Analysis

/// The AI summary, rendered.
struct AnalysisView: View {
    @EnvironmentObject private var model: TranscriptionViewModel
    @EnvironmentObject private var settings: SettingsModel
    let analysis: AnalysisDocument?
    let record: LastRunRecord
    let hasTranscript: Bool
    var onWritten: () -> Void
    @State private var showPrompt = false
    @StateObject private var runner = AnalyzeRunner()

    var body: some View {
        if let analysis {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    HStack(spacing: 8) {
                        if let model = analysis.model { chip("cpu", model) }
                        if let mode = analysis.mode { chip("text.bubble", mode.capitalized) }
                        if analysis.vision == true { chip("eye", "Saw the frames") }
                        Spacer()
                    }
                    MarkdownView(analysis.response)
                    if let prompt = analysis.prompt, !prompt.isEmpty {
                        DisclosureGroup(isExpanded: $showPrompt) {
                            Text(prompt)
                                .font(.system(.callout, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.top, 6)
                        } label: {
                            Text("Prompt used").font(.subheadline.weight(.semibold)).foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.horizontal, 32)
                .padding(.vertical, 24)
                .frame(maxWidth: Theme.readingWidth)
                .frame(maxWidth: .infinity)
            }
        } else if runner.isRunning {
            VStack(spacing: 14) {
                ProgressView()
                Text(runner.status)
                    .font(.headline)
                Text("Your AI model is reading the transcript. Long recordings take a few minutes.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Button("Stop") { runner.stop() }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            EmptyState(symbol: "sparkles", title: "No AI summary for this run",
                       message: emptyMessage) {
                VStack(spacing: 10) {
                    Button("Summarize Now") {
                        runner.run(input: record.inputURL, onWritten: onWritten)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(!hasTranscript || settings.string("ai_model").isEmpty)
                    if let error = runner.error {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .font(.callout)
                            .foregroundStyle(.red)
                            .textSelection(.enabled)
                            .frame(maxWidth: 460)
                    }
                }
            }
        }
    }

    private var emptyMessage: String {
        let model = settings.string("ai_model")
        if model.isEmpty {
            return "Choose an AI model in Settings ▸ AI (or the Setup Assistant) to get a summary and action items."
        }
        return "Get a summary and action items for this transcript from \(model)."
    }

    private func chip(_ symbol: String, _ text: String) -> some View {
        Label(text, systemImage: symbol)
            .font(.caption.weight(.medium))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(Capsule().fill(Color.primary.opacity(0.06)))
    }
}

/// Runs `whiz analyze` for a finished run.
@MainActor
final class AnalyzeRunner: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var status = "Starting…"
    @Published private(set) var error: String?
    private var backend: CLIBackend?

    func run(input: URL, onWritten: @escaping () -> Void) {
        guard !isRunning else { return }
        guard let whiz = WhizLocator.find() else {
            error = "The whiz command was not found."
            return
        }
        let backend = CLIBackend(executable: whiz)
        self.backend = backend
        isRunning = true
        error = nil
        status = "Starting…"
        let (events, continuation) = AsyncStream<TranscriptionEvent>.makeStream()
        Task {
            let consumer = Task {
                for await event in events {
                    if case .phase(let label) = event {
                        status = PhaseStep(id: 0, label: label, startedAt: .now).title + "…"
                    }
                }
            }
            do {
                try await backend.run(arguments: ["analyze", input.path]) { continuation.yield($0) }
                continuation.finish()
                await consumer.value
                onWritten()
            } catch is CancellationError {
                continuation.finish()
                await consumer.value
            } catch let failure as TranscriptionFailure {
                continuation.finish()
                await consumer.value
                error = failure.errorDescription
            } catch {
                continuation.finish()
                await consumer.value
                self.error = error.localizedDescription
            }
            isRunning = false
            self.backend = nil
        }
    }

    func stop() { backend?.cancel() }
}

// MARK: - Files

/// Every file the run touched, with the one action each needs.
struct FilesView: View {
    let record: LastRunRecord

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 8) {
                    SectionLabel("Recording")
                    VStack(spacing: 0) {
                        FileRow(label: MediaKind(url: record.inputURL).label, url: record.inputURL)
                    }
                    .card(padding: 0)
                }
                VStack(alignment: .leading, spacing: 8) {
                    SectionLabel("Written by whiz")
                    VStack(spacing: 0) {
                        ForEach(Array(record.artifacts.enumerated()), id: \.element.id) { index, artifact in
                            if index > 0 { Divider().padding(.leading, 52) }
                            FileRow(label: artifact.label.replacingOccurrences(of: "Wrote ", with: ""),
                                    url: artifact.url)
                        }
                    }
                    .card(padding: 0)
                }
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 24)
            .frame(maxWidth: Theme.readingWidth)
            .frame(maxWidth: .infinity)
        }
    }
}

private struct FileRow: View {
    let label: String
    let url: URL

    var body: some View {
        let exists = FileManager.default.fileExists(atPath: url.path)
        HStack(spacing: 12) {
            Image(nsImage: NSWorkspace.shared.icon(forFile: url.path))
                .resizable()
                .frame(width: 28, height: 28)
                .opacity(exists ? 1 : 0.4)
            VStack(alignment: .leading, spacing: 2) {
                Text(label.prefix(1).uppercased() + label.dropFirst())
                Text(exists ? url.lastPathComponent : "Missing — \(url.lastPathComponent)")
                    .font(.caption)
                    .foregroundStyle(exists ? Color.secondary : Color.red)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(url.path)
            }
            Spacer()
            if exists {
                if let size = MediaInfo.fileSize(of: url) {
                    Text(Format.fileSize(size))
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(.tertiary)
                }
                Button("Open") { NSWorkspace.shared.open(url) }
                    .controlSize(.small)
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                } label: {
                    Image(systemName: "magnifyingglass")
                }
                .controlSize(.small)
                .help("Show in Finder")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .contextMenu {
            Button("Open") { NSWorkspace.shared.open(url) }
            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            Button("Copy Path") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(url.path, forType: .string)
            }
        }
    }
}
