import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The window: runs on the left, the selected one on the right.
struct RootView: View {
    @EnvironmentObject private var model: TranscriptionViewModel
    @EnvironmentObject private var settings: SettingsModel
    @EnvironmentObject private var setup: SetupModel
    @EnvironmentObject private var license: LicenseModel
    @EnvironmentObject private var recorder: RecorderModel
    @ObservedObject private var router = FileOpenRouter.shared
    /// Set once the assistant has been finished or skipped; until then it
    /// opens on launch.
    @AppStorage("dev.reidenxerx.whiz.setupDone") private var setupDone = false

    var body: some View {
        NavigationSplitView {
            SidebarView()
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 360)
        } detail: {
            detail
                .frame(minWidth: 560)
                .background(Color(nsColor: .windowBackgroundColor))
                // On the detail, not beside the setup sheet: one view, one sheet.
                .sheet(isPresented: $license.isActivationPresented) {
                    ActivationView().environmentObject(license)
                }
        }
        .navigationTitle(title)
        .fileImporter(isPresented: $model.isPickingFile,
                      allowedContentTypes: [.movie, .audio, .audiovisualContent],
                      allowsMultipleSelection: false) { result in
            if case .success(let urls) = result, let url = urls.first { model.choose(url) }
        }
        .onDrop(of: [UTType.fileURL], isTargeted: nil) { providers in
            guard !model.isRunning, let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url, !url.hasDirectoryPath else { return }
                Task { @MainActor in model.choose(url) }
            }
            return true
        }
        .onReceive(router.$pending.compactMap { $0 }) { url in
            model.choose(url)
            router.pending = nil
        }
        // A run waiting on a name, or one that just ended, is worth a bounce
        // of the Dock icon when the user is in another app.
        .onChange(of: model.pendingSpeakerPrompt) { prompt in
            if prompt != nil { NSApp?.requestUserAttention(.criticalRequest) }
        }
        .onChange(of: model.isRunning) { running in
            if !running { NSApp?.requestUserAttention(.informationalRequest) }
        }
        // An open sheet stops the quit an update's relaunch needs.
        .onReceive(NotificationCenter.default.publisher(for: .whizWillRelaunchForUpdate)) { _ in
            setup.isPresented = false
            license.isActivationPresented = false
            model.isPickingFile = false
        }
        .sheet(isPresented: $setup.isPresented) {
            SetupAssistantView {
                setupDone = true
                Task { await settings.reload() }
            }
            .environmentObject(setup)
            .environmentObject(settings)
        }
        .task {
            await settings.reload()
            await setup.refresh()
            if !setupDone { setup.isPresented = true }
        }
    }

    /// The detail views carry their own large title; the toolbar names the
    /// section, so the file name is not printed twice.
    private var title: String {
        switch model.selection {
        case .compose, .record: return "Whiz"
        case .live: return model.isRunning ? "Transcribing" : "Last Run"
        case .run: return "Whiz"
        }
    }

    @ViewBuilder
    private var detail: some View {
        switch model.selection {
        case .compose:
            ComposeView()
        case .record:
            RecordView()
        case .live:
            if model.liveRunInput != nil {
                LiveRunView()
            } else {
                ComposeView()
            }
        case .run(let path):
            if let record = model.record(for: path) {
                RunDetailView(record: record)
                    .id(record.inputPath + "\(record.finishedAt.timeIntervalSince1970)")
            } else {
                ComposeView()
            }
        }
    }
}

/// New, now, and recent.
private struct SidebarView: View {
    @EnvironmentObject private var model: TranscriptionViewModel

    var body: some View {
        List(selection: selection) {
            Label("New Transcription", systemImage: "plus.circle")
                .tag(SidebarItem.compose)
            RecordRow()
                .tag(SidebarItem.record)

            if let input = model.liveRunInput {
                Section(model.isRunning ? "In Progress" : "Last Run") {
                    LiveRow(input: input)
                        .tag(SidebarItem.live)
                }
            }

            Section("Recent") {
                if model.recentRuns.isEmpty {
                    Text("Finished runs appear here.")
                        .font(.callout)
                        .foregroundStyle(.tertiary)
                }
                ForEach(model.recentRuns) { record in
                    RecentRow(record: record)
                        .tag(SidebarItem.run(record.inputPath))
                        .contextMenu {
                            Button("Show in Finder") {
                                NSWorkspace.shared.activateFileViewerSelecting([record.inputURL])
                            }
                            Button("Transcribe Again…") { model.prepareRerun(record) }
                                .disabled(model.isRunning)
                            Divider()
                            Button("Remove from History") { model.removeFromHistory(record) }
                        }
                }
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .bottom, spacing: 0) { TrialFooter() }
        .toolbar {
            ToolbarItem {
                Button {
                    model.selection = .compose
                } label: {
                    Label("New Transcription", systemImage: "square.and.pencil")
                }
                .help("New Transcription (⌘N)")
            }
        }
    }

    /// List selection is optional; the model's is not.
    private var selection: Binding<SidebarItem?> {
        Binding(get: { model.selection }, set: { if let item = $0 { model.selection = item } })
    }
}

/// "Record a Call", or the recording's clock while one runs.
private struct RecordRow: View {
    @EnvironmentObject private var recorder: RecorderModel

    var body: some View {
        if recorder.isBusy {
            Label {
                HStack {
                    Text("Recording")
                    Spacer()
                    Text(RecorderModel.clock(recorder.elapsed))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            } icon: {
                Image(systemName: "record.circle.fill").foregroundStyle(.red)
            }
        } else {
            Label("Record a Call", systemImage: "record.circle")
        }
    }
}

private struct LiveRow: View {
    @EnvironmentObject private var model: TranscriptionViewModel
    let input: URL

    var body: some View {
        HStack(spacing: 10) {
            ZStack {
                if model.isRunning {
                    if let percent = model.progressPercent {
                        ProgressView(value: Double(percent), total: 100)
                            .progressViewStyle(.circular)
                            .controlSize(.small)
                    } else {
                        ProgressView().controlSize(.small)
                    }
                } else if model.lastOutcome == .failed {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
                } else {
                    Image(systemName: "stop.circle").foregroundStyle(.secondary)
                }
            }
            .frame(width: 18)
            VStack(alignment: .leading, spacing: 2) {
                Text(input.lastPathComponent)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(status)
                    .font(.caption)
                    .foregroundStyle(model.pendingSpeakerPrompt != nil ? Color.accentColor : Color.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 2)
    }

    private var status: String {
        if model.pendingSpeakerPrompt != nil { return "Waiting for a name" }
        switch model.lastOutcome {
        case .failed: return "Failed"
        case .stopped: return "Stopped"
        default:
            let phase = model.steps.last?.title ?? "Starting"
            return model.progressPercent.map { "\(phase) · \($0)%" } ?? phase
        }
    }
}

private struct RecentRow: View {
    let record: LastRunRecord

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: MediaKind(url: record.inputURL).symbol)
                .foregroundStyle(.secondary)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 2) {
                Text(record.inputURL.deletingPathExtension().lastPathComponent)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(Format.when(record.finishedAt))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
        .help(record.inputPath)
    }
}

/// Files opened from Finder ("Open With", a drop on the Dock icon, `open -a`)
/// arrive at the app delegate before any view exists; this hands them over.
@MainActor
final class FileOpenRouter: ObservableObject {
    static let shared = FileOpenRouter()
    @Published var pending: URL?
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func application(_ application: NSApplication, open urls: [URL]) {
        guard let url = urls.first(where: { $0.isFileURL }) else { return }
        MainActor.assumeIsolated { FileOpenRouter.shared.pending = url }
    }

    /// One window; closing it should not leave a windowless app in the Dock
    /// that the user then has to quit separately — unless a recording is
    /// running, which goes on in the menu bar.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        MainActor.assumeIsolated { !RecorderModel.shared.isBusy }
    }

    /// Quitting mid-recording finishes the file first, so the call is kept.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        MainActor.assumeIsolated {
            let recorder = RecorderModel.shared
            guard recorder.isBusy else { return .terminateNow }
            Task { @MainActor in
                while recorder.state == .starting { try? await Task.sleep(for: .milliseconds(50)) }
                await recorder.stop(transcribe: false)
                // A stop already under way (from the menu bar) finishes too.
                while recorder.isBusy { try? await Task.sleep(for: .milliseconds(50)) }
                sender.reply(toApplicationShouldTerminate: true)
            }
            return .terminateLater
        }
    }
}
