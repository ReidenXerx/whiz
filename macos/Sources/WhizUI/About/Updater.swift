import AppKit
import Combine
import Foundation
import Sparkle
import SwiftUI

/// In-app updates, through Sparkle: Whiz > Check for Updates… and the
/// Updates settings tab.
///
/// The feed (SUFeedURL in Info.plist) is per major version, `updates/v1.xml`
/// for 0.x and 1.x: a licence covers its major version, so a 1.x copy must
/// never be offered 2.0 as an ordinary update and then refuse the buyer's
/// key. 2.0 gets its own feed; the v1 feed can announce it as an
/// informational item that only links to the upgrade page.
///
/// Every update is checked against SUPublicEDKey before it installs (the
/// private half signs releases: publish-update.sh). Sparkle asks the user on
/// the second launch whether to check automatically; until they agree,
/// Whiz contacts nothing.
@MainActor
final class UpdaterModel: ObservableObject {
    @Published private(set) var canCheckForUpdates = false
    @Published private(set) var automaticallyChecks = false
    @Published private(set) var lastCheck: Date?

    private let controller: SPUStandardUpdaterController?
    private let delegate = UpdaterDelegate()
    private var observers: Set<AnyCancellable> = []

    /// Off when the bundle has no feed: `swift run`, the tests, a build
    /// that did not go through build-app.sh. Sparkle would otherwise greet
    /// each launch with "the updater failed to start".
    init(bundle: Bundle = .main) {
        guard Self.feedURL(in: bundle) != nil else {
            controller = nil
            return
        }
        let controller = SPUStandardUpdaterController(startingUpdater: true,
                                                      updaterDelegate: delegate, userDriverDelegate: nil)
        self.controller = controller
        let updater = controller.updater
        updater.publisher(for: \.canCheckForUpdates)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] in self?.canCheckForUpdates = $0 }
            .store(in: &observers)
        updater.publisher(for: \.automaticallyChecksForUpdates)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] in self?.automaticallyChecks = $0 }
            .store(in: &observers)
        updater.publisher(for: \.lastUpdateCheckDate)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] in self?.lastCheck = $0 }
            .store(in: &observers)
    }

    static func feedURL(in bundle: Bundle) -> URL? {
        (bundle.object(forInfoDictionaryKey: "SUFeedURL") as? String).flatMap(URL.init(string:))
    }

    var isAvailable: Bool { controller != nil }

    func checkForUpdates() {
        controller?.checkForUpdates(nil)
    }

    func setAutomaticallyChecks(_ on: Bool) {
        controller?.updater.automaticallyChecksForUpdates = on
    }
}

extension Notification.Name {
    /// Sparkle is about to quit Whiz to install an update: every view that
    /// presents a sheet closes it (RootView). See UpdaterDelegate.
    static let whizWillRelaunchForUpdate = Notification.Name("dev.reidenxerx.whiz.willRelaunchForUpdate")
}

/// "Install and Relaunch" quits Whiz to replace it, and AppKit will not
/// terminate an app while a window has a sheet attached (the setup
/// assistant, the license sheet): the quit is dropped before the app
/// delegate is even asked, and the update waits for a relaunch that never
/// comes. NSWindow.endSheet is not enough for a SwiftUI sheet, which comes
/// straight back while its isPresented is still true, so the views close
/// their own (whizWillRelaunchForUpdate) and the relaunch waits for them.
final class UpdaterDelegate: NSObject, SPUUpdaterDelegate {
    func updater(_ updater: SPUUpdater, shouldPostponeRelaunchForUpdate item: SUAppcastItem,
                 untilInvokingBlock installHandler: @escaping () -> Void) -> Bool {
        // Sparkle calls this on the main thread and wants the block called
        // there; it is not declared Sendable, but never leaves that thread.
        nonisolated(unsafe) let install = installHandler
        Task { @MainActor in
            await Self.closeSheets()
            install()
        }
        return true
    }

    /// Up to three seconds; a sheet that will not close must not cost the
    /// update, so the relaunch goes ahead regardless (and macOS's own quit
    /// handling has the last word).
    @MainActor
    static func closeSheets() async {
        NotificationCenter.default.post(name: .whizWillRelaunchForUpdate, object: nil)
        for attempt in 0..<30 {
            let windows = NSApplication.shared.windows
            if windows.allSatisfy({ $0.attachedSheet == nil }) { return }
            if attempt == 10 {
                // A sheet no view owns (an AppKit panel): close it directly.
                for window in windows {
                    if let sheet = window.attachedSheet { window.endSheet(sheet) }
                }
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
    }
}

struct UpdateSettings: View {
    @EnvironmentObject private var updater: UpdaterModel

    var body: some View {
        Form {
            if updater.isAvailable {
                Section("Updates") {
                    Toggle("Check for updates automatically", isOn: Binding(
                        get: { updater.automaticallyChecks },
                        set: { updater.setAutomaticallyChecks($0) }))
                    LabeledContent("Last checked", value: lastChecked)
                    Button("Check Now") { updater.checkForUpdates() }
                        .disabled(!updater.canCheckForUpdates)
                }
                Section {
                    Text("Your license covers every \(majorLine) update. A check sends duduphudu.app only the Whiz version and your preferred language; your recordings never leave your Mac.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            } else {
                Section("Updates") {
                    Text("This build of Whiz does not update itself. The app from duduphudu.app does.")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }

    private var lastChecked: String {
        updater.lastCheck.map { $0.formatted(date: .abbreviated, time: .shortened) } ?? "Never"
    }

    /// 0.x releases are covered by a 1.x licence (LicenseKey.verify).
    private var majorLine: String { "\(max(1, LicenseModel.bundleMajor)).x" }
}
