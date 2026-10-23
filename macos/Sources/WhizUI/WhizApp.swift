import SwiftUI

/// The native macOS front-end for the whiz CLI.
///
/// No `@main` here: the executable's entry point lives in `main.swift`, which
/// keeps the module importable from tests (see that file for why).
///
/// One window (`Window`, not `WindowGroup`): the sidebar holds every run, so
/// a second window would only be a second view of the same state — and two
/// windows each starting a run is how the old app got confusing.
struct WhizApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var license: LicenseModel
    @StateObject private var model: TranscriptionViewModel
    @StateObject private var settings = SettingsModel()
    @StateObject private var setup = SetupModel()
    @StateObject private var updater = UpdaterModel()

    init() {
        // One license model, shared: the run gate and the views read it.
        let license = LicenseModel()
        _license = StateObject(wrappedValue: license)
        _model = StateObject(wrappedValue: TranscriptionViewModel(license: license))
    }

    var body: some Scene {
        Window("Whiz", id: "main") {
            RootView()
                .environmentObject(model)
                .environmentObject(settings)
                .environmentObject(setup)
                .environmentObject(license)
                .frame(minWidth: 860, minHeight: 560)
        }
        .defaultSize(width: 1180, height: 780)
        .commands { WhizCommands(model: model, setup: setup, license: license, updater: updater) }

        Settings {
            SettingsView()
                .environmentObject(settings)
                .environmentObject(setup)
                .environmentObject(license)
                .environmentObject(updater)
        }

        Window("Acknowledgements", id: "acknowledgements") {
            AcknowledgementsView(acknowledgements: .bundled)
                .frame(minWidth: 640, minHeight: 420)
        }
        .defaultSize(width: 820, height: 580)
    }
}

/// File and Run menus: every action the window offers, on the keyboard.
private struct WhizCommands: Commands {
    @ObservedObject var model: TranscriptionViewModel
    @ObservedObject var setup: SetupModel
    @ObservedObject var license: LicenseModel
    @ObservedObject var updater: UpdaterModel
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(after: .appInfo) {
            if updater.isAvailable {
                Button("Check for Updates…") { updater.checkForUpdates() }
                    .disabled(!updater.canCheckForUpdates)
            }
            Button("Acknowledgements") { openWindow(id: "acknowledgements") }
            Divider()
            if !license.isLicensed {
                Button("Buy Whiz…") { NSWorkspace.shared.open(LicenseModel.purchaseURL) }
            }
            Button(license.isLicensed ? "License…" : "Enter License Key…") { license.presentActivation() }
        }
        CommandGroup(after: .appSettings) {
            Button("Setup Assistant…") { setup.isPresented = true }
        }
        CommandGroup(replacing: .newItem) {
            Button("New Transcription") { model.selection = .compose }
                .keyboardShortcut("n")
            Button("Open Recording…") {
                model.selection = .compose
                model.isPickingFile = true
            }
            .keyboardShortcut("o")
            .disabled(model.isRunning)
        }
        CommandMenu("Run") {
            Button("Transcribe") { model.start() }
                .keyboardShortcut("r")
                .disabled(model.isRunning || model.inputURL == nil)
            Button("Stop") { model.stop() }
                .keyboardShortcut(".")
                .disabled(!model.isRunning)
            Divider()
            Button("Show Progress") { model.selection = .live }
                .keyboardShortcut("l")
                .disabled(model.liveRunInput == nil)
        }
    }
}
