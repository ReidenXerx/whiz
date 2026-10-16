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
    @StateObject private var model = TranscriptionViewModel()
    @StateObject private var settings = SettingsModel()

    var body: some Scene {
        Window("Whiz", id: "main") {
            RootView()
                .environmentObject(model)
                .environmentObject(settings)
                .frame(minWidth: 860, minHeight: 560)
        }
        .defaultSize(width: 1180, height: 780)
        .commands { WhizCommands(model: model) }

        Settings {
            SettingsView()
                .environmentObject(settings)
        }
    }
}

/// File and Run menus: every action the window offers, on the keyboard.
private struct WhizCommands: Commands {
    @ObservedObject var model: TranscriptionViewModel

    var body: some Commands {
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
