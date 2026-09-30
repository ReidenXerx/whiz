import SwiftUI

/// The native macOS front-end for the whiz CLI.
///
/// No `@main` here: the executable's entry point lives in `main.swift`, which
/// keeps the module importable from tests (see that file for why).
struct WhizApp: App {
    var body: some Scene {
        WindowGroup("whiz") {
            MainView()
                .frame(minWidth: 720, minHeight: 540)
        }
    }
}