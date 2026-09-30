import SwiftUI

// Entry point. As a top-level-code file, `main.swift` is what SwiftPM excludes
// when building the module for `swift test`, so the test target gets a clean
// testable library while the executable product still receives a main symbol
// — `@main` alone cannot give both builds what they need.
WhizApp.main()