import Foundation
import SwiftUI

// Entry point. As a top-level-code file, `main.swift` is what SwiftPM excludes
// when building the module for `swift test`, so the test target gets a clean
// testable library while the executable product still receives a main symbol
// — `@main` alone cannot give both builds what they need.

// A speaker-name answer written to a child that already died (it crashed, or
// finished between the prompt and the click) would raise SIGPIPE, whose
// default action kills the whole app. Ignored, the write fails with EPIPE
// instead, which CLIBackend's stdin path swallows (it also sets
// F_SETNOSIGPIPE per descriptor; this covers every other pipe too).
signal(SIGPIPE, SIG_IGN)

WhizApp.main()
