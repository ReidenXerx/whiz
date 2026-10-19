// swift-tools-version: 6.0
import PackageDescription

// The native macOS front-end for the whiz CLI.
//
// This package deliberately contains NO pipeline logic. Decoding, frame
// extraction, diarization, merging, speaker profiles and AI analysis all live
// in the Python package and stay there: duplicating them in Swift would create
// a second implementation of one engine, which is the drift problem NS-8
// records and ARCHITECTURE.md warns about ("if whiz ever grows a second
// implementation of the pipeline, the pattern moves with the need").
//
// What lives here is only what cannot live anywhere else: AppKit/SwiftUI, and
// the process driver that runs the CLI. `TranscriptionBackend` is the seam —
// when the shared core becomes C++, a second conformance replaces
// `CLIBackend` and the UI does not change.
//
// One exception, and it is not pipeline logic: `whiz-media` decodes audio
// and grabs video frames with AVFoundation so Whiz.app can carry its engine
// without ffmpeg. whiz still decides what to extract and when; the helper
// is a codec behind the same argv-and-exit-status contract as ffmpeg.
//
// macOS 13 (Ventura) floor: the oldest release with MenuBarExtra and
// SMAppService, and it still covers Macs back to 2017.
let package = Package(
    name: "WhizUI",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "WhizUI", targets: ["WhizUI"]),
        .executable(name: "whiz-media", targets: ["WhizMedia"]),
    ],
    targets: [
        .executableTarget(
            name: "WhizUI",
            path: "Sources/WhizUI"
        ),
        .testTarget(
            name: "WhizUITests",
            dependencies: ["WhizUI"],
            path: "Tests/WhizUITests"
        ),
        .executableTarget(
            name: "WhizMedia",
            path: "Sources/WhizMedia"
        ),
        .testTarget(
            name: "WhizMediaTests",
            dependencies: ["WhizMedia"],
            path: "Tests/WhizMediaTests"
        ),
    ]
)
