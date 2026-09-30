import Foundation

/// Finds the `whiz` executable.
///
/// A GUI app does not inherit the shell's PATH — a macOS app launched from
/// Finder or Dock gets a minimal environment, so the `~/.local/bin` that pipx
/// installs into is invisible even though `whiz` works perfectly in Terminal.
/// That mismatch is the single most likely "it works for me" bug in this
/// layer, so discovery is explicit rather than trusting PATH.
enum WhizLocator {

    /// Where pipx, Homebrew and uv put console scripts, in the order a user
    /// would expect them to win.
    static var searchPaths: [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return [
            home.appendingPathComponent(".local/bin/whiz"),          // pipx
            URL(fileURLWithPath: "/opt/homebrew/bin/whiz"),          // Homebrew, Apple silicon
            URL(fileURLWithPath: "/usr/local/bin/whiz"),             // Homebrew, Intel
            home.appendingPathComponent(".cargo/bin/whiz"),
        ]
    }

    /// The first executable whiz on disk, or nil.
    ///
    /// An explicit override wins: `WHIZ_EXECUTABLE` lets a developer point the
    /// app at a checkout without installing anything.
    /// `candidates` is injectable for the same reason `resolve(searchDirs:)`
    /// is on the Python side: without it a test cannot tell "found the fake in
    /// PATH" from "found the real whiz in the developer's ~/.local/bin", and
    /// the host silently decides the result.
    static func find(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        candidates: [URL] = searchPaths
    ) -> URL? {
        if let override = environment["WHIZ_EXECUTABLE"], !override.isEmpty {
            let url = URL(fileURLWithPath: (override as NSString).expandingTildeInPath)
            return isExecutable(url) ? url : nil
        }
        if let found = candidates.first(where: isExecutable) { return found }
        // PATH last rather than first: when the app *is* launched from a shell
        // it may be richer than the list above, but it is the unreliable case.
        for directory in (environment["PATH"] ?? "").split(separator: ":") {
            let candidate = URL(fileURLWithPath: String(directory))
                .appendingPathComponent("whiz")
            if isExecutable(candidate) { return candidate }
        }
        return nil
    }

    static func isExecutable(_ url: URL) -> Bool {
        FileManager.default.isExecutableFile(atPath: url.path)
    }
}
