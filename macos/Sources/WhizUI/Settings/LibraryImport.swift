import Foundation

/// Hands the app's run history to whiz's library once, so calls transcribed
/// before the library existed are there for agents (`whiz mcp`) too. Runs
/// since then add themselves as they finish.
enum LibraryImport {
    static let doneKey = "dev.reidenxerx.whiz.libraryImported"

    static func runOnce(history: [LastRunRecord], executable: URL?,
                        defaults: UserDefaults = .standard) async {
        guard !defaults.bool(forKey: doneKey), let executable else { return }
        let inputs = history.map(\.inputPath).filter { FileManager.default.fileExists(atPath: $0) }
        if !inputs.isEmpty {
            guard let result = try? await WhizCommand.run(["library", "add"] + inputs, executable: executable)
            else { return }
            // 1 is "some had no transcript beside them": done all the same.
            // Anything else (an engine too old for `library`) tries again
            // next launch.
            guard result.status == 0 || result.status == 1 else { return }
        }
        defaults.set(true, forKey: doneKey)
    }
}
