import Foundation
import Testing
@testable import WhizUI

/// End-of-run state in the view model, driven through a stub CLI.
@Suite("View model run lifecycle")
@MainActor
struct TranscriptionViewModelTests {

    private func makeSuite() -> UserDefaults {
        UserDefaults(suiteName: "test.viewmodel.\(UUID().uuidString)")!
    }

    /// Polls a main-actor condition; the run's events hop onto the main
    /// actor asynchronously, so state is observed rather than awaited.
    private func waitUntil(_ condition: () -> Bool) async throws -> Bool {
        for _ in 0..<200 {
            if condition() { return true }
            try await Task.sleep(for: .milliseconds(25))
        }
        return condition()
    }

    @Test("a prompt left unanswered is cleared when the run is stopped")
    func stopClearsPendingPrompt() async throws {
        // Left up, the naming bar offers a Submit that writes to a run that
        // no longer exists.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let script = directory.appendingPathComponent("whiz")
        let body = "#!/bin/sh\n"
            + "cat <<'WHIZEOF'\n"
            + "? speaker-name: Speaker A | hello there | Alice\n"
            + "WHIZEOF\n"
            + "read -t 20 name\n"
            + "exit 0\n"
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)

        let model = TranscriptionViewModel(
            lastRunStore: LastRunStore(defaults: makeSuite()),
            optionsStore: OptionsStore(defaults: makeSuite()),
            makeBackend: { CLIBackend(executable: script) })
        model.inputURL = URL(fileURLWithPath: "/tmp/a.mp4")
        model.start()

        #expect(try await waitUntil { model.pendingSpeakerPrompt != nil }, "prompt never arrived")
        model.stop()
        #expect(try await waitUntil { !model.isRunning }, "run never ended")
        #expect(model.pendingSpeakerPrompt == nil)
        #expect(model.lastOutcome == .stopped)
    }
}
