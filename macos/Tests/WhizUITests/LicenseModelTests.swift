import CryptoKit
import Foundation
import Testing
import WhizLicense
@testable import WhizUI

/// The trial and the license, against a test signing key and a throwaway
/// defaults suite — never the real key or the developer's own trial count.
@Suite("License and trial")
@MainActor
struct LicenseModelTests {
    let signer = Curve25519.Signing.PrivateKey()

    private func makeSuite() -> UserDefaults {
        UserDefaults(suiteName: "test.license.\(UUID().uuidString)")!
    }

    private func makeModel(_ defaults: UserDefaults, appMajor: Int = 0) -> LicenseModel {
        LicenseModel(defaults: defaults, publicKey: signer.publicKey, appMajor: appMajor)
    }

    private func key(major: Int = 1) throws -> String {
        try LicenseKey.issue(License(id: "T-1", name: "Grace Hopper", issued: "2026-10-21", major: major),
                             signingKey: signer)
    }

    @Test("the trial counts finished transcriptions, persists, and ends")
    func trialRunsOut() {
        let defaults = makeSuite()
        let model = makeModel(defaults)
        #expect(model.status == .trial(remaining: LicenseModel.freeTranscriptions))
        for _ in 0..<(LicenseModel.freeTranscriptions - 1) { model.recordTranscription() }
        #expect(model.status == .trial(remaining: 1))
        #expect(makeModel(defaults).status == .trial(remaining: 1), "a relaunch keeps the count")
        model.recordTranscription()
        #expect(model.status == .trialOver)
        #expect(!model.canTranscribe)

        // Transcribe past the trial explains itself instead of doing nothing.
        #expect(model.requestTranscription() == false)
        #expect(model.isActivationPresented && model.openedAtTrialEnd)
        model.presentActivation()
        #expect(!model.openedAtTrialEnd)
    }

    @Test("a valid key licenses the app, stops the count, survives a relaunch, and can be removed")
    func activation() throws {
        let defaults = makeSuite()
        let model = makeModel(defaults)
        model.recordTranscription()
        let license = try model.activate("  " + key() + "\n")
        #expect(license.name == "Grace Hopper")
        #expect(model.status == .licensed(license))
        model.recordTranscription()
        #expect(model.trialUsed == 1, "licensed runs are not counted")
        #expect(makeModel(defaults).isLicensed)

        model.deactivate()
        #expect(model.status == .trial(remaining: LicenseModel.freeTranscriptions - 1))
        #expect(!makeModel(defaults).isLicensed)
    }

    @Test("a rejected key changes nothing")
    func rejectedKey() throws {
        let defaults = makeSuite()
        let model = makeModel(defaults)
        let foreign = try LicenseKey.issue(License(id: "X", name: "X", issued: "2026-10-21", major: 1),
                                           signingKey: Curve25519.Signing.PrivateKey())
        #expect(throws: LicenseError.invalidSignature) { try model.activate(foreign) }
        #expect(throws: LicenseError.malformed) { try model.activate("not a key") }
        #expect(!model.isLicensed)
        #expect(defaults.string(forKey: LicenseModel.licenseDefault) == nil)
    }

    @Test("a stored key for an older major version licenses nothing, but is kept")
    func olderMajor() throws {
        let defaults = makeSuite()
        try makeModel(defaults, appMajor: 1).activate(key(major: 1))
        #expect(!makeModel(defaults, appMajor: 2).isLicensed)
        #expect(defaults.string(forKey: LicenseModel.licenseDefault) != nil)
        #expect(makeModel(defaults, appMajor: 1).isLicensed, "going back brings it back")
    }

    // MARK: - The run gate

    private func stubWhiz(exitStatus: Int32, artifact: Bool) throws -> (URL, URL) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let output = directory.appendingPathComponent("talk.speakers.txt")
        try Data("Speaker A: hi\n".utf8).write(to: output)
        let script = directory.appendingPathComponent("whiz")
        let body = "#!/bin/sh\n"
            + (artifact ? "echo '✓ Wrote dialogue TXT: \(output.path)' >&2\n" : "echo 'whiz: failed' >&2\n")
            + "exit \(exitStatus)\n"
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        return (directory, script)
    }

    private func waitUntil(_ condition: () -> Bool) async throws -> Bool {
        for _ in 0..<200 {
            if condition() { return true }
            try await Task.sleep(for: .milliseconds(25))
        }
        return condition()
    }

    @Test("a finished run with results uses one free transcription; a failed one does not")
    func runsCount() async throws {
        let license = makeModel(makeSuite())
        for (status, artifact, expected) in [(Int32(1), false, 0), (Int32(0), true, 1)] {
            let (directory, script) = try stubWhiz(exitStatus: status, artifact: artifact)
            defer { try? FileManager.default.removeItem(at: directory) }
            let model = TranscriptionViewModel(
                lastRunStore: LastRunStore(defaults: makeSuite()),
                optionsStore: OptionsStore(defaults: makeSuite()),
                license: license,
                makeBackend: { CLIBackend(executable: script) })
            model.inputURL = directory.appendingPathComponent("talk.mov")
            model.start()
            #expect(try await waitUntil { model.lastOutcome != nil && model.lastOutcome != .running })
            #expect(license.trialUsed == expected)
        }
    }

    @Test("past the trial, Transcribe opens the license sheet and starts nothing")
    func gateBlocks() throws {
        let license = makeModel(makeSuite())
        for _ in 0..<LicenseModel.freeTranscriptions { license.recordTranscription() }
        var built = false
        let model = TranscriptionViewModel(
            lastRunStore: LastRunStore(defaults: makeSuite()),
            optionsStore: OptionsStore(defaults: makeSuite()),
            license: license,
            makeBackend: { built = true; return CLIBackend(executable: URL(fileURLWithPath: "/bin/false")) })
        model.inputURL = URL(fileURLWithPath: "/tmp/talk.mov")
        model.start()
        #expect(!model.isRunning && !built)
        #expect(license.isActivationPresented && license.openedAtTrialEnd)
    }
}
