import Foundation
import Testing
@testable import WhizUI

/// The setup assistant's model, driven through stub executables: the doctor
/// decides what is done, fixes stream progress, and nothing here touches the
/// developer's real whiz, brew or pipx.
@Suite("Setup assistant")
@MainActor
struct SetupModelTests {

    private func makeDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func script(_ directory: URL, _ name: String, _ body: String) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try ("#!/bin/sh\n" + body).write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    /// A doctor whose model check flips to ok once `marker` exists.
    private func doctorStub(_ directory: URL, extra: String = "") throws -> URL {
        let marker = directory.appendingPathComponent("model-ok").path
        return try script(directory, "whiz", """
        case "$1" in
          doctor)
            if [ -f '\(marker)' ]; then M=true; else M=false; fi
            if [ "$3" = "--ai" ]; then AI='"reachable": true, "models": ["m1"], "model_available": false'; else AI='"configured": false'; fi
            printf '{"version": "1.2.3", "checks": {"whisper_cli": {"ok": true, "path": "/w"}, "ffmpeg": {"ok": true, "path": "/f"}, "model": {"ok": %s, "alias": "turbo", "size_mb": 1500}, "vad_model": {"ok": false}, "speakers": {"ok": false, "package": false}}, "ai": {"configured": false, "base_url": "http://x/v1", "model": "", %s}, "brew": null, "ready": %s}\\n' "$M" "$AI" "$M"
            [ "$M" = true ] || exit 1 ;;
          models)
            echo '▸ downloading speech model' >&2
            echo '~ progress: 50%' >&2
            echo '~ progress: 100%' >&2
            touch '\(marker)' ;;
        \(extra)
        esac
        exit 0
        """)
    }

    private func waitFor(_ condition: () -> Bool) async throws {
        for _ in 0..<200 where !condition() { try await Task.sleep(for: .milliseconds(25)) }
    }

    @Test("the doctor report decodes, AI fields optional")
    func decodeReport() throws {
        let report = try DoctorReport.decode("""
        {"version": "0.14.0", "checks": {"model": {"ok": true, "alias": "turbo", "size_mb": 1550.5, "path": "/m", "fix": "x", "configured": ""}},
         "ai": {"configured": true, "base_url": "http://l/v1", "model": "q"}, "brew": "/opt/homebrew/bin/brew", "ready": false}
        """)
        #expect(report.check("model")?.sizeMb == 1550.5)
        #expect(report.ai.baseUrl == "http://l/v1" && report.ai.reachable == nil)
        #expect(report.brew == "/opt/homebrew/bin/brew")
    }

    @Test("a missing CLI is not ready and points at step one")
    func missingCLI() async {
        let setup = SetupModel(locateWhiz: { nil }, locateTool: { _ in nil })
        await setup.refresh()
        #expect(!setup.isReady)
        #expect(setup.isDone(.cli) == false)
        #expect(setup.firstMissing == .cli)
    }

    @Test("a fix streams progress, then the doctor decides it is done")
    func downloadThenRecheck() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let whiz = try doctorStub(directory)
        let setup = SetupModel(locateWhiz: { whiz }, locateTool: { _ in nil })
        await setup.refresh()
        #expect(setup.report?.version == "1.2.3")
        #expect(setup.isDone(.model) == false)
        #expect(setup.firstMissing == .model)

        setup.downloadModel(SpeechModelChoice.all[0])
        #expect(setup.task?.item == .model)
        try await waitFor { setup.task == nil && setup.isDone(.model) == true }
        #expect(setup.isDone(.model) == true)
        #expect(setup.isReady)
        #expect(setup.failures[.model] == nil)
    }

    @Test("a failing fix reports the CLI's reason and stays not-done")
    func failingFix() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let whiz = try doctorStub(directory, extra: """
          setup) echo 'pip exploded' >&2; exit 1 ;;
        """)
        let setup = SetupModel(locateWhiz: { whiz }, locateTool: { _ in nil })
        await setup.refresh()
        setup.setupSpeakers()
        try await waitFor { setup.task == nil && setup.failures[.speakers] != nil }
        #expect(setup.failures[.speakers]?.contains("pip exploded") == true)
        #expect(setup.isDone(.speakers) == false)
    }

    @Test("installing the CLI installs pipx through brew first when needed")
    func installCLIChain() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let log = directory.appendingPathComponent("log").path
        let pipxPath = directory.appendingPathComponent("pipx")
        // brew "installs" pipx by writing the pipx stub.
        let brew = try script(directory, "brew", """
        echo "brew $*" >> '\(log)'
        printf '#!/bin/sh\\necho "pipx $*" >> \(log)\\n' > '\(pipxPath.path)'
        chmod +x '\(pipxPath.path)'
        """)
        let setup = SetupModel(locateWhiz: { nil }, locateTool: { name in
            switch name {
            case "brew": return brew
            case "pipx": return FileManager.default.isExecutableFile(atPath: pipxPath.path) ? pipxPath : nil
            default: return nil
            }
        })
        setup.installCLI()
        try await waitFor { setup.task == nil }
        let calls = try String(contentsOfFile: log, encoding: .utf8)
        #expect(calls.contains("brew install pipx"))
        #expect(calls.contains("pipx install --force \(SetupModel.installSource)"))
    }

    @Test("a plain re-check keeps the last AI probe for the same server")
    func aiProbeSurvivesRefresh() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let whiz = try doctorStub(directory)
        let setup = SetupModel(locateWhiz: { whiz }, locateTool: { _ in nil })
        await setup.checkAI()
        #expect(setup.report?.ai.reachable == true)
        await setup.refresh()
        #expect(setup.report?.ai.models == ["m1"], "the probe result must not be blanked")
    }

    @Test("every offered speech model resolves to an unquantized download name")
    func modelChoices() {
        #expect(SpeechModelChoice.all.map(\.name) == ["turbo", "large-v3", "small"])
        #expect(SpeechModelChoice.all.allSatisfy { !$0.name.contains("q5") && !$0.name.contains("q8") })
    }

    @Test("voice profiles decode without embeddings")
    func voiceProfiles() throws {
        let voices = try VoiceProfile.decodeList("""
        [{"name": "Ann", "samples": 3, "created": "2026-10-16T15:10:57Z", "source": "user", "path": "/p/Ann.json"},
         {"name": "Bo", "samples": 1, "created": "bad", "source": "auto", "path": "/p/Bo.json"}]
        """)
        #expect(voices.map(\.name) == ["Ann", "Bo"])
        #expect(voices[0].isConfirmed && !voices[1].isConfirmed)
        #expect(voices[0].createdDate != nil && voices[1].createdDate == nil)
    }
}
