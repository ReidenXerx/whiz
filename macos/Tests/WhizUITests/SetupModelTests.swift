import Foundation
import Testing
@testable import WhizUI

/// The setup assistant's model, driven through stub executables: the doctor
/// decides what is done, fixes stream progress, and nothing here touches the
/// developer's real whiz, brew or pipx.
@Suite("Setup assistant", .timeLimit(.minutes(1)))
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

    @Test("a quantized fallback model is flagged; a configured one is the user's choice")
    func quantizedFallback() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        func setup(configured: String) async throws -> SetupModel {
            let whiz = try script(directory, "whiz-\(configured.isEmpty ? "auto" : "set")", """
            printf '{"version": "1", "checks": {"whisper_cli": {"ok": true}, "ffmpeg": {"ok": true}, "model": {"ok": true, "alias": "large-v3-turbo-q5_0", "quantized": true, "configured": "\(configured)"}}, "ai": {"configured": false, "base_url": "", "model": ""}, "ready": true}\\n'
            """)
            let model = SetupModel(locateWhiz: { whiz }, locateTool: { _ in nil })
            await model.refresh()
            return model
        }
        let auto = try await setup(configured: "")
        #expect(auto.hasQuantizedFallback && auto.isReady && auto.isDone(.model) == true)
        let chosen = try await setup(configured: "large-v3-turbo-q5_0")
        #expect(!chosen.hasQuantizedFallback)
    }

    @Test("inside Whiz.app the tools are not a step, and the checklist renumbers")
    func bundledEngine() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        func setup(bundled: String) async throws -> SetupModel {
            let whiz = try script(directory, "whiz-\(bundled)", """
            printf '{"version": "1", "checks": {"whisper_cli": {"ok": true}, "ffmpeg": {"ok": true}, "model": {"ok": false}}, "ai": {"configured": false, "base_url": "", "model": ""}, \(bundled)"ready": false}\\n'
            exit 1
            """)
            let model = SetupModel(locateWhiz: { whiz }, locateTool: { _ in nil })
            await model.refresh()
            return model
        }
        let app = try await setup(bundled: "\"bundled\": true, ")
        #expect(app.isBundled)
        #expect(app.visibleItems == [.cli, .model, .speakers, .ai])
        #expect(app.number(of: .model) == 2 && app.number(of: .ai) == 4)
        #expect(app.isDone(.cli) == true && app.isDone(.model) == false)
        // A whiz from before the bundled engine says nothing: the full list.
        let installed = try await setup(bundled: "")
        #expect(!installed.isBundled)
        #expect(installed.visibleItems == SetupItem.allCases)
        #expect(installed.number(of: .model) == 3)
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

    @Test("Install Homebrew writes a runnable installer and hands it to Terminal")
    func installHomebrew() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var opened: [URL] = []
        let setup = SetupModel(locateWhiz: { nil }, locateTool: { _ in nil }, openFile: { opened.append($0) })
        setup.installHomebrew(for: .tools, in: directory)

        #expect(opened.count == 1 && opened[0].pathExtension == "command")
        #expect(FileManager.default.isExecutableFile(atPath: opened[0].path))
        let text = try String(contentsOf: opened[0], encoding: .utf8)
        #expect(text.hasPrefix("#!/bin/bash\n"))
        #expect(text.contains("curl -fsSL \(SetupModel.homebrewInstaller)"))
        #expect(setup.homebrewStarted && setup.failures[.tools] == nil)

        // The script is valid bash: `bash -n` parses without running it.
        // Waited for off the main actor: waitUntilExit spins the calling
        // thread's run loop, and on the main thread that runs other tests'
        // main-actor jobs re-entrantly in the middle of this one.
        let script = opened[0].path
        let status = try await Task.detached { () throws -> Int32 in
            let check = Process()
            check.executableURL = URL(fileURLWithPath: "/bin/bash")
            check.arguments = ["-n", script]
            try check.run()
            check.waitUntilExit()
            return check.terminationStatus
        }.value
        #expect(status == 0)
    }

    @Test("an installer that cannot be written is reported, not opened")
    func installHomebrewUnwritable() {
        var opened = 0
        let setup = SetupModel(locateWhiz: { nil }, locateTool: { _ in nil }, openFile: { _ in opened += 1 })
        setup.installHomebrew(for: .cli, in: URL(fileURLWithPath: "/nonexistent-\(UUID().uuidString)"))
        #expect(opened == 0 && !setup.homebrewStarted)
        #expect(setup.failures[.cli]?.contains("Homebrew") == true)
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
