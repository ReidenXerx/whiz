import Foundation
import Testing
@testable import WhizUI

/// Navigation, history and Settings plumbing — the state the new window is
/// built on.
@Suite("App state")
@MainActor
struct AppStateTests {

    private func makeSuite() -> UserDefaults {
        UserDefaults(suiteName: "test.appstate.\(UUID().uuidString)")!
    }

    private func artifact(_ path: String) -> Artifact {
        Artifact(label: "Wrote labeled SRT", url: URL(fileURLWithPath: path))
    }

    @Test("sidebar selections survive a storage round trip")
    func sidebarStorage() {
        for item in [SidebarItem.compose, .record, .live, .run("/a/b c:d.mov")] {
            #expect(SidebarItem(storageValue: item.storageValue) == item)
        }
        #expect(SidebarItem(storageValue: "run:") == nil)
        #expect(SidebarItem(storageValue: "garbage") == nil)
    }

    @Test("a restored selection must still point at something")
    func restorableSelection() {
        let record = LastRunRecord(inputPath: "/x.mov", finishedAt: .now, artifacts: [])
        #expect(TranscriptionViewModel.restorable(.run("/x.mov"), history: [record]) == .run("/x.mov"))
        #expect(TranscriptionViewModel.restorable(.run("/gone.mov"), history: [record]) == .compose)
        #expect(TranscriptionViewModel.restorable(.live, history: [record]) == .compose,
                "no run survives a relaunch")
        #expect(TranscriptionViewModel.restorable(nil, history: []) == .compose)
        #expect(TranscriptionViewModel.restorable(.record, history: []) == .record)
    }

    @Test("the view model restores the persisted selection")
    func selectionPersists() {
        let history = LastRunStore(defaults: makeSuite())
        let options = OptionsStore(defaults: makeSuite())
        history.saveReturning(artifacts: [artifact("/x.srt")], input: URL(fileURLWithPath: "/x.mov"))
        options.saveSelection(.run("/x.mov"))
        let model = TranscriptionViewModel(lastRunStore: history, optionsStore: options,
                                           makeBackend: { throw TranscriptionFailure.whizNotFound })
        #expect(model.selection == .run("/x.mov"))
        model.selection = .compose
        #expect(options.loadSelection() == .compose)
    }

    @Test("removing a run forgets it and leaves its selection")
    func removeFromHistory() {
        let history = LastRunStore(defaults: makeSuite())
        history.saveReturning(artifacts: [artifact("/a.srt")], input: URL(fileURLWithPath: "/a.mov"))
        history.saveReturning(artifacts: [artifact("/b.srt")], input: URL(fileURLWithPath: "/b.mov"),
                              duration: 12.5)
        #expect(history.loadHistory().first?.duration == 12.5)

        let model = TranscriptionViewModel(lastRunStore: history, optionsStore: OptionsStore(defaults: makeSuite()),
                                           makeBackend: { throw TranscriptionFailure.whizNotFound })
        model.selection = .run("/b.mov")
        model.removeFromHistory(try! #require(model.record(for: "/b.mov")))
        #expect(model.recentRuns.map(\.inputPath) == ["/a.mov"])
        #expect(model.selection == .compose)
        #expect(history.loadHistory().map(\.inputPath) == ["/a.mov"])
    }

    @Test("records written before `duration` existed still decode")
    func legacyRecordDecodes() throws {
        let json = #"[{"inputPath":"/x.mov","finishedAt":0,"artifacts":[]}]"#
        let records = try JSONDecoder().decode([LastRunRecord].self, from: Data(json.utf8))
        #expect(records.first?.duration == nil)
    }

    @Test("time left appears only once the estimate means something")
    func remainingEstimate() {
        #expect(TranscriptionViewModel.estimatedRemaining(percent: nil, elapsed: 60) == nil)
        #expect(TranscriptionViewModel.estimatedRemaining(percent: 1, elapsed: 60) == nil)
        #expect(TranscriptionViewModel.estimatedRemaining(percent: 50, elapsed: 2) == nil)
        #expect(TranscriptionViewModel.estimatedRemaining(percent: 100, elapsed: 60) == nil)
        #expect(TranscriptionViewModel.estimatedRemaining(percent: 25, elapsed: 30) == 90)
    }

    @Test("phase labels read as plain words; unknown ones still show")
    func stepTitles() {
        let step = { (label: String) in PhaseStep(id: 0, label: label, startedAt: .now).title }
        #expect(step("diarizing") == "Telling speakers apart")
        #expect(step("analyzing (chained)") == "Writing the AI summary")
        #expect(step("brand new phase") == "Brand new phase")
    }

    @Test("a run's phases become a timeline that closes with the run")
    func stepsFromARun() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("a.speakers.srt")
        let script = directory.appendingPathComponent("whiz")
        try """
        #!/bin/sh
        echo '▸ transcribing'
        echo '~ progress: 40%'
        echo '▸ merging speakers'
        echo '✓ Wrote labeled SRT: \(output.path)'
        exit 0
        """.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)

        let options = OptionsStore(defaults: makeSuite())
        let model = TranscriptionViewModel(
            lastRunStore: LastRunStore(defaults: makeSuite()), optionsStore: options,
            makeBackend: { CLIBackend(executable: script) })
        let input = URL(fileURLWithPath: "/tmp/a.mp4")
        model.inputURL = input
        model.start()
        #expect(model.selection == .live, "starting a run shows it")
        #expect(model.liveRunInput == input)

        for _ in 0..<200 where model.isRunning { try await Task.sleep(for: .milliseconds(25)) }
        #expect(!model.isRunning)
        #expect(model.steps.map(\.label) == ["transcribing", "merging speakers"])
        #expect(model.steps.allSatisfy { $0.state == .done && $0.endedAt != nil })
        // A clean run with results moves into history and is shown there.
        #expect(model.liveRunInput == nil)
        #expect(model.selection == .run(input.path))
        #expect(model.recentRuns.first?.duration != nil)
    }

    @Test("config JSON keeps bools, ints, doubles, lists and nulls apart")
    func configDecoding() throws {
        let snapshot = try ConfigSnapshot.decode("""
        {"path": "/c.toml", "values": {"vad": true, "threads": 0, "vad_threshold": 0.5,
         "outputs": ["srt", "json"], "model": "", "auto_diarization_setup": null},
         "secrets_set": {"ai_api_key": false}, "model_search_dirs": ["/m"]}
        """)
        #expect(snapshot.values["vad"] == .bool(true))
        #expect(snapshot.values["threads"] == .int(0))
        #expect(snapshot.values["vad_threshold"] == .double(0.5))
        #expect(snapshot.values["outputs"] == .list(["srt", "json"]))
        #expect(snapshot.values["model"] == .string(""))
        #expect(snapshot.values["auto_diarization_setup"] == .null)
        #expect(snapshot.secretsSet?["ai_api_key"] == false)
        #expect(snapshot.modelSearchDirs == ["/m"])
    }

    @Test("values go back to `config set` in the text form its coercion reads")
    func assignmentText() {
        #expect(ConfigValue.bool(false).assignmentText == "false")
        #expect(ConfigValue.list(["srt", "txt"]).assignmentText == "srt,txt")
        #expect(ConfigValue.list([]).assignmentText == "")
        #expect(ConfigValue.double(0.85).assignmentText == "0.85")
        #expect(ConfigValue.int(50).assignmentText == "50")
    }

    @Test("models list JSON decodes")
    func modelsDecoding() throws {
        let models = try WhisperModelInfo.decodeList(
            #"[{"alias": "turbo", "size_mb": 1550.0, "path": "/m/ggml-large-v3-turbo.bin"}]"#)
        #expect(models == [WhisperModelInfo(alias: "turbo", sizeMb: 1550, path: "/m/ggml-large-v3-turbo.bin")])
    }

    @Test("Settings reports a too-old CLI instead of an empty form")
    func settingsWithOldCLI() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let script = directory.appendingPathComponent("whiz")
        try "#!/bin/sh\necho 'whiz: error: unrecognized arguments: --json' >&2\nexit 2\n"
            .write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)

        let settings = SettingsModel(locate: { script })
        await settings.reload()
        guard case .unavailable(let reason) = settings.state else {
            Issue.record("expected unavailable, got \(settings.state)")
            return
        }
        #expect(reason.contains("whiz upgrade"))
        #expect(!settings.isMissingWhiz, "installed but old is not missing")

        let missing = SettingsModel(locate: { nil })
        await missing.reload()
        #expect(missing.isMissingWhiz)
    }

    @Test("Settings reads a real-shaped config and writes through `config set`")
    func settingsRoundTrip() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let log = directory.appendingPathComponent("calls.log")
        let script = directory.appendingPathComponent("whiz")
        try """
        #!/bin/sh
        echo "$@" >> '\(log.path)'
        case "$1 $2" in
          "config show") echo '{"path": "/c.toml", "values": {"vad": true, "ai_model": "m"}, "secrets_set": {"ai_api_key": false}}' ;;
          "models list") echo '[]' ;;
          "--version ") echo 'whiz 9.9.9' ;;
          "config set") if [ "$4" = "--stdin" ]; then read v; echo "stdin=$v" >> '\(log.path)'; fi ;;
        esac
        exit 0
        """.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)

        let settings = SettingsModel(locate: { script })
        await settings.reload()
        #expect(settings.state == .loaded)
        #expect(settings.bool("vad") && settings.string("ai_model") == "m")
        #expect(settings.version == "whiz 9.9.9")

        settings.set("vad", .bool(false))
        #expect(settings.bool("vad") == false, "the form updates immediately")
        settings.setSecret("ai_api_key", "sk-secret")
        for _ in 0..<200 where !settings.saving.isEmpty { try await Task.sleep(for: .milliseconds(25)) }

        let calls = try String(contentsOf: log, encoding: .utf8)
        #expect(calls.contains("config set vad=false"))
        #expect(calls.contains("config set ai_api_key --stdin"))
        #expect(calls.contains("stdin=sk-secret"))
        #expect(!calls.contains("ai_api_key=sk-secret"), "a secret never travels in argv")
    }
}
