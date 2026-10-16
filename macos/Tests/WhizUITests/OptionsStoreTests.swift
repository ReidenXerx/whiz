import Foundation
import Testing
@testable import WhizUI

/// The options-row form state, persisted across launches.
///
/// The pattern is LastRunStore's: suite injectable so tests never touch the
/// developer's real domain, Codable round-trip through the real encode/decode
/// path, and a corrupt payload reads as defaults rather than crashing.
@Suite("Options persistence")
struct OptionsStoreTests {

    private func makeSuite() -> UserDefaults {
        UserDefaults(suiteName: "test.options.\(UUID().uuidString)")!
    }

    @Test("a saved record loads back identical")
    func roundTrip() {
        let store = OptionsStore(defaults: makeSuite())
        #expect(store.load() == OptionsRecord(), "a fresh suite must read as defaults")

        var record = OptionsRecord()
        record.language = "ru"
        record.speakers = 3
        record.speakerNames = "Alice,Bob"
        record.screenshotsRaw = "Off"
        record.analyze = true
        record.aiModel = "qwen3.5:9b"
        store.save(record)

        #expect(store.load() == record)
    }

    @Test("a corrupt payload decodes as defaults rather than crashing")
    func corruptPayloadLoadsDefaults() {
        let defaults = makeSuite()
        defaults.set(Data("not a record".utf8), forKey: OptionsStore.key)
        #expect(OptionsStore(defaults: defaults).load() == OptionsRecord())
    }

    @Test("the view model restores options at birth and writes them on start")
    @MainActor
    func viewModelRestoresAndSaves() {
        // Arrange: a store holding non-default options.
        let defaults = makeSuite()
        let store = OptionsStore(defaults: defaults)
        var record = OptionsRecord()
        record.language = "ru"
        record.speakers = 3
        record.speakerNames = "Alice,Bob"
        record.screenshotsRaw = "Off"
        record.analyze = true
        record.aiModel = "qwen3.5:9b"
        store.save(record)

        // Act: a model born on that store restores the form...
        let model = TranscriptionViewModel(
            lastRunStore: LastRunStore(defaults: makeSuite()),
            optionsStore: store)
        #expect(model.language == "ru")
        #expect(model.speakers == 3)
        #expect(model.speakerNames == "Alice,Bob")
        #expect(model.screenshots == .off)
        #expect(model.analyze == true)
        #expect(model.aiModel == "qwen3.5:9b")

        // ...and a start() writes the form back for the next launch. A real
        // CLIBackend spawn fails (no whiz in the test world) and start()
        // returns early — but not before the options are saved, so the save
        // is observable exactly when it happens in the real app too.
        model.inputURL = URL(fileURLWithPath: "/tmp/a.mp4")
        model.start()
        #expect(store.load() == record)
    }
}