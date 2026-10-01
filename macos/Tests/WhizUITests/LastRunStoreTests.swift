import Foundation
import Testing
@testable import WhizUI

/// The persisted last-run record.
///
/// One record in UserDefaults is the whole design: a fresh window has to say
/// "here is what the last run wrote" rather than looking like nothing ever
/// happened. These tests pin the three properties that make that honest —
/// round-tripping through the real encode/decode path, never touching the
/// real defaults domain, and refusing to let an empty run erase a good one.
@Suite("Last run persistence")
struct LastRunStoreTests {

    private func makeSuite() -> UserDefaults {
        // A throwaway suite name isolates each test from the developer's real
        // app state and from every other test — the key is the app's constant,
        // so a shared suite would leak records between tests.
        UserDefaults(suiteName: "test.lastrun.\(UUID().uuidString)")!
    }

    private func makeArtifacts() -> [Artifact] {
        [
            Artifact(label: "Wrote labeled SRT", url: URL(fileURLWithPath: "/tmp/a.speakers.srt")),
            Artifact(label: "Wrote analysis", url: URL(fileURLWithPath: "/tmp/a.analysis.md")),
        ]
    }

    @Test("a saved run loads back identical")
    func roundTrip() {
        let store = LastRunStore(defaults: makeSuite())
        #expect(store.load() == nil, "a fresh suite must read as no last run")

        let input = URL(fileURLWithPath: "/Users/vk/Documents/Talk.mov")
        let finished = Date(timeIntervalSince1970: 1_765_000_000)
        let returned = store.saveReturning(artifacts: makeArtifacts(), input: input, finishedAt: finished)

        #expect(returned != nil)
        #expect(returned == LastRunRecord(
            inputPath: "/Users/vk/Documents/Talk.mov",
            finishedAt: finished,
            artifacts: [
                StoredArtifact(label: "Wrote labeled SRT", url: URL(fileURLWithPath: "/tmp/a.speakers.srt")),
                StoredArtifact(label: "Wrote analysis", url: URL(fileURLWithPath: "/tmp/a.analysis.md")),
            ]))
        #expect(store.load() == returned)
    }

    @Test("an empty run does not erase the previous record")
    func emptyRunKeepsPrevious() {
        // A stopped or failed run writes nothing; the previous run's results
        // must survive it — an empty "last run" is worse than an older real one.
        let store = LastRunStore(defaults: makeSuite())
        let good = store.saveReturning(
            artifacts: makeArtifacts(),
            input: URL(fileURLWithPath: "/tmp/a.mov"),
            finishedAt: Date(timeIntervalSince1970: 1_765_000_000))
        #expect(good != nil)

        #expect(store.saveReturning(
            artifacts: [],
            input: URL(fileURLWithPath: "/tmp/b.mov")) == nil)
        #expect(store.load() == good)
    }

    @Test("a newer run replaces the older one")
    func newerRunReplaces() {
        let store = LastRunStore(defaults: makeSuite())
        let first = store.saveReturning(
            artifacts: makeArtifacts(),
            input: URL(fileURLWithPath: "/tmp/first.mov"),
            finishedAt: Date(timeIntervalSince1970: 1_765_000_000))
        #expect(first != nil)

        let second = store.saveReturning(
            artifacts: makeArtifacts(),
            input: URL(fileURLWithPath: "/tmp/second.mov"),
            finishedAt: Date(timeIntervalSince1970: 1_765_000_100))
        #expect(second != nil)
        #expect(store.load() == second)
        #expect(store.load() != first)
    }

    @Test("a corrupt payload decodes as nothing rather than crashing")
    func corruptPayloadLoadsNil() {
        let defaults = makeSuite()
        defaults.set(Data("not a record".utf8), forKey: LastRunStore.key)
        #expect(LastRunStore(defaults: defaults).load() == nil)
    }
}