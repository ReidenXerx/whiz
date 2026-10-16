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

/// The recent-runs list the idle window shows instead of a single card.
///
/// History is a different contract from "the last run": entries accumulate
/// newest-first, a re-run must not leave a stale duplicate advertising
/// artifacts a re-run just replaced, the list is capped, and a user upgrading
/// from the single-record era must not lose the run they had.
@Suite("Run history")
struct RunHistoryTests {

    private func makeSuite() -> UserDefaults {
        UserDefaults(suiteName: "test.history.\(UUID().uuidString)")!
    }

    private func makeArtifacts() -> [Artifact] {
        [
            Artifact(label: "Wrote labeled SRT", url: URL(fileURLWithPath: "/tmp/a.speakers.srt")),
            Artifact(label: "Wrote analysis", url: URL(fileURLWithPath: "/tmp/a.analysis.md")),
        ]
    }

    @Test("runs accumulate newest-first")
    func accumulateNewestFirst() {
        let store = LastRunStore(defaults: makeSuite())
        let first = store.saveReturning(
            artifacts: makeArtifacts(),
            input: URL(fileURLWithPath: "/tmp/first.mov"),
            finishedAt: Date(timeIntervalSince1970: 1_765_000_000))
        let second = store.saveReturning(
            artifacts: makeArtifacts(),
            input: URL(fileURLWithPath: "/tmp/second.mov"),
            finishedAt: Date(timeIntervalSince1970: 1_765_000_100))
        #expect(first != nil)
        #expect(second != nil)

        let history = store.loadHistory()
        #expect(history.count == 2)
        #expect(history.first == second)
        #expect(history.last == first)
        #expect(store.load() == second, "load() stays the newest record")
    }

    @Test("a re-run of the same input replaces the older entry")
    func rerunReplacesOlderEntry() {
        // The artifacts of the newest run are the ones that exist on disk;
        // keeping the older entry would advertise files a re-run just
        // replaced.
        let store = LastRunStore(defaults: makeSuite())
        let old = store.saveReturning(
            artifacts: makeArtifacts(),
            input: URL(fileURLWithPath: "/tmp/same.mov"),
            finishedAt: Date(timeIntervalSince1970: 1_765_000_000))
        #expect(old != nil)
        let other = store.saveReturning(
            artifacts: makeArtifacts(),
            input: URL(fileURLWithPath: "/tmp/other.mov"),
            finishedAt: Date(timeIntervalSince1970: 1_765_000_050))
        #expect(other != nil)
        let rerun = store.saveReturning(
            artifacts: makeArtifacts(),
            input: URL(fileURLWithPath: "/tmp/same.mov"),
            finishedAt: Date(timeIntervalSince1970: 1_765_000_100))
        #expect(rerun != nil)

        let history = store.loadHistory()
        #expect(history.count == 2, "re-running an input must not duplicate it")
        #expect(history.map(\.inputPath) == ["/tmp/same.mov", "/tmp/other.mov"])
        #expect(history.first == rerun)
    }

    @Test("the history is capped at 20 entries")
    func cappedAtTwenty() {
        let store = LastRunStore(defaults: makeSuite())
        for i in 0..<30 {
            let saved = store.saveReturning(
                artifacts: makeArtifacts(),
                input: URL(fileURLWithPath: "/tmp/run\(i).mov"),
                finishedAt: Date(timeIntervalSince1970: 1_765_000_000 + Double(i)))
            #expect(saved != nil)
        }
        let history = store.loadHistory()
        #expect(history.count == LastRunStore.historyCap)
        #expect(history.first?.inputPath == "/tmp/run29.mov", "the newest survives")
        #expect(!history.contains { $0.inputPath == "/tmp/run0.mov" }, "the oldest fell off")
    }

    @Test("a legacy single-run record migrates into the history")
    func legacyRecordMigrates() {
        // The pre-history app stored one record under `key`. Upgrading must
        // not lose that run.
        let defaults = makeSuite()
        let legacy = LastRunRecord(
            inputPath: "/tmp/legacy.mov",
            finishedAt: Date(timeIntervalSince1970: 1_765_000_000),
            artifacts: [
                StoredArtifact(label: "Wrote labeled SRT", url: URL(fileURLWithPath: "/tmp/legacy.speakers.srt")),
            ])
        guard let data = try? JSONEncoder().encode(legacy) else {
            Issue.record("could not encode the legacy record"); return
        }
        defaults.set(data, forKey: LastRunStore.key)

        let store = LastRunStore(defaults: defaults)
        let history = store.loadHistory()
        #expect(history == [legacy])
        #expect(store.load() == legacy)
        // And it persisted — a second load reads the written history.
        #expect(LastRunStore(defaults: defaults).loadHistory() == [legacy])
    }
}
