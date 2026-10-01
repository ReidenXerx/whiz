import Foundation

/// The last finished run, kept across launches and windows.
///
/// The app's windows are stateless by design — each one owns its run from
/// start to finish, and closing it forgets the run. That is honest for the
/// log pane (a transcript belongs to the window that recorded it) but it
/// turns a fresh window into "nothing ever happened" while the artifacts are
/// sitting on disk. This store carries the *artifact list* of the most
/// recent finished run — facts about files, not run history — so a fresh
/// window can offer them.
///
/// Deliberately NOT a run-history database: one record, the newest one, in
/// UserDefaults. The moment the UI wants a real history (multi-run lists,
/// pruning deleted files), this is the wrong home and the change is obvious.
struct LastRunRecord: Codable, Equatable, Sendable {
    /// The input file the run consumed — shown so the user can tell runs
    /// apart without reading artifact paths.
    var inputPath: String
    var finishedAt: Date
    var artifacts: [StoredArtifact]
}

/// The persisted half of `Artifact`. Exists because `Artifact.kind` is a
/// computed property from the URL extension and `Identifiable` derives from
/// the URL — neither needs storing; the URL and label are the facts.
struct StoredArtifact: Codable, Equatable, Sendable, Identifiable {
    var id: URL { url }
    var label: String
    var url: URL
}

/// Reads and writes the single last-run record.
///
/// UserDefaults rather than a file: one small record, no atomicity concerns
/// worth engineering, and it moves with the app's defaults domain. The suite
/// is injectable so tests never touch the real domain (a test writing the
/// real domain leaks into the developer's own app state, and a test reading
/// it would see whatever the last real run left there).
struct LastRunStore {
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    static let key = "dev.reidenxerx.whiz.lastRun"

    func load() -> LastRunRecord? {
        guard let data = defaults.data(forKey: Self.key) else { return nil }
        return try? JSONDecoder().decode(LastRunRecord.self, from: data)
    }

    func save(_ record: LastRunRecord) {
        guard let data = try? JSONEncoder().encode(record) else { return }
        defaults.set(data, forKey: Self.key)
    }

    /// The store only ever holds runs that produced something. A run whose
    /// artifact list is empty (stopped, failed) does not overwrite the
    /// previous run's results — an empty "last run" is worse than an older
    /// real one.
    @discardableResult
    func saveReturning(artifacts: [Artifact], input: URL, finishedAt: Date = .init()) -> LastRunRecord? {
        guard !artifacts.isEmpty else { return nil }
        let record = LastRunRecord(
            inputPath: input.path,
            finishedAt: finishedAt,
            artifacts: artifacts.map { StoredArtifact(label: $0.label, url: $0.url) }
        )
        save(record)
        return record
    }
}
