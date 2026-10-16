import Foundation

/// The last finished run, kept across launches and windows.
///
/// The app's windows are stateless by design — each one owns its run from
/// start to finish, and closing it forgets the run. That is honest for the
/// log pane (a transcript belongs to the window that recorded it) but it
/// turns a fresh window into "nothing ever happened" while the artifacts are
/// sitting on disk. This store carries the *artifact list* of the most
/// recent finished runs — facts about files, not run history — so a fresh
/// window can offer them.
///
/// A capped list (newest first) in UserDefaults rather than a database:
/// twenty records is plenty for "recent", and the moment a real history
/// feature lands (search, pruning deleted files) this is the wrong home and
/// the change is obvious.
struct LastRunRecord: Codable, Equatable, Sendable, Identifiable {
    /// Identity is the input path — the same grain the history dedupes on,
    /// so a re-run replaces the row rather than stacking a twin beside it.
    var id: String { inputPath }

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

/// Reads and writes the recent-run list.
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
    static let historyKey = "dev.reidenxerx.whiz.runHistory"
    /// How many runs the history keeps. Enough for "recent", small enough
    /// that UserDefaults stays a reasonable home.
    static let historyCap = 20

    /// The newest record, for the one card a fresh window used to show.
    func load() -> LastRunRecord? {
        loadHistory().first
    }

    /// The recent-run list, newest first. An empty list when nothing has
    /// ever been recorded.
    func loadHistory() -> [LastRunRecord] {
        guard let data = defaults.data(forKey: Self.historyKey),
              let records = try? JSONDecoder().decode([LastRunRecord].self, from: data)
        else { return migrateLegacy() }
        return records
    }

    /// The pre-history app stored one record under `key`. A user upgrading
    /// must not lose that run: it becomes the single entry of the new list.
    private func migrateLegacy() -> [LastRunRecord] {
        guard let data = defaults.data(forKey: Self.key),
              let record = try? JSONDecoder().decode(LastRunRecord.self, from: data)
        else { return [] }
        saveHistory([record])
        return [record]
    }

    func save(_ record: LastRunRecord) {
        prepend(toHistory: record)
    }

    func saveHistory(_ records: [LastRunRecord]) {
        guard let data = try? JSONEncoder().encode(records) else { return }
        defaults.set(data, forKey: Self.historyKey)
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
        prepend(toHistory: record)
        return record
    }

    /// Newest first, deduplicated by input path (a re-run of the same file
    /// replaces its older entry — the newest artifacts are the ones that
    /// exist), capped.
    private func prepend(toHistory record: LastRunRecord) {
        var history = loadHistory().filter { $0.inputPath != record.inputPath }
        history.insert(record, at: 0)
        saveHistory(Array(history.prefix(Self.historyCap)))
    }
}
