import Foundation

/// The options-row form state, kept across launches.

/// The controls are set once and then never touched — losing them on every
/// relaunch made "options" feel like per-run suggestions rather than
/// preferences. The input file is deliberately NOT persisted: it is the run's
/// subject, not a preference, and a fresh window offering a stale recording
/// would invite transcribing the wrong file.
///
/// UserDefaults with a Codable snapshot, same pattern as `LastRunStore`: one
/// small record, suite injectable so tests never touch the developer's real
/// domain.
struct OptionsRecord: Codable, Equatable, Sendable {
    var language: String = ""
    var speakers: Int = 0
    var speakerNames: String = ""
    var screenshotsRaw: String = "auto"
    var analyze: Bool = false
    var aiModel: String = ""
}

struct OptionsStore {
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    static let key = "dev.reidenxerx.whiz.options"

    func load() -> OptionsRecord {
        guard let data = defaults.data(forKey: Self.key),
              let record = try? JSONDecoder().decode(OptionsRecord.self, from: data)
        else { return OptionsRecord() }
        return record
    }

    func save(_ record: OptionsRecord) {
        guard let data = try? JSONEncoder().encode(record) else { return }
        defaults.set(data, forKey: Self.key)
    }
}