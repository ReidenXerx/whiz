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

    /// The sidebar item the window last showed, so a relaunch lands where
    /// the user left off. Same suite as the options: one injectable domain
    /// for everything the window remembers.
    static let selectionKey = "dev.reidenxerx.whiz.selection"

    func loadSelection() -> SidebarItem? {
        defaults.string(forKey: Self.selectionKey).flatMap(SidebarItem.init(storageValue:))
    }

    func saveSelection(_ item: SidebarItem) {
        defaults.set(item.storageValue, forKey: Self.selectionKey)
    }
}

/// What the sidebar has selected: the new-run form, the recorder, the run in
/// progress (or the one that just failed), or a finished run from history, keyed by its
/// input path — the same identity `LastRunStore` dedupes on.
enum SidebarItem: Hashable, Sendable {
    case compose
    case record
    case live
    case run(String)

    var storageValue: String {
        switch self {
        case .compose: return "compose"
        case .record: return "record"
        case .live: return "live"
        case .run(let path): return "run:" + path
        }
    }

    init?(storageValue: String) {
        switch storageValue {
        case "compose": self = .compose
        case "record": self = .record
        case "live": self = .live
        default:
            guard storageValue.hasPrefix("run:"), storageValue.count > 4 else { return nil }
            self = .run(String(storageValue.dropFirst(4)))
        }
    }
}