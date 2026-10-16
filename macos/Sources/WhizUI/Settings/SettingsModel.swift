import Foundation

/// What a short `whiz` command printed.
struct WhizCommandResult: Sendable, Equatable {
    var status: Int32
    var stdout: String
    var stderr: String
}

/// Runs one short `whiz` command to completion (config, models, version).
///
/// Separate from `CLIBackend` on purpose: that one drives a long run over a
/// live pipe; these are request/response calls whose whole output is the
/// answer. Same environment repair, minus the speaker-prompt opt-in — none
/// of these commands asks anything.
enum WhizCommand {
    static func run(_ arguments: [String], stdin: String? = nil,
                    executable: URL) async throws -> WhizCommandResult {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Result {
                    try runBlocking(arguments, stdin: stdin, executable: executable)
                })
            }
        }
    }

    /// The synchronous body. It blocks until the child exits, so `run`
    /// calls it on a GCD thread: a detached Task would still block one of
    /// the cooperative pool's few threads (as many as there are cores), and
    /// a handful of these at once can starve every other task.
    private static func runBlocking(_ arguments: [String], stdin: String?,
                                    executable: URL) throws -> WhizCommandResult {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        var environment = CLIBackend.childEnvironment(
            from: ProcessInfo.processInfo.environment, home: NSHomeDirectory())
        environment.removeValue(forKey: "WHIZ_UI_PIPE")
        process.environment = environment

        let out = Pipe(), err = Pipe(), input = Pipe()
        process.standardOutput = out
        process.standardError = err
        process.standardInput = input
        try process.run()
        if let stdin {
            try? input.fileHandleForWriting.write(contentsOf: Data(stdin.utf8))
        }
        try? input.fileHandleForWriting.close()

        // Both pipes drain concurrently: a child that fills stderr while
        // we block on stdout would otherwise deadlock.
        let errData = DataBox()
        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            errData.data = err.fileHandleForReading.readDataToEndOfFile()
            group.leave()
        }
        let outData = out.fileHandleForReading.readDataToEndOfFile()
        group.wait()
        process.waitUntilExit()
        return WhizCommandResult(
            status: process.terminationStatus,
            stdout: String(decoding: outData, as: UTF8.self),
            stderr: String(decoding: errData.data, as: UTF8.self))
    }

    private final class DataBox: @unchecked Sendable {
        var data = Data()
    }
}

/// One value of whiz's config, as `config show --json` types it.
enum ConfigValue: Equatable, Sendable, Decodable {
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case list([String])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        // Order matters: JSON `true` must not read as 1, and `50` must stay
        // an Int (ai_max_frames) while `0.5` is a Double.
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Int.self) { self = .int(value) }
        else if let value = try? container.decode(Double.self) { self = .double(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([String].self) { self = .list(value) }
        else { self = .null }
    }

    /// The text `whiz config set KEY=VALUE` parses back into this value.
    var assignmentText: String {
        switch self {
        case .bool(let value): return value ? "true" : "false"
        case .int(let value): return String(value)
        case .double(let value): return String(value)
        case .string(let value): return value
        case .list(let value): return value.joined(separator: ",")
        case .null: return ""
        }
    }
}

/// `whiz config show --json`.
struct ConfigSnapshot: Equatable, Sendable, Decodable {
    var path: String
    var values: [String: ConfigValue]
    var secretsSet: [String: Bool]?
    var modelSearchDirs: [String]?

    static func decode(_ json: String) throws -> ConfigSnapshot {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ConfigSnapshot.self, from: Data(json.utf8))
    }
}

/// One entry of `whiz models list --json`.
struct WhisperModelInfo: Equatable, Sendable, Decodable, Identifiable {
    var id: String { alias }
    var alias: String
    var sizeMb: Double
    var path: String

    static func decodeList(_ json: String) throws -> [WhisperModelInfo] {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([WhisperModelInfo].self, from: Data(json.utf8))
    }
}

/// One entry of `whiz speakers list --json` — names and provenance only;
/// the CLI never hands out the voice embeddings.
struct VoiceProfile: Equatable, Sendable, Decodable, Identifiable {
    var id: String { name }
    var name: String
    var samples: Int
    var created: String
    var source: String
    var path: String

    static func decodeList(_ json: String) throws -> [VoiceProfile] {
        try JSONDecoder().decode([VoiceProfile].self, from: Data(json.utf8))
    }

    /// "named by you" vs adopted from an automatic match.
    var isConfirmed: Bool { source == "user" }

    var createdDate: Date? {
        let formatter = ISO8601DateFormatter()
        return formatter.date(from: created)
    }
}

/// The Settings window's state: whiz's own config, read and written through
/// the CLI.
///
/// The app keeps no settings copy of its own. `~/.config/whiz/config.toml`
/// is the one source of truth — the terminal and the app see the same
/// values, and the CLI's own validation (`config set`) guards every write.
@MainActor
final class SettingsModel: ObservableObject {
    enum LoadState: Equatable {
        case idle
        case loading
        case loaded
        /// The CLI could not be asked: missing, or too old for `--json`.
        case unavailable(String)
    }

    @Published private(set) var state: LoadState = .idle
    @Published private(set) var snapshot: ConfigSnapshot?
    @Published private(set) var models: [WhisperModelInfo] = []
    @Published private(set) var version: String?
    @Published private(set) var executable: URL?
    /// The last write the CLI rejected, shown under the form.
    @Published var lastError: String?
    /// Keys with a write in flight, for a per-row spinner.
    @Published private(set) var saving: Set<String> = []
    /// Stored voice profiles; nil until loaded.
    @Published private(set) var voices: [VoiceProfile]?

    private let locate: () -> URL?

    init(locate: @escaping () -> URL? = { WhizLocator.find() }) {
        self.locate = locate
    }

    func reload() async {
        guard let executable = locate() else {
            self.executable = nil
            state = .unavailable("The whiz command was not found. Install it with `pipx install git+https://github.com/ReidenXerx/whiz.git`.")
            return
        }
        self.executable = executable
        if snapshot == nil { state = .loading }
        do {
            let config = try await WhizCommand.run(["config", "show", "--json"], executable: executable)
            guard config.status == 0 else {
                state = .unavailable(
                    config.stderr.contains("--json")
                        ? "This version of whiz is too old for the Settings window. Run `whiz upgrade`, then reopen Settings."
                        : Self.firstLine(config.stderr, fallback: "whiz config show failed (exit \(config.status))."))
                return
            }
            snapshot = try ConfigSnapshot.decode(config.stdout)
            let list = try await WhizCommand.run(["models", "list", "--json"], executable: executable)
            models = list.status == 0 ? ((try? WhisperModelInfo.decodeList(list.stdout)) ?? []) : []
            let versionResult = try await WhizCommand.run(["--version"], executable: executable)
            version = versionResult.status == 0
                ? versionResult.stdout.trimmingCharacters(in: .whitespacesAndNewlines) : nil
            state = .loaded
        } catch {
            state = .unavailable("Could not run whiz: \(error.localizedDescription)")
        }
    }

    // MARK: - Voices

    func loadVoices() async {
        guard let executable = executable ?? locate() else { return }
        do {
            let result = try await WhizCommand.run(["speakers", "list", "--json"], executable: executable)
            voices = result.status == 0 ? ((try? VoiceProfile.decodeList(result.stdout)) ?? []) : []
            if result.status != 0 {
                lastError = Self.firstLine(result.stderr, fallback: "Could not list voice profiles.")
            }
        } catch {
            lastError = error.localizedDescription
        }
    }

    /// Delete a profile: that voice is a stranger again on the next run.
    func forgetVoice(_ name: String) async {
        guard let executable = executable ?? locate() else { return }
        do {
            let result = try await WhizCommand.run(["speakers", "forget", name], executable: executable)
            if result.status != 0 {
                lastError = Self.firstLine(result.stderr + result.stdout, fallback: "Could not forget \(name).")
            }
        } catch {
            lastError = error.localizedDescription
        }
        await loadVoices()
    }

    // MARK: - Reading

    /// The CLI is not installed (as opposed to installed but failing).
    var isMissingWhiz: Bool {
        if case .unavailable = state, executable == nil { return true }
        return false
    }

    func value(_ key: String) -> ConfigValue? { snapshot?.values[key] }

    func string(_ key: String) -> String {
        if case .string(let value) = value(key) { return value }
        return ""
    }

    func bool(_ key: String) -> Bool {
        if case .bool(let value) = value(key) { return value }
        return false
    }

    func double(_ key: String) -> Double {
        switch value(key) {
        case .double(let value): return value
        case .int(let value): return Double(value)
        default: return 0
        }
    }

    func int(_ key: String) -> Int {
        switch value(key) {
        case .int(let value): return value
        case .double(let value): return Int(value)
        default: return 0
        }
    }

    func list(_ key: String) -> [String] {
        if case .list(let value) = value(key) { return value }
        return []
    }

    func secretIsSet(_ key: String) -> Bool { snapshot?.secretsSet?[key] ?? false }

    // MARK: - Writing

    /// Write one key through `whiz config set`. The form shows the new value
    /// immediately; a rejected write puts the CLI's reason under the form
    /// and re-reads the config so the control snaps back to the truth.
    func set(_ key: String, _ value: ConfigValue) {
        guard let executable, snapshot?.values[key] != value else { return }
        snapshot?.values[key] = value
        saving.insert(key)
        Task {
            defer { saving.remove(key) }
            do {
                let result = try await WhizCommand.run(
                    ["config", "set", "\(key)=\(value.assignmentText)"], executable: executable)
                if result.status != 0 {
                    lastError = Self.firstLine(result.stderr, fallback: "Could not save \(key).")
                    await reload()
                } else {
                    lastError = nil
                }
            } catch {
                lastError = error.localizedDescription
            }
        }
    }

    /// Secrets go through stdin (`config set KEY --stdin`), never argv,
    /// where any local process can read them.
    func setSecret(_ key: String, _ secret: String) {
        guard let executable else { return }
        saving.insert(key)
        Task {
            defer { saving.remove(key) }
            do {
                let result = try await WhizCommand.run(
                    ["config", "set", key, "--stdin"], stdin: secret + "\n", executable: executable)
                if result.status != 0 {
                    lastError = Self.firstLine(result.stderr, fallback: "Could not save \(key).")
                } else {
                    lastError = nil
                }
                await reload()
            } catch {
                lastError = error.localizedDescription
            }
        }
    }

    private static func firstLine(_ text: String, fallback: String) -> String {
        let line = text.split(separator: "\n").map {
            $0.trimmingCharacters(in: .whitespaces)
        }.last { !$0.isEmpty && !$0.hasPrefix("usage:") }
        return line ?? fallback
    }
}
