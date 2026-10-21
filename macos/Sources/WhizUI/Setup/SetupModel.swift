import AppKit
import Foundation

/// `whiz doctor --json`: what a run needs and whether this Mac has it.
struct DoctorReport: Decodable, Equatable, Sendable {
    struct Check: Decodable, Equatable, Sendable {
        var ok: Bool
        var path: String?
        var fix: String?
        var alias: String?
        var sizeMb: Double?
        var package: Bool?
        /// The speech model only: quantized, and whether the user chose it.
        var quantized: Bool?
        var configured: String?
    }

    struct AI: Decodable, Equatable, Sendable {
        var configured: Bool
        var baseUrl: String
        var model: String
        var reachable: Bool?
        var models: [String]?
        var modelAvailable: Bool?
    }

    var version: String
    var checks: [String: Check]
    var ai: AI
    var brew: String?
    /// whiz is the engine inside Whiz.app: whisper.cpp and the media tool
    /// ship with it. Absent from a whiz older than the bundled engine.
    var bundled: Bool?
    var ready: Bool

    func check(_ key: String) -> Check? { checks[key] }

    static func decode(_ json: String) throws -> DoctorReport {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(DoctorReport.self, from: Data(json.utf8))
    }
}

/// The parts of a working setup, in the order the assistant lists them.
enum SetupItem: String, CaseIterable, Identifiable, Sendable {
    case cli, tools, model, speakers, ai
    var id: String { rawValue }
}

/// A speech model the assistant offers. Names are what `whiz models
/// download` takes; each resolves to an unquantized file (NS-15).
struct SpeechModelChoice: Identifiable, Equatable, Sendable {
    var id: String { name }
    var name: String
    var title: String
    var detail: String
    var size: String

    static let all: [SpeechModelChoice] = [
        .init(name: "turbo", title: "Turbo", detail: "Recommended: fast and accurate in 99 languages", size: "1.6 GB"),
        .init(name: "large-v3", title: "Large v3", detail: "The most accurate, about twice as slow", size: "3.1 GB"),
        .init(name: "small", title: "Small", detail: "The quickest; fine for clear speech", size: "488 MB"),
    ]
}

/// The first-run setup: find what is missing, and fix it from the app.
///
/// Every fix is a command the user could type — `brew install`, `pipx
/// install`, `whiz models download`, `whiz setup speakers` — run through
/// `CLIBackend` so progress and failures arrive the same way a
/// transcription's do. After each one the doctor runs again: the report,
/// not the command's exit code, decides what is done.
@MainActor
final class SetupModel: ObservableObject {
    struct RunningTask: Equatable {
        var item: SetupItem
        var title: String
        var percent: Int?
        var detail: String
    }

    @Published private(set) var report: DoctorReport?
    @Published private(set) var whizExecutable: URL?
    @Published private(set) var isChecking = false
    @Published private(set) var task: RunningTask?
    @Published private(set) var failures: [SetupItem: String] = [:]
    /// The AI server probe, run on demand (it touches the network).
    @Published private(set) var isCheckingAI = false
    /// The assistant sheet; ⌘-menu and the compose banner open it.
    @Published var isPresented = false
    /// The Homebrew installer was started in Terminal; the assistant says
    /// so until brew turns up.
    @Published private(set) var homebrewStarted = false

    static let installSource = "git+https://github.com/ReidenXerx/whiz.git"

    private let locateWhiz: () -> URL?
    private let locateTool: (String) -> URL?
    private let openFile: (URL) -> Void
    private var backend: CLIBackend?
    private var runTask: Task<Void, Never>?

    init(locateWhiz: @escaping () -> URL? = { WhizLocator.find() },
         locateTool: @escaping (String) -> URL? = SetupModel.findTool,
         openFile: @escaping (URL) -> Void = { NSWorkspace.shared.open($0) }) {
        self.locateWhiz = locateWhiz
        self.locateTool = locateTool
        self.openFile = openFile
    }

    /// brew and pipx live where whiz's own tools do; a Finder-launched app's
    /// PATH has none of them.
    nonisolated static func findTool(_ name: String) -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let directories = CLIBackend.toolDirectories(home: home)
            + (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":").map(String.init)
        for directory in directories {
            let url = URL(fileURLWithPath: directory).appendingPathComponent(name)
            if FileManager.default.isExecutableFile(atPath: url.path) { return url }
        }
        return nil
    }

    // MARK: - State

    /// The engine came with the app, so there is nothing to install for it.
    var isBundled: Bool { report?.bundled == true }

    /// The checklist, numbered in this order. Inside Whiz.app "whisper.cpp
    /// and ffmpeg" is not a step: both ship with the engine.
    var visibleItems: [SetupItem] {
        isBundled ? SetupItem.allCases.filter { $0 != .tools } : SetupItem.allCases
    }

    func number(of item: SetupItem) -> Int {
        (visibleItems.firstIndex(of: item) ?? 0) + 1
    }

    var hasBrew: Bool { locateTool("brew") != nil }
    var hasPipx: Bool { locateTool("pipx") != nil }

    /// Whether `item` is done, nil while unknown.
    func isDone(_ item: SetupItem) -> Bool? {
        switch item {
        case .cli:
            return whizExecutable != nil && report != nil
        case .tools:
            guard let report else { return whizExecutable == nil ? false : nil }
            return report.check("whisper_cli")?.ok == true && report.check("ffmpeg")?.ok == true
        case .model:
            return report.map { $0.check("model")?.ok == true }
        case .speakers:
            return report.map { $0.check("speakers")?.ok == true }
        case .ai:
            guard let ai = report?.ai else { return nil }
            if let available = ai.modelAvailable { return ai.configured && available }
            return nil
        }
    }

    /// The three parts a plain transcription cannot run without.
    var isReady: Bool { whizExecutable != nil && report?.ready == true }

    /// NS-15: the speech model in use is quantized only because nothing
    /// better is on disk. It works, so the step stays done, but the
    /// assistant offers the full-quality download. A model the user set in
    /// config is their informed choice and is left alone.
    var hasQuantizedFallback: Bool {
        guard let model = report?.check("model"), model.ok, model.quantized == true else { return false }
        return (model.configured ?? "").isEmpty
    }

    /// The first missing essential, for the "finish setup" banner.
    var firstMissing: SetupItem? {
        for item in [SetupItem.cli, .tools, .model] where isDone(item) == false { return item }
        return nil
    }

    // MARK: - Checking

    func refresh() async {
        isChecking = true
        defer { isChecking = false }
        guard let whiz = locateWhiz() else {
            whizExecutable = nil
            report = nil
            return
        }
        whizExecutable = whiz
        await loadReport(arguments: ["doctor", "--json"], whiz: whiz)
    }

    /// Ask the AI server what it has — only when the user is looking at the
    /// AI step, since it is a network call.
    func checkAI() async {
        guard let whiz = whizExecutable ?? locateWhiz() else { return }
        isCheckingAI = true
        defer { isCheckingAI = false }
        await loadReport(arguments: ["doctor", "--json", "--ai"], whiz: whiz)
    }

    private func loadReport(arguments: [String], whiz: URL) async {
        do {
            // Exit 1 just means "not ready"; the JSON is printed either way.
            let result = try await WhizCommand.run(arguments, executable: whiz)
            if var decoded = try? DoctorReport.decode(result.stdout) {
                // A plain re-check does not probe the AI server; keep the
                // last probe's answer while the server and model are the
                // same, or every fix would blank the AI row.
                if decoded.ai.reachable == nil, let previous = report?.ai, previous.reachable != nil,
                   previous.baseUrl == decoded.ai.baseUrl, previous.model == decoded.ai.model {
                    decoded.ai = previous
                }
                report = decoded
                failures[.cli] = nil
            } else {
                report = nil
                failures[.cli] = result.stderr.contains("invalid choice")
                    ? "This whiz is too old for the setup assistant. Update it below."
                    : "whiz doctor failed: \(result.stderr.split(separator: "\n").last.map(String.init) ?? "no output")"
            }
        } catch {
            failures[.cli] = error.localizedDescription
        }
    }

    // MARK: - Fixing

    /// One command of a fix. The executable is resolved when the step runs,
    /// not when the fix starts: `pipx install` must find the pipx that the
    /// step before it just installed.
    struct Step {
        var title: String
        var resolve: @MainActor () -> (URL, [String])?
    }

    /// Install (or update) the CLI with pipx — installing pipx with brew
    /// first when only brew is there.
    func installCLI() {
        var steps: [Step] = []
        if locateTool("pipx") == nil {
            guard let brew = locateTool("brew") else {
                failures[.cli] = "Install Homebrew first (brew.sh), then try again."
                return
            }
            steps.append(Step(title: "Installing pipx") { (brew, ["install", "pipx"]) })
        }
        steps.append(Step(title: "Installing whiz") { [weak self] in
            self?.locateTool("pipx").map { ($0, ["install", "--force", Self.installSource]) }
        })
        perform(.cli, steps: steps)
    }

    /// `brew install` whichever of whisper-cpp / ffmpeg is missing.
    func installTools() {
        guard let brew = locateTool("brew") else {
            failures[.tools] = "Homebrew is needed to install these. Get it at brew.sh, then check again."
            return
        }
        var formulae: [String] = []
        if report?.check("whisper_cli")?.ok != true { formulae.append("whisper-cpp") }
        if report?.check("ffmpeg")?.ok != true { formulae.append("ffmpeg") }
        guard !formulae.isEmpty else { return }
        perform(.tools, steps: [Step(title: "Installing \(formulae.joined(separator: " and "))") {
            (brew, ["install"] + formulae)
        }])
    }

    func downloadModel(_ choice: SpeechModelChoice) {
        guard let whiz = whizExecutable else { return }
        perform(.model, steps: [Step(title: "Downloading \(choice.title) (\(choice.size))") {
            (whiz, ["models", "download", choice.name])
        }])
    }

    func setupSpeakers() {
        guard let whiz = whizExecutable else { return }
        perform(.speakers, steps: [Step(title: "Setting up speaker detection") {
            (whiz, ["setup", "speakers"])
        }])
    }

    /// Homebrew's official installer, run in Terminal rather than in the
    /// app: it asks for an administrator password and a confirmation, and
    /// someone installing a package manager should see what it does. The
    /// app writes a `.command` file and Terminal opens it; the assistant
    /// checks again whenever the app becomes active.
    func installHomebrew(for item: SetupItem, in directory: URL = FileManager.default.temporaryDirectory) {
        let url = directory.appendingPathComponent("Install Homebrew for Whiz.command")
        do {
            try Self.homebrewScript.write(to: url, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        } catch {
            failures[item] = "Could not start the Homebrew installer: \(error.localizedDescription)"
            return
        }
        failures[item] = nil
        homebrewStarted = true
        openFile(url)
    }

    static let homebrewInstaller = "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"

    static var homebrewScript: String {
        """
        #!/bin/bash
        # Written by Whiz's Setup Assistant: installs Homebrew (https://brew.sh)
        # with its official installer.
        clear
        echo "Installing Homebrew for Whiz. The installer asks for your Mac password"
        echo "(typing shows nothing) and for Return to continue."
        echo
        /bin/bash -c "$(curl -fsSL \(homebrewInstaller))"
        status=$?
        echo
        if [ $status -eq 0 ]; then
          echo "Homebrew is installed. Switch back to Whiz to continue."
        else
          echo "The Homebrew installer stopped (exit $status). See https://brew.sh for help."
        fi

        """
    }

    func cancelTask() {
        backend?.cancel()
        runTask?.cancel()
    }

    /// Run steps in order, streaming progress into `task`; stop at the first
    /// failure; re-check afterwards — the doctor, not an exit code, decides
    /// what is done.
    private func perform(_ item: SetupItem, steps: [Step]) {
        guard task == nil else { return }
        failures[item] = nil
        task = RunningTask(item: item, title: steps.first?.title ?? "Working…", percent: nil, detail: "")
        runTask = Task {
            for step in steps {
                guard let (executable, arguments) = step.resolve() else {
                    failures[item] = "Could not find the tool to run for “\(step.title)”."
                    break
                }
                task?.title = step.title
                task?.percent = nil
                task?.detail = ""
                if !(await runStep(item, executable: executable, arguments: arguments)) { break }
            }
            task = nil
            backend = nil
            await refresh()
        }
    }

    private func runStep(_ item: SetupItem, executable: URL, arguments: [String]) async -> Bool {
        let backend = CLIBackend(executable: executable)
        self.backend = backend
        let (events, continuation) = AsyncStream<TranscriptionEvent>.makeStream()
        let consumer = Task {
            for await event in events { apply(event) }
        }
        var succeeded = true
        do {
            try await backend.run(arguments: arguments) { continuation.yield($0) }
        } catch is CancellationError {
            succeeded = false
            failures[item] = "Stopped."
        } catch let failure as TranscriptionFailure {
            succeeded = false
            failures[item] = failure.errorDescription
        } catch {
            succeeded = false
            failures[item] = error.localizedDescription
        }
        continuation.finish()
        await consumer.value
        return succeeded
    }

    private func apply(_ event: TranscriptionEvent) {
        switch event {
        case .progress(let percent):
            task?.percent = percent
        case .phase(let label):
            task?.detail = PhaseStep(id: 0, label: label, startedAt: .now).title
        case .log(let line):
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty, !trimmed.hasPrefix("~ "), !trimmed.hasPrefix("▸ ") {
                task?.detail = String(trimmed.prefix(160))
            }
        case .artifact, .speakerName:
            break
        }
    }
}
