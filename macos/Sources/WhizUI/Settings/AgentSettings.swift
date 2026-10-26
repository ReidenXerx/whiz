import AppKit
import SwiftUI

/// An MCP client whose config file Whiz can add itself to.
///
/// Both keep servers under `mcpServers` in a JSON file; Whiz adds or
/// replaces only its own `whiz` entry and leaves everything else in the
/// file as it was.
enum AgentClient: String, CaseIterable, Identifiable, Sendable {
    case claudeDesktop
    case cursor

    var id: String { rawValue }

    var name: String {
        switch self {
        case .claudeDesktop: return "Claude Desktop"
        case .cursor: return "Cursor"
        }
    }

    /// Where the client keeps its MCP servers.
    func configURL(home: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
        switch self {
        case .claudeDesktop:
            return home.appendingPathComponent("Library/Application Support/Claude/claude_desktop_config.json")
        case .cursor:
            return home.appendingPathComponent(".cursor/mcp.json")
        }
    }

    /// The client is installed (its settings folder exists).
    func isPresent(home: URL = FileManager.default.homeDirectoryForCurrentUser) -> Bool {
        FileManager.default.fileExists(atPath: configURL(home: home).deletingLastPathComponent().path)
    }
}

/// Reads and writes the `whiz` entry of an MCP client's config.
enum MCPConfig {
    static let serverName = "whiz"

    struct Unreadable: LocalizedError {
        var file: URL
        var errorDescription: String? {
            "\(file.lastPathComponent) is not valid JSON, so Whiz left it alone. Fix it, or add Whiz by hand."
        }
    }

    /// The entry Whiz writes: the app's own engine, serving MCP.
    static func entry(executable: URL) -> [String: Any] {
        ["command": executable.path, "args": ["mcp"]]
    }

    /// Is Whiz's entry there and pointing at this executable?
    static func isInstalled(in file: URL, executable: URL) -> Bool {
        guard let root = try? read(file),
              let servers = root["mcpServers"] as? [String: Any],
              let whiz = servers[serverName] as? [String: Any]
        else { return false }
        return whiz["command"] as? String == executable.path && whiz["args"] as? [String] == ["mcp"]
    }

    /// Add (or repoint) Whiz's entry. The file's other content is kept; a
    /// file that is not valid JSON is never overwritten.
    static func install(in file: URL, executable: URL) throws {
        var root = try read(file) ?? [:]
        var servers = root["mcpServers"] as? [String: Any] ?? [:]
        servers[serverName] = entry(executable: executable)
        root["mcpServers"] = servers
        let data = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: file, options: .atomic)
    }

    /// Remove Whiz's entry, keeping the rest.
    static func remove(from file: URL) throws {
        guard var root = try read(file), var servers = root["mcpServers"] as? [String: Any] else { return }
        servers.removeValue(forKey: serverName)
        root["mcpServers"] = servers
        let data = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: file, options: .atomic)
    }

    /// nil when there is no file yet (or it is empty).
    private static func read(_ file: URL) throws -> [String: Any]? {
        guard let data = try? Data(contentsOf: file),
              !data.allSatisfy({ $0 == 0x20 || $0 == 0x0A || $0 == 0x0D || $0 == 0x09 })
        else { return nil }
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Unreadable(file: file)
        }
        return root
    }

    /// The command for Claude Code, which keeps its servers itself.
    static func claudeCodeCommand(executable: URL) -> String {
        "claude mcp add --scope user \(serverName) -- \(shellQuoted(executable.path)) mcp"
    }

    static func shellQuoted(_ text: String) -> String {
        text.allSatisfy { $0.isLetter || $0.isNumber || "/._-".contains($0) }
            ? text : "'" + text.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}

/// Settings → Agents: past calls for Claude, Cursor and other MCP clients.
struct AgentSettings: View {
    @State private var installed: Set<AgentClient> = []
    @State private var message: String?
    @State private var failure: String?
    @State private var copied = false

    private var executable: URL? { WhizLocator.find() }

    var body: some View {
        Form {
            Section {
                Text("AI agents can read your transcribed calls through Whiz: list them, search what was said, read a transcript or summary, and see the screen at a moment. Whiz serves them from this Mac over MCP; nothing is uploaded until an agent reads it.")
                    .fixedSize(horizontal: false, vertical: true)
                Label("What an agent reads goes to its AI provider, as anything else you share with it. Claude Desktop asks before each use unless you allow it.",
                      systemImage: "hand.raised")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let executable {
                Section("Connect") {
                    ForEach(AgentClient.allCases) { client in
                        clientRow(client, executable: executable)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Claude Code")
                            Spacer()
                            Button(copied ? "Copied" : "Copy Command") {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(MCPConfig.claudeCodeCommand(executable: executable), forType: .string)
                                copied = true
                            }
                        }
                        Text("Paste it in Terminal once.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Section {
                    LabeledContent("Command") {
                        Text("\(executable.path) mcp")
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .lineLimit(2)
                    }
                } header: {
                    Text("Other MCP clients")
                } footer: {
                    Text(executable.path.hasPrefix("/Applications/")
                         ? "Clients start this command. If you move Whiz.app, connect them again."
                         : "Move Whiz.app to Applications first: clients start this exact path.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            } else {
                Section {
                    Text("The whiz engine was not found, so there is nothing to connect yet.")
                        .foregroundStyle(.secondary)
                }
            }
            if let message {
                Label(message, systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            }
            if let failure {
                Label(failure, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red)
            }
        }
        .formStyle(.grouped)
        .onAppear(perform: refresh)
    }

    private func clientRow(_ client: AgentClient, executable: URL) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(client.name)
                if !client.isPresent() {
                    Text("Not installed").font(.caption).foregroundStyle(.secondary)
                } else if installed.contains(client) {
                    Text("Connected").font(.caption).foregroundStyle(.green)
                }
            }
            Spacer()
            if installed.contains(client) {
                Button("Disconnect") {
                    change(done: "Disconnected from \(client.name). Restart it to take effect.") {
                        try MCPConfig.remove(from: client.configURL())
                    }
                }
            } else {
                Button("Connect") {
                    change(done: "Connected to \(client.name). Restart \(client.name), then ask it about your calls.") {
                        try MCPConfig.install(in: client.configURL(), executable: executable)
                    }
                }
                .disabled(!client.isPresent())
            }
        }
    }

    private func change(done: String, _ body: () throws -> Void) {
        message = nil
        failure = nil
        do {
            try body()
            message = done
        } catch {
            failure = error.localizedDescription
        }
        refresh()
    }

    private func refresh() {
        guard let executable else { return }
        installed = Set(AgentClient.allCases.filter { MCPConfig.isInstalled(in: $0.configURL(), executable: executable) })
    }
}
