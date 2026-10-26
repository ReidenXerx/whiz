import Foundation
import Testing
@testable import WhizUI

@Suite("Agents: MCP client configs and the library import")
struct AgentTests {
    let whiz = URL(fileURLWithPath: "/Applications/Whiz.app/Contents/Resources/engine/bin/whiz")

    private func temporaryFile(_ contents: String?) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent("claude_desktop_config.json")
        if let contents { try contents.write(to: file, atomically: true, encoding: .utf8) }
        return file
    }

    @Test("connecting adds only Whiz's entry and keeps everything else in the file")
    func installKeepsTheRest() throws {
        let file = try temporaryFile("""
        {"mcpServers": {"github": {"command": "gh-mcp", "args": []}}, "globalShortcut": "Alt+Space"}
        """)
        #expect(!MCPConfig.isInstalled(in: file, executable: whiz))
        try MCPConfig.install(in: file, executable: whiz)
        #expect(MCPConfig.isInstalled(in: file, executable: whiz))
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
        let servers = root["mcpServers"] as! [String: Any]
        #expect((servers["github"] as? [String: Any])?["command"] as? String == "gh-mcp")
        #expect(root["globalShortcut"] as? String == "Alt+Space")
        #expect((servers["whiz"] as? [String: Any])?["args"] as? [String] == ["mcp"])

        try MCPConfig.remove(from: file)
        #expect(!MCPConfig.isInstalled(in: file, executable: whiz))
        let after = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
        #expect((after["mcpServers"] as? [String: Any])?["github"] != nil)
    }

    @Test("a missing or empty file is created; a broken one is never overwritten")
    func missingAndBroken() throws {
        let missing = try temporaryFile(nil)
        try MCPConfig.install(in: missing, executable: whiz)
        #expect(MCPConfig.isInstalled(in: missing, executable: whiz))

        let empty = try temporaryFile("\n")
        try MCPConfig.install(in: empty, executable: whiz)
        #expect(MCPConfig.isInstalled(in: empty, executable: whiz))

        let broken = try temporaryFile("{\"mcpServers\": {,}")
        #expect(throws: MCPConfig.Unreadable.self) { try MCPConfig.install(in: broken, executable: whiz) }
        #expect(try String(contentsOf: broken, encoding: .utf8) == "{\"mcpServers\": {,}")
    }

    @Test("a Whiz entry pointing at another copy of the app reads as not connected")
    func movedApp() throws {
        let file = try temporaryFile(nil)
        try MCPConfig.install(in: file, executable: URL(fileURLWithPath: "/Users/x/Downloads/Whiz.app/Contents/Resources/engine/bin/whiz"))
        #expect(!MCPConfig.isInstalled(in: file, executable: whiz))
    }

    @Test("the Claude Code command quotes a path with spaces")
    func claudeCodeCommand() {
        #expect(MCPConfig.claudeCodeCommand(executable: whiz)
                == "claude mcp add --scope user whiz -- /Applications/Whiz.app/Contents/Resources/engine/bin/whiz mcp")
        #expect(MCPConfig.claudeCodeCommand(executable: URL(fileURLWithPath: "/Users/a b/Whiz.app/whiz"))
                == "claude mcp add --scope user whiz -- '/Users/a b/Whiz.app/whiz' mcp")
    }

    @Test("client config files live where Claude Desktop and Cursor read them")
    func configLocations() {
        let home = URL(fileURLWithPath: "/Users/someone")
        #expect(AgentClient.claudeDesktop.configURL(home: home).path
                == "/Users/someone/Library/Application Support/Claude/claude_desktop_config.json")
        #expect(AgentClient.cursor.configURL(home: home).path == "/Users/someone/.cursor/mcp.json")
    }

    @Test("history is handed to `whiz library add` once; an engine too old for it is asked again")
    func libraryImport() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let recording = directory.appendingPathComponent("call.m4a")
        FileManager.default.createFile(atPath: recording.path, contents: Data())
        let log = directory.appendingPathComponent("argv")
        let script = directory.appendingPathComponent("whiz")
        func stub(exit status: Int) throws {
            try "#!/bin/sh\necho \"$@\" >> '\(log.path)'\nexit \(status)\n".write(to: script, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        }
        let defaults = UserDefaults(suiteName: "test.library.\(UUID().uuidString)")!
        let history = [
            LastRunRecord(inputPath: recording.path, finishedAt: .now, artifacts: []),
            LastRunRecord(inputPath: "/gone/old.mov", finishedAt: .now, artifacts: []),
        ]

        try stub(exit: 2)   // argparse: no such command
        await LibraryImport.runOnce(history: history, executable: script, defaults: defaults)
        #expect(!defaults.bool(forKey: LibraryImport.doneKey))

        try stub(exit: 1)   // some had nothing to add
        await LibraryImport.runOnce(history: history, executable: script, defaults: defaults)
        #expect(defaults.bool(forKey: LibraryImport.doneKey))
        await LibraryImport.runOnce(history: history, executable: script, defaults: defaults)

        let calls = try String(contentsOf: log, encoding: .utf8).split(separator: "\n")
        #expect(calls.count == 2, "ran until it took, then never again")
        #expect(calls.last == "library add \(recording.path)", "files that are gone are not sent")
    }
}
