import Foundation

/// A failure whiz shows the user: the message names the file and, where
/// one exists, the way out.
struct MediaError: Error, CustomStringConvertible {
    var description: String
    init(_ description: String) { self.description = description }
}

/// The command line: parse, dispatch, map failures to an exit status.
///
/// Exit 0 means the output file exists and is complete; 1 is a failure
/// (message on stderr, no output left behind); 2 is a usage error.
enum MediaCommand {
    static let usage = """
        usage: whiz-media audio <input> <output.wav> [--comment TEXT]
               whiz-media frame <input> <seconds> <output.jpg> [--width N] [--quality Q]
        """

    static func run(_ arguments: [String]) async -> Int32 {
        do {
            let parsed = try Arguments(arguments)
            switch parsed.command {
            case "audio":
                try parsed.expectPositionals(2, options: ["--comment"])
                try parsed.requireInput()
                try await AudioExtractor.extract(
                    from: parsed.file(0), to: parsed.file(1), comment: parsed.options["--comment"])
            case "frame":
                try parsed.expectPositionals(3, options: ["--width", "--quality"])
                guard let seconds = Double(parsed.positionals[1]), seconds.isFinite else {
                    throw UsageError("not a time in seconds: \(parsed.positionals[1])")
                }
                try parsed.requireInput()
                try await FrameExtractor.extract(
                    from: parsed.file(0), at: seconds, to: parsed.file(2),
                    width: try parsed.int("--width") ?? 0,
                    quality: try parsed.int("--quality") ?? 2)
            case "--version":
                print("whiz-media 1")
            case "-h", "--help":
                print(usage)
            default:
                throw UsageError("unknown command: \(parsed.command)")
            }
            return 0
        } catch let error as UsageError {
            FileHandle.standardError.write(Data("whiz-media: \(error.message)\n\(usage)\n".utf8))
            return 2
        } catch {
            FileHandle.standardError.write(Data("whiz-media: \(error)\n".utf8))
            return 1
        }
    }
}

struct UsageError: Error {
    var message: String
    init(_ message: String) { self.message = message }
}

/// `<command> <positional>... [--option value]...`, options anywhere.
struct Arguments {
    var command: String
    var positionals: [String] = []
    var options: [String: String] = [:]

    init(_ arguments: [String]) throws {
        guard let first = arguments.first else { throw UsageError("no command") }
        command = first
        var rest = arguments.dropFirst()[...]
        while let argument = rest.popFirst() {
            if argument.hasPrefix("--") {
                guard let value = rest.popFirst() else { throw UsageError("\(argument) needs a value") }
                options[argument] = value
            } else {
                positionals.append(argument)
            }
        }
    }

    func expectPositionals(_ count: Int, options allowed: Set<String>) throws {
        guard positionals.count == count else {
            throw UsageError("\(command) takes \(count) arguments, got \(positionals.count)")
        }
        if let unknown = options.keys.first(where: { !allowed.contains($0) }) {
            throw UsageError("\(command) has no option \(unknown)")
        }
    }

    /// AVFoundation reports a missing file as error -17913; say what it is.
    func requireInput() throws {
        guard FileManager.default.fileExists(atPath: file(0).path) else {
            throw MediaError("no such file: \(file(0).path)")
        }
    }

    func file(_ index: Int) -> URL {
        URL(fileURLWithPath: (positionals[index] as NSString).expandingTildeInPath)
    }

    func int(_ option: String) throws -> Int? {
        guard let text = options[option] else { return nil }
        guard let value = Int(text) else { throw UsageError("\(option) needs a whole number, got \(text)") }
        return value
    }
}

/// Write `output` by way of a temporary file beside it, so a failure (or a
/// kill) never leaves a half-written file under the final name — whiz
/// treats "the file exists" as success.
func writeAtomically(to output: URL, _ body: (URL) throws -> Void) throws {
    let directory = output.deletingLastPathComponent()
    let temporary = directory.appendingPathComponent(".\(output.lastPathComponent).\(UUID().uuidString).part")
    defer { try? FileManager.default.removeItem(at: temporary) }
    try body(temporary)
    if FileManager.default.fileExists(atPath: output.path) {
        _ = try FileManager.default.replaceItemAt(output, withItemAt: temporary)
    } else {
        try FileManager.default.moveItem(at: temporary, to: output)
    }
}
