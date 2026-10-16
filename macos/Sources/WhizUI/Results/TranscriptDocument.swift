import Foundation

/// One line of the transcript as the reader sees it: when, who, what — and
/// the video frame the CLI captured for it, when there is one.
struct TranscriptSegment: Identifiable, Equatable, Sendable {
    var id: Int
    var start: Double
    var end: Double
    /// nil for a transcript without speaker labels (plain whisper SRT).
    var speaker: String?
    var text: String
    var frame: URL? = nil
}

/// A speaker's share of the recording, for the legend above the transcript.
struct SpeakerStat: Identifiable, Equatable, Sendable {
    var id: String { name }
    var name: String
    /// Index into `Theme.speakerColor` — by order of first appearance, the
    /// same rule merge.py's `speaker_palette` uses, so a speaker is the same
    /// color here and in the HTML transcript.
    var colorIndex: Int
    var seconds: Double
    var share: Double
}

/// The transcript of a finished run, read back from the files the CLI wrote.
///
/// The app parses what is already on disk rather than asking the CLI for a
/// new format: the frames manifest is JSON written for exactly this kind of
/// consumer (`screenshots.write_manifest`), and the labeled SRT is a format
/// the CLI's own tests pin. Nothing here re-implements pipeline logic — it is
/// a reader.
struct TranscriptDocument: Equatable, Sendable {
    enum Source: Equatable, Sendable {
        /// `<stem>.frames.json` — segments, speakers, and a frame each.
        case framesManifest
        /// `<stem>.speakers.srt` — segments and speakers.
        case labeledSRT
        /// `<stem>.speakers.txt` — speaker turns with start times only.
        case dialogueText
        /// whisper's own SRT — segments, no speakers.
        case plainSRT
    }

    var segments: [TranscriptSegment]
    var source: Source
    var speakers: [SpeakerStat]

    init(segments: [TranscriptSegment], source: Source) {
        self.segments = segments
        self.source = source
        self.speakers = Self.stats(for: segments)
    }

    var duration: Double { segments.map(\.end).max() ?? 0 }
    var hasFrames: Bool { segments.contains { $0.frame != nil } }
    var hasSpeakers: Bool { !speakers.isEmpty }

    func colorIndex(for speaker: String?) -> Int? {
        guard let speaker else { return nil }
        return speakers.first { $0.name == speaker }?.colorIndex
    }

    /// True where a new speaker turn begins — the view prints the name only
    /// there, so a monologue reads as one block rather than a stutter.
    func startsTurn(at index: Int) -> Bool {
        guard index > 0 else { return true }
        return segments[index].speaker != segments[index - 1].speaker
    }

    /// Positions of the segments whose text or speaker contains `query`
    /// (case- and diacritic-insensitive — "Елена" finds "елена"); every
    /// position for an empty query. Positions, not ids: ids come from the
    /// files and are not guaranteed unique.
    func matchingPositions(_ query: String) -> [Int] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else { return Array(segments.indices) }
        let options: String.CompareOptions = [.caseInsensitive, .diacriticInsensitive]
        return segments.indices.filter {
            segments[$0].text.range(of: needle, options: options) != nil
                || (segments[$0].speaker?.range(of: needle, options: options) != nil)
        }
    }

    /// The position of the segment playing at `time`: the last one that
    /// started at or before it, while `time` is still inside it (a gap
    /// between segments highlights nothing).
    func position(at time: Double) -> Int? {
        var low = 0
        var high = segments.count - 1
        var found: Int?
        while low <= high {
            let mid = (low + high) / 2
            if segments[mid].start <= time {
                found = mid
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        guard let index = found, time <= segments[index].end + 0.25 else { return nil }
        return index
    }

    static func stats(for segments: [TranscriptSegment]) -> [SpeakerStat] {
        var order: [String] = []
        var seconds: [String: Double] = [:]
        for segment in segments {
            guard let speaker = segment.speaker else { continue }
            if seconds[speaker] == nil { order.append(speaker) }
            seconds[speaker, default: 0] += max(segment.end - segment.start, 0)
        }
        let total = seconds.values.reduce(0, +)
        return order.enumerated().map { index, name in
            let s = seconds[name] ?? 0
            return SpeakerStat(name: name, colorIndex: index, seconds: s,
                               share: total > 0 ? s / total : 0)
        }
    }

    // MARK: - Loading

    /// The best transcript the run left behind, or nil.
    ///
    /// Candidates are the run's announced artifacts plus the CLI's sibling
    /// names for the input (`<stem>.frames.json`, `<stem>.speakers.srt`, …):
    /// an output the CLI KEPT rather than rewrote (a named manifest from an
    /// earlier run) is on disk but never announced. Richest source wins.
    static func load(input: URL, artifacts: [URL]) -> TranscriptDocument? {
        let base = input.deletingPathExtension().path
        var candidates = artifacts
        for suffix in [".frames.json", ".speakers.srt", ".speakers.txt", ".srt", ".wav.srt"] {
            candidates.append(URL(fileURLWithPath: base + suffix))
        }
        let existing = candidates.filter { FileManager.default.fileExists(atPath: $0.path) }

        if let manifest = existing.first(where: { $0.lastPathComponent.hasSuffix(".frames.json") }),
           let data = try? Data(contentsOf: manifest),
           let segments = parseManifest(data, manifestURL: manifest), !segments.isEmpty {
            return TranscriptDocument(segments: segments, source: .framesManifest)
        }
        if let srt = existing.first(where: { $0.lastPathComponent.hasSuffix(".speakers.srt") }),
           let text = try? String(contentsOf: srt, encoding: .utf8) {
            let segments = parseSRT(text, labeled: true)
            if !segments.isEmpty { return TranscriptDocument(segments: segments, source: .labeledSRT) }
        }
        if let txt = existing.first(where: { $0.lastPathComponent.hasSuffix(".speakers.txt") }),
           let text = try? String(contentsOf: txt, encoding: .utf8) {
            let segments = parseDialogue(text)
            if !segments.isEmpty { return TranscriptDocument(segments: segments, source: .dialogueText) }
        }
        if let srt = existing.first(where: { $0.pathExtension.lowercased() == "srt" }),
           let text = try? String(contentsOf: srt, encoding: .utf8) {
            let segments = parseSRT(text, labeled: false)
            if !segments.isEmpty { return TranscriptDocument(segments: segments, source: .plainSRT) }
        }
        return nil
    }

    // MARK: - Parsers

    /// `screenshots.write_manifest`'s JSON. Frame paths are relative to the
    /// manifest's `frames_dir`, which sits beside the manifest; a segment
    /// whose frame is missing on disk simply has none.
    static func parseManifest(_ data: Data, manifestURL: URL) -> [TranscriptSegment]? {
        struct Manifest: Decodable {
            struct Row: Decodable {
                var index: Int?
                var start: Double
                var end: Double
                var speaker: String?
                var text: String?
                var frame: String?
            }
            var frames_dir: String?
            var segments: [Row]
        }
        guard let manifest = try? JSONDecoder().decode(Manifest.self, from: data) else { return nil }
        let framesDir = manifestURL.deletingLastPathComponent()
            .appendingPathComponent(manifest.frames_dir ?? "", isDirectory: true)
        return manifest.segments.enumerated().compactMap { offset, row in
            let text = (row.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            var frame: URL?
            if let name = row.frame, !name.isEmpty, manifest.frames_dir != nil {
                let url = framesDir.appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: url.path) { frame = url }
            }
            return TranscriptSegment(id: row.index ?? offset + 1, start: row.start, end: row.end,
                                     speaker: row.speaker, text: text, frame: frame)
        }
    }

    /// SRT cues. `labeled` splits merge.py's `Label: text` on the FIRST
    /// ": " — only for the `.speakers.srt` the CLI labels; whisper's plain
    /// SRT text can contain ": " that is speech, not a speaker.
    static func parseSRT(_ text: String, labeled: Bool) -> [TranscriptSegment] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        var segments: [TranscriptSegment] = []
        for block in normalized.components(separatedBy: "\n\n") {
            let lines = block.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
            guard let timeIndex = lines.firstIndex(where: { $0.contains("-->") }) else { continue }
            let parts = lines[timeIndex].components(separatedBy: "-->")
            guard parts.count == 2,
                  let start = parseClock(parts[0]), let end = parseClock(parts[1]) else { continue }
            var body = lines[(timeIndex + 1)...].joined(separator: " ")
                .trimmingCharacters(in: .whitespaces)
            guard !body.isEmpty else { continue }
            var speaker: String?
            if labeled, let separator = body.range(of: ": ") {
                speaker = String(body[..<separator.lowerBound])
                body = String(body[separator.upperBound...])
            }
            let id = Int(lines[0].trimmingCharacters(in: .whitespaces)) ?? segments.count + 1
            segments.append(TranscriptSegment(id: id, start: start, end: end,
                                              speaker: speaker, text: body, frame: nil))
        }
        return segments
    }

    /// merge.py's `format_dialogue_txt`: `Label (HH:MM:SS): text` turns
    /// separated by blank lines. Only start times are recorded, so a turn
    /// ends where the next one starts.
    static func parseDialogue(_ text: String) -> [TranscriptSegment] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        var rows: [(speaker: String, start: Double, text: String)] = []
        for block in normalized.components(separatedBy: "\n\n") {
            let trimmed = block.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let open = trimmed.range(of: " ("),
                  let close = trimmed.range(of: "): ", range: open.upperBound..<trimmed.endIndex),
                  let start = parseClock(String(trimmed[open.upperBound..<close.lowerBound]))
            else { continue }
            let speaker = String(trimmed[..<open.lowerBound])
            let body = String(trimmed[close.upperBound...]).replacingOccurrences(of: "\n", with: " ")
            rows.append((speaker, start, body))
        }
        return rows.enumerated().map { index, row in
            let end = index + 1 < rows.count ? rows[index + 1].start : row.start
            return TranscriptSegment(id: index + 1, start: row.start, end: end,
                                     speaker: row.speaker, text: row.text, frame: nil)
        }
    }

    /// `HH:MM:SS,mmm` (SRT) or `HH:MM:SS` (dialogue TXT) → seconds.
    static func parseClock(_ raw: String) -> Double? {
        let clock = raw.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: ".")
        let parts = clock.split(separator: ":")
        guard parts.count == 3,
              let h = Double(parts[0]), let m = Double(parts[1]), let s = Double(parts[2])
        else { return nil }
        return h * 3600 + m * 60 + s
    }
}

/// `whiz analyze`'s markdown report, split into what a reader wants (the
/// model's response) and the provenance around it.
struct AnalysisDocument: Equatable, Sendable {
    var model: String?
    var mode: String?
    var vision: Bool?
    var prompt: String?
    /// Markdown — the part worth reading.
    var response: String

    /// The report is `# title`, a `**Model:** … **Vision:** … **Mode:** …`
    /// line, `## Prompt` (fenced), then `## Response`. Anything that does not
    /// follow that shape is shown whole rather than guessed at.
    static func parse(_ text: String) -> AnalysisDocument {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        var doc = AnalysisDocument(response: normalized)

        for line in normalized.split(separator: "\n", omittingEmptySubsequences: false).prefix(8)
        where line.hasPrefix("**Model:**") {
            doc.model = field("Model", in: String(line))
            doc.mode = field("Mode", in: String(line))
            doc.vision = field("Vision", in: String(line)).map { $0.lowercased() == "true" }
        }

        guard let response = normalized.range(of: "\n## Response\n") else { return doc }
        doc.response = String(normalized[response.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let prompt = normalized.range(of: "\n## Prompt\n"), prompt.upperBound <= response.lowerBound {
            var body = String(normalized[prompt.upperBound..<response.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if body.hasPrefix("```") {
                body = String(body.drop { $0 != "\n" }.dropFirst())
            }
            if body.hasSuffix("```") { body = String(body.dropLast(3)) }
            doc.prompt = body.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return doc
    }

    /// The value after `**Name:**`, up to the next `**` or end of line.
    private static func field(_ name: String, in line: String) -> String? {
        guard let marker = line.range(of: "**\(name):**") else { return nil }
        let rest = line[marker.upperBound...]
        let value = rest.range(of: "**").map { rest[..<$0.lowerBound] } ?? rest
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? nil : trimmed
    }
}
