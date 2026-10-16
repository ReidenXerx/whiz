import Foundation
import Testing
@testable import WhizUI

/// The readers behind the results view. Every input shape here is one the
/// CLI writes (merge.py's SRT/TXT formatters, screenshots.write_manifest,
/// ai.py's report), so a change on the Python side that breaks the app's
/// reading of it fails here.
@Suite("Reading a run's files")
struct ResultsParsingTests {

    private func tempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test("labeled SRT splits the speaker on the first ': ' only")
    func labeledSRT() {
        let srt = """
        1
        00:00:00,000 --> 00:00:07,720
        Speaker A: Hello: this colon is speech.

        2
        00:01:02,500 --> 00:01:05,000
        Bob: Second line
        continues here.

        """
        let segments = TranscriptDocument.parseSRT(srt, labeled: true)
        #expect(segments.count == 2)
        #expect(segments[0].speaker == "Speaker A")
        #expect(segments[0].text == "Hello: this colon is speech.")
        #expect(segments[0].end == 7.72)
        #expect(segments[1].start == 62.5)
        #expect(segments[1].text == "Second line continues here.")
    }

    @Test("plain SRT keeps ': ' in the text and has no speakers")
    func plainSRT() {
        let srt = "1\r\n00:00:01,000 --> 00:00:02,000\r\nNote: not a speaker\r\n\r\n"
        let segments = TranscriptDocument.parseSRT(srt, labeled: false)
        #expect(segments.map(\.text) == ["Note: not a speaker"])
        #expect(segments[0].speaker == nil)
        #expect(TranscriptDocument(segments: segments, source: .plainSRT).hasSpeakers == false)
    }

    @Test("dialogue TXT turns end where the next one starts")
    func dialogueText() {
        let txt = "Alice (00:00:05): Hi there.\n\nBob (00:01:00): Hello (again): yes."
        let segments = TranscriptDocument.parseDialogue(txt)
        #expect(segments.map(\.speaker) == ["Alice", "Bob"])
        #expect(segments[0].start == 5 && segments[0].end == 60)
        #expect(segments[1].text == "Hello (again): yes.")
    }

    @Test("the frames manifest resolves frames beside it and drops missing ones")
    func manifest() throws {
        let dir = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let frames = dir.appendingPathComponent("talk.frames")
        try FileManager.default.createDirectory(at: frames, withIntermediateDirectories: true)
        try Data([0xff]).write(to: frames.appendingPathComponent("seg0001.jpg"))
        let json = """
        {"version": 1, "frames_dir": "talk.frames", "count": 3, "segments": [
          {"index": 1, "start": 0.0, "end": 2.0, "speaker": "Ann", "text": " Hi ", "frame": "seg0001.jpg"},
          {"index": 2, "start": 2.0, "end": 4.0, "speaker": "Ann", "text": "Gone", "frame": "seg0002.jpg"},
          {"index": 3, "start": 4.0, "end": 5.0, "speaker": "Ben", "text": "   ", "frame": null}
        ]}
        """
        let manifest = dir.appendingPathComponent("talk.frames.json")
        try json.write(to: manifest, atomically: true, encoding: .utf8)

        let doc = try #require(TranscriptDocument.load(
            input: dir.appendingPathComponent("talk.mov"), artifacts: []))
        #expect(doc.source == .framesManifest, "an unannounced manifest beside the input is still found")
        #expect(doc.segments.map(\.text) == ["Hi", "Gone"], "blank text is dropped, text trimmed")
        #expect(doc.segments[0].frame?.lastPathComponent == "seg0001.jpg")
        #expect(doc.segments[1].frame == nil, "a frame missing on disk is no frame")
        #expect(doc.hasFrames)
    }

    @Test("the richest source wins: manifest over labeled SRT over plain SRT")
    func sourcePriority() throws {
        let dir = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let input = dir.appendingPathComponent("talk.mov")
        let plain = dir.appendingPathComponent("talk.wav.srt")
        try "1\n00:00:00,000 --> 00:00:01,000\nplain\n".write(to: plain, atomically: true, encoding: .utf8)
        #expect(TranscriptDocument.load(input: input, artifacts: [plain])?.source == .plainSRT)

        let labeled = dir.appendingPathComponent("talk.speakers.srt")
        try "1\n00:00:00,000 --> 00:00:01,000\nAnn: labeled\n".write(to: labeled, atomically: true, encoding: .utf8)
        #expect(TranscriptDocument.load(input: input, artifacts: [plain, labeled])?.source == .labeledSRT)

        #expect(TranscriptDocument.load(input: dir.appendingPathComponent("other.mov"), artifacts: []) == nil)
    }

    @Test("speakers are colored by first appearance and share their talk time")
    func speakerStats() {
        let doc = TranscriptDocument(segments: [
            .init(id: 1, start: 0, end: 30, speaker: "Ben", text: "a"),
            .init(id: 2, start: 30, end: 40, speaker: "Ann", text: "b"),
            .init(id: 3, start: 40, end: 100, speaker: "Ben", text: "c"),
        ], source: .labeledSRT)
        #expect(doc.speakers.map(\.name) == ["Ben", "Ann"])
        #expect(doc.speakers.map(\.colorIndex) == [0, 1])
        #expect(abs(doc.speakers[0].share - 0.9) < 1e-9)
        #expect(doc.colorIndex(for: "Ann") == 1)
        #expect(doc.startsTurn(at: 0) && doc.startsTurn(at: 1) && doc.startsTurn(at: 2))
        #expect(doc.duration == 100)
    }

    @Test("search is case- and diacritic-insensitive and matches speakers too")
    func search() {
        let doc = TranscriptDocument(segments: [
            .init(id: 1, start: 0, end: 1, speaker: "Елена", text: "Привет"),
            .init(id: 2, start: 1, end: 2, speaker: "Bob", text: "Café meeting"),
        ], source: .labeledSRT)
        #expect(doc.matchingPositions("") == [0, 1])
        #expect(doc.matchingPositions("cafe") == [1])
        #expect(doc.matchingPositions("елена") == [0])
        #expect(doc.matchingPositions("nothing").isEmpty)
    }

    @Test("playback position finds the segment in progress, not one in a gap")
    func playbackPosition() {
        let doc = TranscriptDocument(segments: [
            .init(id: 1, start: 0, end: 5, speaker: nil, text: "a"),
            .init(id: 2, start: 10, end: 15, speaker: nil, text: "b"),
        ], source: .plainSRT)
        #expect(doc.position(at: 2) == 0)
        #expect(doc.position(at: 12) == 1)
        #expect(doc.position(at: 7) == nil, "between segments nothing is playing")
        #expect(doc.position(at: -1) == nil)
    }

    @Test("an analysis report splits into provenance, prompt and response")
    func analysisReport() {
        let report = """
        # whiz analysis — talk.mov

        **Model:** gpt-oss:20b-cloud  **Vision:** False  **Mode:** meeting

        ## Prompt

        ```
        Summarize this.
        ```

        ## Response

        **Overview**
        - one
        """
        let doc = AnalysisDocument.parse(report)
        #expect(doc.model == "gpt-oss:20b-cloud")
        #expect(doc.mode == "meeting")
        #expect(doc.vision == false)
        #expect(doc.prompt == "Summarize this.")
        #expect(doc.response == "**Overview**\n- one")

        let freeform = AnalysisDocument.parse("Just text.")
        #expect(freeform.response == "Just text." && freeform.model == nil)
    }

    @Test("markdown blocks: headings, lists, code, tables, quotes")
    func markdownBlocks() {
        let blocks = MarkdownBlock.parse("""
        ## Summary
        First line
        second line.

        - item
          - nested
        1. first
        > quoted

        ```
        code | not a table
        ```
        | A | B |
        |---|:-:|
        | 1 | 2 |
        ---
        """)
        #expect(blocks == [
            .heading(level: 2, text: "Summary"),
            .paragraph("First line\nsecond line."),
            .listItem(ordinal: nil, depth: 0, text: "item"),
            .listItem(ordinal: nil, depth: 1, text: "nested"),
            .listItem(ordinal: 1, depth: 0, text: "first"),
            .quote("quoted"),
            .code("code | not a table"),
            .table([["A", "B"], ["1", "2"]]),
            .rule,
        ])
        #expect(MarkdownBlock.parse("#hashtag") == [.paragraph("#hashtag")], "no space, no heading")
    }
}
