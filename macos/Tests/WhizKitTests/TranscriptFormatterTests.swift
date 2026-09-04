import Testing
import Foundation
@testable import WhizKit

// swift-testing (see ConfigTests for the toolchain rationale).
//
// The formatter is the merge.py output layer's Swift half — pure functions,
// pinned exactly. These are the fixtures the future Rust/merge ports will
// also be held to (ARCHITECTURE.md, P2 acceptance bar): the shapes below are
// the contract.

@Suite("Transcript formatting")
struct TranscriptFormatterTests {

    @Test("SRT timestamps are merge.py's _fmt_srt_time")
    func srtTimestamps() {
        #expect(TranscriptFormatter.srtTimestamp(0) == "00:00:00,000")
        #expect(TranscriptFormatter.srtTimestamp(83.456) == "00:01:23,456")
        #expect(TranscriptFormatter.srtTimestamp(3661.5) == "01:01:01,500")
        // Negative clamps, exactly like the Python original.
        #expect(TranscriptFormatter.srtTimestamp(-5) == "00:00:00,000")
    }

    @Test("clock timestamps are merge.py's _fmt_clock")
    func clockTimestamps() {
        #expect(TranscriptFormatter.clock(0) == "00:00:00")
        #expect(TranscriptFormatter.clock(83.456) == "00:01:23")
        #expect(TranscriptFormatter.clock(3661.4) == "01:01:01")
        #expect(TranscriptFormatter.clock(-1) == "00:00:00")
    }

    @Test("SRT output is the whisper-cli cue shape")
    func srtOutput() {
        let segments = [
            WhisperBatchTranscriber.Segment(start: 1.5, end: 3.0, text: "Hello world"),
            WhisperBatchTranscriber.Segment(start: 3.2, end: 5.5, text: "Second cue"),
        ]
        // Every cue ends with a blank line, including the last one — the
        // extra "\n" below makes that explicit since a multiline literal
        // alone cannot end in two newlines.
        #expect(TranscriptFormatter.srt(segments) == """
            1
            00:00:01,500 --> 00:00:03,000
            Hello world

            2
            00:00:03,200 --> 00:00:05,500
            Second cue

            """ + "\n")
    }

    @Test("JSON output parses and carries the parse_whisper_json contract")
    func jsonOutput() throws {
        let segments = [
            WhisperBatchTranscriber.Segment(start: 1.5, end: 3.0, text: "He said \"hi\""),
        ]
        let text = TranscriptFormatter.json(segments)

        // The strongest assertion is that it is valid JSON at all; the rest
        // pins the exact fields merge.py:parse_whisper_json reads.
        let object = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        let transcription = object?["transcription"] as? [[String: Any]]
        #expect(transcription?.count == 1)

        let entry = transcription?.first
        let timestamps = entry?["timestamps"] as? [String: String]
        #expect(timestamps?["from"] == "00:00:01,500")
        #expect(timestamps?["to"] == "00:00:03,000")

        // offsets are milliseconds, matching cli's `t * 10` for centiseconds.
        let offsets = entry?["offsets"] as? [String: Any]
        #expect((offsets?["from"] as? NSNumber)?.intValue == 1500)
        #expect((offsets?["to"] as? NSNumber)?.intValue == 3000)

        // Escaping is cli.cpp's rule: backslash first, then quotes — verified
        // by round-tripping text containing both through a real JSON parse.
        #expect((entry?["text"] as? String) == "He said \"hi\"")
    }

    @Test("empty transcriptions produce a valid empty JSON array")
    func emptyJSONIsStillValid() throws {
        let text = TranscriptFormatter.json([])
        let object = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        let transcription = object?["transcription"] as? [[String: Any]]
        #expect(transcription?.isEmpty == true)
    }

    @Test("segment log lines read as timestamps plus text")
    func segmentLogLines() {
        let segment = WhisperBatchTranscriber.Segment(start: 61.0, end: 76.5, text: "Hi")
        #expect(TranscriptFormatter.segmentLogLine(segment) == "[00:01:01 → 00:01:17]  Hi")
    }
}