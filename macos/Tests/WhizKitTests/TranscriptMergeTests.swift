import Testing
import Foundation
@testable import WhizKit

// swift-testing (see ConfigTests for the toolchain rationale).
//
// This suite mirrors tests/test_merge.py fixture-for-fixture: same segments,
// same diarization overlaps, same expected strings. Both implementations are
// held to one behavior contract — the NS-1 discipline (pins against a shared
// source of truth) extended from the tuning constants to the merge layer.
// The two parse_whisper_json tests have no Swift counterpart because the
// native pipeline produces segments in-process; the JSON it *writes* is
// pinned in TranscriptFormatterTests.

@Suite("Labeled transcript merge")
struct TranscriptMergeTests {

    private func seg(_ start: Double, _ end: Double, _ text: String = "hello") -> WhisperBatchTranscriber.Segment {
        WhisperBatchTranscriber.Segment(start: start, end: end, text: text)
    }

    // MARK: - Speaker labels (merge.py:speaker_label)

    @Test("speaker labels are letters up to Z, then numbers")
    func speakerLabelLetters() {
        #expect(LabeledTranscript.speakerLabel(0) == "Speaker A")
        #expect(LabeledTranscript.speakerLabel(1) == "Speaker B")
        #expect(LabeledTranscript.speakerLabel(25) == "Speaker Z")
        #expect(LabeledTranscript.speakerLabel(26) == "Speaker 26")
    }

    // MARK: - Assignment (merge.py:assign_speakers)

    @Test("a whisper segment is labeled by the diarization speaker it overlaps most")
    func assignSpeakersMaxOverlap() {
        let whisper = [
            seg(0.0, 2.0, "first"),
            seg(5.0, 7.0, "second"),
            seg(10.0, 12.0, "third"),
        ]
        let diar = [
            DiarSegment(start: 0.0, end: 3.0, speaker: 0),    // Speaker A
            DiarSegment(start: 4.0, end: 8.0, speaker: 1),    // Speaker B
            DiarSegment(start: 9.0, end: 13.0, speaker: 0),   // Speaker A again
        ]
        let merged = LabeledTranscript.assignSpeakers(segments: whisper, diar: diar)
        #expect(merged.map(\.speaker) == ["Speaker A", "Speaker B", "Speaker A"])
    }

    @Test("with no diarization segments everything falls back to Speaker A")
    func assignSpeakersNoDiar() {
        let whisper = [seg(0.0, 1.0, "x"), seg(2.0, 3.0, "y")]
        let merged = LabeledTranscript.assignSpeakers(segments: whisper, diar: [])
        #expect(merged.allSatisfy { $0.speaker == "Speaker A" })
    }

    // MARK: - Speaker ordering (merge.py:speakers_by_talk_time / _in_order)

    @Test("talk-time ordering puts the most talkative first")
    func speakersByTalkTimeOrdersMostFirst() {
        let merged = [
            LabeledSegment(segment: seg(0.0, 10.0, "long"), speaker: "Speaker A"),    // 10s
            LabeledSegment(segment: seg(0.0, 2.0, "short"), speaker: "Speaker B"),    // 2s
            LabeledSegment(segment: seg(0.0, 5.0, "mid"), speaker: "Speaker A"),      // A total 15s
            LabeledSegment(segment: seg(0.0, 1.0, "tiny"), speaker: "Speaker C"),     // 1s
        ]
        #expect(LabeledTranscript.speakersByTalkTime(merged) == ["Speaker A", "Speaker B", "Speaker C"])
    }

    @Test("order of appearance keeps first-seen sequence")
    func speakersInOrderOfAppearance() {
        let merged = [
            LabeledSegment(segment: seg(0, 1), speaker: "Speaker B"),
            LabeledSegment(segment: seg(1, 2), speaker: "Speaker A"),
            LabeledSegment(segment: seg(2, 3), speaker: "Speaker B"),
            LabeledSegment(segment: seg(3, 4), speaker: "Speaker C"),
        ]
        #expect(LabeledTranscript.speakersInOrder(merged) == ["Speaker B", "Speaker A", "Speaker C"])
    }

    // MARK: - Relabeling (merge.py:relabel)

    @Test("relabel replaces known labels and leaves others untouched")
    func relabelReplacesNames() {
        let merged = [
            LabeledSegment(segment: seg(0, 1, "hi"), speaker: "Speaker A"),
            LabeledSegment(segment: seg(1, 2, "yo"), speaker: "Speaker B"),
        ]
        let out = LabeledTranscript.relabel(merged, ["Speaker A": "Alice"])
        #expect(out[0].speaker == "Alice")
        #expect(out[1].speaker == "Speaker B")
    }

    // MARK: - Representative quotes (merge.py:representative_quotes)

    @Test("representative quotes pick the longest utterance per speaker")
    func representativeQuotesPicksLongest() {
        let merged = [
            LabeledSegment(segment: seg(0, 1, "Yeah."), speaker: "Speaker A"),
            LabeledSegment(segment: seg(1, 5, "Let me explain the whole plan in detail now."), speaker: "Speaker A"),
            LabeledSegment(segment: seg(5, 6, "Ok."), speaker: "Speaker A"),
            LabeledSegment(segment: seg(6, 7, "Sure."), speaker: "Speaker B"),
        ]
        let quotes = LabeledTranscript.representativeQuotes(merged)
        #expect((quotes["Speaker A"] ?? "").contains("plan in detail"))
        #expect(quotes["Speaker B"] == "Sure.")
    }

    // MARK: - Formats (merge.py:format_labeled_srt / format_dialogue_txt)

    @Test("labeled SRT keeps the cue structure with the speaker prefix")
    func formatLabeledSRTStructure() {
        let merged = [LabeledSegment(segment: seg(0.0, 1.5, "Hello world"), speaker: "Speaker A")]
        let srt = LabeledTranscript.formatLabeledSRT(merged)
        let lines = srt.components(separatedBy: "\n")
        #expect(lines[0] == "1")
        #expect(lines[1].contains("00:00:00,000"))
        #expect(lines[1].contains("00:00:01,500"))
        #expect(lines[2] == "Speaker A: Hello world")
    }

    @Test("dialogue TXT merges consecutive same-speaker segments into one block")
    func formatDialogueTXTMergesConsecutive() {
        let merged = [
            LabeledSegment(segment: seg(0.0, 1.0, "First."), speaker: "Speaker A"),
            LabeledSegment(segment: seg(1.0, 2.0, "Second."), speaker: "Speaker A"),
            LabeledSegment(segment: seg(2.0, 3.0, "Reply."), speaker: "Speaker B"),
        ]
        let txt = LabeledTranscript.formatDialogueTXT(merged)
        let blocks = txt.components(separatedBy: "\n\n")
        #expect(blocks.count == 2)
        #expect(blocks[0] == "Speaker A (00:00:00): First. Second.")
        #expect(blocks[1] == "Speaker B (00:00:02): Reply.")
    }

    // MARK: - HTML (merge.py:format_speakers_html)

    @Test("HTML escapes speaker text and renders cues")
    func htmlBasic() {
        let merged = [LabeledSegment(segment: seg(0.0, 1.5, "Hello & <welcome>"), speaker: "Speaker A")]
        let html = SpeakersHTML.format(merged, title: "My Meeting")
        #expect(html.hasPrefix("<!DOCTYPE html>"))
        #expect(html.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("</html>"))
        #expect(html.contains("My Meeting"))
        #expect(html.contains("&amp;"))
        #expect(html.contains("&lt;welcome&gt;"))
        #expect(html.contains("class=\"cue\""))
    }

    @Test("a frames directory with no matching JPEGs emits no images")
    func htmlNoFramesWhenDirMissing() throws {
        let merged = [LabeledSegment(segment: seg(0.0, 1.0, "hi"), speaker: "Speaker A")]
        let emptyDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-merge-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: emptyDir) }

        let html = SpeakersHTML.format(merged, framesDir: emptyDir)
        #expect(!html.contains("<img"))
    }

    @Test("the sticky header renders a speaker legend and a search input")
    func htmlHasLegendAndSearch() {
        let merged = [
            LabeledSegment(segment: seg(0.0, 1.0, "hi"), speaker: "Speaker A"),
            LabeledSegment(segment: seg(1.0, 2.0, "yo"), speaker: "Speaker B"),
        ]
        let html = SpeakersHTML.format(merged)
        #expect(html.contains("class=\"legend\""))
        #expect(html.contains("Speaker A") && html.contains("Speaker B"))
        #expect(html.contains("id=\"search\""))
        #expect(html.contains("type=\"search\""))
    }

    @Test("a present frame is inlined and the lightbox + script are emitted")
    func htmlFrameClickableOpensLightbox() throws {
        let merged = [LabeledSegment(segment: seg(0.0, 1.0, "hi"), speaker: "Speaker A")]
        let framesDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-merge-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: framesDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: framesDir) }
        // The same fake-JPEG bytes as the Python fixture.
        var frame = Data([0xFF, 0xD8])
        frame.append(Data("jpeg".utf8))
        frame.append(Data([0xFF, 0xD9]))
        try frame.write(to: framesDir.appendingPathComponent("seg0001.jpg"))

        let html = SpeakersHTML.format(merged, framesDir: framesDir)
        #expect(html.contains("class=\"frame\""))
        #expect(html.contains("<img"))
        #expect(html.contains("class=\"lightbox\""))
        #expect(html.contains("id=\"lightbox\""))
        #expect(html.contains("<script>"))
        // Inlined as a data URI, base64 of the exact fixture bytes.
        #expect(html.contains("data:image/jpeg;base64,"))
    }

    @Test("no frames means no lightbox — but the search script still ships")
    func htmlNoLightboxWithoutFrames() throws {
        let merged = [LabeledSegment(segment: seg(0.0, 1.0, "hi"), speaker: "Speaker A")]
        let emptyDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-merge-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: emptyDir) }

        let html = SpeakersHTML.format(merged, framesDir: emptyDir)
        #expect(!html.contains("class=\"lightbox\""))
        #expect(!html.contains("getElementById('lightbox')"))
        // The search script ships regardless — the branch fixed the box
        // filtering only when frames existed.
        #expect(html.contains("getElementById('search')"))
    }

    // MARK: - Colors (merge.py:speaker_palette)

    @Test("speaker colors are stable and cycle through the palette")
    func speakerColorsAreStable() {
        #expect(SpeakersHTML.speakerColor("Speaker A") == SpeakersHTML.speakerColor("Speaker A"))
        #expect(SpeakersHTML.palette.contains(SpeakersHTML.speakerColor("Speaker Z")))
    }
}