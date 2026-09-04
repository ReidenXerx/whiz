import Testing
import Foundation
import CoreGraphics
import CoreText
import ImageIO
@testable import WhizKit

// swift-testing (see ConfigTests for the toolchain rationale).
//
// normalize and new_screen_lines mirror tests/test_ocr.py fixture-for-fixture
// from the feat/llm-management branch. Vision itself gets a real end-to-end
// run against a rendered-text frame — white-on-black Helvetica is the
// reliable end of Vision's accuracy range, and the reading-order assertion
// proves the join, not just the recognition.

@Suite("Frame OCR")
struct FrameOCRTests {

    // MARK: - normalize (test_ocr.py fixtures)

    @Test("normalize collapses whitespace and drops blank lines")
    func normalizeCollapses() {
        #expect(FrameOCR.normalize("  Hello    World \n\n\n  Foo  \n")
                == "Hello World\nFoo")
    }

    @Test("normalize drops short noise below min_chars")
    func normalizeMinChars() {
        #expect(FrameOCR.normalize("ab", minChars: 8) == "")
        #expect(FrameOCR.normalize("abcdefghij", minChars: 8) == "abcdefghij")
    }

    @Test("normalize truncates on a line boundary with the … marker")
    func normalizeMaxCharsBoundary() {
        let out = FrameOCR.normalize("aaaa\nbbbb\ncccc", maxChars: 6)
        #expect(out.hasSuffix("…"))
        #expect(!out.contains("cccc"))
    }

    @Test("a single long line truncates to max_chars plus the marker")
    func normalizeMaxCharsSingleLine() {
        let out = FrameOCR.normalize(String(repeating: "x", count: 100), maxChars: 10)
        #expect(out.count == 11)
        #expect(out.hasSuffix("…"))
    }

    @Test("empty input normalizes to empty")
    func normalizeEmpty() {
        #expect(FrameOCR.normalize("") == "")
        #expect(FrameOCR.normalize(nil) == "")
    }

    @Test("single-character glyph noise is dropped, two-character content kept")
    func normalizeSingleCharacterNoise() {
        // Toolbar icons OCR as stray single characters — never useful content.
        #expect(FrameOCR.normalize("G\nQ\n+\nReal content here") == "Real content here")
        #expect(FrameOCR.normalize("OK\nsomething else entirely").contains("OK"))
    }

    // MARK: - new_screen_lines (test_ocr.py fixtures)

    @Test("carried-over chrome is stated once, not every frame")
    func newScreenLinesDropsChrome() {
        let previous = "Slack\nFile\nEdit\nInbox (3)"
        let current = "Slack\nFile\nEdit\nInbox (4)\nnew message"
        #expect(FrameOCR.newScreenLines(current: current, previous: previous)
                == "Inbox (4)\nnew message")
    }

    @Test("the first frame keeps everything")
    func newScreenLinesFirstFrame() {
        #expect(FrameOCR.newScreenLines(current: "a line\nb line", previous: "")
                == "a line\nb line")
    }

    @Test("an identical frame yields nothing")
    func newScreenLinesIdentical() {
        let same = "Slack\nFile\nEdit"
        #expect(FrameOCR.newScreenLines(current: same, previous: same) == "")
    }

    @Test("order is preserved in the diff")
    func newScreenLinesPreservesOrder() {
        #expect(FrameOCR.newScreenLines(current: "a\nb\nc\nd", previous: "b")
                == "a\nc\nd")
    }

    @Test("empty current yields empty")
    func newScreenLinesEmptyCurrent() {
        #expect(FrameOCR.newScreenLines(current: "", previous: "anything") == "")
    }

    // MARK: - Vision end to end

    @Test("Vision reads a rendered frame and joins it in reading order")
    func recognizesRenderedText() throws {
        let url = tempURL("jpg")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeTextFrame(at: url, top: "HELLO WHIZ", bottom: "NUMBER 42")

        let raw = try FrameOCR.recognize(url, languages: ["en-US"])
        let text = FrameOCR.normalize(raw, minChars: 8).uppercased()
        #expect(text.contains("HELLO"))
        #expect(text.contains("42"))
        // Reading order: the top line (larger y) sorts first.
        if let hello = text.range(of: "HELLO")?.lowerBound,
           let number = text.range(of: "42")?.lowerBound {
            #expect(hello < number)
        } else {
            Issue.record("expected both rendered lines to be recognized: \(text)")
        }
    }

    @Test("the batch reuses identical frames and fails missing ones softly")
    func batchDedupeAndAlignment() async throws {
        let dir = tempURL("dir")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let first = dir.appendingPathComponent("a.jpg")
        let copy = dir.appendingPathComponent("b.jpg")
        try writeTextFrame(at: first, top: "HELLO WHIZ", bottom: "NUMBER 42")
        try FileManager.default.copyItem(at: first, to: copy)
        let missing = dir.appendingPathComponent("absent.jpg")

        let outcome = await FrameOCR.frames(
            [first, copy, missing],
            languages: ["en-US"],
            minChars: 8)

        #expect(outcome.texts.count == 3)
        #expect(outcome.reused == 1)
        #expect(outcome.failed == 1)
        #expect(outcome.ok >= 1)
        // Alignment: the missing frame keeps its slot with empty text; the
        // two identical frames carry identical text.
        #expect(outcome.texts[2].isEmpty)
        #expect(outcome.texts[0] == outcome.texts[1])
        #expect(!outcome.texts[0].isEmpty)
    }

    @Test("OCR text renders as a collapsed, escaped screen block in the HTML")
    func htmlScreenBlock() {
        let merged = [LabeledSegment(
            segment: WhisperBatchTranscriber.Segment(start: 0, end: 1, text: "hello"),
            speaker: "Speaker A")]
        let entries = [FrameExtractor.Entry(
            index: 1, start: 0, end: 1, speaker: "Speaker A",
            text: "hello", frame: "seg0001.jpg", ocr: "Menu & <div>")]
        let html = SpeakersHTML.format(merged, framesDir: nil, entries: entries)
        #expect(html.contains("<details class=\"screen\">"))
        #expect(html.contains("<summary>screen</summary>"))
        #expect(html.contains("Menu &amp; &lt;div&gt;"))
        #expect(html.contains("class=\"screen-text\""))
        // The search script ships even without frames now.
        #expect(html.contains("getElementById('search')"))
    }

    // MARK: - Helpers

    /// White-on-black Helvetica Bold — Vision's reliable end of the range.
    private func writeTextFrame(at url: URL, top: String, bottom: String) throws {
        let width = 800, height = 600
        guard let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw FrameExtractorError.writeFailed(url.path) }

        context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))

        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 72, nil)
        for (text, y) in [(top, CGFloat(420)), (bottom, CGFloat(280))] {
            let attributed = CFAttributedStringCreate(
                nil, text as CFString,
                [
                    kCTFontAttributeName: font,
                    // CTLineDraw uses the attributed color, not the context
                    // fill — without this the text draws black on black.
                    kCTForegroundColorAttributeName: CGColor(red: 1, green: 1, blue: 1, alpha: 1),
                ] as CFDictionary)!
            let line = CTLineCreateWithAttributedString(attributed)
            context.textPosition = CGPoint(x: 40, y: y)
            CTLineDraw(line, context)
        }

        guard let image = context.makeImage(),
              let destination = CGImageDestinationCreateWithURL(
                url as CFURL, "public.jpeg" as CFString, 1, nil)
        else { throw FrameExtractorError.writeFailed(url.path) }
        CGImageDestinationAddImage(
            destination, image,
            [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw FrameExtractorError.writeFailed(url.path)
        }
    }

    private func tempURL(_ ext: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-ocr-\(UUID().uuidString).\(ext)")
    }
}