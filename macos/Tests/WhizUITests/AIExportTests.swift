import AppKit
import Foundation
import PDFKit
import Testing
@testable import WhizUI

/// Export for AI: the right frames, and files an AI reads correctly. The
/// PDF is read back with PDFKit, as a chat app would, to prove the text is
/// real text and not a picture of it.
@Suite("Export for AI", .timeLimit(.minutes(1)))
struct AIExportTests {
    let directory: URL

    init() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("ai-export-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("talk.frames"),
                                                withIntermediateDirectories: true)
    }

    /// A 320×180 JPEG of one color, with optional faint noise.
    private func frame(_ name: String, _ color: NSColor, noise: Bool = false) throws -> URL {
        let url = directory.appendingPathComponent("talk.frames/\(name).jpg")
        let image = NSImage(size: NSSize(width: 320, height: 180))
        image.lockFocus()
        color.setFill()
        NSRect(x: 0, y: 0, width: 320, height: 180).fill()
        if noise {
            NSColor(white: 0.5, alpha: 0.05).setFill()
            NSRect(x: 40, y: 40, width: 6, height: 6).fill()   // a pointer, say
        }
        image.unlockFocus()
        let bitmap = NSBitmapImageRep(data: image.tiffRepresentation!)!
        try bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.8])!.write(to: url)
        return url
    }

    private func document(frames: [URL?]) -> TranscriptDocument {
        let speakers = ["Maya", "Tom", "Arjun"]
        let segments = frames.enumerated().map { index, url in
            TranscriptSegment(id: index, start: Double(index * 10), end: Double(index * 10 + 9),
                              speaker: speakers[index % 3], text: "Line \(index + 1).", frame: url)
        }
        return TranscriptDocument(segments: segments, source: .framesManifest)
    }

    // MARK: - Key frames

    @Test("a frame is kept only when the screen changed; a moved pointer is not a change")
    func keyFramesFromImages() throws {
        let red = try frame("seg0001", .red)
        let redAgain = try frame("seg0002", .red, noise: true)
        let blue = try frame("seg0003", .blue)
        let blueAgain = try frame("seg0004", .blue)
        let green = try frame("seg0005", .green)
        let doc = document(frames: [red, redAgain, blue, blueAgain, green, nil])
        let kept = AIExport.keyFrames(in: doc)
        #expect(kept.map(\.segmentIndex) == [0, 2, 4])
        #expect(kept.map(\.time) == [0, 20, 40])
    }

    @Test("over the limit, the bar rises and the biggest changes survive")
    func limit() {
        let candidates = (0..<10).map {
            AIExport.KeyFrame(segmentIndex: $0, url: URL(fileURLWithPath: "/f\($0).jpg"), time: Double($0))
        }
        // Small changes everywhere (10), big ones at 4 and 8 (200).
        let prints: [[UInt8]?] = (0..<10).map { index in
            let level: UInt8 = index >= 8 ? 230 : index >= 4 ? 120 : UInt8(index * 10)
            return [UInt8](repeating: level, count: 4)
        }
        // Under the limit: every visible change, the repeated 120s and 230s collapsed.
        #expect(AIExport.select(candidates, prints: prints, limit: 10).map(\.segmentIndex) == [0, 1, 2, 3, 4, 8])
        #expect(AIExport.select(candidates, prints: prints, limit: 3).map(\.segmentIndex) == [0, 4, 8])
        #expect(AIExport.select([], prints: [], limit: 3).isEmpty)
    }

    @Test("an unreadable frame counts as a change rather than vanishing")
    func unreadableFrame() {
        let candidates = (0..<3).map {
            AIExport.KeyFrame(segmentIndex: $0, url: URL(fileURLWithPath: "/f\($0).jpg"), time: Double($0))
        }
        let same: [UInt8] = [1, 1, 1, 1]
        #expect(AIExport.select(candidates, prints: [same, nil, same], limit: 10).map(\.segmentIndex) == [0, 1, 2])
    }

    // MARK: - Markdown

    @Test("markdown: reading note, demoted summary, frames linked relative to the file, one line per segment")
    func markdown() throws {
        let red = try frame("seg0001", .red)
        let doc = document(frames: [red, nil])
        let analysis = AnalysisDocument(response: "## Summary\n- Ship on Friday")
        let frames = AIExport.keyFrames(in: doc)
        let text = AIExport.markdown(title: "talk.mov", subtitle: "Video · 1 min", document: doc,
                                     analysis: analysis, frames: frames, directory: directory)
        #expect(text.hasPrefix("# talk.mov\n\nVideo · 1 min\n\n> A transcript made by Whiz."))
        #expect(text.contains("## AI summary\n\n#### Summary\n- Ship on Friday"))
        #expect(text.contains("![Screen at 0:00](<talk.frames/seg0001.jpg>)\n\n[0:00] **Maya:** Line 1."))
        #expect(text.contains("[0:10] **Tom:** Line 2."))
        // Saved somewhere else: the link is absolute, so it still opens.
        let elsewhere = AIExport.markdown(title: "t", subtitle: "s", document: doc, analysis: nil,
                                          frames: frames, directory: URL(fileURLWithPath: "/tmp/other"))
        #expect(elsewhere.contains("](<\(red.standardizedFileURL.path)>)"))
        #expect(!elsewhere.contains("AI summary"))
    }

    // MARK: - PDF

    @Test("the PDF's text reads back exactly, with captions for the frames that changed")
    func pdf() throws {
        let red = try frame("seg0001", .red)
        let blue = try frame("seg0002", .blue)
        // Enough lines to need a second page.
        let many: [URL?] = [red] + Array(repeating: nil, count: 40) + [blue] + Array(repeating: nil, count: 40)
        let doc = document(frames: many)
        let url = directory.appendingPathComponent("talk for AI.pdf")
        try AIExport.writePDF(title: "talk.mov", subtitle: "Video · 14 min · 3 speakers", document: doc,
                              analysis: AnalysisDocument(response: "## Action items\n- **[Tom]** Draft the notes"),
                              frames: AIExport.keyFrames(in: doc), to: url)

        let pdf = try #require(PDFDocument(url: url))
        #expect(pdf.pageCount >= 2)
        let text = try #require(pdf.string)
        #expect(text.contains("talk.mov"))
        #expect(text.contains("A transcript made by Whiz."))
        #expect(text.contains("AI summary\nAction items\n• [Tom] Draft the notes"))
        // Nothing but the transcript between its lines (no page furniture).
        #expect(text.contains("[1:20] Arjun: Line 9.\n[1:30] Maya: Line 10."))
        #expect(text.contains("[0:00] Maya: Line 1."))
        #expect(text.contains("[13:30] Maya: Line 82."))
        #expect(text.contains("Screen at 0:00") && text.contains("Screen at 6:50"))
        #expect(text.components(separatedBy: "Screen at ").count - 1 == 2)
        #expect(pdf.documentAttributes?[PDFDocumentAttribute.creatorAttribute] as? String == "Whiz")
    }

    @Test("a recording without frames still exports, with no screen note")
    func audioOnly() throws {
        let doc = document(frames: [nil, nil])
        let url = directory.appendingPathComponent("call.pdf")
        try AIExport.writePDF(title: "call.m4a", subtitle: "Audio", document: doc, analysis: nil,
                              frames: AIExport.keyFrames(in: doc), to: url)
        let text = try #require(PDFDocument(url: url)?.string)
        #expect(text.contains("[0:10] Tom: Line 2."))
        #expect(!text.contains("Screen at"))
    }
}
