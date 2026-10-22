import AppKit
import CoreGraphics
import Foundation
import ImageIO

/// "Export for AI": a run as something an AI can take in whole, with what
/// was said and what was on screen.
///
/// Two shapes, for two kinds of AI:
/// - **PDF**, for chat apps (ChatGPT, Claude). One file to drop in, and
///   both read a PDF's text and its images. The transcript is real text,
///   so it is extracted exactly, and the screen appears wherever it changed.
/// - **Markdown**, for agents that work with files (Claude Code, Codex,
///   Cursor). The same reading order, with links to the frames on disk.
///
/// Only *key* frames go in. A transcript has a frame per line, and most of
/// them repeat the slide before them. Thirty minutes on twelve slides gives
/// twelve images, not three hundred, which keeps the PDF within what chat
/// apps accept and the reading free of duplicates.
enum AIExport {

    /// A frame worth showing, and the transcript line it belongs to.
    struct KeyFrame: Equatable, Sendable {
        var segmentIndex: Int
        var url: URL
        var time: Double
    }

    /// The most images either format carries. Chat apps cap a PDF at about
    /// a hundred pages; this keeps a long screen share well inside that.
    static let frameLimit = 60

    /// Mean difference per pixel (0…255) between two 32×18 grayscale
    /// thumbnails below which two frames count as the same screen. Video
    /// compression noise and a moving pointer stay under it; a new slide,
    /// a scrolled page or a switched window go well over.
    static let sameScreen = 6.0

    // MARK: - Key frames

    /// The frames to show: the first, then each one that differs from the
    /// last one kept. More than `limit` (a webcam, a video) and the bar for
    /// "different" rises until they fit, so the biggest changes survive.
    static func keyFrames(in document: TranscriptDocument, limit: Int = frameLimit,
                          fingerprint: (URL) -> [UInt8]? = Fingerprint.make) -> [KeyFrame] {
        let candidates = document.segments.enumerated().compactMap { index, segment in
            segment.frame.map { KeyFrame(segmentIndex: index, url: $0, time: segment.start) }
        }
        let prints = candidates.map { fingerprint($0.url) }
        return select(candidates, prints: prints, limit: limit)
    }

    /// The selection itself, separate from loading images so it can be
    /// tested with made-up fingerprints. A frame that cannot be read counts
    /// as a change; it is left in rather than silently dropped.
    static func select(_ candidates: [KeyFrame], prints: [[UInt8]?], limit: Int,
                       threshold: Double = sameScreen) -> [KeyFrame] {
        guard !candidates.isEmpty, limit > 0 else { return [] }
        var bar = threshold
        while true {
            var kept: [Int] = [0]
            for index in candidates.indices.dropFirst() {
                let last = kept[kept.count - 1]
                if Fingerprint.distance(prints[last], prints[index]) > bar { kept.append(index) }
            }
            if kept.count <= limit || bar >= 255 {
                return kept.prefix(limit).map { candidates[$0] }
            }
            bar = min(255, bar * 1.5)
        }
    }

    // MARK: - Markdown

    /// The run as Markdown for an agent. Frame links are relative to
    /// `directory` (where the file is saved) when the frames sit below it,
    /// absolute otherwise, so they open from wherever the agent reads.
    static func markdown(title: String, subtitle: String, document: TranscriptDocument,
                         analysis: AnalysisDocument?, frames: [KeyFrame], directory: URL) -> String {
        var out: [String] = [
            "# \(title)",
            "",
            subtitle,
            "",
            "> \(readingNote(hasFrames: !frames.isEmpty))",
            "",
        ]
        if let analysis, !analysis.response.isEmpty {
            // "AI summary": the reader should know this part is derived, not said.
            out += ["## AI summary", "", demoteHeadings(analysis.response), ""]
        }
        out += ["## Transcript", ""]
        let bySegment = Dictionary(uniqueKeysWithValues: frames.map { ($0.segmentIndex, $0) })
        for (index, segment) in document.segments.enumerated() {
            if let frame = bySegment[index] {
                out += ["![Screen at \(Format.clock(frame.time))](<\(link(to: frame.url, from: directory))>)", ""]
            }
            out.append(line(for: segment, markdown: true))
            out.append("")
        }
        return out.joined(separator: "\n")
    }

    static func link(to file: URL, from directory: URL) -> String {
        let base = directory.standardizedFileURL.path
        let path = file.standardizedFileURL.path
        if path.hasPrefix(base + "/") { return String(path.dropFirst(base.count + 1)) }
        return path
    }

    // MARK: - PDF

    /// Write the run as a PDF: title, the summary when there is one, then
    /// the transcript with each key frame before the line it belongs to.
    static func writePDF(title: String, subtitle: String, document: TranscriptDocument,
                         analysis: AnalysisDocument?, frames: [KeyFrame], to url: URL) throws {
        let writer = try PDFWriter(url: url, title: title)
        writer.paragraph(NSAttributedString(string: title, attributes: Style.title), spacing: 4)
        writer.paragraph(NSAttributedString(string: subtitle, attributes: Style.meta), spacing: 10)
        writer.paragraph(NSAttributedString(string: readingNote(hasFrames: !frames.isEmpty), attributes: Style.note),
                         spacing: 18)

        if let analysis, !analysis.response.isEmpty {
            writer.paragraph(NSAttributedString(string: "AI summary", attributes: Style.heading), spacing: 8)
            for block in plainBlocks(analysis.response) { writer.paragraph(block, spacing: 5) }
            writer.space(14)
        }

        writer.paragraph(NSAttributedString(string: "Transcript", attributes: Style.heading), spacing: 8)
        let bySegment = Dictionary(uniqueKeysWithValues: frames.map { ($0.segmentIndex, $0) })
        for (index, segment) in document.segments.enumerated() {
            if let frame = bySegment[index], let image = NSImage(contentsOf: frame.url) {
                writer.image(image, caption: "Screen at \(Format.clock(frame.time))")
            }
            writer.paragraph(styledLine(for: segment), spacing: 6)
        }
        try writer.finish()
    }

    // MARK: - Shared text

    /// What the reader (an AI) needs to know to read the rest correctly.
    static func readingNote(hasFrames: Bool) -> String {
        var note = "A transcript made by Whiz. Each line is [time] Speaker: what they said, in order."
        if hasFrames {
            note += " A \"Screen at\" image shows what was on screen from that moment until the next one;"
                + " lines that follow it were said while it was showing."
        }
        return note
    }

    static func line(for segment: TranscriptSegment, markdown: Bool) -> String {
        let time = "[\(Format.clock(segment.start))]"
        guard let speaker = segment.speaker else { return "\(time) \(segment.text)" }
        return markdown ? "\(time) **\(speaker):** \(segment.text)" : "\(time) \(speaker): \(segment.text)"
    }

    private static func styledLine(for segment: TranscriptSegment) -> NSAttributedString {
        let text = NSMutableAttributedString(string: "[\(Format.clock(segment.start))] ", attributes: Style.time)
        if let speaker = segment.speaker {
            text.append(NSAttributedString(string: "\(speaker): ", attributes: Style.speaker))
        }
        text.append(NSAttributedString(string: segment.text, attributes: Style.body))
        return text
    }

    /// The summary's own headings move two levels down, under "## AI summary".
    static func demoteHeadings(_ markdown: String) -> String {
        markdown.split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.hasPrefix("#") ? "##" + $0 : String($0) }
            .joined(separator: "\n")
    }

    /// The summary's Markdown as readable PDF paragraphs: headings bold,
    /// list items with a bullet, emphasis markers removed.
    static func plainBlocks(_ markdown: String) -> [NSAttributedString] {
        markdown.split(separator: "\n").compactMap { raw -> NSAttributedString? in
            var line = String(raw).trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, line != "---" else { return nil }
            var attributes = Style.body
            if line.hasPrefix("#") {
                line = String(line.drop(while: { $0 == "#" })).trimmingCharacters(in: .whitespaces)
                attributes = Style.subheading
            } else if line.count > 4, line.hasPrefix("**"), line.hasSuffix("**"), !line.dropFirst(2).dropLast(2).contains("**") {
                attributes = Style.subheading   // a line that is all bold reads as a heading
            } else if line.hasPrefix("- ") || line.hasPrefix("* ") {
                line = "• " + line.dropFirst(2)
            }
            line = line.replacingOccurrences(of: "**", with: "").replacingOccurrences(of: "__", with: "")
            return NSAttributedString(string: line, attributes: attributes)
        }
    }

    enum Style {
        static var title: [NSAttributedString.Key: Any] { [.font: NSFont.boldSystemFont(ofSize: 18), .foregroundColor: NSColor.black] }
        static var heading: [NSAttributedString.Key: Any] { [.font: NSFont.boldSystemFont(ofSize: 14), .foregroundColor: NSColor.black] }
        static var subheading: [NSAttributedString.Key: Any] { [.font: NSFont.boldSystemFont(ofSize: 11), .foregroundColor: NSColor.black] }
        static var meta: [NSAttributedString.Key: Any] { [.font: NSFont.systemFont(ofSize: 10), .foregroundColor: NSColor.darkGray] }
        static var note: [NSAttributedString.Key: Any] { [.font: NSFont.systemFont(ofSize: 9.5), .foregroundColor: NSColor.gray] }
        static var body: [NSAttributedString.Key: Any] { [.font: NSFont.systemFont(ofSize: 10.5), .foregroundColor: NSColor.black] }
        static var speaker: [NSAttributedString.Key: Any] { [.font: NSFont.boldSystemFont(ofSize: 10.5), .foregroundColor: NSColor.black] }
        static var time: [NSAttributedString.Key: Any] { [.font: NSFont.monospacedDigitSystemFont(ofSize: 9.5, weight: .regular), .foregroundColor: NSColor.gray] }
        static var caption: [NSAttributedString.Key: Any] { [.font: NSFont.systemFont(ofSize: 8.5), .foregroundColor: NSColor.gray] }
    }
}

/// A tiny grayscale thumbnail of a frame, enough to tell screens apart.
enum Fingerprint {
    static let width = 32, height = 18

    static func make(_ url: URL) -> [UInt8]? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        var pixels = [UInt8](repeating: 0, count: width * height)
        let drawn: Bool = pixels.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height,
                                          bitsPerComponent: 8, bytesPerRow: width,
                                          space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            context.interpolationQuality = .medium
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        return drawn ? pixels : nil
    }

    /// Mean absolute difference per pixel; the maximum when either is missing.
    static func distance(_ a: [UInt8]?, _ b: [UInt8]?) -> Double {
        guard let a, let b, a.count == b.count, !a.isEmpty else { return 255 }
        var sum = 0
        for index in a.indices { sum += abs(Int(a[index]) - Int(b[index])) }
        return Double(sum) / Double(a.count)
    }
}

/// Lays paragraphs and images down US Letter pages, top to bottom, as real
/// PDF text (selectable, and extracted exactly by the AI reading it).
final class PDFWriter {
    static let page = CGRect(x: 0, y: 0, width: 612, height: 792)
    static let margin: CGFloat = 54

    private let context: CGContext
    private let graphics: NSGraphicsContext
    private var y: CGFloat = 0
    private var pageNumber = 0
    private var contentWidth: CGFloat { Self.page.width - 2 * Self.margin }
    private var bottom: CGFloat { Self.page.height - Self.margin }

    init(url: URL, title: String) throws {
        var box = Self.page
        let info: [CFString: Any] = [kCGPDFContextTitle: title, kCGPDFContextCreator: "Whiz"]
        guard let context = CGContext(url as CFURL, mediaBox: &box, info as CFDictionary) else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path])
        }
        self.context = context
        graphics = NSGraphicsContext(cgContext: context, flipped: true)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = graphics
        startPage()
    }

    func paragraph(_ text: NSAttributedString, spacing: CGFloat) {
        let height = ceil(text.boundingRect(with: CGSize(width: contentWidth, height: .greatestFiniteMagnitude),
                                            options: [.usesLineFragmentOrigin, .usesFontLeading]).height)
        if y + height > bottom, y > Self.margin { startPage() }
        text.draw(with: CGRect(x: Self.margin, y: y, width: contentWidth, height: height),
                  options: [.usesLineFragmentOrigin, .usesFontLeading])
        y += height + spacing
    }

    func image(_ image: NSImage, caption: String) {
        let size = image.size
        guard size.width > 0, size.height > 0 else { return }
        let maxHeight: CGFloat = 300
        let scale = min(contentWidth / size.width, maxHeight / size.height)
        let drawn = CGSize(width: size.width * scale, height: size.height * scale)
        let label = NSAttributedString(string: caption, attributes: AIExport.Style.caption)
        let total = 14 + drawn.height + 10
        if y + total > bottom, y > Self.margin { startPage() }
        y += 4
        label.draw(at: CGPoint(x: Self.margin, y: y))
        y += 12
        image.draw(in: CGRect(x: Self.margin, y: y, width: drawn.width, height: drawn.height),
                   from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil)
        y += drawn.height + 10
    }

    func space(_ points: CGFloat) { y += points }

    func finish() throws {
        context.endPDFPage()
        context.closePDF()
        NSGraphicsContext.restoreGraphicsState()
    }

    private func startPage() {
        if pageNumber > 0 { context.endPDFPage() }
        pageNumber += 1
        context.beginPDFPage(nil)
        // Top-left origin, y down, to match the flipped NSGraphicsContext.
        context.translateBy(x: 0, y: Self.page.height)
        context.scaleBy(x: 1, y: -1)
        // No page numbers: they would land between transcript lines in the
        // text a chat app extracts.
        y = Self.margin
    }
}
