import AVFoundation
import CoreGraphics
import Foundation
import ImageIO

/// One frame per transcribed segment — the Swift half of `whiz/screenshots.py`.
///
/// For video inputs, capturing what was on screen alongside each segment is
/// what lets the transcript + frames feed a vision LLM later, and what the
/// HTML transcript (phase 4 of the Python pipeline) inlines. One JPEG per
/// segment, taken at `segment.start`, named `segNNNN.jpg` to match the 1-based
/// segment index so transcript and frames join cleanly.
///
/// Two deliberate divergences from the ffmpeg implementation, both serving the
/// stated intent in `screenshots.py` ("Downscaling keeps frame sizes
/// reasonable for LLM context windows") rather than the letter:
/// - Python scales with `scale=1280:-1` — width forced to 1280, so a portrait
///   video becomes a giant 1280×2276 image. `maximumSize` here caps the
///   *larger* dimension at `width` instead: landscape gets the same
///   1280×720 Python produces; portrait gets 720×1280, which is the size the
///   intent was after.
/// - Seek accuracy: ffmpeg's seek-before-input and AVAssetImageGenerator with
///   zero tolerance are the same strategy (seek to the keyframe at-or-before
///   the target, decode forward), so timestamps are as exact as the keyframe
///   spacing allows on both.
///
/// Per-frame failures are non-fatal and mirrored from Python: a segment whose
/// capture fails keeps an empty `frame` field so the manifest still aligns by
/// index (`screenshots.py:_extract_one` returns False; the entry is kept).
enum FrameExtractor {

    /// One row of the frames manifest — `screenshots.py:FrameEntry`.
    struct Entry: Sendable, Equatable {
        let index: Int
        let start: Double
        let end: Double
        let speaker: String
        let text: String
        /// Filename within the frames directory; empty if capture failed.
        let frame: String
    }

    /// True if the media file has a video track — "auto-on for video" is
    /// semantic, not extension-based (cli.py:707 gates on `needs_extraction`).
    static func hasVideoTrack(_ url: URL) async -> Bool {
        let tracks = try? await AVURLAsset(url: url).loadTracks(withMediaType: .video)
        return !(tracks ?? []).isEmpty
    }

    /// Extract one JPEG per segment at its start timestamp into `framesDir`,
    /// which is created. Returns one entry per segment, in order, regardless
    /// of capture failures — the manifest aligns by index. Empty input yields
    /// no entries; an input without a video track yields none either, which
    /// the caller distinguishes with `hasVideoTrack` for logging.
    static func extractFrames(
        video: URL,
        segments: [LabeledSegment],
        into framesDir: URL,
        width: Int = 1280,
        onProgress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> [Entry] {
        guard !segments.isEmpty else { return [] }
        guard await hasVideoTrack(video) else { return [] }

        let asset = AVURLAsset(url: video)
        let generator = AVAssetImageGenerator(asset: asset)
        // Honour rotation metadata the way ffmpeg's autorotate does.
        generator.appliesPreferredTrackTransform = true
        // Exact seeks, matching ffmpeg's decode-forward-to-target accuracy.
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        if width > 0 {
            generator.maximumSize = CGSize(width: width, height: width)
        }

        try FileManager.default.createDirectory(at: framesDir, withIntermediateDirectories: true)

        var entries: [Entry] = []
        entries.reserveCapacity(segments.count)
        for (offset, pair) in segments.enumerated() {
            if Task.isCancelled {
                throw CancellationError()
            }

            // Python collapses whitespace before storing (`" ".join(text.split())`).
            let text = pair.segment.text
                .components(separatedBy: .whitespacesAndNewlines)
                .filter { !$0.isEmpty }
                .joined(separator: " ")

            var frameName = ""
            if let image = try? generator.copyCGImage(
                at: CMTime(seconds: max(0.0, pair.segment.start), preferredTimescale: 600),
                actualTime: nil)
            {
                let name = String(format: "seg%04d.jpg", offset + 1)
                if writeJPEG(image, to: framesDir.appendingPathComponent(name), quality: 0.9) {
                    frameName = name
                }
            }

            entries.append(Entry(
                index: offset + 1,
                start: pair.segment.start,
                end: pair.segment.end,
                speaker: pair.speaker,
                text: text,
                frame: frameName))

            onProgress?(Double(offset + 1) / Double(segments.count))
        }
        return entries
    }

    /// Write the frames manifest — `screenshots.py:write_manifest`'s payload
    /// exactly: version, the frames directory *name*, count, and one row per
    /// segment in Python's key order. Paths only, never bytes — the manifest
    /// stays small and re-runnable while the HTML artifact inlines frames as
    /// base64. Shape-compatible with Python's reader (`load_manifest`), not
    /// byte-stable with `json.dumps` — non-ASCII stays UTF-8 where Python
    /// would emit `\uXXXX`.
    static func writeManifest(_ entries: [Entry], framesDir: URL, to path: URL) throws {
        var json = "{\n"
        json += "  \"version\": 1,\n"
        json += "  \"frames_dir\": \(escape(framesDir.lastPathComponent)),\n"
        json += "  \"count\": \(entries.count),\n"
        json += "  \"segments\": ["
        for (offset, entry) in entries.enumerated() {
            json += offset == 0 ? "\n" : ",\n"
            json += "    {\n"
            json += "      \"index\": \(entry.index),\n"
            json += "      \"start\": \(entry.start),\n"
            json += "      \"end\": \(entry.end),\n"
            json += "      \"speaker\": \(escape(entry.speaker)),\n"
            json += "      \"text\": \(escape(entry.text)),\n"
            json += "      \"frame\": \(escape(entry.frame))\n"
            json += "    }"
        }
        json += entries.isEmpty ? "]\n" : "\n  ]\n"
        json += "}"

        guard let data = json.data(using: .utf8) else {
            throw FrameExtractorError.writeFailed(path.path)
        }
        try data.write(to: path, options: .atomic)
    }

    // MARK: - Internals

    /// Encode one JPEG at `url` — ffmpeg `-q:v 2`'s high-quality end, ~0.9.
    private static func writeJPEG(_ image: CGImage, to url: URL, quality: CGFloat) -> Bool {
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, "public.jpeg" as CFString, 1, nil)
        else { return false }
        let options = [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary
        CGImageDestinationAddImage(destination, image, options)
        return CGImageDestinationFinalize(destination)
    }

    /// JSON string escaping: quotes, backslashes, and the whitespace/control
    /// characters that must not appear raw inside a JSON string. Richer than
    /// `TranscriptFormatter.escapeJSON` (which stays minimal for `-oj`
    /// parity) because manifest text can contain anything the transcript did.
    private static func escape(_ text: String) -> String {
        var out = "\""
        for character in text.unicodeScalars {
            switch character {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if character.value < 0x20 {
                    out += String(format: "\\u%04x", character.value)
                } else {
                    out.unicodeScalars.append(character)
                }
            }
        }
        return out + "\""
    }
}

enum FrameExtractorError: LocalizedError, Sendable, Equatable {
    case writeFailed(String)

    var errorDescription: String? {
        switch self {
        case .writeFailed(let path):
            return "Could not write frames manifest at \(path)."
        }
    }
}