import CryptoKit
import Foundation
import Vision

/// On-screen text from captured frames — Vision, directly, where the Python
/// pipeline's `ocr.py` (feat/llm-management branch) reaches Vision through
/// the `ocrmac` wrapper. The engine surface collapses to one implementation
/// natively, so only the behavior is ported: reading-order joining, the
/// normalization rules, byte-identical dedupe, and the batch contract that
/// never fails a transcription over one bad frame.
///
/// Mirrors the Python constants and rules exactly (they are the pipeline's
/// contract, not implementation details): the 0.02 line tolerance, the
/// single-character glyph-noise drop, the min/max character bounds with the
/// "…" truncation marker, and `new_screen_lines`, which the future AI pass
/// will use to diff frames (measured on the Python side: 74% of OCR lines
/// repeat frame-to-frame chrome — the diff is what keeps a screen
/// transcript from swamping the prompt).
enum FrameOCR {

    /// ocr.py:_LINE_TOL — annotations within 2% of the frame height group
    /// into one visual row.
    static let lineTolerance = 0.02

    private static let queue = DispatchQueue(label: "whiz.ocr", qos: .userInitiated)

    // MARK: - Single frame

    /// OCR one image, joining Vision's observations into reading order —
    /// `_ocr_apple` + `_join_apple`. Recognition level mirrors ocrmac's
    /// "accurate"; languages are the same "en-US" style hints.
    static func recognize(_ url: URL, languages: [String]) throws -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        if !languages.isEmpty {
            request.recognitionLanguages = languages
        }
        let handler = VNImageRequestHandler(url: url, options: [:])
        try handler.perform([request])

        // Vision's boundingBox origin is bottom-left, so higher on screen is
        // a *larger* y; the inverted bucket sorts rows top-to-bottom, then
        // left-to-right by x. Same math as _join_apple.
        var rows: [(bucket: Int, x: Double, text: String)] = []
        for observation in request.results ?? [] {
            guard let text = observation.topCandidates(1).first?.string
                .trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty
            else { continue }
            let box = observation.boundingBox
            rows.append((bucket: -Int((box.minY / lineTolerance).rounded()), x: box.minX, text))
        }
        rows.sort {
            if $0.bucket != $1.bucket { return $0.bucket < $1.bucket }
            if $0.x != $1.x { return $0.x < $1.x }
            return $0.text < $1.text
        }
        return rows.map(\.text).joined(separator: "\n")
    }

    // MARK: - Normalization (ocr.py:normalize)

    /// Tidy raw engine output into compact, token-friendly lines: collapse
    /// intra-line whitespace, drop blanks and single-character lines (toolbar
    /// buttons OCR as stray 'G', 'Q', '+' — never useful content), and
    /// enforce the min/max bounds. Below `minChars` is noise (empty); above
    /// `maxChars` truncates on a line boundary with a "…" marker so one
    /// pathological frame can't blow up a prompt.
    static func normalize(
        _ text: String?,
        minChars: Int = 0,
        maxChars: Int = 0
    ) -> String {
        guard let text else { return "" }
        var lines = text
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { line -> String in
                line.components(separatedBy: .whitespacesAndNewlines)
                    .filter { !$0.isEmpty }
                    .joined(separator: " ")
            }
        lines = lines.filter { $0.count > 1 }
        var out = lines.joined(separator: "\n")
        if minChars != 0, out.count < minChars {
            return ""
        }
        if maxChars != 0, out.count > maxChars {
            var kept: [String] = []
            var size = 0
            for line in lines {
                if size + line.count + 1 > maxChars { break }
                kept.append(line)
                size += line.count + 1
            }
            out = kept.joined(separator: "\n")
            if !out.isEmpty {
                out += "\n…"
            } else {
                out = lines.isEmpty ? "" : String(lines[0].prefix(maxChars)) + "…"
            }
        }
        return out
    }

    // MARK: - Frame diffing (ocr.py:new_screen_lines)

    /// Only the lines of `current` that weren't already on `previous`.
    /// Whole-frame dedupe barely fires on real recordings (clock, cursor),
    /// but 74% of *lines* repeat — window chrome, menu bar, sidebars. The
    /// line-level diff turns the screen track into "what changed", which is
    /// what the AI pass actually wants alongside the spoken words.
    static func newScreenLines(current: String, previous: String) -> String {
        if current.isEmpty { return "" }
        if previous.isEmpty { return current }
        let seen = Set(previous.split(separator: "\n", omittingEmptySubsequences: false))
        return current.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !seen.contains($0) }
            .joined(separator: "\n")
    }

    // MARK: - Batch driver (ocr.py:ocr_frames)

    /// ocr.py:OcrRun.
    struct Outcome: Sendable, Equatable {
        var texts: [String] = []
        var ok = 0
        var empty = 0
        var reused = 0
        var failed = 0
    }

    /// Content hash for byte-identical frame reuse. Python used 16-byte
    /// blake2b; the digest is only an internal cache key, so SHA-256 is an
    /// equivalent — it never leaves the pass.
    static func frameDigest(_ url: URL) -> String {
        guard let data = try? Data(contentsOf: url) else { return "" }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// OCR a list of frames, one normalized text per input (aligned by
    /// index). Byte-identical frames reuse the earlier result when `dedupe`
    /// is on — static screens repeat for long stretches, so this usually
    /// saves most of the pass, and stays conservative: only identical bytes
    /// are reused, never merely similar frames. A frame that fails yields
    /// "" and is counted in `failed`; the pass never throws — losing a
    /// transcription to an OCR error would be a bad trade (ocr.py:ocr_frames).
    ///
    /// Vision's perform is blocking, so the batch runs on a dedicated serial
    /// queue; `onProgress(done, total, reused)` fires as work advances.
    /// There is no mid-batch cancellation (Python parity) — the surrounding
    /// Task's cancellation lands between phases.
    static func frames(
        _ paths: [URL],
        languages: [String],
        minChars: Int = 0,
        maxChars: Int = 0,
        dedupe: Bool = true,
        onProgress: (@Sendable (Int, Int, Int) -> Void)? = nil
    ) async -> Outcome {
        let request = BatchRequest(
            paths: paths, languages: languages,
            minChars: minChars, maxChars: maxChars,
            dedupe: dedupe, onProgress: onProgress)
        return await withCheckedContinuation { continuation in
            queue.async {
                continuation.resume(returning: Self.runBatch(request))
            }
        }
    }

    private struct BatchRequest: Sendable {
        let paths: [URL]
        let languages: [String]
        let minChars: Int
        let maxChars: Int
        let dedupe: Bool
        let onProgress: (@Sendable (Int, Int, Int) -> Void)?
    }

    private static func runBatch(_ request: BatchRequest) -> Outcome {
        var run = Outcome()
        var seen: [String: String] = [:]
        let total = request.paths.count
        run.texts.reserveCapacity(total)

        for (index, path) in request.paths.enumerated() {
            let done = index + 1
            var text = ""
            var failedHere = false

            if !FileManager.default.fileExists(atPath: path.path) {
                run.failed += 1
                run.texts.append("")
                request.onProgress?(done, total, run.reused)
                continue
            }

            let digest = request.dedupe ? frameDigest(path) : ""
            if !digest.isEmpty, let cached = seen[digest] {
                text = cached
                run.reused += 1
            } else {
                do {
                    text = normalize(
                        try recognize(path, languages: request.languages),
                        minChars: request.minChars,
                        maxChars: request.maxChars)
                } catch {
                    run.failed += 1
                    failedHere = true
                    text = ""
                }
                if !digest.isEmpty {
                    seen[digest] = text
                }
            }

            if !text.isEmpty {
                run.ok += 1
            } else if !failedHere {
                run.empty += 1
            }
            run.texts.append(text)
            request.onProgress?(done, total, run.reused)
        }
        return run
    }
}