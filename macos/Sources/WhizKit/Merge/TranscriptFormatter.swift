import Foundation

/// Pure transcript formatting for the batch pipeline — the Swift half of
/// `whiz/merge.py`'s output layer, pinned in `TranscriptFormatterTests`.
///
/// Byte-parity notes, deliberately stated rather than implied:
/// - Timestamps are `merge.py:_fmt_srt_time` (round-to-ms, comma form), which
///   is the same shape `whisper.cpp`'s `to_timestamp(t, comma)` emits for
///   centisecond `t0/t1`.
/// - SRT text is trimmed, like `merge.py:format_labeled_srt` strips it — the
///   raw `whisper-cli` SRT writes the segment text untrimmed with its leading
///   space. The trimmed form is what whiz's own merge layer produces, so it
///   is the convention this port follows.
/// - The JSON is the `-oj` *contract* subset: everything
///   `merge.py:parse_whisper_json` reads (`transcription` with
///   `timestamps`/`text`, plus the `offsets` it can use). It is not a
///   byte-identical `whisper-cli -oj` file — the metadata sections
///   (`systeminfo`, `model`, `params`) are absent. Enough for `whiz merge
///   --resume` interop; byte-stable output fixtures are a decision to make
///   when the merge port grows (see ARCHITECTURE.md, P2 acceptance bar).
enum TranscriptFormatter {

    /// `merge.py:_fmt_srt_time`: seconds → "00:01:23,456".
    static func srtTimestamp(_ t: Double) -> String {
        let clamped = max(0.0, t)
        let msTotal = Int((clamped * 1000).rounded())
        let hours = msTotal / (60 * 60 * 1000)
        let minutes = (msTotal % (60 * 60 * 1000)) / (60 * 1000)
        let seconds = (msTotal % (60 * 1000)) / 1000
        let millis = msTotal % 1000
        return String(format: "%02d:%02d:%02d,%03d", hours, minutes, seconds, millis)
    }

    /// `merge.py:_fmt_clock`: seconds → "00:01:23" (whole seconds).
    static func clock(_ t: Double) -> String {
        let clamped = max(0.0, t)
        let total = Int(clamped.rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let seconds = total % 60
        return String(format: "%02d:%02d:%02d", hours, minutes, seconds)
    }

    /// Numbered SRT cues, one per segment — `whisper.cpp` cli's `output_srt`
    /// shape with `merge.py`'s trimming.
    static func srt(_ segments: [WhisperBatchTranscriber.Segment]) -> String {
        var out = ""
        for (index, segment) in segments.enumerated() {
            out += "\(index + 1)\n"
            out += "\(srtTimestamp(segment.start)) --> \(srtTimestamp(segment.end))\n"
            out += "\(segment.text)\n\n"
        }
        return out
    }

    /// The `-oj` subset: `{"transcription": [...]}` with tab indentation,
    /// matching the field order `output_json` emits (timestamps, offsets, text).
    static func json(_ segments: [WhisperBatchTranscriber.Segment]) -> String {
        var out = "{\n\t\"transcription\": ["
        for (index, segment) in segments.enumerated() {
            out += index == 0 ? "\n" : ",\n"
            out += "\t\t{\n"
            out += "\t\t\t\"timestamps\": {\n"
            out += "\t\t\t\t\"from\": \"\(srtTimestamp(segment.start))\",\n"
            out += "\t\t\t\t\"to\": \"\(srtTimestamp(segment.end))\"\n"
            out += "\t\t\t},\n"
            out += "\t\t\t\"offsets\": {\n"
            out += "\t\t\t\t\"from\": \(ms(segment.start)),\n"
            out += "\t\t\t\t\"to\": \(ms(segment.end))\n"
            out += "\t\t\t},\n"
            out += "\t\t\t\"text\": \"\(escapeJSON(segment.text))\"\n"
            out += "\t\t}"
        }
        out += segments.isEmpty ? "]\n" : "\n\t]\n"
        out += "}\n"
        return out
    }

    private static func ms(_ seconds: Double) -> Int {
        Int((max(0.0, seconds) * 1000).rounded())
    }

    /// `cli.cpp:escape_double_quotes_and_backslashes` — the only two
    /// characters that need escaping for these JSON strings.
    static func escapeJSON(_ text: String) -> String {
        text
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    /// Log-window line for a recognized segment: `[00:41 → 00:55]  text`.
    static func segmentLogLine(_ segment: WhisperBatchTranscriber.Segment) -> String {
        "[\(clock(segment.start)) → \(clock(segment.end))]  \(segment.text)"
    }
}