import Foundation

/// The speaker-aware half of `whiz/merge.py` — assignment, relabeling and the
/// labeled output formats.
///
/// Everything here is pure and label-driven: diarization's job is to produce
/// `DiarSegment`s; this layer turns segments + diarization into labeled
/// transcripts without knowing where either came from. The tests mirror
/// `tests/test_merge.py` fixture-for-fixture, so the two implementations are
/// held to the same behavior contract — the same discipline that pinned the
/// tuning constants (NS-1's pattern, extended to the merge layer).
///
/// `DiarSegment` mirrors `whiz/diarize.py`'s dataclass: speaker is an int
/// cluster id; `speakerLabel` renders it as "Speaker A/B/…" exactly like the
/// Python side.
struct DiarSegment: Sendable, Equatable {
    let start: Double
    let end: Double
    let speaker: Int
}

/// A transcript segment carrying its resolved speaker label — the Swift shape
/// of merge.py's `(WhisperSeg, label)` tuples, shared with the frames manifest.
struct LabeledSegment: Sendable, Equatable {
    let segment: WhisperBatchTranscriber.Segment
    let speaker: String
}

enum LabeledTranscript {

    private static let speakerLetters = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    /// merge.py:speaker_label — 0 → "Speaker A" … 25 → "Speaker Z",
    /// 26 → "Speaker 26" (also for negative ids).
    static func speakerLabel(_ id: Int) -> String {
        if 0 <= id, id < speakerLetters.count {
            return "Speaker \(speakerLetters[id])"
        }
        return "Speaker \(id)"
    }

    /// Unique speaker labels in order of first appearance.
    static func speakersInOrder(_ merged: [LabeledSegment]) -> [String] {
        var seen: [String] = []
        for entry in merged where !seen.contains(entry.speaker) {
            seen.append(entry.speaker)
        }
        return seen
    }

    /// Unique speaker labels by total speaking time, most first — the order
    /// `--speakers-names` assigns names in, so the most talkative speaker gets
    /// the first name. Ties fall back to order of appearance (Python's sort is
    /// made deterministic by keying on first-seen index, mirrored here).
    static func speakersByTalkTime(_ merged: [LabeledSegment]) -> [String] {
        var totals: [String: Double] = [:]
        var firstSeen: [String: Int] = [:]
        for (index, entry) in merged.enumerated() {
            let duration = max(0.0, entry.segment.end - entry.segment.start)
            totals[entry.speaker, default: 0] += duration
            if firstSeen[entry.speaker] == nil {
                firstSeen[entry.speaker] = index
            }
        }
        return totals.keys.sorted {
            let timeA = totals[$0] ?? 0, timeB = totals[$1] ?? 0
            if timeA != timeB { return timeA > timeB }
            return (firstSeen[$0] ?? 0) < (firstSeen[$1] ?? 0)
        }
    }

    /// The longest (most identifying) utterance per speaker — more words
    /// recognize better than one-word replies. Word-count wins, character
    /// length breaks ties; displayed quotes truncate to `maxChars`.
    static func representativeQuotes(
        _ merged: [LabeledSegment],
        maxChars: Int = 140
    ) -> [String: String] {
        var best: [String: String] = [:]
        var bestWords: [String: Int] = [:]
        for entry in merged {
            let text = entry.segment.text
                .components(separatedBy: .whitespacesAndNewlines)
                .filter { !$0.isEmpty }
                .joined(separator: " ")
            if text.isEmpty { continue }
            let wordCount = text
                .components(separatedBy: .whitespaces)
                .filter { !$0.isEmpty }
                .count
            let currentBest = best[entry.speaker]
            let currentWords = bestWords[entry.speaker] ?? 0
            let isBetter = currentBest == nil
                || wordCount > currentWords
                || (wordCount == currentWords && text.count > (currentBest?.count ?? 0))
            if isBetter {
                best[entry.speaker] = text
                bestWords[entry.speaker] = wordCount
            }
        }
        return best.mapValues { quote in
            quote.count <= maxChars ? quote : String(quote.prefix(maxChars - 3)) + "..."
        }
    }

    /// Replace speaker labels with names; labels absent from the map are
    /// left unchanged.
    static func relabel(
        _ merged: [LabeledSegment],
        _ nameMap: [String: String]
    ) -> [LabeledSegment] {
        merged.map {
            LabeledSegment(segment: $0.segment, speaker: nameMap[$0.speaker] ?? $0.speaker)
        }
    }

    /// merge.py:assign_speakers — label each whisper segment by whichever
    /// diarization segment it overlaps most. No diarization at all means
    /// every segment falls back to the first speaker's label, exactly like
    /// Python's diar_segs[0].speaker base case.
    static func assignSpeakers(
        segments: [WhisperBatchTranscriber.Segment],
        diar: [DiarSegment]
    ) -> [LabeledSegment] {
        guard let fallback = diar.first else {
            return segments.map {
                LabeledSegment(segment: $0, speaker: speakerLabel(0))
            }
        }
        var merged: [LabeledSegment] = []
        merged.reserveCapacity(segments.count)
        for segment in segments {
            var bestSpeaker = fallback.speaker
            var bestOverlap = 0.0
            for diarEntry in diar {
                let overlap = max(
                    0.0,
                    min(segment.end, diarEntry.end) - max(segment.start, diarEntry.start))
                if overlap > bestOverlap {
                    bestOverlap = overlap
                    bestSpeaker = diarEntry.speaker
                }
            }
            merged.append(LabeledSegment(segment: segment, speaker: speakerLabel(bestSpeaker)))
        }
        return merged
    }

    // MARK: - Output formats

    /// SRT with "Speaker X: text" per cue — merge.py:format_labeled_srt.
    /// Empty-text segments are skipped but still consume their cue number,
    /// matching Python's enumerate-then-continue numbering.
    static func formatLabeledSRT(_ merged: [LabeledSegment]) -> String {
        var lines: [String] = []
        for (index, entry) in merged.enumerated() {
            let text = entry.segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { continue }
            lines.append("\(index + 1)")
            lines.append(
                "\(TranscriptFormatter.srtTimestamp(entry.segment.start)) --> "
                    + "\(TranscriptFormatter.srtTimestamp(entry.segment.end))")
            lines.append("\(entry.speaker): \(text)")
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    /// Readable "Speaker A (00:01:23): text" transcript —
    /// merge.py:format_dialogue_txt. Consecutive same-speaker segments merge
    /// into one block carrying the first segment's timestamp.
    static func formatDialogueTXT(_ merged: [LabeledSegment]) -> String {
        var lines: [String] = []
        var previousSpeaker: String?
        var previousStart: Double?
        var buffer: [String] = []

        func flush() {
            if let speaker = previousSpeaker, let start = previousStart, !buffer.isEmpty {
                lines.append("\(speaker) (\(TranscriptFormatter.clock(start))): \(buffer.joined(separator: " "))")
            }
            buffer.removeAll()
        }

        for entry in merged {
            let text = entry.segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { continue }
            if entry.speaker != previousSpeaker {
                flush()
                previousSpeaker = entry.speaker
                previousStart = entry.segment.start
            }
            buffer.append(text)
        }
        flush()
        return lines.joined(separator: "\n\n")
    }
}