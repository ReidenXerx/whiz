"""Merge whisper transcription with diarization segments.

whisper-cli produces text segments with timestamps; sherpa-onnx produces
(start, end, speaker) segments. We assign each whisper segment to the
diarization speaker with the maximum temporal overlap, then emit labeled
SRT and a readable dialogue transcript.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from whiz.diarize import DiarSegment

# Speaker id -> human label: 0 -> "Speaker A", 1 -> "Speaker B", ...
_SPEAKER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def speaker_label(speaker_id: int) -> str:
    """0 -> 'Speaker A', 1 -> 'Speaker B', ..., 26 -> 'Speaker 26'."""
    if 0 <= speaker_id < len(_SPEAKER_LETTERS):
        return f"Speaker {_SPEAKER_LETTERS[speaker_id]}"
    return f"Speaker {speaker_id}"


def speakers_in_order(merged: list[tuple[WhisperSeg, str]]) -> list[str]:
    """Unique speaker labels in order of first appearance."""
    seen: list[str] = []
    for _, label in merged:
        if label not in seen:
            seen.append(label)
    return seen


def speakers_by_talk_time(merged: list[tuple[WhisperSeg, str]]) -> list[str]:
    """Unique speaker labels ordered by total speaking time (most first).

    Used by `--speakers-names Enric,Vadim,...`: names are assigned to speakers
    in this order so the most talkative speaker gets the first name. Falls
    back to order-of-appearance for ties (stable sort).
    """
    totals: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for i, (seg, label) in enumerate(merged):
        dur = max(0.0, seg.end - seg.start)
        totals[label] = totals.get(label, 0.0) + dur
        if label not in first_seen:
            first_seen[label] = i
    return sorted(totals, key=lambda lbl: (-totals[lbl], first_seen[lbl]))


def representative_quotes(
    merged: list[tuple[WhisperSeg, str]],
    max_chars: int = 140,
) -> dict[str, str]:
    """Pick the longest (most identifying) utterance per speaker.

    Longer utterances with more words are more recognizable than one-word
    replies like "Yeah", so we maximize word count (ties broken by char length).
    """
    best: dict[str, str] = {}
    best_words: dict[str, int] = {}
    for seg, label in merged:
        text = " ".join(seg.text.split())  # collapse whitespace
        if not text:
            continue
        n_words = len(text.split())
        if label not in best or n_words > best_words[label] or (
            n_words == best_words[label] and len(text) > len(best[label])
        ):
            best[label] = text
            best_words[label] = n_words
    # Truncate for display.
    return {k: (v if len(v) <= max_chars else v[: max_chars - 3] + "...") for k, v in best.items()}


def relabel(
    merged: list[tuple[WhisperSeg, str]],
    name_map: dict[str, str],
) -> list[tuple[WhisperSeg, str]]:
    """Return a new merged list with speaker labels replaced by names.

    Labels absent from name_map are left unchanged.
    """
    return [(seg, name_map.get(label, label)) for seg, label in merged]


@dataclass
class WhisperSeg:
    start: float
    end: float
    text: str


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Temporal overlap in seconds between [a_start,a_end] and [b_start,b_end]."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(
    whisper_segs: list[WhisperSeg],
    diar_segs: list[DiarSegment],
) -> list[tuple[WhisperSeg, str]]:
    """Assign a speaker label to each whisper segment by max overlap."""
    if not diar_segs:
        return [(seg, speaker_label(0)) for seg in whisper_segs]
    merged: list[tuple[WhisperSeg, str]] = []
    for wseg in whisper_segs:
        best_speaker = diar_segs[0].speaker
        best_overlap = 0.0
        for dseg in diar_segs:
            ov = _overlap(wseg.start, wseg.end, dseg.start, dseg.end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = dseg.speaker
        merged.append((wseg, speaker_label(best_speaker)))
    return merged


def _fmt_srt_time(t: float) -> str:
    """0.000 -> 00:00:00,000 (SRT timestamp)."""
    if t < 0:
        t = 0.0
    ms_total = int(round(t * 1000))
    hours, rem = divmod(ms_total, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_labeled_srt(merged: list[tuple[WhisperSeg, str]]) -> str:
    """Emit SRT with 'Speaker X: text' per cue."""
    lines: list[str] = []
    for i, (seg, label) in enumerate(merged, start=1):
        text = seg.text.strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}")
        lines.append(f"{label}: {text}")
        lines.append("")
    return "\n".join(lines)


def _fmt_clock(t: float) -> str:
    """0.000 -> 00:01:23 (readable HH:MM:SS)."""
    if t < 0:
        t = 0.0
    total = int(round(t))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_dialogue_txt(merged: list[tuple[WhisperSeg, str]]) -> str:
    """Emit a readable 'Speaker A (00:01:23): text' transcript.

    Consecutive segments from the same speaker are merged into one block.
    """
    lines: list[str] = []
    prev_label: str | None = None
    prev_start: float | None = None
    buf: list[str] = []

    def flush() -> None:
        if prev_label is not None and buf:
            lines.append(f"{prev_label} ({_fmt_clock(prev_start)}): {' '.join(buf)}")
        buf.clear()

    for seg, label in merged:
        text = seg.text.strip()
        if not text:
            continue
        if label != prev_label:
            flush()
            prev_label = label
            prev_start = seg.start
        buf.append(text)
    flush()
    return "\n\n".join(lines)


# ---------- whisper-cli output parsing ----------

def parse_whisper_json(path, json_full: bool = False) -> list[WhisperSeg]:
    """Parse a whisper-cli JSON output file into WhisperSeg list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    segs: list[WhisperSeg] = []
    # whisper-cli -oj produces {"transcription": [{"timestamps":{"from","to"}, "text":"..."}, ...]}
    # -ojf adds more fields but the segment shape is the same.
    transcription = data.get("transcription", [])
    for entry in transcription:
        ts = entry.get("timestamps", {})
        start = _ts_to_seconds(ts.get("from", "00:00:00,000"))
        end = _ts_to_seconds(ts.get("to", "00:00:00,000"))
        text = entry.get("text", "").strip()
        if text:
            segs.append(WhisperSeg(start=start, end=end, text=text))
    return segs


def parse_whisper_srt(path) -> list[WhisperSeg]:
    """Parse a whisper-cli SRT output file into WhisperSeg list."""
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", content.strip())
    segs: list[WhisperSeg] = []
    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        # Line 0 = index, line 1 = timestamps, line 2+ = text.
        ts_line = lines[1]
        m = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", ts_line
        )
        if not m:
            continue
        start = _ts_to_seconds(m.group(1))
        end = _ts_to_seconds(m.group(2))
        text = " ".join(lines[2:]).strip()
        if text:
            segs.append(WhisperSeg(start=start, end=end, text=text))
    return segs


def _ts_to_seconds(ts: str) -> float:
    """'00:01:23,456' or '00:01:23.456' -> 83.456."""
    ts = ts.strip().replace(".", ",")
    m = re.match(r"(\d+):(\d{2}):(\d{2}),(\d{3})", ts)
    if not m:
        return 0.0
    h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mi * 60 + s + ms / 1000.0