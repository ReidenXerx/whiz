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
from pathlib import Path

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

    Used by `--speakers-names Alice,Bob,...`: names are assigned to speakers
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
    words: list[dict] | None = None  # per-word timestamps if -ojf/verbose_json provides them


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


# ---------- HTML transcript ----------

# Deterministic per-speaker colors (cycle for >8 speakers).
_SPEAKER_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#16a085", "#d35400",
]


def _speaker_color(label: str) -> str:
    """Pick a stable color for a speaker label by its first-appearance index."""
    # Hash the label to a stable index so colors are consistent across runs.
    h = sum(ord(c) for c in label)
    return _SPEAKER_COLORS[h % len(_SPEAKER_COLORS)]


def speaker_palette(label: str) -> str:
    """Public alias for the HTML/terminal speaker color palette.

    Returns a hex color string (e.g. ``#e74c3c``) stable across runs for a
    given label. Used by both the HTML transcript and the terminal speaker
    tally so colors match between the two.
    """
    return _speaker_color(label)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def format_speakers_html(
    merged: list[tuple[WhisperSeg, str]],
    frames_dir: Path | None = None,
    title: str = "whiz transcript",
) -> str:
    """Emit a self-contained HTML transcript.

    Color-coded per-speaker transcript with timestamps. If ``frames_dir`` is
    given and contains ``segNNNN.jpg`` files (from a prior --screenshots run),
    frames are inlined as ``data:image/jpeg;base64`` URIs so the single HTML
    file is portable with every screenshot embedded — no external files needed.

    The page has a sticky header with the title, a speaker legend, and a
    live search box that filters cues by text or speaker. Clicking any frame
    thumbnail opens a fullscreen lightbox overlay (close with the button, the
    backdrop, or the Escape key).
    """
    import base64

    # Gather speakers in order of appearance for the legend.
    speakers_in_order: list[str] = []
    for _seg, label in merged:
        if label not in speakers_in_order:
            speakers_in_order.append(label)
    legend = [
        (label, _speaker_color(label)) for label in speakers_in_order
    ]

    css = """
:root {
  --bg: #fafafa;
  --card: #ffffff;
  --text: #1f2328;
  --muted: #6e7781;
  --border: #e4e7eb;
  --accent: #2f81f7;
  --shadow: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.04);
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
header.bar {
  position: sticky; top: 0; z-index: 20;
  background: rgba(255,255,255,.92);
  backdrop-filter: saturate(180%) blur(12px);
  -webkit-backdrop-filter: saturate(180%) blur(12px);
  border-bottom: 1px solid var(--border);
  padding: .65em 1em;
  display: flex; align-items: center; gap: 1em; flex-wrap: wrap;
}
header.bar h1 { font-size: 1.05em; margin: 0; font-weight: 650; letter-spacing: -.01em; }
header.bar .spacer { flex: 1; }
header.bar input.search {
  font: inherit; font-size: .9em; padding: .35em .6em .35em 1.8em;
  border: 1px solid var(--border); border-radius: 8px;
  width: 16em; max-width: 40vw; background: var(--card) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%236e7781' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='7'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg>") .55em .5em no-repeat; outline: none;
}
header.bar input.search:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(47,129,247,.18); }
.legend { display: flex; gap: .4em; flex-wrap: wrap; align-items: center; }
.legend .chip { font-size: .72em; padding: .15em .55em; border-radius: 999px; color: #fff; font-weight: 600; white-space: nowrap; }
main { max-width: 920px; margin: 0 auto; padding: 1em; }
.cue {
  display: flex; gap: .9em; align-items: flex-start;
  margin: .35em 0; padding: .7em .8em;
  background: var(--card);
  border: 1px solid var(--border); border-left: 3px solid var(--c, var(--border));
  border-radius: 10px; box-shadow: var(--shadow);
  transition: transform .06s ease, box-shadow .12s ease;
}
.cue:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,.08); }
.cue .frame {
  flex-shrink: 0; cursor: zoom-in; position: relative;
  border-radius: 8px; overflow: hidden; line-height: 0;
  border: 1px solid var(--border);
}
.cue .frame img { width: 180px; height: 116px; object-fit: cover; display: block; }
.cue .frame::after {
  content: ""; position: absolute; inset: 0; background: rgba(0,0,0,0); transition: background .12s;
}
.cue .frame:hover::after { background: rgba(0,0,0,.12); }
.cue .body { flex: 1; min-width: 0; }
.cue .meta { display: flex; align-items: baseline; gap: .55em; margin-bottom: .15em; }
.cue .ts { color: var(--muted); font-size: .8em; font-variant-numeric: tabular-nums; text-decoration: none; border-radius: 4px; padding: 0 .2em; }
.cue .ts:hover { background: var(--border); color: var(--text); }
.cue .speaker { font-weight: 650; font-size: .92em; }
.cue .text { font-size: .95em; white-space: pre-wrap; word-wrap: break-word; }
.cue.hidden { display: none; }
footer.foot { text-align: center; color: var(--muted); font-size: .8em; padding: 1.5em; }
/* Lightbox */
.lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.86); z-index: 100;
  display: none; align-items: center; justify-content: center; padding: 2em; }
.lightbox.open { display: flex; }
.lightbox img { max-width: 96vw; max-height: 92vh; border-radius: 8px; box-shadow: 0 10px 40px rgba(0,0,0,.5); }
.lightbox .close { position: absolute; top: 1em; right: 1.4em; background: rgba(255,255,255,.12); color: #fff;
  border: none; font-size: 1.6em; line-height: 1; width: 2.2em; height: 2.2em; border-radius: 50%; cursor: pointer; }
.lightbox .close:hover { background: rgba(255,255,255,.22); }
@media (max-width: 640px) {
  .cue .frame img { width: 120px; height: 78px; }
  header.bar input.search { width: 10em; }
}
"""

    js = r"""
(function () {
  var box = document.getElementById('lightbox');
  var boxImg = box.querySelector('img');
  function open(src) { boxImg.src = src; box.classList.add('open'); }
  function close() { box.classList.remove('open'); boxImg.src = ''; }
  document.querySelectorAll('.cue .frame').forEach(function (el) {
    el.addEventListener('click', function () {
      var img = el.querySelector('img'); if (img) open(img.src);
    });
  });
  box.addEventListener('click', function (e) { if (e.target === box || e.target.classList.contains('close')) close(); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
  var search = document.getElementById('search');
  if (search) {
    search.addEventListener('input', function () {
      var q = search.value.trim().toLowerCase();
      document.querySelectorAll('.cue').forEach(function (cue) {
        var hay = (cue.textContent || '').toLowerCase();
        cue.classList.toggle('hidden', q && hay.indexOf(q) === -1);
      });
    });
  }
})();
"""

    parts: list[str] = []
    parts.append("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">")
    parts.append(f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
    parts.append(f"<title>{_html_escape(title)}</title>")
    parts.append(f"<style>{css}</style>")
    parts.append("</head>\n<body>")

    parts.append("<header class=\"bar\">")
    parts.append(f"<h1>{_html_escape(title)}</h1>")
    if legend:
        parts.append('<div class="legend">')
        for label, color in legend:
            parts.append(
                f'<span class="chip" style="background:{color}">{_html_escape(label)}</span>'
            )
        parts.append('</div>')
    parts.append('<div class="spacer"></div>')
    parts.append(
        '<input id="search" class="search" type="search" placeholder="Search transcript&hellip;" aria-label="Search transcript">'
    )
    parts.append('</header>')

    parts.append('<main>')
    cue_count = 0
    has_frame = False
    for i, (seg, label) in enumerate(merged, start=1):
        text = seg.text.strip()
        if not text:
            continue
        cue_count += 1
        color = _speaker_color(label)
        ts = _fmt_clock(seg.start)
        parts.append(f'<div class="cue" style="--c:{color}">')
        # Inline frame thumbnail if it exists (clickable -> lightbox).
        if frames_dir is not None:
            frame_path = frames_dir / f"seg{i:04d}.jpg"
            if frame_path.exists():
                has_frame = True
                b64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
                parts.append(
                    f'<div class="frame"><img src="data:image/jpeg;base64,{b64}" alt="frame {i}"></div>'
                )
        parts.append('<div class="body">')
        parts.append('<div class="meta">')
        parts.append(f'<a class="ts" href="#cue-{i}" id="cue-{i}">{ts}</a>')
        parts.append(f'<span class="speaker" style="color:{color}">{_html_escape(label)}</span>')
        parts.append('</div>')  # .meta
        parts.append(f'<div class="text">{_html_escape(text)}</div>')
        parts.append('</div></div>')  # .body and .cue
    parts.append('</main>')

    if cue_count:
        parts.append(f'<div class="foot">{cue_count} cue(s)</div>')

    # Lightbox overlay (only when at least one frame was inlined, so a
    # frames-less transcript emits no <img> tags at all).
    if has_frame:
        parts.append('<div class="lightbox" id="lightbox" aria-hidden="true">')
        parts.append('<button class="close" aria-label="Close">&times;</button>')
        parts.append('<img alt="fullscreen frame">')
        parts.append('</div>')
        parts.append(f'<script>{js}</script>')

    parts.append('</body>\n</html>')
    return "\n".join(parts)


# ---------- whisper-cli output parsing ----------

def parse_whisper_json(path, json_full: bool = False) -> list[WhisperSeg]:
    """Parse a whisper-cli JSON output file into WhisperSeg list.

    Handles both -oj (standard) and -ojf (json-full) formats. Both use a
    ``transcription`` array with ``timestamps``/``text`` per segment. The
    json-full format additionally has ``offsets``; per-word timestamps (when
    present, e.g. from verbose_json-style ``words`` arrays) are captured on
    the optional ``words`` field for HTML karaoke-style highlighting.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    segs: list[WhisperSeg] = []
    # whisper-cli -oj/-ojf produces {"transcription": [{"timestamps":{"from","to"}, "text":"..."}, ...]}
    # -ojf adds more fields but the segment shape is the same.
    transcription = data.get("transcription", [])
    for entry in transcription:
        ts = entry.get("timestamps", {})
        start = _ts_to_seconds(ts.get("from", "00:00:00,000"))
        end = _ts_to_seconds(ts.get("to", "00:00:00,000"))
        text = entry.get("text", "").strip()
        words = entry.get("words")  # present in verbose_json-style output
        if text:
            segs.append(WhisperSeg(start=start, end=end, text=text, words=words))
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