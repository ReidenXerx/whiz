"""Language-aware transcription, and the guard against repetition loops.

whisper-cli's ``-l auto`` detects the language ONCE, from the first 30 s,
and decodes the whole file in it. A recording that switches language (an
interview that opens in Ukrainian and moves to English) then comes out
with the English rendered, or translated, into Ukrainian. That mismatched
decode is also what tips whisper into a repetition loop: one phrase
emitted hundreds of times while the real speech is lost.

So, when the language is "auto":

1. :func:`windows_for` cuts the audio into windows: the speaker turns when
   diarization ran, otherwise ~30 s stretches cut at a quiet moment.
   :func:`detect_languages` asks whisper-cli for each window's language
   (``-dl``: one process, an encoder pass per window, no decoding).
2. :func:`smooth` settles the guesses (an unsure or tiny window takes its
   neighbours' language, a short stray island is absorbed) and
   :func:`spans_for` turns them into spans: stretches of one language that
   cover the whole recording, cut at the quietest moment between them.
3. One language: the usual single whisper-cli run, told that language
   instead of "auto". Several: :func:`transcribe_spans` decodes each span
   in its own language and stitches the results into the one whisper JSON
   (and srt/txt/...) every later step reads, timestamps on the original
   timeline.

After every run, :func:`find_loops` looks for the loop signature (the same
line three or more times in a row) and :func:`repair_loops` decodes that
stretch again on its own with no carried-over context (``-mc 0``); a loop
that survives is collapsed to one line, and reported.
"""

from __future__ import annotations

import array
import contextlib
import json
import re
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from whiz import audio as aud

RATE = 16000

# Windows: whisper detects language from (up to) 30 s of audio.
WINDOW_SECONDS = 30.0
# A window shorter than this is too little speech to name its language.
MIN_WINDOW_SECONDS = 2.0
# Below this probability a window's guess is not trusted on its own.
MIN_CONFIDENCE = 0.7
# A language needs this much confident speech IN ONE STRETCH to get spans
# of its own. Scattered windows are misdetections (Ukrainian heard as
# Russian here and there, an English term in Ukrainian speech); a real
# switch is sustained.
MIN_LANGUAGE_SECONDS = 20.0
# Around each switch, the two windows either side are checked again in
# pieces this long, to place the cut within ~10 s rather than ~30 s.
# (Each detection costs a full encoder pass, whatever the length, so the
# whole recording is not checked this finely.)
REFINE_SECONDS = 10.0
# A run of windows shorter than this, between two runs of one language,
# joins them.
ISLAND_SECONDS = 10.0

# Run is the whisper-cli runner: (argv, progress callback) -> return code.
Run = Callable[[list[str], Callable[[float], None]], int]


@dataclass
class Window:
    start: float
    end: float
    language: str | None = None
    confidence: float = 0.0
    # Seconds of speech in it, per diarization; None when diarization did
    # not run.
    speech: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Span:
    start: float
    end: float
    language: str


# ---------- audio ----------

class Workspace:
    """A temp dir, and the recording as 16 kHz mono 16-bit PCM to cut clips
    from (the extracted WAV itself when it already is, else a converted
    copy)."""

    def __init__(self, pcm: Path, tmp: Path) -> None:
        self.pcm = pcm
        self.tmp = tmp
        with wave.open(str(pcm), "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, RATE):
                raise ValueError(f"{pcm} is not 16 kHz mono 16-bit PCM")
            self.frames = w.getnframes()
        self._count = 0

    @classmethod
    def create(cls, source: Path, ffmpeg: Callable[[], str], stack: contextlib.ExitStack) -> "Workspace":
        """``ffmpeg`` resolves the converter, asked only when ``source`` needs
        converting."""
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="whiz-lang-")))
        pcm = source if _is_pcm16k(source) else aud.extract_audio(source, ffmpeg(), dest_dir=tmp)
        return cls(pcm, tmp)

    @property
    def duration(self) -> float:
        return self.frames / RATE

    def read(self, start: float, end: float) -> bytes:
        a = max(0, min(self.frames, int(start * RATE)))
        b = max(a, min(self.frames, int(end * RATE)))
        with wave.open(str(self.pcm), "rb") as w:
            w.setpos(a)
            return w.readframes(b - a)

    def clip(self, start: float, end: float, name: str = "clip") -> Path:
        """Write [start, end) as its own WAV and return its path."""
        self._count += 1
        path = self.tmp / f"{name}{self._count:05d}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(self.read(start, end))
        return path

    def quietest(self, around: float, radius: float) -> float:
        """The middle of the quietest 300 ms within ``radius`` of ``around``:
        where a cut splits no word."""
        lo = max(0.0, around - radius)
        hi = min(self.duration, around + radius)
        samples = array.array("h", self.read(lo, hi))
        if sys.byteorder == "big":
            samples.byteswap()
        step = RATE // 100  # 10 ms
        energy = [
            sum(s * s for s in samples[i:i + step])
            for i in range(0, len(samples) - step + 1, step)
        ]
        span = 30  # 300 ms
        if len(energy) <= span:
            return min(max(around, lo), hi)
        total = sum(energy[:span])
        best, best_at = total, 0
        for i in range(span, len(energy)):
            total += energy[i] - energy[i - span]
            if total < best:
                best, best_at = total, i - span + 1
        return lo + (best_at + span / 2) * 0.01


def _is_pcm16k(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
    except (wave.Error, EOFError, OSError):
        return False


# ---------- planning ----------

def windows_for(
    duration: float,
    diar_segments: Sequence | None = None,
    quietest: Callable[[float, float], float] | None = None,
) -> list[Window]:
    """The stretches to detect a language for: 30 s tiles of the whole
    recording, each cut moved to a quiet moment within 3 s.

    Fixed tiles, not speaker turns: every detection costs a full encoder
    pass (~1 s on an M1 Pro) however short the clip, so a 50-minute call
    is ~100 of them rather than one per turn; :func:`refine` then places
    each switch to ~10 s. With diarization, each tile also carries how much
    speech it holds, so silent ones are not asked about.
    """
    cuts = [0.0]
    t = WINDOW_SECONDS
    while t < duration - MIN_WINDOW_SECONDS:
        cut = quietest(t, 3.0) if quietest else t
        if cut > cuts[-1] + MIN_WINDOW_SECONDS:
            cuts.append(cut)
        t += WINDOW_SECONDS
    cuts.append(duration)
    windows = [Window(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
    if diar_segments is not None:
        for w in windows:
            w.speech = sum(
                max(0.0, min(w.end, seg.end) - max(w.start, seg.start)) for seg in diar_segments
            )
    return windows


_PROCESSING_RE = re.compile(r"processing '(.+?)' \(")
_DETECTED_RE = re.compile(r"auto-detected language: ([a-z]{2,3}) \(p = ([0-9.]+)\)")


def parse_detections(lines: Iterable[str], clips: Sequence[Path]) -> list[tuple[str, float] | None]:
    """Map whisper-cli ``-dl`` output to one (language, p) per clip."""
    found: dict[str, tuple[str, float]] = {}
    current = None
    for line in lines:
        if m := _PROCESSING_RE.search(line):
            current = m.group(1)
        elif (m := _DETECTED_RE.search(line)) and current is not None:
            found[current] = (m.group(1), float(m.group(2)))
            current = None
    return [found.get(str(c)) for c in clips]


def detect_languages(
    whisper_cli: str,
    model: str,
    threads: str,
    clips: Sequence[Path],
    progress: Callable[[float], None] | None = None,
) -> list[tuple[str, float] | None]:
    """Each clip's language and its probability, from one whisper-cli run.

    No ``-np``: that flag also silences the detection lines this parses.
    """
    cmd = [whisper_cli, "-m", model, "-t", threads, "-l", "auto", "-dl"]
    for c in clips:
        cmd += ["-f", str(c)]
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert proc.stdout is not None
    done = 0
    try:
        for line in proc.stdout:
            lines.append(line)
            if _DETECTED_RE.search(line):
                done += 1
                if progress and clips:
                    progress(100.0 * done / len(clips))
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if proc.wait() != 0:
        tail = "".join(lines[-5:]).strip()
        raise RuntimeError(f"language detection failed (exit {proc.returncode}): {tail}")
    return parse_detections(lines, clips)


def smooth(windows: Sequence[Window]) -> list[str] | None:
    """One language per window, or None when nothing was detected with
    confidence. Unsure windows inherit from their neighbours; a language with
    too little confident speech is treated as a misdetection; a short run
    between two runs of one language joins them."""
    langs: list[str | None] = [
        w.language if w.language and w.confidence >= MIN_CONFIDENCE and w.duration >= MIN_WINDOW_SECONDS
        else None
        for w in windows
    ]
    # The longest stretch of each language: its confident windows in a row,
    # unsure ones between them not breaking it.
    best: dict[str, float] = {}
    run_lang, run_total = None, 0.0
    for w, lang in zip(windows, langs):
        if not lang:
            continue
        if lang != run_lang:
            run_lang, run_total = lang, 0.0
        run_total += w.duration
        best[lang] = max(best.get(lang, 0.0), run_total)
    keep = {lang for lang, longest in best.items() if longest >= MIN_LANGUAGE_SECONDS}
    if not keep:
        # Nothing reached the bar: trust whatever held longest.
        if not best:
            return None
        keep = {max(best, key=best.get)}
    langs = [lang if lang in keep else None for lang in langs]
    # Unsure windows take the language before them (the first ones, after).
    first = next(lang for lang in langs if lang)
    filled: list[str] = []
    for lang in langs:
        filled.append(lang or (filled[-1] if filled else first))
    # Absorb short islands: a run shorter than ISLAND_SECONDS whose
    # neighbours on both sides share a language.
    runs = _runs(filled)
    for k in range(1, len(runs) - 1):
        lang, a, b = runs[k]
        before, after = runs[k - 1][0], runs[k + 1][0]
        if before == after != lang and sum(windows[i].duration for i in range(a, b)) < ISLAND_SECONDS:
            for i in range(a, b):
                filled[i] = before
    return filled


def _runs(langs: Sequence[str]) -> list[tuple[str, int, int]]:
    """(language, first index, end index) for each run of equal entries."""
    runs: list[tuple[str, int, int]] = []
    for i, lang in enumerate(langs):
        if runs and runs[-1][0] == lang:
            runs[-1] = (lang, runs[-1][1], i + 1)
        else:
            runs.append((lang, i, i + 1))
    return runs


def split(window: Window, size: float = REFINE_SECONDS) -> list[Window]:
    """``window`` in equal pieces of at most ``size`` seconds."""
    pieces = max(1, int(-(-window.duration // size)))
    step = window.duration / pieces
    return [Window(window.start + k * step, window.start + (k + 1) * step) for k in range(pieces)]


def refine(
    windows: Sequence[Window],
    langs: Sequence[str],
    detect: Callable[[list[Window]], None],
) -> dict[int, tuple[float, float]]:
    """Place each language switch more precisely.

    For the switch before ``windows[b]``, the windows either side are split
    into ~10 s pieces and ``detect`` fills in each piece's language (one
    call for every switch). The cut goes where the most pieces agree: the
    pieces before it in the old language, those after in the new.
    Returns {b: (left, right)}, the stretch to cut within.
    """
    switches = [(b, langs[b - 1], langs[b]) for b in range(1, len(langs)) if langs[b] != langs[b - 1]]
    pieces = {b: split(windows[b - 1]) + split(windows[b]) for b, _, _ in switches}
    detect([p for b, _, _ in switches for p in pieces[b]])
    cuts: dict[int, tuple[float, float]] = {}
    for b, old, new in switches:
        subs = pieces[b]
        labels = [
            p.language if p.language in (old, new) and p.confidence >= MIN_CONFIDENCE
            and p.duration >= MIN_WINDOW_SECONDS else None
            for p in subs
        ]
        middle = len(split(windows[b - 1]))
        # score(c) = pieces before c heard as `old` + pieces from c heard as `new`
        def score(c: int) -> tuple[int, int]:
            agree = sum(lab == old for lab in labels[:c]) + sum(lab == new for lab in labels[c:])
            return agree, -abs(c - middle)  # ties: stay nearest the coarse cut
        c = max(range(len(subs) + 1), key=score)
        # Unsure pieces next to the cut could be either side: cut at the
        # quietest pause anywhere among them, not at their edge.
        lo, hi = c, c
        while lo > 0 and labels[lo - 1] is None:
            lo -= 1
        while hi < len(subs) and labels[hi] is None:
            hi += 1
        if lo < hi:
            cuts[b] = (subs[lo].start, subs[hi - 1].end)
        elif c == 0:
            cuts[b] = (subs[0].start, subs[0].start)
        elif c == len(subs):
            cuts[b] = (subs[-1].end, subs[-1].end)
        else:
            cuts[b] = (subs[c - 1].end, subs[c].start)
    return cuts


def spans_for(
    windows: Sequence[Window],
    langs: Sequence[str],
    duration: float,
    quietest: Callable[[float, float], float] | None = None,
    cuts: dict[int, tuple[float, float]] | None = None,
) -> list[Span]:
    """Contiguous spans covering [0, duration], one per run of a language,
    each cut at the quietest moment between the two windows it falls
    between (or within the stretch :func:`refine` found)."""
    runs = _runs(langs)
    spans: list[Span] = []
    start = 0.0
    for k, (lang, _a, b) in enumerate(runs):
        if k == len(runs) - 1:
            end = duration
        else:
            left, right = (cuts or {}).get(b, (windows[b - 1].end, windows[b].start))
            around = (left + right) / 2
            radius = max(0.5, abs(right - left) / 2)
            end = quietest(around, radius) if quietest else around
            end = min(max(end, start), duration)
        spans.append(Span(start, end, lang))
        start = end
    return [s for s in spans if s.end > s.start] or [Span(0.0, duration, runs[0][0])]


# ---------- transcription ----------

def _shift(entry: dict, offset_ms: int) -> None:
    """Move a whisper JSON segment (or token) onto the original timeline."""
    offsets = entry.get("offsets")
    if isinstance(offsets, dict):
        f = int(offsets.get("from", 0)) + offset_ms
        t = int(offsets.get("to", 0)) + offset_ms
        entry["offsets"] = {"from": f, "to": t}
        entry["timestamps"] = {"from": timestamp(f), "to": timestamp(t)}
    for token in entry.get("tokens") or []:
        _shift(token, offset_ms)


def timestamp(ms: int, sep: str = ",") -> str:
    """whisper-cli's ``HH:MM:SS,mmm``."""
    ms = max(0, int(ms))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _decode(
    run: Run,
    base_cmd: list[str],
    jobs: Sequence[tuple[Path, str]],
    json_kind: str,
    progress: Callable[[float], None],
    extra: Sequence[str] = (),
) -> dict[Path, dict]:
    """Decode clips, one whisper-cli process per language, and return each
    clip's JSON. ``progress`` gets 0-100 over all of them, by audio length."""
    sizes = {clip: max(clip.stat().st_size, 1) for clip, _ in jobs}
    total = sum(sizes.values())
    done = 0
    results: dict[Path, dict] = {}
    by_lang: dict[str, list[Path]] = {}
    for clip, lang in jobs:
        by_lang.setdefault(lang, []).append(clip)
    for lang, clips in by_lang.items():
        cmd = list(base_cmd) + ["-l", lang, json_kind, *extra]
        for c in clips:
            cmd += ["-f", str(c)]
        tracker = _FileProgress([sizes[c] for c in clips], done, total, progress)
        rc = run(cmd, tracker)
        if rc != 0:
            raise RuntimeError(f"whisper-cli exited {rc} transcribing the {lang} parts")
        for c in clips:
            out = Path(str(c) + ".json")
            if not out.exists():
                raise RuntimeError(f"whisper-cli wrote no transcript for {c.name}")
            results[c] = json.loads(out.read_text(encoding="utf-8", errors="replace"))
        done += sum(sizes[c] for c in clips)
    return results


class _FileProgress:
    """whisper-cli reports 0-100% per input file; turn that into one
    percentage over every file of every process. A drop in the percentage
    means it moved on to the next file. Never goes backwards."""

    def __init__(self, sizes: Sequence[int], before: int, total: int, emit: Callable[[float], None]) -> None:
        self.sizes, self.before, self.total, self.emit = list(sizes), before, total, emit
        self.index, self.last, self.shown = 0, 0.0, -1.0

    def __call__(self, percent: float) -> None:
        if percent + 10 < self.last and self.index < len(self.sizes) - 1:
            self.index += 1
        self.last = percent
        done = self.before + sum(self.sizes[:self.index]) + self.sizes[self.index] * percent / 100
        value = 100.0 * done / self.total
        if value > self.shown:
            self.shown = value
            self.emit(value)


def _tail_hallucination(seg: dict, length_ms: int) -> bool:
    """A line whisper invented over the silence padding a clip's end: it
    starts in the clip's last two seconds and claims to run more than a
    second past the audio (whisper pads the last window to 30 s, and
    fills quiet with lines from its training subtitles: "Субтитрувальниця
    Оля Шор", "Thanks for watching")."""
    return _ms(seg, "from") >= length_ms - 2000 and _ms(seg, "to") > length_ms + 1000


def transcribe_spans(
    work: Workspace,
    spans: Sequence[Span],
    base_cmd: list[str],
    json_kind: str,
    run: Run,
    progress: Callable[[float], None],
) -> dict:
    """Decode each span in its own language; one whisper JSON document on
    the original timeline."""
    clips = [(work.clip(s.start, s.end, "span"), s) for s in spans]
    results = _decode(run, base_cmd, [(c, s.language) for c, s in clips], json_kind, progress)
    first = results[clips[0][0]]
    segments: list[dict] = []
    for clip, span in clips:
        length = round((span.end - span.start) * 1000)
        for seg in results[clip].get("transcription") or []:
            if _tail_hallucination(seg, length):
                continue
            _shift(seg, round(span.start * 1000))
            segments.append(seg)
    totals: dict[str, float] = {}
    for s in spans:
        totals[s.language] = totals.get(s.language, 0.0) + (s.end - s.start)
    doc = {key: first[key] for key in ("systeminfo", "model", "params") if key in first}
    if isinstance(doc.get("params"), dict):
        doc["params"] = {**doc["params"], "language": "auto"}
    doc["result"] = {"language": max(totals, key=totals.get)}
    doc["languages"] = [
        {"from": timestamp(round(s.start * 1000)), "to": timestamp(round(s.end * 1000)), "language": s.language}
        for s in spans
    ]
    doc["transcription"] = segments
    return doc


# ---------- repetition loops ----------

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def _norm(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


def find_loops(segments: Sequence[dict]) -> list[tuple[int, int]]:
    """(first, last) indices of each run of one line repeated in a row:
    three times for a line of three words or more, six for a shorter one
    (people do say "Yes." three times running)."""
    loops = []
    texts = [_norm(s.get("text", "")) for s in segments]
    i = 0
    while i < len(texts):
        j = i
        while j + 1 < len(texts) and texts[i] and texts[j + 1] == texts[i]:
            j += 1
        count = j - i + 1
        if texts[i] and (count >= 6 or (count >= 3 and len(texts[i].split()) >= 3)):
            loops.append((i, j))
        i = j + 1
    return loops


def _ms(seg: dict, end: str) -> int:
    return int((seg.get("offsets") or {}).get(end, 0))


def _collapse(segments: list[dict]) -> list[dict]:
    """Each loop reduced to its first line, stretched over the loop's time."""
    out = list(segments)
    for i, j in reversed(find_loops(out)):
        keep = dict(out[i])
        f, t = _ms(out[i], "from"), _ms(out[j], "to")
        keep["offsets"] = {"from": f, "to": t}
        keep["timestamps"] = {"from": timestamp(f), "to": timestamp(t)}
        out[i:j + 1] = [keep]
    return out


@dataclass
class LoopReport:
    start: float
    end: float
    lines: int
    repaired: bool


def repair_loops(
    doc: dict,
    work: Workspace,
    base_cmd: list[str],
    language_at: Callable[[float], str],
    json_kind: str,
    run: Run,
    progress: Callable[[float], None],
) -> list[LoopReport]:
    """Decode each looped stretch again, alone and with no carried-over text
    (``-mc 0``), and put the result in its place. What still loops after
    that is collapsed to one line. Changes ``doc`` in place."""
    segments = doc.get("transcription") or []
    loops = find_loops(segments)
    if not loops:
        return []
    jobs = []
    for i, j in loops:
        start, end = _ms(segments[i], "from") / 1000, _ms(segments[j], "to") / 1000
        jobs.append((work.clip(start, end, "loop"), start, end, i, j))
    results = _decode(
        run, base_cmd, [(clip, language_at(start)) for clip, start, *_ in jobs],
        json_kind, progress, extra=["-mc", "0"],
    )
    reports = []
    for clip, start, end, i, j in reversed(jobs):
        fresh = results[clip].get("transcription") or []
        for seg in fresh:
            _shift(seg, round(start * 1000))
        repaired = bool(fresh) and not find_loops(fresh)
        segments[i:j + 1] = _collapse(fresh) if fresh else _collapse(segments[i:j + 1])
        reports.append(LoopReport(start, end, j - i + 1, repaired))
    doc["transcription"] = segments
    return list(reversed(reports))


# ---------- writing ----------

def write_outputs(doc: dict, base: str, formats: Iterable[str]) -> list[Path]:
    """Write ``doc`` in each whisper-cli format asked for, at the paths
    whisper-cli would have used (``<base>.<ext>``), in its layout."""
    segments = doc.get("transcription") or []
    texts = [(seg, (seg.get("text") or "")) for seg in segments]
    written = []
    for fmt in dict.fromkeys(formats):
        if fmt in ("json", "json-full"):
            path, body = Path(base + ".json"), json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
        elif fmt == "txt":
            path, body = Path(base + ".txt"), "".join(t.lstrip(" \t") + "\n" for _, t in texts)
        elif fmt == "srt":
            path = Path(base + ".srt")
            body = "".join(
                f"{n}\n{timestamp(_ms(s, 'from'))} --> {timestamp(_ms(s, 'to'))}\n{t}\n\n"
                for n, (s, t) in enumerate(texts, 1)
            )
        elif fmt == "vtt":
            path = Path(base + ".vtt")
            body = "WEBVTT\n\n" + "".join(
                f"{timestamp(_ms(s, 'from'), '.')} --> {timestamp(_ms(s, 'to'), '.')}\n{t}\n\n"
                for s, t in texts
            )
        elif fmt == "csv":
            path = Path(base + ".csv")
            body = "start,end,text\n" + "".join(
                f'{_ms(s, "from")},{_ms(s, "to")},"{t.replace(chr(34), chr(34) * 2)}"\n' for s, t in texts
            )
        elif fmt == "lrc":
            path = Path(base + ".lrc")
            lines = []
            for s, t in texts:
                ms = _ms(s, "from")
                lines.append(f"[{ms // 60000:02d}:{ms // 1000 % 60:02d}.{ms % 1000 // 10:02d}]{t}\n")
            body = "[by:whisper.cpp]\n" + "".join(lines)
        else:
            continue
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


def clock(t: float) -> str:
    """5:12, or 1:05:12 past an hour."""
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def describe(spans: Sequence[Span]) -> str:
    """'uk 0:00–5:12, en 5:12–45:40, uk 45:40–49:44'."""
    return ", ".join(f"{s.language} {clock(s.start)}–{clock(s.end)}" for s in spans)
