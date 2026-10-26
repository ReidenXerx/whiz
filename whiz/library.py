"""The library: every recording whiz has transcribed, wherever it lives.

whiz writes its outputs next to the recording, so finding past calls means
knowing where they were. ``transcribe``, ``merge`` and ``analyze`` add a
line to ``library.jsonl`` in the config directory when they finish; the MCP
server (and search) read it back. One JSON object per line, appended — two
runs finishing at once cannot tear each other's record — and the newest
line for a recording wins on read. ``whiz library add`` registers recordings
transcribed before the library existed; ``whiz library forget`` hides one.

A call is read from whatever its run left, best first: the frames manifest
(speaker, text and a frame per segment), the labeled ``.speakers.srt``, the
plain whisper JSON. Nothing is copied: delete the files and the call is
gone from the library too.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from whiz import config as cfg

LIBRARY_NAME = "library.jsonl"


def library_path() -> Path:
    return cfg.CONFIG_DIR / LIBRARY_NAME


def call_id(input_path: Path | str) -> str:
    """A short, stable id for a recording: its path, hashed."""
    return hashlib.sha1(str(Path(input_path).expanduser().resolve()).encode("utf-8")).hexdigest()[:10]


@dataclass
class Line:
    """One line of a transcript."""
    start: float
    end: float
    speaker: str
    text: str
    frame: Path | None = None


@dataclass
class Call:
    input: Path
    base: Path
    added: str
    whisper_json: Path | None = None
    _lines: list[Line] | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return call_id(self.input)

    @property
    def title(self) -> str:
        return self.input.stem

    @property
    def manifest(self) -> Path:
        return Path(str(self.base) + ".frames.json")

    @property
    def labeled_srt(self) -> Path:
        return Path(str(self.base) + ".speakers.srt")

    @property
    def analysis(self) -> Path:
        return Path(str(self.base) + ".analysis.md")

    @property
    def frames_dir(self) -> Path:
        return Path(str(self.base) + ".frames")

    @property
    def has_transcript(self) -> bool:
        return self.manifest.exists() or self.labeled_srt.exists() or bool(
            self.whisper_json and self.whisper_json.exists())

    @property
    def has_summary(self) -> bool:
        return self.analysis.exists()

    @property
    def has_frames(self) -> bool:
        return self.manifest.exists() and self.frames_dir.is_dir()

    def lines(self) -> list[Line]:
        if self._lines is None:
            self._lines = _read_lines(self)
        return self._lines

    @property
    def duration(self) -> float:
        lines = self.lines()
        return max((line.end for line in lines), default=0.0)

    def speakers(self) -> list[str]:
        """Speakers by talk time, most first."""
        talk: dict[str, float] = {}
        for line in self.lines():
            talk[line.speaker] = talk.get(line.speaker, 0.0) + max(line.end - line.start, 0.0)
        return [name for name, _ in sorted(talk.items(), key=lambda item: -item[1]) if name]

    def recorded(self) -> datetime:
        """When the recording was made: the file's own date, else when it was added."""
        try:
            return datetime.fromtimestamp(self.input.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return datetime.fromisoformat(self.added)


# ---------- reading transcripts ----------

_SRT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})")


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_labeled_srt(text: str) -> list[Line]:
    """``.speakers.srt``: each cue's text is ``Label: words``."""
    lines: list[Line] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        rows = block.strip().splitlines()
        for i, row in enumerate(rows):
            match = _SRT_TIME.search(row)
            if not match:
                continue
            body = " ".join(r.strip() for r in rows[i + 1:]).strip()
            speaker, sep, words = body.partition(": ")
            if not sep:
                speaker, words = "", body
            lines.append(Line(_seconds(*match.groups()[:4]), _seconds(*match.groups()[4:]),
                              speaker.strip(), words.strip()))
            break
    return lines


def _read_lines(call: Call) -> list[Line]:
    if call.manifest.exists():
        from whiz import screenshots as SC

        entries = SC.load_manifest(call.manifest)
        if entries:
            return [Line(e.start, e.end, e.speaker, e.text.strip(),
                         (call.frames_dir / e.frame) if e.frame else None)
                    for e in entries if e.text.strip()]
    if call.labeled_srt.exists():
        return parse_labeled_srt(call.labeled_srt.read_text(encoding="utf-8", errors="replace"))
    if call.whisper_json and call.whisper_json.exists():
        from whiz import merge as MR

        return [Line(seg.start, seg.end, "", seg.text.strip())
                for seg in MR.parse_whisper_json(call.whisper_json) if seg.text.strip()]
    return []


# ---------- the index ----------

def remember(input_path: Path, of_base: Path, whisper_json: Path | None = None,
             *, path: Path | None = None) -> Call | None:
    """Add (or refresh) a recording in the library. Returns the call, or None
    when its run left no transcript to read."""
    record_input = Path(input_path).expanduser().resolve()
    call = Call(record_input, Path(of_base).expanduser().resolve(),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                Path(whisper_json).expanduser().resolve() if whisper_json else None)
    if not call.has_transcript:
        return None
    _append({
        "input": str(call.input),
        "base": str(call.base),
        "whisper_json": str(call.whisper_json) if call.whisper_json else None,
        "added": call.added,
    }, path)
    return call


def forget(input_path: Path, *, path: Path | None = None) -> None:
    _append({"input": str(Path(input_path).expanduser().resolve()), "forget": True,
             "added": datetime.now(timezone.utc).isoformat(timespec="seconds")}, path)


def _append(record: dict, path: Path | None) -> None:
    target = path or library_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    # One write() of one line on an O_APPEND descriptor: concurrent runs
    # interleave whole lines, never halves.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def calls(*, path: Path | None = None) -> list[Call]:
    """Every call whose transcript still exists, newest recording first."""
    target = path or library_path()
    latest: dict[str, dict] = {}
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return []
    for raw in text.splitlines():
        try:
            record = json.loads(raw)
            key = record["input"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # a hand-edited or torn line costs that line only
        latest[key] = record
    found: list[Call] = []
    for record in latest.values():
        if record.get("forget"):
            continue
        try:
            call = Call(Path(record["input"]), Path(record["base"]), record.get("added", ""),
                        Path(record["whisper_json"]) if record.get("whisper_json") else None)
        except (KeyError, TypeError):
            continue
        if call.has_transcript:
            found.append(call)
    found.sort(key=lambda c: c.recorded(), reverse=True)
    return found


def find(ref: str, *, path: Path | None = None) -> Call | None:
    """A call by id, by path, or by a title fragment that names only one."""
    everything = calls(path=path)
    ref = ref.strip()
    for call in everything:
        if call.id == ref or str(call.input) == ref:
            return call
    needle = ref.lower()
    named = [c for c in everything if needle and needle in c.title.lower()]
    return named[0] if len(named) == 1 else None


def discover(input_path: Path) -> tuple[Path, Path | None]:
    """Where an earlier run of whiz put a recording's outputs, for ``library add``:
    the output base, and the whisper JSON if there is one.

    Outputs hang off ``<stem>`` for a video or audio whiz decoded first
    (``talk.mov`` → ``talk.speakers.srt``, ``talk.wav.json``), and off the
    whole name for audio whisper reads itself (``talk.mp3.speakers.srt``,
    ``talk.mp3.json``)."""
    in_path = Path(input_path).expanduser().resolve()
    bases = [in_path.with_suffix(""), in_path]
    jsons = [Path(str(bases[0]) + ".wav.json"), Path(str(in_path) + ".json"), Path(str(bases[0]) + ".json")]
    whisper_json = next((j for j in jsons if j.exists()), None)
    for base in bases:
        if Call(in_path, base, "", whisper_json).has_transcript and (
                Path(str(base) + ".frames.json").exists() or Path(str(base) + ".speakers.srt").exists()):
            return base, whisper_json
    return bases[0], whisper_json
