"""Persistent user configuration for whiz.

Config lives at ~/.config/whiz/config.toml (created on demand).
Only Python 3.11+ stdlib tomllib is used for reading; writing is a tiny
hand-rolled TOML emitter so we don't depend on a third-party package.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import tempfile
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.environ.get("WHIZ_CONFIG_DIR", Path.home() / ".config" / "whiz"))
CONFIG_PATH = CONFIG_DIR / "config.toml"


@dataclass
class Config:
    # Model alias or absolute path preferred by default (empty => auto-pick best).
    model: str = ""
    # Extra directories to scan for models.
    model_dirs: list[str] = field(default_factory=list)
    # whisper-cli binary path. Empty => auto-detect on PATH.
    whisper_cli: str = ""
    # ffmpeg binary path. Empty => auto-detect on PATH.
    ffmpeg: str = ""
    # Number of CPU threads (0 => auto: min(8, cpu_count)).
    threads: int = 0
    # Spoken language code or "auto".
    language: str = "auto"
    # Enable VAD by default.
    vad: bool = True
    # Path to Silero VAD model (empty => auto-discover or download ggml-silero-vad.bin).
    vad_model: str = ""
    # VAD threshold.
    vad_threshold: float = 0.5
    # Output formats to produce by default.
    outputs: list[str] = field(default_factory=lambda: ["srt", "json"])
    # Print progress to stderr.
    verbose: bool = True
    # Additional flags passed verbatim to whisper-cli.
    extra_args: list[str] = field(default_factory=list)
    # --- Diarization (sherpa-onnx) ---
    # Enable speaker diarization by default.
    diarize: bool = False
    # Known number of speakers (0 => auto-detect via cluster_threshold).
    num_speakers: int = 0
    # Clustering threshold when auto-detecting (larger = fewer speakers; default 0.9 per sherpa-onnx guidance).
    cluster_threshold: float = 0.9
    # Explicit paths to diarization models (empty => auto-discover).
    diarization_segmentation_model: str = ""
    diarization_embedding_model: str = ""
    # Remembered answer to the one-time diarization auto-setup prompt.
    # None (unset) => ask on a TTY / proceed automatically when scripted;
    # true/false answers permanently for both. Written by the prompt and
    # settable by hand: whiz config set auto_diarization_setup=false
    auto_diarization_setup: bool | None = None
    # --- AI analysis (Ollama / OpenAI-compatible) ---
    # Base URL of the chat completions endpoint (without /chat/completions).
    ai_base_url: str = "http://localhost:11434/v1"
    # Model name (e.g. 'llava', 'qwen2.5-vl', 'gpt-4o-mini'). Empty => error with hint.
    ai_model: str = ""
    # API key (Ollama ignores this; set for cloud OpenAI-compatible providers).
    ai_api_key: str = ""
    # Max frames sent to a vision model (spread evenly across the video).
    ai_max_frames: int = 50
    # --- Speaker voice profiles ---
    # Cosine-similarity threshold for auto-matching a cluster to a stored profile.
    # Higher = stricter (fewer auto-assignments); 0.8 suits 3D-Speaker embeddings.
    speaker_match_threshold: float = 0.8
    # When True, save a voice profile for each named speaker after transcription/merge.
    save_voice_profiles: bool = True
    # Dictation used to live here as dictate_* keys. It is Mynah now
    # (github.com/ReidenXerx/mynah), which reads them out of this file once and
    # then keeps its own. save() preserves keys it does not know, so an existing
    # config file keeps them until Mynah has imported them.

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_MODEL_SEARCH_DIRS: list[Path] = [
    Path.home() / ".cache" / "whisper",
    Path.home() / "Library" / "Application Support" / "com.unspoken.app" / "WhisperModels",
    Path.home() / "Library" / "Caches" / "whisper",
    Path("/usr/local/share/whisper"),
    Path("/opt/homebrew/share/whisper"),
    Path("/usr/share/whisper"),
]


# Characters TOML forbids raw inside a basic string: every C0 control and
# DEL. \n/\r/\t get their short escapes; the rest go out as \uXXXX.
_TOML_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_TOML_SHORT_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
# Keys that may be written bare; anything else is emitted as a quoted key.
_TOML_BARE_KEY_RE = re.compile(r"[A-Za-z0-9_-]+")


def _escape_toml_string(value: str) -> str:
    """Escape a string for a basic TOML double-quoted literal.

    Order matters: backslash first (or its own output would be re-escaped
    by later steps), then the quote, then control characters — a raw
    ``\n`` inside a quoted string makes the file invalid TOML for the
    NEXT ``load()`` of every command (C1, wave-1 audit; the Swift reader
    drops the line outright). Multi-line prompts saved from the settings
    window are the known trigger.

    Wave-3: escaping only newline/CR/tab left every OTHER control character
    (an ANSI ESC 0x1b pasted into a prompt, a stray NUL, DEL) raw —
    equally illegal in a basic string, equally fatal to the next load.
    Those now go out as ``\\uXXXX``.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return _TOML_CONTROL_RE.sub(
        lambda m: _TOML_SHORT_ESCAPES.get(m.group(0), f"\\u{ord(m.group(0)):04X}"),
        escaped,
    )


def _toml_key(key: str) -> str:
    """A TOML key: bare when it can be, quoted otherwise."""
    if _TOML_BARE_KEY_RE.fullmatch(key):
        return key
    return f'"{_escape_toml_string(key)}"'


def _toml_value(value: Any) -> str:
    """One TOML value, inline. Raises ValueError for anything unrepresentable.

    Covers everything ``tomllib`` can hand back — so a foreign key written by
    another tool (Mynah's ``[mynah]`` table, a datetime, a nested or mixed
    array) round-trips instead of going out as Python ``repr`` and making the
    file unreadable for every later command (wave-3 audit).
    """
    # bool before int: bool is an int subclass.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # repr keeps full precision; inf/-inf/nan spell the same in TOML.
        return repr(value)
    if isinstance(value, str):
        return f'"{_escape_toml_string(value)}"'
    # datetime before date: datetime is a date subclass.
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(
            f"{_toml_key(str(k))} = {_toml_value(v)}"
            for k, v in value.items() if v is not None
        )
        return "{ " + items + " }" if items else "{}"
    raise ValueError(f"cannot represent {type(value).__name__} value {value!r} in TOML")


def _emit_table(data: dict[str, Any], prefix: tuple[str, ...], lines: list[str]) -> None:
    """Emit one table's plain keys, then each sub-table as a ``[section]``."""
    tables: list[tuple[str, dict[str, Any]]] = []
    for key, value in data.items():
        if value is None:
            # Tri-state fields (e.g. auto_diarization_setup) are "unset" by
            # None — an emitted `= None` would be invalid TOML and break the
            # NEXT load() for every command, not just the one that saved.
            continue
        if isinstance(value, dict):
            tables.append((key, value))
            continue
        try:
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        except ValueError as e:
            raise ValueError(f"{'.'.join(prefix + (key,))}: {e}") from e
    for key, sub in tables:
        path = prefix + (key,)
        if lines:
            lines.append("")
        lines.append("[" + ".".join(_toml_key(k) for k in path) + "]")
        _emit_table(sub, path, lines)


def _emit_toml(data: dict[str, Any]) -> str:
    """Minimal TOML writer: our flat schema plus whatever foreign keys hold.

    Top-level scalars and arrays first, then each dict value as a
    ``[table]`` section (recursively, ``[a.b]``) — keys after a table header
    would otherwise land INSIDE that table. Raises ValueError when a value
    has no TOML spelling, so ``save()`` refuses rather than writing a file
    the next ``load()`` cannot parse.
    """
    lines: list[str] = []
    _emit_table(data, (), lines)
    return "\n".join(lines) + "\n"


def load() -> Config:
    """Load config from disk, falling back to defaults (missing file only).

    A MISSING file means defaults — fine. A CORRUPT file must not be
    silently swallowed into defaults either: that would make every
    ``whiz config set`` read-modify-write from an empty table and rewrite
    the file, permanently deleting every key the user had (C1, wave-1
    audit). Raise a RuntimeError naming the file with the fix instead;
    ``main()`` catches RuntimeError and prints it as a clean error.
    """
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as e:
            raise RuntimeError(
                f"config.toml is corrupt and could not be read: {e}\n"
                f"Fix or delete the file by hand: {CONFIG_PATH}\n"
                "(a deleted file regenerates from defaults; a corrupted one "
                "left in place breaks every command until fixed)"
            ) from e
        # Only keep known keys so older configs don't break dataclass init.
        known = {k: v for k, v in data.items() if k in Config.__dataclass_fields__}
        return Config(**known)
    return Config()


def save(cfg: Config) -> Path:
    """Write config to disk, preserving keys this version does not know about.

    Read-modify-write rather than a plain overwrite. ``load()`` already filters
    to known dataclass fields, so a naive ``write_text(_emit_toml(cfg.to_dict()))``
    silently deletes every key the running build has never heard of.

    That is not hypothetical. The config file is shared by several writers that
    do not agree on the schema: the Swift app owns ``dictate_*``, feature
    branches add their own keys (``ocr_*``), and a user may be running a pipx
    install that is older or newer than the checkout. Any ``whiz config set``
    from the wrong one wiped the others' settings without a word.

    Wave-3 audit: the emitted text is re-parsed BEFORE anything touches disk
    — a value with no TOML spelling (or an emitter bug) raises RuntimeError
    and the existing file is left byte-for-byte intact, never replaced by one
    the next ``load()`` rejects for every command. The write is atomic (temp
    file in the same directory + ``os.replace``), so a crash mid-write cannot
    truncate the file either. The file is created 0600 (it can hold an
    ``ai_api_key``) and a directory whiz creates is 0700.
    """
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    merged: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("rb") as fh:
                merged.update(tomllib.load(fh))
        except (OSError, tomllib.TOMLDecodeError):
            # An unreadable file should not block saving; fall back to a
            # clean write rather than refusing to persist the change.
            merged = {}
    # Skip fields the in-memory config never answered (None): they must not
    # clobber a value a previous session persisted (a fresh Config() has
    # auto_diarization_setup=None even when the file says true).
    updates = {k: v for k, v in cfg.to_dict().items() if v is not None}
    merged.update(updates)

    try:
        text = _emit_toml(merged)
        tomllib.loads(text)
    except (ValueError, tomllib.TOMLDecodeError) as e:
        raise RuntimeError(
            f"refusing to save config: the result would not be valid TOML ({e}).\n"
            f"{CONFIG_PATH} was left unchanged."
        ) from e
    _write_atomic(CONFIG_PATH, text)
    return CONFIG_PATH


def _write_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` atomically, mode 0600.

    ``mkstemp`` creates the temp file 0600 in the SAME directory, so the
    ``os.replace`` is a same-filesystem rename: readers see the old file or
    the complete new one, never a partial write. A symlinked config (dotfile
    managers) is written through to its target instead of being replaced by
    a regular file.
    """
    target = path.resolve() if path.is_symlink() else path
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def model_search_dirs(cfg: Config) -> list[Path]:
    """Built-in defaults plus user-configured extra dirs."""
    dirs = list(DEFAULT_MODEL_SEARCH_DIRS)
    for d in cfg.model_dirs:
        dirs.append(Path(d).expanduser())
    # De-dup preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        key = str(d.expanduser().resolve())
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out