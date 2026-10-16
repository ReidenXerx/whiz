"""Audio extraction via ffmpeg.

whisper-cli only decodes audio files; it cannot demux video containers
like .mov/.mp4. This module extracts a 16 kHz mono PCM WAV — the format
Whisper natively expects — so there are no resampling surprises.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

# Extensions whisper-cli accepts directly (no extraction needed).
AUDIO_EXTS = {".flac", ".mp3", ".ogg", ".wav", ".m4a", ".aac", ".opus", ".wma"}
# Video containers we extract from.
VIDEO_EXTS = {".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ts"}


def find_ffmpeg(configured: str = "") -> str:
    """Locate ffmpeg: explicit path or PATH lookup."""
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("ffmpeg not found on PATH — install it (brew install ffmpeg) or set ffmpeg in config.")
    return found


def needs_extraction(path: Path) -> bool:
    """True if the file is a video container that whisper-cli can't read."""
    return path.suffix.lower() in VIDEO_EXTS


def is_audio(path: Path) -> bool:
    """True if whisper-cli can read this format directly."""
    return path.suffix.lower() in AUDIO_EXTS


# Written into every WAV whiz extracts (RIFF INFO/ICMT, via ffmpeg's
# `comment` metadata) followed by the source file's name. It is how whiz
# tells its own intermediate `<stem>.wav` from a user's recording that
# happens to share the name — the second must never be overwritten, reused
# as the video's audio, or deleted as "intermediate".
_MARKER = b"whiz-extracted:"
# The INFO chunk sits right after `fmt `, well inside the first few KB.
_MARKER_SCAN_BYTES = 64 * 1024


def extracted_wav_path(src: Path) -> Path:
    """Where whiz puts (and names its outputs after) a video's audio.

    Output naming hangs off this path — whisper-cli's ``<stem>.wav.json``,
    the diarization cache, ``whiz merge``'s lookup — whether or not the
    audio actually lives there (see :func:`extraction_dir`).
    """
    return src.parent / (src.stem + ".wav")


def is_whiz_extracted(path: Path, src: Path | None = None) -> bool:
    """True if ``path`` is a WAV whiz extracted (from ``src``, when given).

    A WAV written before the marker existed reads as not-ours: treating it
    as foreign costs a re-extraction, treating a user's file as ours costs
    the recording.
    """
    try:
        with path.open("rb") as f:
            head = f.read(_MARKER_SCAN_BYTES)
    except OSError:
        return False
    at = head.find(_MARKER)
    if at < 0:
        return False
    if src is None:
        return True
    name = src.name.encode("utf-8")
    end = at + len(_MARKER) + len(name)
    # The ICMT value is NUL-terminated; requiring it rejects a.mov matching
    # a marker written for a.mov.bak.
    return head[at + len(_MARKER):end] == name and head[end:end + 1] in (b"\0", b"")


def extraction_dir(src: Path) -> Path | None:
    """Where to extract ``src``'s audio: ``None`` for beside the source.

    Beside the source is the normal place. When a file NOT extracted by whiz
    from this very source already sits at :func:`extracted_wav_path` — the
    user's own ``talk.wav`` next to ``talk.mov``, or the audio of a
    same-stem ``talk.mp4`` — a fresh private temp dir is returned instead,
    so the existing file is never overwritten or deleted.
    """
    target = extracted_wav_path(src)
    if not target.exists() or is_whiz_extracted(target, src):
        return None
    return Path(tempfile.mkdtemp(prefix="whiz-audio-"))


def is_diarization_ready(path: Path) -> bool:
    """True if ``path`` is the 16 kHz mono 16-bit PCM WAV diarization reads.

    Anything else (m4a, mp3, a 44.1 kHz or stereo WAV) has to be converted
    first; the diarization reader accepts nothing else.
    """
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "rb") as w:
            return (
                w.getframerate() == 16000
                and w.getnchannels() == 1
                and w.getsampwidth() == 2
                and w.getcomptype() == "NONE"
            )
    except (wave.Error, EOFError, OSError):
        return False


def extract_audio(
    src: Path,
    ffmpeg: str,
    dest_dir: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Extract 16 kHz mono WAV from a media file.

    Returns the path to the WAV (which would be produced, if dry_run).
    Refuses to overwrite a file whiz did not extract from ``src`` — choose
    the destination with :func:`extraction_dir`.
    """
    out_dir = dest_dir or src.parent
    out = out_dir / (src.stem + ".wav")
    if not dry_run and out.exists() and not is_whiz_extracted(out, src):
        raise RuntimeError(
            f"{out} already exists and was not extracted by whiz from {src.name}; "
            "refusing to overwrite it."
        )

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(src),
        "-vn",            # no video
        "-ac", "1",       # mono
        "-ar", "16000",   # 16 kHz, Whisper's native rate
        "-c:a", "pcm_s16le",
        "-metadata", f"comment={_MARKER.decode()}{src.name}",
        str(out),
    ]
    if dry_run:
        print("DRY-RUN audio extraction:")
        print("  " + " ".join(cmd))
        return out

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    if not out.exists():
        raise RuntimeError(f"ffmpeg reported success but {out} was not created")
    return out