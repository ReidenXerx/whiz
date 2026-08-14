"""Audio extraction via ffmpeg.

whisper-cli only decodes audio files; it cannot demux video containers
like .mov/.mp4. This module extracts a 16 kHz mono PCM WAV — the format
Whisper natively expects — so there are no resampling surprises.
"""

from __future__ import annotations

import shutil
import subprocess
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


def extract_audio(
    src: Path,
    ffmpeg: str,
    dest_dir: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Extract 16 kHz mono WAV from a media file.

    Returns the path to the WAV (which would be produced, if dry_run).
    """
    out_dir = dest_dir or src.parent
    out = out_dir / (src.stem + ".wav")

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(src),
        "-vn",            # no video
        "-ac", "1",       # mono
        "-ar", "16000",   # 16 kHz, Whisper's native rate
        "-c:a", "pcm_s16le",
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