"""Audio extraction via ffmpeg (or Whiz.app's whiz-media).

whisper-cli only decodes audio files; it cannot demux video containers
like .mov/.mp4. This module extracts a 16 kHz mono PCM WAV — the format
Whisper natively expects — so there are no resampling surprises.

The tool is ffmpeg whenever one is installed. Without it, ``whiz-media``
stands in: the AVFoundation helper Whiz.app bundles so a buyer needs no
Homebrew (macos/Sources/WhizMedia). Same WAV, same marker, same exit-status
contract; it reads what macOS reads (mov, mp4, m4a, mp3, wav…), not mkv
or webm.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

# Audio formats whiz takes.
AUDIO_EXTS = {".flac", ".mp3", ".ogg", ".wav", ".m4a", ".aac", ".opus", ".wma"}
# The ones whisper-cli decodes itself (miniaudio + stb_vorbis). The rest —
# m4a from Voice Memos and Zoom, aac, opus, wma — reach it as a WAV whiz
# made, exactly like a video's audio; handed over as-is they failed with
# "failed to read audio data".
WHISPER_READS = {".wav", ".mp3", ".flac", ".ogg"}
# Video containers we extract from.
VIDEO_EXTS = {".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ts"}


# The bundled stand-in for ffmpeg, found on PATH after ffmpeg itself.
MEDIA_HELPER = "whiz-media"


def find_ffmpeg(configured: str = "") -> str:
    """Locate the media tool: explicit path, ffmpeg on PATH, then whiz-media."""
    if configured:
        return configured
    found = shutil.which("ffmpeg") or shutil.which(MEDIA_HELPER)
    if not found:
        raise RuntimeError("ffmpeg not found on PATH — install it (brew install ffmpeg) or set ffmpeg in config.")
    return found


def is_media_helper(tool: str) -> bool:
    """True if ``tool`` is whiz-media, which takes its own argv, not ffmpeg's."""
    return Path(tool).name == MEDIA_HELPER


def needs_extraction(path: Path) -> bool:
    """True if the file is a video container that whisper-cli can't read."""
    return path.suffix.lower() in VIDEO_EXTS


def needs_decoding(path: Path) -> bool:
    """True if whiz must make the WAV whisper-cli reads: a video, or audio
    whisper-cli cannot decode itself. (``needs_extraction`` stays "is this a
    video" — frames and speaker auto-detection hang off that.)"""
    suffix = path.suffix.lower()
    return suffix in VIDEO_EXTS or (suffix in AUDIO_EXTS and suffix not in WHISPER_READS)


def is_audio(path: Path) -> bool:
    """True if this is an audio format whiz takes (see WHISPER_READS)."""
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

    comment = f"{_MARKER.decode()}{src.name}"
    if is_media_helper(ffmpeg):
        cmd = [ffmpeg, "audio", str(src), str(out), "--comment", comment]
    else:
        cmd = [
            ffmpeg,
            "-y",
            "-i", str(src),
            "-vn",            # no video
            "-ac", "1",       # mono
            "-ar", "16000",   # 16 kHz, Whisper's native rate
            "-c:a", "pcm_s16le",
            "-metadata", f"comment={comment}",
            str(out),
        ]
    if dry_run:
        print("DRY-RUN audio extraction:")
        print("  " + " ".join(cmd))
        return out

    tool = Path(ffmpeg).name
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{tool} failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    if not out.exists():
        raise RuntimeError(f"{tool} reported success but {out} was not created")
    return out