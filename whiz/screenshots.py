"""Video screenshot extraction — one frame per transcribed segment.

For video inputs, capturing what was on screen alongside each transcribed
segment lets the transcript + frames be fed to a vision LLM (phase 3) for
summary/action-items/analysis. This module extracts one JPEG per segment,
taken at ``segment.start``, using ffmpeg's seek-before-input mode (fast,
accurate enough for keyframe-adjacent timestamps).

Frames are named ``segNNNN.jpg`` to match the 1-based segment index so the
transcript and frames join cleanly. A lightweight manifest (``.frames.json``)
records the per-segment metadata + frame *path* (never bytes) — see the
two-layer note in the plan: the manifest stays small and re-runnable, while
the self-contained HTML artifact (phase 4) inlines frames as base64.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from whiz import audio as aud
from whiz.merge import WhisperSeg


@dataclass
class FrameEntry:
    """One row of the frames manifest: a transcript segment + its frame path."""
    index: int
    start: float
    end: float
    speaker: str
    text: str
    frame: str  # filename within the frames dir (empty if extraction failed)


def find_ffmpeg(configured: str = "") -> str:
    """Locate the media tool (mirrors audio.find_ffmpeg, whiz-media included)."""
    if configured:
        return configured
    found = shutil.which("ffmpeg") or shutil.which(aud.MEDIA_HELPER)
    if not found:
        raise RuntimeError("ffmpeg not found on PATH — set ffmpeg in config or install it.")
    return found


def _frame_name(index: int) -> str:
    return f"seg{index:04d}.jpg"


def extract_segment_frames(
    video: Path,
    segments: list[tuple[WhisperSeg, str]],
    out_dir: Path,
    ffmpeg: str = "",
    quality: int = 2,
    width: int = 1280,
    dry_run: bool = False,
) -> list[FrameEntry]:
    """Extract one JPEG per segment at its start timestamp.

    ``segments`` is the merged ``[(WhisperSeg, label), ...]`` list. Frames go
    into ``out_dir`` (a ``<stem>.frames/`` directory). Returns the manifest
    entries (one per segment, in order). Segments whose frame extraction fails
    keep an empty ``frame`` field so the manifest still aligns by index.

    ``quality`` is ffmpeg's ``-q:v`` (2 = high, lower = better). ``width`` is
    the target pixel width; 0 disables scaling (native resolution). Downscaling
    keeps frame sizes reasonable for LLM context windows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[FrameEntry] = []
    for i, (seg, label) in enumerate(segments, start=1):
        text = " ".join(seg.text.split())
        entry = FrameEntry(
            index=i,
            start=seg.start,
            end=seg.end,
            speaker=label,
            text=text,
            frame="",
        )
        frame_path = out_dir / _frame_name(i)
        if dry_run:
            print(f"DRY-RUN frame {i}: -ss {seg.start:.3f} -> {frame_path}")
            entry.frame = frame_path.name
            entries.append(entry)
            continue

        ok = _extract_one(video, seg.start, frame_path, ffmpeg, quality, width)
        if ok:
            entry.frame = frame_path.name
        else:
            print(f"Warning: failed to extract frame {i} at t={seg.start:.3f}s", file=sys.stderr)
        entries.append(entry)
    return entries


def _extract_one(
    video: Path,
    ts: float,
    out: Path,
    ffmpeg: str,
    quality: int,
    width: int,
) -> bool:
    """Extract a single frame at timestamp ``ts`` via seek-before-input.

    Returns True on success. Seek-before-input (``-ss`` before ``-i``) is fast
    because ffmpeg seeks to the nearest keyframe before decoding.

    Success means THIS run produced the file: any ``out`` left by a previous
    run is deleted first. ffmpeg exits 0 without writing a frame when the
    seek lands past the last frame, and "rc 0 and the file exists" then
    passed off the stale ``segNNNN.jpg`` of an older video as this
    segment's frame (wave-3 audit). An empty file is a failure too.
    """
    out.unlink(missing_ok=True)
    scale = width if width and width > 0 else 0
    if aud.is_media_helper(ffmpeg):
        # whiz-media fails (rc 1) past the last frame rather than exiting 0
        # with nothing written; the file checks below hold either way.
        cmd = [ffmpeg, "frame", str(video), f"{max(0.0, ts):.3f}", str(out),
               "--quality", str(quality), "--width", str(scale)]
    else:
        cmd = [
            ffmpeg,
            "-y",
            "-ss", f"{max(0.0, ts):.3f}",
            "-i", str(video),
            "-frames:v", "1",
            "-q:v", str(quality),
        ]
        if scale:
            cmd += ["-vf", f"scale={scale}:-1"]
        cmd.append(str(out))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and out.exists() and out.stat().st_size > 0


def frames_dir_for(of_base: Path) -> Path:
    """The ``<stem>.frames/`` directory that holds extracted JPEGs."""
    return Path(str(of_base) + ".frames")


def frames_manifest_path(of_base: Path) -> Path:
    """The ``<stem>.frames.json`` manifest path (paths only, no bytes)."""
    return Path(str(of_base) + ".frames.json")


def write_manifest(entries: list[FrameEntry], out_dir: Path, manifest_path: Path) -> Path:
    """Write the frames manifest JSON (paths relative to out_dir, never bytes)."""
    payload = {
        "version": 1,
        "frames_dir": out_dir.name,
        "count": len(entries),
        "segments": [
            {
                "index": e.index,
                "start": e.start,
                "end": e.end,
                "speaker": e.speaker,
                "text": e.text,
                "frame": e.frame,
            }
            for e in entries
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def load_manifest(manifest_path: Path) -> list[FrameEntry] | None:
    """Load a frames manifest. Returns None if missing or unreadable.

    Malformed rows are skipped (they can't be reconstructed), but never
    silently anymore (wave-1 audit, L-low): the count of skipped rows is
    reported in one stderr warning, so a manifest that silently lost rows
    to partial writes / manual edits surfaces instead of quietly producing
    a shorter transcript.
    """
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    segs = data.get("segments", [])
    out: list[FrameEntry] = []
    skipped = 0
    for s in segs:
        try:
            out.append(FrameEntry(
                index=int(s["index"]),
                start=float(s["start"]),
                end=float(s["end"]),
                speaker=str(s.get("speaker", "")),
                text=str(s.get("text", "")),
                frame=str(s.get("frame", "")),
            ))
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
    if skipped:
        print(
            f"Warning: skipped {skipped} malformed row(s) in "
            f"{manifest_path.name}.",
            file=sys.stderr,
        )
    return out
