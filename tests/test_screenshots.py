"""Tests for whiz.screenshots — frame naming, manifest round-trip, path helpers.

Run with: pytest tests/test_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whiz import merge as MR
from whiz import screenshots as SC


def _seg(start, end, text="hello"):
    return MR.WhisperSeg(start=start, end=end, text=text)


def test_frame_name_zero_padded():
    assert SC._frame_name(1) == "seg0001.jpg"
    assert SC._frame_name(272) == "seg0272.jpg"
    assert SC._frame_name(7) == "seg0007.jpg"


def test_frames_dir_for_uses_string_append():
    """Dotted stems like '...16.03.40' must not be split by with_suffix."""
    base = Path("/tmp/Screen Recording 2026-08-14 at 16.03.40")
    d = SC.frames_dir_for(base)
    assert d.name == "Screen Recording 2026-08-14 at 16.03.40.frames"
    assert "16.03.40.frames" in str(d)


def test_frames_manifest_path_uses_string_append():
    base = Path("/tmp/rec.16.03.40")
    p = SC.frames_manifest_path(base)
    assert p.name == "rec.16.03.40.frames.json"
    # The '.40' part survives (not turned into '.frames.json' alone).
    assert ".16.03.40.frames.json" in p.name


def test_extract_frames_dry_run_names_entries(tmp_path):
    """Dry run doesn't call ffmpeg but still assigns frame filenames."""
    merged = [
        (_seg(0.0, 2.0, "first"), "Speaker A"),
        (_seg(2.0, 4.0, "second"), "Speaker B"),
    ]
    out_dir = tmp_path / "frames"
    entries = SC.extract_segment_frames(
        tmp_path / "fake.mp4", merged, out_dir, ffmpeg="ffmpeg", dry_run=True,
    )
    assert len(entries) == 2
    assert entries[0].index == 1
    assert entries[0].frame == "seg0001.jpg"
    assert entries[1].frame == "seg0002.jpg"
    assert entries[0].speaker == "Speaker A"
    assert entries[0].start == 0.0


def test_manifest_round_trip(tmp_path):
    entries = [
        SC.FrameEntry(index=1, start=0.0, end=2.5, speaker="Alice", text="hello", frame="seg0001.jpg"),
        SC.FrameEntry(index=2, start=2.5, end=5.0, speaker="Bob", text="world", frame="seg0002.jpg"),
    ]
    out_dir = tmp_path / "rec.frames"
    out_dir.mkdir()
    manifest = tmp_path / "rec.frames.json"
    SC.write_manifest(entries, out_dir, manifest)
    assert manifest.exists()

    loaded = SC.load_manifest(manifest)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].speaker == "Alice"
    assert loaded[1].frame == "seg0002.jpg"
    assert loaded[0].index == 1


def test_load_manifest_missing_returns_none(tmp_path):
    assert SC.load_manifest(tmp_path / "nope.json") is None


def test_load_manifest_skips_malformed_rows(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text(
        '{"segments": ['
        '{"index":1,"start":0.0,"end":1.0,"speaker":"A","text":"ok","frame":"seg0001.jpg"},'
        '{"index":"oops"}'
        ']}',
        encoding="utf-8",
    )
    loaded = SC.load_manifest(manifest)
    assert loaded is not None
    assert len(loaded) == 1  # malformed row skipped, valid one kept