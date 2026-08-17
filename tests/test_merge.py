"""Tests for whiz.merge — overlap assignment, relabeling, formatting.

Run with: pytest tests/test_merge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whiz import merge as MR
from whiz.diarize import DiarSegment


def _seg(start, end, text="hello", words=None):
    return MR.WhisperSeg(start=start, end=end, text=text, words=words)


def test_speaker_label_letters():
    assert MR.speaker_label(0) == "Speaker A"
    assert MR.speaker_label(1) == "Speaker B"
    assert MR.speaker_label(25) == "Speaker Z"
    assert MR.speaker_label(26) == "Speaker 26"


def test_assign_speakers_max_overlap():
    """A whisper segment is labeled by the diarization speaker it overlaps most."""
    whisper = [
        _seg(0.0, 2.0, "first"),
        _seg(5.0, 7.0, "second"),
        _seg(10.0, 12.0, "third"),
    ]
    diar = [
        DiarSegment(start=0.0, end=3.0, speaker=0),    # Speaker A
        DiarSegment(start=4.0, end=8.0, speaker=1),    # Speaker B
        DiarSegment(start=9.0, end=13.0, speaker=0),   # Speaker A again
    ]
    merged = MR.assign_speakers(whisper, diar)
    labels = [lbl for _, lbl in merged]
    assert labels == ["Speaker A", "Speaker B", "Speaker A"]


def test_assign_speakers_no_diar():
    """With no diarization segments, everything falls back to Speaker A."""
    whisper = [_seg(0.0, 1.0, "x"), _seg(2.0, 3.0, "y")]
    merged = MR.assign_speakers(whisper, [])
    assert all(lbl == "Speaker A" for _, lbl in merged)


def test_speakers_by_talk_time_orders_most_first():
    merged = [
        (_seg(0.0, 10.0, "long"), "Speaker A"),    # 10s
        (_seg(0.0, 2.0, "short"), "Speaker B"),    # 2s
        (_seg(0.0, 5.0, "mid"), "Speaker A"),      # A total = 15s
        (_seg(0.0, 1.0, "tiny"), "Speaker C"),     # 1s
    ]
    order = MR.speakers_by_talk_time(merged)
    assert order == ["Speaker A", "Speaker B", "Speaker C"]


def test_speakers_in_order_of_appearance():
    merged = [
        (_seg(0, 1), "Speaker B"),
        (_seg(1, 2), "Speaker A"),
        (_seg(2, 3), "Speaker B"),
        (_seg(3, 4), "Speaker C"),
    ]
    assert MR.speakers_in_order(merged) == ["Speaker B", "Speaker A", "Speaker C"]


def test_relabel_replaces_names():
    merged = [(_seg(0, 1, "hi"), "Speaker A"), (_seg(1, 2, "yo"), "Speaker B")]
    out = MR.relabel(merged, {"Speaker A": "Alice"})
    assert out[0][1] == "Alice"
    assert out[1][1] == "Speaker B"  # untouched


def test_representative_quotes_picks_longest():
    merged = [
        (_seg(0, 1, "Yeah."), "Speaker A"),
        (_seg(1, 5, "Let me explain the whole plan in detail now."), "Speaker A"),
        (_seg(5, 6, "Ok."), "Speaker A"),
        (_seg(6, 7, "Sure."), "Speaker B"),
    ]
    quotes = MR.representative_quotes(merged)
    assert "plan in detail" in quotes["Speaker A"]
    assert quotes["Speaker B"] == "Sure."


def test_format_labeled_srt_structure():
    merged = [(_seg(0.0, 1.5, "Hello world"), "Speaker A")]
    srt = MR.format_labeled_srt(merged)
    lines = srt.split("\n")
    assert lines[0] == "1"
    assert "00:00:00,000" in lines[1]
    assert "00:00:01,500" in lines[1]
    assert lines[2] == "Speaker A: Hello world"


def test_format_dialogue_txt_merges_consecutive():
    merged = [
        (_seg(0.0, 1.0, "First."), "Speaker A"),
        (_seg(1.0, 2.0, "Second."), "Speaker A"),
        (_seg(2.0, 3.0, "Reply."), "Speaker B"),
    ]
    txt = MR.format_dialogue_txt(merged)
    blocks = txt.split("\n\n")
    assert len(blocks) == 2
    assert "Speaker A (00:00:00): First. Second." == blocks[0]
    assert "Speaker B (00:00:02): Reply." == blocks[1]


def test_parse_whisper_json_oj_format(tmp_path):
    """The standard -oj format: transcription array with timestamps/text."""
    jf = tmp_path / "out.json"
    jf.write_text(
        '{"transcription": ['
        '{"timestamps":{"from":"00:00:00,000","to":"00:00:02,000"},"text":"hi"},'
        '{"timestamps":{"from":"00:00:02,000","to":"00:00:04,500"},"text":"bye"}'
        "]}",
        encoding="utf-8",
    )
    segs = MR.parse_whisper_json(jf)
    assert len(segs) == 2
    assert segs[0].start == 0.0
    assert segs[0].end == 2.0
    assert segs[0].text == "hi"
    assert segs[1].end == 4.5
    assert segs[0].words is None  # not present in -oj


def test_parse_whisper_json_with_words(tmp_path):
    """verbose_json-style `words` arrays are captured on the words field."""
    jf = tmp_path / "out.json"
    jf.write_text(
        '{"transcription": ['
        '{"timestamps":{"from":"00:00:00,000","to":"00:00:01,000"},'
        '"text":"hello there","words":[{"word":"hello","start":0.0,"end":0.5},'
        '{"word":"there","start":0.5,"end":1.0}]}'
        "]}",
        encoding="utf-8",
    )
    segs = MR.parse_whisper_json(jf)
    assert segs[0].words is not None
    assert len(segs[0].words) == 2
    assert segs[0].words[0]["word"] == "hello"


def test_format_speakers_html_basic(tmp_path):
    merged = [(_seg(0.0, 1.5, "Hello & <welcome>"), "Speaker A")]
    html = MR.format_speakers_html(merged, title="My Meeting")
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert "My Meeting" in html
    # HTML escaping of speaker text.
    assert "&amp;" in html
    assert "&lt;welcome&gt;" in html
    assert 'class="cue"' in html


def test_format_speakers_html_no_frames_when_dir_missing(tmp_path):
    """When frames_dir has no matching segNNNN.jpg, no <img> is emitted."""
    merged = [(_seg(0.0, 1.0, "hi"), "Speaker A")]
    empty_dir = tmp_path / "frames"
    empty_dir.mkdir()
    html = MR.format_speakers_html(merged, frames_dir=empty_dir)
    assert "<img" not in html