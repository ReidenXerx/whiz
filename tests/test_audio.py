"""Tests for whiz.audio: whiz's own extractions vs the user's files.

The rule these pin: whiz overwrites, reuses or deletes a ``<stem>.wav`` only
when it carries whiz's extraction marker for that very source. A user's
``talk.wav`` next to ``talk.mov`` used to be overwritten by ffmpeg ``-y`` and
then unlinked as "intermediate".
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from whiz import audio as aud


def _riff(*, comment: bytes | None = None, rate: int = 16000, channels: int = 1) -> bytes:
    """A minimal PCM WAV, optionally with the RIFF INFO/ICMT chunk ffmpeg
    writes for ``-metadata comment=...``."""
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * channels * 2, channels * 2, 16)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if comment is not None:
        value = comment + b"\0"
        if len(value) % 2:
            value += b"\0"
        info = b"INFO" + b"ICMT" + struct.pack("<I", len(comment) + 1) + value
        chunks += b"LIST" + struct.pack("<I", len(info)) + info
    data = b"\0\0" * 160
    chunks += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def test_marker_identifies_whiz_extraction_from_this_source(tmp_path):
    wav = tmp_path / "talk.wav"
    wav.write_bytes(_riff(comment=b"whiz-extracted:talk.mov"))
    assert aud.is_whiz_extracted(wav)
    assert aud.is_whiz_extracted(wav, tmp_path / "talk.mov")
    # A same-stem sibling's audio is not this video's audio.
    assert not aud.is_whiz_extracted(wav, tmp_path / "talk.mp4")
    # The NUL terminator stops a prefix match.
    assert not aud.is_whiz_extracted(wav, tmp_path / "talk.mo")


def test_unmarked_wav_is_not_whiz_extracted(tmp_path):
    wav = tmp_path / "talk.wav"
    wav.write_bytes(_riff())
    assert not aud.is_whiz_extracted(wav, tmp_path / "talk.mov")
    assert not aud.is_whiz_extracted(tmp_path / "missing.wav")


def test_extraction_dir_spares_a_foreign_wav(tmp_path):
    video = tmp_path / "talk.mov"
    assert aud.extraction_dir(video) is None  # nothing there: beside the video

    (tmp_path / "talk.wav").write_bytes(_riff(comment=b"whiz-extracted:talk.mov"))
    assert aud.extraction_dir(video) is None  # whiz's own: reuse the spot

    (tmp_path / "talk.wav").write_bytes(_riff())  # the user's recording
    dest = aud.extraction_dir(video)
    assert dest is not None and dest.is_dir() and dest != tmp_path


def test_extract_audio_refuses_to_overwrite_a_foreign_wav(tmp_path):
    video = tmp_path / "talk.mov"
    video.write_bytes(b"video")
    mine = tmp_path / "talk.wav"
    mine.write_bytes(b"my studio recording")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        aud.extract_audio(video, "ffmpeg-must-not-run")
    assert mine.read_bytes() == b"my studio recording"


def test_extract_audio_writes_the_marker(tmp_path, monkeypatch):
    video = tmp_path / "talk.mov"
    video.write_bytes(b"video")
    seen = {}

    def fake_run(cmd, capture_output, text):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(_riff())
        return type("P", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(aud.subprocess, "run", fake_run)
    aud.extract_audio(video, "ffmpeg")
    i = seen["cmd"].index("-metadata")
    assert seen["cmd"][i + 1] == "comment=whiz-extracted:talk.mov"


def test_is_diarization_ready(tmp_path):
    good = tmp_path / "a.wav"
    good.write_bytes(_riff())
    assert aud.is_diarization_ready(good)

    hi_rate = tmp_path / "b.wav"
    hi_rate.write_bytes(_riff(rate=44100))
    assert not aud.is_diarization_ready(hi_rate)

    stereo = tmp_path / "c.wav"
    with wave.open(str(stereo), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0" * 64)
    assert not aud.is_diarization_ready(stereo)

    m4a = tmp_path / "d.m4a"
    m4a.write_bytes(b"not a wav")
    assert not aud.is_diarization_ready(m4a)
