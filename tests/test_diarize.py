"""Tests for whiz.diarize — model downloads, WAV reading, progress throttle.

No network, no sherpa-onnx: urlopen is faked with in-memory bodies and the
segmentation archive is built in memory.

Run with: pytest tests/test_diarize.py
"""

from __future__ import annotations

import io
import sys
import tarfile
import wave
from array import array
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whiz import config as cfg
from whiz import diarize as D


# ---------- fakes ----------

class _FakeResp(io.BytesIO):
    """urlopen() stand-in: a context-manager body with an HTTP status."""

    status = 200

    def __init__(self, body: bytes, fail_after: int | None = None):
        super().__init__(body)
        self._fail_after = fail_after

    def read(self, n=-1):
        if self._fail_after is not None and self.tell() >= self._fail_after:
            raise KeyboardInterrupt  # ^C mid-download
        return super().read(n if self._fail_after is None else min(n, 4))


def _seg_archive(extra: list[tarfile.TarInfo] | None = None, with_model: bool = True) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        names = ["model.onnx", "model.int8.onnx"] if with_model else ["README.md"]
        for name in names:
            data = f"onnx:{name}".encode()
            info = tarfile.TarInfo(f"{D.SEG_DIR_NAME}/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for info in extra or []:
            tf.addfile(info, io.BytesIO(b"x" * info.size) if info.isfile() else None)
    return buf.getvalue()


def _fake_urlopen(monkeypatch, bodies: dict[str, bytes], fail: set[str] = frozenset()):
    calls: list[str] = []

    def urlopen(req):
        url = req.full_url
        calls.append(url)
        return _FakeResp(bodies[url], fail_after=8 if url in fail else None)

    monkeypatch.setattr(D.urllib.request, "urlopen", urlopen)
    return calls


def _evil(name: str, kind: bytes = tarfile.REGTYPE, linkname: str = "") -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    info.size = 1 if kind == tarfile.REGTYPE else 0
    return info


# ---------- downloads (H, wave-3) ----------

def test_download_models_happy_path(tmp_path, monkeypatch):
    calls = _fake_urlopen(monkeypatch, {D.SEG_URL: _seg_archive(), D.EMB_URL: b"EMB" * 100})
    seg, emb = D.download_diarization_models(dest_dir=tmp_path)
    assert seg == tmp_path / D.SEG_DIR_NAME / "model.onnx"  # NS-15: unquantized
    assert emb.read_bytes() == b"EMB" * 100
    assert calls == [D.SEG_URL, D.EMB_URL]
    # No archive, staging dir, or .part left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([D.SEG_DIR_NAME, D.EMB_MODEL_FILE])


def test_interrupted_embedding_download_leaves_nothing(tmp_path, monkeypatch):
    """Wave-3: the .onnx streamed straight to its final path, so ^C left a
    truncated model that find_embedding_model reported as present forever.
    Now nothing is left and the next run downloads again."""
    body = b"E" * 1000
    _fake_urlopen(monkeypatch, {D.SEG_URL: _seg_archive(), D.EMB_URL: body}, fail={D.EMB_URL})
    with pytest.raises(KeyboardInterrupt):
        D.download_diarization_models(dest_dir=tmp_path)
    emb = tmp_path / D.EMB_MODEL_FILE
    assert not emb.exists()
    assert not emb.with_name(emb.name + ".part").exists()

    calls = _fake_urlopen(monkeypatch, {D.SEG_URL: _seg_archive(), D.EMB_URL: body})
    D.download_diarization_models(dest_dir=tmp_path)
    assert calls == [D.EMB_URL]  # segmentation already complete; embedding retried
    assert emb.read_bytes() == body


def test_interrupted_extraction_leaves_no_model_dir(tmp_path, monkeypatch):
    """The archive extracts into a staging dir; an interruption must not
    leave a half-populated seg dir that looks installed."""
    _fake_urlopen(monkeypatch, {D.SEG_URL: _seg_archive(), D.EMB_URL: b"E"})

    def boom(tf, dest):
        (dest / D.SEG_DIR_NAME).mkdir()
        (dest / D.SEG_DIR_NAME / "model.int8.onnx").write_bytes(b"trunc")
        raise KeyboardInterrupt

    monkeypatch.setattr(D, "_safe_extractall", boom)
    with pytest.raises(KeyboardInterrupt):
        D.download_diarization_models(dest_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []  # no seg dir, no staging, no tarball


def test_stale_partial_seg_dir_is_replaced(tmp_path, monkeypatch):
    """Debris of an older non-atomic extraction (dir present, no model) is
    replaced by the freshly extracted dir."""
    stale = tmp_path / D.SEG_DIR_NAME
    stale.mkdir()
    (stale / "junk").write_text("half")
    _fake_urlopen(monkeypatch, {D.SEG_URL: _seg_archive(), D.EMB_URL: b"E"})
    seg, _ = D.download_diarization_models(dest_dir=tmp_path)
    assert seg.read_bytes() == b"onnx:model.onnx"
    assert not (stale / "junk").exists()


def test_archive_without_model_raises_and_cleans_up(tmp_path, monkeypatch):
    _fake_urlopen(monkeypatch, {D.SEG_URL: _seg_archive(with_model=False), D.EMB_URL: b"E"})
    with pytest.raises(RuntimeError, match="did not produce expected model"):
        D.download_diarization_models(dest_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "member",
    [
        _evil("../escaped.txt"),
        _evil(f"{D.SEG_DIR_NAME}/../../escaped.txt"),
        _evil("/tmp/whiz-absolute-escape.txt"),
        _evil(f"{D.SEG_DIR_NAME}/link", tarfile.SYMTYPE, "../../outside"),
        _evil(f"{D.SEG_DIR_NAME}/abslink", tarfile.SYMTYPE, "/etc/passwd"),
        _evil(f"{D.SEG_DIR_NAME}/dev", tarfile.CHRTYPE),
    ],
    ids=["dotdot", "nested-dotdot", "absolute", "symlink-out", "symlink-abs", "device"],
)
def test_malicious_archive_member_rejected(tmp_path, monkeypatch, member):
    """No member may land outside the cache — checked on every Python, with
    filter="data" layered on top where the stdlib has it."""
    base = tmp_path / "cache"
    _fake_urlopen(monkeypatch, {D.SEG_URL: _seg_archive(extra=[member]), D.EMB_URL: b"E"})
    with pytest.raises(RuntimeError, match="Refusing"):
        D.download_diarization_models(dest_dir=base)
    assert not (tmp_path / "escaped.txt").exists()
    assert not Path("/tmp/whiz-absolute-escape.txt").exists()
    assert list(base.iterdir()) == []  # nothing installed, nothing left over


# ---------- _read_wav_pcm (wave-3) ----------

def _write_wav(path: Path, frames: bytes, rate=16000, channels=1, width=2) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(frames)
    return path


def test_read_wav_pcm_returns_float32_array(tmp_path):
    """array('f') (4 bytes/sample), not a list of Python floats (~4 GB per
    hour). Same values as the old divide-by-32768."""
    ints = array("h", [0, 16384, -32768, 32767, -1])
    if sys.byteorder == "big":
        ints.byteswap()
    wav = _write_wav(tmp_path / "a.wav", ints.tobytes())
    samples, sr = D._read_wav_pcm(wav)
    assert sr == 16000
    assert isinstance(samples, array) and samples.typecode == "f"
    assert list(samples) == [0.0, 0.5, -1.0, 32767 / 32768, -1 / 32768]
    # The sequence contract callers rely on: len + slicing.
    assert len(samples[1:3]) == 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"channels": 2}, "mono"),
        ({"rate": 8000}, "16000 Hz"),
        ({"width": 1}, "16-bit"),
        ({"width": 4}, "16-bit"),
    ],
)
def test_read_wav_pcm_rejects_unusable_format_with_runtime_error(tmp_path, kwargs, match):
    wav = _write_wav(tmp_path / "b.wav", b"\0" * 64, **kwargs)
    with pytest.raises(RuntimeError, match=match):
        D._read_wav_pcm(wav)


@pytest.mark.parametrize("body", [b"", b"ID3 not a wave file at all......", b"RIFF\0\0\0\0WAVE"])
def test_read_wav_pcm_non_riff_raises_runtime_error(tmp_path, body):
    """wave.Error / EOFError escaped the callers' RuntimeError mapping as a
    traceback; they are RuntimeError now."""
    bad = tmp_path / "c.wav"
    bad.write_bytes(body)
    with pytest.raises(RuntimeError, match="Not a readable PCM WAV"):
        D._read_wav_pcm(bad)


def test_read_wav_pcm_rate_check_optional(tmp_path):
    wav = _write_wav(tmp_path / "d.wav", b"\0\0" * 4, rate=8000)
    samples, sr = D._read_wav_pcm(wav, expected_rate=None)
    assert sr == 8000 and len(samples) == 4


# ---------- progress throttle (wave-3) ----------

@pytest.fixture
def piped_progress(monkeypatch, capsys):
    from whiz import ui
    monkeypatch.setattr(ui, "_is_tty", lambda: False)
    monkeypatch.setattr(D, "_ui_tty", lambda: False)
    D._reset_progress()
    yield lambda: capsys.readouterr().err.strip().splitlines()
    D._reset_progress()


def test_progress_rounds_once_and_never_duplicates_100(piped_progress):
    """Throttle on int() floor + ui.progress rounding showed 100% at 99.5%
    and then emitted 100% again. One rounded integer drives both now."""
    for done, total in [(994, 1000), (995, 1000), (999, 1000), (1000, 1000), (1000, 1000)]:
        D._progress_callback(done, total)
    assert piped_progress() == ["~ progress: 99%", "~ progress: 100%"]


def test_second_run_in_same_process_emits_again(piped_progress, tmp_path, monkeypatch):
    """_last_emitted_percent was never reset, so a second diarization in the
    same process emitted nothing until it passed the first run's last
    percent. run_diarization resets it at the start of every run."""
    for i in range(0, 101, 50):
        D._progress_callback(i, 100)
    assert piped_progress() == ["~ progress: 0%", "~ progress: 50%", "~ progress: 100%"]

    monkeypatch.setattr(D, "find_segmentation_model", lambda c: tmp_path / "seg.onnx")
    monkeypatch.setattr(D, "find_embedding_model", lambda c: tmp_path / "emb.onnx")
    D.run_diarization(tmp_path / "x.wav", cfg.Config(), dry_run=True)
    piped_progress()  # drop the dry-run banner

    for i in range(0, 101, 50):
        D._progress_callback(i, 100)
    assert piped_progress() == ["~ progress: 0%", "~ progress: 50%", "~ progress: 100%"]


def test_tty_newline_on_completion_even_when_percent_repeats(monkeypatch, capsys):
    """The completion newline went out only if the final call was a NEW
    percent; when 100% was already drawn (99.6% rounds to 100) the counter
    line was left unterminated. Exactly one newline per run now."""
    monkeypatch.setattr(D, "_ui_tty", lambda: True)
    D._reset_progress()
    for done, total in [(500, 1000), (996, 1000), (1000, 1000), (1000, 1000)]:
        D._progress_callback(done, total)
    err = capsys.readouterr().err
    assert err.endswith("\n")
    assert err.count("\n") == 1
    assert "100.0%" in err
    D._reset_progress()
