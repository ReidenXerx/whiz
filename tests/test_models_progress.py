"""Download progress: machine lines for the app, silence for a terminal,
and the atomic write unchanged either way."""

from __future__ import annotations

import io

import pytest

from whiz import models as M
from whiz import ui


class _Resp(io.BytesIO):
    def __init__(self, data: bytes, length: str | None):
        super().__init__(data)
        self.headers = {} if length is None else {"Content-Length": length}


def test_piped_download_reports_whole_percent_steps(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ui, "_is_tty", lambda: False)
    data = b"x" * (3 * 1024 * 1024)
    target = tmp_path / "model.bin"
    M._stream_to_atomic(_Resp(data, str(len(data))), target)
    assert target.read_bytes() == data
    lines = [l for l in capsys.readouterr().err.splitlines() if l.startswith("~ progress:")]
    assert lines == ["~ progress: 33%", "~ progress: 66%", "~ progress: 100%"]


def test_terminal_download_stays_silent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ui, "_is_tty", lambda: True)
    M._stream_to_atomic(_Resp(b"abc", "3"), tmp_path / "m.bin")
    captured = capsys.readouterr()
    assert "progress" not in captured.err + captured.out


@pytest.mark.parametrize("length", [None, "0", "garbage"])
def test_unknown_length_downloads_without_progress(tmp_path, monkeypatch, capsys, length):
    monkeypatch.setattr(ui, "_is_tty", lambda: False)
    M._stream_to_atomic(_Resp(b"abc", length), tmp_path / "m.bin")
    assert (tmp_path / "m.bin").read_bytes() == b"abc"
    assert "~ progress" not in capsys.readouterr().err


def test_interrupted_download_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "_is_tty", lambda: False)

    class Broken(_Resp):
        def read(self, n=-1):
            raise ConnectionResetError("drop")

    target = tmp_path / "m.bin"
    with pytest.raises(ConnectionResetError):
        M._stream_to_atomic(Broken(b"abc", "3"), target)
    assert not target.exists() and not (tmp_path / "m.bin.part").exists()
