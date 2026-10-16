"""Tests for whiz.config — the TOML emitter and save() durability.

Run with: pytest tests/test_config.py
"""

from __future__ import annotations

import datetime as dt
import os
import stat
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whiz import config as cfg


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    """Point CONFIG_DIR/CONFIG_PATH at a not-yet-existing tmp dir."""
    d = tmp_path / "whiz"
    monkeypatch.setattr(cfg, "CONFIG_DIR", d)
    monkeypatch.setattr(cfg, "CONFIG_PATH", d / "config.toml")
    return d


# A config.toml as another writer (Mynah, a feature branch) may leave it:
# a [mynah] table with a nested sub-table, datetimes, mixed and nested arrays.
_FOREIGN = '''\
model = "large-v3"
threads = 4
dictate_hotkey = "cmd+shift+d"
imported_at = 2026-09-22T17:01:14Z
local_dt = 2026-09-22T17:01:14.250000
day = 2026-09-22
alarm = 07:30:00
mixed = [1, "two", 3.5, true]
nested = [[1, 2], ["a", "b"], []]
points = [{ x = 1, y = 2 }, { x = 3, y = 4 }]
"odd key" = "quoted"

[mynah]
engine = "cpp"
version = 2
ratio = 0.25
words = ["alpha", "beta"]

[mynah.tuning]
vad_threshold = 0.5
'''


def test_save_roundtrips_foreign_tables_datetimes_and_nested_lists(cfg_home):
    """Wave-3: save() wrote foreign dicts/datetimes/nested lists as Python
    repr — invalid TOML, so the NEXT load() failed "config.toml is corrupt"
    for every command. load → save → load must now preserve every value."""
    cfg_home.mkdir()
    cfg.CONFIG_PATH.write_text(_FOREIGN, encoding="utf-8")
    before = tomllib.loads(_FOREIGN)

    config = cfg.load()
    config.language = "en"
    cfg.save(config)

    after = tomllib.loads(cfg.CONFIG_PATH.read_text(encoding="utf-8"))
    for key, value in before.items():
        assert after[key] == value, key
    assert after["mynah"]["tuning"] == {"vad_threshold": 0.5}
    assert isinstance(after["imported_at"], dt.datetime)
    assert after["language"] == "en"
    # And the regular load path still works.
    assert cfg.load().language == "en"


def test_emit_toml_tables_follow_top_level_keys():
    """Keys after a [table] header would land INSIDE that table."""
    text = cfg._emit_toml({"t": {"a": 1}, "top": "x"})
    assert text.index("top =") < text.index("[t]")
    assert tomllib.loads(text) == {"t": {"a": 1}, "top": "x"}


def test_escape_all_control_characters_roundtrip(cfg_home):
    """Wave-3: only \\n \\r \\t were escaped; ESC (an ANSI color pasted into
    a prompt), NUL, DEL etc. are just as illegal raw in a basic string."""
    nasty = "red \x1b[31m bell \x07 nul \x00 del \x7f nl \n tab \t quote \" bs \\"
    config = cfg.Config(ai_model=nasty, extra_args=[nasty, "plain"])
    cfg.save(config)
    raw = cfg.CONFIG_PATH.read_bytes()
    assert b"\x1b" not in raw and b"\x00" not in raw and b"\x7f" not in raw
    assert b"\\u001B" in raw
    loaded = cfg.load()
    assert loaded.ai_model == nasty
    assert loaded.extra_args == [nasty, "plain"]


def test_save_refuses_unrepresentable_and_keeps_file(cfg_home):
    """A value with no TOML spelling must never reach disk as invalid TOML:
    save() raises RuntimeError and the existing file is left byte-for-byte."""
    cfg_home.mkdir()
    cfg.CONFIG_PATH.write_text('model = "keep"\n', encoding="utf-8")
    before = cfg.CONFIG_PATH.read_bytes()
    config = cfg.load()
    config.extra_args = [object()]  # type: ignore[list-item]
    with pytest.raises(RuntimeError, match="refusing to save config"):
        cfg.save(config)
    assert cfg.CONFIG_PATH.read_bytes() == before
    assert list(cfg_home.iterdir()) == [cfg.CONFIG_PATH]  # no temp debris


def test_save_omits_none_still(cfg_home):
    """Tri-state None is never emitted (`= None` is invalid TOML)."""
    cfg.save(cfg.Config())
    text = cfg.CONFIG_PATH.read_text(encoding="utf-8")
    assert "auto_diarization_setup" not in text
    assert "None" not in text


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_save_creates_private_file_and_dir(cfg_home):
    """Wave-3: config.toml can hold ai_api_key — 0600, and a dir whiz
    creates is 0700."""
    cfg.save(cfg.Config(ai_api_key="sk-secret"))
    assert stat.S_IMODE(cfg.CONFIG_PATH.stat().st_mode) == 0o600
    assert stat.S_IMODE(cfg_home.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_save_tightens_existing_world_readable_file(cfg_home):
    cfg_home.mkdir()
    cfg.CONFIG_PATH.write_text('model = "x"\n', encoding="utf-8")
    os.chmod(cfg.CONFIG_PATH, 0o644)
    cfg.save(cfg.load())
    assert stat.S_IMODE(cfg.CONFIG_PATH.stat().st_mode) == 0o600


def test_save_is_atomic_on_write_failure(cfg_home, monkeypatch):
    """A crash mid-write must leave the old file intact (temp + os.replace),
    and no temp file behind."""
    cfg_home.mkdir()
    cfg.CONFIG_PATH.write_text('model = "old"\n', encoding="utf-8")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(cfg.os, "replace", boom)
    with pytest.raises(OSError):
        cfg.save(cfg.Config(model="new"))
    assert cfg.CONFIG_PATH.read_text(encoding="utf-8") == 'model = "old"\n'
    assert list(cfg_home.iterdir()) == [cfg.CONFIG_PATH]


def test_save_writes_through_symlink(cfg_home, tmp_path):
    """A dotfile-managed (symlinked) config keeps its link."""
    cfg_home.mkdir()
    real = tmp_path / "dotfiles" / "config.toml"
    real.parent.mkdir()
    real.write_text('model = "x"\n', encoding="utf-8")
    cfg.CONFIG_PATH.symlink_to(real)
    cfg.save(cfg.Config(model="y"))
    assert cfg.CONFIG_PATH.is_symlink()
    assert tomllib.loads(real.read_text(encoding="utf-8"))["model"] == "y"
