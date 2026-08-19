"""Tests for whiz.models — alias resolution and best-pick preference.

Run with: pytest tests/test_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whiz import config as cfg
from whiz import models as M


def _make_config(search_dirs: list[Path]) -> cfg.Config:
    """Config that only scans the given dirs (no built-in defaults polluting tests)."""
    config = cfg.Config()
    # We can't easily override the built-in dirs, so point model_dirs at our
    # tmp dir and rely on discover() scanning it. Built-in dirs may not exist
    # in CI, so they won't contribute models.
    config.model_dirs = [str(d) for d in search_dirs]
    return config


def _make_isolated_config(search_dirs: list[Path], monkeypatch) -> cfg.Config:
    """Config whose model_search_dirs is ONLY the given dirs (built-ins cleared).

    Use this for tests that assert on pick_best / discover with specific model
    sets, so real models on the host machine don't leak in via the built-in
    DEFAULT_MODEL_SEARCH_DIRS.
    """
    monkeypatch.setattr(cfg, "DEFAULT_MODEL_SEARCH_DIRS", [])
    config = cfg.Config()
    config.model_dirs = [str(d) for d in search_dirs]
    return config


def _touch(path: Path, size_bytes: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size_bytes)


def test_alias_from_name_strips_ggml_and_bin():
    assert M._alias_from_name("ggml-large-v3-turbo-q5_0.bin") == "large-v3-turbo-q5_0"
    assert M._alias_from_name("ggml-medium.bin") == "medium"
    assert M._alias_from_name("foo.bin") == "foo"
    assert M._alias_from_name("ggml-foo") == "foo"


def test_short_alias_turbo_collapse():
    assert M._short_alias("large-v3-turbo-q5_0") == "turbo"
    assert M._short_alias("large-v3-turbo") == "turbo"
    assert M._short_alias("large-v3") == "large-v3"
    assert M._short_alias("medium-q5_0") == "medium-q5_0"


def test_discover_finds_ggml_bin(tmp_path):
    _touch(tmp_path / "ggml-large-v3-turbo-q5_0.bin")
    _touch(tmp_path / "ggml-medium.bin")
    _touch(tmp_path / "not-a-model.txt")  # ignored
    _touch(tmp_path / "random.bin")       # ignored (no ggml- prefix)
    config = _make_config([tmp_path])
    found = {m.alias: m for m in M.discover(config)}
    assert "large-v3-turbo-q5_0" in found
    assert "medium" in found
    assert "random" not in found


def test_resolve_exact_alias(tmp_path):
    _touch(tmp_path / "ggml-large-v3.bin")
    config = _make_config([tmp_path])
    p = M.resolve("large-v3", config)
    assert p is not None
    assert p.name == "ggml-large-v3.bin"


def test_resolve_turbo_short_alias(tmp_path):
    _touch(tmp_path / "ggml-large-v3-turbo-q5_0.bin")
    config = _make_config([tmp_path])
    p = M.resolve("turbo", config)
    assert p is not None
    assert p.name == "ggml-large-v3-turbo-q5_0.bin"


def test_resolve_direct_path(tmp_path):
    f = tmp_path / "my-custom-model.bin"
    _touch(f)
    config = _make_config([tmp_path])
    p = M.resolve(str(f), config)
    assert p == f


def test_resolve_unknown_returns_none(tmp_path):
    config = _make_config([tmp_path])
    assert M.resolve("nonexistent-model", config) is None


def test_pick_best_prefers_turbo_q5(tmp_path, monkeypatch):
    # Place a lower-preference and the top-preference model; pick_best must
    # choose turbo-q5_0 even if medium is also present.
    _touch(tmp_path / "ggml-medium.bin")
    _touch(tmp_path / "ggml-large-v3-turbo-q5_0.bin")
    _touch(tmp_path / "ggml-small.bin")
    config = _make_isolated_config([tmp_path], monkeypatch)
    best = M.pick_best(config)
    assert best is not None
    assert best.name == "ggml-large-v3-turbo-q5_0.bin"


def test_pick_best_falls_back_to_anything(tmp_path, monkeypatch):
    # Only a small model present (not in PREFERENCE but discoverable).
    _touch(tmp_path / "ggml-tiny.bin")
    config = _make_isolated_config([tmp_path], monkeypatch)
    best = M.pick_best(config)
    assert best is not None
    assert best.name == "ggml-tiny.bin"


def test_pick_best_empty_returns_none(tmp_path, monkeypatch):
    config = _make_isolated_config([tmp_path], monkeypatch)
    assert M.pick_best(config) is None


def test_list_known_returns_canonical_set():
    known = M.list_known()
    assert "ggml-large-v3-turbo-q5_0.bin" in known
    assert "ggml-tiny.bin" in known
    assert len(known) == len(M.KNOWN_MODELS)


def test_find_vad_model_finds_silero(tmp_path, monkeypatch):
    _touch(tmp_path / "ggml-silero-v5.1.2.bin")
    config = _make_isolated_config([tmp_path], monkeypatch)
    p = M.find_vad_model(config)
    assert p is not None
    assert "silero" in p.name


def test_find_vad_model_prefers_known_versions(tmp_path, monkeypatch):
    _touch(tmp_path / "ggml-silero-v6.2.0.bin")
    _touch(tmp_path / "ggml-silero-v5.1.2.bin")
    config = _make_isolated_config([tmp_path], monkeypatch)
    p = M.find_vad_model(config)
    # v5.1.2 is listed first in VAD_MODELS (preferred).
    assert p is not None
    assert p.name == "ggml-silero-v5.1.2.bin"


# ---------- download name resolution (regression: `download turbo` 404'd) ----------

def test_resolve_download_name_expands_turbo_alias():
    """`whiz models download turbo` must not become the nonexistent ggml-turbo.bin."""
    assert M.resolve_download_name("turbo") == "ggml-large-v3-turbo-q5_0.bin"


def test_resolve_download_name_prefers_exact_known_model():
    assert M.resolve_download_name("large-v3") == "ggml-large-v3.bin"
    assert M.resolve_download_name("medium") == "ggml-medium.bin"
    assert M.resolve_download_name("base") == "ggml-base.bin"


def test_resolve_download_name_passes_through_full_filenames():
    assert M.resolve_download_name("ggml-large-v3.bin") == "ggml-large-v3.bin"
    assert M.resolve_download_name("ggml-large-v3") == "ggml-large-v3.bin"


def test_resolve_download_name_handles_quantized_alias():
    assert M.resolve_download_name("large-v3-turbo-q5_0") == "ggml-large-v3-turbo-q5_0.bin"


def test_resolve_download_name_unknown_falls_through():
    """An unrecognized name still gets a plausible URL rather than an exception."""
    assert M.resolve_download_name("brand-new-model") == "ggml-brand-new-model.bin"


def test_resolve_download_name_every_alias_maps_to_a_known_model():
    for alias in ("turbo", "large-v3", "medium", "small", "base", "tiny"):
        assert M.resolve_download_name(alias) in M.KNOWN_MODELS
