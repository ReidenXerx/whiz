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


def test_resolve_turbo_short_alias(tmp_path, monkeypatch):
    # Isolated, not _make_config: a short alias resolves only when it matches
    # exactly one model, and _make_config leaves DEFAULT_MODEL_SEARCH_DIRS in
    # play. On any machine with both ggml-large-v3-turbo.bin and
    # ggml-large-v3-turbo-q5_0.bin in ~/.cache/whisper, "turbo" matched two and
    # resolve() correctly returned None — so the test failed for a real user
    # while passing in CI and on machines with one turbo model.
    _touch(tmp_path / "ggml-large-v3-turbo-q5_0.bin")
    config = _make_isolated_config([tmp_path], monkeypatch)
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


def test_pick_best_prefers_unquantized_over_quantized(tmp_path, monkeypatch):
    # NS-15: pick_best must never choose a quantized model while its
    # unquantized class exists on disk. turbo-q5_0 is the old default —
    # now it loses to unquantized turbo, and even to medium when no
    # unquantized large-class model exists.
    _touch(tmp_path / "ggml-large-v3-turbo-q5_0.bin")
    _touch(tmp_path / "ggml-medium.bin")
    config = _make_isolated_config([tmp_path], monkeypatch)
    best = M.pick_best(config)
    assert best is not None
    assert best.name == "ggml-medium.bin"


def test_pick_best_prefers_turbo_when_unquantized_exists(tmp_path, monkeypatch):
    # The happy case: unquantized turbo beats its q5_0 sibling even when
    # the quantized file was "the default" for years.
    _touch(tmp_path / "ggml-large-v3-turbo-q5_0.bin")
    _touch(tmp_path / "ggml-large-v3-turbo.bin")
    config = _make_isolated_config([tmp_path], monkeypatch)
    best = M.pick_best(config)
    assert best is not None
    assert best.name == "ggml-large-v3-turbo.bin"


def test_pick_best_quantized_only_as_last_resort(tmp_path, monkeypatch):
    # A quantized model resolves only when NOTHING unquantized is on disk
    # — the fallback keeps a disk-constrained machine working, but it is
    # the last resort, not a preference.
    _touch(tmp_path / "ggml-large-v3-turbo-q5_0.bin")
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
    assert "ggml-large-v3-turbo.bin" in known
    assert "ggml-tiny.bin" in known
    assert len(known) == len(M.KNOWN_MODELS)
    # NS-15: every quantized variant in the canonical list must sit behind
    # its unquantized class — both lists stay honest if a class is added.
    for name in known:
        if "-q" in name:
            base = name.replace("-q8_0", "").replace("-q5_0", "")
            assert base in known, f"quantized {name} has no unquantized {base}"


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


# ---------- NS-15: PREFERENCE ordering invariants ----------


def _unquantized_base(name: str) -> str:
    return name.replace("-q8_0", "").replace("-q5_0", "")


def test_preference_covers_exactly_known_models():
    """PREFERENCE must be a reordering of KNOWN_MODELS: a model added to
    one list without a deliberate slot in the other fails loudly here."""
    assert set(M.PREFERENCE) == set(M.KNOWN_MODELS)
    assert len(M.PREFERENCE) == len(set(M.PREFERENCE))  # no duplicates


def test_preference_quantized_always_behind_unquantized_base():
    """NS-15: every -q* entry must sit AFTER its unquantized class, so
    pick_best never chooses a quantized model while its base exists on disk."""
    for name in M.PREFERENCE:
        if "-q" not in name:
            continue
        base = _unquantized_base(name)
        assert base in M.PREFERENCE, f"{name} has no unquantized {base}"
        assert M.PREFERENCE.index(base) < M.PREFERENCE.index(name), (
            f"{name} must rank behind {base} (NS-15)")


def test_preference_all_unquantized_before_any_quantized():
    """NS-15: the unquantized models form one leading batch — no quantized
    model is reachable while ANY unquantized model exists on disk."""
    quantized = [i for i, n in enumerate(M.PREFERENCE) if "-q" in n]
    unquantized = [i for i, n in enumerate(M.PREFERENCE) if "-q" not in n]
    assert unquantized, "PREFERENCE must list unquantized models"
    assert quantized, "PREFERENCE must keep quantized fallbacks (last resort)"
    assert max(unquantized) < min(quantized)
