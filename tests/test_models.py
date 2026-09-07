"""Tests for whiz.models — alias resolution and best-pick preference.

Run with: pytest tests/test_models.py
"""

from __future__ import annotations

import re
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
    # NS-15, per-class: turbo-q5_0 loses only to unquantized TURBO — never
    # to an unrelated unquantized class. Its class outranks medium, so with
    # no unquantized large-class model on disk the quantized turbo wins
    # (the old global batch let `medium` win here, which also meant `tiny`
    # could beat `large-v3-turbo-q8_0` — see the per-class repro below).
    _touch(tmp_path / "ggml-large-v3-turbo-q5_0.bin")
    _touch(tmp_path / "ggml-medium.bin")
    config = _make_isolated_config([tmp_path], monkeypatch)
    best = M.pick_best(config)
    assert best is not None
    assert best.name == "ggml-large-v3-turbo-q5_0.bin"


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
    # Only a non-canonical model present (not in PREFERENCE but
    # discoverable) — the alphabetical fallback keeps such machines working.
    _touch(tmp_path / "ggml-custom.bin")
    config = _make_isolated_config([tmp_path], monkeypatch)
    best = M.pick_best(config)
    assert best is not None
    assert best.name == "ggml-custom.bin"


def test_pick_best_empty_returns_none(tmp_path, monkeypatch):
    config = _make_isolated_config([tmp_path], monkeypatch)
    assert M.pick_best(config) is None


def test_list_known_returns_canonical_set():
    known = M.list_known()
    assert "ggml-large-v3-turbo-q5_0.bin" in known
    assert "ggml-large-v3-turbo.bin" in known
    # tiny is excluded from the canonical set entirely (NS-15, user
    # decision: useless quality — it still downloads when named explicitly).
    assert "ggml-tiny.bin" not in known
    assert "ggml-tiny-q5_0.bin" not in known
    assert len(known) == len(M.KNOWN_MODELS)
    # NS-15: every quantized variant in the canonical list must sit behind
    # its unquantized class — both lists stay honest if a class is added.
    for name in known:
        m = re.search(r"-q\d+_\d+\.bin$", name)
        if m:
            base = name[: m.start()] + ".bin"
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
    """Strip any -qN_M quantization suffix (q8_0, q5_0, and future variants
    like q4_0 — a hardcoded pair would silently mis-handle those and pin a
    misleading failure)."""
    return re.sub(r"-q\d+_\d+\.bin$", ".bin", name)


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


def _class_of(name: str) -> str:
    """Model class: 'large-v3-turbo' for ggml-large-v3-turbo-q8_0.bin."""
    return re.sub(r"-q\d+_\d+$", "", M._alias_from_name(name))


def test_preference_groups_each_class_with_unquantized_first():
    """NS-15, per-class: every class's variants are contiguous, its
    unquantized model first, quantized variants best-quality-first (q8_0
    before q5_0). The old global unquantized-batch let `tiny` outrank
    `large-v3-turbo-q8_0` — the batch property is NOT the requirement."""
    order = [_class_of(n) for n in M.PREFERENCE]
    classes = ["large-v3-turbo", "large-v3", "medium", "small", "base"]
    seen: list[str] = []
    for cls in order:
        if not seen or seen[-1] != cls:
            assert cls not in seen, f"{cls} class is not contiguous in PREFERENCE"
            seen.append(cls)
    assert seen == classes, f"classes must rank {classes}, got {seen}"
    # Within the turbo class: q8_0 ranks ahead of q5_0 (higher quality).
    assert (M.PREFERENCE.index("ggml-large-v3-turbo-q8_0.bin")
            < M.PREFERENCE.index("ggml-large-v3-turbo-q5_0.bin"))


def test_preference_and_known_models_exclude_tiny():
    """tiny is useless quality (user decision, NS-15): never listed,
    recommended, or auto-picked — explicit download still works (the name
    passes through _resolve_download_filename unchanged)."""
    for name in M.KNOWN_MODELS + M.PREFERENCE:
        assert "tiny" not in name


def test_pick_best_prefers_higher_class_quantized_over_tiny(tmp_path, monkeypatch):
    """Per-class regression (review repro): with only tiny + turbo-q8_0 on
    disk, turbo-q8_0 wins — tiny must not outrank a large-class quantized
    model just because it happens to be unquantized."""
    _touch(tmp_path / "ggml-tiny.bin")
    _touch(tmp_path / "ggml-large-v3-turbo-q8_0.bin")
    config = _make_isolated_config([tmp_path], monkeypatch)
    best = M.pick_best(config)
    assert best is not None
    assert best.name == "ggml-large-v3-turbo-q8_0.bin"


# ---------- download short-alias expansion (NS-15) ----------


def test_download_filename_expands_turbo_to_unquantized():
    # `whiz models download turbo` used to fetch ggml-turbo.bin, which does
    # not exist upstream — expansion must land on the unquantized class.
    assert M._resolve_download_filename("turbo") == "ggml-large-v3-turbo.bin"


def test_download_filename_exact_and_short_aliases():
    assert M._resolve_download_filename("large-v3") == "ggml-large-v3.bin"
    assert M._resolve_download_filename("medium") == "ggml-medium.bin"
    assert M._resolve_download_filename("large-v3-turbo-q8_0") == "ggml-large-v3-turbo-q8_0.bin"
    assert M._resolve_download_filename("ggml-large-v3-turbo.bin") == "ggml-large-v3-turbo.bin"


def test_download_filename_expands_short_quantized_class(tmp_path):
    # "turbo-q8_0" names a quantized variant explicitly: expand the class
    # (turbo -> large-v3-turbo) but honor the requested quantization —
    # never silently swap it for the unquantized file.
    assert (M._resolve_download_filename("turbo-q8_0")
            == "ggml-large-v3-turbo-q8_0.bin")


def test_download_filename_passthrough_for_unknown_names():
    # tiny and non-canonical names: no expansion, explicit informed use
    # stays possible (tiny downloads if the user insists).
    assert M._resolve_download_filename("tiny") == "ggml-tiny.bin"
    assert M._resolve_download_filename("tiny-q5_0") == "ggml-tiny-q5_0.bin"
    assert M._resolve_download_filename("ggml-large-v3-turbo-q4_0") == "ggml-large-v3-turbo-q4_0.bin"
    assert M._resolve_download_filename("my-custom-model") == "ggml-my-custom-model.bin"
