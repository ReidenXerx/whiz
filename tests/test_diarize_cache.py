"""Tests for diarize cache and profiles matching (pure-Python, no sherpa-onnx).

Run with: pytest tests/test_diarize_cache.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whiz.diarize import DiarSegment, load_diarization_cache, _write_diarization_cache, diar_cache_path
from whiz import profiles as P


# ---------- diarization cache ----------

def test_diar_cache_path_uses_string_append(tmp_path):
    """Dotted stems survive (no with_suffix splitting)."""
    wav = tmp_path / "rec.16.03.40.wav"
    wav.write_bytes(b"x")
    p = diar_cache_path(wav)
    assert p.name == "rec.16.03.40.wav.diar.json"


def test_diar_cache_round_trip(tmp_path):
    wav = tmp_path / "rec.wav"
    wav.write_bytes(b"x")
    segs = [
        DiarSegment(start=0.0, end=2.0, speaker=0),
        DiarSegment(start=2.0, end=5.0, speaker=1),
    ]
    _write_diarization_cache(wav, segs, num_speakers=2, threshold=0.9)
    loaded = load_diarization_cache(wav, num_speakers=2, threshold=0.9)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].speaker == 0
    assert loaded[1].end == 5.0


def test_diar_cache_miss_on_param_mismatch(tmp_path):
    wav = tmp_path / "rec.wav"
    wav.write_bytes(b"x")
    segs = [DiarSegment(start=0.0, end=1.0, speaker=0)]
    _write_diarization_cache(wav, segs, num_speakers=2, threshold=0.9)
    # Different num_speakers -> cache miss.
    assert load_diarization_cache(wav, num_speakers=3, threshold=0.9) is None
    # Different threshold -> cache miss.
    assert load_diarization_cache(wav, num_speakers=2, threshold=0.5) is None


def test_diar_cache_missing_file_returns_none(tmp_path):
    wav = tmp_path / "nope.wav"
    assert load_diarization_cache(wav, num_speakers=2, threshold=0.9) is None


def test_diar_cache_threshold_epsilon_tolerance(tmp_path):
    """Tiny float differences (formatting round-trips) don't invalidate the cache."""
    wav = tmp_path / "rec.wav"
    wav.write_bytes(b"x")
    segs = [DiarSegment(start=0.0, end=1.0, speaker=0)]
    _write_diarization_cache(wav, segs, num_speakers=0, threshold=0.95)
    # 0.95 vs 0.9500000001 is within epsilon -> hit.
    loaded = load_diarization_cache(wav, num_speakers=0, threshold=0.9500000001)
    assert loaded is not None


def test_diar_cache_records_and_matches_model_paths(tmp_path):
    """H1 (wave-1): the cache is keyed on the resolved model files too —
    same WAV + same params + same models hit, and the payload records the
    input identity for later verification."""
    wav = tmp_path / "rec.wav"
    wav.write_bytes(b"x" * 1000)
    seg = tmp_path / "seg.onnx"
    emb = tmp_path / "emb.onnx"
    segs = [DiarSegment(start=0.0, end=1.0, speaker=0)]
    _write_diarization_cache(wav, segs, num_speakers=2, threshold=0.9,
                             seg_model=seg, emb_model=emb)
    loaded = load_diarization_cache(wav, num_speakers=2, threshold=0.9,
                                    seg_model=seg, emb_model=emb)
    assert loaded is not None
    # The payload records the input identity.
    payload = json.loads(diar_cache_path(wav).read_text())
    assert payload["seg_model"] == str(seg)
    assert payload["emb_model"] == str(emb)
    assert payload["wav_size"] == 1000


def test_diar_cache_miss_on_model_swap(tmp_path):
    """H1: a different resolved model must never serve the old cache — a
    model upgrade would silently re-serve stale speaker labels."""
    wav = tmp_path / "rec.wav"
    wav.write_bytes(b"x" * 1000)
    segs = [DiarSegment(start=0.0, end=1.0, speaker=0)]
    _write_diarization_cache(wav, segs, num_speakers=2, threshold=0.9,
                             seg_model=tmp_path / "seg.onnx",
                             emb_model=tmp_path / "emb.onnx")
    # Same WAV, same params, different segmentation model -> miss.
    assert load_diarization_cache(
        wav, num_speakers=2, threshold=0.9,
        seg_model=tmp_path / "model.onnx", emb_model=tmp_path / "emb.onnx",
    ) is None
    # Different embedding model -> miss too.
    assert load_diarization_cache(
        wav, num_speakers=2, threshold=0.9,
        seg_model=tmp_path / "seg.onnx", emb_model=tmp_path / "emb2.onnx",
    ) is None


def test_diar_cache_miss_on_stale_wav_bytes(tmp_path):
    """H1: a re-exported video under the same filename must be a MISS —
    different size, or same size with a bumped mtime, both mean the cached
    speaker labels belong to different audio."""
    wav = tmp_path / "rec.wav"
    wav.write_bytes(b"x" * 1000)
    segs = [DiarSegment(start=0.0, end=1.0, speaker=0)]
    _write_diarization_cache(wav, segs, num_speakers=2, threshold=0.9,
                             seg_model=tmp_path / "seg.onnx",
                             emb_model=tmp_path / "emb.onnx")

    # Re-export with different content -> different size -> miss.
    wav.write_bytes(b"y" * 2000)
    assert load_diarization_cache(
        wav, num_speakers=2, threshold=0.9,
        seg_model=tmp_path / "seg.onnx", emb_model=tmp_path / "emb.onnx",
    ) is None

    # Same size, newer mtime -> miss as well.
    wav.write_bytes(b"x" * 1000)
    st = wav.stat()
    os.utime(wav, (st.st_atime, st.st_mtime + 10))
    assert load_diarization_cache(
        wav, num_speakers=2, threshold=0.9,
        seg_model=tmp_path / "seg.onnx", emb_model=tmp_path / "emb.onnx",
    ) is None


def test_diar_cache_miss_when_models_unverifiable(tmp_path):
    """H1: a cache recorded WITH models must not hit a load that cannot
    verify them (None) — unverified is never a hit."""
    wav = tmp_path / "rec.wav"
    wav.write_bytes(b"x")
    segs = [DiarSegment(start=0.0, end=1.0, speaker=0)]
    _write_diarization_cache(wav, segs, num_speakers=2, threshold=0.9,
                             seg_model=tmp_path / "seg.onnx",
                             emb_model=tmp_path / "emb.onnx")
    assert load_diarization_cache(wav, num_speakers=2, threshold=0.9) is None


def test_diar_cache_version_bump_invalidates(tmp_path):
    """A cache written by an older payload version is a MISS, not a parse
    error or a silent hit."""
    wav = tmp_path / "rec.wav"
    wav.write_bytes(b"x")
    segs = [DiarSegment(start=0.0, end=1.0, speaker=0)]
    _write_diarization_cache(wav, segs, num_speakers=2, threshold=0.9)
    path = diar_cache_path(wav)
    payload = json.loads(path.read_text())
    payload["version"] = payload["version"] - 1
    path.write_text(json.dumps(payload))
    assert load_diarization_cache(wav, num_speakers=2, threshold=0.9) is None


# ---------- profiles: cosine similarity + matching ----------

def test_cosine_similarity_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert abs(P.cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(P.cosine_similarity(a, b)) < 1e-9


def test_cosine_similarity_empty_returns_zero():
    assert P.cosine_similarity([], [1.0]) == 0.0
    assert P.cosine_similarity([1.0], []) == 0.0


def test_cosine_similarity_different_lengths_returns_none():
    """M3: mismatched dims are incomparable (a swapped embedding model
    changes every dimension's meaning) — reject with None, never a
    truncated projection that reads like a confident score."""
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0]  # shorter
    assert P.cosine_similarity(a, b) is None
    assert P.cosine_similarity(b, a) is None


def test_match_speakers_skips_dim_mismatched_profile(capsys):
    """M3: a mismatched-dim profile is skipped with a warning (no match),
    not silently truncated into a bogus score."""
    profiles = [
        P.Profile(name="Alice", embedding=[1.0, 0.0, 0.0], dim=3, created=""),
    ]
    clusters = {0: [1.0, 0.0]}  # 2-dim vs 3-dim profile
    matches = P.match_speakers(clusters, profiles, threshold=0.5)
    assert matches[0] is None
    err = capsys.readouterr().err
    assert "dimension mismatch" in err


def test_match_speakers_clean_pairs_no_warning(capsys):
    """M3: the dim-mismatch warning only fires when a pair was skipped."""
    profiles = [P.Profile(name="Alice", embedding=[1.0, 0.0], dim=2, created="")]
    clusters = {0: [1.0, 0.0]}
    matches = P.match_speakers(clusters, profiles, threshold=0.8)
    assert matches[0] is not None and matches[0][0] == "Alice"
    assert capsys.readouterr().err == ""


def test_match_speakers_assigns_above_threshold():
    profiles = [
        P.Profile(name="Alice", embedding=[1.0, 0.0], dim=2, created=""),
        P.Profile(name="Bob", embedding=[0.0, 1.0], dim=2, created=""),
    ]
    clusters = {
        0: [1.0, 0.0],  # matches Alice
        1: [0.0, 1.0],  # matches Bob
    }
    matches = P.match_speakers(clusters, profiles, threshold=0.8)
    assert matches[0] is not None and matches[0][0] == "Alice"
    assert matches[1] is not None and matches[1][0] == "Bob"


def test_match_speakers_below_threshold_returns_none():
    profiles = [P.Profile(name="Alice", embedding=[1.0, 0.0], dim=2, created="")]
    clusters = {0: [0.0, 1.0]}  # orthogonal -> score 0
    matches = P.match_speakers(clusters, profiles, threshold=0.8)
    assert matches[0] is None


def test_match_speakers_one_to_one_no_double_assignment():
    """A profile can't be assigned to two clusters even if it's the best for both."""
    profiles = [P.Profile(name="Alice", embedding=[1.0, 0.0], dim=2, created="")]
    clusters = {
        0: [1.0, 0.0],  # matches Alice at 1.0
        1: [0.99, 0.01], # also near Alice but lower score
    }
    matches = P.match_speakers(clusters, profiles, threshold=0.8)
    # Only the best cluster (0) gets Alice; cluster 1 gets None.
    assert matches[0] is not None and matches[0][0] == "Alice"
    assert matches[1] is None


def test_match_speakers_empty_profiles_all_none():
    clusters = {0: [1.0, 0.0], 1: [0.0, 1.0]}
    matches = P.match_speakers(clusters, profiles=[], threshold=0.8)
    assert matches[0] is None
    assert matches[1] is None


def test_match_speakers_empty_clusters():
    profiles = [P.Profile(name="Alice", embedding=[1.0], dim=1, created="")]
    assert P.match_speakers({}, profiles, threshold=0.8) == {}


def test_save_load_forget_profile(tmp_path, monkeypatch):
    """Profile persistence round-trip with an isolated config dir."""
    monkeypatch.setattr("whiz.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("whiz.profiles.cfg.CONFIG_DIR", tmp_path)
    # save_profile writes to profiles_dir() which reads cfg.CONFIG_DIR at call
    # time, so the monkeypatch takes effect.
    P.save_profile("Alice", [1.0, 2.0, 3.0], samples=5)
    profiles = P.load_profiles()
    assert len(profiles) == 1
    assert profiles[0].name == "Alice"
    assert profiles[0].dim == 3
    assert profiles[0].samples == 5

    assert P.forget_profile("Alice") is True
    assert P.load_profiles() == []
    assert P.forget_profile("Alice") is False  # already gone