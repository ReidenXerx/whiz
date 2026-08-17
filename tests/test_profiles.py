"""Tests for whiz.profiles — embedding merge + save_profile merge behavior.

Run with: pytest tests/test_profiles.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whiz import profiles as P


# ---------- merge_embeddings math ----------

def test_merge_embeddings_no_old_returns_new():
    out, n = P.merge_embeddings([], 0, [0.1, 0.2, 0.3], new_samples=1)
    assert out == [0.1, 0.2, 0.3]
    assert n == 1


def test_merge_embeddings_no_new_returns_old():
    out, n = P.merge_embeddings([0.5, 0.5], 3, [], new_samples=0)
    assert out == [0.5, 0.5]
    assert n == 3


def test_merge_embeddings_equal_weighted_average():
    # old_samples=1, new_samples=1 -> simple mean
    out, n = P.merge_embeddings([0.0, 2.0], 1, [2.0, 0.0], new_samples=1)
    assert out == [1.0, 1.0]
    assert n == 2


def test_merge_embeddings_weighted_by_samples():
    # old has 3 samples, new has 1 -> (0*3 + 4*1)/4 = 1.0
    out, n = P.merge_embeddings([0.0], 3, [4.0], new_samples=1)
    assert out == [1.0]
    assert n == 4


def test_merge_embeddings_history_capped_at_max_weight():
    # With a huge old_samples, the old weight is capped at _MAX_HISTORY_WEIGHT (5).
    # old=0.0 with 100 samples (capped weight 5), new=6.0 with 1 sample.
    # merged = (0*5 + 6*1)/6 = 1.0  (NOT (0*100 + 6*1)/101 ~= 0.059)
    out, n = P.merge_embeddings([0.0], 100, [6.0], new_samples=1)
    assert out == [1.0]
    assert n == 101


def test_merge_embeddings_preserves_total_count_even_when_capped():
    # The cap affects the WEIGHT, not the recorded sample count.
    out, n = P.merge_embeddings([1.0], 50, [1.0], new_samples=2)
    assert n == 52


def test_merge_embeddings_zero_new_samples_treated_as_one():
    # new_samples=0 is bumped to 1 internally so it still counts.
    out, n = P.merge_embeddings([0.0, 0.0], 2, [2.0, 2.0], new_samples=0)
    # weight: old=2, new=1 -> (0*2 + 2*1)/3
    assert out == [2/3, 2/3]
    assert n == 2


# ---------- save_profile merge behavior (filesystem) ----------

def test_save_profile_creates_new(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    path = P.save_profile("Alice", [0.1, 0.2, 0.3], samples=1)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["name"] == "Alice"
    assert data["embedding"] == [0.1, 0.2, 0.3]
    assert data["samples"] == 1
    assert data["dim"] == 3


def test_save_profile_merges_with_existing_same_dim(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    # First save: samples=2, embedding all 0.0
    P.save_profile("Bob", [0.0, 0.0], samples=2)
    # Second save: samples=1, embedding all 3.0
    # merged = (0*2 + 3*1)/3 = 1.0, total samples = 3
    P.save_profile("Bob", [3.0, 3.0], samples=1)
    path = P._profile_path("Bob")
    data = json.loads(path.read_text())
    assert data["embedding"] == [1.0, 1.0]
    assert data["samples"] == 3
    assert data["dim"] == 2


def test_save_profile_dim_mismatch_replaces(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    # Save a 3-dim profile
    P.save_profile("Carol", [1.0, 2.0, 3.0], samples=4)
    # Re-save with a different dim (model swapped) -> old discarded, replaced
    P.save_profile("Carol", [9.0, 9.0], samples=1)
    path = P._profile_path("Carol")
    data = json.loads(path.read_text())
    assert data["dim"] == 2
    assert data["embedding"] == [9.0, 9.0]
    assert data["samples"] == 1


def test_save_profile_accumulates_across_many_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    # Write the same embedding 3 times; samples should accumulate to 3.
    for _ in range(3):
        P.save_profile("Dave", [0.5, 0.5], samples=1)
    data = json.loads(P._profile_path("Dave").read_text())
    assert data["samples"] == 3
    assert data["embedding"] == [0.5, 0.5]


def test_load_profiles_round_trips_merged(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    P.save_profile("Eve", [1.0, 2.0], samples=2)
    P.save_profile("Eve", [3.0, 4.0], samples=1)
    # merged = (1*2 + 3*1)/3 = 5/3, (2*2 + 4*1)/3 = 8/3
    profiles = P.load_profiles()
    assert len(profiles) == 1
    p = profiles[0]
    assert p.name == "Eve"
    assert p.samples == 3
    assert abs(p.embedding[0] - 5/3) < 1e-9
    assert abs(p.embedding[1] - 8/3) < 1e-9