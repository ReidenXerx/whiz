"""Tests for whiz.profiles — embedding merge + save_profile merge behavior.

Run with: pytest tests/test_profiles.py
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

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


# ---------- wave-1 M3: auto-match provenance ----------


def test_save_profile_marks_auto_match_source(tmp_path, monkeypatch):
    """M3: an auto-match creating a NEW profile records its provenance as
    'auto' — no more silently-merged auto-matches with no provenance."""
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    path = P.save_profile("Alice", [0.1, 0.2], samples=1, auto_match=True)
    data = json.loads(path.read_text())
    assert data["source"] == "auto"
    assert P.load_profiles()[0].source == "auto"


def test_save_profile_auto_match_never_touches_existing(tmp_path, monkeypatch):
    """M3: an auto-match must not merge into or replace an existing profile
    — that is how a chain of self-confirming matches silently drifted the
    stored centroid. The existing file is returned untouched."""
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    P.save_profile("Alice", [1.0, 1.0], samples=3)  # user-confirmed base
    before = P._profile_path("Alice").read_text()
    # Same name, different dim — would REPLACE under old rules; must not.
    P.save_profile("Alice", [9.0, 9.0, 9.0], samples=1, auto_match=True)
    # Same name, same dim — would MERGE under old rules; must not either.
    P.save_profile("Alice", [0.0, 0.0], samples=1, auto_match=True)
    assert P._profile_path("Alice").read_text() == before


def test_save_profile_user_confirmation_merges_and_upgrades_source(tmp_path, monkeypatch):
    """M3: a later human confirmation still merges normally and upgrades
    provenance from 'auto' back to 'user'."""
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    P.save_profile("Alice", [0.0, 0.0], samples=1, auto_match=True)
    # User confirms: same dim -> merged mean = (0*1 + 2*1)/2 = 1.0
    P.save_profile("Alice", [2.0, 2.0], samples=1)
    data = json.loads(P._profile_path("Alice").read_text())
    assert data["source"] == "user"
    assert data["samples"] == 2
    assert data["embedding"] == [1.0, 1.0]


def test_save_profile_default_is_user_source(tmp_path, monkeypatch):
    """M3: existing call-sites (cli's _save_named_profiles) keep the old
    behavior — default source is 'user' and merging still works."""
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    P.save_profile("Bob", [1.0], samples=1)
    P.save_profile("Bob", [3.0], samples=1)
    data = json.loads(P._profile_path("Bob").read_text())
    assert data["source"] == "user"
    assert data["samples"] == 2


# ---------- wave-3 audit: durability + filename collisions ----------


def test_load_profiles_warns_on_corrupt_file(tmp_path, monkeypatch, capsys):
    """Wave-3: a corrupt profile used to be skipped silently — the speaker
    just stopped being recognized. The skip now names the file."""
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    P.save_profile("Alice", [1.0, 2.0], samples=1)
    (tmp_path / "Broken.json").write_text('{"name": "Broken", "embedd', encoding="utf-8")
    (tmp_path / "Empty.json").write_text('{"name": "Empty", "embedding": []}', encoding="utf-8")
    (tmp_path / "List.json").write_text("[1, 2]", encoding="utf-8")
    profiles = P.load_profiles()
    assert [p.name for p in profiles] == ["Alice"]
    err = capsys.readouterr().err
    for name in ("Broken.json", "Empty.json", "List.json"):
        assert name in err


def test_save_profile_refuses_to_overwrite_corrupt_file(tmp_path, monkeypatch):
    """Wave-3: save_profile read a corrupt prior file as "no profile" and
    overwrote the accumulated history. It now raises (a RuntimeError the CLI
    already catches and warns on) and leaves the file byte-for-byte."""
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    bad = tmp_path / "Bob.json"
    bad.write_text('{"name": "Bob", "samples": 40, "embedding": [0.1,', encoding="utf-8")
    before = bad.read_bytes()
    for auto in (False, True):
        with pytest.raises(P.CorruptProfileError, match="not overwriting"):
            P.save_profile("Bob", [1.0, 1.0], samples=1, auto_match=auto)
        assert bad.read_bytes() == before
    assert issubclass(P.CorruptProfileError, RuntimeError)


def test_save_profile_atomic_write_keeps_old_on_failure(tmp_path, monkeypatch):
    """A crash mid-write leaves the previous profile intact, no temp debris."""
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    path = P.save_profile("Carol", [1.0, 1.0], samples=3)
    before = path.read_bytes()

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(P.os, "replace", boom)
    with pytest.raises(OSError):
        P.save_profile("Carol", [5.0, 5.0], samples=1)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Carol.json"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_save_profile_private_permissions(tmp_path, monkeypatch):
    """A voiceprint is biometric data: file 0600, created dir 0700."""
    d = tmp_path / "speakers"
    monkeypatch.setattr(P, "profiles_dir", lambda: d)
    path = P.save_profile("Dana", [1.0], samples=1)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(d.stat().st_mode) == 0o700


def test_distinct_names_with_same_sanitized_stem_stay_separate(tmp_path, monkeypatch):
    """Wave-3: "Bob Smith" / "Bob_Smith" / "Bob.Smith" all sanitized to
    Bob_Smith.json and MERGED three people's voices into one profile. The
    first keeps the plain filename; later ones get a stable hashed name."""
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    names = ["Bob Smith", "Bob_Smith", "Bob.Smith"]
    paths = [P.save_profile(n, [float(i), float(i)], samples=1) for i, n in enumerate(names)]
    assert paths[0].name == "Bob_Smith.json"
    assert len({p.name for p in paths}) == 3
    for p in paths[1:]:
        assert p.name.startswith("Bob_Smith-") and p.parent == tmp_path
    by_name = {p.name: p for p in P.load_profiles()}
    assert set(by_name) == set(names)
    for i, n in enumerate(names):
        assert by_name[n].embedding == [float(i), float(i)]
        assert by_name[n].samples == 1
        assert P._profile_path(n) == paths[i]  # stable across calls
    # Re-saving merges into the RIGHT file only.
    P.save_profile("Bob.Smith", [2.0, 2.0], samples=1)
    assert json.loads(paths[2].read_text())["samples"] == 2
    assert json.loads(paths[0].read_text())["samples"] == 1
    # Forgetting one leaves the others, and the hashed one keeps resolving.
    assert P.forget_profile("Bob Smith") is True
    assert P._profile_path("Bob.Smith") == paths[2]
    assert {p.name for p in P.load_profiles()} == {"Bob_Smith", "Bob.Smith"}


def test_legacy_profile_files_still_load_and_match(tmp_path, monkeypatch):
    """Backward compatibility: a file written under the old scheme (plain
    sanitized stem, possibly with no 'name' field) still loads, still
    resolves for its stored name, and still merges."""
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    (tmp_path / "Mary_Ann.json").write_text(
        json.dumps({"name": "Mary Ann", "dim": 2, "embedding": [1.0, 1.0], "samples": 3}),
        encoding="utf-8",
    )
    (tmp_path / "Old.json").write_text(json.dumps({"embedding": [0.5]}), encoding="utf-8")
    assert {p.name for p in P.load_profiles()} == {"Mary Ann", "Old"}
    assert P._profile_path("Mary Ann") == tmp_path / "Mary_Ann.json"
    assert P._profile_path("Old") == tmp_path / "Old.json"
    P.save_profile("Mary Ann", [1.0, 1.0], samples=1)
    assert json.loads((tmp_path / "Mary_Ann.json").read_text())["samples"] == 4


def test_profile_path_blocks_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "profiles_dir", lambda: tmp_path)
    for evil in ("../evil", "../../etc/passwd", "/abs/path", "a/../../b", "..", ""):
        p = P._profile_path(evil)
        assert p.parent == tmp_path
        assert "/" not in p.name and ".." not in p.name
