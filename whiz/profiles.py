"""Speaker voice profiles — cross-recording speaker recognition.

When a user names a speaker (via `--name-speakers` or `--speakers-names`),
whiz can save a *voice profile*: a fixed-size embedding vector for that
speaker cluster, computed with the same sherpa-onnx embedding extractor used
for diarization. On later recordings, each detected cluster's embedding is
compared (cosine similarity) to the stored profiles, and a name is
auto-assigned when the best match exceeds ``speaker_match_threshold``
(config, default 0.8).

Profiles live at ``~/.config/whiz/speakers/<Name>.json``::

    {
      "name": "Alice",
      "dim": 256,
      "embedding": [0.0123, -0.0456, ...],
      "created": "2026-08-17T01:11:55Z",
      "samples": 14
    }

The store uses simple JSON files (one per name) so they're inspectable and
easy to delete. Profiles **merge** across recordings: each time a speaker is
re-named, the new cluster's embedding is combined with any existing one via a
sample-weighted running mean (see ``save_profile``), so the stored voice
profile grows more accurate over time instead of being overwritten by each
run. sherpa-onnx is an optional dependency, imported lazily.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from whiz import config as cfg
from whiz.diarize import DiarSegment, _import_sherpa, _read_wav_pcm, find_embedding_model


def profiles_dir() -> Path:
    """Directory holding per-name speaker profile JSON files."""
    return cfg.CONFIG_DIR / "speakers"


def _profile_path(name: str) -> Path:
    """Path of a single profile, sanitized so the name is filename-safe."""
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip().replace(" ", "_")
    if not safe:
        safe = "speaker"
    return profiles_dir() / f"{safe}.json"


@dataclass
class Profile:
    name: str
    embedding: list[float]
    dim: int
    created: str
    samples: int = 0
    # "user" (named interactively / via --speakers-names) or "auto"
    # (adopted from a profile auto-match). Auto-sourced profiles are
    # treated as derived data: a later run cannot silently merge into a
    # profile that was never actually confirmed by a human (M3, wave-1).
    source: str = "user"


def load_profiles() -> list[Profile]:
    """Load all stored voice profiles, sorted by name."""
    d = profiles_dir()
    if not d.exists():
        return []
    out: list[Profile] = []
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        emb = data.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        out.append(
            Profile(
                name=str(data.get("name", p.stem)),
                embedding=[float(x) for x in emb],
                dim=int(data.get("dim", len(emb))),
                created=str(data.get("created", "")),
                samples=int(data.get("samples", 0)),
                source=str(data.get("source", "user")),
            )
        )
    return out


def _load_profile_raw(name: str) -> dict | None:
    """Load a single stored profile's raw JSON, or None if it doesn't exist."""
    path = _profile_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# Cap on how much historical samples can outweigh a new one. Keeps the profile
# adapting to a changed mic/voice instead of freezing after a few recordings.
_MAX_HISTORY_WEIGHT = 5


def merge_embeddings(
    old: list[float],
    old_samples: int,
    new: list[float],
    new_samples: int = 1,
) -> tuple[list[float], int]:
    """Combine two speaker embeddings via a sample-weighted running mean.

    The old vector's weight is capped at ``_MAX_HISTORY_WEIGHT`` so a new
    recording still moves the centroid once enough history has accumulated —
    the profile keeps adapting to a changed mic/voice instead of freezing.
    Returns the merged embedding and the new total sample count. The two vectors
    must be equal-length; the caller is responsible for the dimension check
    (mismatched dims should discard the old profile rather than average).
    """
    if not old:
        return [float(x) for x in new], int(new_samples)
    if not new:
        return [float(x) for x in old], int(old_samples)
    w_old = float(min(max(0, old_samples), _MAX_HISTORY_WEIGHT))
    w_new = float(max(1, new_samples))
    total = w_old + w_new
    out = [(o * w_old + n * w_new) / total for o, n in zip(old, new)]
    return out, int(old_samples + new_samples)


def save_profile(
    name: str,
    embedding: list[float],
    samples: int = 1,
    auto_match: bool = False,
) -> Path:
    """Persist a voice profile for ``name``, merging with any existing one.

    If a profile already exists for ``name`` with the same embedding dimension,
    the new embedding is combined with the stored one via a sample-weighted
    running mean (see ``merge_embeddings``) so the profile grows more accurate
    across recordings instead of being overwritten. If the dimension differs
    (e.g. the embedding model was swapped), the old profile is discarded and
    the new one replaces it. ``samples`` is the count this run contributes.

    M3 (wave-1 audit): ``auto_match=True`` marks this save as sourced from a
    profile auto-match rather than a user-confirmed name. An auto-match can
    CREATE a profile (first sighting, provenance recorded as ``"auto"``) but
    can never MERGE into or REPLACE an existing one — auto-matches used to
    flow back into stored profiles with no provenance, so a chain of
    self-confirming matches silently drifted the stored centroid. An
    existing profile is returned untouched (the caller decides whether to
    warn), and a new file is marked ``source: "auto"`` so a later human
    confirmation (``auto_match=False``) still merges normally and upgrades
    the provenance to ``"user"``.
    """
    d = profiles_dir()
    d.mkdir(parents=True, exist_ok=True)
    prior = _load_profile_raw(name)
    if auto_match and prior is not None:
        # An auto-match must not adopt an existing profile's slot.
        return _profile_path(name)
    total_samples = int(samples)
    final_embedding = [float(x) for x in embedding]
    source = "auto" if auto_match else "user"
    if prior is not None:
        old_emb = prior.get("embedding")
        old_dim = int(prior.get("dim", len(old_emb) if isinstance(old_emb, list) else 0))
        if isinstance(old_emb, list) and old_emb and old_dim == len(final_embedding):
            final_embedding, total_samples = merge_embeddings(
                [float(x) for x in old_emb],
                int(prior.get("samples", 0)),
                final_embedding,
                new_samples=samples,
            )
            # A user-confirmed save upgrades any earlier auto provenance;
            # `source` was set above and stays "user" for this branch
            # (auto_match=True never reaches here — it returns early).
        # Dim mismatch: drop the old profile (incompatible model) and start fresh.
    payload = {
        "name": name,
        "dim": len(final_embedding),
        "embedding": final_embedding,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "samples": total_samples,
        "source": source,
    }
    path = _profile_path(name)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def forget_profile(name: str) -> bool:
    """Delete a profile by name. Returns True if a file was removed."""
    path = _profile_path(name)
    if path.exists():
        path.unlink()
        return True
    return False


# ---------- embedding extraction ----------

def compute_speaker_embeddings(
    wav: Path,
    segments: list[DiarSegment],
    config: cfg.Config,
) -> dict[int, list[float]]:
    """Compute one averaged embedding per speaker cluster.

    For each speaker id present in ``segments``, the audio for that speaker's
    segments is concatenated and fed to the sherpa-onnx
    ``SpeakerEmbeddingExtractor``. The extractor is a streaming model, so the
    audio is split into chunks (~30 s) that fit its context. When a speaker
    has multiple segments, their embeddings are averaged into a single vector.

    Returns ``{speaker_id: embedding}``. Speakers whose total audio is too
    short to produce an embedding are omitted.
    """
    if not segments:
        return {}

    emb_model = find_embedding_model(config)
    if emb_model is None:
        raise RuntimeError(
            "Embedding model not found. Run `whiz models download-diarization` first."
        )

    sherpa_onnx = _import_sherpa()

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(str(emb_model))
    )
    dim = extractor.dim
    sample_rate = 16000  # whiz extracts 16 kHz mono WAV

    samples, sr = _read_wav_pcm(wav)
    if sr != sample_rate:
        raise RuntimeError(f"Expected {sample_rate} Hz audio, got {sr} Hz.")

    # Group sample ranges by speaker.
    by_speaker: dict[int, list[tuple[int, int]]] = {}
    for s in segments:
        if s.speaker not in by_speaker:
            by_speaker[s.speaker] = []
        start_i = max(0, int(s.start * sample_rate))
        end_i = min(len(samples), int(s.end * sample_rate))
        if end_i > start_i:
            by_speaker[s.speaker].append((start_i, end_i))

    out: dict[int, list[float]] = {}
    # Feed audio in chunks so the streaming extractor's context isn't exceeded.
    chunk = sample_rate * 30  # 30 s
    for spk, ranges in by_speaker.items():
        vecs: list[list[float]] = []
        for start_i, end_i in ranges:
            seg_samples = samples[start_i:end_i]
            # Skip very short utterances (< 0.3 s) — not enough for an embedding.
            if len(seg_samples) < int(sample_rate * 0.3):
                continue
            off = 0
            while off < len(seg_samples):
                block = seg_samples[off : off + chunk]
                if len(block) < int(sample_rate * 0.3):
                    break
                stream = extractor.create_stream()
                stream.accept_waveform(sample_rate, block)
                stream.input_finished()
                if extractor.is_ready(stream):
                    vecs.append(list(extractor.compute(stream)))
                off += chunk
        if vecs:
            out[spk] = _average_vectors(vecs, dim)
    return out


def _average_vectors(vecs: list[list[float]], dim: int) -> list[float]:
    """Element-wise mean of equal-length vectors."""
    acc = [0.0] * dim
    for v in vecs:
        for i in range(min(dim, len(v))):
            acc[i] += v[i]
    n = float(len(vecs))
    return [x / n for x in acc]


# ---------- matching ----------

def cosine_similarity(a: list[float], b: list[float]) -> float | None:
    """Cosine similarity of two equal-length vectors (range -1..1).

    M3 (wave-1 audit): mismatched dimensions used to be silently truncated
    to the shorter length, comparing a partial projection and reporting a
    confident-looking score for two embeddings that are not comparable (a
    swapped embedding model changes every dimension's meaning). It now
    returns ``None`` — callers treat that as "no comparison possible"
    (``match_speakers`` skips the pair with a warning) instead of a number
    derived from a truncated vector.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return None
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def match_speakers(
    cluster_embeddings: dict[int, list[float]],
    profiles: list[Profile] | None = None,
    threshold: float = 0.8,
) -> dict[int, tuple[str, float] | None]:
    """Match each cluster to the best stored profile above ``threshold``.

    Returns ``{cluster_id: (name, score) | None}``. A cluster maps to ``None``
    when no profile reaches the threshold (i.e. an unknown speaker). Ties are
    broken by higher score; the same profile is never assigned to two clusters
    — each name is claimed by its single best-scoring cluster.
    """
    profiles = profiles if profiles is not None else load_profiles()
    if not profiles or not cluster_embeddings:
        return {cid: None for cid in cluster_embeddings}

    # Score every (cluster, profile) pair.
    scored: list[tuple[float, int, str]] = []
    dim_skips = 0
    for cid, cemb in cluster_embeddings.items():
        for prof in profiles:
            score = cosine_similarity(cemb, prof.embedding)
            if score is None:
                dim_skips += 1
                continue
            scored.append((score, cid, prof.name))
    if dim_skips:
        print(
            f"Warning: {dim_skips} cluster/profile pair(s) skipped — embedding "
            "dimension mismatch (a profile saved with a different embedding "
            "model is not comparable; re-create it under the current model).",
            file=sys.stderr,
        )
    scored.sort(reverse=True)

    matched: dict[int, tuple[str, float] | None] = {cid: None for cid in cluster_embeddings}
    used_names: set[str] = set()
    used_clusters: set[int] = set()
    for score, cid, name in scored:
        if score < threshold:
            break
        if cid in used_clusters or name in used_names:
            continue
        matched[cid] = (name, score)
        used_clusters.add(cid)
        used_names.add(name)
    return matched


def auto_assign_names(
    cluster_embeddings: dict[int, list[float]],
    threshold: float = 0.8,
    profiles: list[Profile] | None = None,
) -> tuple[dict[str, str], dict[int, tuple[str, float] | None]]:
    """Build a {speaker_label: name} map from profile matches.

    Only speakers whose match score exceeds ``threshold`` are named; others
    are left as ``Speaker X`` for the interactive prompt or
    ``--speakers-names`` to fill in. Returns the name map keyed by
    ``Speaker A/B/...`` labels and the raw per-cluster match info.
    """
    from whiz.merge import speaker_label

    matches = match_speakers(cluster_embeddings, profiles, threshold)
    name_map: dict[str, str] = {}
    for cid, m in matches.items():
        if m is not None:
            name_map[speaker_label(cid)] = m[0]
    return name_map, matches
