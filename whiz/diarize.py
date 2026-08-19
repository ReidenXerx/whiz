"""Speaker diarization via sherpa-onnx Python API.

sherpa-onnx combines a pyannote segmentation model with a speaker-embedding
extractor and clustering to produce (start, end, speaker) segments. This
module lazily imports the sherpa_onnx package (an optional dependency,
installed via `pipx inject whiz sherpa-onnx`), locates the two required
model files, downloads them if needed, and returns structured segments.
"""

from __future__ import annotations

import json
import shutil
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from whiz import config as cfg

_DIAR_CACHE_VERSION = 1

# GitHub release asset URLs.
SEG_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMB_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)

SEG_DIR_NAME = "sherpa-onnx-pyannote-segmentation-3-0"
SEG_MODEL_FILE = "model.int8.onnx"
EMB_MODEL_FILE = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"


@dataclass
class DiarSegment:
    start: float
    end: float
    speaker: int


def _default_diarization_dir() -> Path:
    return Path.home() / ".cache" / "whiz" / "diarization"


def _import_sherpa():
    """Lazily import sherpa_onnx, with a clear error if missing."""
    try:
        import sherpa_onnx  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "sherpa_onnx is not installed.\n"
            "Install it into whiz with:  pipx inject whiz sherpa-onnx\n"
            f"(underlying error: {e})"
        ) from e
    return sherpa_onnx


def find_segmentation_model(config: cfg.Config) -> Path | None:
    """Find the pyannote segmentation model.onnx/.int8.onnx."""
    if config.diarization_segmentation_model:
        p = Path(config.diarization_segmentation_model).expanduser()
        if p.exists():
            return p
    candidates: list[Path] = []
    seg_dir = _default_diarization_dir() / SEG_DIR_NAME
    candidates.append(seg_dir / SEG_MODEL_FILE)
    candidates.append(seg_dir / "model.onnx")
    for d in cfg.model_search_dirs(config):
        candidates.append(d / SEG_DIR_NAME / SEG_MODEL_FILE)
        candidates.append(d / SEG_DIR_NAME / "model.onnx")
        candidates.append(d / SEG_MODEL_FILE)
    for c in candidates:
        if c.exists():
            return c
    return None


def find_embedding_model(config: cfg.Config) -> Path | None:
    """Find the 3D-Speaker embedding extractor .onnx."""
    if config.diarization_embedding_model:
        p = Path(config.diarization_embedding_model).expanduser()
        if p.exists():
            return p
    candidates: list[Path] = []
    diar_dir = _default_diarization_dir()
    candidates.append(diar_dir / EMB_MODEL_FILE)
    for d in cfg.model_search_dirs(config):
        candidates.append(d / EMB_MODEL_FILE)
    for c in candidates:
        if c.exists():
            return c
    return None


def _download(url: str, target: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "whiz/0.3"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - trusted release URL
        if resp.status >= 400:
            raise RuntimeError(f"Download failed: HTTP {resp.status} for {url}")
        with target.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=1024 * 1024)


def download_diarization_models(dest_dir: Path | None = None) -> tuple[Path, Path]:
    """Download segmentation (tar.bz2, extracted) and embedding models."""
    base = dest_dir or _default_diarization_dir()
    base.mkdir(parents=True, exist_ok=True)

    seg_dir = base / SEG_DIR_NAME
    seg_int8 = seg_dir / SEG_MODEL_FILE

    if not seg_int8.exists() and not (seg_dir / "model.onnx").exists():
        tar_path = base / f"{SEG_DIR_NAME}.tar.bz2"
        print(f"Downloading segmentation model from {SEG_URL} ...", flush=True)
        _download(SEG_URL, tar_path)
        print(f"Extracting {tar_path.name} ...", flush=True)
        with tarfile.open(tar_path, "r:bz2") as tf:
            tf.extractall(base)  # noqa: S202 - trusted release asset
        tar_path.unlink(missing_ok=True)
        if not seg_int8.exists() and not (seg_dir / "model.onnx").exists():
            raise RuntimeError(f"Extraction did not produce expected model in {seg_dir}")

    seg_path = seg_int8 if seg_int8.exists() else seg_dir / "model.onnx"

    emb_path = base / EMB_MODEL_FILE
    if not emb_path.exists():
        print(f"Downloading embedding model from {EMB_URL} ...", flush=True)
        _download(EMB_URL, emb_path)

    print(
        f"Diarization models ready:\n  segmentation: {seg_path}\n  embedding:    {emb_path}",
        flush=True,
    )
    return seg_path, emb_path


def diar_cache_path(wav: Path) -> Path:
    """Path of the diarization cache file for a given WAV."""
    # Append (not with_suffix) so dots in stems like "...16.03.40" survive.
    return Path(str(wav) + ".diar.json")


def load_diarization_cache(
    wav: Path,
    num_speakers: int = 0,
    threshold: float = 0.5,
) -> list[DiarSegment] | None:
    """Load a cached diarization result if params match.

    Returns the cached segments when the cache exists and was produced with
    the same num_speakers/threshold; otherwise returns None. The expensive
    part of diarization is embedding extraction, so reusing a matching cache
    skips the ~3 minute embedding pass and only needs the cheap merge step.
    """
    path = diar_cache_path(wav)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("version") != _DIAR_CACHE_VERSION:
        return None
    if data.get("num_speakers") != num_speakers:
        return None
    # Compare threshold with a small epsilon for float formatting round-trips.
    cached_thr = data.get("threshold")
    if cached_thr is None or abs(float(cached_thr) - threshold) > 1e-9:
        return None
    segs = data.get("segments", [])
    out: list[DiarSegment] = []
    for s in segs:
        try:
            out.append(DiarSegment(start=float(s["start"]), end=float(s["end"]), speaker=int(s["speaker"])))
        except (KeyError, TypeError, ValueError):
            return None
    return out


def _write_diarization_cache(
    wav: Path,
    segments: list[DiarSegment],
    num_speakers: int,
    threshold: float,
) -> Path:
    """Persist the diarization result so later `whiz merge` runs can reuse it."""
    path = diar_cache_path(wav)
    payload = {
        "version": _DIAR_CACHE_VERSION,
        "num_speakers": num_speakers,
        "threshold": threshold,
        "segments": [
            {"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_diarization(
    wav: Path,
    config: cfg.Config,
    num_speakers: int = 0,
    threshold: float = 0.5,
    dry_run: bool = False,
    use_cache: bool = True,
) -> list[DiarSegment]:
    """Run sherpa-onnx diarization on a 16kHz mono WAV.

    Returns parsed segments sorted by start time. If dry_run, returns []
    and prints what would run.

    When ``use_cache`` is True (the default) and a matching cache exists for
    this WAV + (num_speakers, threshold), the embedding pass is skipped and
    the cached segments are returned. A fresh result is always written back
    to the cache after a real run.
    """
    if use_cache and not dry_run:
        cached = load_diarization_cache(wav, num_speakers=num_speakers, threshold=threshold)
        if cached is not None:
            from whiz import ui
            ui.muted(
                f"Reusing diarization cache ({len(cached)} segments, "
                f"num_speakers={num_speakers or 'auto'}, threshold={threshold}): "
                f"{diar_cache_path(wav)}"
            )
            return cached

    seg_model = find_segmentation_model(config)
    emb_model = find_embedding_model(config)
    if seg_model is None or emb_model is None:
        raise RuntimeError(
            "Diarization models not found. Run `whiz models download-diarization` first."
        )

    if dry_run:
        print("DRY-RUN diarization (Python API):")
        print(f"  segmentation model: {seg_model}")
        print(f"  embedding model:    {emb_model}")
        print(f"  num_speakers:       {num_speakers}")
        print(f"  cluster_threshold:  {threshold}")
        print(f"  wav:                {wav}")
        return []

    sherpa_onnx = _import_sherpa()

    from whiz import ui
    ui.muted("Loading sherpa-onnx diarization ...")
    seg_cfg = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(str(seg_model))
    segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(pyannote=seg_cfg)
    embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(str(emb_model))
    clustering = sherpa_onnx.FastClusteringConfig(
        num_clusters=num_speakers if num_speakers and num_speakers > 0 else -1,
        threshold=threshold,
    )
    sd_cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        clustering=clustering,
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not sd_cfg.validate():
        raise RuntimeError("sherpa-onnx diarization config validation failed; check model paths.")

    sd = sherpa_onnx.OfflineSpeakerDiarization(sd_cfg)

    samples, sample_rate = _read_wav_pcm(wav)
    if sample_rate != sd.sample_rate:
        raise RuntimeError(
            f"Expected {sd.sample_rate} Hz audio, got {sample_rate} Hz. "
            "whiz should have extracted 16kHz audio."
        )

    from whiz import ui
    ui.muted("Running speaker diarization ...")
    result = sd.process(samples, callback=_progress_callback).sort_by_start_time()
    segments = [
        DiarSegment(start=r.start, end=r.end, speaker=r.speaker) for r in result
    ]
    ui.muted(f"Diarization found {len(segments)} segments.")
    cache_path = _write_diarization_cache(wav, segments, num_speakers, threshold)
    ui.muted(f"Saved diarization cache: {cache_path}")
    return segments


def _progress_callback(num_processed: int, num_total: int) -> int:
    """sherpa-onnx diarization progress hook (prints % to stderr)."""
    if num_total > 0:
        pct = num_processed / num_total * 100.0
        print(f"\rDiarization: {pct:5.1f}% ({num_processed}/{num_total})", end="", file=sys.stderr, flush=True)
        if num_processed >= num_total:
            print(file=sys.stderr)  # newline after completion
    return 0


def _read_wav_pcm(path: Path) -> tuple[list[float], int]:
    """Read a 16kHz mono PCM WAV into a float32 sample list.

    Uses the wave stdlib to avoid a numpy/soundfile dependency.
    """
    import struct
    import wave

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width != 2:
        raise RuntimeError(f"Expected 16-bit PCM WAV, got sample_width={sample_width}")

    n = n_frames * n_channels
    ints = struct.unpack(f"<{n}h", raw)
    if n_channels > 1:
        mono: list[float] = []
        for i in range(0, n, n_channels):
            chunk = ints[i : i + n_channels]
            mono.append(sum(chunk) / n_channels / 32768.0)
        return mono, sample_rate
    return [s / 32768.0 for s in ints], sample_rate