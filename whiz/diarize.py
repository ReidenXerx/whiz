"""Speaker diarization via sherpa-onnx Python API.

sherpa-onnx combines a pyannote segmentation model with a speaker-embedding
extractor and clustering to produce (start, end, speaker) segments. This
module lazily imports the sherpa_onnx package (an optional dependency,
installed via `pipx inject whiz sherpa-onnx`), locates the two required
model files, downloads them if needed, and returns structured segments.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from array import array
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from whiz import config as cfg

_DIAR_CACHE_VERSION = 2

class DiarizationUnavailable(RuntimeError):
    """Diarization cannot run: missing package, models, or invalid config.

    Setup problems that callers DEGRADE on (skip speakers / generic labels
    with a hint), not crash on. Distinct from RuntimeError because callers
    must not string-match messages to tell the two apart — the wave-1 audit
    found the string taxonomy had drifted between sites and missed the
    validate-failure path entirely (M2). Transient/runtime failures keep
    raising plain RuntimeError and stay loud.
    """

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
        raise DiarizationUnavailable(
            "sherpa_onnx is not installed.\n"
            "Install it into whiz with:  pipx inject whiz sherpa-onnx\n"
            f"(underlying error: {e})"
        ) from e
    return sherpa_onnx


def find_segmentation_model(config: cfg.Config) -> Path | None:
    """Find the pyannote segmentation model.onnx/.int8.onnx.

    Default candidates prefer the UNQUANTIZED model.onnx over
    model.int8.onnx (NS-15: quantization corrupts quality; the wave-1
    audit flagged the int8-first order as a contradiction, M13). An
    explicit ``diarization_segmentation_model`` config path is used
    verbatim when it exists.
    """
    if config.diarization_segmentation_model:
        p = Path(config.diarization_segmentation_model).expanduser()
        if p.exists():
            return p
    candidates: list[Path] = []
    seg_dir = _default_diarization_dir() / SEG_DIR_NAME
    candidates.append(seg_dir / "model.onnx")
    candidates.append(seg_dir / SEG_MODEL_FILE)
    for d in cfg.model_search_dirs(config):
        candidates.append(d / SEG_DIR_NAME / "model.onnx")
        candidates.append(d / SEG_DIR_NAME / SEG_MODEL_FILE)
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
    """Fetch ``url`` into ``target`` atomically (``.part`` + rename).

    H (wave-3 audit): the body used to stream straight into the final path,
    so a ^C mid-download left a truncated .onnx that ``find_*_model`` then
    reported as present forever — never re-downloaded, failing to load
    with an opaque onnxruntime error. Shares ``models._stream_to_atomic``.
    """
    from whiz.models import _stream_to_atomic

    req = urllib.request.Request(url, headers={"User-Agent": "whiz/0.3"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - trusted release URL
        if resp.status >= 400:
            raise RuntimeError(f"Download failed: HTTP {resp.status} for {url}")
        _stream_to_atomic(resp, target)


def _check_member(member: tarfile.TarInfo) -> None:
    """Reject archive members that could land outside the extraction dir."""
    for name in (member.name, member.linkname if (member.issym() or member.islnk()) else ""):
        if not name:
            continue
        p = PurePosixPath(name.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts or (len(name) > 1 and name[1] == ":"):
            raise RuntimeError(f"Refusing unsafe path in diarization archive: {name!r}")
    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
        raise RuntimeError(f"Refusing special file in diarization archive: {member.name!r}")


def _safe_extractall(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract ``tf`` into ``dest`` with no member escaping it.

    Every member is checked first (absolute paths, ``..``, device files —
    on every Python), then ``filter="data"`` adds the stdlib's own checks
    where it exists (3.12+, and 3.11.4+ backports).
    """
    members = tf.getmembers()
    for m in members:
        _check_member(m)
    if hasattr(tarfile, "data_filter"):
        tf.extractall(dest, members=members, filter="data")
    else:  # pragma: no cover - pre-3.11.4 interpreters only
        tf.extractall(dest, members=members)  # noqa: S202 - members vetted above


def download_diarization_models(dest_dir: Path | None = None) -> tuple[Path, Path]:
    """Download segmentation (tar.bz2, extracted) and embedding models.

    Both land atomically (wave-3 audit): files via ``.part`` + rename, and
    the archive is extracted into a temp dir INSIDE ``base`` (same
    filesystem) whose model dir is renamed into place only once it holds a
    model — an interrupted run leaves nothing that ``find_segmentation_model``
    would mistake for an installed model, so the next run downloads again.
    """
    base = dest_dir or _default_diarization_dir()
    base.mkdir(parents=True, exist_ok=True)

    seg_dir = base / SEG_DIR_NAME
    seg_int8 = seg_dir / SEG_MODEL_FILE

    if not seg_int8.exists() and not (seg_dir / "model.onnx").exists():
        tar_path = base / f"{SEG_DIR_NAME}.tar.bz2"
        print(f"Downloading segmentation model from {SEG_URL} ...", flush=True)
        _download(SEG_URL, tar_path)
        print(f"Extracting {tar_path.name} ...", flush=True)
        staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=base))
        try:
            with tarfile.open(tar_path, "r:bz2") as tf:
                _safe_extractall(tf, staging)
            staged = staging / SEG_DIR_NAME
            if not (staged / SEG_MODEL_FILE).exists() and not (staged / "model.onnx").exists():
                raise RuntimeError(f"Extraction did not produce expected model in {seg_dir}")
            # Whatever sits at seg_dir now holds no model (checked above) —
            # the debris of an older, non-atomic extraction. Replace it.
            if seg_dir.is_symlink() or seg_dir.is_file():
                seg_dir.unlink()
            elif seg_dir.exists():
                shutil.rmtree(seg_dir)
            os.replace(staged, seg_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            tar_path.unlink(missing_ok=True)

    seg_path = (seg_dir / "model.onnx") if (seg_dir / "model.onnx").exists() else seg_int8
    # NS-15: prefer the unquantized model.onnx when the archive provided
    # both; the int8 variant is an explicit-use fallback only.

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
    seg_model: Path | None = None,
    emb_model: Path | None = None,
) -> list[DiarSegment] | None:
    """Load a cached diarization result if params AND inputs match.

    Returns the cached segments when the cache exists and was produced with
    the same num_speakers/threshold AND the same WAV bytes and model files;
    otherwise returns None. The expensive part of diarization is embedding
    extraction, so reusing a matching cache skips the ~3 minute embedding
    pass and only needs the cheap merge step.

    H1 (wave-1 audit): the cache used to be keyed only on
    (num_speakers, threshold). A re-exported video replaced the WAV under
    the same filename and the cache silently served STALE speaker labels
    for different audio, which then poisoned stored voice profiles via
    compute_speaker_embeddings + save_profile. The payload now records
    the WAV's size+mtime and the resolved model paths; a mismatch is a
    cache MISS (recompute) — never a silent stale hit.
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
    # Input identity (H1): the WAV bytes (size + mtime) and both resolved
    # model paths must match what produced the cache. ``None`` model args
    # mean the caller could not resolve models — never trust a cache that
    # cannot be verified.
    try:
        st = wav.stat()
    except OSError:
        return None
    if (
        data.get("wav_size") != st.st_size
        or data.get("wav_mtime") != st.st_mtime
        or data.get("seg_model") != (str(seg_model) if seg_model else None)
        or data.get("emb_model") != (str(emb_model) if emb_model else None)
    ):
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
    seg_model: Path | None = None,
    emb_model: Path | None = None,
) -> Path:
    """Persist the diarization result so later `whiz merge` runs can reuse it.

    Records the input identity (H1) — WAV size+mtime and the resolved model
    paths — alongside the params so a later load can tell a matching cache
    from a stale one.
    """
    path = diar_cache_path(wav)
    try:
        st = wav.stat()
        wav_size: int | None = st.st_size
        wav_mtime: float | None = st.st_mtime
    except OSError:
        wav_size = None
        wav_mtime = None
    payload = {
        "version": _DIAR_CACHE_VERSION,
        "num_speakers": num_speakers,
        "threshold": threshold,
        "wav_size": wav_size,
        "wav_mtime": wav_mtime,
        "seg_model": str(seg_model) if seg_model else None,
        "emb_model": str(emb_model) if emb_model else None,
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

    When ``use_cache`` is True (the default) and a matching cache exists
    for this WAV + (num_speakers, threshold) + resolved model files, the
    embedding pass is skipped and the cached segments are returned. A
    fresh result is always written back to the cache after a real run.
    """
    # Fresh throttle state per run: a second diarization in the same process
    # (merge after transcribe, the macOS UI's repeat runs) otherwise emitted
    # nothing until its percent passed the previous run's last one.
    _reset_progress()
    # Resolve the models BEFORE the cache check (H1): the cache key now
    # includes the resolved model paths, and dry_run needs them anyway.
    seg_model = find_segmentation_model(config)
    emb_model = find_embedding_model(config)
    if seg_model is None or emb_model is None:
        raise DiarizationUnavailable(
            "Diarization models not found. Run `whiz models download-diarization` first."
        )

    if use_cache and not dry_run:
        cached = load_diarization_cache(
            wav, num_speakers=num_speakers, threshold=threshold,
            seg_model=seg_model, emb_model=emb_model,
        )
        if cached is not None:
            from whiz import ui
            ui.muted(
                f"Reusing diarization cache ({len(cached)} segments, "
                f"num_speakers={num_speakers or 'auto'}, threshold={threshold}): "
                f"{diar_cache_path(wav)}"
            )
            return cached

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
    # Which segmentation variant is in use — model.onnx (unquantized) or
    # model.int8.onnx — matters for quality (NS-15); make it visible.
    ui.muted(f"  segmentation model: {seg_model.name}")
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
        raise DiarizationUnavailable(
            "sherpa-onnx diarization config validation failed; check model paths."
        )

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
    cache_path = _write_diarization_cache(
        wav, segments, num_speakers, threshold,
        seg_model=seg_model, emb_model=emb_model,
    )
    ui.muted(f"Saved diarization cache: {cache_path}")
    return segments


def _progress_callback(num_processed: int, num_total: int) -> int:
    """sherpa-onnx diarization progress hook.

    Called very frequently (per chunk); the whole point of throttling here is
    that ONE integer percent is what both renderers want. On a TTY the line
    redraws in place with ``\r`` (the classic counter). When piped, each new
    whole percent is a ``~ progress: N%`` machine line for the macOS UI — the
    raw ``\r`` line would corrupt the merged stream a pipe parser reads.
    """
    if num_total <= 0:
        return 0
    pct = num_processed / num_total * 100.0
    # ONE rounded integer drives both the throttle and the emitted value:
    # throttling on int() floor while ui.progress rounds showed 100% at
    # 99.5% and then emitted "100%" again (wave-3 audit).
    whole = int(round(pct))
    done = num_processed >= num_total
    if not done:
        _progress_finished[0] = False
    elif _progress_finished[0]:
        return 0
    if whole == _last_emitted_percent[0] and not done:
        return 0
    fresh = whole != _last_emitted_percent[0]
    _last_emitted_percent[0] = whole
    if _ui_tty():
        print(f"\rDiarization: {pct:5.1f}% ({num_processed}/{num_total})",
              end="", file=sys.stderr, flush=True)
        if done:
            # The newline after completion goes out even when the final
            # percent was already drawn — the next output must not glue
            # onto the counter line.
            print(file=sys.stderr)
    elif fresh:
        from whiz import ui
        ui.progress(whole)
    if done:
        _progress_finished[0] = True
    return 0


# The last percent the callback emitted, so repeated per-chunk calls at the
# same percentage emit exactly one line. Starts at -1 (a percent outside the
# real range) so a first call at 0% still emits once. Reset by
# ``run_diarization`` at the start of every run (``_reset_progress``).
_last_emitted_percent = [-1]
# Whether the completion call has been rendered (one TTY newline per run).
_progress_finished = [False]


def _reset_progress() -> None:
    """Fresh throttle state for a new diarization run."""
    _last_emitted_percent[0] = -1
    _progress_finished[0] = False


def _ui_tty() -> bool:
    """Whether whiz's ui sees a TTY — imported lazily to avoid a cycle."""
    from whiz import ui
    return ui._is_tty()


def _read_wav_pcm(path: Path, expected_rate: int | None = 16000) -> tuple[array, int]:
    """Read a 16 kHz mono 16-bit PCM WAV into float32 samples in [-1, 1).

    Returns ``(samples, sample_rate)``; ``samples`` is an ``array('f')`` —
    a float sequence (``len``, slicing, iteration) that sherpa-onnx accepts
    wherever it took the old list. Uses the wave stdlib, no numpy.

    Wave-3 audit: the old ``struct.unpack`` tuple + list of Python floats
    cost ~70 bytes/sample (~4 GB per hour of audio, built twice per run —
    diarization and voice profiles); ``array`` costs 4 bytes/sample.

    Every unusable input — not RIFF/WAVE, not 16-bit, not mono, not
    ``expected_rate`` (None skips the rate check) — raises RuntimeError
    with a clear message: callers map RuntimeError to a clean error, and a
    raw ``wave.Error`` escaped that as a traceback. whiz always extracts
    16 kHz mono itself, so any of these means a bad file, not a setting.
    """
    import wave

    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            if sample_width != 2:
                raise RuntimeError(
                    f"Expected 16-bit PCM WAV, got sample_width={sample_width}: {path}"
                )
            if n_channels != 1:
                raise RuntimeError(f"Expected mono WAV, got {n_channels} channels: {path}")
            if expected_rate is not None and sample_rate != expected_rate:
                raise RuntimeError(
                    f"Expected {expected_rate} Hz audio, got {sample_rate} Hz: {path}"
                )
            raw = wf.readframes(n_frames)
    except (wave.Error, EOFError) as e:
        raise RuntimeError(f"Not a readable PCM WAV file ({e}): {path}") from e

    ints = array("h")
    # A truncated data chunk can end mid-sample; drop the odd byte.
    ints.frombytes(raw[: len(raw) - (len(raw) % 2)])
    del raw
    if sys.byteorder == "big":
        ints.byteswap()  # WAV is little-endian
    # 1/32768 is a power of two, so the multiply is exact (== the old divide).
    samples = array("f", map((1.0 / 32768.0).__mul__, ints))
    return samples, sample_rate
