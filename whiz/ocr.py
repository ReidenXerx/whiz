"""On-screen text extraction (OCR) from captured frames.

whiz captures one frame per transcript segment. Those frames only reach a model
through the vision path, which needs a VLM — so on a text-only model the visible
screen is lost. OCR closes that gap: it turns each frame into text that rides
along in the transcript, so any model (a 4B one included) can read what was on
screen, and the analyst posture in ``whiz.ai`` can reconcile screen against
speech.

Three engines are supported, all optional and lazily imported (the same pattern
``whiz.diarize`` uses for sherpa-onnx):

``apple``
    Apple's Vision framework via the ``ocrmac`` wrapper. macOS only, ~130-210 ms
    per frame, no model download, and more accurate than Tesseract on UI text.
``rapidocr``
    PP-OCR models on onnxruntime. Cross-platform, Apache 2.0, and nearly free on
    machines that already installed sherpa-onnx (which brings onnxruntime).
``tesseract``
    The ``tesseract`` binary, driven over a subprocess like ffmpeg and
    whisper-cli so no Python bindings (or Pillow) are needed.

OCR is the slowest stage in the pipeline — one pass per segment frame, and an
hour-long recording is several hundred frames — so it is always opt-in, results
are deduped by frame bytes, and every per-frame failure degrades to an empty
string rather than aborting the run.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from whiz import config as cfg

# All engines whiz knows how to drive, in no particular order.
ENGINES: tuple[str, ...] = ("apple", "rapidocr", "tesseract")

# Per-platform auto-detect preference. Apple Vision is far and away the best
# option on macOS (fastest, most accurate, no model download); RapidOCR is the
# portable default everywhere else, with Tesseract as the last resort.
_ORDER_BY_PLATFORM: dict[str, tuple[str, ...]] = {
    "darwin": ("apple", "rapidocr", "tesseract"),
}
_DEFAULT_ORDER: tuple[str, ...] = ("rapidocr", "tesseract")

# Vertical tolerance (in normalized 0..1 image height) for grouping Apple Vision
# annotations onto the same output line before sorting left-to-right.
_LINE_TOL = 0.02

# whiz asks for language hints in BCP-47-ish form ("en-US"); Tesseract wants
# ISO 639-2/T. Unknown codes simply skip the -l flag (Tesseract defaults to eng).
_TESSERACT_LANGS = {
    "en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita",
    "pt": "por", "nl": "nld", "ru": "rus", "uk": "ukr", "pl": "pol",
    "tr": "tur", "ja": "jpn", "ko": "kor", "zh": "chi_sim",
}


@dataclass
class EngineInfo:
    """Availability and install story for one OCR engine."""
    name: str
    available: bool
    detail: str = ""
    # Runnable argv to install this engine, after confirmation. Empty means the
    # install can't be automated (needs sudo, or the platform doesn't support it)
    # and ``install_hint`` should just be printed.
    install_argv: list[str] = field(default_factory=list)
    install_hint: str = ""


def engine_order() -> tuple[str, ...]:
    """Auto-detect preference order for the current platform."""
    return _ORDER_BY_PLATFORM.get(sys.platform, _DEFAULT_ORDER)


def preferred_engine() -> str:
    """The engine whiz would install by default on this platform."""
    order = engine_order()
    return order[0] if order else ""


def _module_present(name: str) -> bool:
    """True if ``name`` is importable, without paying the import cost."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _install_argv(pkgs: list[str]) -> list[str]:
    """Build the install command for the environment whiz is running in.

    A pipx-installed whiz needs ``pipx inject`` (its venv isn't pip-managed in
    the usual way, and this is the form the README already documents for
    sherpa-onnx). Anything else gets a plain pip install against the running
    interpreter.
    """
    if "/pipx/venvs/" in sys.prefix or "\\pipx\\venvs\\" in sys.prefix:
        return ["pipx", "inject", "whiz", *pkgs]
    return [sys.executable, "-m", "pip", "install", *pkgs]


def _tesseract_version(exe: str) -> str:
    try:
        proc = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "tesseract"
    first = (proc.stdout or proc.stderr or "").strip().splitlines()
    return first[0].strip() if first else "tesseract"


def detect(name: str) -> EngineInfo:
    """Report whether ``name`` is usable here, and how to install it if not."""
    name = (name or "").strip().lower()
    if name == "apple":
        argv = _install_argv(["ocrmac"])
        hint = " ".join(argv)
        if sys.platform != "darwin":
            return EngineInfo("apple", False, "macOS only", [], "")
        if _module_present("ocrmac"):
            return EngineInfo("apple", True, "Apple Vision (ocrmac)", argv, hint)
        return EngineInfo("apple", False, "ocrmac not installed", argv, hint)
    if name == "rapidocr":
        argv = _install_argv(["rapidocr", "onnxruntime"])
        hint = " ".join(argv)
        if _module_present("rapidocr"):
            return EngineInfo("rapidocr", True, "RapidOCR (onnxruntime)", argv, hint)
        return EngineInfo("rapidocr", False, "rapidocr not installed", argv, hint)
    if name == "tesseract":
        exe = shutil.which("tesseract")
        if exe:
            return EngineInfo("tesseract", True, _tesseract_version(exe), [], "")
        if sys.platform == "darwin" and shutil.which("brew"):
            argv = ["brew", "install", "tesseract"]
            return EngineInfo("tesseract", False, "tesseract not on PATH", argv, " ".join(argv))
        # apt/dnf need sudo; offer the command but don't run it for the user.
        return EngineInfo(
            "tesseract", False, "tesseract not on PATH", [],
            "sudo apt-get install tesseract-ocr  (or: sudo dnf install tesseract)",
        )
    return EngineInfo(name or "?", False, "unknown engine", [], "")


def available_engines() -> list[EngineInfo]:
    """Detect every known engine, in this platform's preference order."""
    order = list(engine_order())
    order += [e for e in ENGINES if e not in order]
    return [detect(e) for e in order]


def resolve_engine(config: cfg.Config, requested: str = "") -> str:
    """Pick the engine to use: explicit request, config, or auto-detect.

    An explicit name is returned as-is even when it isn't installed, so the
    caller can offer to install exactly what was asked for. ``auto`` returns the
    first *available* engine, or ``""`` when none is installed.
    """
    name = (requested or config.ocr_engine or "auto").strip().lower()
    if name and name != "auto":
        if name not in ENGINES:
            raise RuntimeError(
                f"Unknown OCR engine '{name}'. Choose one of: {', '.join(ENGINES)}, or 'auto'."
            )
        return name
    for candidate in engine_order():
        if detect(candidate).available:
            return candidate
    return ""


def ensure_engine(name: str, *, interactive: bool = True, on_message=None) -> bool:
    """Make ``name`` usable, offering to install it when it isn't.

    Returns True when the engine can be used after this call. Installs are only
    run after an explicit y/N confirmation, and only when they don't need sudo.
    After a successful install the import caches are invalidated so the engine
    works in this same process rather than requiring a re-run.
    """
    def say(msg: str) -> None:
        if on_message:
            on_message(msg)

    info = detect(name)
    if info.available:
        return True
    if not info.install_argv:
        if info.install_hint:
            say(f"{name}: {info.detail}. Install it with:  {info.install_hint}")
        else:
            say(f"{name}: {info.detail}.")
        return False
    if not interactive:
        say(f"{name}: {info.detail}. Install it with:  {info.install_hint}")
        return False

    say(f"{name}: {info.detail}.")
    try:
        answer = input(f"Install it now with `{info.install_hint}`? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if answer and answer not in {"y", "yes"}:
        return False

    say(f"running: {info.install_hint}")
    try:
        proc = subprocess.run(info.install_argv)
    except OSError as e:
        say(f"install failed: {e}")
        return False
    if proc.returncode != 0:
        say(f"install failed (exit {proc.returncode}). Run it manually:  {info.install_hint}")
        return False
    # A package installed into the running interpreter's site-packages is only
    # importable after the finder caches are dropped.
    importlib.invalidate_caches()
    return detect(name).available


# ---------- engine implementations ----------

# RapidOCR loads its ONNX models on construction. Across several hundred frames,
# rebuilding it per call is the difference between minutes and hours, so the
# engine is built once and reused.
_rapid_engine = None


def _ocr_apple(path: Path, languages: list[str] | None) -> str:
    from ocrmac import ocrmac  # type: ignore[import-not-found]

    kwargs = {"recognition_level": "accurate"}
    if languages:
        kwargs["language_preference"] = list(languages)
    annotations = ocrmac.OCR(str(path), **kwargs).recognize()
    return _join_apple(annotations)


def _join_apple(annotations) -> str:
    """Order Vision annotations into reading order and join them into lines.

    Each annotation is ``(text, confidence, bbox)`` with a normalized bbox whose
    origin is bottom-left, so "higher on screen" is a *larger* y. Group by y
    within ``_LINE_TOL`` then sort left-to-right within each line.
    """
    rows: list[tuple[int, float, str]] = []
    for item in annotations or ():
        try:
            text = str(item[0]).strip()
        except (IndexError, TypeError):
            continue
        if not text:
            continue
        x = y = 0.0
        if len(item) > 2 and isinstance(item[2], (list, tuple)) and len(item[2]) >= 2:
            try:
                x, y = float(item[2][0]), float(item[2][1])
            except (TypeError, ValueError):
                x = y = 0.0
        rows.append((-int(round(y / _LINE_TOL)), x, text))
    rows.sort()
    return "\n".join(text for _, _, text in rows)


def _ocr_rapidocr(path: Path, languages: list[str] | None) -> str:
    global _rapid_engine
    if _rapid_engine is None:
        from rapidocr import RapidOCR  # type: ignore[import-not-found]

        _rapid_engine = RapidOCR()
    result = _rapid_engine(str(path))
    return _join_rapidocr(result)


def _join_rapidocr(result) -> str:
    """Extract text from a RapidOCR result (v3 object or legacy tuple shape)."""
    if result is None:
        return ""
    txts = getattr(result, "txts", None)
    if txts:
        return "\n".join(str(t).strip() for t in txts if str(t).strip())
    # Legacy: (boxes_texts_scores, elapse) where each row is [box, text, score].
    rows = result[0] if isinstance(result, (tuple, list)) and result else None
    if not rows:
        return ""
    out: list[str] = []
    for row in rows:
        if isinstance(row, (tuple, list)) and len(row) >= 2:
            text = str(row[1]).strip()
            if text:
                out.append(text)
    return "\n".join(out)


def _tesseract_lang(languages: list[str] | None) -> str:
    """Map whiz language hints ('en-US') to a Tesseract -l value ('eng')."""
    codes: list[str] = []
    for lang in languages or ():
        base = str(lang).replace("_", "-").split("-")[0].strip().lower()
        mapped = _TESSERACT_LANGS.get(base)
        if mapped and mapped not in codes:
            codes.append(mapped)
    return "+".join(codes)


def _ocr_tesseract(path: Path, languages: list[str] | None) -> str:
    exe = shutil.which("tesseract")
    if not exe:
        raise RuntimeError("tesseract not found on PATH")
    cmd = [exe, str(path), "stdout"]
    lang = _tesseract_lang(languages)
    if lang:
        cmd += ["-l", lang]
    # errors="replace": tesseract can emit non-UTF-8 bytes on stderr, and a
    # strict decode would raise UnicodeDecodeError instead of returning text.
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
    return proc.stdout


_IMPLS = {
    "apple": _ocr_apple,
    "rapidocr": _ocr_rapidocr,
    "tesseract": _ocr_tesseract,
}


# ---------- normalization + batch driver ----------

def normalize(text: str, *, min_chars: int = 0, max_chars: int = 0) -> str:
    """Tidy raw engine output into compact, token-friendly lines.

    Collapses intra-line whitespace, drops blank lines, and enforces the
    min/max character bounds: results below ``min_chars`` are treated as noise
    (empty), and results above ``max_chars`` are truncated on a line boundary so
    a dense code screen can't blow up the prompt.
    """
    lines = [" ".join(line.split()) for line in (text or "").splitlines()]
    # Single-character lines are almost always icon/glyph misreads (toolbar
    # buttons come back as 'G', 'Q', '+'), never useful content.
    lines = [line for line in lines if len(line) > 1]
    out = "\n".join(lines)
    if min_chars and len(out) < min_chars:
        return ""
    if max_chars and len(out) > max_chars:
        kept: list[str] = []
        size = 0
        for line in lines:
            if size + len(line) + 1 > max_chars:
                break
            kept.append(line)
            size += len(line) + 1
        out = "\n".join(kept)
        if out:
            out += "\n…"
        else:
            out = (lines[0][:max_chars] + "…") if lines else ""
    return out


def new_screen_lines(current: str, previous: str) -> str:
    """Return only the lines of ``current`` that weren't already on ``previous``.

    Whole-frame dedupe barely fires on a real screen recording: the clock and
    cursor change every frame, so consecutive frames are never byte-identical
    even when the screen is visually static. Measured on a 4.8-minute Slack
    recording, 74% of OCR *lines* repeated frame to frame — window chrome, the
    macOS menu bar, the sidebar — which would otherwise dominate the prompt
    (96% of it) and split a one-chunk transcript into sixteen.

    Diffing per line fixes the signal rather than the volume alone: the screen
    line becomes "what changed on screen at this moment", which is what the
    model actually needs alongside the spoken words.

    Order is preserved, and a frame whose lines are all carried over yields "".
    """
    if not current:
        return ""
    if not previous:
        return current
    seen = set(previous.splitlines())
    kept = [line for line in current.splitlines() if line not in seen]
    return "\n".join(kept)


def frame_digest(path: Path) -> str:
    """Content hash of a frame, used to reuse OCR across identical frames."""
    try:
        return hashlib.blake2b(Path(path).read_bytes(), digest_size=16).hexdigest()
    except OSError:
        return ""


def ocr_image(
    path: Path,
    engine: str,
    languages: list[str] | None = None,
    *,
    min_chars: int = 0,
    max_chars: int = 0,
) -> str:
    """OCR one image with ``engine`` and return normalized text."""
    impl = _IMPLS.get((engine or "").strip().lower())
    if impl is None:
        raise RuntimeError(f"Unknown OCR engine '{engine}'. Choose one of: {', '.join(ENGINES)}.")
    return normalize(impl(Path(path), languages), min_chars=min_chars, max_chars=max_chars)


@dataclass
class OcrRun:
    """Outcome of an OCR pass over a batch of frames."""
    texts: list[str]
    ok: int = 0
    empty: int = 0
    reused: int = 0
    failed: int = 0
    elapsed: float = 0.0


def ocr_frames(
    paths: list[Path],
    engine: str,
    *,
    languages: list[str] | None = None,
    min_chars: int = 0,
    max_chars: int = 0,
    dedupe: bool = True,
    on_progress=None,
) -> OcrRun:
    """OCR a list of frames, returning one text per input path (aligned by index).

    Identical frames (same bytes) reuse the earlier result when ``dedupe`` is on
    — screen recordings hold a static screen for long stretches, so this is
    usually a large saving. It is deliberately conservative: only byte-identical
    frames are reused, never merely similar ones.

    A frame that fails to OCR yields ``""`` and is counted in ``failed``; the
    pass never raises, because losing a transcription to an OCR error would be a
    bad trade. ``on_progress(done, total, reused)`` is called as work advances.
    """
    run = OcrRun(texts=[])
    started = time.monotonic()
    seen: dict[str, str] = {}
    total = len(paths)
    for i, path in enumerate(paths, start=1):
        path = Path(path)
        text = ""
        if not path.exists():
            run.failed += 1
            run.texts.append("")
            if on_progress:
                on_progress(i, total, run.reused)
            continue
        digest = frame_digest(path) if dedupe else ""
        failed_here = False
        if digest and digest in seen:
            text = seen[digest]
            run.reused += 1
        else:
            try:
                text = ocr_image(path, engine, languages, min_chars=min_chars, max_chars=max_chars)
            except Exception:  # noqa: BLE001 - one bad frame must not kill the run
                run.failed += 1
                failed_here = True
                text = ""
            if digest:
                seen[digest] = text
        if text:
            run.ok += 1
        elif not failed_here:
            run.empty += 1
        run.texts.append(text)
        if on_progress:
            on_progress(i, total, run.reused)
    run.elapsed = time.monotonic() - started
    return run
