"""Model discovery, alias resolution, and download.

whisper-cli (whisper.cpp) ships ggml models named like `ggml-large-v3-q5_0.bin`.
whiz scans known directories for these, indexes them by a friendly alias,
and can download new ones from the HuggingFace whisper.cpp repo.
"""

from __future__ import annotations

import re
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from whiz import config as cfg

HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
# VAD models live in a separate repo and are versioned.
VAD_HF_BASE = "https://huggingface.co/ggml-org/whisper-vad/resolve/main"
# Preference order: v5.1.2 has broader whisper-cli compatibility; v6.2.0 is newer.
VAD_MODELS: list[str] = ["ggml-silero-v5.1.2.bin", "ggml-silero-v6.2.0.bin"]
VAD_DEFAULT = VAD_MODELS[0]
# Glob pattern for discovering any Silero VAD model on disk.
VAD_GLOB = "ggml-silero-v*.bin"

# Canonical whisper.cpp models. Order matters for "best" auto-pick (prefer
# turbo/quantized for speed, then full large).
KNOWN_MODELS: list[str] = [
    "ggml-large-v3-turbo-q5_0.bin",
    "ggml-large-v3-turbo.bin",
    "ggml-large-v3-q5_0.bin",
    "ggml-large-v3.bin",
    "ggml-large-v3-turbo-q8_0.bin",
    "ggml-medium-q5_0.bin",
    "ggml-medium.bin",
    "ggml-small-q5_0.bin",
    "ggml-small.bin",
    "ggml-base-q5_0.bin",
    "ggml-base.bin",
    "ggml-tiny-q5_0.bin",
    "ggml-tiny.bin",
]

# Auto-pick preference: prefer turbo quantized, then turbo, then large quantized, then large.
PREFERENCE: list[str] = [
    "ggml-large-v3-turbo-q5_0.bin",
    "ggml-large-v3-turbo.bin",
    "ggml-large-v3-turbo-q8_0.bin",
    "ggml-large-v3-q5_0.bin",
    "ggml-large-v3.bin",
    "ggml-medium-q5_0.bin",
    "ggml-medium.bin",
    "ggml-small-q5_0.bin",
    "ggml-small.bin",
]


@dataclass
class ModelInfo:
    path: Path
    alias: str
    size_mb: float


def _alias_from_name(name: str) -> str:
    """ggml-large-v3-turbo-q5_0.bin -> large-v3-turbo-q5_0"""
    base = name
    if base.startswith("ggml-"):
        base = base[5:]
    if base.endswith(".bin"):
        base = base[:-4]
    return base


def _short_alias(alias: str) -> str:
    """large-v3-turbo-q5_0 -> turbo; large-v3 -> large-v3; medium-q5_0 -> medium-q5."""
    if "turbo" in alias:
        return "turbo"
    return alias


def discover(config: cfg.Config) -> list[ModelInfo]:
    """Scan configured dirs for ggml-*.bin model files."""
    found: dict[str, ModelInfo] = {}
    for d in cfg.model_search_dirs(config):
        if not d.exists():
            continue
        for p in d.iterdir():
            if not p.is_file():
                continue
            if not p.name.endswith(".bin"):
                continue
            if not p.name.startswith("ggml-"):
                continue
            alias = _alias_from_name(p.name)
            if alias not in found:
                found[alias] = ModelInfo(
                    path=p,
                    alias=alias,
                    size_mb=round(p.stat().st_size / (1024 * 1024), 1),
                )
    return sorted(found.values(), key=lambda m: m.alias)


def resolve(name: str, config: cfg.Config) -> Path | None:
    """Resolve a user-supplied model reference to a path.

    Accepts:
      - absolute/relative file path (returned if exists)
      - full alias like 'large-v3-turbo-q5_0'
      - short alias like 'turbo' / 'large-v3' / 'medium'
      - bare 'large-v3' matching ggml-large-v3*.bin
    """
    # Direct path.
    candidate = Path(name).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate

    found = discover(config)
    aliases = {m.alias: m for m in found}

    # Exact alias match.
    if name in aliases:
        return aliases[name].path

    # Short alias: 'turbo' -> any alias containing 'turbo'.
    matches = [m for a, m in aliases.items() if _short_alias(a) == _short_alias(name)]
    if len(matches) == 1:
        return matches[0].path
    # Bare 'large-v3' should match 'large-v3' exactly if present.
    for m in found:
        if m.alias == name:
            return m.path

    # Prefix match: 'large-v3' matches 'large-v3-q5_0' etc. Pick preferred.
    pref = [m for m in found if m.alias.startswith(name)]
    if pref:
        for wanted in PREFERENCE:
            for m in pref:
                if _alias_from_name(wanted) == m.alias:
                    return m.path
        return pref[0].path

    return None


def pick_best(config: cfg.Config) -> Path | None:
    """Auto-pick the best available model by preference order."""
    found = {m.alias: m for m in discover(config)}
    for wanted in PREFERENCE:
        if wanted in found:
            return found[wanted].path
    # Fallback: anything we found.
    all_models = sorted(found.values(), key=lambda m: m.alias)
    return all_models[0].path if all_models else None


def download(model: str, config: cfg.Config, dest_dir: Path | None = None) -> Path:
    """Download a model from the HuggingFace whisper.cpp repo.

    `model` may be a bare name like 'large-v3' (resolved to ggml-large-v3.bin)
    or a full filename like 'ggml-large-v3-turbo-q5_0.bin'.
    """
    filename = model if model.startswith("ggml-") else f"ggml-{model}.bin"
    if not filename.endswith(".bin"):
        filename += ".bin"

    target_dir = dest_dir or (Path.home() / ".cache" / "whisper")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename

    if target.exists():
        raise FileExistsError(f"Already exists: {target}")

    url = f"{HF_BASE}/{filename}"
    # urllib doesn't follow HF redirects to CDN by default; use a redirect-aware fetch.
    print(f"Downloading {filename} from {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "whiz/0.1"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - trusted HF URL
        if resp.status >= 400:
            raise RuntimeError(f"Download failed: HTTP {resp.status} for {url}")
        with target.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=1024 * 1024)
    print(f"Saved to {target} ({round(target.stat().st_size / (1024*1024), 1)} MB)", flush=True)
    return target


def list_known() -> list[str]:
    """Return the canonical list of known whisper.cpp model filenames."""
    return list(KNOWN_MODELS)


def find_vad_model(config: cfg.Config) -> Path | None:
    """Find the Silero VAD model file.

    Precedence: explicit config.vad_model path, then any ggml-silero-v*.bin
    in the model search dirs (preferring v5.1.2, then v6.2.0).
    """
    if config.vad_model:
        p = Path(config.vad_model).expanduser()
        if p.exists():
            return p
    for d in cfg.model_search_dirs(config):
        if not d.exists():
            continue
        # Prefer known versions in order, then any other silero-v*.bin.
        for name in VAD_MODELS:
            candidate = d / name
            if candidate.exists():
                return candidate
        for p in sorted(d.glob(VAD_GLOB)):
            return p
    return None


def download_vad(config: cfg.Config, dest_dir: Path | None = None, version: str = "") -> Path:
    """Download a Silero VAD model from the ggml-org/whisper-vad repo.

    `version` may be empty (picks default v5.1.2), 'v5.1.2', 'v6.2.0',
    or a full filename like 'ggml-silero-v6.2.0.bin'.
    """
    if version and version.startswith("ggml-"):
        filename = version
    elif version:
        filename = f"ggml-silero-{version}.bin" if not version.startswith("silero-") else f"ggml-{version}.bin"
    else:
        filename = VAD_DEFAULT
    if not filename.endswith(".bin"):
        filename += ".bin"

    target_dir = dest_dir or (Path.home() / ".cache" / "whisper")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.exists():
        raise FileExistsError(f"Already exists: {target}")
    url = f"{VAD_HF_BASE}/{filename}"
    print(f"Downloading {filename} from {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "whiz/0.2"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - trusted HF URL
        if resp.status >= 400:
            raise RuntimeError(f"Download failed: HTTP {resp.status} for {url}")
        with target.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=1024 * 1024)
    print(f"Saved to {target} ({round(target.stat().st_size / (1024*1024), 1)} MB)", flush=True)
    return target


def ensure_vad_model(config: cfg.Config, auto_download: bool = True) -> Path | None:
    """Find the VAD model, optionally downloading it if missing."""
    found = find_vad_model(config)
    if found:
        return found
    if not auto_download:
        return None
    try:
        return download_vad(config)
    except FileExistsError:
        return find_vad_model(config)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not download VAD model: {e}", file=sys.stderr)
        return None
