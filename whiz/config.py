"""Persistent user configuration for whiz.

Config lives at ~/.config/whiz/config.toml (created on demand).
Only Python 3.11+ stdlib tomllib is used for reading; writing is a tiny
hand-rolled TOML emitter so we don't depend on a third-party package.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.environ.get("WHIZ_CONFIG_DIR", Path.home() / ".config" / "whiz"))
CONFIG_PATH = CONFIG_DIR / "config.toml"


@dataclass
class Config:
    # Model alias or absolute path preferred by default (empty => auto-pick best).
    model: str = ""
    # Extra directories to scan for models.
    model_dirs: list[str] = field(default_factory=list)
    # whisper-cli binary path. Empty => auto-detect on PATH.
    whisper_cli: str = ""
    # ffmpeg binary path. Empty => auto-detect on PATH.
    ffmpeg: str = ""
    # Number of CPU threads (0 => auto: min(8, cpu_count)).
    threads: int = 0
    # Spoken language code or "auto".
    language: str = "auto"
    # Enable VAD by default.
    vad: bool = True
    # Path to Silero VAD model (empty => auto-discover or download ggml-silero-vad.bin).
    vad_model: str = ""
    # VAD threshold.
    vad_threshold: float = 0.5
    # Output formats to produce by default.
    outputs: list[str] = field(default_factory=lambda: ["srt", "json"])
    # Print progress to stderr.
    verbose: bool = True
    # Additional flags passed verbatim to whisper-cli.
    extra_args: list[str] = field(default_factory=list)
    # --- Diarization (sherpa-onnx) ---
    # Enable speaker diarization by default.
    diarize: bool = False
    # Known number of speakers (0 => auto-detect via cluster_threshold).
    num_speakers: int = 0
    # Clustering threshold when auto-detecting (larger = fewer speakers; default 0.9 per sherpa-onnx guidance).
    cluster_threshold: float = 0.9
    # Explicit paths to diarization models (empty => auto-discover).
    diarization_segmentation_model: str = ""
    diarization_embedding_model: str = ""
    # --- AI analysis (Ollama / OpenAI-compatible) ---
    # Base URL of the chat completions endpoint (without /chat/completions).
    ai_base_url: str = "http://localhost:11434/v1"
    # Model name (e.g. 'llava', 'qwen2.5-vl', 'gpt-4o-mini'). Empty => error with hint.
    ai_model: str = ""
    # API key (Ollama ignores this; set for cloud OpenAI-compatible providers).
    ai_api_key: str = ""
    # Max frames sent to a vision model (spread evenly across the video).
    ai_max_frames: int = 50
    # --- Speaker voice profiles ---
    # Cosine-similarity threshold for auto-matching a cluster to a stored profile.
    # Higher = stricter (fewer auto-assignments); 0.8 suits 3D-Speaker embeddings.
    speaker_match_threshold: float = 0.8
    # When True, save a voice profile for each named speaker after transcription/merge.
    save_voice_profiles: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_MODEL_SEARCH_DIRS: list[Path] = [
    Path.home() / ".cache" / "whisper",
    Path.home() / "Library" / "Application Support" / "com.unspoken.app" / "WhisperModels",
    Path.home() / "Library" / "Caches" / "whisper",
    Path("/usr/local/share/whisper"),
    Path("/opt/homebrew/share/whisper"),
    Path("/usr/share/whisper"),
]


def _emit_toml(data: dict[str, Any]) -> str:
    """Minimal TOML writer for our flat config schema."""
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, int):
            lines.append(f"{key} = {value}")
        elif isinstance(value, float):
            lines.append(f"{key} = {value}")
        elif isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
        elif isinstance(value, list):
            if not value:
                lines.append(f"{key} = []")
            elif all(isinstance(v, str) for v in value):
                items = ", ".join(
                    '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"' for v in value
                )
                lines.append(f"{key} = [{items}]")
            else:
                items = ", ".join(str(v) for v in value)
                lines.append(f"{key} = [{items}]")
        else:
            lines.append(f"{key} = {value!r}")
    return "\n".join(lines) + "\n"


def load() -> Config:
    """Load config from disk, falling back to defaults."""
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as fh:
            data = tomllib.load(fh)
        # Only keep known keys so older configs don't break dataclass init.
        known = {k: v for k, v in data.items() if k in Config.__dataclass_fields__}
        return Config(**known)
    return Config()


def save(cfg: Config) -> Path:
    """Write config to disk, creating dirs as needed. Returns the path."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(_emit_toml(cfg.to_dict()), encoding="utf-8")
    return CONFIG_PATH


def model_search_dirs(cfg: Config) -> list[Path]:
    """Built-in defaults plus user-configured extra dirs."""
    dirs = list(DEFAULT_MODEL_SEARCH_DIRS)
    for d in cfg.model_dirs:
        dirs.append(Path(d).expanduser())
    # De-dup preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        key = str(d.expanduser().resolve())
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out