"""Tests for whiz.cli helpers — model-picker recommendation heuristic.

Run with: pytest tests/test_cli.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whiz import ai as AI
from whiz import cli
from whiz import config as cfg
from whiz import ocr as OCR
from whiz import screenshots as SC


def test_recommend_model_empty_returns_zero():
    assert cli._recommend_model([], prefer_vision=False) == 0


def test_recommend_model_prefers_non_cloud():
    models = ["gpt-4o-mini:cloud", "qwen2.5:3b", "llava:latest"]
    # qwen2.5:3b (non-cloud + text token) beats gpt-4o-mini (cloud + text token)
    idx = cli._recommend_model(models, prefer_vision=False)
    assert models[idx] == "qwen2.5:3b"


def test_recommend_model_prefers_vision_when_requested():
    models = ["gpt-4o-mini:cloud", "qwen2.5:3b", "llava:latest"]
    # llava matches vision tokens; qwen2.5:3b does not (prefer_vision=True)
    idx = cli._recommend_model(models, prefer_vision=True)
    assert models[idx] == "llava:latest"


def test_recommend_model_all_cloud_picks_best_token_match():
    models = ["devstral-small-2:24b-cloud", "glm-5.1:cloud", "qwen3-coder-next:cloud"]
    idx = cli._recommend_model(models, prefer_vision=False)
    # All cloud (score 0 base); text-token matches add 5. First one with a text
    # token wins. 'devstral' contains 'devstral' token -> score 5.
    assert models[idx] == "devstral-small-2:24b-cloud"


def test_recommend_model_first_wins_on_ties():
    models = ["alpha:cloud", "beta:cloud", "gamma:cloud"]
    # All cloud, no token matches -> tie at score 0 -> first wins (index 0).
    assert cli._recommend_model(models, prefer_vision=False) == 0


def test_recommend_model_prefers_cloud_vision_when_requested():
    models = ["gpt-oss:20b-cloud", "qwen3.5:cloud", "glm-5.1:cloud"]
    # qwen3.5 is cloud vision-capable; gpt-oss and glm-5.1 are not.
    idx = cli._recommend_model(models, prefer_vision=True)
    assert models[idx] == "qwen3.5:cloud"


# ---------- _looks_vision_capable ----------

def test_looks_vision_capable_true_for_known_vision_models():
    for name in ("llava", "llava:latest", "qwen2.5-vl", "minicpm-v", "gpt-4o",
                "gpt-4o-mini", "pixtral-12b", "internvl2", "phi-3.5-vision",
                # Cloud vision-capable models (no 'vl'/'vision' in name).
                "qwen3.5:cloud", "qwen3.5:397b", "kimi-k2.6:cloud",
                "kimi-k2.7-code:cloud", "gemma4:31b", "gemma4:31b-cloud",
                "mistral-large-3:675b", "minimax-m3:cloud"):
        assert cli._looks_vision_capable(name) is True, name


def test_looks_vision_capable_false_for_text_models():
    for name in ("gpt-oss:20b-cloud", "gpt-oss:120b", "llama3.1", "qwen2.5:3b",
                "deepseek-coder", "devstral-small", "gpt-3.5-turbo",
                "glm-5.1:cloud", "qwen3-coder-next:cloud"):
        assert cli._looks_vision_capable(name) is False, name


def test_looks_vision_capable_empty_or_none():
    assert cli._looks_vision_capable("") is False
    assert cli._looks_vision_capable(None) is False


# ---------- _resolve_vision ----------

def _resolve(explicit=False, no=False, frames=False, model="llava"):
    return cli._resolve_vision(
        explicit_vision=explicit, no_vision=no,
        has_frames=frames, model=model,
    )


def test_resolve_vision_no_vision_always_disables():
    # --no-vision wins even if --vision was also set and frames exist.
    use, kind, msg = _resolve(explicit=True, no=True, frames=True, model="llava")
    assert use is False
    assert msg == ""


def test_resolve_vision_explicit_with_frames_enables():
    use, kind, msg = _resolve(explicit=True, no=False, frames=True, model="llava")
    assert use is True
    assert kind == ""
    assert msg == ""


def test_resolve_vision_explicit_without_frames_warns():
    use, kind, msg = _resolve(explicit=True, no=False, frames=False, model="llava")
    assert use is False
    assert kind == "warn"
    assert "no frames manifest" in msg


def test_resolve_vision_explicit_but_text_model_overrides():
    # Explicit --vision is a user override: even with a text-looking model we
    # send the frames (the HTTP layer surfaces a rejection hint if it fails).
    use, kind, msg = _resolve(explicit=True, no=False, frames=True, model="gpt-oss:20b")
    assert use is True
    assert kind == ""
    assert msg == ""


def test_resolve_vision_auto_enables_for_vision_model_with_frames():
    use, kind, msg = _resolve(explicit=False, no=False, frames=True, model="llava")
    assert use is True
    assert kind == "info"
    assert "auto-enabling" in msg


def test_resolve_vision_auto_enables_for_cloud_qwen3_5():
    use, kind, msg = _resolve(explicit=False, no=False, frames=True, model="qwen3.5:cloud")
    assert use is True
    assert kind == "info"
    assert "auto-enabling" in msg


def test_resolve_vision_auto_enables_for_cloud_kimi_k2_6():
    use, kind, msg = _resolve(explicit=False, no=False, frames=True, model="kimi-k2.6:cloud")
    assert use is True
    assert kind == "info"


def test_resolve_vision_text_model_with_frames_stays_text_only_with_hint():
    use, kind, msg = _resolve(explicit=False, no=False, frames=True, model="gpt-oss:20b")
    assert use is False
    assert kind == "hint"
    assert "vision-capable" in msg


def test_resolve_vision_no_frames_is_text_only_silently():
    use, kind, msg = _resolve(explicit=False, no=False, frames=False, model="llava")
    assert use is False
    assert kind == ""
    assert msg == ""


def test_resolve_vision_no_vision_overrides_auto_enable():
    use, kind, msg = _resolve(explicit=False, no=True, frames=True, model="llava")
    assert use is False
    assert kind == ""
    assert msg == ""


# ---------- config coercion (regression: PEP 563 string annotations) ----------

def test_coerce_uses_real_types_not_annotation_strings():
    """Config uses `from __future__ import annotations`, so field .type is a
    string like 'bool'. Coercion must resolve the real type or every value
    would be stored as a raw string (vad="false" is truthy!)."""
    defaults = cfg.Config()
    assert cli._coerce("false", type(defaults.vad)) is False
    assert cli._coerce("true", type(defaults.vad)) is True
    assert cli._coerce("8", type(defaults.threads)) == 8
    assert cli._coerce("0.7", type(defaults.vad_threshold)) == 0.7
    assert cli._coerce("srt,txt", type(defaults.outputs)) == ["srt", "txt"]
    assert cli._coerce("apple", type(defaults.ocr_engine)) == "apple"


def test_config_set_persists_real_bool(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.toml")
    cli.cmd_config_set(SimpleNamespace(assignment="vad=false"))
    assert cfg.load().vad is False
    cli.cmd_config_set(SimpleNamespace(assignment="ocr=true"))
    assert cfg.load().ocr is True
    cli.cmd_config_set(SimpleNamespace(assignment="outputs=srt,html"))
    assert cfg.load().outputs == ["srt", "html"]


def test_config_set_rejects_unknown_key(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.toml")
    with pytest.raises(SystemExit):
        cli.cmd_config_set(SimpleNamespace(assignment="nope=1"))


# ---------- vision classification fixes ----------

def test_looks_vision_capable_recognizes_moondream_and_deepseek_ocr():
    assert cli._looks_vision_capable("moondream") is True
    assert cli._looks_vision_capable("deepseek-ocr:3b") is True


def test_recommend_model_does_not_pick_a_vision_model_for_text():
    """'deepseek-ocr' contains the text token 'deepseek' but is a VLM."""
    models = ["deepseek-ocr:3b", "llama3.2:3b"]
    assert cli._recommend_model(models, prefer_vision=False) == 1


def test_resolve_vision_uses_injected_capability_check():
    use, kind, msg = cli._resolve_vision(
        explicit_vision=False, no_vision=False, has_frames=True,
        model="mystery", is_vision_model=lambda m: True,
    )
    assert use is True and kind == "info"


def test_resolve_vision_text_model_with_ocr_is_informational_not_a_hint():
    """With OCR the screen is already in the transcript, so nothing is lost."""
    use, kind, msg = cli._resolve_vision(
        explicit_vision=False, no_vision=False, has_frames=True,
        model="llama3.2:3b", has_ocr=True,
    )
    assert use is False
    assert kind == "info"
    assert "OCR" in msg


def test_resolve_vision_text_model_without_ocr_still_hints():
    use, kind, msg = cli._resolve_vision(
        explicit_vision=False, no_vision=False, has_frames=True,
        model="llama3.2:3b", has_ocr=False,
    )
    assert use is False
    assert kind == "hint"
    assert "whiz ocr run" in msg


# ---------- OCR wiring ----------

def test_resolve_ocr_off_by_default():
    args = SimpleNamespace(ocr=False, no_ocr=False, ocr_engine="")
    assert cli._resolve_ocr(args, cfg.Config()) == ""


def test_resolve_ocr_no_ocr_overrides_config():
    args = SimpleNamespace(ocr=True, no_ocr=True, ocr_engine="")
    assert cli._resolve_ocr(args, cfg.Config(ocr=True)) == ""


def test_resolve_ocr_enabled_by_flag(monkeypatch):
    monkeypatch.setattr(OCR, "resolve_engine", lambda c, r="": "tesseract")
    monkeypatch.setattr(OCR, "detect", lambda n: OCR.EngineInfo(n, True, "ready"))
    args = SimpleNamespace(ocr=True, no_ocr=False, ocr_engine="")
    assert cli._resolve_ocr(args, cfg.Config()) == "tesseract"


def test_resolve_ocr_enabled_by_config(monkeypatch):
    monkeypatch.setattr(OCR, "resolve_engine", lambda c, r="": "apple")
    monkeypatch.setattr(OCR, "detect", lambda n: OCR.EngineInfo(n, True, "ready"))
    args = SimpleNamespace(ocr=False, no_ocr=False, ocr_engine="")
    assert cli._resolve_ocr(args, cfg.Config(ocr=True)) == "apple"


def test_resolve_ocr_missing_engine_degrades_to_empty(monkeypatch):
    """OCR must never abort a transcription just because no engine is installed."""
    monkeypatch.setattr(OCR, "resolve_engine", lambda c, r="": "")
    args = SimpleNamespace(ocr=True, no_ocr=False, ocr_engine="")
    assert cli._resolve_ocr(args, cfg.Config()) == ""


def test_resolve_ocr_unknown_engine_exits(monkeypatch):
    args = SimpleNamespace(ocr=True, no_ocr=False, ocr_engine="banana")
    with pytest.raises(SystemExit):
        cli._resolve_ocr(args, cfg.Config())


def test_ocr_frame_width_bumps_small_widths():
    config = cfg.Config()
    assert cli._ocr_frame_width(1280, "apple", config) == config.ocr_min_width


def test_ocr_frame_width_leaves_large_and_native_alone():
    config = cfg.Config()
    assert cli._ocr_frame_width(2560, "apple", config) == 2560
    assert cli._ocr_frame_width(0, "apple", config) == 0  # 0 = native resolution


def test_ocr_frame_width_untouched_without_ocr():
    assert cli._ocr_frame_width(1280, "", cfg.Config()) == 1280


# ---------- analyze end-to-end (offline) ----------

def _write_manifest_with_ocr(tmp_path):
    base = tmp_path / "rec"
    frames_dir = SC.frames_dir_for(base)
    frames_dir.mkdir(parents=True)
    entries = [
        SC.FrameEntry(1, 0.0, 4.0, "Vadim", "use GET for export", "seg0001.jpg",
                      ocr="Export API\nMethod: POST"),
        SC.FrameEntry(2, 4.0, 8.0, "Anna", "it returns 400", "seg0002.jpg", ocr="status 400"),
    ]
    SC.write_manifest(entries, frames_dir, SC.frames_manifest_path(base), ocr_engine="tesseract")
    video = tmp_path / "rec.mov"
    video.write_bytes(b"placeholder")
    return video


def test_cmd_analyze_sends_ocr_text_to_a_text_only_model(tmp_path, monkeypatch, capsys):
    """The whole point: a text-only model still receives the on-screen text."""
    video = _write_manifest_with_ocr(tmp_path)
    sent: list[str] = []

    monkeypatch.setattr(cfg, "load", lambda: cfg.Config(ai_model="llama3.2:3b"))
    monkeypatch.setattr(AI, "probe_model", lambda *a, **k: (True, ""))
    monkeypatch.setattr(AI, "_post_chat",
                        lambda base_url, model, messages, api_key, timeout=600:
                        sent.append(str(messages)) or "MEETING")

    args = SimpleNamespace(file=str(video), model="", base_url="", api_key=None,
                           max_frames=None, summary=True, actions=False, plan=False,
                           prompt="", vision=False, no_vision=False)
    assert cli.cmd_analyze(args) == 0

    prompt = "\n".join(sent)
    assert "screen: Export API · Method: POST" in prompt
    assert "screen: status 400" in prompt
    # Text-only model must never be handed image content.
    assert "image_url" not in prompt

    analysis = (tmp_path / "rec.analysis.md").read_text(encoding="utf-8")
    assert "**Vision:** False" in analysis


def test_cmd_analyze_uses_ai_vision_model_when_vision_enabled(tmp_path, monkeypatch):
    video = _write_manifest_with_ocr(tmp_path)
    used: list[str] = []

    monkeypatch.setattr(cfg, "load", lambda: cfg.Config(
        ai_model="llama3.2:3b", ai_vision_model="qwen3-vl:8b"))
    monkeypatch.setattr(AI, "probe_model", lambda *a, **k: (True, ""))
    monkeypatch.setattr(AI, "_post_chat",
                        lambda base_url, model, messages, api_key, timeout=600:
                        used.append(model) or "ok")

    args = SimpleNamespace(file=str(video), model="", base_url="", api_key=None,
                           max_frames=None, summary=True, actions=False, plan=False,
                           prompt="", vision=False, no_vision=False)
    assert cli.cmd_analyze(args) == 0
    assert used and all(m == "qwen3-vl:8b" for m in used)


def _analyze_args(video, **overrides):
    """A cmd_analyze namespace with the common defaults."""
    base = dict(file=str(video), model="", base_url="", api_key=None,
                max_frames=None, summary=True, actions=False, plan=False,
                prompt="", vision=False, no_vision=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _capture_analyze_kwargs(monkeypatch, config):
    """Run cmd_analyze with AI.analyze stubbed; return the captured kwargs dict."""
    captured: dict = {}
    monkeypatch.setattr(cfg, "load", lambda: config)
    monkeypatch.setattr(AI, "probe_model", lambda *a, **k: (True, ""))
    monkeypatch.setattr(AI, "analyze",
                        lambda *a, **kw: captured.update(kw) or "ok")
    return captured


def test_cmd_analyze_passes_chunk_knobs_from_config(tmp_path, monkeypatch):
    """ai_chunk_chars / ai_context_turns reach analyze() instead of its defaults."""
    video = _write_manifest_with_ocr(tmp_path)
    captured = _capture_analyze_kwargs(monkeypatch, cfg.Config(
        ai_model="llama3.2:3b", ai_chunk_chars=40000, ai_context_turns=1))

    assert cli.cmd_analyze(_analyze_args(video)) == 0
    assert captured["chunk_chars"] == 40000
    assert captured["context_turns"] == 1


def test_cmd_analyze_chunk_flags_override_config(tmp_path, monkeypatch):
    """An explicit --chunk-chars/--context-turns beats the config value."""
    video = _write_manifest_with_ocr(tmp_path)
    captured = _capture_analyze_kwargs(monkeypatch, cfg.Config(
        ai_model="llama3.2:3b", ai_chunk_chars=6000, ai_context_turns=3))

    args = _analyze_args(video, chunk_chars=25000, context_turns=0)
    assert cli.cmd_analyze(args) == 0
    assert captured["chunk_chars"] == 25000
    # 0 is explicit "disable rolling context", not "unset" — must not fall back to 3.
    assert captured["context_turns"] == 0


def test_cmd_analyze_chunk_knobs_default_when_attrs_absent(tmp_path, monkeypatch):
    """A namespace predating these knobs falls back to config, not AttributeError."""
    video = _write_manifest_with_ocr(tmp_path)
    captured = _capture_analyze_kwargs(monkeypatch, cfg.Config(ai_model="llama3.2:3b"))

    args = _analyze_args(video)
    assert not hasattr(args, "chunk_chars")  # the pre-existing caller shape
    assert cli.cmd_analyze(args) == 0
    assert captured["chunk_chars"] == cfg.Config().ai_chunk_chars
    assert captured["context_turns"] == cfg.Config().ai_context_turns


def test_config_set_coerces_chunk_chars_to_int(tmp_path, monkeypatch):
    """`whiz config set ai_chunk_chars=40000` stores an int, not a string."""
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.toml")
    assert cli.cmd_config_set(SimpleNamespace(assignment="ai_chunk_chars=40000")) == 0
    assert cfg.load().ai_chunk_chars == 40000
