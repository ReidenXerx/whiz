"""Tests for whiz.cli helpers — model-picker recommendation heuristic,
vision resolution, and output fallbacks (HTML without diarization).

Run with: pytest tests/test_cli.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whiz import cli


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


# ---------- output fallbacks (HTML without diarization) ----------

# whisper-cli -oj fixture: two segments the merge/HTML path can parse.
_WHISPER_JSON = (
    '{"transcription": ['
    '{"timestamps":{"from":"00:00:00,000","to":"00:00:02,000"},"text":"hello world"},'
    '{"timestamps":{"from":"00:00:02,000","to":"00:00:04,500"},"text":"second line"}'
    "]}"
)


def _transcribe_args(file, outputs="srt,html", speakers=1):
    return SimpleNamespace(
        file=str(file),
        output="",
        outputs=outputs,
        model="",
        threads=0,
        language="",
        vad=False,
        no_timestamps=False,
        print_progress=False,
        no_progress=True,
        keep_wav=False,
        no_auto_vad_download=True,
        translate=False,
        speakers=speakers,
        no_speakers=False,
        cluster_threshold=None,
        name_speakers=False,
        no_name_speakers=True,
        speakers_names=None,
        screenshots=False,
        no_screenshots=True,
        screenshot_width=None,
        no_voice_profiles=True,
        resume=False,
        verbose=False,
        extra=[],
        dry_run=False,
        analyze=False,
        vision=False,
        no_vision=False,
    )


def _setup_transcribe(monkeypatch, tmp_path, *, diarize_enabled, screenshots=False, name="meeting"):
    """Create a fake audio input + whisper JSON and stub out the heavy machinery.

    Returns the input Path. ``diarize_enabled``/``screenshots`` are what the
    stubbed _build_transcribe_args reports (the real one derives them from
    --speakers/video detection).
    """
    audio = tmp_path / f"{name}.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / f"{name}.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")

    def fake_build(args, config):
        return (["whisper-cli"], "model.bin", audio, audio, False,
                audio.with_suffix(""), diarize_enabled, screenshots)

    monkeypatch.setattr(cli, "_build_transcribe_args", fake_build)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    return audio


def test_transcribe_html_fallback_when_diarization_unavailable(tmp_path, monkeypatch, capsys):
    """--speakers with --outputs html must not silently skip the HTML when
    diarization is unavailable: it degrades to generic 'Speaker' labels."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli, "_run_diarize_or_fallback", lambda wav, config, args: [])

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=1))

    assert rc == 0
    html_path = tmp_path / "meeting.speakers.html"
    assert html_path.exists(), "HTML transcript was silently skipped"
    content = html_path.read_text(encoding="utf-8")
    assert "hello world" in content
    assert ">Speaker<" in content  # generic label, not 'Speaker A'
    assert "Speaker A" not in content
    # The labeled outputs are NOT faked — they need real diarization.
    assert not (tmp_path / "meeting.speakers.srt").exists()
    assert not (tmp_path / "meeting.speakers.txt").exists()
    # Loud degradation, not silence.
    assert "Speakers unavailable" in capsys.readouterr().err


def test_transcribe_html_without_speakers(tmp_path, monkeypatch, capsys):
    """--outputs html on an audio run without diarization still writes the
    HTML (previously silently skipped); no speaker warning is needed."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=False)

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=None))

    assert rc == 0
    assert (tmp_path / "meeting.speakers.html").exists()
    assert not (tmp_path / "meeting.speakers.srt").exists()
    assert "Speakers unavailable" not in capsys.readouterr().err


def test_transcribe_html_and_frames_fallback_for_video(tmp_path, monkeypatch, capsys):
    """Video + --speakers + --outputs html with diarization unavailable:
    the frames manifest AND the HTML are written with generic labels, and
    frames are still inlined into the HTML."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    (tmp_path / "recording.wav.json").write_text(_WHISPER_JSON, encoding="utf-8")
    frames_dir = tmp_path / "recording.frames"
    frames_dir.mkdir()
    (frames_dir / "seg0001.jpg").write_bytes(b"\xff\xd8jpeg\xff\xd9")
    manifest = tmp_path / "recording.frames.json"

    def fake_build(args, config):
        wav = tmp_path / "recording.wav"
        return (["whisper-cli"], "model.bin", wav, video, False,
                tmp_path / "recording", True, True)

    monkeypatch.setattr(cli, "_build_transcribe_args", fake_build)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli, "_run_diarize_or_fallback", lambda wav, config, args: [])
    monkeypatch.setattr(
        cli, "_extract_and_manifest_screenshots",
        lambda in_path, merged, of_base, ffmpeg, width, dry_run: (frames_dir, manifest),
    )
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())

    rc = cli.cmd_transcribe(_transcribe_args(video, outputs="html", speakers=1))

    assert rc == 0
    html = (tmp_path / "recording.speakers.html").read_text(encoding="utf-8")
    assert "hello world" in html
    assert "<img" in html  # frame inlined even without speaker labels
    assert "Speakers unavailable" in capsys.readouterr().err
    # Labeled outputs are not faked.
    assert not (tmp_path / "recording.speakers.srt").exists()


def test_transcribe_no_crash_when_json_missing(tmp_path, monkeypatch):
    """A missing whisper JSON on the unlabeled path must warn, not crash
    (the old screenshots-only block read an unbound 'result')."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")

    def fake_build(args, config):
        wav = tmp_path / "recording.wav"
        return (["whisper-cli"], "model.bin", wav, video, False,
                tmp_path / "recording", False, True)

    monkeypatch.setattr(cli, "_build_transcribe_args", fake_build)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())

    rc = cli.cmd_transcribe(_transcribe_args(video, outputs="html", speakers=None))

    assert rc == 0
    assert not (tmp_path / "recording.speakers.html").exists()


def test_build_args_forces_json_with_html_output(tmp_path, monkeypatch):
    """--outputs html must force -oj so the HTML can be rendered even when
    diarization is unavailable (segments are needed to build the page)."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.M, "pick_best", lambda config: Path("/models/turbo.bin"))
    monkeypatch.setattr(cli, "_find_whisper_cli", lambda configured="": "whisper-cli")

    cmd, *_rest = cli._build_transcribe_args(
        _transcribe_args(audio, outputs="html", speakers=None),
        cli.cfg.Config(vad=False),
    )

    assert "-oj" in cmd


# ---------- merge fallback ----------


def _merge_args(file, outputs="html", speakers=1):
    return SimpleNamespace(
        file=str(file), json="", outputs=outputs, speakers=speakers,
        no_speakers=False, cluster_threshold=None, name_speakers=False,
        no_name_speakers=True, speakers_names=None, screenshots=False,
        no_screenshots=False, screenshot_width=None, no_voice_profiles=True,
    )


def _raise_sherpa_missing(wav, config, num_speakers=0, threshold=0.9):
    raise RuntimeError("The 'sherpa_onnx' package is required for diarization")


def test_merge_html_fallback_when_diarization_unavailable(tmp_path, monkeypatch, capsys):
    """whiz merge --speakers --outputs html with sherpa-onnx missing degrades
    to a generic-label HTML transcript instead of exiting."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 0
    html_path = tmp_path / "meeting.m4a.speakers.html"
    assert html_path.exists(), "HTML transcript was silently skipped"
    content = html_path.read_text(encoding="utf-8")
    assert "hello world" in content
    assert ">Speaker<" in content
    # Labeled outputs are not faked.
    assert not (tmp_path / "meeting.speakers.srt").exists()
    assert "diarization unavailable" in capsys.readouterr().err


def test_merge_still_raises_when_nothing_else_requested(tmp_path, monkeypatch):
    """Without --outputs html (or screenshots) there is nothing to fall back
    to: an explicit --speakers merge against missing sherpa-onnx stays loud."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    with pytest.raises(SystemExit, match="sherpa_onnx"):
        cli.cmd_merge(_merge_args(audio, outputs="", speakers=1))
