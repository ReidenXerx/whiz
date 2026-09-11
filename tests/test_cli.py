"""Tests for whiz.cli helpers — model-picker recommendation heuristic,
vision resolution, output fallbacks (HTML without diarization), and the
proactive diarization auto-setup (user decision, 2026-09-05).

Run with: pytest tests/test_cli.py
"""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whiz import cli
from whiz.diarize import DiarSegment


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
        vad_threshold=None,
        no_timestamps=False,
        print_progress=False,
        no_progress=True,
        keep_wav=False,
        no_auto_vad_download=True,
        no_auto_diarization_setup=False,
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
    --speakers/video detection). The proactive diarization auto-setup is
    stubbed ready=True so no test ever runs pip or downloads models.
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
    monkeypatch.setattr(cli, "_ensure_diarization_ready", lambda config, dry_run=False, setup_allowed=True: True)
    return audio


def test_transcribe_html_fallback_when_diarization_unavailable(tmp_path, monkeypatch, capsys):
    """--speakers with --outputs html must not silently skip the HTML when
    diarization is unavailable: it degrades to generic 'Speaker' labels.
    Uses the real _run_diarize_or_fallback (with run_diarization stubbed to
    raise the sherpa-missing error) so the 'diarization unavailable' warn is
    exercised end-to-end."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=1))

    assert rc == 0
    html_path = tmp_path / "meeting.speakers.html"
    assert html_path.exists(), "HTML transcript was silently skipped"
    content = html_path.read_text(encoding="utf-8")
    assert "hello world" in content
    assert ">Speaker<" in content  # generic label, not 'Speaker A'
    assert "Speaker A" not in content
    # The labeled SRT is NOT faked — it needs real diarization. (The
    # generic-label .speakers.txt IS written on this audio run: `whiz analyze`
    # needs a frames manifest or a .speakers.txt to find a transcript.)
    assert not (tmp_path / "meeting.speakers.srt").exists()
    txt = (tmp_path / "meeting.speakers.txt").read_text(encoding="utf-8")
    assert "Speaker (00:00:00):" in txt
    # The degraded page self-identifies with a muted note line.
    assert 'class="note"' in content
    assert "No speaker diarization" in content
    # Loud degradation via the real fallback helper, honest about what
    # happens next (audio run, explicit html → generic labels written).
    err = capsys.readouterr().err
    assert "diarization unavailable" in err
    assert "Falling back to generic 'Speaker' labels" in err


def test_transcribe_html_without_speakers(tmp_path, monkeypatch, capsys):
    """--outputs html on an audio run without diarization still writes the
    HTML (previously silently skipped); no speaker warning is needed."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=False)

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=None))

    assert rc == 0
    assert (tmp_path / "meeting.speakers.html").exists()
    assert (tmp_path / "meeting.speakers.txt").exists()
    assert not (tmp_path / "meeting.speakers.srt").exists()
    err = capsys.readouterr().err
    assert "diarization unavailable" not in err  # diarization was never on


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
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    monkeypatch.setattr(
        cli, "_extract_and_manifest_screenshots",
        lambda in_path, merged, of_base, ffmpeg, width, dry_run: (frames_dir, manifest, False),
    )
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())

    rc = cli.cmd_transcribe(_transcribe_args(video, outputs="html", speakers=1))

    assert rc == 0
    html = (tmp_path / "recording.speakers.html").read_text(encoding="utf-8")
    assert "hello world" in html
    assert "<img" in html  # frame inlined even without speaker labels
    assert 'class="note"' in html  # degraded page self-identifies
    # Explicit --speakers + artifacts to write → loud warn via the real
    # fallback helper, honest that generic labels are being written.
    err = capsys.readouterr().err
    assert "diarization unavailable" in err
    assert "Falling back to generic 'Speaker' labels" in err
    # Labeled outputs are not faked; video runs have a frames manifest, so
    # no generic-label .speakers.txt is needed for `whiz analyze`.
    assert not (tmp_path / "recording.speakers.srt").exists()
    assert not (tmp_path / "recording.speakers.txt").exists()


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


def test_find_whisper_json_dotted_output_stem(tmp_path):
    """-o /x/out.v2 -> the JSON is out.v2.json, not out.json (with_suffix
    would eat the dotted stem and read a stale transcript from an old run)."""
    of_base = tmp_path / "out.v2"
    wav = tmp_path / "in.wav"
    wanted = tmp_path / "out.v2.json"
    wanted.write_text("{}", encoding="utf-8")
    # A stale out.json (from a run of the old, buggy naming) must never win
    # over the run's real out.v2.json.
    (tmp_path / "out.json").write_text("{}", encoding="utf-8")
    found = cli._find_whisper_json(of_base, wav, of_passed=True)
    assert found == wanted


def test_find_whisper_json_of_passed_stale_only_never_wins(tmp_path):
    """with -of out.v2, whisper-cli writes out.v2.json and nothing else —
    when only the stale out.json exists, it must NOT be ingested: the
    caller warns on the missing out.v2.json instead of merging old data."""
    of_base = tmp_path / "out.v2"
    wav = tmp_path / "in.wav"
    (tmp_path / "out.json").write_text("{}", encoding="utf-8")  # stale
    found = cli._find_whisper_json(of_base, wav, of_passed=True)
    assert found == tmp_path / "out.v2.json"  # reported missing, not stale


# ---------- merge fallback ----------


def _merge_args(file, outputs="html", speakers=1, speakers_names=None, no_speakers=False):
    return SimpleNamespace(
        file=str(file), json="", outputs=outputs, speakers=speakers,
        no_speakers=no_speakers, no_auto_diarization_setup=False,
        cluster_threshold=None, name_speakers=False,
        no_name_speakers=True, speakers_names=speakers_names, screenshots=False,
        no_screenshots=False, screenshot_width=None, no_voice_profiles=True,
    )


def _raise_sherpa_missing(wav, config, num_speakers=0, threshold=0.9):
    # M2 (wave-1 audit): unavailability arrives as the TYPED exception now;
    # the "sherpa_onnx" token stays in the message for the SystemExit
    # match assertions below.
    raise cli.D.DiarizationUnavailable("The 'sherpa_onnx' package is required for diarization")


def _stub_setup_unavailable(monkeypatch):
    """Simulate the proactive setup having run and failed (pip offline,
    model download error, ...) — the safety-net case every degraded path
    below still guards. Without this stub cmd_merge would attempt a REAL
    `pip install sherpa-onnx`, since merge now calls the setup before
    diarizing."""
    monkeypatch.setattr(
        cli, "_ensure_diarization_ready",
        lambda config, dry_run=False, setup_allowed=True: False,
    )


def _stub_setup_ready(monkeypatch):
    """Simulate a successful one-time setup (package + models ready)."""
    monkeypatch.setattr(
        cli, "_ensure_diarization_ready",
        lambda config, dry_run=False, setup_allowed=True: True,
    )


def test_merge_html_fallback_when_diarization_unavailable(tmp_path, monkeypatch, capsys):
    """whiz merge --speakers --outputs html with sherpa-onnx missing degrades
    to a generic-label HTML transcript instead of exiting."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 0
    html_path = tmp_path / "meeting.m4a.speakers.html"
    assert html_path.exists(), "HTML transcript was silently skipped"
    content = html_path.read_text(encoding="utf-8")
    assert "hello world" in content
    assert ">Speaker<" in content
    assert 'class="note"' in content  # degraded page self-identifies
    # Labeled SRT is not faked (this previously asserted meeting.speakers.srt
    # — a file this code path never writes, so it guarded nothing).
    assert not (tmp_path / "meeting.m4a.speakers.srt").exists()
    # Audio fallback also writes a generic-label .speakers.txt so
    # `whiz analyze` finds a transcript.
    txt = (tmp_path / "meeting.m4a.speakers.txt").read_text(encoding="utf-8")
    assert "Speaker (00:00:00):" in txt
    assert "diarization unavailable" in capsys.readouterr().err


def test_merge_still_raises_when_nothing_else_requested(tmp_path, monkeypatch):
    """Without --outputs html (or screenshots) there is nothing to fall back
    to: an explicit --speakers merge against missing sherpa-onnx stays loud."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    with pytest.raises(SystemExit, match="sherpa_onnx"):
        cli.cmd_merge(_merge_args(audio, outputs="", speakers=1))


# ---------- command-level success paths and new fallback behaviors ----------


def test_transcribe_diarized_success_writes_labeled_outputs(tmp_path, monkeypatch):
    """Happy path: diarization succeeds -> labeled .speakers.srt/.txt/.html
    all written with letterized labels, and no degraded-run note."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    diar = [
        DiarSegment(start=0.0, end=3.0, speaker=0),   # Speaker A
        DiarSegment(start=3.0, end=5.0, speaker=1),   # Speaker B
    ]
    monkeypatch.setattr(cli, "_run_diarize_or_fallback", lambda wav, config, args: diar)

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=2))

    assert rc == 0
    srt = (tmp_path / "meeting.speakers.srt").read_text(encoding="utf-8")
    assert "Speaker A:" in srt and "Speaker B:" in srt
    txt = (tmp_path / "meeting.speakers.txt").read_text(encoding="utf-8")
    assert "Speaker A (00:00:00):" in txt
    html = (tmp_path / "meeting.speakers.html").read_text(encoding="utf-8")
    assert "Speaker A" in html
    assert 'class="note"' not in html  # not a degraded run


def test_merge_diarized_success_writes_labeled_outputs(tmp_path, monkeypatch):
    """whiz merge happy path: diarization succeeds -> labeled srt/txt/html
    written under the JSON stem with letterized labels, no note."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
            DiarSegment(start=3.0, end=5.0, speaker=1),
        ],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=2))

    assert rc == 0
    srt = (tmp_path / "meeting.m4a.speakers.srt").read_text(encoding="utf-8")
    assert "Speaker A:" in srt and "Speaker B:" in srt
    txt = (tmp_path / "meeting.m4a.speakers.txt").read_text(encoding="utf-8")
    assert "Speaker A (00:00:00):" in txt
    html = (tmp_path / "meeting.m4a.speakers.html").read_text(encoding="utf-8")
    assert "Speaker A" in html
    assert 'class="note"' not in html


def test_transcribe_fallback_warns_discarded_speakers_names(tmp_path, monkeypatch, capsys):
    """The fallback must say --speakers-names was discarded, not let the
    user believe the names were applied."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    args = _transcribe_args(audio, outputs="html", speakers=1)
    args.speakers_names = ["Alice,Bob"]
    rc = cli.cmd_transcribe(args)

    assert rc == 0
    # Whitespace-normalized: rich wraps long status lines at the console
    # width, which may split the phrase across lines (assert on content,
    # not on wrap luck).
    assert "--speakers-names had no effect" in " ".join(capsys.readouterr().err.split())


def test_merge_fallback_warns_discarded_speakers_names(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1,
                                   speakers_names=["Alice,Bob"]))

    assert rc == 0
    # Wrap-insensitive (see the transcribe twin above).
    assert "--speakers-names had no effect" in " ".join(capsys.readouterr().err.split())


def test_merge_zero_segments_falls_back_to_unlabeled_html(tmp_path, monkeypatch, capsys):
    """Diarization runs but finds no speech: warn + generic-label HTML
    (and .speakers.txt on audio runs) instead of crashing or skipping."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 0
    html = (tmp_path / "meeting.m4a.speakers.html").read_text(encoding="utf-8")
    assert ">Speaker<" in html
    assert 'class="note"' in html
    assert (tmp_path / "meeting.m4a.speakers.txt").exists()
    assert "Diarization produced no segments; writing unlabeled output" in capsys.readouterr().err


def test_merge_returns_1_when_nothing_written(tmp_path, monkeypatch, capsys):
    """No html/screenshots requested + no segments -> nothing written, AND
    nothing was kept either -> rc=1: a silent rc=0 would read as success.
    (The kept-only case is now rc=0 — see the no-clobber test above.)"""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="", speakers=1))

    assert rc == 1
    assert "Diarization produced no segments; nothing to merge." in capsys.readouterr().err


def _capture_status(monkeypatch):
    calls: list[tuple[str, str, str | None]] = []

    def fake(msg, kind="info", detail=None):
        calls.append((msg, kind, detail))

    monkeypatch.setattr(cli.ui, "status", fake)
    return calls


def test_diarize_fallback_warns_for_explicit_speakers(tmp_path, monkeypatch):
    """Explicit --speakers degrades loudly (warn), not with the quiet hint
    used for merely auto-enabled diarization."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    calls = _capture_status(monkeypatch)

    cli._run_diarize_or_fallback(audio, cli.cfg.Config(), _transcribe_args(audio, speakers=1))

    kinds = [k for _m, k, _d in calls]
    assert "warn" in kinds


def test_diarize_fallback_stays_hint_when_auto_enabled(tmp_path, monkeypatch):
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    calls = _capture_status(monkeypatch)

    cli._run_diarize_or_fallback(audio, cli.cfg.Config(), _transcribe_args(audio, speakers=None))

    kinds = [k for _m, k, _d in calls]
    assert "hint" in kinds
    assert "warn" not in kinds


# ---------- blocker fixes: no-clobber + explicit-html gating ----------


def test_transcribe_fallback_never_clobbers_existing_named_outputs(tmp_path, monkeypatch, capsys):
    """Blocker 2 regression: a diarized run (with real names) left a named
    .speakers.txt/.html; a later degraded run must keep them, not collapse
    them to a one-line generic 'Speaker' wall."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    # Earlier diarized run's outputs, with real speaker names.
    named_txt = tmp_path / "meeting.speakers.txt"
    named_html = tmp_path / "meeting.speakers.html"
    named_srt = tmp_path / "meeting.speakers.srt"
    named_txt.write_text("Vadim (00:00:00): real named content\n", encoding="utf-8")
    named_html.write_text("<html>named run</html>", encoding="utf-8")
    named_srt.write_text("1\n00:00:00,000 --> ...\nVadim: real named content\n", encoding="utf-8")

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=1))

    assert rc == 0
    assert named_txt.read_text(encoding="utf-8") == "Vadim (00:00:00): real named content\n"
    assert named_html.read_text(encoding="utf-8") == "<html>named run</html>"
    assert "real named content" in named_srt.read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert "kept" in err and "meeting.speakers.txt" in err
    assert "kept" in err and "meeting.speakers.html" in err


def test_merge_fallback_never_clobbers_existing_named_outputs(tmp_path, monkeypatch, capsys):
    """Same no-clobber contract on the merge path. The run writes nothing
    (both artifacts were kept) — which is a no-op SUCCESS, not a failure:
    the named outputs were correctly preserved, and an rc=1 here would
    false-alarm `whiz merge ... || alert` wrappers on identical re-runs
    (review follow-up). Warnings explain the keeps."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    named_txt = tmp_path / "meeting.m4a.speakers.txt"
    named_html = tmp_path / "meeting.m4a.speakers.html"
    named_txt.write_text("Vadim (00:00:00): real named content\n", encoding="utf-8")
    named_html.write_text("\u003chtml\u003enamed run\u003c/html\u003e", encoding="utf-8")

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 0  # kept-only is a no-op success (review follow-up)
    assert named_txt.read_text(encoding="utf-8") == "Vadim (00:00:00): real named content\n"
    assert named_html.read_text(encoding="utf-8") == "\u003chtml\u003enamed run\u003c/html\u003e"
    err = capsys.readouterr().err
    assert "kept" in err and "meeting.m4a.speakers.txt" in err
    assert "kept" in err and "meeting.m4a.speakers.html" in err


def test_transcribe_fallback_rerun_overwrites_existing_degraded_outputs(tmp_path, monkeypatch, capsys):
    """Review follow-up (idempotence): the second identical degraded run
    must NOT say "kept" for its own degraded output — that file has no
    speaker names to destroy, and keeping it would leave a stale transcript
    forever whenever --model/--language/the audio change. The degraded
    file is cheaply detectable (provenance note in the HTML, all-generic
    label lines in the txt), so the run rewrites both files with rc=0."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    # Run 1: writes the degraded outputs.
    rc1 = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=1))
    assert rc1 == 0
    degraded_html = tmp_path / "meeting.speakers.html"
    degraded_txt = tmp_path / "meeting.speakers.txt"
    assert degraded_html.exists() and degraded_txt.exists()
    assert cli._GENERIC_LABEL_NOTE in degraded_html.read_text(encoding="utf-8")

    # Run 2: identical command. A naive .exists() guard keeps the stale
    # degraded files and (on merge) exits 1; the fix rewrites them.
    rc2 = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=1))

    assert rc2 == 0
    html2 = degraded_html.read_text(encoding="utf-8")
    txt2 = degraded_txt.read_text(encoding="utf-8")
    assert "hello world" in html2  # fresh content, not the stale run-1 file
    assert cli._GENERIC_LABEL_NOTE in html2
    assert "Speaker (00:00:00):" in txt2
    err = capsys.readouterr().err
    assert "overwriting" in err  # says what it did to the degraded files
    assert "kept" not in err      # ...and does NOT claim to keep them


def test_merge_fallback_rerun_overwrites_existing_degraded_outputs(tmp_path, monkeypatch, capsys):
    """Merge half of the idempotence fix: the second identical degraded
    merge rewrites its own degraded outputs (rc=0, "overwriting" note),
    instead of keeping them and exiting 1 on a no-op run."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc1 = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))
    assert rc1 == 0
    degraded_html = tmp_path / "meeting.m4a.speakers.html"
    degraded_txt = tmp_path / "meeting.m4a.speakers.txt"
    assert degraded_html.exists() and degraded_txt.exists()

    rc2 = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc2 == 0
    assert "hello world" in degraded_html.read_text(encoding="utf-8")
    assert "Speaker (00:00:00):" in degraded_txt.read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert "overwriting" in err
    assert "kept" not in err


def test_transcribe_fallback_mixed_named_html_kept_degraded_txt_overwritten(tmp_path, monkeypatch, capsys):
    """Per-file decision (review follow-up): an earlier diarized run left a
    NAMED html but a degraded txt on disk; the fallback must keep the
    named file and still refresh the degraded one — the guard protects
    speaker names, not whiz's own fallback output."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    named_html = tmp_path / "meeting.speakers.html"
    degraded_txt = tmp_path / "meeting.speakers.txt"
    named_html.write_text("<html>named run</html>", encoding="utf-8")
    degraded_txt.write_text("Speaker (00:00:00): stale degraded content\n", encoding="utf-8")

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=1))

    assert rc == 0
    assert named_html.read_text(encoding="utf-8") == "<html>named run</html>"  # kept
    assert "stale degraded content" not in degraded_txt.read_text(encoding="utf-8")  # refreshed
    assert "Speaker (00:00:00):" in degraded_txt.read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert "kept" in err and "meeting.speakers.html" in err
    assert "overwriting" in err


def test_transcribe_config_html_is_not_degraded(tmp_path, monkeypatch, capsys):
    """Blocker 3: html in config.outputs alone must NOT trigger the
    degraded fallback on a failed-diarization run (a typed --outputs html
    is a promise; a config default describes the success path)."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    def fake_load():
        config = cli.cfg.Config()
        config.outputs = ["srt", "html"]
        return config

    monkeypatch.setattr(cli.cfg, "load", fake_load)

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="", speakers=1))

    # L (wave-1 audit): rc symmetry with merge — the explicit --speakers
    # degraded and nothing speaker-related was written, so the run exits
    # nonzero instead of reporting success.
    assert rc == 1
    assert not (tmp_path / "meeting.speakers.html").exists()
    assert not (tmp_path / "meeting.speakers.txt").exists()
    # The diarization-unavailable status still fires (it is honest about the
    # run) — it just does not promise degraded artifacts it will not write.
    err = capsys.readouterr().err
    assert "diarization unavailable" in err
    assert "Falling back to generic" not in err


def test_merge_config_html_is_not_degraded(tmp_path, monkeypatch, capsys):
    """Blocker 3 on the merge path: config-only html + explicit --speakers
    with sherpa missing stays loud (SystemExit), exactly like master."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")

    def fake_load():
        config = cli.cfg.Config()
        config.outputs = ["srt", "html"]
        return config

    monkeypatch.setattr(cli.cfg, "load", fake_load)
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    with pytest.raises(SystemExit, match="sherpa_onnx"):
        cli.cmd_merge(_merge_args(audio, outputs="", speakers=1))


def test_transcribe_fallback_honest_skip_when_nothing_to_write(tmp_path, monkeypatch, capsys):
    """Message fidelity: audio run, NO explicit html (and no video frames):
    diarization fails → the message must say "skipping", not promise a
    generic-label fallback that will not happen."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt", speakers=1))

    # L (wave-1 audit): rc symmetry with merge — nothing speaker-related was
    # written for the explicit --speakers, so nonzero. (A degraded run that
    # DOES write generic-label outputs still exits 0 — see the html fallback
    # tests above.)
    assert rc == 1
    err = capsys.readouterr().err
    assert "Skipping speaker labels" in err
    assert "Falling back to generic" not in err
    assert not (tmp_path / "meeting.speakers.html").exists()


def test_merge_sherpa_missing_does_not_double_warn(tmp_path, monkeypatch, capsys):
    """Review fix: the except branch already said 'falling back to generic
    labels'; the follow-up 'produced no segments' block must not repeat it.
    It stays for the genuinely-new case (ran and returned nothing) — covered
    by test_merge_zero_segments_falls_back_to_unlabeled_html."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 0
    err = capsys.readouterr().err
    assert err.count("diarization unavailable") == 1
    assert err.count("Diarization produced no segments") == 0


# ---------- proactive diarization auto-setup (user decision, 2026-09-05) ----------
#
# Policy: when diarization is about to run — auto-enabled for video or
# explicitly requested — whiz performs the one-time setup itself (pip
# install sherpa-onnx + ~90 MB model download) instead of degrading; the
# fallbacks above remain only as the safety net for a failed setup or
# --no-auto-diarization-setup. These tests run the REAL setup helpers by
# stubbing their inputs (availability probe, pip, model download) so no
# test ever touches the network or mutates the venv.


def test_no_auto_diarization_setup_flags_registered():
    """--no-auto-diarization-setup exists on transcribe, merge AND speakers
    match (review round 3: cmd_speakers_match read the flag via getattr but
    the subparser never registered it, so the default always won and setup
    was unconditionally allowed on a command whose help calls itself a dry
    run). Defaults to False everywhere: setup-on-first-use is on by default."""
    parser = cli.build_parser()
    args = parser.parse_args(["transcribe", "x.wav", "--no-auto-diarization-setup"])
    assert args.no_auto_diarization_setup is True
    args = parser.parse_args(["transcribe", "x.wav"])
    assert args.no_auto_diarization_setup is False
    args = parser.parse_args(["merge", "x.wav", "--no-auto-diarization-setup"])
    assert args.no_auto_diarization_setup is True
    args = parser.parse_args(["merge", "x.wav"])
    assert args.no_auto_diarization_setup is False
    args = parser.parse_args(["speakers", "match", "x.wav", "--no-auto-diarization-setup"])
    assert args.no_auto_diarization_setup is True
    args = parser.parse_args(["speakers", "match", "x.wav"])
    assert args.no_auto_diarization_setup is False


def test_ensure_diarization_ready_short_circuits_when_available(monkeypatch):
    """Already-available diarization returns True with no install, no
    download, and no status output — every later run stays quiet."""
    calls = _capture_status(monkeypatch)

    def _boom(*_a, **_k):
        raise AssertionError("nothing may run when diarization is ready")

    monkeypatch.setattr(cli, "_diarization_available", lambda config: True)
    monkeypatch.setattr(cli, "_install_sherpa_onnx", _boom)
    monkeypatch.setattr(cli.D, "download_diarization_models", _boom)

    assert cli._ensure_diarization_ready(cli.cfg.Config()) is True
    assert calls == []


def test_ensure_diarization_ready_happy_path_installs_then_downloads(monkeypatch, capsys):
    """Fresh machine: install the package FIRST, then download the models —
    models without the package that runs them would leave half a setup —
    then confirm readiness with a final availability re-check."""
    events: list[str] = []
    # Not ready at entry; ready once the models are on disk — the re-check
    # then sees the freshly installed package + downloaded models.
    monkeypatch.setattr(cli, "_diarization_available", lambda config: "download" in events)
    # A None entry makes `import sherpa_onnx` raise ImportError, forcing the
    # install branch deterministically regardless of the host venv.
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    monkeypatch.setattr(cli, "_install_sherpa_onnx", lambda: events.append("install") or True)
    monkeypatch.setattr(cli.D, "download_diarization_models", lambda: events.append("download"))
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)
    # Consent auto-allows on non-tty stdin; pin it so `pytest -s` (a real
    # terminal stdin) can't turn this wiring test into a live y/N prompt.
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))

    assert cli._ensure_diarization_ready(cli.cfg.Config()) is True
    assert events == ["install", "download"]
    assert "Diarization models downloaded" in capsys.readouterr().err


def test_ensure_diarization_ready_opt_out_skips_setup(monkeypatch, capsys):
    """--no-auto-diarization-setup (setup_allowed=False): report the honest
    False but install and download NOTHING — silent degradation is the
    user's explicit choice."""
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)

    def _boom(*_a, **_k):
        raise AssertionError("opted-out setup must not install or download")

    monkeypatch.setattr(cli, "_install_sherpa_onnx", _boom)
    monkeypatch.setattr(cli.D, "download_diarization_models", _boom)
    # --no-auto-diarization-setup short-circuits BEFORE any prompt (consent
    # ordering): an opted-out run must never sit at a y/N question.
    monkeypatch.setattr(builtins, "input", _boom_input)

    assert cli._ensure_diarization_ready(cli.cfg.Config(), setup_allowed=False) is False
    assert capsys.readouterr().err == ""


def test_ensure_diarization_ready_dry_run_never_sets_up(monkeypatch, capsys):
    """dry_run announces the setup it WOULD perform and runs none of it."""
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)

    def _boom(*_a, **_k):
        raise AssertionError("dry-run must not install or download")

    monkeypatch.setattr(cli, "_install_sherpa_onnx", _boom)
    monkeypatch.setattr(cli.D, "download_diarization_models", _boom)
    # dry_run reports and returns BEFORE consent too: a dry-run must never
    # sit at a prompt (or pip-install) anything.
    monkeypatch.setattr(builtins, "input", _boom_input)

    assert cli._ensure_diarization_ready(cli.cfg.Config(), dry_run=True) is False
    err = capsys.readouterr().err
    assert "DRY-RUN" in err
    assert "~90 MB" in err


def test_install_sherpa_onnx_targets_running_venv_and_reports_progress(monkeypatch, capsys):
    """The installer runs pip via sys.executable (installs into the RUNNING
    venv — dev uv venv, pipx venv, anything; a bare `pipx` binary would
    miss dev venvs), announces itself with the --no-auto-diarization-setup
    opt-out, and verifies the fresh wheel is importable."""
    seen: dict[str, list[str]] = {}

    def fake_run(cmd, check=False):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    assert cli._install_sherpa_onnx() is True
    # The spec the diarize extra declares, not a bare package name (review
    # round 3): a bare `pip install sherpa-onnx` could land an older version
    # than the documented manual path (`pipx inject whiz 'whiz[diarize]'`).
    assert seen["cmd"] == [sys.executable, "-m", "pip", "install", cli._DIARIZE_REQUIREMENT]
    assert ">=" in cli._DIARIZE_REQUIREMENT
    err = capsys.readouterr().err
    assert "installing the diarize extra" in err    # status line up front
    assert "--no-auto-diarization-setup" in err      # the opt-out is surfaced
    assert "sherpa-onnx installed" in err


def test_ensure_diarization_ready_install_failure_returns_false(monkeypatch, capsys):
    """A failed pip install (offline, disk full, ...) returns False so the
    caller stays on its degraded path — with the manual remediation hint.
    No model download may follow the miss."""
    monkeypatch.setattr(cli, "_diarization_available", lambda config: False)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)  # force the install branch
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, check=False: SimpleNamespace(returncode=1))

    def _boom(*_a, **_k):
        raise AssertionError("models must not download when the install failed")

    monkeypatch.setattr(cli.D, "download_diarization_models", _boom)
    # Pin non-tty stdin: consent must auto-allow (plain pytest already is
    # non-tty; `-s` on a terminal would otherwise hit the live prompt).
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))

    assert cli._ensure_diarization_ready(cli.cfg.Config()) is False
    err = capsys.readouterr().err
    assert "pip install sherpa-onnx failed" in err
    assert "pipx inject whiz 'whiz[diarize]'" in err


def _fresh_machine_stubs(monkeypatch, events):
    """Shared wiring-test setup: stub the setup's inputs so the REAL
    _ensure_diarization_ready runs inside the real command paths, with
    ``events`` recording the (stubbed) install + download steps."""
    monkeypatch.setattr(cli, "_diarization_available", lambda config: "download" in events)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    monkeypatch.setattr(cli, "_install_sherpa_onnx", lambda: events.append("install") or True)
    monkeypatch.setattr(cli.D, "download_diarization_models", lambda: events.append("download"))
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    monkeypatch.setattr(cli.M, "pick_best", lambda config: Path("/models/turbo.bin"))
    monkeypatch.setattr(cli, "_find_whisper_cli", lambda configured="": "whisper-cli")
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda configured="": "ffmpeg")
    monkeypatch.setattr(cli.aud, "extract_audio", lambda src, ffmpeg, dest_dir=None, dry_run=False: src.with_suffix(".wav"))
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    # The real _ensure_diarization_ready now consults _auto_setup_consent,
    # which auto-allows on non-tty stdin. Pin it so the wiring tests behave
    # identically under `pytest -s` (real terminal stdin) as under capture.
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))


def test_transcribe_auto_diarization_setup_success_writes_speakers_html(tmp_path, monkeypatch, capsys):
    """The headline behavior: `whiz transcribe recording.mov --outputs html`
    on a fresh machine just works — the setup runs (real
    _build_transcribe_args, real _ensure_diarization_ready), diarization
    produces real labels, and the old quiet auto-skip hint never appears."""
    events: list[str] = []
    _fresh_machine_stubs(monkeypatch, events)
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    (tmp_path / "recording.wav.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    # Frames: stub the extractor (the diarized HTML path inlines frames);
    # the real one would run ffmpeg on the fake video bytes.
    frames_dir = tmp_path / "recording.frames"
    frames_dir.mkdir()
    monkeypatch.setattr(
        cli, "_extract_and_manifest_screenshots",
        lambda in_path, merged, of_base, ffmpeg, width, dry_run: (frames_dir, tmp_path / "recording.frames.json", False),
    )

    rc = cli.cmd_transcribe(_transcribe_args(video, outputs="html", speakers=None))

    assert rc == 0
    assert events == ["install", "download"]  # the one-time setup ran
    html = (tmp_path / "recording.speakers.html").read_text(encoding="utf-8")
    assert "Speaker A" in html  # real labels, not the degraded generic 'Speaker'
    err = capsys.readouterr().err
    assert "skipping speaker labels" not in err  # the skip hint is obsolete now


def test_transcribe_auto_diarization_skips_after_failed_setup(tmp_path, monkeypatch, capsys):
    """Setup failed + merely auto-enabled diarization: the run still
    completes, and the hint says the setup is incomplete and offers
    --no-speakers — a hint, never a crash."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    monkeypatch.setattr(cli.M, "pick_best", lambda config: Path("/models/turbo.bin"))
    monkeypatch.setattr(cli, "_find_whisper_cli", lambda configured="": "whisper-cli")
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda configured="": "ffmpeg")
    monkeypatch.setattr(cli.aud, "extract_audio", lambda src, ffmpeg, dest_dir=None, dry_run=False: src.with_suffix(".wav"))
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    _stub_setup_unavailable(monkeypatch)
    ran = []
    monkeypatch.setattr(cli.D, "run_diarization", lambda *a, **k: ran.append(1) or [])

    rc = cli.cmd_transcribe(_transcribe_args(video, outputs="srt", speakers=None))

    assert rc == 0
    assert ran == []  # diarization never ran after the failed setup
    err = capsys.readouterr().err
    assert "setup incomplete" in err
    assert "--no-speakers" in err


def test_merge_auto_diarization_setup_success_writes_labeled_outputs(tmp_path, monkeypatch):
    """cmd_merge's half of the wiring: explicit --speakers triggers the
    setup, and labeled outputs land afterwards — the first `whiz merge`
    on a fresh machine just works."""
    events: list[str] = []
    _fresh_machine_stubs(monkeypatch, events)
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 0
    assert events == ["install", "download"]
    html = (tmp_path / "meeting.m4a.speakers.html").read_text(encoding="utf-8")
    assert "Speaker A" in html
    srt = (tmp_path / "meeting.m4a.speakers.srt").read_text(encoding="utf-8")
    assert "Speaker A:" in srt


def test_merge_zero_segments_message_is_actionable(tmp_path, monkeypatch, capsys):
    """Zero segments is an audio-content verdict, not a defect: the warning
    must say what to DO next, not just what happened."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 0
    # Wrap-insensitive: the hint lines are muted lines near the console
    # width, so any phrase may be split by rich's wrapping.
    err = " ".join(capsys.readouterr().err.split())
    assert "Diarization produced no segments" in err
    assert "--speakers N" in err           # the biggest accuracy lever first
    assert "--cluster-threshold" in err     # loosen the clustering
    assert "silence" in err                # silence is content, not a defect


def test_speakers_match_setup_failure_exits_with_hint(tmp_path, monkeypatch):
    """`whiz speakers match` needs diarization by definition: when the
    setup cannot make it work, exit loudly with the manual command —
    there is no degraded path to fall back to here."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_unavailable(monkeypatch)

    args = SimpleNamespace(file=str(audio), speakers=1, cluster_threshold=None,
                           no_auto_diarization_setup=False)
    with pytest.raises(SystemExit, match="pipx inject whiz 'whiz\\[diarize\\]'"):
        cli.cmd_speakers_match(args)


def test_speakers_match_runs_after_setup_success(tmp_path, monkeypatch, capsys):
    """Setup made diarization work → the match command proceeds (and
    reports honestly when there are no profiles yet)."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    monkeypatch.setattr(cli.P, "load_profiles", lambda: [])

    args = SimpleNamespace(file=str(audio), speakers=1, cluster_threshold=None,
                           no_auto_diarization_setup=False)
    rc = cli.cmd_speakers_match(args)

    assert rc == 0
    assert "No stored voice profiles" in capsys.readouterr().err


# ---------- auto-setup consent (review round 3, 2026-09-07) ----------
#
# The auto-setup writes into site-packages (unlike the VAD model's cache-file
# download), so an interactive terminal is ASKED first (y/N) and the answer
# persists in auto_diarization_setup. Non-interactive sessions auto-allow so
# a scripted fresh machine still just works; --no-auto-diarization-setup
# short-circuits before any prompt.


class _FakeTtyErr:
    """A stderr stand-in that claims to be a terminal.

    _auto_setup_consent prompts only when BOTH stdin and stderr are ttys,
    and capsys's stderr is always non-tty, so the TTY branch is unreachable
    in tests without faking the stream. Output is collected (not asserted on
    unless a message matters) so rich can render into it freely.
    """

    encoding = "utf-8"

    def __init__(self):
        self.buf: list[str] = []

    def isatty(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self.buf.append(s)
        return len(s)

    def flush(self) -> None:
        pass


def _pin_ttys(monkeypatch, *, stdin_tty: bool, stderr: _FakeTtyErr | None = None):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: stdin_tty))
    if stderr is not None:
        monkeypatch.setattr(sys, "stderr", stderr)


def _boom_input(prompt=""):
    raise AssertionError("the consent prompt must not run here")


def test_consent_answered_true_allows_without_prompting(monkeypatch):
    """auto_diarization_setup=true in config answers permanently — a TTY or
    not, no prompt, setup proceeds."""
    monkeypatch.setattr(builtins, "input", _boom_input)
    config = cli.cfg.Config()
    config.auto_diarization_setup = True
    assert cli._auto_setup_consent(config) is True


def test_consent_answered_false_declines_without_prompting(monkeypatch):
    """auto_diarization_setup=false answers permanently the other way."""
    monkeypatch.setattr(builtins, "input", _boom_input)
    config = cli.cfg.Config()
    config.auto_diarization_setup = False
    assert cli._auto_setup_consent(config) is False


def test_consent_tty_yes_persists_true(tmp_path, monkeypatch):
    """Interactive y: allow, and remember the answer so the question is
    once-ever (config object AND on-disk file)."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    config = cli.cfg.Config()

    assert cli._auto_setup_consent(config) is True
    assert config.auto_diarization_setup is True
    saved = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "auto_diarization_setup = true" in saved


def test_consent_tty_no_persists_false_with_way_back_hint(tmp_path, monkeypatch):
    """A decline must say how to change the answer later — a one-shot 'no'
    can't strand the user with no path back to auto-setup."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    err = _FakeTtyErr()
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=err)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    config = cli.cfg.Config()

    assert cli._auto_setup_consent(config) is False
    saved = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "auto_diarization_setup = false" in saved
    # Wrap-insensitive: rich wraps the long hint lines at console width.
    flat = " ".join("".join(err.buf).split())
    assert "pipx inject whiz 'whiz[diarize]'" in flat  # the manual path
    assert "whiz config set auto_diarization_setup=true" in flat  # the way back


def test_consent_tty_eof_declines_and_persists_false(tmp_path, monkeypatch):
    """Piped stdin under a tty stderr (whiz t rec.mov < /dev/null): EOF is a
    decline, not a crash — and it persists like any other answer."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())

    def _eof(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", _eof)
    config = cli.cfg.Config()

    assert cli._auto_setup_consent(config) is False
    saved = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "auto_diarization_setup = false" in saved


def test_consent_tty_ctrl_c_declines_without_persisting(tmp_path, monkeypatch):
    """^C at the prompt is a plain decline: the question stays UNANSWERED so
    a reflexive interrupt doesn't permanently disable auto-setup."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())

    def _interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", _interrupt)
    config = cli.cfg.Config()

    assert cli._auto_setup_consent(config) is False
    assert config.auto_diarization_setup is None
    assert not (tmp_path / "config.toml").exists()


def test_consent_non_tty_allows_without_persisting(tmp_path, monkeypatch):
    """Scripts/cron (piped stdin): proceed automatically — a prompt would
    hang a headless run — but persist NOTHING (nobody answered)."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    _pin_ttys(monkeypatch, stdin_tty=False)  # capsys stderr is non-tty too
    monkeypatch.setattr(builtins, "input", _boom_input)
    config = cli.cfg.Config()

    assert cli._auto_setup_consent(config) is True
    assert config.auto_diarization_setup is None
    assert not (tmp_path / "config.toml").exists()


def test_consent_persist_failure_does_not_crash(tmp_path, monkeypatch):
    """An unwritable config must not turn a user's y/N into a crash — the
    answer still governs this run."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(cli.cfg, "save", lambda _config: (_ for _ in ()).throw(OSError("disk full")))
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    config = cli.cfg.Config()

    assert cli._auto_setup_consent(config) is True


def test_ensure_diarization_ready_tty_consent_yes_runs_setup(tmp_path, monkeypatch):
    """End-to-end: fresh machine on a real terminal — the user answers y at
    the prompt and the FULL setup runs (install, download), with the answer
    remembered for every future run."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    events: list[str] = []
    monkeypatch.setattr(cli, "_diarization_available", lambda config: "download" in events)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    monkeypatch.setattr(cli, "_install_sherpa_onnx", lambda: events.append("install") or True)
    monkeypatch.setattr(cli.D, "download_diarization_models", lambda: events.append("download"))
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    config = cli.cfg.Config()

    assert cli._ensure_diarization_ready(config) is True
    assert events == ["install", "download"]
    saved = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "auto_diarization_setup = true" in saved


def test_ensure_diarization_ready_tty_consent_no_skips_setup(tmp_path, monkeypatch):
    """End-to-end: a decline at the prompt installs/downloads NOTHING and
    lands the caller on its existing degraded/skip path (False return)."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")

    def _boom(*_a, **_k):
        raise AssertionError("declined consent must not install or download")

    monkeypatch.setattr(cli, "_diarization_available", lambda config: False)
    monkeypatch.setattr(cli, "_install_sherpa_onnx", _boom)
    monkeypatch.setattr(cli.D, "download_diarization_models", _boom)
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    config = cli.cfg.Config()

    assert cli._ensure_diarization_ready(config) is False
    assert config.auto_diarization_setup is False  # declined and remembered
    saved = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "auto_diarization_setup = false" in saved


def test_config_set_auto_diarization_setup_roundtrip(tmp_path, monkeypatch):
    """`whiz config set auto_diarization_setup=false` must store a real bool:
    the tri-state 'bool | None' type string once missed _coerce's bool branch,
    which would persist the STRING 'false' — truthy on every later load."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    args = SimpleNamespace(assignment="auto_diarization_setup=false")

    assert cli.cmd_config_set(args) == 0
    assert cli.cfg.load().auto_diarization_setup is False

    args = SimpleNamespace(assignment="auto_diarization_setup=true")
    assert cli.cmd_config_set(args) == 0
    assert cli.cfg.load().auto_diarization_setup is True


def test_config_save_tri_state_none_semantics(tmp_path, monkeypatch):
    """Unset (None) is omitted from the file — an emitted `= None` would be
    invalid TOML that breaks the NEXT load for every command — and a fresh
    never-answered session must not clobber a previously persisted answer."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")

    cli.cfg.save(cli.cfg.Config())  # a session that never answered
    assert "auto_diarization_setup" not in (tmp_path / "config.toml").read_text(encoding="utf-8")

    config = cli.cfg.Config()
    config.auto_diarization_setup = True
    cli.cfg.save(config)
    assert "auto_diarization_setup = true" in (tmp_path / "config.toml").read_text(encoding="utf-8")

    cli.cfg.save(cli.cfg.Config())  # a later fresh session: None must not win
    assert cli.cfg.load().auto_diarization_setup is True


def test_config_show_renders_tri_state_unset_cleanly(tmp_path, monkeypatch, capsys):
    """`whiz config show` renders the unset tri-state as <unset>, not the
    Python None repr."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")

    assert cli.cmd_config_show(SimpleNamespace()) == 0
    assert "auto_diarization_setup = <unset>" in capsys.readouterr().out


# ---------- wave-1 audit fix batch (branch fix/wave1-cli, 2026-09-10) ----------
#
# Regression tests for the wave-1 silent-failure audit items: M1 (merge
# diarization gate), M2 (typed DiarizationUnavailable at both call sites),
# H4 (chained --analyze honors SystemExit), H5 (a degraded run never
# clobbers a NAMED frames manifest), M14 (degraded-output info without
# diarization), L-a (pip rc-0-but-not-importable), L-b (declined setup +
# cache hit), L-c (non-bool consent value).


def test_merge_no_speakers_gate_skips_diarization_entirely(tmp_path, monkeypatch):
    """M1: an audio merge with no --speakers never reaches run_diarization
    (or the one-time setup) — it goes straight to the JSON merge path and
    writes the generic-label fallback output."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())

    def _boom(*_a, **_k):
        raise AssertionError("diarization must not run when speakers were not requested")

    monkeypatch.setattr(cli.D, "run_diarization", _boom)
    monkeypatch.setattr(cli, "_ensure_diarization_ready", _boom)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=None))

    assert rc == 0
    html = (tmp_path / "meeting.m4a.speakers.html").read_text(encoding="utf-8")
    assert ">Speaker<" in html  # explicit html degraded to generic labels


def test_merge_degraded_info_fires_without_diarization(tmp_path, monkeypatch, capsys):
    """M14: a merge requesting degraded output (explicit html) without
    speakers must explain WHY the artifacts carry generic labels — the old
    explanation only existed inside the diarization branch."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=None))

    assert rc == 0
    flat = " ".join(capsys.readouterr().err.split())
    assert "diarization not requested" in flat


def test_diarize_or_fallback_catches_typed_validate_failure(tmp_path, monkeypatch, capsys):
    """M2 (transcribe site): the config-validation DiarizationUnavailable —
    the path the old RuntimeError string matcher never matched — degrades
    with a hint instead of crashing."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")

    def _raise_validate(wav, config, num_speakers=0, threshold=0.9):
        raise cli.D.DiarizationUnavailable(
            "sherpa-onnx diarization config validation failed; check model paths."
        )

    monkeypatch.setattr(cli.D, "run_diarization", _raise_validate)

    segs = cli._run_diarize_or_fallback(
        audio, cli.cfg.Config(), _transcribe_args(audio, speakers=1))

    assert segs == []
    flat = " ".join(capsys.readouterr().err.split())
    assert "diarization unavailable" in flat
    assert "config validation failed" in flat


def test_merge_validate_failure_raises_systemexit_with_hint(tmp_path, monkeypatch):
    """M2 (merge site): with nothing else requested, the typed catch turns
    the validate failure into the loud SystemExit + enable hint. (Under the
    old string matcher this escaped as a raw RuntimeError.)"""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_unavailable(monkeypatch)

    def _raise_validate(wav, config, num_speakers=0, threshold=0.9):
        raise cli.D.DiarizationUnavailable(
            "sherpa-onnx diarization config validation failed; check model paths."
        )

    monkeypatch.setattr(cli.D, "run_diarization", _raise_validate)

    with pytest.raises(SystemExit) as excinfo:
        cli.cmd_merge(_merge_args(audio, outputs="", speakers=1))
    assert "config validation failed" in str(excinfo.value)
    assert "pipx inject whiz 'whiz[diarize]'" in str(excinfo.value)


def test_transcribe_chained_analyze_failure_surfaces_message_and_rc(tmp_path, monkeypatch, capsys):
    """H4: a --analyze chain that raises SystemExit is not swallowed — the
    message is surfaced, the exit code honored, and the user told the
    transcription itself is fine."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=False)

    def _no_transcript(_args):
        raise SystemExit(
            "No transcript found for recording.mov (need a frames manifest or .speakers.txt)."
        )

    monkeypatch.setattr(cli, "cmd_analyze", _no_transcript)
    args = _transcribe_args(audio, outputs="srt", speakers=None)
    args.analyze = True

    rc = cli.cmd_transcribe(args)

    assert rc == 1
    flat = " ".join(capsys.readouterr().err.split())
    assert "Chained analysis failed" in flat
    assert "No transcript found" in flat
    assert "transcription itself succeeded" in flat


def test_transcribe_chained_analyze_honors_int_exit_code(tmp_path, monkeypatch, capsys):
    """H4 (bare code): SystemExit(3) from the chained analysis propagates
    as rc 3 — no fabricated 'Chained analysis failed: 3' message line."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=False)

    def _boom(_args):
        raise SystemExit(3)

    monkeypatch.setattr(cli, "cmd_analyze", _boom)
    args = _transcribe_args(audio, outputs="srt", speakers=None)
    args.analyze = True

    rc = cli.cmd_transcribe(args)

    assert rc == 3
    flat = " ".join(capsys.readouterr().err.split())
    assert "Chained analysis failed" not in flat


def test_consent_non_bool_string_is_not_authoritative(tmp_path, monkeypatch):
    """L-c: a hand-edited `auto_diarization_setup = "false"` string is not a
    stored answer (it is truthy and used to read as consent) — it falls
    through to the interactive prompt, where the user's 'n' declines."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    config = cli.cfg.Config()
    config.auto_diarization_setup = "false"

    assert cli._auto_setup_consent(config) is False
    saved = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "auto_diarization_setup = false" in saved


def test_install_sherpa_rc0_but_not_importable_warns(monkeypatch, capsys):
    """L-a: pip exits 0 but sherpa_onnx still does not resolve (partial
    install, wrong venv) — a loud warning with the manual path, not a silent
    False that reads as a fresh-install loop on the next run."""
    monkeypatch.setattr(
        cli.subprocess, "run",
        lambda cmd, check=False: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    assert cli._install_sherpa_onnx() is False
    flat = " ".join(capsys.readouterr().err.split())
    assert "still not importable" in flat
    assert "pipx inject whiz 'whiz[diarize]'" in flat


def test_merge_declined_setup_then_success_explains_cache_reuse(tmp_path, monkeypatch, capsys):
    """L-b: setup failed/declined yet diarization succeeded — only possible
    via the diarization cache; the run says so instead of leaving the
    surprise unexplained."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 0
    flat = " ".join(capsys.readouterr().err.split())
    assert "ran WITHOUT the one-time setup" in flat
    assert ".diar.json" in flat


def test_manifest_is_named_detection(tmp_path):
    """H5 unit: a NAMED manifest reads as named (keep), an all-generic one
    as degraded (refresh), a missing one as writable."""
    frames_dir = tmp_path / "rec.frames"
    frames_dir.mkdir()
    manifest = tmp_path / "rec.frames.json"
    named = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Alice",
                               text="hi", frame="seg0001.jpg")]
    cli.SC.write_manifest(named, frames_dir, manifest)
    assert cli._manifest_is_named(manifest) is True

    degraded = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Speaker",
                                  text="hi", frame="seg0001.jpg")]
    cli.SC.write_manifest(degraded, frames_dir, manifest)
    assert cli._manifest_is_named(manifest) is False

    assert cli._manifest_is_named(tmp_path / "missing.frames.json") is False


def test_extract_manifests_degraded_run_never_clobbers_named_manifest(tmp_path, monkeypatch, capsys):
    """H5 unit: an incoming all-generic shot list KEEPS a NAMED manifest
    (with a warning) and never runs the frame extraction."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    frames_dir = tmp_path / "recording.frames"
    frames_dir.mkdir()
    manifest = tmp_path / "recording.frames.json"
    named = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Alice",
                               text="hi", frame="seg0001.jpg")]
    cli.SC.write_manifest(named, frames_dir, manifest)

    def _boom(*_a, **_k):
        raise AssertionError("extraction must not run when the named manifest is kept")

    monkeypatch.setattr(cli.SC, "extract_segment_frames", _boom)
    merged = [(cli.MR.WhisperSeg(start=0.0, end=2.0, text="hi"), "Speaker")]

    result = cli._extract_and_manifest_screenshots(
        video, merged, tmp_path / "recording", ffmpeg="ffmpeg", width=1280,
    )

    assert result == (frames_dir, manifest, True)
    loaded = cli.SC.load_manifest(manifest)
    assert loaded[0].speaker == "Alice"  # untouched
    flat = " ".join(capsys.readouterr().err.split())
    assert "already exists with named speakers" in flat


def test_extract_manifests_refreshes_existing_degraded_manifest(tmp_path, monkeypatch):
    """H5 flip side: an existing DEGRADED manifest has no speaker names to
    protect — the degraded re-run refreshes it (idempotence), so re-runs
    with a different --model/--language can update the transcript."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    frames_dir = tmp_path / "recording.frames"
    frames_dir.mkdir()
    manifest = tmp_path / "recording.frames.json"
    degraded = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Speaker",
                                  text="stale", frame="seg0001.jpg")]
    cli.SC.write_manifest(degraded, frames_dir, manifest)
    fresh = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Speaker",
                               text="fresh", frame="seg0001.jpg")]
    monkeypatch.setattr(
        cli.SC, "extract_segment_frames",
        lambda video, merged, out_dir, ffmpeg="ffmpeg", width=1280, dry_run=False: fresh,
    )
    merged = [(cli.MR.WhisperSeg(start=0.0, end=2.0, text="fresh"), "Speaker")]

    result = cli._extract_and_manifest_screenshots(
        video, merged, tmp_path / "recording", ffmpeg="ffmpeg", width=1280,
    )

    assert result == (frames_dir, manifest, False)
    loaded = cli.SC.load_manifest(manifest)
    assert loaded[0].text == "fresh"  # refreshed, not kept


def test_transcribe_degraded_run_keeps_named_frames_manifest_rc0(tmp_path, monkeypatch, capsys):
    """H5 end-to-end: a degraded video re-run (explicit --speakers, sherpa
    missing) KEEPS the named manifest an earlier diarized run wrote — and a
    kept-only run is a no-op success (rc 0), not a failure."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    (tmp_path / "recording.wav.json").write_text(_WHISPER_JSON, encoding="utf-8")
    wav = tmp_path / "recording.wav"
    frames_dir = tmp_path / "recording.frames"
    frames_dir.mkdir()
    manifest = tmp_path / "recording.frames.json"
    named = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Alice",
                               text="hello world", frame="seg0001.jpg")]
    cli.SC.write_manifest(named, frames_dir, manifest)

    def fake_build(args, config):
        return (["whisper-cli"], "model.bin", wav, video, False,
                tmp_path / "recording", True, True)

    monkeypatch.setattr(cli, "_build_transcribe_args", fake_build)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())

    def _boom(*_a, **_k):
        raise AssertionError("extraction must not run when the named manifest is kept")

    monkeypatch.setattr(cli.SC, "extract_segment_frames", _boom)

    rc = cli.cmd_transcribe(_transcribe_args(video, outputs="", speakers=1))

    assert rc == 0  # kept-only run: no-op success
    loaded = cli.SC.load_manifest(manifest)
    assert loaded[0].speaker == "Alice"
    flat = " ".join(capsys.readouterr().err.split())
    assert "already exists with named speakers" in flat


# ---------- wave-1 M3 adoption: profiles API at the CLI call sites ----------
#
# py-support-2 (fbd04d5) gave profiles.py an auto-match provenance API:
# cosine_similarity returns None on dim mismatch, and save_profile(auto_match)
# creates-but-never-merges. These tests pin the CLI's ADOPTION of it — the
# speakers-match score table under a dim-mismatched stored profile, and the
# auto-vs-confirmed provenance threaded through _write_labeled_outputs →
# _save_named_profiles.


def test_speakers_match_dim_mismatch_renders_na_not_crash(tmp_path, monkeypatch, capsys):
    """`whiz speakers match` against a stored profile saved with a DIFFERENT
    embedding dim (embedding model swapped): cosine_similarity returns None
    and the score table must render an honest n/a — the old code crashed
    sorting None among floats, and read scores[0][0] on what could be an
    empty list."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    cli.P.save_profile("OldDim", [1.0, 2.0, 3.0, 4.0], samples=2)  # dim 4
    monkeypatch.setattr(
        cli.P, "compute_speaker_embeddings",
        lambda wav, segments, config: {0: [1.0, 1.0]},  # dim 2
    )

    args = SimpleNamespace(file=str(audio), speakers=1, cluster_threshold=None,
                           no_auto_diarization_setup=False)
    rc = cli.cmd_speakers_match(args)

    assert rc == 0
    err = capsys.readouterr().err
    assert "OldDim=n/a" in err        # incomparable pair rendered, not crashed
    assert "n/a" in err               # best score is n/a too — no scores existed
    assert "dimension mismatch" in err  # match_speakers' skip warning fired


def test_merge_auto_match_never_merges_existing_profile(tmp_path, monkeypatch, capsys):
    """M3 cli adoption: a name that arrived via VOICE-PROFILE auto-match
    never merges into an existing profile — save_profile's guard keeps the
    stored centroid byte-identical and the run says so. (Before the adoption
    every named speaker was merged, so a self-confirming auto-match chain
    silently drifted the stored centroid.)"""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    cli.P.save_profile("Alice", [0.5, 0.5], samples=4)  # user-confirmed base
    before = (tmp_path / "Alice.json").read_text(encoding="utf-8")
    # The run's cluster embedding auto-matches Alice at threshold 0.8.
    monkeypatch.setattr(
        cli.P, "compute_speaker_embeddings",
        lambda wav, segments, config: {0: [0.5, 0.5]},
    )

    args = _merge_args(audio, outputs="html", speakers=1)
    args.no_voice_profiles = False  # enable the feature under test
    rc = cli.cmd_merge(args)

    assert rc == 0
    # The stored centroid is untouched — no merge, no replace.
    assert (tmp_path / "Alice.json").read_text(encoding="utf-8") == before
    flat = " ".join(capsys.readouterr().err.split())
    assert "auto-match not merged" in flat
    assert "Merged voice profile" not in flat  # no merge that didn't happen


def test_merge_confirmed_name_merges_existing_profile(tmp_path, monkeypatch, capsys):
    """M3 flip side: --speakers-names is a HUMAN confirmation — it overrides
    any auto-match for the same label and merges into the stored profile
    normally (auto_match=False), upgrading provenance to 'user'."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    cli.P.save_profile("Alice", [0.0, 0.0], samples=2)
    monkeypatch.setattr(
        cli.P, "compute_speaker_embeddings",
        lambda wav, segments, config: {0: [2.0, 2.0]},
    )

    args = _merge_args(audio, outputs="html", speakers=1, speakers_names=["Alice"])
    args.no_voice_profiles = False
    rc = cli.cmd_merge(args)

    assert rc == 0
    data = json.loads((tmp_path / "Alice.json").read_text(encoding="utf-8"))
    assert data["samples"] == 3                       # merged: 2 stored + 1 new
    # (0*2 + 2*1)/3 = 2/3 — plain abs checks, matching test_profiles.py's
    # idiom (pytest.approx routes through np.isscalar, gone in numpy 2.x).
    assert all(abs(v - 2/3) < 1e-9 for v in data["embedding"])
    assert data["source"] == "user"
    flat = " ".join(capsys.readouterr().err.split())
    assert "Merged voice profile: Alice" in flat


def test_write_labeled_outputs_prompt_confirmation_upgrades_auto_label(tmp_path, monkeypatch, capsys):
    """M3 provenance threading: interactive confirmation — even just pressing
    Enter on the auto-matched suggestion — upgrades the label from auto to
    user-confirmed, so the save MERGES instead of no-clobbering."""
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    merged = [(cli.MR.WhisperSeg(start=0.0, end=2.0, text="hi"), "Speaker A")]
    # The interactive prompt returns the auto-suggested name (Enter accepted).
    monkeypatch.setattr(
        cli, "_prompt_speaker_names",
        lambda merged, default_names=None: {"Speaker A": "Alice"},
    )
    cli.P.save_profile("Alice", [0.0, 0.0], samples=2, auto_match=True)  # source 'auto'

    _srt, _txt, _html, name_map = cli._write_labeled_outputs(
        merged, tmp_path / "rec", name_speakers=True,
        profile_names={"Speaker A": "Alice"},
        cluster_embeddings={0: [2.0, 2.0]},
        save_profiles=True,
    )

    assert name_map == {"Speaker A": "Alice"}
    data = json.loads((tmp_path / "Alice.json").read_text(encoding="utf-8"))
    assert data["samples"] == 3   # merged, not skipped — Enter is confirmation
    assert data["source"] == "user"  # provenance upgraded from 'auto'
