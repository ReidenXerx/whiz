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
        lambda in_path, merged, of_base, ffmpeg, width, dry_run: (frames_dir, manifest),
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


def _merge_args(file, outputs="html", speakers=1, speakers_names=None):
    return SimpleNamespace(
        file=str(file), json="", outputs=outputs, speakers=speakers,
        no_speakers=False, cluster_threshold=None, name_speakers=False,
        no_name_speakers=True, speakers_names=speakers_names, screenshots=False,
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
    assert "--speakers-names had no effect" in capsys.readouterr().err


def test_merge_fallback_warns_discarded_speakers_names(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1,
                                   speakers_names=["Alice,Bob"]))

    assert rc == 0
    assert "--speakers-names had no effect" in capsys.readouterr().err


def test_merge_zero_segments_falls_back_to_unlabeled_html(tmp_path, monkeypatch, capsys):
    """Diarization runs but finds no speech: warn + generic-label HTML
    (and .speakers.txt on audio runs) instead of crashing or skipping."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
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
    """No html/screenshots requested + no segments -> nothing written; a
    silent rc=0 would read as success."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
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
    (both artifacts were kept), so merge exits 1 per the approved
    nothing-written-is-not-success semantics — with warnings explaining
    the keeps."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config())
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    named_txt = tmp_path / "meeting.m4a.speakers.txt"
    named_html = tmp_path / "meeting.m4a.speakers.html"
    named_txt.write_text("Vadim (00:00:00): real named content\n", encoding="utf-8")
    named_html.write_text("\u003chtml\u003enamed run\u003c/html\u003e", encoding="utf-8")

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 1  # this run wrote nothing: the named outputs were kept
    assert named_txt.read_text(encoding="utf-8") == "Vadim (00:00:00): real named content\n"
    assert named_html.read_text(encoding="utf-8") == "\u003chtml\u003enamed run\u003c/html\u003e"
    err = capsys.readouterr().err
    assert "kept" in err and "meeting.m4a.speakers.txt" in err
    assert "kept" in err and "meeting.m4a.speakers.html" in err


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

    assert rc == 0
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

    assert rc == 0
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
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 0
    err = capsys.readouterr().err
    assert err.count("diarization unavailable") == 1
    assert err.count("Diarization produced no segments") == 0
