"""Per-stretch language detection and the repetition-loop guard."""

from __future__ import annotations

import contextlib
import json
import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from whiz import cli
from whiz import config as cfg
from whiz import languages as LG


# ---------- helpers ----------

def _wav(path: Path, seconds: float, loud: list[tuple[float, float]] = ()) -> Path:
    """16 kHz mono PCM: silence, with a 440 Hz tone over each (start, end)."""
    n = int(seconds * LG.RATE)
    samples = [0] * n
    for a, b in loud:
        for i in range(int(a * LG.RATE), min(n, int(b * LG.RATE))):
            samples[i] = int(8000 * math.sin(2 * math.pi * 440 * i / LG.RATE))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(LG.RATE)
        w.writeframes(struct.pack(f"<{n}h", *samples))
    return path


def _seg(f_ms: int, t_ms: int, text: str) -> dict:
    return {
        "timestamps": {"from": LG.timestamp(f_ms), "to": LG.timestamp(t_ms)},
        "offsets": {"from": f_ms, "to": t_ms},
        "text": text,
    }


def _doc(segments: list[dict], language: str = "uk") -> dict:
    return {
        "systeminfo": "test", "model": {"type": "large"},
        "params": {"model": "m.bin", "language": "auto", "translate": False},
        "result": {"language": language}, "transcription": segments,
    }


class FakeWhisper:
    """Stands in for whisper-cli: for each -f clip writes <clip>.json from
    ``reply(language, clip, argv)`` and reports 0..100% per clip."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], progress) -> int:
        self.calls.append(argv)
        lang = argv[argv.index("-l") + 1]
        clips = [argv[i + 1] for i, a in enumerate(argv) if a == "-f"]
        for clip in clips:
            for p in (0, 50, 100):
                progress(p)
            Path(clip + ".json").write_text(json.dumps(_doc(self.reply(lang, clip, argv), lang)))
        return 0


# ---------- planning ----------

def test_parse_detections_maps_each_clip():
    clips = [Path("/t/a.wav"), Path("/t/b.wav"), Path("/t/c.wav")]
    lines = [
        "whisper_model_load: n_langs = 100\n",
        "main: processing '/t/a.wav' (480000 samples, 30.0 sec), 4 threads ...\n",
        "whisper_full_with_state: auto-detected language: uk (p = 0.996432)\n",
        "main: processing '/t/b.wav' (480000 samples, 30.0 sec), 4 threads ...\n",
        "main: processing '/t/c.wav' (480000 samples, 30.0 sec), 4 threads ...\n",
        "whisper_full_with_state: auto-detected language: en (p = 0.951769)\n",
    ]
    assert LG.parse_detections(lines, clips) == [("uk", 0.996432), None, ("en", 0.951769)]


def _windows(*specs):
    """(duration, language, confidence) → consecutive Windows."""
    out, t = [], 0.0
    for dur, lang, p in specs:
        out.append(LG.Window(t, t + dur, lang, p))
        t += dur
    return out


def test_smooth_unsure_windows_inherit_their_neighbours():
    ws = _windows((30, "uk", 0.99), (30, "en", 0.4), (30, None, 0.0), (30, "en", 0.98), (30, "en", 0.99))
    assert LG.smooth(ws) == ["uk", "uk", "uk", "en", "en"]


def test_smooth_drops_a_language_with_too_little_speech():
    # 5 s of confident "en" inside Ukrainian: a stray English term.
    ws = _windows((30, "uk", 0.99), (5, "en", 0.95), (30, "uk", 0.99))
    assert LG.smooth(ws) == ["uk", "uk", "uk"]


def test_smooth_absorbs_a_short_island_but_keeps_a_long_one():
    island = _windows((30, "uk", 0.99), (20, "en", 0.95), (4, "de", 0.95), (30, "uk", 0.99), (30, "de", 0.99))
    # "de" passes the 15 s bar overall, but its 4 s run sits between en and uk: kept.
    assert LG.smooth(island) == ["uk", "en", "de", "uk", "de"]
    short = _windows((30, "uk", 0.99), (8, "en", 0.95), (30, "uk", 0.99), (30, "en", 0.99))
    assert LG.smooth(short) == ["uk", "uk", "uk", "en"]


def test_smooth_with_nothing_confident_is_none():
    assert LG.smooth(_windows((30, None, 0.0), (30, "en", 0.3))) is None


def test_windows_carry_how_much_speech_they_hold():
    diar = [
        SimpleNamespace(start=0.0, end=10.0, speaker=0),
        SimpleNamespace(start=25.0, end=40.0, speaker=1),
    ]
    ws = LG.windows_for(90.0, diar)
    assert [(w.start, w.end, w.speech) for w in ws] == [(0.0, 30.0, 15.0), (30.0, 60.0, 10.0), (60.0, 90.0, 0.0)]
    assert all(w.speech is None for w in LG.windows_for(90.0, None))


def test_fixed_windows_tile_the_recording():
    ws = LG.windows_for(95.0, None, quietest=lambda around, radius: around + 1)
    assert ws[0].start == 0.0 and ws[-1].end == 95.0
    assert all(a.end == b.start for a, b in zip(ws, ws[1:]))
    assert [w.end for w in ws[:-1]] == [31.0, 61.0, 91.0]


def test_spans_cover_everything_and_cut_between_windows():
    ws = [LG.Window(2, 30), LG.Window(32, 60), LG.Window(64, 90)]
    spans = LG.spans_for(ws, ["uk", "en", "en"], 100.0)
    assert [(s.start, s.end, s.language) for s in spans] == [(0.0, 31.0, "uk"), (31.0, 100.0, "en")]


def test_workspace_clips_and_finds_the_quiet_moment(tmp_path):
    wav = _wav(tmp_path / "a.wav", 10.0, loud=[(0, 4.0), (5.0, 10.0)])
    with contextlib.ExitStack() as stack:
        work = LG.Workspace.create(wav, lambda: "ffmpeg", stack)
        assert work.pcm == wav  # already 16 kHz mono: used as-is
        assert work.duration == 10.0
        assert 4.0 <= work.quietest(5.0, 3.0) <= 5.0
        clip = work.clip(2.0, 3.5)
        with wave.open(str(clip)) as w:
            assert w.getnframes() == int(1.5 * LG.RATE)


# ---------- loops ----------

def test_find_loops_thresholds():
    long = "МІЦП має бути такі,"
    texts = ["Hi there.", long, long, " " + long.upper(), "Так.", "Так.", "Так.", "ok"]
    segs = [_seg(i * 1000, i * 1000 + 900, t) for i, t in enumerate(texts)]
    # The 4-word line three times is a loop; "Так." three times is speech.
    assert LG.find_loops(segs) == [(1, 3)]
    six = [_seg(i * 1000, i * 1000 + 900, "Дякую.") for i in range(6)]
    assert LG.find_loops(six) == [(0, 5)]


def test_repair_replaces_a_loop_with_a_fresh_decode(tmp_path):
    wav = _wav(tmp_path / "a.wav", 20.0, loud=[(0, 20)])
    loop = "the same words again"
    doc = _doc([_seg(0, 2000, "Start.")] + [_seg(2000 + i * 2000, 4000 + i * 2000, loop) for i in range(5)]
               + [_seg(12000, 14000, "End.")])
    fake = FakeWhisper(lambda lang, clip, argv: [_seg(0, 3000, "First real line."), _seg(3000, 10000, "Second.")])
    with contextlib.ExitStack() as stack:
        work = LG.Workspace.create(wav, lambda: "ffmpeg", stack)
        reports = LG.repair_loops(doc, work, ["whisper-cli", "-m", "m"], lambda t: "en", "-oj", fake, lambda p: None)
    assert [(r.start, r.end, r.lines, r.repaired) for r in reports] == [(2.0, 12.0, 5, True)]
    assert [s["text"] for s in doc["transcription"]] == ["Start.", "First real line.", "Second.", "End."]
    # The fresh lines sit on the original timeline, decoded with no carried context.
    assert doc["transcription"][1]["offsets"] == {"from": 2000, "to": 5000}
    assert doc["transcription"][2]["timestamps"]["to"] == "00:00:12,000"
    assert fake.calls[0][-4:-2] == ["-mc", "0"] and "en" in fake.calls[0]


def test_repair_collapses_a_loop_that_survives(tmp_path):
    wav = _wav(tmp_path / "a.wav", 20.0)
    loop = "thanks for watching everyone"
    doc = _doc([_seg(i * 2000, i * 2000 + 2000, loop) for i in range(4)])
    fake = FakeWhisper(lambda lang, clip, argv: [_seg(i * 1000, i * 1000 + 1000, loop) for i in range(3)])
    with contextlib.ExitStack() as stack:
        work = LG.Workspace.create(wav, lambda: "ffmpeg", stack)
        reports = LG.repair_loops(doc, work, ["whisper-cli"], lambda t: "en", "-oj", fake, lambda p: None)
    assert [r.repaired for r in reports] == [False]
    assert len(doc["transcription"]) == 1
    assert doc["transcription"][0]["text"] == loop


# ---------- stitching and writing ----------

def test_transcribe_spans_decodes_each_language_once_and_stitches(tmp_path):
    wav = _wav(tmp_path / "a.wav", 30.0)
    spans = [LG.Span(0, 10, "uk"), LG.Span(10, 26, "en"), LG.Span(26, 30, "uk")]
    fake = FakeWhisper(lambda lang, clip, argv: [_seg(500, 1500, f"{lang} line")])
    shown = []
    with contextlib.ExitStack() as stack:
        work = LG.Workspace.create(wav, lambda: "ffmpeg", stack)
        doc = LG.transcribe_spans(work, spans, ["whisper-cli", "-m", "m.bin", "-pp"], "-oj", fake, shown.append)
    assert [c[c.index("-l") + 1] for c in fake.calls] == ["uk", "en"]  # one process per language
    assert sum(a == "-f" for a in fake.calls[0]) == 2
    assert [(s["offsets"]["from"], s["text"]) for s in doc["transcription"]] == [
        (500, "uk line"), (10500, "en line"), (26500, "uk line"),
    ]
    assert doc["result"]["language"] == "en"  # 16 s of en beats 14 s of uk
    assert doc["languages"][1] == {"from": "00:00:10,000", "to": "00:00:26,000", "language": "en"}
    assert shown == sorted(shown) and shown[-1] == pytest.approx(100.0)


def test_write_outputs_matches_whisper_layouts(tmp_path):
    doc = _doc([_seg(0, 1500, " Hello, \"you\"."), _seg(61_230, 62_000, " Bye.")])
    base = str(tmp_path / "rec.wav")
    LG.write_outputs(doc, base, ["srt", "txt", "vtt", "csv", "lrc", "json"])
    assert Path(base + ".srt").read_text() == (
        "1\n00:00:00,000 --> 00:00:01,500\n Hello, \"you\".\n\n"
        "2\n00:01:01,230 --> 00:01:02,000\n Bye.\n\n"
    )
    assert Path(base + ".txt").read_text() == "Hello, \"you\".\nBye.\n"
    assert Path(base + ".vtt").read_text().startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n")
    assert Path(base + ".csv").read_text().splitlines()[1] == '0,1500," Hello, ""you""."'
    assert Path(base + ".lrc").read_text().splitlines()[2] == "[01:01.23] Bye."
    assert json.loads(Path(base + ".json").read_text())["transcription"][1]["text"] == " Bye."


# ---------- the transcribe step ----------

def _cmd(wav: Path, *formats: str) -> list[str]:
    cmd = ["whisper-cli", "-m", "m.bin", "-f", str(wav), "-t", "4", "-l", "auto"]
    cmd += [cli.OUTPUT_FLAGS[f] for f in formats]
    return cmd + ["-pp", "-np"]


@pytest.mark.language_detection
def test_transcribe_mixed_recording_writes_one_stitched_transcript(tmp_path, monkeypatch):
    wav = _wav(tmp_path / "rec.wav", 30.0)
    monkeypatch.setattr(cli, "_language_plan",
                        lambda workspace, cmd, diar: [LG.Span(0, 12, "uk"), LG.Span(12, 30, "en")])
    fake = FakeWhisper(lambda lang, clip, argv: [_seg(0, 2000, f"{lang} speech")])
    monkeypatch.setattr(cli, "_run_whisper_streaming",
                        lambda argv, progress=None: SimpleNamespace(returncode=fake(argv, progress or (lambda p: None))))
    with contextlib.ExitStack() as stack:
        rc = cli._transcribe(_cmd(wav, "srt", "json"), cfg.Config(), None, stack)
    assert rc == 0
    doc = json.loads(Path(str(wav) + ".json").read_text())
    assert [(s["offsets"]["from"], s["text"]) for s in doc["transcription"]] == [(0, "uk speech"), (12000, "en speech")]
    assert "en speech" in Path(str(wav) + ".srt").read_text()
    # The per-language runs never write next to the recording themselves.
    assert all("-of" not in c and "-osrt" not in c for c in fake.calls)


@pytest.mark.language_detection
def test_transcribe_one_language_runs_whisper_once_with_it(tmp_path, monkeypatch):
    wav = _wav(tmp_path / "rec.wav", 30.0)
    monkeypatch.setattr(cli, "_language_plan", lambda workspace, cmd, diar: [LG.Span(0, 30, "en")])
    seen = []

    def whisper(argv, progress=None):
        seen.append(argv)
        Path(str(wav) + ".json").write_text(json.dumps(_doc([_seg(0, 1000, "hello")], "en")))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli, "_run_whisper_streaming", whisper)
    with contextlib.ExitStack() as stack:
        assert cli._transcribe(_cmd(wav, "srt", "json"), cfg.Config(), None, stack) == 0
    assert len(seen) == 1
    assert seen[0][seen[0].index("-l") + 1] == "en" and seen[0].count("-l") == 1
    assert "-osrt" in seen[0] and "-f" in seen[0]


@pytest.mark.language_detection
def test_transcribe_falls_back_when_detection_fails(tmp_path, monkeypatch, capsys):
    wav = _wav(tmp_path / "rec.wav", 5.0)

    def broken(workspace, cmd, diar):
        raise RuntimeError("no model")

    monkeypatch.setattr(cli, "_language_plan", broken)
    seen = []
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda argv, progress=None: seen.append(argv) or SimpleNamespace(returncode=0))
    with contextlib.ExitStack() as stack:
        assert cli._transcribe(_cmd(wav, "srt"), cfg.Config(), None, stack) == 0
    assert seen[0][seen[0].index("-l") + 1] == "auto"
    out = capsys.readouterr()
    assert "language detection skipped: no model" in out.out + out.err


def test_transcribe_repairs_a_loop_in_a_single_language_run(tmp_path, monkeypatch):
    wav = _wav(tmp_path / "rec.wav", 20.0)
    loop = "one phrase over and over"

    def whisper(argv, progress=None):
        if "-mc" in argv:  # the repair pass
            for i, a in enumerate(argv):
                if a == "-f":
                    Path(argv[i + 1] + ".json").write_text(json.dumps(_doc([_seg(0, 4000, "What was said.")])))
        else:
            Path(str(wav) + ".json").write_text(json.dumps(_doc(
                [_seg(0, 1000, "Hi.")] + [_seg(1000 + i * 1000, 2000 + i * 1000, loop) for i in range(4)])))
            Path(str(wav) + ".srt").write_text("whisper's own srt")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli, "_run_whisper_streaming", whisper)
    cmd = _cmd(wav, "srt", "json")
    cmd[cmd.index("-l") + 1] = "uk"
    with contextlib.ExitStack() as stack:
        assert cli._transcribe(cmd, cfg.Config(), None, stack) == 0
    doc = json.loads(Path(str(wav) + ".json").read_text())
    assert [s["text"] for s in doc["transcription"]] == ["Hi.", "What was said."]
    assert "What was said." in Path(str(wav) + ".srt").read_text()


def test_file_progress_spans_files_and_never_goes_back():
    shown = []
    p = LG._FileProgress([100, 300], before=0, total=400, emit=shown.append)
    for v in (0, 50, 100, 0, 50, 100):
        p(v)
    assert shown == [0.0, 12.5, 25.0, 62.5, 100.0]


def test_refine_moves_the_cut_to_where_the_pieces_switch():
    # Coarse: 0-30 uk, 30-60 en. Really the switch is at ~50 s.
    windows = [LG.Window(0, 30), LG.Window(30, 60)]
    truth = lambda w: ("uk", 0.99) if w.end <= 50 else ("en", 0.98)
    asked = []

    def detect(pieces):
        asked.append(len(pieces))
        for p in pieces:
            p.language, p.confidence = truth(p)

    cuts = LG.refine(windows, ["uk", "en"], detect)
    assert asked == [6]  # both windows, in 10 s pieces, one call
    assert cuts == {1: (50.0, 50.0)}
    spans = LG.spans_for(windows, ["uk", "en"], 60.0, cuts=cuts)
    assert [(s.start, s.end, s.language) for s in spans] == [(0.0, 50.0, "uk"), (50.0, 60.0, "en")]


def test_refine_ignores_pieces_heard_as_a_third_language():
    windows = [LG.Window(0, 30), LG.Window(30, 60)]
    heard = iter([("uk", 0.99), ("uk", 0.99), ("ru", 0.83), ("en", 0.97), ("en", 0.99), ("en", 0.99)])

    def detect(pieces):
        for p in pieces:
            p.language, p.confidence = next(heard)

    # The "ru" piece counts for neither side: the cut may fall anywhere in it.
    assert LG.refine(windows, ["uk", "en"], detect) == {1: (20.0, 30.0)}


def test_transcribe_spans_drops_a_line_invented_over_the_end_padding(tmp_path):
    wav = _wav(tmp_path / "a.wav", 20.0)
    spans = [LG.Span(0, 10, "uk"), LG.Span(10, 20, "en")]

    def reply(lang, clip, argv):
        if lang == "uk":  # a real line, then one starting at 9.3 s and "lasting" to 39.3 s
            return [_seg(0, 9000, "Справжня фраза."), _seg(9300, 39300, "Субтитрувальниця Оля Шор")]
        return [_seg(0, 9500, "Real words, ending close to the cut.")]

    with contextlib.ExitStack() as stack:
        work = LG.Workspace.create(wav, lambda: "ffmpeg", stack)
        doc = LG.transcribe_spans(work, spans, ["whisper-cli"], "-oj", FakeWhisper(reply), lambda p: None)
    assert [s["text"] for s in doc["transcription"]] == ["Справжня фраза.", "Real words, ending close to the cut."]
