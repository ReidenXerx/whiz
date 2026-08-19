"""Tests for whiz.ocr — engine detection, install argv, normalization, batching.

No OCR engine is required: the engine implementations are monkeypatched, so
these run identically on a machine with no ocrmac/rapidocr/tesseract installed.

Run with: pytest tests/test_ocr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from whiz import config as cfg
from whiz import ocr as OCR


# ---------- engine order / resolution ----------

def test_engine_order_prefers_apple_on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert OCR.engine_order()[0] == "apple"
    assert OCR.preferred_engine() == "apple"


def test_engine_order_falls_back_to_rapidocr_elsewhere(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    order = OCR.engine_order()
    assert order[0] == "rapidocr"
    assert "apple" not in order


def test_available_engines_covers_every_known_engine():
    names = {i.name for i in OCR.available_engines()}
    assert names == set(OCR.ENGINES)


def test_detect_apple_unavailable_off_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    info = OCR.detect("apple")
    assert info.available is False
    assert "macOS" in info.detail
    # Nothing to offer: installing ocrmac on Linux wouldn't help.
    assert info.install_argv == []


def test_detect_unknown_engine():
    info = OCR.detect("nope")
    assert info.available is False
    assert "unknown" in info.detail


def test_resolve_engine_explicit_returned_even_when_missing(monkeypatch):
    """An explicit choice is honored so the caller can offer to install it."""
    monkeypatch.setattr(OCR, "detect", lambda n: OCR.EngineInfo(n, False, "nope"))
    config = cfg.Config()
    assert OCR.resolve_engine(config, "tesseract") == "tesseract"


def test_resolve_engine_auto_picks_first_available(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        OCR, "detect",
        lambda n: OCR.EngineInfo(n, n == "rapidocr", ""),
    )
    assert OCR.resolve_engine(cfg.Config(), "") == "rapidocr"


def test_resolve_engine_auto_returns_empty_when_none_available(monkeypatch):
    monkeypatch.setattr(OCR, "detect", lambda n: OCR.EngineInfo(n, False, ""))
    assert OCR.resolve_engine(cfg.Config(), "") == ""


def test_resolve_engine_uses_config_value(monkeypatch):
    monkeypatch.setattr(OCR, "detect", lambda n: OCR.EngineInfo(n, False, ""))
    config = cfg.Config(ocr_engine="tesseract")
    assert OCR.resolve_engine(config) == "tesseract"


def test_resolve_engine_rejects_unknown_name():
    with pytest.raises(RuntimeError, match="Unknown OCR engine"):
        OCR.resolve_engine(cfg.Config(), "banana")


# ---------- install argv ----------

def test_install_argv_uses_pipx_inject_when_pipx_installed(monkeypatch):
    monkeypatch.setattr(sys, "prefix", "/Users/me/.local/pipx/venvs/whiz")
    assert OCR._install_argv(["ocrmac"]) == ["pipx", "inject", "whiz", "ocrmac"]


def test_install_argv_uses_pip_outside_pipx(monkeypatch):
    monkeypatch.setattr(sys, "prefix", "/usr/local")
    monkeypatch.setattr(sys, "executable", "/usr/local/bin/python3")
    argv = OCR._install_argv(["rapidocr", "onnxruntime"])
    assert argv == ["/usr/local/bin/python3", "-m", "pip", "install", "rapidocr", "onnxruntime"]


def test_ensure_engine_returns_true_when_already_available(monkeypatch):
    monkeypatch.setattr(OCR, "detect", lambda n: OCR.EngineInfo(n, True, "ready"))
    assert OCR.ensure_engine("tesseract") is True


def test_ensure_engine_does_not_prompt_when_install_cannot_be_automated(monkeypatch):
    """A sudo-requiring install is reported, never run."""
    monkeypatch.setattr(
        OCR, "detect",
        lambda n: OCR.EngineInfo(n, False, "not on PATH", [], "sudo apt-get install tesseract-ocr"),
    )
    messages: list[str] = []
    assert OCR.ensure_engine("tesseract", on_message=messages.append) is False
    assert any("sudo apt-get" in m for m in messages)


def test_ensure_engine_non_interactive_only_hints(monkeypatch):
    monkeypatch.setattr(
        OCR, "detect",
        lambda n: OCR.EngineInfo(n, False, "missing", ["pipx", "inject", "whiz", "ocrmac"], "pipx inject whiz ocrmac"),
    )
    called: list = []
    monkeypatch.setattr(OCR.subprocess, "run", lambda *a, **k: called.append(a))
    messages: list[str] = []
    assert OCR.ensure_engine("apple", interactive=False, on_message=messages.append) is False
    assert called == []  # nothing installed without consent
    assert any("pipx inject" in m for m in messages)


def test_ensure_engine_declined_does_not_install(monkeypatch):
    monkeypatch.setattr(
        OCR, "detect",
        lambda n: OCR.EngineInfo(n, False, "missing", ["pipx", "inject", "whiz", "ocrmac"], "hint"),
    )
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    called: list = []
    monkeypatch.setattr(OCR.subprocess, "run", lambda *a, **k: called.append(a))
    assert OCR.ensure_engine("apple") is False
    assert called == []


# ---------- normalization ----------

def test_normalize_collapses_whitespace_and_blank_lines():
    assert OCR.normalize("  Hello    World \n\n\n  Foo  \n") == "Hello World\nFoo"


def test_normalize_min_chars_drops_noise():
    assert OCR.normalize("ab", min_chars=8) == ""
    assert OCR.normalize("abcdefghij", min_chars=8) == "abcdefghij"


def test_normalize_max_chars_truncates_on_line_boundary():
    out = OCR.normalize("aaaa\nbbbb\ncccc", max_chars=6)
    assert out.endswith("…")
    assert "cccc" not in out


def test_normalize_max_chars_handles_single_long_line():
    out = OCR.normalize("x" * 100, max_chars=10)
    assert len(out) == 11 and out.endswith("…")


def test_normalize_empty_input():
    assert OCR.normalize("") == ""
    assert OCR.normalize(None) == ""


# ---------- engine output shapes ----------

def test_join_apple_orders_top_to_bottom_then_left_to_right():
    # bbox origin is bottom-left, so a larger y is higher on screen.
    annotations = [
        ("bottom", 0.9, (0.1, 0.10, 0.2, 0.05)),
        ("top-right", 0.9, (0.5, 0.90, 0.2, 0.05)),
        ("top-left", 0.9, (0.1, 0.90, 0.2, 0.05)),
    ]
    assert OCR._join_apple(annotations) == "top-left\ntop-right\nbottom"


def test_join_apple_skips_empty_and_malformed():
    assert OCR._join_apple([("", 1.0, (0, 0, 0, 0)), ("ok", 1.0, None), ()]) == "ok"


def test_join_rapidocr_v3_object_shape():
    class Result:
        txts = ("first", " second ", "")
    assert OCR._join_rapidocr(Result()) == "first\nsecond"


def test_join_rapidocr_legacy_tuple_shape():
    legacy = ([[[0, 0], "hello", 0.99], [[0, 1], "world", 0.98]], 0.5)
    assert OCR._join_rapidocr(legacy) == "hello\nworld"


def test_join_rapidocr_none():
    assert OCR._join_rapidocr(None) == ""


def test_tesseract_lang_maps_and_skips_unknown():
    assert OCR._tesseract_lang(["en-US", "de-DE"]) == "eng+deu"
    assert OCR._tesseract_lang(["zz"]) == ""
    assert OCR._tesseract_lang([]) == ""


def test_ocr_image_rejects_unknown_engine(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="Unknown OCR engine"):
        OCR.ocr_image(img, "banana")


# ---------- batch driver ----------

def _fake_impl(texts_by_name):
    def impl(path, languages):
        return texts_by_name.get(Path(path).name, "")
    return impl


def test_ocr_frames_aligns_results_with_inputs(tmp_path, monkeypatch):
    a, b = tmp_path / "seg0001.jpg", tmp_path / "seg0002.jpg"
    a.write_bytes(b"AAA")
    b.write_bytes(b"BBB")
    monkeypatch.setitem(OCR._IMPLS, "fake", _fake_impl({"seg0001.jpg": "one one one", "seg0002.jpg": "two two two"}))
    run = OCR.ocr_frames([a, b], "fake")
    assert run.texts == ["one one one", "two two two"]
    assert run.ok == 2 and run.failed == 0


def test_ocr_frames_reuses_identical_frames(tmp_path, monkeypatch):
    """Identical bytes must be OCR'd once — the whole point of the dedupe."""
    calls: list[str] = []

    def impl(path, languages):
        calls.append(str(path))
        return "same screen text"

    monkeypatch.setitem(OCR._IMPLS, "fake", impl)
    paths = []
    for i in range(1, 4):
        p = tmp_path / f"seg000{i}.jpg"
        p.write_bytes(b"IDENTICAL")
        paths.append(p)
    run = OCR.ocr_frames(paths, "fake", dedupe=True)
    assert len(calls) == 1
    assert run.reused == 2
    assert run.texts == ["same screen text"] * 3


def test_ocr_frames_dedupe_off_reads_every_frame(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setitem(OCR._IMPLS, "fake", lambda p, l: calls.append(str(p)) or "text here")
    paths = []
    for i in range(1, 4):
        p = tmp_path / f"seg000{i}.jpg"
        p.write_bytes(b"IDENTICAL")
        paths.append(p)
    OCR.ocr_frames(paths, "fake", dedupe=False)
    assert len(calls) == 3


def test_ocr_frames_survives_a_failing_frame(tmp_path, monkeypatch):
    """One bad frame must not lose the whole run."""
    def impl(path, languages):
        if "0002" in str(path):
            raise RuntimeError("engine exploded")
        return "fine text here"

    monkeypatch.setitem(OCR._IMPLS, "fake", impl)
    paths = []
    for i, payload in enumerate((b"A", b"B", b"C"), start=1):
        p = tmp_path / f"seg000{i}.jpg"
        p.write_bytes(payload)
        paths.append(p)
    run = OCR.ocr_frames(paths, "fake")
    assert run.failed == 1
    assert run.ok == 2
    assert run.texts[1] == ""
    assert len(run.texts) == 3  # alignment preserved


def test_ocr_frames_missing_file_counts_as_failed(tmp_path, monkeypatch):
    monkeypatch.setitem(OCR._IMPLS, "fake", lambda p, l: "unused")
    run = OCR.ocr_frames([tmp_path / "gone.jpg"], "fake")
    assert run.failed == 1
    assert run.texts == [""]


def test_ocr_frames_reports_progress(tmp_path, monkeypatch):
    monkeypatch.setitem(OCR._IMPLS, "fake", lambda p, l: "some text here")
    paths = []
    for i in range(1, 3):
        p = tmp_path / f"seg000{i}.jpg"
        p.write_bytes(bytes([i]))
        paths.append(p)
    seen: list[tuple[int, int, int]] = []
    OCR.ocr_frames(paths, "fake", on_progress=lambda d, t, r: seen.append((d, t, r)))
    assert seen == [(1, 2, 0), (2, 2, 0)]


def test_ocr_frames_empty_input():
    run = OCR.ocr_frames([], "tesseract")
    assert run.texts == [] and run.ok == 0


def test_frame_digest_stable_and_distinct(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c = tmp_path / "c.jpg"
    c.write_bytes(b"different")
    assert OCR.frame_digest(a) == OCR.frame_digest(b)
    assert OCR.frame_digest(a) != OCR.frame_digest(c)
    assert OCR.frame_digest(tmp_path / "missing.jpg") == ""


# ---------- line-level screen diffing ----------

def test_new_screen_lines_drops_carried_over_chrome():
    """The real win: static window chrome is stated once, not every frame."""
    prev = "Slack\nFile\nEdit\nInbox (3)"
    cur = "Slack\nFile\nEdit\nInbox (4)\nnew message"
    assert OCR.new_screen_lines(cur, prev) == "Inbox (4)\nnew message"


def test_new_screen_lines_first_frame_keeps_everything():
    assert OCR.new_screen_lines("a line\nb line", "") == "a line\nb line"


def test_new_screen_lines_identical_frame_yields_nothing():
    same = "Slack\nFile\nEdit"
    assert OCR.new_screen_lines(same, same) == ""


def test_new_screen_lines_preserves_order():
    prev = "b"
    assert OCR.new_screen_lines("a\nb\nc\nd", prev) == "a\nc\nd"


def test_new_screen_lines_empty_current():
    assert OCR.new_screen_lines("", "anything") == ""


def test_normalize_drops_single_character_glyph_noise():
    """Toolbar icons OCR as stray single characters ('G', 'Q', '+')."""
    assert OCR.normalize("G\nQ\n+\nReal content here") == "Real content here"
    # Two-character content is kept — 'OK', 'No' are meaningful.
    assert "OK" in OCR.normalize("OK\nsomething else entirely")
