"""Tests for whiz.ai — prompt resolution, subsampling, base64, HTTP mocking.

Run with: pytest tests/test_ai.py
"""

from __future__ import annotations

import base64
import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whiz import ai as AI


# ---------- prompt resolution ----------

def test_resolve_prompt_custom_wins():
    args = SimpleNamespace(prompt="Why? {transcript}", summary=True, actions=True)
    assert AI.resolve_prompt(args) == "Why? {transcript}"


def test_resolve_prompt_summary_only():
    args = SimpleNamespace(prompt="", summary=True, actions=False)
    assert AI.resolve_prompt(args) == AI.SUMMARY_PROMPT


def test_resolve_prompt_actions_only():
    args = SimpleNamespace(prompt="", summary=False, actions=True)
    assert AI.resolve_prompt(args) == AI.ACTIONS_PROMPT


def test_resolve_prompt_default_summary_and_actions():
    args = SimpleNamespace(prompt="", summary=False, actions=False)
    assert AI.resolve_prompt(args) == AI.SUMMARY_AND_ACTIONS_PROMPT


def test_resolve_prompt_summary_and_actions_explicit():
    args = SimpleNamespace(prompt="", summary=True, actions=True)
    assert AI.resolve_prompt(args) == AI.SUMMARY_AND_ACTIONS_PROMPT


# ---------- transcript_text ----------

def test_transcript_text_with_frame_entries():
    from whiz.screenshots import FrameEntry
    entries = [
        FrameEntry(index=1, start=0.0, end=2.0, speaker="Alice", text="hello", frame="seg0001.jpg"),
        FrameEntry(index=2, start=2.0, end=4.0, speaker="Bob", text="world", frame="seg0002.jpg"),
    ]
    txt = AI.transcript_text(entries)
    assert "[00:00:00] Alice: hello" in txt
    assert "[00:00:02] Bob: world" in txt


def test_transcript_text_with_merged_tuples():
    from whiz.merge import WhisperSeg
    entries = [
        (WhisperSeg(start=0.0, end=1.0, text="  hi  "), "Speaker A"),
        (WhisperSeg(start=5.0, end=6.0, text="bye"), "Speaker B"),
    ]
    txt = AI.transcript_text(entries)
    assert "[00:00:00] Speaker A: hi" in txt
    assert "[00:00:05] Speaker B: bye" in txt


# ---------- subsampling ----------

def test_subsample_under_cap_returns_all():
    frames = [Path(f"f{i}.jpg") for i in range(10)]
    assert AI._subsample(frames, 50) == frames


def test_subsample_over_cap_even_spread():
    frames = [Path(f"f{i}.jpg") for i in range(100)]
    out = AI._subsample(frames, 5)
    assert len(out) == 5
    # First and last sampled indices should span the list.
    names = [p.name for p in out]
    assert names[0] == "f0.jpg"
    assert names[-1] == "f80.jpg"  # int(4 * 100/5) = 80


def test_subsample_one_returns_middle():
    frames = [Path(f"f{i}.jpg") for i in range(10)]
    out = AI._subsample(frames, 1)
    assert len(out) == 1
    assert out[0].name == "f5.jpg"  # middle


def test_subsample_zero_or_negative_returns_all():
    frames = [Path("a.jpg"), Path("b.jpg")]
    assert AI._subsample(frames, 0) == frames
    assert AI._subsample(frames, -1) == frames


# ---------- base64 ----------

def test_base64_jpeg_round_trip(tmp_path):
    raw = b"\xff\xd8\xff\xe0fake-jpeg-data\xff\xd9"
    f = tmp_path / "frame.jpg"
    f.write_bytes(raw)
    b64 = AI._base64_jpeg(f)
    assert base64.b64decode(b64) == raw


def test_base64_jpeg_missing_file_returns_empty(tmp_path):
    assert AI._base64_jpeg(tmp_path / "nope.jpg") == ""


# ---------- HTTP: chat_text / chat_vision with mocked urlopen ----------

def _mock_response(payload: dict):
    """Build a fake urllib response object with .read() and context manager."""
    body = json.dumps(payload).encode("utf-8")
    resp = io.BytesIO(body)
    resp.status = 200
    resp.__enter__ = lambda: resp
    resp.__exit__ = lambda *a: None
    return resp


def test_chat_text_returns_content(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=600):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data.decode("utf-8"))
        captured["headers"] = req.headers
        return _mock_response({"choices": [{"message": {"content": "  Summary!  "}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = AI.chat_text(
        "Summarize: {transcript}", "the transcript",
        base_url="http://localhost:11434/v1", model="llama", api_key="",
    )
    assert out == "Summary!"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["data"]["model"] == "llama"
    assert captured["data"]["messages"][0]["role"] == "user"
    assert "the transcript" in captured["data"]["messages"][0]["content"]


def test_chat_text_sends_api_key_header(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=600):
        captured["auth"] = req.headers.get("Authorization")
        return _mock_response({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    AI.chat_text("p: {transcript}", "t", base_url="http://x/v1", model="m", api_key="secret-key")
    assert captured["auth"] == "Bearer secret-key"


def test_chat_vision_builds_image_content(monkeypatch, tmp_path):
    captured = {}

    # Make two fake JPEG frames.
    f1 = tmp_path / "f1.jpg"
    f2 = tmp_path / "f2.jpg"
    f1.write_bytes(b"\xff\xd8img1\xff\xd9")
    f2.write_bytes(b"\xff\xd8img2\xff\xd9")

    def fake_urlopen(req, timeout=600):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _mock_response({"choices": [{"message": {"content": "seen"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = AI.chat_vision(
        "Describe: {transcript}", "text", [f1, f2],
        base_url="http://x/v1", model="llava", api_key="", max_frames=5,
    )
    assert out == "seen"
    content = captured["body"]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    image_parts = [c for c in content if c["type"] == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_chat_text_http_error_raises_with_body(monkeypatch):
    def fake_urlopen(req, timeout=600):
        raise urllib.error.HTTPError(
            req.full_url, 500, "Server Error", hdrs=None, fp=io.BytesIO(b"boom"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    try:
        AI.chat_text("p: {transcript}", "t", base_url="http://x/v1", model="m", api_key="")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "HTTP 500" in str(e)
        assert "boom" in str(e)


def test_chat_text_url_error_suggests_ollama(monkeypatch):
    def fake_urlopen(req, timeout=600):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    try:
        AI.chat_text("p: {transcript}", "t", base_url="http://x/v1", model="m", api_key="")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "Could not reach AI server" in str(e)
        assert "ollama serve" in str(e)


def test_chat_text_no_choices_raises(monkeypatch):
    def fake_urlopen(req, timeout=600):
        return _mock_response({"choices": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    try:
        AI.chat_text("p: {transcript}", "t", base_url="http://x/v1", model="m", api_key="")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "no choices" in str(e)