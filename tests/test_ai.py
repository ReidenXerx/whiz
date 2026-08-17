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


def test_resolve_prompt_plan_flag():
    args = SimpleNamespace(prompt="", plan=True, summary=False, actions=False)
    assert AI.resolve_prompt(args) == AI.PLAN_PROMPT


def test_resolve_prompt_plan_overridden_by_prompt():
    args = SimpleNamespace(prompt="custom {transcript}", plan=True, summary=True, actions=True)
    assert AI.resolve_prompt(args) == "custom {transcript}"


def test_plan_prompt_has_required_sections():
    for heading in ("Overview", "Goal", "Proposed approach", "Steps", "Risks",
                    "Open questions", "Acceptance criteria"):
        assert heading in AI.PLAN_PROMPT
    assert "{transcript}" in AI.PLAN_PROMPT


def test_classify_prompt_has_tokens():
    assert "MEETING" in AI.CLASSIFY_PROMPT
    assert "PLAN" in AI.CLASSIFY_PROMPT
    assert "{transcript}" in AI.CLASSIFY_PROMPT


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


# ---------- resolve_prompt_auto (classifier routing) ----------

def test_resolve_prompt_auto_routes_to_plan(monkeypatch):
    calls = []

    def fake_chat_text(prompt_template, transcript, *, base_url, model, api_key):
        calls.append(prompt_template)
        if prompt_template is AI.CLASSIFY_PROMPT:
            return "PLAN"
        return "plan body"

    monkeypatch.setattr(AI, "chat_text", fake_chat_text)
    prompt, mode = AI.resolve_prompt_auto(
        "some transcript", base_url="http://x/v1", model="m", api_key="",
    )
    assert prompt is AI.PLAN_PROMPT
    assert mode == "plan"
    assert calls and calls[0] is AI.CLASSIFY_PROMPT


def test_resolve_prompt_auto_routes_to_meeting(monkeypatch):
    monkeypatch.setattr(AI, "chat_text", lambda pt, t, **kw: "MEETING"
                       if pt is AI.CLASSIFY_PROMPT else "meeting body")
    prompt, mode = AI.resolve_prompt_auto(
        "some transcript", base_url="http://x/v1", model="m", api_key="",
    )
    assert prompt is AI.SUMMARY_AND_ACTIONS_PROMPT
    assert mode == "meeting"


def test_resolve_prompt_auto_lowercase_token(monkeypatch):
    monkeypatch.setattr(AI, "chat_text", lambda pt, t, **kw: "plan" if pt is AI.CLASSIFY_PROMPT else "x")
    prompt, mode = AI.resolve_prompt_auto(
        "t", base_url="http://x/v1", model="m", api_key="",
    )
    assert prompt is AI.PLAN_PROMPT
    assert mode == "plan"


def test_resolve_prompt_auto_fallback_on_error(monkeypatch):
    def fake_chat_text(prompt_template, transcript, *, base_url, model, api_key):
        if prompt_template is AI.CLASSIFY_PROMPT:
            raise RuntimeError("boom")
        return "x"

    monkeypatch.setattr(AI, "chat_text", fake_chat_text)
    prompt, mode = AI.resolve_prompt_auto(
        "t", base_url="http://x/v1", model="m", api_key="",
    )
    assert prompt is AI.SUMMARY_AND_ACTIONS_PROMPT
    assert "fallback" in mode


def test_resolve_prompt_auto_garbled_reply_defaults_to_meeting(monkeypatch):
    monkeypatch.setattr(AI, "chat_text", lambda pt, t, **kw: "banana" if pt is AI.CLASSIFY_PROMPT else "x")
    prompt, mode = AI.resolve_prompt_auto(
        "t", base_url="http://x/v1", model="m", api_key="",
    )
    assert prompt is AI.SUMMARY_AND_ACTIONS_PROMPT
    assert mode == "meeting"


def test_explicit_mode_set_detects_flags():
    args = SimpleNamespace(prompt="", plan=True, summary=False, actions=False)
    assert AI._explicit_mode_set(args) == {"plan"}
    args = SimpleNamespace(prompt="custom", plan=False, summary=True, actions=False)
    assert AI._explicit_mode_set(args) == {"prompt", "summary"}
    args = SimpleNamespace(prompt="", plan=False, summary=False, actions=False)
    assert AI._explicit_mode_set(args) == set()


# ---------- list_ollama_models ----------

def test_list_ollama_models_native_tags(monkeypatch):
    payload = json.dumps({"models": [
        {"name": "llava:latest"},
        {"name": "qwen2.5:3b"},
    ]}).encode()
    resp = io.BytesIO(payload); resp.status = 200
    resp.__enter__ = lambda: resp; resp.__exit__ = lambda *a: None

    seen = []
    def fake_urlopen(req, timeout=10):
        seen.append(req.full_url)
        return resp

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    names = AI.list_ollama_models("http://localhost:11434/v1")
    assert names == ["llava:latest", "qwen2.5:3b"]
    assert seen and seen[0] == "http://localhost:11434/api/tags"


def test_list_ollama_models_fallback_to_openai_shape(monkeypatch):
    tags_resp = urllib.error.HTTPError(
        "http://x/api/tags", 404, "NF", hdrs=None, fp=io.BytesIO(b"{}"),
    )
    models_payload = json.dumps({"data": [
        {"id": "gpt-4o-mini"},
        {"id": "llama3"},
    ]}).encode()
    models_resp = io.BytesIO(models_payload); models_resp.status = 200
    models_resp.__enter__ = lambda: models_resp; models_resp.__exit__ = lambda *a: None

    calls = []
    def fake_urlopen(req, timeout=10):
        calls.append(req.full_url)
        if req.full_url.endswith("/api/tags"):
            raise tags_resp
        return models_resp

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    names = AI.list_ollama_models("http://x/v1")
    assert names == ["gpt-4o-mini", "llama3"]
    assert any(u.endswith("/api/tags") for u in calls)
    assert any(u.endswith("/v1/models") for u in calls)


def test_list_ollama_models_server_down_returns_empty(monkeypatch):
    def fake_urlopen(req, timeout=10):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert AI.list_ollama_models("http://localhost:11434/v1") == []


def test_list_ollama_models_strips_base_url_without_v1(monkeypatch):
    payload = json.dumps({"models": [{"name": "m1"}]}).encode()
    resp = io.BytesIO(payload); resp.status = 200
    resp.__enter__ = lambda: resp; resp.__exit__ = lambda *a: None

    seen = []
    def fake_urlopen(req, timeout=10):
        seen.append(req.full_url)
        return resp

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    AI.list_ollama_models("http://localhost:11434/")
    assert seen[0] == "http://localhost:11434/api/tags"


# ---------- probe_model ----------

def test_probe_model_ok(monkeypatch):
    def fake_urlopen(req, timeout=30):
        return _mock_response({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, err = AI.probe_model("http://x/v1", "live-model", api_key="")
    assert ok is True
    assert err == ""


def test_probe_model_retired_returns_failure(monkeypatch):
    def fake_urlopen(req, timeout=30):
        raise urllib.error.HTTPError(
            req.full_url, 410, "Gone", hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"model was retired"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, err = AI.probe_model("http://x/v1", "dead-model", api_key="")
    assert ok is False
    assert "HTTP 410" in err


def test_probe_model_server_down_returns_failure(monkeypatch):
    def fake_urlopen(req, timeout=30):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, err = AI.probe_model("http://x/v1", "m", api_key="")
    assert ok is False
    assert "Could not reach AI server" in err
