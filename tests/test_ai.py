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


def test_essentials_instruction_has_required_markers():
    # The always-on Essentials augmentation instruction (appended to every
    # analysis prompt) must ask for the Essentials section and the markers.
    assert "## Essentials" in AI._ESSENTIALS_INSTRUCTION
    assert "OPEN:" in AI._ESSENTIALS_INSTRUCTION
    assert "REJECTED:" in AI._ESSENTIALS_INSTRUCTION
    assert "(inferred)" in AI._ESSENTIALS_INSTRUCTION
    assert "## Essentials" in AI._ESSENTIALS_TASK_SUFFIX
    # The shared analyst posture (thoroughness + frame reconciliation) is folded
    # into every augmentation so it applies to the whole analysis.
    assert "thorough" in AI._ANALYST_POSTURE
    assert "reconcile" in AI._ANALYST_POSTURE
    assert AI._ANALYST_POSTURE in AI._ESSENTIALS_TASK_SUFFIX
    assert AI._ANALYST_POSTURE in AI._ESSENTIALS_INSTRUCTION
    # Conservative screen claims: confidence tags + legibility guard.
    assert "[HIGH]" in AI._ANALYST_POSTURE
    assert "[MEDIUM]" in AI._ANALYST_POSTURE
    assert "[LOW]" in AI._ANALYST_POSTURE
    assert "legibly readable" in AI._ANALYST_POSTURE


def test_plan_prompt_has_required_sections():
    for heading in ("Overview", "Goal", "Proposed approach", "Steps", "Risks",
                    "Open questions", "Acceptance criteria"):
        assert heading in AI.PLAN_PROMPT
    assert "{transcript}" in AI.PLAN_PROMPT
    # Owner must be the named speaker, not a generic role.
    assert "named speaker" in AI.PLAN_PROMPT
    assert "NOT a generic role" in AI.PLAN_PROMPT
    # Effort must include a justification, not a bare size.
    assert "one-line justification" in AI.PLAN_PROMPT
    # Open questions must be deduplicated.
    assert "DEDUPLICATE" in AI.PLAN_PROMPT


def test_classify_prompt_has_tokens():
    assert "MEETING" in AI.CLASSIFY_PROMPT
    assert "PLAN" in AI.CLASSIFY_PROMPT
    assert "{transcript}" in AI.CLASSIFY_PROMPT


def test_synth_prompt_dedupes_open_questions():
    # The synth/reduce step must instruct deduplication of Open questions
    # across partial chunks.
    assert "Deduplicate" in AI.SYNTH_PROMPT
    assert "Open questions" in AI.SYNTH_PROMPT


def test_plan_task_label_mentions_speaker_and_dedup():
    label = AI._task_label(AI.PLAN_PROMPT)
    assert "named speaker" in label
    assert "deduplicated" in label

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
    args = SimpleNamespace(prompt="", plan=False, summary=False, actions=True)
    assert AI._explicit_mode_set(args) == {"actions"}


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


# ---------- chunking: chunk_entries / _chunk_text ----------

def test_chunk_entries_splits_to_size():
    items = list(range(20))
    chunks = AI.chunk_entries(items, chunk_size=8)
    assert [len(c) for c in chunks] == [8, 8, 4]
    # Order preserved.
    flat = [x for c in chunks for x in c]
    assert flat == items


def test_chunk_entries_under_size_is_single_chunk():
    items = [1, 2, 3]
    assert AI.chunk_entries(items, chunk_size=8) == [[1, 2, 3]]


def test_chunk_entries_empty_returns_empty():
    assert AI.chunk_entries([], chunk_size=8) == []


def test_chunk_entries_size_floor_is_one():
    items = [1, 2, 3]
    # chunk_size <= 1 is coerced to 1 so we get single-item chunks.
    assert AI.chunk_entries(items, chunk_size=0) == [[1], [2], [3]]


def test_chunk_text_short_returns_single_chunk():
    text = "line one\nline two\nline three"
    chunks = AI._chunk_text(text, target_chars=10_000)
    assert chunks == [text]


def test_chunk_text_long_splits_on_line_boundaries():
    # 6 lines of ~10 chars each = 60 chars. target 25 => roughly 2-3 chunks.
    lines = [f"line number {i:02d} here" for i in range(6)]
    text = "\n".join(lines)
    chunks = AI._chunk_text(text, target_chars=25)
    assert len(chunks) >= 2
    # Reassembling preserves all the lines.
    rejoined = "\n".join(chunks)
    for ln in lines:
        assert ln in rejoined


def test_chunk_text_empty_returns_empty():
    assert AI._chunk_text("   \n  \n  ", target_chars=10) == []


def test_chunk_text_zero_target_returns_single_chunk():
    text = "hello world"
    assert AI._chunk_text(text, target_chars=0) == [text]


# ---------- chunking: _task_label / _is_built_in_prompt ----------

def test_task_label_built_in_prompts():
    assert "summary" in AI._task_label(AI.SUMMARY_PROMPT)
    assert "action" in AI._task_label(AI.ACTIONS_PROMPT)
    assert "summary" in AI._task_label(AI.SUMMARY_AND_ACTIONS_PROMPT)
    assert "implementation plan" in AI._task_label(AI.PLAN_PROMPT)


def test_task_label_custom_prompt_fallback():
    label = AI._task_label("What risks? {transcript}")
    assert "user's question" in label


def test_is_built_in_prompt_recognizes_presets():
    assert AI._is_built_in_prompt(AI.SUMMARY_PROMPT) is True
    assert AI._is_built_in_prompt(AI.PLAN_PROMPT) is True
    assert AI._is_built_in_prompt("custom {transcript}") is False


# ---------- analyze: short input single call (no chunking) ----------

def test_analyze_short_text_single_call(monkeypatch):
    """A short transcript uses one chat_text call (no map-reduce). The prompt is
    augmented with the always-on Essentials instruction."""
    calls = []

    def fake_chat_text(prompt_template, transcript, *, base_url, model, api_key):
        calls.append((prompt_template, transcript))
        return "final answer"

    monkeypatch.setattr(AI, "chat_text", fake_chat_text)
    out = AI.analyze(
        AI.SUMMARY_PROMPT, "a short transcript",
        base_url="http://x/v1", model="m", api_key="",
    )
    assert out == "final answer"
    assert len(calls) == 1
    # The single-call prompt carries the Essentials instruction (always on) but
    # is still recognizable as the summary prompt.
    assert "## Essentials" in calls[0][0]
    assert AI.SUMMARY_PROMPT.split("{transcript}")[0] in calls[0][0]
    assert calls[0][1] == "a short transcript"


def test_analyze_short_vision_single_call(monkeypatch, tmp_path):
    """With one chunk of entries, vision analyze uses one chat_vision call.
    The prompt carries the always-on Essentials instruction."""
    from whiz.screenshots import FrameEntry
    entries = [
        FrameEntry(index=1, start=0.0, end=1.0, speaker="A", text="hi", frame="seg0001.jpg"),
        FrameEntry(index=2, start=1.0, end=2.0, speaker="B", text="yo", frame="seg0002.jpg"),
    ]
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for e in entries:
        (frames_dir / e.frame).write_bytes(b"\xff\xd8img\xff\xd9")

    calls = []
    def fake_chat_vision(prompt, transcript, frames, *, base_url, model, api_key, max_frames):
        calls.append((prompt, transcript, len(frames)))
        return "vision answer"
    monkeypatch.setattr(AI, "chat_vision", fake_chat_vision)
    monkeypatch.setattr(AI, "chat_text", lambda *a, **k: "SHOULD NOT BE CALLED")

    out = AI.analyze(
        AI.SUMMARY_PROMPT, AI.transcript_text(entries),
        base_url="http://x/v1", model="m", api_key="",
        entries=entries, frames_dir=frames_dir, use_vision=True, max_frames=50,
    )
    assert out == "vision answer"
    assert len(calls) == 1
    # Single-call path uses the augmented prompt (Essentials instruction on it)
    # and all frames.
    assert "## Essentials" in calls[0][0]
    assert calls[0][2] == 2


# ---------- analyze: long input map-reduce ----------

def test_analyze_long_text_map_reduce(monkeypatch):
    """A long transcript is chunked: map per chunk, then synth reduce."""
    long_text = "\n".join(f"line {i} some words here" for i in range(50))
    assert len(AI._chunk_text(long_text, target_chars=120)) >= 2

    chat_calls = []

    def fake_chat_text(prompt_template, transcript, *, base_url, model, api_key):
        chat_calls.append((prompt_template, transcript))
        # Map calls use MAP_PROMPT (contains "chunk"); synth uses SYNTH_PROMPT.
        if "combining" in prompt_template:
            return "SYNTH FINAL"
        return f"partial({transcript[:8]})"

    monkeypatch.setattr(AI, "chat_text", fake_chat_text)
    out = AI.analyze(
        AI.SUMMARY_AND_ACTIONS_PROMPT, long_text,
        base_url="http://x/v1", model="m", api_key="",
        chunk_chars=120,
    )
    assert out == "SYNTH FINAL"
    # Expect N map calls + 1 synth call.
    map_calls = [c for c in chat_calls if "chunk" in c[0]]
    synth_calls = [c for c in chat_calls if "combining" in c[0]]
    assert len(map_calls) >= 2
    assert len(synth_calls) == 1
    # Map prompts carry the task label for built-in prompts.
    assert "summary" in map_calls[0][0]
    # Synth prompt receives the concatenated partials.
    assert "partial(" in synth_calls[0][0]


def test_analyze_long_vision_map_reduce(monkeypatch, tmp_path):
    """Many entries with frames chunk by entries; each chunk's frames stay local."""
    from whiz.screenshots import FrameEntry
    entries = [
        FrameEntry(index=i, start=float(i), end=float(i + 1),
                   speaker="A", text=f"seg {i}", frame=f"seg{i:04d}.jpg")
        for i in range(1, 21)  # 20 entries -> 3 chunks of 8
    ]
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for e in entries:
        (frames_dir / e.frame).write_bytes(b"\xff\xd8img\xff\xd9")

    vision_calls = []
    text_calls = []

    def fake_chat_vision(prompt, transcript, frames, *, base_url, model, api_key, max_frames):
        vision_calls.append((prompt, len(frames)))
        return f"p{len(vision_calls)}"
    def fake_chat_text(prompt, transcript, *, base_url, model, api_key):
        text_calls.append(prompt)
        return "SYNTH"

    monkeypatch.setattr(AI, "chat_vision", fake_chat_vision)
    monkeypatch.setattr(AI, "chat_text", fake_chat_text)

    out = AI.analyze(
        AI.PLAN_PROMPT, AI.transcript_text(entries),
        base_url="http://x/v1", model="m", api_key="",
        entries=entries, frames_dir=frames_dir, use_vision=True,
        max_frames=50, chunk_size=8,
    )
    assert out == "SYNTH"
    # 3 map vision calls (8 + 8 + 4), each carrying only its chunk's frames.
    assert len(vision_calls) == 3
    assert vision_calls[0][1] == 8
    assert vision_calls[1][1] == 8
    assert vision_calls[2][1] == 4
    # 1 synth text call.
    assert len(text_calls) == 1
    assert "combining" in text_calls[0]


def test_analyze_custom_prompt_long_with_rolling_context(monkeypatch):
    """Custom --prompt with rolling context: chunk 1 is verbatim, chunk 2+ has
    the context block prepended."""
    long_text = "\n".join(f"line {i} content here" for i in range(40))
    assert len(AI._chunk_text(long_text, target_chars=120)) >= 2

    calls = []
    def fake_chat_text(prompt, transcript, *, base_url, model, api_key):
        calls.append(prompt)
        if "Partial answers" in prompt:
            return "MERGED"
        return "chunk-answer"
    monkeypatch.setattr(AI, "chat_text", fake_chat_text)

    out = AI.analyze(
        "What risks? {transcript}", long_text,
        base_url="http://x/v1", model="m", api_key="",
        chunk_chars=120,
    )
    assert out == "MERGED"
    map_prompts = [p for p in calls if "What risks?" in p]
    synth_prompts = [p for p in calls if "Partial answers" in p]
    assert len(map_prompts) >= 2
    # Map calls use the user's prompt verbatim (no MAP_PROMPT wrapper). Chunk 1
    # has no running context yet; later chunks get the context block prepended.
    assert map_prompts[0].startswith("What risks?")
    assert "What risks?" in map_prompts[1]
    assert len(synth_prompts) == 1


def test_analyze_custom_prompt_no_context_when_disabled(monkeypatch):
    """context_turns=0 disables rolling context: every map call starts with
    the user's prompt verbatim (the old independent-chunk behavior)."""
    long_text = "\n".join(f"line {i} content here" for i in range(40))
    assert len(AI._chunk_text(long_text, target_chars=120)) >= 2

    calls = []
    def fake_chat_text(prompt, transcript, *, base_url, model, api_key):
        calls.append(prompt)
        if "Partial answers" in prompt:
            return "MERGED"
        return "chunk-answer"
    monkeypatch.setattr(AI, "chat_text", fake_chat_text)

    out = AI.analyze(
        "What risks? {transcript}", long_text,
        base_url="http://x/v1", model="m", api_key="",
        chunk_chars=120, context_turns=0,
    )
    assert out == "MERGED"
    map_prompts = [p for p in calls if "What risks?" in p]
    synth_prompts = [p for p in calls if "Partial answers" in p]
    assert len(map_prompts) >= 2
    for p in map_prompts:
        assert p.startswith("What risks?")
    assert len(synth_prompts) == 1


# ---------- rolling context across chunks ----------

def test_running_context_chunk1_is_empty(monkeypatch):
    """The first chunk has no prior partials, so its context block is empty."""
    seen_prompts = []
    def fake_chat_text(prompt, transcript, *, base_url, model, api_key):
        seen_prompts.append(prompt)
        return "partial"
    monkeypatch.setattr(AI, "chat_text", fake_chat_text)
    long_text = "\n".join(f"line {i} content here" for i in range(40))
    AI.analyze(
        AI.SUMMARY_PROMPT, long_text,
        base_url="http://x/v1", model="m", api_key="",
        chunk_chars=120, context_turns=3,
    )
    # The first map call is chunk 1 — no running context block.
    first = seen_prompts[0]
    assert "Running context" not in first


def test_running_context_later_chunks_carry_prior_partials(monkeypatch):
    """Chunk 2+ injects the prior chunks' partial analyses as running context."""
    seen_prompts = []
    def fake_chat_text(prompt, transcript, *, base_url, model, api_key):
        seen_prompts.append(prompt)
        if "combining" in prompt:
            return "SYNTH"
        return f"PARTIAL-{transcript[:6]}"
    monkeypatch.setattr(AI, "chat_text", fake_chat_text)
    long_text = "\n".join(f"line {i} content here" for i in range(40))
    AI.analyze(
        AI.SUMMARY_PROMPT, long_text,
        base_url="http://x/v1", model="m", api_key="",
        chunk_chars=120, context_turns=3,
    )
    # Map calls contain "Transcript chunk"; the synth call contains "combining".
    map_prompts = [p for p in seen_prompts if "Transcript chunk" in p]
    assert len(map_prompts) >= 2
    # Chunk 1 has no running context; chunk 2 must contain chunk 1's partial.
    assert "Running context" not in map_prompts[0]
    assert "Running context" in map_prompts[1]
    # The context block in chunk 2 carries the first partial's body.
    assert "PARTIAL-" in map_prompts[1]


def test_running_context_window_caps(monkeypatch):
    """context_turns limits how many prior partials are injected."""
    seen_prompts = []
    def fake_chat_text(prompt, transcript, *, base_url, model, api_key):
        seen_prompts.append(prompt)
        return f"PARTIAL-{transcript[:6]}"
    monkeypatch.setattr(AI, "chat_text", fake_chat_text)
    long_text = "\n".join(f"line {i} content here" for i in range(80))
    # Force small chunks + only 1 turn of context.
    AI.analyze(
        AI.SUMMARY_PROMPT, long_text,
        base_url="http://x/v1", model="m", api_key="",
        chunk_chars=120, context_turns=1,
    )
    map_prompts = [p for p in seen_prompts if "Transcript chunk" in p]
    assert len(map_prompts) >= 3
    # The last chunk's context block contains only the immediately-preceding
    # partial (window of 1), not all prior partials.
    last = map_prompts[-1]
    assert "Running context" in last
    markers = [m for m in last.splitlines() if m.startswith("PARTIAL-")]
    assert len(markers) == 1


def test_running_context_disabled_when_zero(monkeypatch):
    """context_turns=0 disables rolling context (old independent-chunk behavior)."""
    seen_prompts = []
    def fake_chat_text(prompt, transcript, *, base_url, model, api_key):
        seen_prompts.append(prompt)
        return "partial"
    monkeypatch.setattr(AI, "chat_text", fake_chat_text)
    long_text = "\n".join(f"line {i} content here" for i in range(40))
    AI.analyze(
        AI.SUMMARY_PROMPT, long_text,
        base_url="http://x/v1", model="m", api_key="",
        chunk_chars=120, context_turns=0,
    )
    map_prompts = [p for p in seen_prompts if "Transcript chunk" in p]
    assert len(map_prompts) >= 2
    for p in map_prompts:
        assert "Running context" not in p


def test_analyze_progress_callback_invoked(monkeypatch):
    """on_progress is called before each map call and the synth call."""
    long_text = "\n".join(f"line {i} content" for i in range(30))
    msgs = []

    def fake_chat_text(prompt, transcript, *, base_url, model, api_key):
        return "x"
    monkeypatch.setattr(AI, "chat_text", fake_chat_text)

    AI.analyze(
        AI.SUMMARY_PROMPT, long_text,
        base_url="http://x/v1", model="m", api_key="",
        chunk_chars=120,
        on_progress=lambda m: msgs.append(m),
    )
    assert any("analyzing chunk" in m for m in msgs)
    assert any("synthesizing" in m for m in msgs)
