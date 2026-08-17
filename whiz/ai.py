"""AI analysis via an OpenAI-compatible chat API (Ollama by default).

Sends a transcript (and optionally on-screen frames) to a chat model for
summary, action items, implementation plans, or freeform analysis. The
default analyze path auto-detects whether the transcript is a meeting (→
summary + action items) or a feature/task discussion (→ a structured
implementation plan); explicit ``--summary`` / ``--actions`` / ``--plan`` /
``--prompt`` flags override the detector.

Uses only ``urllib.request`` — no new dependency. The request format is the
OpenAI vision message shape:

    content: [
        {"type": "text", "text": ...},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]

This works with Ollama (``/v1/chat/completions``), LM Studio, vLLM, and any
OpenAI-compatible server. Frames are base64-encoded only at send time so the
on-disk manifest stays small (paths only).
"""

from __future__ import annotations

import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


# Built-in prompt presets. Each is a system/instruction string with a
# ``{transcript}`` placeholder replaced at call time.
SUMMARY_PROMPT = (
    "You are an expert meeting assistant. Below is a transcript of a meeting "
    "(with speaker labels and timestamps). Produce a concise summary:\n"
    "- A 2-3 sentence overview of what the meeting was about.\n"
    "- Key topics discussed (bullet points).\n"
    "- Any decisions made.\n\n"
    "Transcript:\n{transcript}\n\nSummary:"
)

ACTIONS_PROMPT = (
    "You are an expert meeting assistant. Below is a transcript of a meeting "
    "(with speaker labels and timestamps). Extract action items:\n"
    "- One bullet per action item.\n"
    "- Format: '- [owner] action (by deadline if mentioned)'.\n"
    "- Owner = the speaker who committed to it; use '?' if unclear.\n"
    "- Only include concrete tasks, not general discussion points.\n"
    "If there are no action items, say 'No action items found.'\n\n"
    "Transcript:\n{transcript}\n\nAction items:"
)

SUMMARY_AND_ACTIONS_PROMPT = (
    "You are an expert meeting assistant. Below is a transcript of a meeting "
    "(with speaker labels and timestamps). Produce:\n\n"
    "## Summary\n"
    "A 2-3 sentence overview, then key topics as bullet points, then any "
    "decisions made.\n\n"
    "## Action items\n"
    "One bullet per action item: '- [owner] action (by deadline if mentioned)'. "
    "Owner = the speaker who committed to it; '?' if unclear. Only concrete "
    "tasks. If none, write 'No action items found.'\n\n"
    "Transcript:\n{transcript}"
)

PLAN_PROMPT = (
    "You are an expert technical lead. Below is a transcript of a discussion "
    "(with speaker labels and timestamps) that is about building a feature, "
    "fixing a bug, or carrying out a technical task. Turn it into a concrete, "
    "actionable implementation plan.\n\n"
    "Produce exactly these sections, in this order, using Markdown headings:\n\n"
    "## Overview\n"
    "2-4 sentences: what is being built/changed and why.\n\n"
    "## Goal\n"
    "The single concrete outcome that 'done' looks like.\n\n"
    "## Proposed approach\n"
    "A short narrative of how the work will be done — the key design choice and "
    "its rationale.\n\n"
    "## Steps\n"
    "A numbered list. Each step MUST have:\n"
    "- **Step N.** <title> — one line of what to do.\n"
    "- **Owner:** the speaker who raised/owns it, or '?' if unclear.\n"
    "- **Effort:** S / M / L with a one-line justification.\n"
    "List every concrete task inferred from the transcript; if a task is only "
    "implied, mark it '(inferred)'. Keep the list in a sensible execution order.\n\n"
    "## Risks\n"
    "Bullet list of risks and unknowns raised or implied, each with a short "
    "mitigation.\n\n"
    "## Open questions\n"
    "Bullet list of unresolved questions that need a decision or more info. "
    "If none, write 'None.'\n\n"
    "## Acceptance criteria\n"
    "A checklist ('- [ ] ...') of the conditions the finished work must meet to "
    "be considered done. Pull these from the transcript; infer reasonable ones if "
    "the discussion was thin.\n\n"
    "Transcript:\n{transcript}"
)

CLASSIFY_PROMPT = (
    "You are a fast content classifier. Read the transcript below and decide "
    "which of these two categories it belongs to:\n"
    "- MEETING — people discussing a past/ongoing topic, a standup, a review, a "
    "decision meeting, an interview, or general conversation.\n"
    "- PLAN — people discussing building, implementing, fixing, or changing a "
    "specific feature, bug, product, or technical task, where the output should "
    "be an actionable implementation plan rather than meeting notes.\n\n"
    "Reply with EXACTLY ONE token: MEETING or PLAN. No other text, no "
    "punctuation.\n\n"
    "Transcript:\n{transcript}\n\nClassification:"
)


def resolve_prompt(args) -> str:
    """Pick the prompt based on --summary/--actions/--plan/--prompt flags.

    Explicit flags skip the auto-detect path. Precedence:
      --prompt > --plan > --summary/--actions combos > default (auto-detect).
    The default branch returns ``SUMMARY_AND_ACTIONS_PROMPT`` so callers that
    just need a static prompt (and existing tests) keep working; the auto path
    that actually runs the classifier is ``resolve_prompt_auto``.
    """
    # --prompt wins; then --plan; then --summary/--actions combine; default = summary+actions.
    if getattr(args, "prompt", None):
        return args.prompt
    if getattr(args, "plan", False):
        return PLAN_PROMPT
    want_summary = getattr(args, "summary", False)
    want_actions = getattr(args, "actions", False)
    if want_summary and want_actions:
        return SUMMARY_AND_ACTIONS_PROMPT
    if want_summary:
        return SUMMARY_PROMPT
    if want_actions:
        return ACTIONS_PROMPT
    # Default: summary + actions (auto-detect is layered on by resolve_prompt_auto).
    return SUMMARY_AND_ACTIONS_PROMPT


def _explicit_mode_set(args) -> set[str]:
    """Names of the explicit mode flags the user passed (for auto-detect gating)."""
    modes: set[str] = set()
    if getattr(args, "prompt", None):
        modes.add("prompt")
    if getattr(args, "plan", False):
        modes.add("plan")
    if getattr(args, "summary", False):
        modes.add("summary")
    if getattr(args, "actions", False):
        modes.add("actions")
    return modes


def resolve_prompt_auto(transcript: str, *, base_url: str, model: str, api_key: str = "") -> tuple[str, str]:
    """Auto-detect and return (prompt_template, detected_mode).

    Runs ``CLASSIFY_PROMPT`` against the model and routes to ``PLAN_PROMPT`` if the
    reply is ``PLAN`` (case-insensitive, single token), else to
    ``SUMMARY_AND_ACTIONS_PROMPT``. ``detected_mode`` is the lowercased token we
    actually routed on (``'plan'`` or ``'meeting'``).

    On any failure (network/model error, empty/garbled reply), falls back to
    ``SUMMARY_AND_ACTIONS_PROMPT`` with ``detected_mode`` set to ``'meeting'`` plus
    the suffix ``' (fallback)'`` so callers can surface that the detector failed.
    """
    try:
        reply = chat_text(CLASSIFY_PROMPT, transcript, base_url=base_url, model=model, api_key=api_key)
    except RuntimeError as exc:
        # Caller logs the warning; we just pick the safe default.
        _last_classifier_error[0] = str(exc)
        return SUMMARY_AND_ACTIONS_PROMPT, "meeting (fallback)"
    token = reply.strip().splitlines()[0].strip().upper() if reply else ""
    # The model may prepend a stray label or wrap the token in prose; accept the
    # first occurrence of either keyword to be forgiving.
    if "PLAN" in token and "MEETING" not in token:
        return PLAN_PROMPT, "plan"
    if "MEETING" in token:
        return SUMMARY_AND_ACTIONS_PROMPT, "meeting"
    if token == "PLAN":
        return PLAN_PROMPT, "plan"
    # Anything else (including empty) -> safe default.
    return SUMMARY_AND_ACTIONS_PROMPT, "meeting"


# Last classifier error text, stashed here so resolve_prompt_auto stays pure-ish
# while still letting cmd_analyze surface the reason in its warning. Reset on
# each call to resolve_prompt_auto before the classifier runs.
_last_classifier_error: list[str] = [""]


def transcript_text(entries) -> str:
    """Render manifest entries (FrameEntry) or (WhisperSeg, label) pairs as text."""
    lines: list[str] = []
    for e in entries:
        # FrameEntry has .start/.end/.speaker/.text; (WhisperSeg, label) tuples
        # are unwrapped below.
        if hasattr(e, "speaker"):
            ts = _fmt_clock(e.start)
            lines.append(f"[{ts}] {e.speaker}: {e.text}")
        elif isinstance(e, tuple) and len(e) == 2:
            seg, label = e
            ts = _fmt_clock(seg.start)
            lines.append(f"[{ts}] {label}: {seg.text.strip()}")
    return "\n".join(lines)


def _fmt_clock(t: float) -> str:
    total = int(round(max(0.0, t)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _post_chat(base_url: str, model: str, messages: list[dict], api_key: str, timeout: int = 600) -> str:
    """POST to /v1/chat/completions and return the assistant text."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.3,
    }
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    # HTTPError is a subclass of URLError, so it must be caught first —
    # otherwise a 400 (e.g. text model rejecting vision images) is misreported
    # as "could not reach the server".
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local/known server
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        hint = ""
        if e.code in (400, 404) and any(
            "image" in str(m.get("content", "")) or isinstance(m.get("content"), list)
            for m in messages
        ):
            hint = (
                "\nHint: this looks like a vision request. Is the configured model "
                "vision-capable? (e.g. llava, qwen2.5-vl, minicpm-v). "
                "A text-only model will reject image inputs."
            )
        raise RuntimeError(
            f"AI server returned HTTP {e.code} for {url}.\n"
            f"Response: {body_text[:500]}{hint}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach AI server at {url}: {e.reason}\n"
            "Is Ollama running? Start it with:  ollama serve"
        ) from e
    # OpenAI shape: {"choices": [{"message": {"content": "..."}}]}
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"AI server returned no choices: {json.dumps(data)[:300]}")
    content = choices[0].get("message", {}).get("content", "")
    return content.strip()


def chat_text(prompt_template: str, transcript: str, *, base_url: str, model: str, api_key: str = "") -> str:
    """Send a text-only analysis request. ``prompt_template`` has {transcript}."""
    prompt = prompt_template.replace("{transcript}", transcript)
    messages = [
        {"role": "user", "content": prompt},
    ]
    return _post_chat(base_url, model, messages, api_key)


def chat_vision(
    prompt_template: str,
    transcript: str,
    frames: list[Path],
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    max_frames: int = 50,
) -> str:
    """Send a vision request: transcript + frames as base64 image_url content.

    Frames are spread evenly across the list when ``len(frames) > max_frames``
    so the vision model sees a representative sample of the whole video rather
    than just the first N minutes. Frames that don't exist on disk are skipped.
    """
    selected = _subsample(frames, max_frames)
    prompt = prompt_template.replace("{transcript}", transcript)
    content: list[dict] = [{"type": "text", "text": prompt}]
    for fp in selected:
        if not fp.exists():
            continue
        b64 = _base64_jpeg(fp)
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
    messages = [{"role": "user", "content": content}]
    return _post_chat(base_url, model, messages, api_key)


def list_ollama_models(base_url: str, *, timeout: int = 10) -> list[str]:
    """List model names available on an Ollama / OpenAI-compatible server.

    ``base_url`` is the chat base URL (e.g. ``http://localhost:11434/v1``); the
    Ollama native tags endpoint lives at the server root, so we strip a trailing
    ``/v1`` and query ``/api/tags`` first (``{"models":[{"name":...}]}``), then
    fall back to ``/v1/models`` (OpenAI shape ``{"data":[{"id":...}]}``) if the
    native endpoint is unavailable.

    Returns model names in server order (empty list on any failure).
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    # Try the native Ollama tags endpoint first.
    names = _get_json(root + "/api/tags", timeout)
    if names is not None:
        return names
    # Fallback to the OpenAI-style list endpoint.
    return _get_json(base_url.rstrip("/") + "/models", timeout) or []


def probe_model(base_url: str, model: str, api_key: str = "", *, timeout: int = 30) -> tuple[bool, str]:
    """Send a trivial chat completion to check ``model`` is actually usable.

    Ollama's ``/api/tags`` can list models that are retired/unavailable
    server-side (e.g. cloud-tagged models whose upstream was retired) — the
    retirement only surfaces as an HTTP 410 when you actually call the model.
    This probe lets the interactive picker avoid saving a dead model.

    Returns ``(ok, error_message)``; ``error_message`` is empty on success.
    """
    try:
        _post_chat(
            base_url, model,
            [{"role": "user", "content": "Reply with: ok"}],
            api_key, timeout=timeout,
        )
    except RuntimeError as exc:
        return False, str(exc)
    return True, ""


def _get_json(url: str, timeout: int) -> list[str] | None:
    """GET ``url`` and return model names, or None if the endpoint failed.

    Handles both the Ollama shape (``models[].name``) and the OpenAI shape
    (``data[].id``); returns the first shape that matches.
    """
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local server
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None
    models = data.get("models") if isinstance(data, dict) else None
    if isinstance(models, list):
        out = [m.get("name", "") for m in models if isinstance(m, dict)]
        if any(out):
            return [n for n in out if n]
    data_list = data.get("data") if isinstance(data, dict) else None
    if isinstance(data_list, list):
        out = [m.get("id", "") for m in data_list if isinstance(m, dict)]
        if any(out):
            return [n for n in out if n]
    return None


def _subsample(frames: list[Path], max_frames: int) -> list[Path]:
    """Evenly spread ``frames`` to at most ``max_frames`` entries."""
    if max_frames <= 0 or len(frames) <= max_frames:
        return list(frames)
    if max_frames == 1:
        return [frames[len(frames) // 2]]
    step = len(frames) / max_frames
    indices = [int(i * step) for i in range(max_frames)]
    # De-dup while preserving order (can happen for small lists).
    seen: set[int] = set()
    out: list[Path] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            out.append(frames[idx])
    return out


def _base64_jpeg(path: Path) -> str:
    """Read a JPEG file and return its base64 encoding (empty string on error)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return base64.b64encode(raw).decode("ascii")