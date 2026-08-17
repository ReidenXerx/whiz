"""AI analysis via an OpenAI-compatible chat API (Ollama by default).

Sends a transcript (and optionally on-screen frames) to a chat model for
summary, action items, or freeform analysis. Uses only ``urllib.request`` —
no new dependency. The request format is the OpenAI vision message shape:

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


def resolve_prompt(args) -> str:
    """Pick the prompt based on --summary/--actions/--prompt flags."""
    # --prompt wins; else --summary/--actions combine; default = summary+actions.
    if getattr(args, "prompt", None):
        return args.prompt
    want_summary = getattr(args, "summary", False)
    want_actions = getattr(args, "actions", False)
    if want_summary and want_actions:
        return SUMMARY_AND_ACTIONS_PROMPT
    if want_summary:
        return SUMMARY_PROMPT
    if want_actions:
        return ACTIONS_PROMPT
    # Default: summary + actions.
    return SUMMARY_AND_ACTIONS_PROMPT


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