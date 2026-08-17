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

# Essentials: a dense, concentrated extraction of EVERY meaningful point from
# the whole recording — facts, decisions, requirements, names, numbers, UI
# details, workflows, constraints, open questions — as one tight bullet list.
# Prioritizes completeness and density over prose. With vision (frames) it
# also captures on-screen UI/schema/label detail. Designed as a standalone
# artifact you can feed back to a later `whiz analyze` (or any AI) as context.
ESSENTIALS_PROMPT = (
    "You are an expert analyst. Below is a transcript of a recording (with "
    "speaker labels and timestamps). "
    "Extract EVERY meaningful point into one dense, concentrated section. "
    "This is for feeding to a later AI analysis pass, so be exhaustive and "
    "specific — do not summarize away detail. Capture:\n"
    "- Facts, decisions, and conclusions stated (who decided what).\n"
    "- Requirements, constraints, and rules (feature flags, names, ids, enums).\n"
    "- Concrete numbers, dates, versions, URLs, field names, table/column names, "
    "API methods, status values.\n"
    "- UI/UX specifics: labels, button names, tabs, modals, workflows, and "
    "what changed (before/after). When frames are provided, read visible text "
    "and schema on screen.\n"
    "- Names of people, teams, products, and vendors mentioned.\n"
    "- Open questions, TODOs, and unresolved threads (mark them 'OPEN:').\n"
    "- Explicit disagreements or rejected alternatives (mark them 'REJECTED:').\n\n"
    "Format as Markdown:\n"
    "## Essentials\n"
    "A flat, grouped bullet list (group by topic with bold sub-headers as "
    "needed). Each bullet is ONE concise point. Prefix with a timestamp and "
    "speaker when useful, e.g. '- [00:12:03] Vadim: must use GET for the Export "
    "endpoint (Enric confirmed)'. Do not invent anything not in the transcript "
    "or frames; if something is only implied, mark it '(inferred)'. Be as "
    "complete as possible — prefer more short bullets over fewer long ones.\n\n"
    "Transcript:\n{transcript}"
)


# Chunked map-reduce prompts. For long transcripts (or many frames) the input is
# split into contiguous chunks; each chunk is analyzed (map) and the partial
# results are merged into one final answer (reduce). Chunking keeps each model
# call focused on a small, coherent window so the model isn't overwhelmed by
# one giant blob — this measurably improves analysis quality.
#
# The map phase is **rolling-context** (chat-style): each chunk carries forward
# the prior chunks' partial analyses as a running context block, so chunk N can
# refer back to speakers/decisions/entities/open threads established in chunks
# 1..N-1 instead of analyzing in a vacuum. This keeps the per-chunk partials
# coherent with each other (the reduce step then just merges them).
MAP_PROMPT = (
    "You are analyzing chunk {k} of {n} of a longer recorded transcript. "
    "Your job: {task}\n\n"
    "{context_block}"
    "Analyze the transcript chunk below. Keep speaker labels and timestamps. "
    "Build on the running context above when present — refer back to speakers, "
    "decisions, entities, and open threads already established; do not re-derive "
    "them from scratch. Do not invent anything not supported by this chunk or the "
    "running context. Produce a partial result for THIS chunk that fits coherently "
    "with what came before; a later step will merge all chunks.\n\n"
    "Transcript chunk ({k}/{n}):\n{transcript}"
)

# Injected before a chunk's analysis when prior chunks exist. {context} is the
# concatenation of the prior chunks' partial analyses (windowed — see
# ``context_turns``), in order. Empty string for chunk 1 (no context yet).
_CONTEXT_BLOCK = (
    "Running context from prior chunks (their partial analyses, in order):\n"
    "{context}\n\n"
    "Continue from this context.\n\n"
)

SYNTH_PROMPT = (
    "You are combining {n} partial analyses of a long recorded transcript into "
    "one final answer. Your job: {task}\n\n"
    "Below are the {n} partial analyses, one per contiguous chunk, in time order. "
    "They were produced with rolling context, so later partials already refer back "
    "to earlier ones. Merge them into a single coherent answer: remove duplicates, "
    "reconcile conflicts, keep the chronological order, and preserve specific "
    "speaker/time references. Produce the final answer in the exact format the task "
    "expects.\n\n"
    "Partial analyses:\n{partials}"
)
# Used only for custom --prompt: the user's prompt is applied per chunk (map,
# with rolling context prepended for chunks after the first), then the per-chunk
# answers are merged with this generic reduce prompt.
_CUSTOM_REDUCE_PROMPT = (
    "Below are {n} partial answers, one per contiguous chunk of a long "
    "transcript, in time order. They were produced with rolling context, so "
    "later answers already refer back to earlier ones. Merge them into one final "
    "answer: remove duplicates, reconcile conflicts, and preserve specific "
    "speaker/time references. Do not add anything not supported by the partials.\n\n"
    "Partial answers:\n{partials}"
)

# Human-readable task description per built-in prompt, used to fill {task} in
# MAP_PROMPT / SYNTH_PROMPT. Looked up by identity (the constants are module
# globals, so `is` is safe and avoids matching user prompts by accident).
_BUILT_IN_TASKS: list[tuple[str, str]] = [
    (SUMMARY_PROMPT,
     "produce a concise meeting summary — a 2-3 sentence overview, key topics as "
     "bullet points, and any decisions made"),
    (ACTIONS_PROMPT,
     "extract action items — one bullet per item as '- [owner] action "
     "(by deadline if mentioned)'; owner = the speaker who committed to it, '?' "
     "if unclear; only concrete tasks"),
    (SUMMARY_AND_ACTIONS_PROMPT,
     "produce a meeting summary (overview + key topics + decisions) followed by "
     "action items (one bullet per item as '- [owner] action (by deadline)')"),
    (PLAN_PROMPT,
     "produce a structured implementation plan with sections: Overview, Goal, "
     "Proposed approach, Steps (each with Owner + Effort S/M/L), Risks, Open "
     "questions, Acceptance criteria"),
    (ESSENTIALS_PROMPT,
     "produce a dense Essentials section — one bullet per meaningful point "
     "(facts, decisions, requirements, names, numbers, UI/workflow details, open "
     "questions), exhaustive and specific, for use as context by a later analysis"),
]


def _task_label(prompt_template: str) -> str:
    """Human-readable description of what ``prompt_template`` asks for.

    Built-in prompts get a tailored label (so MAP_PROMPT/SYNTH_PROMPT reproduce
    the same output structure the non-chunked path would). Custom ``--prompt``
    text falls back to a generic description; the user's prompt is applied per
    chunk verbatim in that case (see ``analyze``).
    """
    for prompt, label in _BUILT_IN_TASKS:
        if prompt_template is prompt:
            return label
    return ("answer the user's question about the transcript (the user's prompt "
            "is applied to each chunk verbatim)")


def _is_built_in_prompt(prompt_template: str) -> bool:
    """True when ``prompt_template`` is one of the module-level presets."""
    return any(prompt_template is prompt for prompt, _ in _BUILT_IN_TASKS)


def resolve_prompt(args) -> str:
    """Pick the prompt based on --summary/--actions/--plan/--prompt flags.

    Explicit flags skip the auto-detect path. Precedence:
      --prompt > --plan > --summary/--actions combos > default (auto-detect).
    The default branch returns ``SUMMARY_AND_ACTIONS_PROMPT`` so callers that
    just need a static prompt (and existing tests) keep working; the auto path
    that actually runs the classifier is ``resolve_prompt_auto``.
    """
    # --prompt wins; then --plan; then --essentials; then --summary/--actions
    # combine; default = summary+actions.
    if getattr(args, "prompt", None):
        return args.prompt
    if getattr(args, "plan", False):
        return PLAN_PROMPT
    if getattr(args, "essentials", False):
        return ESSENTIALS_PROMPT
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
    if getattr(args, "essentials", False):
        modes.add("essentials")
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


def chunk_entries(entries: list, chunk_size: int = 8) -> list[list]:
    """Split a list into contiguous sublists of at most ``chunk_size`` items."""
    if chunk_size <= 1:
        chunk_size = 1
    if not entries:
        return []
    return [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]


def _chunk_text(text: str, target_chars: int = 6000) -> list[str]:
    """Split ``text`` into contiguous chunks near ``target_chars`` on line breaks.

    The transcript text uses ``\n`` line breaks (one per segment), so we split
    on line boundaries to keep each chunk coherent (a whole set of segments).
    """
    if target_chars <= 0 or len(text) <= target_chars:
        return [text] if text.strip() else []
    lines = text.splitlines()
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        line_len = len(line) + 1  # +1 for the '\n' we re-add on join
        if buf and size + line_len > target_chars:
            chunks.append("\n".join(buf))
            buf = []
            size = 0
        buf.append(line)
        size += line_len
    if buf:
        chunks.append("\n".join(buf))
    return [c for c in chunks if c.strip()]


def analyze(
    prompt_template: str,
    transcript: str,
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    entries: list | None = None,
    frames_dir=None,
    use_vision: bool = False,
    max_frames: int = 50,
    chunk_size: int = 8,
    chunk_chars: int = 6000,
    context_turns: int = 3,
    on_progress=None,
) -> str:
    """Run an analysis, chunking long inputs with rolling-context map-reduce.

    Short inputs (one chunk) use a single ``chat_text`` / ``chat_vision`` call —
    identical to the old behavior, so existing single-call tests keep passing.

    Long inputs are split into contiguous chunks. The map phase is
    **rolling-context** (chat-style): each chunk's prompt carries forward the
    prior chunks' partial analyses as a running context block, so chunk N can
    refer back to speakers/decisions/entities/open threads established in chunks
    1..N-1 instead of analyzing in a vacuum. The reduce step then merges the
    (already coherent) partials into one final answer. With frames, each chunk
    carries only the frames for its own segments, so the vision model sees a
    small, coherent window of "these frames + this text" plus the running
    context instead of one giant blob.

    ``context_turns`` caps how many prior partials are injected as context for
    a given chunk (a sliding window — the last ``context_turns`` partials),
    bounding prompt growth on very long inputs. ``0`` disables the rolling
    context (each chunk analyzed independently), which is the old behavior.

    Built-in prompts (summary / actions / summary+actions / plan) route through
    ``MAP_PROMPT`` + ``SYNTH_PROMPT`` so the final answer has the same structure
    the non-chunked path would produce. Custom ``--prompt`` text is applied per
    chunk (with rolling context prepended for chunks after the first) and the
    per-chunk answers are merged with a generic reduce.

    ``on_progress(msg)``, if given, is called with a short status string before
    each map call and the reduce call so callers can surface progress.
    """
    built_in = _is_built_in_prompt(prompt_template)
    task = _task_label(prompt_template)

    # --- Vision path: chunk by entries so each chunk's frames stay local ---
    if use_vision and entries:
        chunks = chunk_entries(entries, chunk_size)
        # One chunk (or none): single call, preserves prior behavior.
        if len(chunks) <= 1:
            frame_paths = _frames_for_entries(entries, frames_dir)
            return chat_vision(
                prompt_template, transcript, frame_paths,
                base_url=base_url, model=model, api_key=api_key, max_frames=max_frames,
            )
        return _map_reduce_vision(
            chunks, task, built_in, prompt_template, max_frames,
            context_turns=context_turns,
            base_url=base_url, model=model, api_key=api_key,
            frames_dir=frames_dir, on_progress=on_progress,
        )

    # --- Text path: chunk the transcript string ---
    chunks = _chunk_text(transcript, target_chars=chunk_chars)
    if len(chunks) <= 1:
        return chat_text(
            prompt_template, transcript,
            base_url=base_url, model=model, api_key=api_key,
        )
    return _map_reduce_text(
        chunks, task, built_in, prompt_template,
        context_turns=context_turns,
        base_url=base_url, model=model, api_key=api_key, on_progress=on_progress,
    )


def _frames_for_entries(entries, frames_dir) -> list[Path]:
    """Resolve the frame image paths for a chunk of manifest entries."""
    if frames_dir is None:
        return []
    out: list[Path] = []
    for e in entries:
        frame = getattr(e, "frame", "")
        if frame:
            out.append(Path(frames_dir) / frame)
    return out


def _running_context(partials: list[str], context_turns: int) -> str:
    """Build the running-context block from prior partials (sliding window).

    Returns the filled ``_CONTEXT_BLOCK`` with the last ``context_turns``
    partials joined in order, or an empty string when there are no prior
    partials (chunk 1) or rolling context is disabled (``context_turns`` <= 0).
    """
    if context_turns <= 0 or not partials:
        return ""
    window = partials[-context_turns:]
    context = "\n\n".join(window)
    return _CONTEXT_BLOCK.replace("{context}", context)


def _map_reduce_vision(
    chunks, task, built_in, prompt_template, max_frames, *,
    context_turns=3, base_url, model, api_key, frames_dir, on_progress=None,
) -> str:
    n = len(chunks)
    partials: list[str] = []
    for k, chunk in enumerate(chunks, start=1):
        chunk_transcript = transcript_text(chunk)
        frames = _frames_for_entries(chunk, frames_dir)
        if on_progress:
            on_progress(f"analyzing chunk {k}/{n} ({len(chunk)} segments, {len(frames)} frames)")
        context_block = _running_context(partials, context_turns)
        if built_in:
            mp = (MAP_PROMPT
                  .replace("{task}", task)
                  .replace("{k}", str(k))
                  .replace("{n}", str(n))
                  .replace("{context_block}", context_block)
                  .replace("{transcript}", chunk_transcript))
        else:
            # Custom prompt: prepend the running context (no MAP_PROMPT wrapper).
            mp = context_block + prompt_template.replace("{transcript}", chunk_transcript)
        partial = chat_vision(
            mp, chunk_transcript, frames,
            base_url=base_url, model=model, api_key=api_key, max_frames=max_frames,
        )
        partials.append(f"### Part {k} of {n}\n{partial}")
    if on_progress:
        on_progress(f"synthesizing {n} partial analyses")
    reduce_prompt = (SYNTH_PROMPT if built_in else _CUSTOM_REDUCE_PROMPT)
    synth_prompt = (reduce_prompt
                    .replace("{task}", task)
                    .replace("{n}", str(n))
                    .replace("{partials}", "\n\n".join(partials)))
    return chat_text(
        synth_prompt, "",  # transcript placeholder not used by SYNTH_PROMPT
        base_url=base_url, model=model, api_key=api_key,
    )


def _map_reduce_text(
    chunks, task, built_in, prompt_template, *,
    context_turns=3, base_url, model, api_key, on_progress=None,
) -> str:
    n = len(chunks)
    partials: list[str] = []
    for k, chunk in enumerate(chunks, start=1):
        if on_progress:
            on_progress(f"analyzing chunk {k}/{n}")
        context_block = _running_context(partials, context_turns)
        if built_in:
            mp = (MAP_PROMPT
                  .replace("{task}", task)
                  .replace("{k}", str(k))
                  .replace("{n}", str(n))
                  .replace("{context_block}", context_block)
                  .replace("{transcript}", chunk))
        else:
            mp = context_block + prompt_template.replace("{transcript}", chunk)
        partial = chat_text(
            mp, chunk,
            base_url=base_url, model=model, api_key=api_key,
        )
        partials.append(f"### Part {k} of {n}\n{partial}")
    if on_progress:
        on_progress(f"synthesizing {n} partial analyses")
    reduce_prompt = (SYNTH_PROMPT if built_in else _CUSTOM_REDUCE_PROMPT)
    synth_prompt = (reduce_prompt
                    .replace("{task}", task)
                    .replace("{n}", str(n))
                    .replace("{partials}", "\n\n".join(partials)))
    return chat_text(
        synth_prompt, "",
        base_url=base_url, model=model, api_key=api_key,
    )


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