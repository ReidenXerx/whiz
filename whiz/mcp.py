"""``whiz mcp``: past calls for AI agents, over the Model Context Protocol.

Claude Desktop, Claude Code, Cursor and other MCP clients start this as a
subprocess and talk JSON-RPC over its stdin/stdout, one message per line
(the protocol's stdio transport). Everything is read-only and local: the
server reads the library (``whiz.library``) and the files each run wrote.
What an agent then does with a transcript — send it to its model provider —
is the agent's business, which is why the app says so where it offers this.

No SDK: the protocol surface needed here (initialize, ping, tools/list,
tools/call) is small, and whiz keeps its dependencies to ``rich``.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import unicodedata
from datetime import datetime
from typing import Any, Callable, TextIO

from whiz import __version__
from whiz import library as L

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = (
    "Whiz holds the user's transcribed calls and recordings, on their Mac. "
    "Start with list_calls (recent first) or search_calls (words said in any call); "
    "both give each call an id. Then get_transcript reads it (optionally a time range), "
    "get_summary gives the AI summary if one was made, and get_frame shows what was on "
    "screen at a moment of a screen recording. Times are hh:mm:ss from the start."
)

# A long call is ~10k words; this keeps a runaway transcript from flooding
# the agent's context. Narrow with start/end to read past it.
TRANSCRIPT_CHAR_CAP = 200_000


class ToolError(Exception):
    """A tool's failure, worded for the agent (and its user)."""


# ---------- formatting ----------

def clock(seconds: float) -> str:
    total = int(round(max(seconds, 0.0)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_time(value: Any, name: str) -> float:
    """``hh:mm:ss``, ``mm:ss`` or seconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    if 2 <= len(parts) <= 3 and all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
        return seconds
    raise ToolError(f"{name} must be a time like 00:12:30, 12:30 or 750, not {value!r}")


def _fold(text: str) -> str:
    """Case- and accent-insensitive form for matching."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _call_summary(call: L.Call) -> dict[str, Any]:
    return {
        "id": call.id,
        "title": call.title,
        "recorded": call.recorded().astimezone().strftime("%Y-%m-%d %H:%M"),
        "duration": clock(call.duration),
        "speakers": call.speakers(),
        "has_summary": call.has_summary,
        "has_frames": call.has_frames,
        "path": str(call.input),
    }


def _describe(info: dict[str, Any]) -> str:
    extras = [name for name, present in (("summary", info["has_summary"]), ("frames", info["has_frames"])) if present]
    speakers = ", ".join(info["speakers"]) or "speakers not told apart"
    return (f"{info['id']} · {info['title']} · {info['recorded']} · {info['duration']} · {speakers}"
            + (f" · has {' and '.join(extras)}" if extras else ""))


def _resolve(arguments: dict[str, Any]) -> L.Call:
    ref = str(arguments.get("call", "")).strip()
    if not ref:
        raise ToolError("call is required: an id from list_calls or search_calls")
    call = L.find(ref)
    if call is None:
        raise ToolError(f"No single call matches {ref!r}. Use list_calls or search_calls to find its id.")
    return call


def _limit(arguments: dict[str, Any], default: int, most: int) -> int:
    value = arguments.get("limit", default)
    try:
        return max(1, min(int(value), most))
    except (TypeError, ValueError):
        raise ToolError(f"limit must be a whole number, not {value!r}") from None


# ---------- tools ----------

Content = list[dict[str, Any]]


def tool_list_calls(arguments: dict[str, Any]) -> tuple[Content, Any]:
    limit = _limit(arguments, 20, 200)
    query = _fold(str(arguments.get("query", "")).strip())
    since = str(arguments.get("since", "")).strip()
    since_date = None
    if since:
        try:
            since_date = datetime.strptime(since, "%Y-%m-%d").date()
        except ValueError:
            raise ToolError(f"since must be a date like 2026-10-01, not {since!r}") from None
    found = []
    for call in L.calls():
        if since_date and call.recorded().astimezone().date() < since_date:
            continue
        if query and query not in _fold(call.title) and not any(query in _fold(s) for s in call.speakers()):
            continue
        found.append(_call_summary(call))
        if len(found) == limit:
            break
    if not found:
        text = "No calls match." if (query or since_date) else (
            "No calls yet. Recordings appear here once Whiz (or `whiz transcribe`) has transcribed them.")
    else:
        text = "\n".join(_describe(info) for info in found)
    return [{"type": "text", "text": text}], {"calls": found}


def tool_search_calls(arguments: dict[str, Any]) -> tuple[Content, Any]:
    raw = str(arguments.get("query", "")).strip()
    if not raw:
        raise ToolError("query is required: the words to look for")
    terms = [_fold(t) for t in raw.split() if t.strip()]
    speaker = _fold(str(arguments.get("speaker", "")).strip())
    limit = _limit(arguments, 30, 200)
    matches: list[dict[str, Any]] = []
    for call in L.calls():
        for line in call.lines():
            if speaker and speaker not in _fold(line.speaker):
                continue
            folded = _fold(line.text)
            if all(term in folded for term in terms):
                matches.append({"call": call.id, "title": call.title, "at": clock(line.start),
                                "speaker": line.speaker, "text": line.text})
                if len(matches) == limit:
                    break
        if len(matches) == limit:
            break
    if not matches:
        text = f"Nothing said matches {raw!r}" + (" from that speaker." if speaker else ".")
    else:
        text = "\n".join(f"{m['call']} · {m['title']} · [{m['at']}] {m['speaker'] or 'Speaker'}: {m['text']}"
                         for m in matches)
        if len(matches) == limit:
            text += f"\n(first {limit} matches; raise limit or narrow the query for more)"
    return [{"type": "text", "text": text}], {"matches": matches}


def tool_get_transcript(arguments: dict[str, Any]) -> tuple[Content, Any]:
    call = _resolve(arguments)
    start = parse_time(arguments["start"], "start") if arguments.get("start") not in (None, "") else None
    end = parse_time(arguments["end"], "end") if arguments.get("end") not in (None, "") else None
    lines = [line for line in call.lines()
             if (start is None or line.end >= start) and (end is None or line.start <= end)]
    info = _call_summary(call)
    header = _describe(info)
    body: list[str] = []
    size = 0
    cut = None
    for line in lines:
        row = f"[{clock(line.start)}] {line.speaker + ': ' if line.speaker else ''}{line.text}"
        if size + len(row) > TRANSCRIPT_CHAR_CAP:
            cut = line.start
            break
        body.append(row)
        size += len(row) + 1
    text = header + "\n\n" + ("\n".join(body) if body else "(nothing said in that range)")
    if cut is not None:
        text += f"\n\n(transcript continues; call again with start={clock(cut)})"
    return [{"type": "text", "text": text}], None


def tool_get_summary(arguments: dict[str, Any]) -> tuple[Content, Any]:
    call = _resolve(arguments)
    if not call.has_summary:
        raise ToolError(f"{call.title} has no AI summary. Read it with get_transcript instead.")
    text = call.analysis.read_text(encoding="utf-8", errors="replace")
    # The file is "# whiz analysis …", the model line, the prompt, then
    # "## Response": the agent wants the response.
    _, marker, response = text.partition("\n## Response\n")
    return [{"type": "text", "text": f"Summary of {call.title}:\n\n" + (response.strip() if marker else text)}], None


def tool_get_frame(arguments: dict[str, Any]) -> tuple[Content, Any]:
    call = _resolve(arguments)
    if "at" not in arguments:
        raise ToolError("at is required: the moment, like 00:12:30")
    at = parse_time(arguments["at"], "at")
    framed = [line for line in call.lines() if line.frame is not None and line.frame.exists()]
    if not framed:
        raise ToolError(f"{call.title} has no screen frames (it is audio, or was transcribed without them).")
    # The line on screen at that moment, else the nearest one.
    line = next((l for l in framed if l.start <= at <= l.end), None) or min(
        framed, key=lambda l: min(abs(l.start - at), abs(l.end - at)))
    data = base64.b64encode(line.frame.read_bytes()).decode("ascii")
    caption = f"{call.title} at [{clock(line.start)}] {line.speaker + ': ' if line.speaker else ''}{line.text}"
    return [{"type": "text", "text": caption}, {"type": "image", "data": data, "mimeType": "image/jpeg"}], None


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


_CALL = {"type": "string", "description": "The call's id (from list_calls or search_calls), its file path, or a unique part of its title."}
_READ_ONLY = {"readOnlyHint": True, "openWorldHint": False}

TOOLS: dict[str, tuple[dict[str, Any], Callable[[dict[str, Any]], tuple[Content, Any]]]] = {
    "list_calls": ({
        "name": "list_calls",
        "title": "List calls",
        "description": "The user's transcribed calls and recordings, most recent first: id, title, when, how long, who spoke, and whether there is an AI summary and screen frames.",
        "inputSchema": _schema({
            "query": {"type": "string", "description": "Only calls whose title or a speaker's name contains this."},
            "since": {"type": "string", "description": "Only calls recorded on or after this date, YYYY-MM-DD."},
            "limit": {"type": "integer", "description": "How many, 1-200 (default 20)."},
        }),
        "annotations": _READ_ONLY,
    }, tool_list_calls),
    "search_calls": ({
        "name": "search_calls",
        "title": "Search what was said",
        "description": "Find the lines, across every call, that contain all of the given words (case and accents ignored). Each match has the call id, the time and the speaker.",
        "inputSchema": _schema({
            "query": {"type": "string", "description": "Words that must all appear in the line."},
            "speaker": {"type": "string", "description": "Only lines from speakers whose name contains this."},
            "limit": {"type": "integer", "description": "How many matches, 1-200 (default 30)."},
        }, ["query"]),
        "annotations": _READ_ONLY,
    }, tool_search_calls),
    "get_transcript": ({
        "name": "get_transcript",
        "title": "Read a transcript",
        "description": "A call's transcript, one line per utterance as [hh:mm:ss] Speaker: text, optionally only between start and end.",
        "inputSchema": _schema({
            "call": _CALL,
            "start": {"type": "string", "description": "From this time, hh:mm:ss."},
            "end": {"type": "string", "description": "Up to this time, hh:mm:ss."},
        }, ["call"]),
        "annotations": _READ_ONLY,
    }, tool_get_transcript),
    "get_summary": ({
        "name": "get_summary",
        "title": "Read the AI summary",
        "description": "The AI summary Whiz made of a call (overview, decisions, action items), when one was made.",
        "inputSchema": _schema({"call": _CALL}, ["call"]),
        "annotations": _READ_ONLY,
    }, tool_get_summary),
    "get_frame": ({
        "name": "get_frame",
        "title": "See the screen at a moment",
        "description": "The screen frame Whiz captured for the line being said at a moment of a screen recording, as an image, with that line.",
        "inputSchema": _schema({
            "call": _CALL,
            "at": {"type": "string", "description": "The moment, hh:mm:ss."},
        }, ["call", "at"]),
        "annotations": _READ_ONLY,
    }, tool_get_frame),
}


# ---------- protocol ----------

def handle(message: Any) -> dict[str, Any] | None:
    """One JSON-RPC message in, its response out (None for a notification)."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return _error(message.get("id") if isinstance(message, dict) else None, -32600, "Invalid Request")
    method = message["method"]
    request_id = message.get("id")
    is_notification = "id" not in message
    params = message.get("params") or {}
    if is_notification:
        return None  # notifications/initialized, cancelled, …: nothing to answer
    try:
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
            return _result(request_id, {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "whiz", "title": "Whiz", "version": __version__},
                "instructions": INSTRUCTIONS,
            })
        if method == "ping":
            return _result(request_id, {})
        if method == "tools/list":
            return _result(request_id, {"tools": [spec for spec, _ in TOOLS.values()]})
        if method == "tools/call":
            name = params.get("name")
            if name not in TOOLS:
                return _error(request_id, -32602, f"Unknown tool: {name}")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return _error(request_id, -32602, "arguments must be an object")
            try:
                content, structured = TOOLS[name][1](arguments)
            except ToolError as exc:
                return _result(request_id, {"content": [{"type": "text", "text": str(exc)}], "isError": True})
            except OSError as exc:
                return _result(request_id, {"content": [{"type": "text", "text": f"Could not read the call's files: {exc}"}], "isError": True})
            result: dict[str, Any] = {"content": content}
            if structured is not None:
                result["structuredContent"] = structured
            return _result(request_id, result)
        return _error(request_id, -32601, f"Method not found: {method}")
    except Exception as exc:  # a bug must answer the request, not kill the server
        return _error(request_id, -32603, f"Internal error: {exc}")


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """Answer messages until the client closes stdin."""
    stdin = stdin or sys.stdin
    original = sys.stdout
    out = stdout or original
    # stdout carries the protocol and nothing else: anything else printing
    # (a warning, a library's chatter) goes to stderr, which clients log.
    sys.stdout = sys.stderr
    try:
        for raw in stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                response: Any = _error(None, -32700, "Parse error")
            else:
                if isinstance(message, list):  # a batch (2025-03-26 and older)
                    response = [r for r in (handle(m) for m in message) if r is not None] or None
                else:
                    response = handle(message)
            if response is not None:
                out.write(json.dumps(response, ensure_ascii=False) + "\n")
                out.flush()
    finally:
        sys.stdout = original
    return 0
