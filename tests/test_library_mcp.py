"""The library of transcribed calls, and the MCP server that reads it."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from whiz import library as L
from whiz import mcp


SRT = (
    "1\n00:00:01,000 --> 00:00:04,000\nAlice: We ship the Record button on Friday.\n\n"
    "2\n00:00:05,000 --> 00:00:09,500\nBob: Then the MCP server, and search after that.\n\n"
    "3\n00:01:00,000 --> 00:01:30,000\nAlice: Résumé: Friday.\n"
)


def make_call(folder: Path, stem: str = "Standup", srt: str = SRT) -> Path:
    recording = folder / f"{stem}.m4a"
    recording.write_bytes(b"")
    Path(str(folder / stem) + ".speakers.srt").write_text(srt, encoding="utf-8")
    return recording


@pytest.fixture
def library(tmp_path):
    """The conftest's private library, with one call in it."""
    recording = make_call(tmp_path)
    L.remember(recording, recording.with_suffix(""))
    return tmp_path


# ---------- library ----------

def test_parses_labeled_srt():
    lines = L.parse_labeled_srt(SRT)
    assert [(l.start, l.end, l.speaker) for l in lines] == [
        (1.0, 4.0, "Alice"), (5.0, 9.5, "Bob"), (60.0, 90.0, "Alice")]
    # Only the first ": " separates the speaker; the rest is what was said.
    assert lines[2].text == "Résumé: Friday."


def test_remember_needs_a_transcript(tmp_path):
    bare = tmp_path / "nothing.m4a"
    bare.write_bytes(b"")
    assert L.remember(bare, bare.with_suffix("")) is None
    assert L.calls() == []


def test_newest_record_wins_and_forget_hides(tmp_path, library):
    recording = tmp_path / "Standup.m4a"
    L.remember(recording, recording.with_suffix(""))  # a re-run
    assert len(L.calls()) == 1
    L.forget(recording)
    assert L.calls() == []
    L.remember(recording, recording.with_suffix(""))  # transcribed again
    assert len(L.calls()) == 1


def test_deleted_transcripts_leave_the_library(tmp_path, library):
    Path(str(tmp_path / "Standup") + ".speakers.srt").unlink()
    assert L.calls() == []


def test_torn_lines_cost_only_themselves(tmp_path, library):
    with L.library_path().open("a", encoding="utf-8") as handle:
        handle.write('{"input": "/half\n')
    assert [c.title for c in L.calls()] == ["Standup"]


def test_find_by_id_path_or_unique_title(tmp_path, library):
    other = make_call(tmp_path, "Standup retro")
    L.remember(other, other.with_suffix(""))
    call = next(c for c in L.calls() if c.title == "Standup")
    assert L.find(call.id) == call or L.find(call.id).input == call.input
    assert L.find(str(call.input)).input == call.input
    assert L.find("retro").title == "Standup retro"
    assert L.find("standup") is None, "two titles match: ambiguous, not a guess"


def test_speakers_by_talk_time(library):
    call = L.calls()[0]
    assert call.speakers() == ["Alice", "Bob"]
    assert call.duration == 90.0


def test_frames_manifest_is_preferred(tmp_path):
    recording = tmp_path / "Demo.mov"
    recording.write_bytes(b"")
    frames = tmp_path / "Demo.frames"
    frames.mkdir()
    (frames / "0001.jpg").write_bytes(b"\xff\xd8jpeg")
    (tmp_path / "Demo.frames.json").write_text(json.dumps({
        "version": 1, "frames_dir": "Demo.frames", "count": 1,
        "segments": [{"index": 1, "start": 2.0, "end": 6.0, "speaker": "Vadim",
                      "text": "Here is the dashboard.", "frame": "0001.jpg"}],
    }), encoding="utf-8")
    call = L.remember(recording, recording.with_suffix(""))
    assert call is not None and call.has_frames
    assert call.lines()[0].frame == frames / "0001.jpg"


def test_discover_finds_an_earlier_runs_whisper_json(tmp_path):
    recording = tmp_path / "memo.m4a"
    recording.write_bytes(b"")
    (tmp_path / "memo.wav.json").write_text('{"transcription": []}', encoding="utf-8")
    assert L.discover(recording) == (recording.resolve().with_suffix(""), (tmp_path / "memo.wav.json").resolve())


# ---------- MCP ----------

def rpc(*messages: dict) -> list[dict]:
    out = io.StringIO()
    mcp.serve(io.StringIO("".join(json.dumps(m) + "\n" for m in messages)), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def call_tool(name: str, **arguments) -> dict:
    [response] = rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": name, "arguments": arguments}})
    return response["result"]


def test_handshake_and_tool_list():
    responses = rpc(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
    )
    assert [r["id"] for r in responses] == [1, 2, 3], "notifications get no answer"
    assert responses[0]["result"]["protocolVersion"] == "2025-03-26"
    assert responses[0]["result"]["capabilities"] == {"tools": {"listChanged": False}}
    tools = responses[1]["result"]["tools"]
    assert [t["name"] for t in tools] == ["list_calls", "search_calls", "get_transcript", "get_summary", "get_frame"]
    assert all(t["annotations"]["readOnlyHint"] for t in tools)
    assert responses[2]["result"] == {}


def test_unknown_version_gets_ours_and_bad_input_gets_errors():
    [init] = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}})
    assert init["result"]["protocolVersion"] == mcp.SUPPORTED_VERSIONS[0]
    out = io.StringIO()
    mcp.serve(io.StringIO("not json\n" + json.dumps({"jsonrpc": "2.0", "id": 9, "method": "nope"}) + "\n"), out)
    errors = [json.loads(l)["error"]["code"] for l in out.getvalue().splitlines()]
    assert errors == [-32700, -32601]


def test_stdout_carries_only_the_protocol(capsys):
    out = io.StringIO()

    def chatty(arguments):
        print("a library printing to stdout")
        return [{"type": "text", "text": "ok"}], None

    spec = mcp.TOOLS["list_calls"][0]
    original = mcp.TOOLS["list_calls"]
    mcp.TOOLS["list_calls"] = (spec, chatty)
    try:
        mcp.serve(io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                          "params": {"name": "list_calls"}}) + "\n"), out)
    finally:
        mcp.TOOLS["list_calls"] = original
    assert json.loads(out.getvalue())["result"]["content"][0]["text"] == "ok"
    assert "a library printing" in capsys.readouterr().err


def test_list_and_search(library):
    listed = call_tool("list_calls")
    assert listed["structuredContent"]["calls"][0]["speakers"] == ["Alice", "Bob"]
    assert "Standup" in listed["content"][0]["text"]
    assert "No calls match" in call_tool("list_calls", query="retro")["content"][0]["text"]

    found = call_tool("search_calls", query="mcp SERVER")
    assert [m["at"] for m in found["structuredContent"]["matches"]] == ["00:00:05"]
    # Accents and case are ignored; every word must appear.
    assert len(call_tool("search_calls", query="resume friday")["structuredContent"]["matches"]) == 1
    assert call_tool("search_calls", query="friday", speaker="bob")["structuredContent"]["matches"] == []


def test_transcript_range_summary_and_frames(tmp_path, library):
    text = call_tool("get_transcript", call="standup", start="00:00:05", end="1:00")["content"][0]["text"]
    assert "[00:00:01]" not in text and "[00:00:05] Bob:" in text and "[00:01:00] Alice:" in text

    missing = call_tool("get_summary", call="standup")
    assert missing["isError"] and "no AI summary" in missing["content"][0]["text"]
    Path(str(tmp_path / "Standup") + ".analysis.md").write_text(
        "# whiz analysis\n\n## Prompt\n\nignored\n\n## Response\n\nShip Friday.\n", encoding="utf-8")
    assert call_tool("get_summary", call="standup")["content"][0]["text"].endswith("Ship Friday.")

    no_frames = call_tool("get_frame", call="standup", at="0:02")
    assert no_frames["isError"]


def test_frame_is_the_line_on_screen(tmp_path):
    test_frames_manifest_is_preferred(tmp_path)
    result = call_tool("get_frame", call="demo", at="00:00:04")
    assert result["content"][0]["text"].endswith("Vadim: Here is the dashboard.")
    assert result["content"][1]["type"] == "image" and result["content"][1]["mimeType"] == "image/jpeg"


def test_tool_errors_are_results_not_crashes(library):
    assert call_tool("get_transcript", call="nobody")["isError"]
    assert call_tool("get_transcript", call="standup", start="noon")["isError"]
    assert call_tool("search_calls")["isError"]
    [unknown] = rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "rm_rf"}})
    assert unknown["error"]["code"] == -32602


def test_time_parsing():
    assert mcp.parse_time("01:02:03", "t") == 3723
    assert mcp.parse_time("2:30", "t") == 150
    assert mcp.parse_time(75, "t") == 75
    with pytest.raises(mcp.ToolError):
        mcp.parse_time("1:2:3:4", "t")
