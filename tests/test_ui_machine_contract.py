"""The piped-output contract the native UI parses.

`ui.py` deliberately degrades to escape-free plain text when stderr is not a
TTY — that branch exists so logs and redirects stay clean. The macOS app turns
that into an interface: it runs the `whiz` CLI as a subprocess and reads the
three line shapes below to drive its progress view, to learn which artifacts
a run produced, and to answer the run's speaker-naming questions.

Nothing pinned those shapes before. A reasonable-looking edit to `ui.wrote`
(dropping the colon, reordering label and path, switching the marker) would
leave every Python test green and silently break the UI, which has no way
to notice beyond showing no artifacts. These tests are the pin.

The contract is deliberately narrow — three prefixes and a separator — so
the prose inside a label stays free to change:

    ▸ <phase label>
    ✓ <artifact label>: <path>
    ? speaker-name: <label> | <quote>[ | <suggestion>]
"""

from __future__ import annotations

import io
import os
import re
import sys

import pytest

from whiz import cli
from whiz import ui


class _Piped:
    """Reads back everything a subprocess would see on stderr.

    Two sinks, and both are needed: `phase`/`status` go through the
    module-level rich Console, while `wrote`'s non-TTY branch prints straight
    to `sys.stderr`. Reassigning `sys.stderr` does not work here — pytest has
    already replaced it for capture and takes it back — so the Console is
    redirected to a buffer and the raw prints are read through `capsys`.
    """

    def __init__(self, buffer, capsys):
        self._buffer = buffer
        self._capsys = capsys

    def read(self) -> str:
        return self._buffer.getvalue() + self._capsys.readouterr().err


@pytest.fixture
def piped(monkeypatch, capsys):
    """Render ui output as a non-TTY."""
    buffer = io.StringIO()
    monkeypatch.setattr(ui, "_is_tty", lambda: False)
    from rich.console import Console

    monkeypatch.setattr(ui, "_console", Console(file=buffer, force_terminal=False))
    return _Piped(buffer, capsys)


def test_wrote_emits_marker_label_colon_path(piped):
    """`✓ <label>: <path>` — how the UI discovers a run's artifacts."""
    ui.wrote("Wrote labeled SRT", "/tmp/recording.speakers.srt")
    line = piped.read().strip()
    assert line.startswith("✓ "), f"artifact marker changed: {line!r}"
    assert ": " in line, f"label/path separator changed: {line!r}"
    label, path = line[2:].split(": ", 1)
    assert label == "Wrote labeled SRT"
    assert path == "/tmp/recording.speakers.srt"


def test_wrote_is_a_single_line_when_piped(piped):
    """The TTY branch renders a two-line aligned block; piped must not.

    A parser reading line-by-line would otherwise take the path as a phase.
    """
    ui.wrote("Wrote HTML transcript", "/tmp/a.speakers.html")
    assert len(piped.read().strip().splitlines()) == 1


def test_phase_emits_marker_then_label(piped):
    """`▸ <label>` — how the UI shows which stage is running."""
    ui.phase("diarizing")
    line = piped.read().strip()
    assert line.startswith("▸ "), f"phase marker changed: {line!r}"
    assert line[2:] == "diarizing"


def test_piped_output_carries_no_ansi_escapes(piped):
    """Escape-free is the whole reason the non-TTY branch exists.

    Escape sequences would land in the UI's log pane verbatim and corrupt any
    prefix match.
    """
    ui.phase("transcribing")
    ui.wrote("Wrote analysis", "/tmp/a.analysis.md")
    ui.status("degraded: no speaker labels", kind="warn")
    assert "\x1b[" not in piped.read()


def test_paths_with_spaces_survive_the_separator(piped):
    """Split on the FIRST ': ' only — macOS paths routinely contain spaces,
    and 'Wrote X: /Users/a b/My Video.srt' must not lose the tail."""
    ui.wrote("Wrote dialogue TXT", "/Users/a b/My Recording.speakers.txt")
    line = piped.read().strip()
    _, path = line[2:].split(": ", 1)
    assert path == "/Users/a b/My Recording.speakers.txt"


ARTIFACT_LABELS = {
    "Wrote labeled SRT",
    "Wrote dialogue TXT",
    "Wrote HTML transcript",
    "Wrote frames manifest",
    "Wrote analysis",
}


def test_every_artifact_label_still_exists_in_the_cli():
    """The labels the UI maps to artifact kinds.

    Not a style rule — the UI keys off these strings to decide what to offer
    ("Open transcript", "Open analysis"). A rename here is a UI change, and
    this test is where that gets noticed.
    """
    source = ui.__file__.rsplit("/", 1)[0] + "/cli.py"
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    emitted = set(re.findall(r'ui\.wrote\(\s*"([^"]+)"', text))
    missing = ARTIFACT_LABELS - emitted
    assert not missing, (
        f"artifact labels the UI parses no longer emitted by cli.py: {sorted(missing)}"
    )


def test_transcribe_parses_the_argv_the_ui_sends():
    """The flags the native UI builds must exist on the real parser.

    The Swift side pins its argv against stub scripts, which cannot notice a
    flag the CLI does not define: the first revision of the UI sent
    `--ai-model` before `whiz transcribe` had it, and every UI run with an
    analysis model died at spawn with argparse exit 2. Only the real parser
    is the contract.
    """
    from whiz.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "transcribe", "/tmp/a.mp4",
            "--language", "ru",
            "--speakers", "3",
            "--no-screenshots",
            "--analyze",
            "--ai-model", "qwen3.5:9b",
        ]
    )
    assert args.language == "ru"
    assert args.speakers == 3
    assert args.no_screenshots is True
    assert args.analyze is True
    assert args.ai_model == "qwen3.5:9b"


def test_speaker_prompt_emits_question_marker_label_quote_suggestion(piped):
    """`? speaker-name: <label> | <quote>[ | <suggestion>]` — how the UI learns
    the run wants a speaker named, and what it would suggest.

    Third machine line after ▸ and ✓. The suggestion is a voice-profile
    auto-match: the UI pre-fills it, and writing it back is a human
    confirmation (M3 provenance).
    """
    ui.speaker_prompt("Speaker A", "hello world", "Alice")
    line = piped.read().strip()
    assert line.startswith("? speaker-name: "), f"prompt marker changed: {line!r}"
    body = line[len("? speaker-name: "):]
    parts = body.split(" | ")
    assert parts[0] == "Speaker A"
    assert parts[1] == "hello world"
    assert parts[2] == "Alice"


def test_speaker_prompt_without_suggestion_has_no_third_segment(piped):
    """An unknown speaker is two segments — an empty trailing ' | ' would
    parse as a suggestion of "".
    """
    ui.speaker_prompt("Speaker B", "some words", None)
    body = piped.read().strip()[len("? speaker-name: "):]
    assert body == "Speaker B | some words"


def test_speaker_prompt_substitutes_the_separator_inside_fields(piped):
    """A quote containing the separator itself cannot tear the line's field
    structure: every field has ' | ' substituted with the lookalike U+01C0
    stroke (visually identical to a pipe), so the UI's split — label before
    the first separator, suggestion after the last — is exact for any
    speech. The exact body is pinned: this is the emit side, and the literal
    IS the contract.
    """
    ui.speaker_prompt("Speaker A", "he said ' | ' out loud", "Bob")
    body = piped.read().strip()[len("? speaker-name: "):]
    assert body == "Speaker A | he said ' ǀ ' out loud | Bob"


def test_prompted_name_answer_flows_through_stdin_pipe(monkeypatch):
    """The piped prompt's whole reason to exist: a UI holding the pipe answers
    it by writing one line per question, exactly as the macOS naming bar does
    — through the REAL input() and a REAL pipe, not a monkeypatched stand-in
    (GP-1: a claim from reading is unverified).
    """
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=2.0, text="hello there"), "Speaker A"),
        (cli.MR.WhisperSeg(start=2.0, end=4.0, text="general kenobi"), "Speaker B"),
    ]

    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
        # Non-TTY stdin is the machine world; the `piped` fixture covers
        # ui.speaker_prompt's renderer, this drives the prompt LOOP.
        monkeypatch.setattr(ui, "_is_tty", lambda: False)
        os.write(write_fd, b"Alice\n")
        os.write(write_fd, b"\n")  # second speaker: Enter keeps the default
        os.close(write_fd)

        name_map = cli._prompt_speaker_names(merged)
    finally:
        sys.stdin.close()

    assert name_map == {"Speaker A": "Alice"}


def test_piped_enter_confirms_the_suggestion(monkeypatch):
    """M3 provenance over the pipe: an empty answer CONFIRMS the suggested
    name — pressing Enter on a voice-profile match is a human confirmation,
    exactly as on a TTY. This is the upgrade path auto-matches rely on.
    """
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=2.0, text="hello there"), "Speaker A"),
    ]
    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
        monkeypatch.setattr(ui, "_is_tty", lambda: False)
        os.write(write_fd, b"\n")  # just Enter
        os.close(write_fd)

        name_map = cli._prompt_speaker_names(
            merged, default_names={"Speaker A": "Alice"})
    finally:
        sys.stdin.close()

    assert name_map == {"Speaker A": "Alice"}


def test_decline_sentinel_beats_the_suggestion(monkeypatch):
    """'-' declines: the answer is None — the signal the caller must read as
    "drop any name this label already has". Without it the GUI could not
    reject a WRONG voice-profile auto-match — every 'no name' answer would
    confirm it, merging the wrong name into the profile as user-confirmed
    where it would stick for every future run.
    """
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=2.0, text="hello there"), "Speaker A"),
    ]
    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
        monkeypatch.setattr(ui, "_is_tty", lambda: False)
        os.write(write_fd, b"-\n")
        os.close(write_fd)

        name_map = cli._prompt_speaker_names(
            merged, default_names={"Speaker A": "WrongName"})
    finally:
        sys.stdin.close()

    assert name_map == {"Speaker A": None}


def test_piped_prompt_writes_no_stdout_echo(monkeypatch, capsys):
    """The deadlock half of the contract: with piped stdin, the prompt loop
    must write its prompt text NOWHERE. input() echoes a non-empty prompt to
    stdout — flushed, without a trailing newline — and the UI merges stdout
    and stderr into one stream, so the echo would glue itself onto the next
    ? speaker-name: machine line and the run would block on an answer the UI
    never saw. On a TTY the echo IS the prompt; over a pipe the machine line
    is.
    """
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=2.0, text="hello there"), "Speaker A"),
    ]

    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
        monkeypatch.setattr(ui, "_is_tty", lambda: False)
        os.close(write_fd)  # EOF at the first question: the loop breaks

        cli._prompt_speaker_names(merged)
    finally:
        sys.stdin.close()

    captured = capsys.readouterr()
    assert "Name for Speaker A" not in captured.out, (
        "piped stdin must not echo the prompt to stdout — it corrupts the merged stream"
    )
    assert "Name for Speaker A" not in captured.err


def test_prompt_loop_ends_on_eof_over_pipe(monkeypatch):
    """The UI went away (stopped run, killed app): its end of the pipe closes
    and the CLI reads EOF mid-question. The loop must end — never hang — with
    no half-answered state.
    """
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=2.0, text="hello there"), "Speaker A"),
        (cli.MR.WhisperSeg(start=2.0, end=4.0, text="general kenobi"), "Speaker B"),
    ]

    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
        monkeypatch.setattr(ui, "_is_tty", lambda: False)
        os.close(write_fd)  # EOF at the first question

        name_map = cli._prompt_speaker_names(merged)
    finally:
        sys.stdin.close()

    assert name_map == {}
