"""The piped-output contract the native UI parses.

`ui.py` deliberately degrades to escape-free plain text when stderr is not a
TTY — that branch exists so logs and redirects stay clean. The macOS app turns
that into an interface: it runs the `whiz` CLI as a subprocess and reads the
two line shapes below to drive its progress view and to learn which artifacts a
run produced.

Nothing pinned those shapes before. A reasonable-looking edit to `ui.wrote`
(dropping the colon, reordering label and path, switching the marker) would
leave every Python test green and silently break the UI, which has no way to
notice beyond showing no artifacts. These tests are the pin.

The contract is deliberately narrow — two prefixes and a separator — so the
prose inside a label stays free to change:

    ▸ <phase label>
    ✓ <artifact label>: <path>
"""

from __future__ import annotations

import io
import re

import pytest

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
