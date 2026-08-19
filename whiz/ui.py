"""whiz UI — styled terminal output helpers (rich-backed, plain-text fallback).

All user-facing CLI output goes through this module so the look is consistent
and degrades to clean plain text when stderr isn't a TTY (logs/redirects stay
escape-free). Rich is the only runtime dependency.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from whiz.merge import speaker_palette

# A single Console bound to stderr; rich auto-detects isatty, so piped/redirected
# output has no ANSI escapes. force_terminal is left to auto so color still
# shows when a pager like `less -R` is in the chain.
_THEME = Theme({
    "whiz.brand": "bold cyan",
    "whiz.dim": "dim",
    "whiz.kv.label": "bold",
    "whiz.phase": "bold magenta",
    "whiz.ok": "bold green",
    "whiz.warn": "bold yellow",
    "whiz.hint": "cyan",
    "whiz.info": "blue",
    "whiz.timestamp": "dim cyan",
    "whiz.muted": "dim",
    "whiz.rule": "dim cyan",
})
_console = Console(stderr=True, theme=_THEME, highlight=False)


def _is_tty() -> bool:
    return _console.is_terminal


def header(title: str, subtitle: str = "") -> None:
    """Print a compact branded banner inside a subtle bordered panel.

    Suppresses the panel entirely when piped (plain ``title — subtitle``).
    """
    if not _is_tty():
        if subtitle:
            print(f"{title} — {subtitle}", file=sys.stderr)
        else:
            print(title, file=sys.stderr)
        return
    # Left: ⚡ brand + title. Right: subtitle, right-aligned on the same line.
    left = Text.assemble(("⚡ ", "whiz.brand"), (title, "bold"))
    if subtitle:
        # Use a two-column table so the subtitle sits flush right.
        t = Table.grid(expand=True)
        t.add_column(ratio=1)
        t.add_column(justify="right")
        t.add_row(left, Text(subtitle, "whiz.dim"))
        body = t
    else:
        body = left
    _console.print(Panel(
        body,
        border_style="whiz.rule",
        padding=(0, 1),
        expand=True,
    ))


def rule() -> None:
    """Print a thin horizontal rule to separate major phases. No-op when piped."""
    if not _is_tty():
        return
    _console.print(Rule(style="whiz.rule"))


def phase(label: str) -> None:
    """Print a phase-step line, e.g. '▸ diarizing', preceded by a soft rule."""
    if _is_tty():
        _console.print(Rule(style="whiz.rule"))
    _console.print(f"[whiz.phase]▸[/] [bold]{label}[/]")


def kv(label: str, value: Any) -> None:
    """Print an aligned 'label: value' line with a styled label."""
    _console.print(f"[whiz.kv.label]{label:<7}[/] {value}")


def status(msg: str, kind: str = "info", detail: str | None = None) -> None:
    """Print a colored status line.

    kind is one of: ok, warn, hint, info. ``detail`` (if given) is printed on a
    dimmed indented follow-up line — used for remediation hints under warnings.
    """
    style = {
        "ok": "whiz.ok",
        "warn": "whiz.warn",
        "hint": "whiz.hint",
        "info": "whiz.info",
    }.get(kind, "whiz.info")
    _console.print(f"[{style}]{msg}[/]")
    if detail:
        _console.print(f"    [whiz.muted]{detail}[/]")


def info(msg: str) -> None:
    """Plain info line (not a warning). Kept for non-status informational prints."""
    _console.print(f"[whiz.info]{msg}[/]")


def muted(msg: str) -> None:
    """A dimmed line, e.g. 'removed intermediate foo.wav'."""
    _console.print(f"[whiz.muted]{msg}[/]")


def note(msg: str) -> None:
    """A neutral note, no color emphasis."""
    _console.print(msg)


def wrote(label: str, path: Any) -> None:
    """An aligned artifact-written line: a green check + label, then the path.

    Renders as a two-line block so paths line up regardless of label length:
        ✓ Wrote labeled SRT
          recording.speakers.srt
    """
    mark = Text("✓ ", style="whiz.ok")
    lbl = Text(label, style="whiz.ok")
    if _is_tty():
        t = Table.grid(expand=False)
        t.add_column()
        t.add_row(Text.assemble(mark, lbl))
        t.add_row(Text("  " + str(path), style="whiz.dim"))
        _console.print(t)
    else:
        print(f"✓ {label}: {path}", file=sys.stderr)


def speaker_label_line(label: str) -> None:
    """Print a speaker label with its palette color and a 'said:' suffix."""
    hex_color = speaker_palette(label)
    _console.print(f"  [{hex_color}]●[/] [bold]{label}[/] said:")


def tally(counts: list[tuple[str, int]]) -> None:
    """Render the speaker tally with per-speaker colors matching the HTML palette."""
    if not counts:
        return
    _console.print(f"[whiz.kv.label]{'Speakers':<7}[/] [bold]{len(counts)}[/] detected")
    # Align speaker names so the segment counts line up.
    width = max((len(label) for label, _ in counts), default=0)
    for label, n in counts:
        hex_color = speaker_palette(label)
        _console.print(
            f"    [{hex_color}]●[/] [bold]{label:<{width}}[/]  [whiz.muted]{n} segments[/]"
        )


def summary(items: list[str], title: str = "Done") -> None:
    """Final footer: a check-marked panel of written artifacts. Suppressed when empty."""
    if not items:
        return
    if not _is_tty():
        print(f"✓ {title}", file=sys.stderr)
        for it in items:
            print(f"  · {it}", file=sys.stderr)
        return
    body_lines = [Text(f"  · {it}", style="whiz.dim") for it in items]
    head = Text.assemble(("✓ ", "whiz.ok"), (f"{title} · {len(items)} file(s)", "whiz.ok"))
    content = Group(head, *body_lines)
    _console.print(Panel(
        content,
        border_style="whiz.rule",
        padding=(0, 1),
        expand=True,
    ))


def table(
    title: str | None,
    columns: list[tuple[str, str]],  # (header, justify: left|right|center)
    rows: list[list[Any]],
) -> None:
    """Render a table; rich handles plain-text fallback (boxes stripped) when piped."""
    t = Table(title=title, show_header=True, header_style="bold", expand=False)
    for header, justify in columns:
        t.add_column(header, justify=justify)
    for row in rows:
        t.add_row(*[str(c) for c in row])
    _console.print(t)


@contextmanager
def spinner(label: str) -> Iterator:
    """A live spinner for a long-running phase.

    On a TTY: a rich ``Status`` spinner with the given label, updated via the
    yielded ``update(msg)`` callable (e.g. progress from the map-reduce loop).
    When piped/non-TTY: prints the label once and ``update`` becomes a no-op
    (the underlying work still streams its own lines where applicable).
    """
    if not _is_tty():
        _console.print(f"[whiz.phase]▸[/] [bold]{label}[/]")

        def update(msg: str) -> None:
            pass

        yield update
        return
    with _console.status(f"[whiz.phase]▸[/] {label}", spinner="dots") as status:
        def update(msg: str) -> None:
            status.update(f"[whiz.phase]▸[/] {label} · {msg}")

        yield update


@contextmanager
def streaming_progress(_cmd: list[str]) -> Iterator:
    """Yield a writer for streamed subprocess output with a styled elapsed prefix.

    The yielded callable takes ``(line, elapsed)`` and prints the line with a
    dimmed timestamp prefix. Used by the whisper-cli streaming loop. When piped,
    output stays plain. The ``_cmd`` arg is accepted for API symmetry with a
    future live-region version but isn't used here.
    """
    def write(line: str, elapsed: float) -> None:
        # line includes its trailing newline from the subprocess
        prefix = f"[{_fmt_elapsed(elapsed)}] "
        if _is_tty():
            _console.print(f"[whiz.timestamp]{prefix}[/][whiz.muted]{line.rstrip()}[/]")
        else:
            sys.stderr.write(prefix + line)
            if not line.endswith("\n"):
                sys.stderr.write("\n")
            sys.stderr.flush()
    yield write


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as M:SS or H:MM:SS."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"