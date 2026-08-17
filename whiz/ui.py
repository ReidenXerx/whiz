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

from rich.console import Console
from rich.table import Table
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
})
_console = Console(stderr=True, theme=_THEME, highlight=False)


def _is_tty() -> bool:
    return _console.is_terminal


def header(title: str, subtitle: str = "") -> None:
    """Print a compact branded banner. Suppressed entirely when piped."""
    if not _is_tty():
        if subtitle:
            print(f"{title} — {subtitle}", file=sys.stderr)
        else:
            print(title, file=sys.stderr)
        return
    _console.print(f"[whiz.brand]⚡ {title}[/]", end="")
    if subtitle:
        _console.print(f" [whiz.dim]{subtitle}[/]")
    else:
        _console.print()


def phase(label: str) -> None:
    """Print a colored phase-step line, e.g. '▸ diarizing'."""
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


def speaker_label_line(label: str) -> None:
    """Print a speaker label with its palette color and a 'said:' suffix."""
    hex_color = speaker_palette(label)
    _console.print(f"  [{hex_color}]●[/] [bold]{label}[/] said:")


def tally(counts: list[tuple[str, int]]) -> None:
    """Render the speaker tally with per-speaker colors matching the HTML palette."""
    if not counts:
        return
    _console.print(f"[whiz.kv.label]{'Speakers':<7}[/] [bold]{len(counts)}[/] detected")
    for label, n in counts:
        hex_color = speaker_palette(label)
        _console.print(
            f"    [{hex_color}]●[/] [bold]{label}[/] [whiz.muted]{n} segments[/]"
        )


def summary(items: list[str], title: str = "Done") -> None:
    """Final footer: a check mark per written artifact. Suppressed when empty."""
    if not items:
        return
    _console.print(f"[whiz.ok]✓ {title}[/]")
    for it in items:
        _console.print(f"  [whiz.ok]·[/] {it}")


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