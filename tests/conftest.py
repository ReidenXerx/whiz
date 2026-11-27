"""pytest config — ensure the whiz package is importable from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

# Insert repo root so `import whiz` works without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest


@pytest.fixture(autouse=True)
def _private_library(tmp_path, monkeypatch):
    """Runs that finish add themselves to the library: keep every test's
    additions in its own temp file, never the developer's real one."""
    from whiz import library

    monkeypatch.setattr(library, "library_path", lambda: tmp_path / "library.jsonl")


@pytest.fixture(autouse=True)
def _no_language_detection(request, monkeypatch):
    """Per-stretch language detection runs whisper-cli on the audio; the
    transcribe tests feed fake audio to a stubbed whisper. Off for them,
    as if detection found nothing (whisper's own -l auto). Tests of the
    detection itself opt back in with @pytest.mark.language_detection."""
    if request.node.get_closest_marker("language_detection"):
        return
    from whiz import cli

    monkeypatch.setattr(cli, "_language_plan", lambda workspace, cmd, diar_segments: None)


def pytest_configure(config):
    config.addinivalue_line("markers", "language_detection: run the real per-stretch language plan")
