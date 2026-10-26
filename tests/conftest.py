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
