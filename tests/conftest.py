from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"


def _fake_output_paths() -> set[Path]:
    if not OUTPUT_ROOT.exists():
        return set()
    return {
        path
        for path in OUTPUT_ROOT.rglob("*")
        if any("fake" in part.lower() for part in path.relative_to(OUTPUT_ROOT).parts)
    }


def pytest_sessionstart(session):
    session.config._fake_output_paths_before = _fake_output_paths()


def pytest_sessionfinish(session, exitstatus):
    if exitstatus != pytest.ExitCode.OK:
        return

    before = getattr(session.config, "_fake_output_paths_before", set())
    created_paths = sorted(_fake_output_paths() - before, key=lambda path: len(path.parts), reverse=True)
    for path in created_paths:
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
