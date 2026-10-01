"""Behavioral tests: the evolution daemon MUST gate batch spawning on the
flatline halt state (and on explicit user pause).

Regression guard for the recurring "value written but never read" defect
class: `_maybe_apply_flatline_halt()` used to hardcode `return False`
("evolution never pauses") and `main()` ignored its return value, so the
daemon respawned flatlined batches indefinitely.

Red-check: if the gate line in scripts/continuous_evolution_daemon.py is
mutated back (e.g., `return False` hardcoded / main ignoring the result),
these tests fail.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.continuous_evolution_daemon as daemon  # noqa: E402


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    agents = tmp_path / ".agents"
    agents.mkdir()
    return tmp_path


def _write_flatline(project_root: Path, halted: bool) -> None:
    state = {
        "halted_at": "2026-09-09T02:20:25Z",
        "halted": halted,
        "report": {"halted": halted, "window": 40, "reasons": ["oscillation trap"]},
        "resume_requires": "development work; run scripts/flatline_guard.py --resume",
    }
    (project_root / ".agents" / "flatline_state.json").write_text(json.dumps(state), encoding="utf-8")


def _write_tick(project_root: Path, user_paused: bool) -> None:
    state = {
        "evolution_paused_user": user_paused,
        "loop_paused": user_paused,
        "status": "in_progress",
    }
    (project_root / ".agents" / "tick_state.json").write_text(json.dumps(state), encoding="utf-8")


def test_gate_returns_true_when_flatline_halted(project_root: Path, monkeypatch) -> None:
    """Halted flatline state MUST block batch spawning."""
    _write_flatline(project_root, halted=True)
    monkeypatch.setenv("EE_FLATLINE_GUARD", "1")
    monkeypatch.setattr(daemon, "PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(daemon, "AGENTS", str(project_root / ".agents"))

    assert daemon._maybe_apply_flatline_halt() is True


def test_gate_returns_false_when_no_halt(project_root: Path, monkeypatch) -> None:
    """Clean state MUST allow batch spawning."""
    _write_flatline(project_root, halted=False)
    monkeypatch.setenv("EE_FLATLINE_GUARD", "1")
    monkeypatch.setattr(daemon, "PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(daemon, "AGENTS", str(project_root / ".agents"))

    assert daemon._maybe_apply_flatline_halt() is False


def test_gate_honors_user_pause(project_root: Path, monkeypatch) -> None:
    """Explicit user pause (tick_state.evolution_paused_user) MUST block spawning."""
    _write_flatline(project_root, halted=False)
    _write_tick(project_root, user_paused=True)
    monkeypatch.setenv("EE_FLATLINE_GUARD", "1")
    monkeypatch.setattr(daemon, "PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(daemon, "AGENTS", str(project_root / ".agents"))

    assert daemon._maybe_apply_flatline_halt() is True


def test_gate_disabled_when_env_guard_off(project_root: Path, monkeypatch) -> None:
    """EE_FLATLINE_GUARD=0 disables enforcement (escape hatch)."""
    _write_flatline(project_root, halted=True)
    monkeypatch.setenv("EE_FLATLINE_GUARD", "0")
    monkeypatch.setattr(daemon, "PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(daemon, "AGENTS", str(project_root / ".agents"))

    assert daemon._maybe_apply_flatline_halt() is False


def test_wait_loop_blocks_while_halted_and_exits_after_resume(project_root: Path, monkeypatch) -> None:
    """The wait loop must stay blocked while halted, and return once the halt
    is cleared (resume path). Real threads + a tiny poll interval."""
    _write_flatline(project_root, halted=True)
    monkeypatch.setattr(daemon, "PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(daemon, "AGENTS", str(project_root / ".agents"))
    monkeypatch.setattr(daemon, "FLATLINE_POLL_SEC", 0.05)
    heartbeats: list[str] = []
    monkeypatch.setattr(daemon, "_heartbeat", lambda *a, **k: heartbeats.append(a[0]))
    monkeypatch.setattr(daemon, "_log", lambda *a, **k: None)

    import threading
    import time

    def _run() -> None:
        daemon._wait_while_evolution_paused_or_halted()

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.2)
    # While halted, the loop must still be blocked and heartbeating.
    assert t.is_alive(), "wait loop must remain blocked while halted"
    assert "evolution_paused" in heartbeats

    # Simulate resume: clear the flatline halt.
    _write_flatline(project_root, halted=False)
    t.join(timeout=5)
    assert not t.is_alive(), "wait loop must return once halt is cleared"
