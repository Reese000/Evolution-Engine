"""Regression: pytest must never write the production phase-timing log.

EE-W7 / t_526975b3 (item 2). ``evolution.phase_timing.default_log_path()``
falls back to ``<project root>/generation_phase_timing.jsonl``, so engine-running
tests used to append their cycles (1 / 99999) to the production rolling log.
``tests/conftest.py`` now exports ``EE_GEN_TIMING_LOG`` to a per-session temp
file for the whole pytest session; these tests pin that behaviour down.

The production log is append-only history -- nothing here truncates, rewrites
or rolls it.  The guard assertions run *before* any write so a broken redirect
fails the test instead of polluting the production log.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from evolution.phase_timing import PhaseTimer, default_log_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_LOG = PROJECT_ROOT / "generation_phase_timing.jsonl"
_PRODUCTION_ABS = os.path.abspath(str(PRODUCTION_LOG))


def _redirected_path() -> str:
    """Where the phase-timing log resolves to under this pytest session."""
    return os.path.abspath(default_log_path(str(PROJECT_ROOT)))


def test_pytest_session_exports_phase_timing_log_override() -> None:
    """conftest must export EE_GEN_TIMING_LOG for the whole pytest session."""
    exported = os.environ.get("EE_GEN_TIMING_LOG")
    assert exported, (
        "tests/conftest.py must set EE_GEN_TIMING_LOG so engine-running tests "
        "cannot append to the production generation_phase_timing.jsonl"
    )
    assert os.path.isabs(exported)


def test_resolved_log_path_is_not_the_production_log() -> None:
    """default_log_path() must not resolve to the repo-root production log."""
    resolved = _redirected_path()
    assert resolved != _PRODUCTION_ABS, f"phase-timing log still resolves to the production log: {resolved}"


def test_resolved_log_path_lives_outside_the_project_root() -> None:
    """The redirected log must not sit anywhere under the checkout."""
    root_prefix = os.path.abspath(str(PROJECT_ROOT)) + os.sep
    resolved = _redirected_path()
    assert not resolved.startswith(root_prefix), f"phase-timing log redirected inside the project root: {resolved}"


def test_probe_row_lands_in_redirected_log_only() -> None:
    """Drive the real emitter: the row must reach the temp log, not prod."""
    redirected = _redirected_path()
    # Fail fast before writing anything -- a broken guard must not pollute prod.
    assert redirected != _PRODUCTION_ABS
    assert not redirected.startswith(os.path.abspath(str(PROJECT_ROOT)) + os.sep)

    nonce = f"ee-w7-log-isolation-{os.getpid()}-{id(redirected)}"
    timer = PhaseTimer(enabled=True)
    timer.set_log_path(redirected)
    timer.set_cycle(99999)
    timer.begin_generation(0)
    record = timer.end_generation(extra={"log_isolation_nonce": nonce})
    assert record is not None, "PhaseTimer.end_generation() produced no record"
    assert record["cycle"] == 99999

    redirected_abs = os.path.abspath(redirected)
    assert os.path.exists(redirected_abs), f"redirected log missing: {redirected}"
    with open(redirected_abs, encoding="utf-8") as fh:
        redirected_rows = fh.read()
    assert nonce in redirected_rows, "probe row did not reach the redirected log"

    if PRODUCTION_LOG.exists():
        with open(PRODUCTION_LOG, encoding="utf-8", errors="replace") as fh:
            production_rows = fh.read()
        assert nonce not in production_rows, "probe row reached the production generation_phase_timing.jsonl"
        # No row of this session may have been parsed out of the production log.
        leaked = [line for line in production_rows.splitlines() if "log_isolation_nonce" in line]
        assert not leaked, f"test rows leaked into the production log: {leaked[:3]}"


def test_redirected_log_is_append_only_jsonl() -> None:
    """The redirected log must hold well-formed JSONL (no history rewriting)."""
    redirected = _redirected_path()
    if not os.path.exists(redirected):
        return
    with open(redirected, encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            assert json.loads(line), f"redirected log line {number} is not JSON"
