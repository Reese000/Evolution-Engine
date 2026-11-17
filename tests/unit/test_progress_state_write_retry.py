"""Regression tests for the progress-state atomic write retry.

`_patch_progress_state` used to swallow the FIRST OSError from its
tmp-file + fsync + `os.replace` dance, silently dropping the update. Losing
the `terminal_efficiency_scan_floor_*` keys makes
`should_rescan_terminal_efficiency_compression` return True on every cycle
(expensive ranked sweep), and fault injection reproduces exactly the
`assert 2 == 1` signature seen once in a 2026-09-30 full-file run of
test_archive_scan_floor_probes — consistent with, but not proven to be, the
cause of that particular run. The write now retries the transient class
briefly, does not back off on deterministic errors, warns when it gives up,
and still never raises out of a production progress update.
"""

import json
import os

import pytest

from evolution import terminal_stagnation as ts


@pytest.fixture()
def agents(tmp_path, monkeypatch):
    path = tmp_path / ".agents"
    path.mkdir()
    monkeypatch.setattr(ts, "_project_agents_dir", lambda: str(path))
    return path


def _flaky_replace(monkeypatch, fail_times, only_for="progress_controller_state.json"):
    """Fail `os.replace` for the progress-state file `fail_times` times."""
    real = os.replace
    state = {"remaining": int(fail_times), "calls": 0}

    def fake(src, dst):
        if str(dst).endswith(only_for):
            state["calls"] += 1
            if state["remaining"] > 0:
                state["remaining"] -= 1
                raise PermissionError("simulated transient write failure")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", fake)
    return state


@pytest.fixture()
def sleeps(monkeypatch):
    """Record backoff calls instead of really sleeping."""
    calls: list[float] = []
    monkeypatch.setattr(ts.time, "sleep", calls.append)
    return calls


def test_progress_state_write_recovers_from_transient_replace_failures(agents, monkeypatch, sleeps):
    replace = _flaky_replace(monkeypatch, fail_times=2)

    ts.record_terminal_efficiency_scan_floor(6543, 48, 1.0)

    assert replace["remaining"] == 0, "write retry never consumed the injected faults"
    assert replace["calls"] == 3, "one attempt per injected fault, then one win"
    assert sleeps == [pytest.approx(0.05), pytest.approx(0.10)]
    state = json.loads((agents / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert state["terminal_efficiency_scan_floor_words"] == 48
    assert state["terminal_efficiency_scan_floor_cycle"] == 6543
    assert ts.terminal_efficiency_at_scan_floor(48) is True


def test_progress_state_write_recovers_from_transient_fsync_fault(agents, monkeypatch, sleeps):
    real_fsync = os.fsync
    state = {"remaining": 1}

    def flaky_fsync(fd):
        if state["remaining"] > 0:
            state["remaining"] -= 1
            raise OSError("simulated transient fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", flaky_fsync)

    ts.record_terminal_efficiency_scan_floor(6543, 48, 1.0)

    assert state["remaining"] == 0, "retry never consumed the injected fsync fault"
    assert sleeps == [pytest.approx(0.05)]
    assert ts.terminal_efficiency_at_scan_floor(48) is True


def test_progress_state_write_retries_are_bounded_and_logged(agents, monkeypatch, sleeps, caplog):
    replace = _flaky_replace(monkeypatch, fail_times=10**6)

    # Historic contract: a dead/unwritable state file must not raise out of a
    # production progress update.
    with caplog.at_level("WARNING", logger="EvolutionEngine"):
        ts.record_terminal_efficiency_scan_floor(6543, 48, 1.0)

    assert replace["calls"] == 3, "retry must be bounded at 3 attempts"
    assert len(sleeps) == 2, "only the first two failures back off"
    assert not (agents / "progress_controller_state.json").exists()
    assert not (agents / "progress_controller_state.json.tmp").exists()
    assert ts.terminal_efficiency_at_scan_floor(48) is False
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "progress-controller state write failed" in r.getMessage() for r in warnings
    ), "giving up must leave a visible trace"


def test_deterministic_write_error_is_not_retried(monkeypatch, sleeps, caplog):
    # A missing agents dir is ENOENT: retrying cannot help, so it must neither
    # back off nor reach os.replace before the historic swallow.
    gone = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_no_such_dir_")
    assert not os.path.isdir(gone)
    replace = {"calls": 0}
    monkeypatch.setattr(
        os,
        "replace",
        lambda src, dst: replace.__setitem__("calls", replace["calls"] + 1),
    )
    monkeypatch.setattr(ts, "_project_agents_dir", lambda: gone)

    with caplog.at_level("WARNING", logger="EvolutionEngine"):
        ts.record_terminal_efficiency_scan_floor(6543, 48, 1.0)

    assert replace["calls"] == 0, "deterministic ENOENT must not reach os.replace"
    assert sleeps == [], "deterministic errors must not back off"
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "progress-controller state write failed" in r.getMessage() for r in warnings
    ), "the deterministic branch must warn exactly like the exhausted-retry branch"
