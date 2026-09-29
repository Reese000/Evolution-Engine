"""Unit tests for evolution.phase_timing — per-generation phase split.

Invariants under test:
  * the four named phases + vm_eval form an EXACT partition of a
    generation's wall time (vm seconds are counted once, at the source,
    and subtracted from the enclosing orchestration phase);
  * one JSON line per generation lands in the rolling log;
  * the run summary exposes only additive ``gen_phase_*`` keys.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from evolution.phase_timing import (
    NAMED_PHASES,
    PHASES,
    PhaseTimer,
    default_log_path,
    reset_vm_eval,
    roll_log,
    timed_vm_eval,
    timers_enabled,
    vm_eval_totals,
)


@pytest.fixture(autouse=True)
def _clean_vm_counter():
    reset_vm_eval()
    yield
    reset_vm_eval()


def test_vm_counter_accumulates_seconds_and_calls():
    with timed_vm_eval():
        time.sleep(0.01)
    seconds, calls = vm_eval_totals()
    assert seconds >= 0.01
    assert calls == 1


def test_partition_is_exact_with_and_without_nested_vm():
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        time.sleep(0.005)
    with timer.phase("vm_eval"):
        with timed_vm_eval():
            time.sleep(0.01)
    with timer.phase("selection_archives"):
        time.sleep(0.003)
    with timer.phase("diagnostics_logging"):
        time.sleep(0.002)
    rec = timer.end_generation()

    phases = rec["phase_seconds"]
    assert set(phases) == set(PHASES)
    # the evaluation phase reports VM time + the orchestration around it
    assert rec["phase_seconds"]["vm_eval"] >= 0.01
    assert rec["vm_eval_seconds"] >= 0.01

    total = sum(phases.values())
    # sum(phases) == generation_seconds is an IDENTITY (unattributed is defined
    # as wall - accounted), so it cannot detect vm double counting. The guards
    # that can: a double count drives unattributed negative and coverage > 1
    # (measured: unattr=-0.01, coverage=1.4546 for a 10ms double count).
    assert total == pytest.approx(rec["generation_seconds"], abs=0.003)
    assert phases["unattributed"] >= -1e-9
    assert 0.0 <= rec["coverage_ratio"] <= 1.0 + 1e-9
    # the evaluation phase reports VM time + the orchestration around it, and
    # can never exceed the generation itself
    assert phases["vm_eval"] <= rec["generation_seconds"] + 1e-9
    assert rec["vm_eval_seconds"] <= rec["generation_seconds"] + 1e-9


def test_nested_vm_is_subtracted_from_enclosing_phase_only_once():
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("selection_archives"):
        with timed_vm_eval():
            time.sleep(0.02)
    rec = timer.end_generation()

    # orchestration around the VM call is ~0; the 20ms is reported as vm_eval
    assert rec["phase_seconds"]["selection_archives"] < 0.005
    assert rec["vm_eval_seconds"] >= 0.02
    assert rec["phase_seconds"]["mutation_propagation"] == 0.0


def test_one_json_line_per_generation(tmp_path):
    log = tmp_path / "gen_timing.jsonl"
    timer = PhaseTimer()
    timer.set_cycle(4242)
    timer.set_log_path(str(log))
    for gen in range(3):
        timer.begin_generation(gen)
        with timer.phase("diagnostics_logging"):
            time.sleep(0.001)
        timer.end_generation()

    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    rows = [json.loads(ln) for ln in lines]
    assert [r["generation"] for r in rows] == [0, 1, 2]
    assert all(r["cycle"] == 4242 for r in rows)
    assert all(set(r["phase_seconds"]) == set(PHASES) for r in rows)


def test_roll_log_keeps_newest_lines(tmp_path):
    log = tmp_path / "roll.jsonl"
    for i in range(200):
        log.write_text(log.read_text(encoding="utf-8") if log.exists() else "", encoding="utf-8")
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"generation": i}) + "\n")
    before = log.stat().st_size
    roll_log(str(log), max_bytes=before // 2)
    kept = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert log.stat().st_size <= before
    assert len(kept) < 200
    assert kept[-1]["generation"] == 199  # newest retained


def test_summary_fields_are_additive_and_named():
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        pass
    timer.end_generation()
    summary = timer.summary(run_wall=1.0)

    assert summary["gen_phase_generations"] == 1
    for key in (
        "gen_phase_seconds",
        "gen_phase_vm_eval_seconds",
        "gen_phase_vm_farm_seconds",
        "gen_phase_mutation_propagation_seconds",
        "gen_phase_selection_archives_seconds",
        "gen_phase_diagnostics_logging_seconds",
        "gen_phase_unattributed_seconds",
        "gen_phase_named_seconds",
        "gen_phase_vm_eval_calls",
        "gen_phase_run_init_seconds",
        "gen_phase_run_finalize_seconds",
        "gen_phase_run_seconds",
        "gen_phase_log_path",
    ):
        assert key in summary, key
    assert all(k.startswith("gen_phase_") for k in summary)


def test_disabled_timer_emits_nothing():
    timer = PhaseTimer(enabled=False)
    timer.begin_generation(0)
    with timer.phase("vm_eval"):
        time.sleep(0.001)
    assert timer.end_generation() is None
    assert timer.summary() == {}


def test_named_phases_exclude_residual():
    assert NAMED_PHASES == (
        "vm_eval",
        "mutation_propagation",
        "selection_archives",
        "diagnostics_logging",
    )
    assert PHASES[-1] == "unattributed"


def test_default_log_path_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("EE_GEN_TIMING_LOG", raising=False)
    assert default_log_path(str(tmp_path)).endswith("generation_phase_timing.jsonl")
    monkeypatch.setenv("EE_GEN_TIMING_LOG", str(tmp_path / "custom.jsonl"))
    assert default_log_path(str(tmp_path)).endswith("custom.jsonl")


def test_timers_enabled_env_gate(monkeypatch):
    monkeypatch.delenv("EE_GEN_TIMING", raising=False)
    assert timers_enabled() is True
    monkeypatch.setenv("EE_GEN_TIMING", "0")
    assert timers_enabled() is False


def test_emit_failure_disables_log_path_without_raising(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    timer = PhaseTimer()
    timer.set_log_path(os.path.join(str(blocker), "nested", "gen.jsonl"))
    timer.begin_generation(0)
    rec = timer.end_generation()
    assert rec is not None
    assert timer.log_path is None


def test_non_serializable_extra_does_not_escape(tmp_path):
    """A bad `extra` payload must never abort the generation loop.

    Regression: `json.dumps` TypeError used to escape end_generation and
    reach run()'s blanket `except Exception`, truncating the run (EE-W5).
    """
    timer = PhaseTimer()
    timer.set_log_path(str(tmp_path / "gen.jsonl"))
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        time.sleep(0.001)
    rec = timer.end_generation({"bad_payload": object()})
    assert rec is not None                      # record still returned
    assert rec["bad_payload"] is not None
    assert timer.log_path is None               # log disabled, not raised
    # open(..., "a") creates the file before json.dumps raises, so it may
    # exist -- but no record line may have been written to it.
    log = tmp_path / "gen.jsonl"
    assert not log.exists() or log.read_text(encoding="utf-8") == ""


def test_aborted_generation_does_not_leak_into_the_next():
    """begin_generation resets residual phase time from an aborted generation."""
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        time.sleep(0.020)
    # generation 0 aborts here: end_generation is never called

    timer.begin_generation(1)
    with timer.phase("mutation_propagation"):
        time.sleep(0.005)
    rec = timer.end_generation()

    phases = rec["phase_seconds"]
    assert phases["mutation_propagation"] < 0.015   # 20ms of gen 0 must be gone
    assert phases["unattributed"] >= -1e-9          # no negative residual
    assert 0.0 <= rec["coverage_ratio"] <= 1.0 + 1e-9
    assert rec["generation"] == 1
