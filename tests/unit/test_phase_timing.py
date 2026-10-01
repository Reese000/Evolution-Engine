"""Unit tests for evolution.phase_timing — per-generation phase split.

Invariants under test:
  * the four named phases + vm_eval form an EXACT partition of a
    generation's wall time (vm seconds are counted once, at the source,
    and subtracted from the enclosing orchestration phase);
  * the source-level guard counts a nested VM call exactly once
    (evaluate_batch -> evaluate, and direct `_execute_vm` callers);
  * one JSON line per generation lands in the rolling log;
  * the run summary exposes only additive ``gen_phase_*`` keys and never
    raises (a raise there would discard the completed cycle's results);
  * EE-W6: the ``mutation_propagation`` sub-phases form an ADDITIVE
    partition of that phase (named + residual == phase seconds), with the
    same VM accounting, and surface as additive ``gen_phase_sub_*`` keys.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from evolution import phase_timing as _phase_timing
from evolution.phase_timing import (
    MUTATION_SUBPHASES,
    NAMED_PHASES,
    PHASES,
    SUBPHASE_COVERED_KEY,
    SUBPHASE_RECORD_KEY,
    PhaseTimer,
    default_log_path,
    record_vm_eval,
    reset_vm_eval,
    roll_log,
    timed_vm_eval,
    timers_enabled,
    vm_eval_totals,
)


@pytest.fixture(autouse=True)
def _clean_vm_counter():
    reset_vm_eval()
    # The unknown-phase warning set is process-global (warn-once); clearing it
    # keeps test_unknown_phase_name_warns_once order-independent under
    # pytest-randomly (review N5).  The sub-phase set is cleared for the same
    # reason (EE-W6).
    _phase_timing._unknown_phases.clear()
    _phase_timing._unknown_subphases.clear()
    _phase_timing._subphase_overcount_seen = False
    yield
    reset_vm_eval()
    _phase_timing._unknown_phases.clear()
    _phase_timing._unknown_subphases.clear()
    _phase_timing._subphase_overcount_seen = False


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
    assert rec is not None  # record still returned
    assert rec["bad_payload"] is not None
    assert timer.log_path is None  # log disabled, not raised
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
    assert phases["mutation_propagation"] < 0.015  # 20ms of gen 0 must be gone
    assert phases["unattributed"] >= -1e-9  # no negative residual
    assert 0.0 <= rec["coverage_ratio"] <= 1.0 + 1e-9
    assert rec["generation"] == 1


# ---------------------------------------------------------------------------
# EE-W5 review findings (B1 / S1 / N1 / N3 / N4)
# ---------------------------------------------------------------------------


def test_reentrant_timed_vm_eval_counts_the_outer_call_once():
    """evaluate_batch -> evaluate must be timed once, not twice (S1).

    The outer wrapper measures its OWN window, which *contains* the inner
    call, so the recorded time is the outer window (~30ms here).  If the
    guard were broken the inner window would be recorded as well: two
    recordings and ~40ms total -- over-reported `vm_eval`, which on a real
    cycle drives `unattributed` negative.
    """
    t_outer = time.perf_counter()
    with timed_vm_eval():
        time.sleep(0.020)
        with timed_vm_eval():
            time.sleep(0.010)
    # Measure the outer window independently: a fixed upper bound on the
    # recorded time would be a load-sensitive assertion (sleeps stretch on a
    # busy host), while `seconds > outer window` is exactly what a broken
    # guard would produce (review SF4).
    outer = time.perf_counter() - t_outer
    seconds, calls = vm_eval_totals()
    assert calls == 1  # the guard, not the sleep, is the proof
    assert seconds >= 0.029  # the outer window covers both sleeps
    assert seconds <= outer + 0.002  # ...counted once, not outer + inner


def test_vm_work_outside_a_vmfarm_block_is_booked_as_vm_eval():
    """Direct VM calls inside an orchestration phase move to vm_eval (B1).

    Terminal/scan-floor gate scoring calls `gpu_vm_farm._execute_vm`
    directly instead of `VMFarm.evaluate*`.  With the source-level wrapper
    that time is subtracted from the enclosing `mutation_propagation` block
    and reported under `vm_eval` -- otherwise the split would misreport
    VM work as "Python orchestration", which is the verdict this card needs.
    """
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timed_vm_eval():  # what _execute_vm now does at source
            time.sleep(0.020)
    rec = timer.end_generation()

    phases = rec["phase_seconds"]
    assert rec["vm_eval_seconds"] >= 0.019  # booked as VM
    # Generous bound only -- the structural proof is the line above plus the
    # partition identity below: if the 20ms were still booked as orchestration
    # vm_eval would be ~0, and if it were booked twice coverage would exceed 1.
    # A fixed-tight bound here is load-sensitive (review SF5).
    assert phases["mutation_propagation"] < 0.05  # not as orchestration
    assert phases["unattributed"] >= -1e-9
    assert rec["coverage_ratio"] <= 1.0 + 1e-9
    # the partition still holds: nothing was double counted
    assert sum(phases.values()) == pytest.approx(rec["generation_seconds"], abs=0.003)


def test_execute_vm_records_wall_time_at_the_source():
    """`gpu_vm_farm._execute_vm` itself is wrapped (B1 regression).

    A direct call -- the path terminal/scan-floor gate scoring takes --
    must increase the process-wide VM accumulator, so its cost lands in
    `vm_eval` instead of the enclosing orchestration phase.  A straight-line
    program is used because the VM short-circuits self-loops (a `jal x0,0`
    reports the full budget in microseconds and would be too fast to
    measure); 20k real instructions take ~15ms.
    """
    gpu_vm_farm = pytest.importorskip("gpu_vm_farm")
    from mutation_proposer import EBREAK, make_rv32i

    reset_vm_eval()
    program = [make_rv32i("ADDI", rd=3, rs1=1, imm=0)] * 20000 + [EBREAK]
    started = time.perf_counter()
    result = gpu_vm_farm._execute_vm(program, {1: 1}, {}, 100000)
    elapsed = time.perf_counter() - started

    assert isinstance(result, dict)
    # Deterministic proof the program really executed (20000 ADDIs, EBREAK
    # not counted) instead of being short-circuited like a self-loop; the
    # cycle count replaces a perf floor, which is machine-dependent (review
    # SF5).
    assert result.get("cycles") == 20000
    seconds, calls = vm_eval_totals()
    assert calls == 1  # the source wrapper recorded it
    assert seconds > 0  # record_vm_eval only counts real time
    assert seconds <= elapsed + 0.01  # the wrapper brackets the call


def test_vm_seconds_are_clamped_to_the_generation_wall():
    """VM time recorded past a generation boundary cannot go negative (N1)."""
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        record_vm_eval(5.0)  # a batch that "spans" the boundary
        time.sleep(0.001)
    rec = timer.end_generation()

    assert rec["vm_eval_seconds"] <= rec["generation_seconds"] + 1e-9
    assert rec["phase_seconds"]["unattributed"] >= -1e-9
    assert rec["coverage_ratio"] <= 1.0 + 1e-9


def test_extra_cannot_shadow_measurement_keys():
    """`extra` diagnostics must not overwrite the measurement fields (S3)."""
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        time.sleep(0.005)
    rec = timer.end_generation({"generations_stalled": 3, "phase_seconds": "evil", "generation_seconds": -1})
    assert isinstance(rec["phase_seconds"], dict)  # reserved key wins
    assert rec["generation_seconds"] > 0  # reserved key wins
    assert rec["generations_stalled"] == 3  # non-colliding extra kept


def test_summary_never_raises_on_a_corrupt_record():
    """A malformed record degrades the summary instead of killing the cycle (S3)."""
    timer = PhaseTimer()
    timer.begin_generation(0)
    timer.end_generation()
    timer.records[0]["phase_seconds"] = "corrupted"

    summary = timer.summary(run_wall=1.0)  # must not raise
    assert summary["gen_phase_generations"] == 1
    assert all(k.startswith("gen_phase_") for k in summary)


def test_unknown_phase_name_warns_once(capsys):
    """A typo in `_pt("...")` is reported instead of silently lost (N4)."""
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("phase_typo_never_registered"):
        pass
    with timer.phase("phase_typo_never_registered"):
        pass
    rec = timer.end_generation()

    err = capsys.readouterr().err
    assert "phase_typo_never_registered" in err
    assert err.count("phase_typo_never_registered") == 1  # warned once
    assert rec["phase_seconds"]["unattributed"] >= 0.0  # fell into residual


# ---------------------------------------------------------------------------
# EE-W6: mutation_propagation sub-phase split
# ---------------------------------------------------------------------------


def test_subphase_names_cover_the_card_and_reserve_the_residual():
    """The documented set is the card's candidate list; `unattributed` is computed."""
    assert MUTATION_SUBPHASES == (
        "generation_init",
        "challenger_generation",
        "diversity_topup",
        "length_capping",
        "dedupe",
        "unattributed",
    )
    assert SUBPHASE_RECORD_KEY == "mutation_subphase_seconds"
    # the record field is a SIBLING of phase_seconds, so the phase partition
    # (asserted exact elsewhere) cannot be perturbed by sub-phase keys
    assert SUBPHASE_RECORD_KEY != "phase_seconds"


def test_subphases_partition_the_mutation_phase():
    """Named sub-phases + residual == mutation_propagation seconds."""
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("generation_init"):
            time.sleep(0.005)
        with timer.sub_phase("challenger_generation"):
            time.sleep(0.010)
    with timer.phase("selection_archives"):
        time.sleep(0.002)
    rec = timer.end_generation()

    subs = rec[SUBPHASE_RECORD_KEY]
    assert set(subs) <= set(MUTATION_SUBPHASES)
    assert subs["generation_init"] >= 0.005
    assert subs["challenger_generation"] >= 0.010
    # the additive identity this card exists for
    assert sum(subs.values()) == pytest.approx(rec["phase_seconds"]["mutation_propagation"], abs=0.003)
    assert subs["unattributed"] >= 0.0  # residual never negative
    # sub-phases belong to their phase only: the 2ms of selection work was
    # not absorbed by the mutation markers
    assert rec["phase_seconds"]["selection_archives"] >= 0.002
    assert rec["phase_seconds"]["selection_archives"] < rec["phase_seconds"]["mutation_propagation"]


def test_subphase_vm_time_is_booked_as_vm_eval_not_as_sub_phase():
    """VM work inside a marker is subtracted exactly like at phase level."""
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("challenger_generation"):
            with timed_vm_eval():  # what _execute_vm does at source
                time.sleep(0.020)
            time.sleep(0.002)  # orchestration around it
    rec = timer.end_generation()

    subs = rec[SUBPHASE_RECORD_KEY]
    assert rec["vm_eval_seconds"] >= 0.019  # counted once, as VM
    # the sub-phase keeps only the orchestration around the VM call
    assert subs["challenger_generation"] < 0.015
    assert rec["phase_seconds"]["mutation_propagation"] < 0.015
    assert sum(subs.values()) == pytest.approx(rec["phase_seconds"]["mutation_propagation"], abs=0.003)
    assert rec["coverage_ratio"] <= 1.0 + 1e-9


def test_unknown_subphase_warns_once_and_keeps_time_in_the_residual(capsys):
    """An unlisted marker neither mints a field nor loses its time (EE-W6)."""
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("sub_typo_never_registered"):
            time.sleep(0.006)
        with timer.sub_phase("sub_typo_never_registered"):
            time.sleep(0.006)
    rec = timer.end_generation()

    err = capsys.readouterr().err
    assert "sub_typo_never_registered" in err
    assert err.count("sub_typo_never_registered") == 1  # warned once
    subs = rec[SUBPHASE_RECORD_KEY]
    assert "sub_typo_never_registered" not in subs  # did not mint a field
    assert subs["unattributed"] >= 0.010  # ...and nothing was lost
    assert sum(subs.values()) == pytest.approx(rec["phase_seconds"]["mutation_propagation"], abs=0.003)


def test_reserved_unattributed_marker_is_recomputed_as_the_residual(capsys):
    """`unattributed` is never taken from a marker: the residual wins."""
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("generation_init"):
            time.sleep(0.004)
        with timer.sub_phase("unattributed"):
            time.sleep(0.020)
    rec = timer.end_generation()

    subs = rec[SUBPHASE_RECORD_KEY]
    assert subs["unattributed"] == pytest.approx(
        rec["phase_seconds"]["mutation_propagation"] - subs["generation_init"], abs=0.003
    )
    assert subs["unattributed"] >= 0.015  # the ignored marker's time is still there
    assert capsys.readouterr().err.count("unattributed") >= 1


def test_subphase_does_not_leak_into_the_next_generation():
    """An aborted generation's sub-phase time must not reach the next record."""
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("challenger_generation"):
            time.sleep(0.020)
    # generation 0 aborts here: end_generation is never called

    timer.begin_generation(1)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("generation_init"):
            time.sleep(0.002)
    rec = timer.end_generation()

    subs = rec[SUBPHASE_RECORD_KEY]
    # The key is always present now (stable key set), so the leak check is on
    # the VALUE: gen 0's 20ms of challenger work must not reappear as a
    # non-zero share of gen 1.
    assert subs["challenger_generation"] == 0.0  # gen 0's 20ms is gone
    assert subs["generation_init"] >= 0.002
    assert rec["phase_seconds"]["mutation_propagation"] < 0.015
    assert sum(subs.values()) == pytest.approx(rec["phase_seconds"]["mutation_propagation"], abs=0.003)


def test_extra_cannot_shadow_the_subphase_record_key():
    """`extra` diagnostics must not overwrite the measurement fields (S3)."""
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("generation_init"):
            time.sleep(0.003)
    rec = timer.end_generation({SUBPHASE_RECORD_KEY: "evil", "generations_stalled": 2})
    assert isinstance(rec[SUBPHASE_RECORD_KEY], dict)
    assert rec[SUBPHASE_RECORD_KEY]["generation_init"] > 0
    assert rec["generations_stalled"] == 2  # non-colliding extra kept


def test_disabled_timer_sub_phase_is_a_noop():
    timer = PhaseTimer(enabled=False)
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("generation_init"):
            time.sleep(0.001)
    assert timer.end_generation() is None
    assert timer.summary() == {}


def test_summary_exposes_additive_gen_phase_sub_keys():
    """The cycle metrics pick these up unchanged.

    run_multi_cycle.py:552-554 copies every ``gen_phase_*`` key from the
    engine result into the cycle metrics, so no runner edit is needed; the
    end-to-end proof is the isolated cycle record of t_9b0b3680 (cycle 7678),
    which carries all seven ``gen_phase_sub_*`` keys.
    """
    timer = PhaseTimer()
    for gen in range(2):
        timer.begin_generation(gen)
        with timer.phase("mutation_propagation"):
            with timer.sub_phase("generation_init"):
                time.sleep(0.002)
            with timer.sub_phase("challenger_generation"):
                time.sleep(0.004)
        timer.end_generation()
    summary = timer.summary(run_wall=1.0)

    assert summary["gen_phase_sub_generation_init_seconds"] > 0
    assert summary["gen_phase_sub_challenger_generation_seconds"] > 0
    assert summary["gen_phase_sub_unattributed_seconds"] >= 0
    covered = summary["gen_phase_sub_covered_seconds"]
    assert covered == pytest.approx(
        summary["gen_phase_sub_generation_init_seconds"] + summary["gen_phase_sub_challenger_generation_seconds"],
        abs=0.01,
    )
    # the sub split reconciles with the phase it splits
    assert covered + summary["gen_phase_sub_unattributed_seconds"] == pytest.approx(
        summary["gen_phase_mutation_propagation_seconds"], abs=0.01
    )
    assert all(k.startswith("gen_phase_") for k in summary)


def test_summary_tolerates_a_corrupt_subphase_record():
    """One malformed row must not hide the other 29."""
    timer = PhaseTimer()
    for gen in range(2):
        timer.begin_generation(gen)
        with timer.phase("mutation_propagation"):
            with timer.sub_phase("generation_init"):
                time.sleep(0.002)
        timer.end_generation()
    timer.records[1][SUBPHASE_RECORD_KEY] = "corrupted"
    timer.records[1][SUBPHASE_RECORD_KEY + "_list"] = ["also", "wrong"]

    summary = timer.summary(run_wall=1.0)
    assert summary["gen_phase_generations"] == 2
    assert summary["gen_phase_sub_generation_init_seconds"] > 0
    assert all(k.startswith("gen_phase_") for k in summary)


def test_summary_skips_a_corrupt_phase_row_without_losing_the_others():
    """A row with an unreadable ``phase_seconds`` must not degrade the summary.

    The sub-phase loop already skipped malformed rows, but the phase totals
    read ``r["phase_seconds"][p]`` unguarded, so ONE bad row raised and
    summary() fell back to its 4-key degraded dict -- losing every phase key
    and every ``gen_phase_sub_*`` key of an otherwise good cycle (review of
    t_9b0b3680).
    """
    timer = PhaseTimer()
    for gen in range(2):
        timer.begin_generation(gen)
        with timer.phase("mutation_propagation"):
            with timer.sub_phase("generation_init"):
                time.sleep(0.002)
        timer.end_generation()
    del timer.records[0]["phase_seconds"]
    timer.records[0]["generation_seconds"] = "corrupted"

    summary = timer.summary(run_wall=1.0)
    assert summary["gen_phase_generations"] == 2
    assert "gen_phase_error" not in summary  # no degradation to the fallback
    assert summary["gen_phase_skipped_rows"] == 1  # and not silently zeroed either
    # the good row still contributes its phase and sub-phase totals
    assert summary["gen_phase_mutation_propagation_seconds"] > 0
    assert summary["gen_phase_sub_generation_init_seconds"] > 0
    assert summary[SUBPHASE_COVERED_KEY] > 0
    assert all(k.startswith("gen_phase_") for k in summary)


def test_summary_ignores_unlisted_subphase_names():
    """A record-borne unknown name must not mint a ``gen_phase_sub_*`` field.

    In particular a row carrying ``covered`` must not overwrite the computed
    ``SUBPHASE_COVERED_KEY`` (review of t_9b0b3680); unknown names are
    dropped exactly like the mark-time warning for an unlisted marker.
    """
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("generation_init"):
            time.sleep(0.002)
    rec = timer.end_generation()
    rec[SUBPHASE_RECORD_KEY] = {**rec[SUBPHASE_RECORD_KEY], "covered": 99.0, "bogus": 5.0}

    summary = timer.summary(run_wall=1.0)
    assert "gen_phase_sub_bogus_seconds" not in summary
    assert summary[SUBPHASE_COVERED_KEY] == pytest.approx(summary["gen_phase_sub_generation_init_seconds"], abs=0.001)
    assert summary[SUBPHASE_COVERED_KEY] < 99.0
    assert all(k.startswith("gen_phase_") for k in summary)


def test_summary_survives_a_non_dict_record_row():
    """A non-dict entry in ``records`` must not raise either (round-2 review).

    Every other row read in ``_build_summary`` is guarded; the sub-phase loop
    and ``_phase_row`` now are too, so the symmetric case is pinned by a test
    instead of by the ``list[dict]`` type hint alone.
    """
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("generation_init"):
            time.sleep(0.002)
    timer.end_generation()
    timer.records.append(None)  # type: ignore[arg-type]

    summary = timer.summary(run_wall=1.0)
    assert "gen_phase_error" not in summary
    assert summary["gen_phase_generations"] == 2
    assert summary["gen_phase_skipped_rows"] == 1
    assert summary["gen_phase_sub_generation_init_seconds"] > 0
    assert all(k.startswith("gen_phase_") for k in summary)


def test_engine_pt_sub_is_a_noop_without_a_phase_timer():
    """`EvolutionEngine._pt_sub` must never raise when the timer is absent."""
    engine_mod = pytest.importorskip("evolution_engine")

    class _NoTimer:
        pass

    with engine_mod.EvolutionEngine._pt_sub(_NoTimer(), "challenger_generation"):
        pass  # nullcontext: the engine still runs with EE_GEN_TIMING=0

    class _Disabled:
        _phase_timer = PhaseTimer(enabled=False)

    with engine_mod.EvolutionEngine._pt_sub(_Disabled(), "challenger_generation"):
        pass


# ---------------------------------------------------------------------------
# EE-W6 independent review (S1 / S2 / S3 / NIT4)
# ---------------------------------------------------------------------------


def test_record_keeps_a_stable_subphase_key_set(capsys):
    """Blocks that did not run are emitted as 0.0, not omitted (review NIT4).

    Consumers reading `gen_phase_sub_<name>_seconds` must not hit a KeyError
    just because a generation never reached the scan-floor cap.
    """
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        time.sleep(0.003)  # unmarked work: only the residual will see it
    rec = timer.end_generation()
    summary = timer.summary(run_wall=1.0)

    assert set(rec[SUBPHASE_RECORD_KEY]) == set(MUTATION_SUBPHASES)
    for name in MUTATION_SUBPHASES:
        assert f"gen_phase_sub_{name}_seconds" in summary
    # and the values are real: the unmarked 3ms is in the residual
    assert rec[SUBPHASE_RECORD_KEY]["unattributed"] >= 0.003
    assert rec[SUBPHASE_RECORD_KEY]["challenger_generation"] == 0.0
    assert capsys.readouterr().err == ""  # nothing to warn about


def test_unmarked_time_in_the_phase_is_neither_lost_nor_counted_as_covered():
    """Review S3: a partially marked phase must show a real residual.

    Without unmarked time the residual is ~0, which made two earlier
    assertions insensitive to a broken residual computation.
    """
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("generation_init"):
            time.sleep(0.005)
        time.sleep(0.030)  # deliberately unmarked work inside the phase
    rec = timer.end_generation()
    summary = timer.summary(run_wall=1.0)

    subs = rec[SUBPHASE_RECORD_KEY]
    assert subs["unattributed"] >= 0.025
    assert subs["generation_init"] >= 0.005
    assert sum(subs.values()) == pytest.approx(rec["phase_seconds"]["mutation_propagation"], abs=0.003)
    # covered counts the NAMED sub-phases only, so it must be strictly
    # smaller than the phase it splits by (about) the unmarked 30ms
    covered = summary[SUBPHASE_COVERED_KEY]
    assert covered == pytest.approx(summary["gen_phase_sub_generation_init_seconds"], abs=0.003)
    assert covered < summary["gen_phase_mutation_propagation_seconds"] - 0.02
    assert covered + summary["gen_phase_sub_unattributed_seconds"] == pytest.approx(
        summary["gen_phase_mutation_propagation_seconds"], abs=0.01
    )


def test_nested_or_misplaced_marker_is_reported_not_silently_absorbed(capsys):
    """Review S1: an over-counting marker set must warn, and never go negative.

    Nesting a marker inside another marker double counts the inner window;
    the residual would be negative, and clamping it silently would make the
    additive identity look like it holds when it does not.
    """
    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"):
        with timer.sub_phase("challenger_generation"):
            with timer.sub_phase("generation_init"):  # NESTED marker (misuse)
                time.sleep(0.010)
    rec = timer.end_generation()

    err = capsys.readouterr().err
    assert "nested or outside the phase" in err
    assert "residual clamped to 0" in err
    subs = rec[SUBPHASE_RECORD_KEY]
    assert subs["unattributed"] >= 0.0  # never negative
    # ...and the record is honest about the over-count instead of hiding it
    assert sum(subs.values()) >= rec["phase_seconds"]["mutation_propagation"] - 1e-9


def test_engine_marker_sites_fire_on_a_real_generation(monkeypatch):
    """Wiring proof for all five committed marker sites (review S2, EE-W6b).

    The timer-level tests above cannot see a MISSING marker: deleting any one
    of the five `_pt_sub(...)` calls in evolution_engine.py must fail this
    test, because it runs `_run_one_generation` end to end on stubbed-out
    heavy work and gives every block a measurable (>=10 ms) body.  Blocks
    whose body is inline (the length-capping try/except and the dedupe call)
    get theirs by patching the functions they import from *inside* the
    block, so what is timed is the marker site itself.
    """
    engine_mod = pytest.importorskip("evolution_engine")
    from types import SimpleNamespace

    from evolution import mutation_helpers as _mh
    from evolution import terminal_stagnation as _ts

    def _sleep_cap(challengers, champ, **kwargs):
        time.sleep(0.010)
        return challengers

    def _sleep_dedupe(challengers):
        time.sleep(0.010)
        return list(challengers), {}

    # length_capping: force the stalled path and make its (normally sub-ms)
    # body measurable.
    monkeypatch.setattr(_ts, "_scan_floor_probe_stall_mode", lambda words: True)
    monkeypatch.setattr(_ts, "get_scan_floor_gate", lambda champ: None)
    monkeypatch.setattr(_mh, "scan_floor_stalled_challenger_max_words", lambda *a, **k: 64)
    monkeypatch.setattr(_mh, "cap_scan_floor_challenger_lengths", _sleep_cap)
    # dedupe: same treatment for the one-line call.
    monkeypatch.setattr(_mh, "dedupe_programs_for_eval", _sleep_dedupe)
    # not a marker site; must not run real archive writes in a unit test.
    monkeypatch.setattr(_mh, "record_l10_lift_growth_discoveries", lambda *a, **k: None)

    engine = engine_mod.EvolutionEngine.__new__(engine_mod.EvolutionEngine)
    timer = PhaseTimer()
    engine._phase_timer = timer
    # EE-W6c: capture the [PhaseSub] lines this generation emits.
    logged: list[str] = []
    engine._log = logged.append
    # Both [PhaseSub] systems are env-gated; pin them so an ambient value
    # in the developer shell cannot flip the assertions below.
    monkeypatch.setenv("EE_PHASE_SUB_LOG_MIN", "3")
    monkeypatch.setenv("EE_PHASE_SUBPART_LOG_MIN", "0.005")
    engine.config = SimpleNamespace(
        population_size=4,
        use_mutation_scheduler=True,
        mutation_rate_start=0.3,
        mutation_rate_end=0.1,
        mutation_schedule_generations=100,
        max_generations=100,
    )
    engine.champion = [0x00000013, 0x00100073]
    engine.champion_level = 1
    engine.generations_stalled = 0
    engine.current_generation = 6

    # Heavy work is stubbed with a measurable sleep so a marker that fires
    # shows a non-zero share (a bare 0.0 would be indistinguishable from a
    # marker that never ran, now that the key set is stable).
    engine._compute_adaptive_mutation_rate = lambda: (time.sleep(0.010), 0.3)[1]
    engine._choose_mutation_strategy = lambda idx: SimpleNamespace(value="STUB")
    engine._pick_mutation_parent = lambda: (list(engine.champion), None)
    engine._apply_mutation = lambda strategy, **kw: (time.sleep(0.010), [0x00000013])[1]
    engine._generate_diversity_injection = lambda challengers, pop_size: None
    engine._inject_terminal_verify_probes = lambda challengers, pop_size: None
    engine._apply_model_guided_slots = lambda challengers: None
    engine._inject_qd_archive_hybrids = lambda challengers: None
    # diversity_topup marker site body
    engine._maybe_topup_scan_floor_diversity = lambda challengers: time.sleep(0.010)
    engine._compute_population_diversity = lambda challengers: 0.0
    engine._evaluate_batch = lambda challengers: []
    engine._fill_empty_qd_cells_from_evaluated = lambda *a, **k: None
    engine._select_elite = lambda challengers, results: ([], [], None)
    engine._evaluate_batch_detailed = lambda elite: []
    engine._run_generation_update = lambda *a, **k: False
    engine._log_generation = lambda *a, **k: None
    engine._run_generation_maintenance = lambda *a, **k: None

    engine._run_one_generation(6, 0, False, 25, 10, None, 30)

    assert len(timer.records) == 1
    rec = timer.records[0]
    subs = rec[SUBPHASE_RECORD_KEY]
    for name in ("generation_init", "challenger_generation", "diversity_topup", "length_capping", "dedupe"):
        assert subs[name] >= 0.010, f"{name} marker did not fire"
    # the identity holds on the real code path, not just in synthetic tests
    assert sum(subs.values()) == pytest.approx(rec["phase_seconds"]["mutation_propagation"], abs=0.01)
    assert rec["phase_seconds"]["mutation_propagation"] >= 0.050
    # EE-W6b: the sibling lane's diagnostic markers ride the same record
    # under their own key -- two systems, two keys, two meanings.
    assert SUBPHASE_RECORD_KEY != "subphase_seconds"
    assert "subphase_seconds" in rec

    # EE-W6c: the additive split this record carries is operator-visible on
    # its own [PhaseSub] line, leading with the dominant named sub-phase.
    # Filters are anchored on the [PhaseSub] prefix so a future line that
    # merely MENTIONS "mutation:" cannot satisfy them.
    part_lines = [m for m in logged if m.startswith("[PhaseSub]") and " mutation: " in m]
    assert len(part_lines) == 1, logged
    dominant = max((k for k in subs if k != "unattributed"), key=lambda k: subs[k])
    line = part_lines[0]
    assert f"gen={rec['generation']}" in line
    assert f"{dominant}=" in line
    assert "unattributed=" in line
    # biggest-first ordering: whatever leads the listing IS the dominant
    # block (an index-vs-residual comparison would pass for ascending too)
    assert line.split("mutation: ", 1)[1].startswith(f"{dominant}=")
    # the sibling hotspot line is a DIFFERENT system with its own gate
    # (3 s >> these ~10 ms blocks), so it stays independent -- and quiet.
    assert not [m for m in logged if m.startswith("[PhaseSub]") and " top: " in m]

    # Quiet generation: raise the partition gate above every named block so
    # only residual time remains -> nothing at all is emitted.
    logged.clear()
    monkeypatch.setenv("EE_PHASE_SUBPART_LOG_MIN", "999")
    engine._log_subphase_marker(rec)
    assert logged == []


def test_sibling_diagnostic_markers_and_additive_subphases_share_a_record():
    """EE-W6b reconciliation: both marker systems, neither borrowing a key.

    The sibling lane's ``subphase()`` markers are VM-inclusive and nest
    inside each other, so they are a HOTSPOT MAP (their sum overshoots the
    phase).  EE-W6's ``sub_phase()`` markers are VM-subtracted and
    partition the phase (named + residual == phase).  Both are written to
    one generation record -- each under its own key, and only the EE-W6 key
    may ever be read as a partition.
    """
    assert SUBPHASE_RECORD_KEY != "subphase_seconds"
    assert SUBPHASE_RECORD_KEY == "mutation_subphase_seconds"

    timer = PhaseTimer()
    timer.begin_generation(0)
    with timer.phase("mutation_propagation"), _phase_timing.subphase("generate_challengers"):
        with timer.sub_phase("challenger_generation"):
            time.sleep(0.030)
        with _phase_timing.subphase("topup_diversity"):  # nests inside the marker above
            with timer.sub_phase("diversity_topup"):
                time.sleep(0.060)
    rec = timer.end_generation()

    subs = rec[SUBPHASE_RECORD_KEY]
    diag = rec["subphase_seconds"]
    phase = rec["phase_seconds"]["mutation_propagation"]

    # additive partition: named sub-phases + residual == the phase
    assert sum(subs.values()) == pytest.approx(phase, abs=0.005)
    assert subs["challenger_generation"] == pytest.approx(0.030, abs=0.005)
    assert subs["diversity_topup"] == pytest.approx(0.060, abs=0.005)
    assert subs["generation_init"] == 0.0  # stable key set, block did not run

    # diagnostic markers nest and do NOT partition: 0.090 + 0.060 > 0.090
    assert diag["generate_challengers"] == pytest.approx(0.090, abs=0.005)
    assert diag["topup_diversity"] == pytest.approx(0.060, abs=0.005)
    assert sum(diag.values()) > phase * 1.5


def test_phase_sub_partition_line_gates_on_largest_named_subphase(monkeypatch):
    """EE-W6c: ``_log_subphase_marker`` prints the additive split, gated on
    the largest NAMED sub-phase.

    The ``mutation_subphase_seconds`` partition used to be invisible to an
    operator watching logs (it only surfaced in cycle metrics /
    ``generation_phase_timing.jsonl``).  It now gets its own ``[PhaseSub]``
    line -- separate from the sibling lane's ``top:`` hotspot line, with an
    independent threshold -- and it must:

    * fire when one named sub-phase dominates, naming it first;
    * stay QUIET when the time all sits in the ``unattributed`` residual
      (the gate excludes the residual by construction);
    * stay QUIET when every named sub-phase is under the gate;
    * not disturb the sibling ``subphase_seconds`` line, in either
      direction (two systems, two keys, two independently tuned gates);
    * tolerate records that lack the new key entirely (old records,
      sibling-only code paths) without raising -- markers must never
      break evolution -- and ignore garbage INSIDE the partition (unknown
      names, non-numeric, non-finite) exactly the way ``summary()``
      ignores it, so it can neither lead the line nor abort it.
    """
    engine_mod = pytest.importorskip("evolution_engine")
    engine = engine_mod.EvolutionEngine.__new__(engine_mod.EvolutionEngine)
    logged: list[str] = []
    engine._log = logged.append
    monkeypatch.setenv("EE_PHASE_SUB_LOG_MIN", "3")
    monkeypatch.setenv("EE_PHASE_SUBPART_LOG_MIN", "3")

    def _record(**split):
        """A generation record with the stable EE-W6 key set + overrides."""
        full = {name: 0.0 for name in MUTATION_SUBPHASES}
        full.update(split)
        return {
            "generation": 29,
            "generation_seconds": 132.7,
            "subphase_seconds": {},
            SUBPHASE_RECORD_KEY: full,
        }

    # 1. dominant named sub-phase above the gate -> line fires, dominant first
    engine._log_subphase_marker(_record(challenger_generation=67.7, length_capping=0.04))
    assert len(logged) == 1, logged
    line = logged[0]
    assert line.startswith("[PhaseSub] gen=29 mutation: challenger_generation=67.70s")
    assert line.index("challenger_generation=") < line.index("length_capping=")
    assert "unattributed=0.00s" in line  # residual is always reported
    assert "generation_init=" not in line  # 0.0 s blocks are noise, not shown

    # 2. residual-only generation -> quiet (the gate ignores unattributed)
    logged.clear()
    engine._log_subphase_marker(_record(unattributed=42.0))
    assert logged == []

    # 3. every named sub-phase under the gate -> quiet, residual or not
    logged.clear()
    engine._log_subphase_marker(_record(challenger_generation=2.9, unattributed=40.0))
    assert logged == []

    # 4. the two systems tune independently.  Build a record where BOTH
    #    would fire, then close one gate at a time.
    both = _record(challenger_generation=67.7)
    both["subphase_seconds"] = {"generate_challengers": 50.0}

    logged.clear()
    monkeypatch.setenv("EE_PHASE_SUBPART_LOG_MIN", "100")  # close partition gate
    engine._log_subphase_marker(both)
    assert [m for m in logged if " top: " in m], logged
    assert not [m for m in logged if " mutation: " in m]

    logged.clear()
    monkeypatch.setenv("EE_PHASE_SUBPART_LOG_MIN", "3")  # reopen
    monkeypatch.setenv("EE_PHASE_SUB_LOG_MIN", "100")  # close hotspot gate
    engine._log_subphase_marker(both)
    assert [m for m in logged if " mutation: " in m], logged
    assert not [m for m in logged if " top: " in m]

    # 5. records without the additive key (pre-EE-W6) behave exactly as
    #    before: no partition line, no exception, sibling line untouched.
    logged.clear()
    monkeypatch.setenv("EE_PHASE_SUB_LOG_MIN", "3")
    engine._log_subphase_marker({"generation": 5, "subphase_seconds": {}})
    engine._log_subphase_marker({})
    engine._log_subphase_marker(None)
    assert logged == []
    legacy = {"generation": 5, "generation_seconds": 9.0, "subphase_seconds": {"enrich": 4.0}}
    engine._log_subphase_marker(legacy)
    assert len(logged) == 1 and " top: " in logged[0] and " mutation: " not in logged[0]

    # 6. a truthy NON-DICT record must not raise: the pre-EE-W6c form read
    #    `record.get(...)` inside its try and swallowed the AttributeError;
    #    the guard keeps that "diagnostics never break evolution" promise
    #    for both lines.
    logged.clear()
    for junk in ([1, 2, 3], "hello", 42, object()):
        engine._log_subphase_marker(junk)  # type: ignore[arg-type]
    assert logged == []

    # 7. a gate of 0 ("log every generation") still cannot print a line
    #    that names no stage: residual-only stays quiet, a real stage prints.
    monkeypatch.setenv("EE_PHASE_SUBPART_LOG_MIN", "0")
    logged.clear()
    engine._log_subphase_marker(_record(unattributed=42.0))
    assert logged == []
    engine._log_subphase_marker(_record(challenger_generation=0.5))
    assert len(logged) == 1 and "challenger_generation=0.50s" in logged[0]

    # 8. sub-10 ms dominant under a sub-10 ms gate: the 10 ms display filter
    #    must not eat the block the gate just admitted, so the line still
    #    names a stage and still leads with it.
    monkeypatch.setenv("EE_PHASE_SUBPART_LOG_MIN", "0.005")
    logged.clear()
    engine._log_subphase_marker(_record(challenger_generation=0.007, unattributed=89.5))
    assert len(logged) == 1, logged
    assert logged[0].split("mutation: ", 1)[1].startswith("challenger_generation=")
    assert "unattributed=89.50s" in logged[0]

    # 9. garbage inside the partition: unknown names and non-numeric /
    #    non-finite values are ignored the way summary() ignores them --
    #    they neither lead the line nor abort it.
    monkeypatch.setenv("EE_PHASE_SUBPART_LOG_MIN", "3")
    dirty = _record(challenger_generation=67.7)
    dirty[SUBPHASE_RECORD_KEY]["covered"] = 500.0
    dirty[SUBPHASE_RECORD_KEY]["length_capping"] = "oops"
    dirty[SUBPHASE_RECORD_KEY]["dedupe"] = float("nan")
    logged.clear()
    engine._log_subphase_marker(dirty)
    assert len(logged) == 1, logged
    body = logged[0].split("mutation: ", 1)[1]
    assert body.startswith("challenger_generation=67.70s")
    assert "covered=" not in body
    assert "length_capping=" not in body
    assert "dedupe=" not in body
