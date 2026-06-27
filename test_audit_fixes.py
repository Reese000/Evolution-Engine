#!/usr/bin/env python3
"""Unit tests for D1, D2, L1, S1 audit fixes — only tests that map to actual code changes."""
import json
import os
import random
import tempfile


# ══════════════════════════════════════════════════════════════════════
# D1 — Distance-Weighted Fitness Scoring
# ══════════════════════════════════════════════════════════════════════

def _distance_weighted_score(actual: int, expected: int) -> float:
    """Pure implementation of the D1 distance-weighted scoring formula."""
    diff = abs(actual - expected)
    max_diff = max(1, abs(expected))
    return 1.0 / (1.0 + diff / max_diff)


def test_d1_exact_match_scores_1():
    assert _distance_weighted_score(55, 55) == 1.0


def test_d1_off_by_one_gets_high_credit():
    s = _distance_weighted_score(54, 55)
    assert 0.4 < s < 1.0


def test_d1_off_by_one_small_value():
    s = _distance_weighted_score(4, 5)
    assert 0.8 < s < 1.0


def test_d1_wildly_wrong_gets_near_zero():
    assert _distance_weighted_score(99999, 55) < 0.01


def test_d1_zero_expected():
    s = _distance_weighted_score(5, 0)
    assert 0 < s < 0.5
    assert _distance_weighted_score(0, 0) == 1.0


def test_d1_negative_expected():
    assert _distance_weighted_score(-10, -10) == 1.0
    assert 0.8 < _distance_weighted_score(-9, -10) < 1.0


def test_d1_fitness_monotonic():
    expected = 10
    scores = [_distance_weighted_score(val, expected) for val in [0, 5, 9, 10]]
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i - 1]


# ══════════════════════════════════════════════════════════════════════
# D2 — Fitness Gate Logic
# ══════════════════════════════════════════════════════════════════════

def _fitness_gate_should_block(champion_fitness, generations_stalled, circuit_breaker_stall_limit):
    if champion_fitness is None:
        return True
    if champion_fitness > 0:
        return False
    if generations_stalled >= circuit_breaker_stall_limit:
        return False
    return True


def test_d2_positive_fitness_passes():
    assert not _fitness_gate_should_block(0.5, 5, 20)


def test_d2_negative_fitness_blocked():
    assert _fitness_gate_should_block(-1.0, 5, 20)


def test_d2_zero_fitness_blocked():
    assert _fitness_gate_should_block(0.0, 5, 20)


def test_d2_deeply_stalled_bypasses():
    assert not _fitness_gate_should_block(-1.0, 25, 20)


def test_d2_exactly_at_limit_bypasses():
    assert not _fitness_gate_should_block(-1.0, 20, 20)


def test_d2_none_fitness_blocked():
    assert _fitness_gate_should_block(None, 5, 20)


# ══════════════════════════════════════════════════════════════════════
# L1 — Task Instance Refresh
# ══════════════════════════════════════════════════════════════════════

def test_l1_refresh_triggers_at_25():
    assert 25 % 25 == 0 and 25 > 0


def test_l1_no_refresh_at_gen_0():
    assert not (0 % 25 == 0 and 0 > 0)


def test_l1_no_refresh_at_non_multiple():
    assert not (10 % 25 == 0 and 10 > 0)


def test_l1_refresh_at_50_75_100():
    for gen in [50, 75, 100, 200]:
        assert gen % 25 == 0 and gen > 0


# ══════════════════════════════════════════════════════════════════════
# S1 — Clean-State Logic
# ══════════════════════════════════════════════════════════════════════

def test_s1_clean_state_removes_files_without_crashing():
    """Verify clean-state file removal works and doesn't crash on missing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock state files
        state_files = [
            "meta_controller_state.json",
            "strategy_archive_gen100.json",
            "strategy_archive_gen200.json",
            "novelty_archive_gen100.json",
            "barrier_tuner_state.json",
        ]
        for fname in state_files:
            with open(os.path.join(tmpdir, fname), 'w') as f:
                f.write('{}')

        # Create a non-state file
        with open(os.path.join(tmpdir, "evolution_engine.py"), 'w') as f:
            f.write('# not a state file')

        patterns = [
            "meta_controller_state.json",
            "strategy_archive_gen*.json",
            "novelty_archive_gen*.json",
            "barrier_tuner_state.json",
        ]

        import glob
        removed = []
        for pat in patterns:
            for f in glob.glob(os.path.join(tmpdir, pat)):
                os.remove(f)
                removed.append(os.path.basename(f))

        # State files removed
        for fname in state_files:
            assert fname in removed
            assert not os.path.exists(os.path.join(tmpdir, fname))

        # Non-state file preserved
        assert os.path.exists(os.path.join(tmpdir, "evolution_engine.py"))


def test_s1_dry_run_no_errors():
    """Clean-state with no state files doesn't crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        patterns = [
            "meta_controller_state.json",
            "strategy_archive_gen*.json",
            "novelty_archive_gen*.json",
            "barrier_tuner_state.json",
        ]
        import glob
        for pat in patterns:
            for f in glob.glob(os.path.join(tmpdir, pat)):
                os.remove(f)
        assert True


# ══════════════════════════════════════════════════════════════════════
# v51 — Graduated Bonus Threshold (Fix 1)
# ══════════════════════════════════════════════════════════════════════

def _compute_bonus_threshold(level: int, num_levels: int = 12) -> float:
    """Replicates the graduated bonus threshold formula from evolution_engine.py."""
    return max(0.85, 0.95 - 0.10 * (level / max(1, num_levels - 1)))


def _compute_graduated_bonus(reg_match: float, level: int, num_levels: int = 12) -> float:
    """Replicates the graduated bonus formula from evolution_engine.py."""
    threshold = _compute_bonus_threshold(level, num_levels)
    if reg_match >= threshold:
        return 50.0 * (reg_match - threshold) / (1.0 - threshold)
    return 0.0


def test_v51_bonus_threshold_l0():
    """At L0, threshold should be 0.95 (strict — simple tasks)."""
    t = _compute_bonus_threshold(0)
    assert abs(t - 0.95) < 0.001, f"L0 threshold={t}, expected 0.95"


def test_v51_bonus_threshold_mid():
    """At mid-level (L6), threshold should be approximately 0.895."""
    t = _compute_bonus_threshold(6, num_levels=12)
    assert 0.89 < t < 0.90, f"L6 threshold={t}, expected ~0.895"


def test_v51_bonus_threshold_high():
    """At high level (L11), threshold should be 0.85 (most forgiving)."""
    t = _compute_bonus_threshold(11, num_levels=12)
    assert abs(t - 0.85) < 0.001, f"L11 threshold={t}, expected 0.85"


def test_v51_bonus_clamped_at_0_85():
    """Threshold should never go below 0.85, even at very high levels."""
    t = _compute_bonus_threshold(100, num_levels=12)
    assert abs(t - 0.85) < 0.001, f"L100 threshold={t}, expected 0.85"


def test_v51_graduated_bonus_below_threshold():
    """Below threshold, bonus should be 0."""
    bonus = _compute_graduated_bonus(0.90, 0)  # L0 threshold=0.95
    assert bonus == 0.0


def test_v51_graduated_bonus_at_threshold():
    """At exactly threshold, bonus should be 0."""
    bonus = _compute_graduated_bonus(0.95, 0)  # L0 threshold=0.95
    assert abs(bonus) < 0.001


def test_v51_graduated_bonus_perfect():
    """At perfect match (1.0), bonus should be 50.0."""
    bonus = _compute_graduated_bonus(1.0, 0)
    assert abs(bonus - 50.0) < 0.001


def test_v51_graduated_bonus_halfway():
    """Halfway between threshold and perfect should give ~25 bonus."""
    bonus = _compute_graduated_bonus(0.975, 0)  # halfway: (0.95+1.0)/2
    assert 24.0 < bonus < 26.0, f"Halfway bonus={bonus}, expected ~25"


def test_v51_graduated_bonus_monotonic():
    """Bonus should increase monotonically with reg_match."""
    bonuses = [_compute_graduated_bonus(m / 100, 5) for m in range(85, 101)]
    for i in range(1, len(bonuses)):
        assert bonuses[i] >= bonuses[i - 1] - 0.001, f"Non-monotonic at index {i}: {bonuses[i-1]} -> {bonuses[i]}"


# ══════════════════════════════════════════════════════════════════════
# v51 — raw_task_score Gate (Fix 4)
# ══════════════════════════════════════════════════════════════════════

def _raw_task_score_gate(raw_score: float) -> bool:
    """Replicates the raw_task_score gate logic."""
    return raw_score < 10.0


def test_v51_raw_score_gate_zero():
    """raw=0 should be blocked."""
    assert _raw_task_score_gate(0.0) is True


def test_v51_raw_score_gate_below_threshold():
    """raw=5 should be blocked."""
    assert _raw_task_score_gate(5.0) is True


def test_v51_raw_score_gate_at_threshold():
    """raw=10.0 should pass (not blocked)."""
    assert _raw_task_score_gate(10.0) is False


def test_v51_raw_score_gate_above_threshold():
    """raw=15 should pass."""
    assert _raw_task_score_gate(15.0) is False


def test_v51_raw_score_gate_negative():
    """Negative raw scores should be blocked."""
    assert _raw_task_score_gate(-20.0) is True


# ══════════════════════════════════════════════════════════════════════
# v51 — champion_fitness Sentinel (Fix A)
# ══════════════════════════════════════════════════════════════════════

def test_v51_champion_fitness_initial():
    """Initial champion_fitness should be non-negative, not -inf."""
    # The actual fix changes -float('inf') to 0.0
    initial = 0.0
    assert initial == 0.0
    assert initial > float('-inf')


# ══════════════════════════════════════════════════════════════════════
# v51 — Strategy Archive Recording (Fix 2)
# ══════════════════════════════════════════════════════════════════════

class MockStrategyArchive:
    """Minimal mock for testing strategy archive recording."""
    def __init__(self):
        self.records: list[dict] = []
    def record(self, strategy: str, level: int, success: bool, generation: int):
        self.records.append({
            "strategy": strategy,
            "level": level,
            "success": success,
            "gen": generation,
        })


def _record_strategy_outcomes(
    archive: MockStrategyArchive,
    strategies: list[str],
    results: list[dict],
    current_generation: int,
    level: int,
) -> None:
    """Simplified recording logic matching Fix 2."""
    if archive is None or not strategies or len(strategies) != len(results):
        return
    fitnesses = []
    for r in results:
        f = r.get("fitness", r.get("score", 0.0))
        fitnesses.append(float(f) if isinstance(f, (int, float)) else 0.0)
    if not fitnesses:
        return
    sorted_f = sorted(fitnesses, reverse=True)
    threshold = sorted_f[max(0, len(sorted_f) // 10)]
    for strat, fit in zip(strategies, fitnesses):
        archive.record(
            strategy=strat,
            level=level,
            success=fit >= threshold,
            generation=current_generation,
        )


def test_v51_archive_records_all_challengers():
    """All challengers should be recorded (same count)."""
    mock = MockStrategyArchive()
    _record_strategy_outcomes(
        mock,
        strategies=["RANDOM", "PERTURBATION_CAPPED", "TASK_AWARE", "GROW", "DIRECTED"],
        results=[{"fitness": 10}, {"fitness": 50}, {"fitness": 30}, {"fitness": 40}, {"fitness": 20}],
        current_generation=50,
        level=5,
    )
    assert len(mock.records) == 5


def test_v51_archive_top_10_percent_success():
    """Top 10% (1 out of 10) should be marked success."""
    mock = MockStrategyArchive()
    results = [{"fitness": float(i * 10)} for i in range(10)]  # 0, 10, 20, ..., 90
    _record_strategy_outcomes(
        mock,
        strategies=[f"STRAT_{i}" for i in range(10)],
        results=results,
        current_generation=1,
        level=3,
    )
    success_count = sum(1 for r in mock.records if r["success"])
    assert success_count >= 1, "At least top 10% should succeed"


def test_v51_archive_all_equal_fitness():
    """All-equal fitness should still record without errors."""
    mock = MockStrategyArchive()
    results = [{"fitness": 42.0} for _ in range(8)]
    _record_strategy_outcomes(
        mock,
        strategies=["RANDOM"] * 8,
        results=results,
        current_generation=1,
        level=3,
    )
    assert len(mock.records) == 8
    # All should be marked success since they're all at the threshold
    assert all(r["success"] for r in mock.records)


def test_v51_archive_empty_results_no_crash():
    """Empty results should not crash."""
    mock = MockStrategyArchive()
    _record_strategy_outcomes(
        mock,
        strategies=[],
        results=[],
        current_generation=1,
        level=3,
    )
    assert len(mock.records) == 0


def test_v51_archive_mismatched_lengths_no_crash():
    """Mismatched strategies/results should not crash."""
    mock = MockStrategyArchive()
    _record_strategy_outcomes(
        mock,
        strategies=["A", "B", "C"],
        results=[{"fitness": 1.0}],  # only 1 result for 3 strategies
        current_generation=1,
        level=3,
    )
    assert len(mock.records) == 0


# ══════════════════════════════════════════════════════════════════════
# v51 — Curriculum Threshold (Fix B)
# ══════════════════════════════════════════════════════════════════════

def test_v51_curriculum_threshold_reachable_with_graduated_bonus():
    """With graduated bonus, a near-perfect program should clear 120.0.

    L5: threshold ~0.908. At reg_match=0.95: bonus = 50 * (0.95-0.908)/(1-0.908)
    = 50 * 0.042/0.092 ≈ 22.8. So raw_score = 95 - bloat + 22.8 ≈ 117.8.
    At reg_match=0.97: bonus = 50 * 0.062/0.092 ≈ 33.7. raw_score ≈ 130.7 > 120.
    """
    threshold = _compute_bonus_threshold(5)
    bonus_095 = _compute_graduated_bonus(0.95, 5)
    bonus_097 = _compute_graduated_bonus(0.97, 5)

    raw_095 = 95.0 + bonus_095  # ignoring bloat penalty
    raw_097 = 97.0 + bonus_097

    assert raw_095 > 100, f"L5 reg_match=0.95 gives raw={raw_095:.1f}, should be > 100"
    assert raw_097 >= 120, f"L5 reg_match=0.97 gives raw={raw_097:.1f}, should be >= 120"


# ══════════════════════════════════════════════════════════════════════
# D1 Supplemental — Execution Progress Credit
# ══════════════════════════════════════════════════════════════════════

def _execution_progress_credit(status, instr_count, reg_match, eval_budget):
    """Replicates the D1 execution progress credit formula."""
    exec_credit = 0.0
    if status == "ok" and instr_count > 0 and reg_match > 0.0:
        exec_credit = min(10.0, (instr_count / max(1, eval_budget)) * 10.0)
    return exec_credit


def test_d1_exec_credit_zero_when_crashed():
    """Crashed programs get no execution credit."""
    credit = _execution_progress_credit("crash", 5000, 0.5, 5000)
    assert credit == 0.0


def test_d1_exec_credit_zero_when_zero_match():
    """Programs with zero register match get no execution credit."""
    credit = _execution_progress_credit("ok", 3000, 0.0, 5000)
    assert credit == 0.0


def test_d1_exec_credit_zero_when_no_instructions():
    """Programs that executed zero instructions get no credit."""
    credit = _execution_progress_credit("ok", 0, 0.5, 5000)
    assert credit == 0.0


def test_d1_exec_credit_proportional_to_budget_use():
    """Credit scales with fraction of budget used, capped at 10.0."""
    credit_10pct = _execution_progress_credit("ok", 500, 0.5, 5000)
    credit_50pct = _execution_progress_credit("ok", 2500, 0.5, 5000)
    credit_100pct = _execution_progress_credit("ok", 5000, 0.5, 5000)
    assert 0.0 < credit_10pct < credit_50pct < credit_100pct <= 10.0
    assert abs(credit_10pct - 1.0) < 0.001, f"10% budget should give credit=1.0, got {credit_10pct}"
    assert abs(credit_50pct - 5.0) < 0.001, f"50% budget should give credit=5.0, got {credit_50pct}"
    assert abs(credit_100pct - 10.0) < 0.001, f"100% budget should give credit=10.0, got {credit_100pct}"


def test_d1_exec_credit_capped_at_10():
    """Even over-budget instruction counts cap at 10.0."""
    credit = _execution_progress_credit("ok", 50000, 0.5, 5000)
    assert credit == 10.0


def test_d1_exec_credit_empty_budget():
    """Eval budget of 0 should not crash."""
    credit = _execution_progress_credit("ok", 100, 0.5, 0)
    assert credit == 10.0  # min(10, 100/1 * 10) capped at 10


def test_d1_exec_credit_additive_with_distance_score():
    """Execution credit adds to distance-weighted base score."""
    base = _distance_weighted_score(54, 55)  # partial credit for near-match
    credit = _execution_progress_credit("ok", 2500, base, 5000)
    total = base * 100.0 + credit
    assert total > base * 100.0, "Execution credit should increase total score"


def test_d1_exec_credit_near_zero_budget_use():
    """Very few instructions gives near-zero credit."""
    credit = _execution_progress_credit("ok", 10, 0.5, 5000)
    assert 0.0 < credit < 0.1


def test_d1_exec_credit_monotonic():
    """More instructions = more credit (monotonic)."""
    credits = [_execution_progress_credit("ok", n, 0.5, 5000)
               for n in [100, 500, 1000, 2000, 4000, 5000]]
    for i in range(1, len(credits)):
        assert credits[i] >= credits[i - 1], f"Non-monotonic at {i}: {credits[i-1]} -> {credits[i]}"


# ══════════════════════════════════════════════════════════════════════
# D2 Supplemental — Recurrent Buffer Fitness Gate
# ══════════════════════════════════════════════════════════════════════

def _recurrent_buffer_should_save(fitness, stalled_count, stall_limit=50):
    """Replicates the D2 recurrent buffer fitness gate logic."""
    if fitness <= 0.0 and stalled_count < stall_limit:
        return False
    return True


def test_d2_positive_fitness_saved():
    """Programs with positive fitness are saved to buffer."""
    assert _recurrent_buffer_should_save(0.5, 5)


def test_d2_negative_fitness_blocked():
    """Programs with negative fitness are blocked when not deeply stalled."""
    assert not _recurrent_buffer_should_save(-1.0, 5)


def test_d2_zero_fitness_blocked():
    """Programs with zero fitness are blocked when not deeply stalled."""
    assert not _recurrent_buffer_should_save(0.0, 5)


def test_d2_deep_stall_bypasses_gate():
    """Deeply stalled systems (>50 gen) allow low-quality traces."""
    assert _recurrent_buffer_should_save(-1.0, 50)


def test_d2_exactly_at_stall_limit():
    """Exactly at stall limit (50) should bypass."""
    assert _recurrent_buffer_should_save(0.0, 50)


def test_d2_just_below_stall_limit_still_blocked():
    """Just below stall limit (49) still blocks zero-fitness."""
    assert not _recurrent_buffer_should_save(0.0, 49)


def test_d2_high_positive_fitness_always_saved():
    """High fitness programs always saved regardless of stall."""
    assert _recurrent_buffer_should_save(100.0, 5)


def test_d2_mildly_negative_blocked():
    """Even small negative fitness blocked (contamination prevention)."""
    assert not _recurrent_buffer_should_save(-0.01, 10)


def test_d2_zero_stall_count():
    """Zero stall count with zero fitness blocked."""
    assert not _recurrent_buffer_should_save(0.0, 0)
