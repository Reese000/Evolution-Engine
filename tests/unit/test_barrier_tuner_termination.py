"""Tests for BarrierTuner batch auto-termination."""

from __future__ import annotations

from barrier_tuner import BarrierTuner, CycleMetrics


def _terminal_metrics(cycle: int) -> CycleMetrics:
    return CycleMetrics(
        cycle=cycle,
        fitness=264.04,
        level=30,
        champion_words=73,
        verify_score=1.0,
    )


def test_should_not_terminate_certified_terminal_plateau():
    tuner = BarrierTuner()
    for cycle in range(1, 20):
        tuner.history.append(_terminal_metrics(cycle))
    assert tuner.should_terminate() is False


def test_should_terminate_low_level_stagnation():
    tuner = BarrierTuner()
    for cycle in range(1, 20):
        tuner.history.append(CycleMetrics(cycle=cycle, fitness=50.0, level=22, verify_score=0.6))
    assert tuner.should_terminate() is True


def test_auto_terminate_disabled_by_env(monkeypatch):
    monkeypatch.setenv("EE_BARRIER_AUTO_TERMINATE", "0")
    tuner = BarrierTuner()
    for cycle in range(1, 20):
        tuner.history.append(CycleMetrics(cycle=cycle, fitness=50.0, level=22, verify_score=0.6))
    assert tuner.should_terminate() is False


def test_tune_archive_skips_gen_escalation_at_terminal_plateau():
    from barrier_tuner import BarrierInfo, BarrierType

    tuner = BarrierTuner()
    for cycle in range(1, 6):
        tuner.history.append(_terminal_metrics(cycle))
    tuner._last_barrier_info = BarrierInfo(barrier=BarrierType.LEVEL_STALL, consecutive_count=10, severity=0.8)
    assert tuner.tune_archive_params() == {}


def test_tune_archive_escalates_gens_below_terminal_plateau():
    from barrier_tuner import BarrierInfo, BarrierType

    tuner = BarrierTuner()
    for cycle in range(1, 6):
        tuner.history.append(CycleMetrics(cycle=cycle, fitness=120.0, level=22, verify_score=0.85))
    tuner._last_barrier_info = BarrierInfo(barrier=BarrierType.LEVEL_STALL, consecutive_count=10, severity=0.8)
    assert tuner.tune_archive_params() == {"max_generations": 80}
