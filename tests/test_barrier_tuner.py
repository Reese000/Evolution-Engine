#!/usr/bin/env python3
"""Tests for barrier_tuner.BarrierTuner.

BarrierTuner drives autonomous-run auto-termination (should_terminate) and
barrier detection (fitness plateau / level stall / diversity collapse / val_loss
regression). It governs when unattended runs stop, but had no test coverage.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from barrier_tuner import BarrierTuner, BarrierType, CycleMetrics


def _feed(tuner, specs):
    """specs: list of dicts with cycle/fitness/level (+ optional overrides)."""
    last = None
    for i, s in enumerate(specs):
        m = CycleMetrics(
            cycle=s.get("cycle", i + 1),
            fitness=s["fitness"],
            level=s["level"],
            diversity=s.get("diversity", 0.3),
            val_loss=s.get("val_loss"),
        )
        last = tuner.update(m)
    return last


class TestShouldTerminate:
    def test_too_few_cycles(self):
        t = BarrierTuner()
        _feed(t, [{"fitness": 1.0, "level": 0}, {"fitness": 1.0, "level": 0}])
        assert t.should_terminate() is False

    def test_flat_fitness_and_level_terminates(self):
        t = BarrierTuner()
        _feed(t, [{"fitness": 100.0, "level": 5} for _ in range(15)])
        assert t.should_terminate() is True
        reason = t.get_termination_reason()
        assert "stagnation" in reason.lower()

    def test_level_advancement_prevents_termination(self):
        t = BarrierTuner()
        _feed(t, [{"fitness": 100.0, "level": i} for i in range(15)])
        assert t.should_terminate() is False

    def test_fitness_progress_prevents_termination(self):
        t = BarrierTuner()
        # fitness range = 14 (>= 2.0) over the window
        _feed(t, [{"fitness": float(i), "level": 5} for i in range(15)])
        assert t.should_terminate() is False

    def test_get_reason_empty_when_not_terminating(self):
        t = BarrierTuner()
        _feed(t, [{"fitness": float(i), "level": i} for i in range(5)])
        assert t.get_termination_reason() == ""


class TestBarrierDetection:
    def test_level_stall(self):
        t = BarrierTuner()
        info = _feed(t, [{"fitness": 100.0 + i * 5, "level": 5} for i in range(5)])
        # level never advances -> level stall (takes precedence)
        assert info.barrier == BarrierType.LEVEL_STALL
        assert t.level_stall_cycles >= 3

    def test_fitness_plateau(self):
        t = BarrierTuner()
        # level advances each cycle (no stall); fitness flat within 1.0
        info = _feed(t, [{"fitness": 50.0, "level": i} for i in range(6)])
        assert info.barrier == BarrierType.FITNESS_PLATEAU

    def test_diversity_collapse(self):
        t = BarrierTuner()
        # level advances, fitness varies widely, diversity below threshold
        info = _feed(
            t,
            [{"fitness": 10.0 * i, "level": i, "diversity": 0.05} for i in range(1, 4)],
        )
        assert info.barrier == BarrierType.DIVERSITY_COLLAPSE
        assert t.diversity_collapse_cycles >= 2

    def test_val_loss_regression(self):
        t = BarrierTuner()
        info = _feed(
            t,
            [
                {"fitness": 10.0, "level": 1, "val_loss": 2.0},
                {"fitness": 20.0, "level": 2, "val_loss": 3.0},  # > 2.0*1.1
                {"fitness": 30.0, "level": 3, "val_loss": 3.0},  # regression cycle 2
            ],
        )
        assert info.barrier == BarrierType.VAL_LOSS_REGRESSION
        assert t.val_loss_regression_cycles >= 2

    def test_normal_when_progressing(self):
        t = BarrierTuner()
        info = _feed(t, [{"fitness": 10.0 * i, "level": i, "diversity": 0.5} for i in range(1, 4)])
        assert info.barrier == BarrierType.NORMAL


class TestStatePersistence:
    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "barrier_state.json")
        t = BarrierTuner()
        _feed(t, [{"fitness": 100.0, "level": 5} for _ in range(5)])
        t.save_state(path)

        t2 = BarrierTuner()
        assert t2.load_state(path) is True
        assert t2.level_stall_cycles == t.level_stall_cycles
        assert t2.fitness_plateau_cycles == t.fitness_plateau_cycles
        assert len(t2.history) == len(t.history)

    def test_load_missing_returns_false(self):
        t = BarrierTuner()
        assert t.load_state("/nonexistent/barrier_state.json") is False
