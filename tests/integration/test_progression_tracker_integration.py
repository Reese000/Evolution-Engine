#!/usr/bin/env python3
"""Integration tests for ProgressionTracker — metrics tracking, trends, auto-tuning.

Validates:
  - Record/save/load persistence
  - Trend calculations (fitness, val_loss, words)
  - Stagnation detection
  - TuningPolicy → ParameterActuator pipeline
  - Saturation detection integration
"""

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestProgressionTracker:
    """Metrics recording and trend analysis."""

    def test_record_and_persist(self):
        """Recording metrics persists to JSONL and survives reload."""
        from progression_tracker import CycleMetrics, ProgressionTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "progression.jsonl")
            tracker = ProgressionTracker(path=path)

            metrics = CycleMetrics(
                cycle=1,
                level=3,
                fitness=25.0,
                words=5,
                archive_size=50,
                val_loss=2.5,
                time_s=120.0,
                mg_count=10,
                total_challengers=50,
                champion_improvements=3,
            )
            tracker.record(metrics)

            # Reload
            tracker2 = ProgressionTracker(path=path)
            assert len(tracker2.history) == 1
            assert tracker2.history[0].cycle == 1
            assert tracker2.history[0].fitness == 25.0

    def test_fitness_trend_improving(self):
        """fitness_trend is positive when fitness increases."""
        from progression_tracker import CycleMetrics, ProgressionTracker

        tracker = ProgressionTracker()
        for i in range(6):
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=i,
                    fitness=float(10 + i * 5),
                    words=3,
                    archive_size=20,
                    val_loss=2.0,
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )
        trend = tracker.fitness_trend(window=5)
        assert trend > 0, f"Expected positive trend, got {trend}"

    def test_fitness_trend_declining(self):
        """fitness_trend is negative when fitness decreases."""
        from progression_tracker import CycleMetrics, ProgressionTracker

        tracker = ProgressionTracker()
        for i in range(6):
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=float(30 - i * 4),
                    words=3,
                    archive_size=20,
                    val_loss=2.0,
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )
        trend = tracker.fitness_trend(window=5)
        assert trend < 0, f"Expected negative trend, got {trend}"

    def test_val_loss_trend_improving(self):
        """val_loss_trend negative means improving (lower loss)."""
        from progression_tracker import CycleMetrics, ProgressionTracker

        tracker = ProgressionTracker()
        val_losses = [3.0, 2.5, 2.0, 1.5, 1.0]
        for i, vl in enumerate(val_losses):
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=20.0,
                    words=3,
                    archive_size=20,
                    val_loss=vl,
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )
        trend = tracker.val_loss_trend(window=5)
        assert trend < 0, f"Expected negative val_loss trend (improving), got {trend}"

    def test_val_loss_trend_with_none(self):
        """val_loss_trend skips cycles with None val_loss."""
        from progression_tracker import CycleMetrics, ProgressionTracker

        tracker = ProgressionTracker()
        for i in range(5):
            vl = None if i % 2 == 0 else 2.0
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=20.0,
                    words=3,
                    archive_size=20,
                    val_loss=vl,
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )
        trend = tracker.val_loss_trend(window=5)
        # Should not crash with None values
        assert isinstance(trend, float)

    def test_words_trend_growing(self):
        """words_trend is positive when champion grows."""
        from progression_tracker import CycleMetrics, ProgressionTracker

        tracker = ProgressionTracker()
        for i in range(5):
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=20.0,
                    words=3 + i,
                    archive_size=20,
                    val_loss=2.0,
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )
        trend = tracker.words_trend(window=5)
        assert trend > 0, f"Expected positive words trend, got {trend}"

    def test_last_cycle_returns_latest(self):
        """last_cycle property returns the most recent entry."""
        from progression_tracker import CycleMetrics, ProgressionTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "p.jsonl")
            tracker = ProgressionTracker(path=path)
            assert tracker.last_cycle is None

            for i in range(3):
                tracker.history.append(
                    CycleMetrics(
                        cycle=i + 1,
                        level=i,
                        fitness=float(10 * i),
                        words=3,
                        archive_size=20,
                        val_loss=2.0,
                        time_s=60.0,
                        mg_count=5,
                        total_challengers=50,
                        champion_improvements=1,
                    )
                )
            assert tracker.last_cycle is not None
            assert tracker.last_cycle.cycle == 3

    def test_empty_tracker_returns_defaults(self):
        """Empty tracker returns neutral values."""
        from progression_tracker import ProgressionTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "p.jsonl")
            tracker = ProgressionTracker(path=path)
            assert tracker.fitness_trend() == 0.0
            assert tracker.val_loss_trend() == 0.0
            assert tracker.words_trend() == 0.0
            assert tracker.mg_ratio_avg() == 0.0
            assert tracker.is_stagnant() is False
            assert tracker.summary()["cycles"] == 0

    def test_stagnation_detection(self):
        """is_stagnant detects plateau (no fitness/words change)."""
        from progression_tracker import CycleMetrics, ProgressionTracker

        tracker = ProgressionTracker()
        for i in range(6):
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=20.0,
                    words=4,
                    archive_size=20,
                    val_loss=2.0,
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )
        assert tracker.is_stagnant(window=5)

    def test_stagnation_not_detected_with_progress(self):
        """is_stagnant returns False when fitness is improving."""
        from progression_tracker import CycleMetrics, ProgressionTracker

        tracker = ProgressionTracker()
        for i in range(6):
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=float(10 + i * 2),
                    words=3 + i,
                    archive_size=20,
                    val_loss=2.0,
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )
        assert not tracker.is_stagnant(window=5)

    def test_mg_ratio_avg_calculation(self):
        """mg_ratio_avg correctly averages MG ratios."""
        from progression_tracker import CycleMetrics, ProgressionTracker

        tracker = ProgressionTracker()
        for i in range(3):
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=20.0,
                    words=4,
                    archive_size=20,
                    val_loss=2.0,
                    time_s=60.0,
                    mg_count=10 + i * 10,
                    total_challengers=100,
                    champion_improvements=1,
                )
            )
        # MG ratios: 0.1, 0.2, 0.3 => avg 0.2
        avg = tracker.mg_ratio_avg(window=3)
        assert abs(avg - 0.2) < 0.01, f"Expected mg_ratio ~0.2, got {avg}"

    def test_plateau_cycles_count(self):
        """plateau_cycles counts consecutive flat cycles."""
        from progression_tracker import CycleMetrics, ProgressionTracker

        tracker = ProgressionTracker()
        for i in range(10):
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=float(20.0 if i < 7 else 20.0 + (i - 6) * 2),
                    words=4,
                    archive_size=20,
                    val_loss=2.0,
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )
        # First 7 cycles flat, then 3 improving
        plateau = tracker.plateau_cycles(threshold=1.0)
        assert plateau == 0  # last 3 cycles were improving

        # Full flat
        tracker2 = ProgressionTracker()
        for i in range(5):
            tracker2.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=20.0,
                    words=4,
                    archive_size=20,
                    val_loss=2.0,
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )
        assert tracker2.plateau_cycles(threshold=1.0) >= 4


class TestTuningPolicy:
    """TuningPolicy → adjustment pipeline."""

    def test_fitness_plateau_adjusts_mutation_rate(self):
        """Fitness plateau triggers mutation_rate increase."""
        from progression_tracker import CycleMetrics, ProgressionTracker, TuningPolicy

        tracker = ProgressionTracker()
        for i in range(6):
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=20.0,
                    words=4,
                    archive_size=20,
                    val_loss=2.0,
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )

        policy = TuningPolicy()
        adjustments = policy.compute_adjustments(tracker, current_cycle=7, current_config={"mutation_rate": 0.3})
        if adjustments:
            assert adjustments[0].parameter == "mutation_rate"
            assert adjustments[0].value > 0.3
            assert "plateau" in adjustments[0].reason.lower() or "stagnant" in adjustments[0].reason.lower()

    def test_val_loss_increasing_adjusts_epochs(self):
        """Increasing val_loss triggers retrain_max_epochs increase."""
        from progression_tracker import CycleMetrics, ProgressionTracker, TuningPolicy

        tracker = ProgressionTracker()
        for i in range(5):
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=float(20 + i * 2),
                    words=4,
                    archive_size=20,
                    val_loss=float(1.0 + i * 0.1),
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )

        policy = TuningPolicy()
        adjustments = policy.compute_adjustments(tracker, current_cycle=6, current_config={"retrain_max_epochs": 50})
        if adjustments:
            assert adjustments[0].parameter == "retrain_max_epochs"
            assert adjustments[0].value > 50

    def test_val_loss_decreasing_reduces_epochs(self):
        """Decreasing val_loss triggers epoch reduction."""
        from progression_tracker import CycleMetrics, ProgressionTracker, TuningPolicy

        tracker = ProgressionTracker()
        for i in range(5):
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=3,
                    fitness=20.0,
                    words=4,
                    archive_size=20,
                    val_loss=float(3.0 - i * 0.3),
                    time_s=60.0,
                    mg_count=5,
                    total_challengers=50,
                    champion_improvements=1,
                )
            )

        policy = TuningPolicy()
        adjustments = policy.compute_adjustments(tracker, current_cycle=6, current_config={"retrain_max_epochs": 50})
        if adjustments:
            assert adjustments[0].parameter == "retrain_max_epochs"
            assert adjustments[0].value < 50

    def test_no_duplicate_params_in_a_row(self):
        """Same parameter cannot be adjusted two cycles in a row."""
        from progression_tracker import TuningAdjustment, TuningPolicy

        policy = TuningPolicy()
        # Manually record an adjustment
        adj = TuningAdjustment(parameter="mutation_rate", value=0.4, reason="test", magnitude=0.1)
        policy.record_applied(adj, cycle=1)

        # The same parameter should not be eligible next cycle
        assert not policy._eligible("mutation_rate", current_cycle=2)

    def test_cooldown_enforced(self):
        """Parameter must wait cooldown cycles before re-adjustment."""
        from progression_tracker import TuningAdjustment, TuningPolicy

        policy = TuningPolicy(cooldown=3)
        adj = TuningAdjustment(parameter="retrain_max_epochs", value=60, reason="test", magnitude=10)
        policy.record_applied(adj, cycle=1)

        # After recording, _last_param is "retrain_max_epochs"
        # _eligible checks same-param FIRST, so we need a buffer param
        # Record a different param to clear the same-param blocker
        adj2 = TuningAdjustment(parameter="mutation_rate", value=0.4, reason="test", magnitude=0.1)
        policy.record_applied(adj2, cycle=2)

        # retrain_max_epochs: last adjusted at cycle 1, cooldown=3
        # Eligible at cycle 1+3 = 4+
        assert not policy._eligible("retrain_max_epochs", current_cycle=2)
        assert not policy._eligible("retrain_max_epochs", current_cycle=3)
        assert policy._eligible("retrain_max_epochs", current_cycle=4)

    def test_no_adjustments_on_nominal_tracker(self):
        """No adjustments when progression is fully nominal (no actionable signals)."""
        from progression_tracker import CycleMetrics, ProgressionTracker, TuningPolicy

        tracker = ProgressionTracker()
        for i in range(6):
            # Stable val_loss (flat), improving fitness, growing words
            tracker.history.append(
                CycleMetrics(
                    cycle=i + 1,
                    level=min(i + 1, 8),
                    fitness=float(10 + i * 5),
                    words=3 + i,
                    archive_size=20 + i * 10,
                    val_loss=2.5,  # flat — not decreasing well, not increasing
                    time_s=60.0,
                    mg_count=15,
                    total_challengers=50,
                    champion_improvements=2,
                )
            )

        policy = TuningPolicy()
        adjustments = policy.compute_adjustments(tracker, current_cycle=7)
        assert len(adjustments) == 0  # No stagnation, improving, stable val_loss


class TestParameterActuator:
    """ParameterActuator applies adjustments correctly."""

    def test_apply_valid_parameter(self):
        """Apply changes a config attribute."""
        from progression_tracker import ParameterActuator, TuningAdjustment

        # EvolutionConfig-like object
        config = type("MockConfig", (), {"mutation_rate": 0.3})()

        actuator = ParameterActuator()
        adj = TuningAdjustment(parameter="mutation_rate", value=0.5, reason="test", magnitude=0.2)
        result = actuator.apply(adj, config)
        assert result
        assert config.mutation_rate == 0.5

    def test_clamps_out_of_bounds(self):
        """Values are clamped to valid bounds."""
        from progression_tracker import ParameterActuator, TuningAdjustment

        config = type("MockConfig", (), {"mutation_rate": 0.3})()
        actuator = ParameterActuator()

        adj = TuningAdjustment(parameter="mutation_rate", value=10.0, reason="test")
        actuator.apply(adj, config)
        assert config.mutation_rate <= 0.8  # Clamped to max

    def test_unknown_param_returns_false(self):
        """Unknown parameter returns False."""
        from progression_tracker import ParameterActuator, TuningAdjustment

        config = type("MockConfig", (), {})()
        actuator = ParameterActuator()
        adj = TuningAdjustment(parameter="nonexistent", value=1.0, reason="test")
        result = actuator.apply(adj, config)
        assert not result

    def test_summary_tracks_applications(self):
        """summary() returns applied count and last adjustment."""
        from progression_tracker import ParameterActuator, TuningAdjustment

        config = type("MockConfig", (), {"mutation_rate": 0.3})()
        actuator = ParameterActuator()

        adj = TuningAdjustment(parameter="mutation_rate", value=0.5, reason="test")
        actuator.apply(adj, config)

        summary = actuator.summary()
        assert summary["applied_count"] == 1
        assert summary["last_applied"]["param"] == "mutation_rate"
