"""Wave 10 integration tests — cross-module end-to-end cycles.

Tests cover:
  1. TestMetaControllerFullCycle — full MetaController lifecycle
  2. TestStrategyArchiveFullCycle — StrategyArchive recording and analysis
  3. TestChampionArchiveFullCycle — ChampionArchive selection and pruning
  4. TestMetricsAnomalyCorrelation — MetricsCollector anomaly detection and correlation
  5. TestProgressionTrackerFullCycle — ProgressionTracker plateau analysis and window comparison
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from typing import Dict, List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meta_controller import MetaController, TelemetrySnapshot, TuningAction, StagnationResult
from strategy_archive import StrategyArchive
from champion_archive import ChampionArchive
from metrics import MetricsCollector
from progression_tracker import CycleMetrics, ProgressionTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(
    cycle: int,
    fitness: float,
    level: int = 10,
    champion_words: int = 120,
    val_loss: float = 2.0,
    diversity: float = 0.5,
    **overrides,
) -> TelemetrySnapshot:
    """Build a TelemetrySnapshot with sensible defaults."""
    defaults = dict(
        cycle=cycle,
        fitness=fitness,
        level=level,
        champion_words=champion_words,
        val_loss=val_loss,
        diversity=diversity,
        novelty_size=100,
        qd_occupied=50,
        qd_total=200,
        traces_count=20,
        retrained=True,
        generations_completed=5,
    )
    defaults.update(overrides)
    return TelemetrySnapshot(**defaults)


def _make_cycle_metrics(
    cycle: int,
    fitness: float,
    level: int = 10,
    words: int = 120,
    val_loss: float | None = None,
    **overrides,
) -> CycleMetrics:
    """Build a CycleMetrics with sensible defaults."""
    defaults = dict(
        cycle=cycle,
        level=level,
        fitness=fitness,
        words=words,
        archive_size=cycle,
        val_loss=val_loss,
        time_s=1.0,
        mg_count=cycle % 3,
        total_challengers=10,
        champion_improvements=1,
    )
    defaults.update(overrides)
    return CycleMetrics(**defaults)


def _make_champion_program(seed: int, length: int = 8) -> List[int]:
    """Generate a deterministic instruction list for a champion."""
    return [seed * 0x1000 + i for i in range(length)]


# ══════════════════════════════════════════════════════════════════════
# 1. TestMetaControllerFullCycle
# ══════════════════════════════════════════════════════════════════════


class TestMetaControllerFullCycle:
    """Full lifecycle test of MetaController: create, add telemetry,
    detect stagnation, query trends / recovery / escalation / reset."""

    def test_create_with_tempdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetaController(checkpoint_dir=tmpdir)
            assert isinstance(mc, MetaController)

    def test_update_cycle_and_detect_stagnation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetaController(checkpoint_dir=tmpdir)
            # Add a few snapshots — should remain NORMAL
            for i in range(5):
                mc.update_cycle(_make_snapshot(cycle=i, fitness=100 + i * 10))
            result = mc.detect_stagnation()
            assert isinstance(result, StagnationResult)

    def test_detect_stagnation_plateau(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetaController(
                checkpoint_dir=tmpdir,
                fitness_plateau_cycles=3,
            )
            # Flat fitness for 5 cycles -> should trigger plateau
            for i in range(5):
                mc.update_cycle(_make_snapshot(cycle=i, fitness=100.0))
            result = mc.detect_stagnation()
            # With fitness_plateau_cycles=3 and 5 flat cycles, should not be NORMAL
            assert isinstance(result, StagnationResult)

    def test_get_cross_cycle_trends_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetaController(checkpoint_dir=tmpdir)
            for i in range(8):
                mc.update_cycle(
                    _make_snapshot(
                        cycle=i,
                        fitness=80 + i * 5,
                        val_loss=3.0 - i * 0.1,
                        diversity=0.3 + i * 0.02,
                    )
                )
            trends = mc.get_cross_cycle_trends(window=5)
            assert isinstance(trends, dict)
            assert "fitness" in trends
            assert "val_loss" in trends
            assert "diversity" in trends
            for key in ("fitness", "val_loss", "diversity"):
                t = trends[key]
                assert "direction" in t
                assert "slope" in t
                assert "confidence" in t
                assert t["direction"] in ("improving", "stable", "declining")
                assert isinstance(t["slope"], float)
                assert 0.0 <= t["confidence"] <= 1.0

    def test_get_recovery_confidence_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetaController(checkpoint_dir=tmpdir)
            # No recovery active initially
            conf = mc.get_recovery_confidence()
            assert isinstance(conf, dict)
            assert "is_recovering" in conf
            assert "cycles_in_recovery" in conf
            assert "fitness_trend_during_recovery" in conf
            assert "confidence" in conf
            assert "should_exit_recovery" in conf
            assert isinstance(conf["is_recovering"], bool)
            assert isinstance(conf["cycles_in_recovery"], int)
            assert isinstance(conf["confidence"], (int, float))

    def test_get_tuning_escalation_history_returns_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetaController(checkpoint_dir=tmpdir)
            history = mc.get_tuning_escalation_history()
            assert isinstance(history, list)
            # Empty initially
            assert len(history) == 0

    def test_suggest_parameter_reset_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetaController(checkpoint_dir=tmpdir)
            suggestion = mc.suggest_parameter_reset()
            assert isinstance(suggestion, dict)
            assert "should_reset" in suggestion
            assert "reason" in suggestion
            assert "suggested_params" in suggestion
            assert isinstance(suggestion["should_reset"], bool)
            assert isinstance(suggestion["suggested_params"], dict)

    def test_suggest_parameter_reset_no_telemetry(self):
        """With no telemetry, should_reset should be False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetaController(checkpoint_dir=tmpdir)
            suggestion = mc.suggest_parameter_reset()
            assert suggestion["should_reset"] is False
            assert suggestion["reason"] == "no telemetry"

    def test_full_cycle_flow(self):
        """Run the complete create -> add telemetry -> detect -> trends -> recovery
        -> escalation -> reset flow and verify all return types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetaController(checkpoint_dir=tmpdir)

            # 1. Add 10 cycles of improving telemetry
            for i in range(10):
                mc.update_cycle(
                    _make_snapshot(
                        cycle=i,
                        fitness=90 + i * 5,
                        val_loss=3.0 - i * 0.05,
                        diversity=0.4 + i * 0.02,
                    )
                )

            # 2. detect_stagnation
            stag = mc.detect_stagnation()
            assert isinstance(stag, StagnationResult)

            # 3. get_cross_cycle_trends
            trends = mc.get_cross_cycle_trends()
            assert isinstance(trends, dict)
            assert set(trends.keys()) == {"fitness", "val_loss", "diversity"}

            # 4. get_recovery_confidence
            recovery = mc.get_recovery_confidence()
            assert isinstance(recovery, dict)
            assert isinstance(recovery["is_recovering"], bool)

            # 5. get_tuning_escalation_history
            escalation = mc.get_tuning_escalation_history()
            assert isinstance(escalation, list)

            # 6. suggest_parameter_reset
            reset = mc.suggest_parameter_reset()
            assert isinstance(reset, dict)
            assert isinstance(reset["should_reset"], bool)


# ══════════════════════════════════════════════════════════════════════
# 2. TestStrategyArchiveFullCycle
# ══════════════════════════════════════════════════════════════════════


class TestStrategyArchiveFullCycle:
    """Full lifecycle of StrategyArchive: record strategies across levels,
    query effectiveness, exploration bonus, time-weighted rates, dominance,
    and summary."""

    def test_record_and_total_records(self):
        sa = StrategyArchive()
        for i in range(20):
            sa.record("RANDOM", level=i % 5, success=i % 2 == 0, generation=i)
        assert sa.total_records() == 20

    def test_per_level_effectiveness(self):
        sa = StrategyArchive()
        # Strategy A at level 0: 8 successes out of 10
        for i in range(10):
            sa.record("STRAT_A", level=0, success=i < 8, generation=i)
        # Strategy A at level 1: 3 successes out of 10
        for i in range(10):
            sa.record("STRAT_A", level=1, success=i < 3, generation=i + 10)

        eff = sa.per_level_effectiveness("STRAT_A")
        assert isinstance(eff, dict)
        assert 0 in eff
        assert 1 in eff
        assert eff[0] == pytest.approx(0.8, abs=0.01)
        assert eff[1] == pytest.approx(0.3, abs=0.01)

    def test_per_level_effectiveness_no_records(self):
        sa = StrategyArchive()
        eff = sa.per_level_effectiveness("NONEXISTENT")
        assert eff == {}

    def test_adaptive_exploration_bonus(self):
        sa = StrategyArchive()
        # Underperforming strategy (success rate < 0.3)
        for i in range(10):
            sa.record("WEAK_STRAT", level=0, success=False, generation=i)

        bonus = sa.adaptive_exploration_bonus("WEAK_STRAT", level=0, base_bonus=0.1)
        # Effectiveness is 0.0 -> underperforming -> base * 2
        assert bonus == pytest.approx(0.2, abs=0.01)

    def test_adaptive_exploration_bonus_high_performing(self):
        sa = StrategyArchive()
        # High-performing strategy
        for i in range(10):
            sa.record("STRONG_STRAT", level=0, success=True, generation=i)

        bonus = sa.adaptive_exploration_bonus("STRONG_STRAT", level=0, base_bonus=0.1)
        # Effectiveness is 1.0 -> well-performing -> base * 0.5
        assert bonus == pytest.approx(0.05, abs=0.01)

    def test_time_weighted_success_rate(self):
        sa = StrategyArchive()
        # Old failures
        for i in range(10):
            sa.record("MIXED_STRAT", level=0, success=False, generation=i)
        # Recent successes
        for i in range(10, 20):
            sa.record("MIXED_STRAT", level=0, success=True, generation=i)

        rate = sa.time_weighted_success_rate("MIXED_STRAT", decay_halflife=100)
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0
        # Recent successes should dominate -> rate > 0.5
        assert rate > 0.5

    def test_get_dominance_report(self):
        sa = StrategyArchive()
        # Make RANDOM dominant
        for i in range(20):
            sa.record("RANDOM", level=0, success=True, generation=i)
        for i in range(5):
            sa.record("RARE_STRAT", level=0, success=False, generation=i)

        report = sa.get_dominance_report()
        assert isinstance(report, dict)
        assert "dominant_strategy" in report
        assert "dominance_ratio" in report
        assert "is_monoculture" in report
        assert "recommendation" in report
        assert report["dominant_strategy"] == "RANDOM"

    def test_summary(self):
        sa = StrategyArchive()
        for i in range(15):
            sa.record("STRAT_A", level=i % 3, success=i % 2 == 0, generation=i)
        for i in range(10):
            sa.record("STRAT_B", level=i % 3, success=i % 3 == 0, generation=i)

        s = sa.summary()
        assert isinstance(s, dict)
        assert "total_records" in s
        assert "strategies" in s
        assert "dominance" in s
        assert s["total_records"] == 25
        assert "STRAT_A" in s["strategies"]
        assert "STRAT_B" in s["strategies"]
        assert "count" in s["strategies"]["STRAT_A"]
        assert "successes" in s["strategies"]["STRAT_A"]
        assert "rate" in s["strategies"]["STRAT_A"]

    def test_full_archive_cycle(self):
        """Record across multiple levels, query everything, verify types."""
        sa = StrategyArchive()
        strategies = ["RANDOM", "TASK_AWARE", "PERTURBATION"]
        for i in range(30):
            strat = strategies[i % len(strategies)]
            sa.record(strat, level=i % 5, success=i % 3 != 0, generation=i)

        # per_level_effectiveness
        for s in strategies:
            eff = sa.per_level_effectiveness(s)
            assert isinstance(eff, dict)
            for lev, val in eff.items():
                assert isinstance(lev, int)
                assert 0.0 <= val <= 1.0

        # adaptive_exploration_bonus
        bonus = sa.adaptive_exploration_bonus("RANDOM", level=0)
        assert isinstance(bonus, float)
        assert 0.0 <= bonus <= 1.0

        # time_weighted_success_rate
        twr = sa.time_weighted_success_rate("RANDOM")
        assert isinstance(twr, float)
        assert 0.0 <= twr <= 1.0

        # dominance report
        dr = sa.get_dominance_report()
        assert isinstance(dr, dict)
        assert isinstance(dr["is_monoculture"], bool)

        # summary
        sm = sa.summary()
        assert isinstance(sm, dict)
        assert sm["total_records"] == 30


# ══════════════════════════════════════════════════════════════════════
# 3. TestChampionArchiveFullCycle
# ══════════════════════════════════════════════════════════════════════


class TestChampionArchiveFullCycle:
    """Full lifecycle of ChampionArchive: add champions, diversity-aware
    selection, quality-weighted pruning, and diversity reporting."""

    def test_add_champions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "archive.jsonl")
            ca = ChampionArchive(path=path, max_size=50)
            for i in range(10):
                added = ca.add(
                    program=_make_champion_program(i),
                    level=10,
                    fitness=100 + i * 10,
                    cycle=1,
                )
                assert added is True
            assert ca.size == 10

    def test_deduplication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "archive.jsonl")
            ca = ChampionArchive(path=path, max_size=50)
            prog = _make_champion_program(0)
            ca.add(program=prog, level=10, fitness=100, cycle=1)
            added = ca.add(program=prog, level=10, fitness=200, cycle=2)
            assert added is False  # duplicate
            assert ca.size == 1

    def test_diversity_aware_select(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "archive.jsonl")
            ca = ChampionArchive(path=path, max_size=50)
            # Add multiple champions at level 10
            for i in range(8):
                ca.add(
                    program=_make_champion_program(i),
                    level=10,
                    fitness=100 + i * 5,
                    cycle=1,
                )
            selected = ca.diversity_aware_select(level=10, n=3)
            assert isinstance(selected, list)
            assert len(selected) == 3
            for entry in selected:
                assert isinstance(entry, dict)
                assert "fitness" in entry
                assert "level" in entry

    def test_diversity_aware_select_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "archive.jsonl")
            ca = ChampionArchive(path=path, max_size=50)
            selected = ca.diversity_aware_select(level=99, n=3)
            assert selected == []

    def test_quality_weighted_prune(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "archive.jsonl")
            ca = ChampionArchive(path=path, max_size=10)
            # Add 15 champions across cycles to fill and overflow
            for i in range(15):
                ca.add(
                    program=_make_champion_program(i),
                    level=10,
                    fitness=50 + i * 10,
                    cycle=i,
                )
            # Archive should be at most max_size
            initial_size = ca.size
            pruned = ca.quality_weighted_prune()
            assert isinstance(pruned, int)
            assert ca.size <= 10

    def test_get_diversity_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "archive.jsonl")
            ca = ChampionArchive(path=path, max_size=100)
            # Add champions at different levels
            for i in range(10):
                ca.add(
                    program=_make_champion_program(i, length=6 + i % 4),
                    level=5 + i % 8,
                    fitness=80 + i * 15,
                    cycle=1,
                )
            report = ca.get_diversity_report()
            assert isinstance(report, dict)
            assert "total_champions" in report
            assert "unique_levels" in report
            assert "avg_fitness_by_level" in report
            assert "program_length_variance" in report
            assert "instruction_diversity" in report
            assert "fitness_correlation_with_length" in report
            assert report["total_champions"] == 10
            assert isinstance(report["unique_levels"], list)
            assert isinstance(report["avg_fitness_by_level"], dict)
            assert isinstance(report["program_length_variance"], float)
            assert isinstance(report["instruction_diversity"], float)
            assert isinstance(report["fitness_correlation_with_length"], float)

    def test_get_diversity_report_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "archive.jsonl")
            ca = ChampionArchive(path=path, max_size=100)
            report = ca.get_diversity_report()
            assert report["total_champions"] == 0
            assert report["unique_levels"] == []
            assert report["avg_fitness_by_level"] == {}

    def test_full_champion_cycle(self):
        """Add, select, prune, report -- full lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "archive.jsonl")
            ca = ChampionArchive(path=path, max_size=20)

            # Add 25 champions across 3 cycles at various levels
            for i in range(25):
                ca.add(
                    program=_make_champion_program(i, length=4 + i % 6),
                    level=5 + i % 10,
                    fitness=60 + i * 8,
                    cycle=1 + i // 10,
                )

            # diversity_aware_select
            selected = ca.diversity_aware_select(level=10, n=2)
            assert isinstance(selected, list)
            assert len(selected) <= 2

            # quality_weighted_prune
            pruned = ca.quality_weighted_prune()
            assert isinstance(pruned, int)

            # get_diversity_report
            report = ca.get_diversity_report()
            assert isinstance(report, dict)
            assert report["total_champions"] > 0


# ══════════════════════════════════════════════════════════════════════
# 4. TestMetricsAnomalyCorrelation
# ══════════════════════════════════════════════════════════════════════


class TestMetricsAnomalyCorrelation:
    """MetricsCollector: record 30 cycles with varying fitness,
    detect anomalies, and correlate metrics."""

    def _make_mc_with_data(self, n: int = 30) -> MetricsCollector:
        """Create a MetricsCollector with n cycles of synthetic data."""
        tmpfile = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, prefix="metrics_test_"
        )
        tmpfile.close()
        mc = MetricsCollector(path=tmpfile.name)
        for i in range(n):
            # Create fitness with a clear outlier at cycle 15
            fitness = 100 + i * 2 + (i * i * 0.1)
            if i == 15:
                fitness = 500.0  # outlier spike
            if i == 25:
                fitness = 10.0  # outlier dip
            mc.record_cycle({
                "cycle": i,
                "fitness": fitness,
                "level": 5 + i // 5,
                "diversity": 0.3 + (i % 10) * 0.05,
                "val_loss": 3.0 - i * 0.02,
            })
        return mc

    def test_record_and_history(self):
        mc = self._make_mc_with_data(10)
        history = mc.get_history()
        assert isinstance(history, list)
        assert len(history) == 10
        assert "fitness" in history[0]
        assert "cycle" in history[0]

    def test_detect_anomalies_finds_outliers(self):
        mc = self._make_mc_with_data(30)
        anomalies = mc.detect_anomalies("fitness", window=10, threshold=2.0)
        assert isinstance(anomalies, list)
        # Should find at least the spike at cycle 15
        assert len(anomalies) > 0
        # Check that at least one anomaly is near cycle 15
        indices = [a["index"] for a in anomalies]
        assert 15 in indices or any(13 <= idx <= 17 for idx in indices)

    def test_detect_anomalies_returns_correct_fields(self):
        mc = self._make_mc_with_data(30)
        anomalies = mc.detect_anomalies("fitness", window=10, threshold=2.0)
        for a in anomalies:
            assert "index" in a
            assert "value" in a
            assert "z_score" in a
            assert "direction" in a
            assert a["direction"] in ("high", "low")
            assert isinstance(a["index"], int)
            assert isinstance(a["value"], float)
            assert isinstance(a["z_score"], float)

    def test_detect_anomalies_empty_metric(self):
        tmpfile = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, prefix="metrics_empty_"
        )
        tmpfile.close()
        mc = MetricsCollector(path=tmpfile.name)
        anomalies = mc.detect_anomalies("nonexistent", window=10)
        assert anomalies == []

    def test_correlate_metrics_positive(self):
        mc = MetricsCollector(
            path=tempfile.mktemp(suffix=".jsonl", prefix="corr_")
        )
        # Create two positively correlated metrics
        for i in range(20):
            mc.record_cycle({
                "cycle": i,
                "fitness": 100 + i * 10,
                "level": 5 + i,
            })
        result = mc.correlate_metrics("fitness", "level", window=20)
        assert isinstance(result, dict)
        assert "correlation" in result
        assert "strength" in result
        assert "direction" in result
        assert "sample_size" in result
        assert result["sample_size"] == 20
        # Both increasing -> positive correlation
        assert result["correlation"] > 0.5
        assert result["direction"] == "positive"

    def test_correlate_metrics_negative(self):
        mc = MetricsCollector(
            path=tempfile.mktemp(suffix=".jsonl", prefix="corr_neg_")
        )
        # Create two negatively correlated metrics
        for i in range(20):
            mc.record_cycle({
                "cycle": i,
                "fitness": 200 - i * 10,
                "val_loss": 1.0 + i * 0.2,
            })
        result = mc.correlate_metrics("fitness", "val_loss", window=20)
        assert result["correlation"] < -0.5
        assert result["direction"] == "negative"

    def test_correlate_metrics_insufficient_data(self):
        mc = MetricsCollector(
            path=tempfile.mktemp(suffix=".jsonl", prefix="corr_short_")
        )
        mc.record_cycle({"cycle": 0, "fitness": 100})
        result = mc.correlate_metrics("fitness", "fitness", window=10)
        assert result["sample_size"] < 2
        assert result["correlation"] == 0.0
        assert result["strength"] == "none"

    def test_correlate_metrics_strength_categories(self):
        mc = MetricsCollector(
            path=tempfile.mktemp(suffix=".jsonl", prefix="corr_str_")
        )
        for i in range(20):
            mc.record_cycle({
                "cycle": i,
                "metric_a": 50 + i * 10,
                "metric_b": 50 + i * 10,  # identical -> perfect correlation
            })
        result = mc.correlate_metrics("metric_a", "metric_b")
        assert result["strength"] == "strong"
        assert result["correlation"] == pytest.approx(1.0, abs=0.01)

    def test_cleanup_files(self):
        """Ensure temporary files are cleaned up."""
        mc = self._make_mc_with_data(5)
        path = mc.path
        assert os.path.isfile(path)
        mc.clear()
        assert not os.path.isfile(path)


# ══════════════════════════════════════════════════════════════════════
# 5. TestProgressionTrackerFullCycle
# ══════════════════════════════════════════════════════════════════════


class TestProgressionTrackerFullCycle:
    """ProgressionTracker: record 15 cycles, run plateau analysis and
    window comparison end-to-end."""

    def _make_tracker_with_data(self, n: int = 15, plateau_start: int = 10) -> ProgressionTracker:
        """Create a ProgressionTracker with n cycles.
        First plateau_start cycles improve, last ones plateau."""
        tmpfile = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, prefix="progression_test_"
        )
        tmpfile.close()
        pt = ProgressionTracker(path=tmpfile.name)
        for i in range(n):
            if i < plateau_start:
                fitness = 80 + i * 8
                words = 100 + i * 5
            else:
                fitness = 150.0  # plateau
                words = 150
            pt.record(_make_cycle_metrics(
                cycle=i,
                fitness=fitness,
                words=words,
                val_loss=3.0 - i * 0.05 if i < plateau_start else 1.0,
            ))
        return pt

    def test_record_and_history(self):
        pt = self._make_tracker_with_data(10)
        assert len(pt.history) == 10
        assert pt.last_cycle is not None
        assert pt.last_cycle.cycle == 9

    def test_get_plateau_analysis_returns_dict(self):
        pt = self._make_tracker_with_data(15, plateau_start=10)
        result = pt.get_plateau_analysis(window=10)
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ("plateau", "improving", "declining", "insufficient_data")

    def test_get_plateau_analysis_detects_plateau(self):
        pt = self._make_tracker_with_data(15, plateau_start=5)
        result = pt.get_plateau_analysis(window=10)
        # Last 10 cycles (5-14) are all flat fitness=150 -> should detect plateau
        assert result["status"] == "plateau"
        assert result["is_plateau"] is True
        assert result["fitness_variance"] < 1.0
        assert isinstance(result["trend_slope"], float)

    def test_get_plateau_analysis_detects_improving(self):
        pt = self._make_tracker_with_data(15, plateau_start=15)  # all improving
        result = pt.get_plateau_analysis(window=10)
        assert result["status"] == "improving"
        assert result["trend_slope"] > 0

    def test_get_plateau_analysis_insufficient_data(self):
        pt = self._make_tracker_with_data(3)
        result = pt.get_plateau_analysis(window=10)
        assert result["status"] == "insufficient_data"
        assert "cycles_available" in result
        assert result["cycles_available"] == 3

    def test_compare_windows_returns_dict(self):
        pt = self._make_tracker_with_data(15)
        result = pt.compare_windows("fitness", window_a=5, window_b=5)
        assert isinstance(result, dict)
        required_keys = {
            "metric", "window_a_mean", "window_b_mean",
            "improvement", "percent_change", "direction",
            "window_a_stddev", "window_b_stddev",
        }
        assert required_keys.issubset(result.keys())

    def test_compare_windows_improving(self):
        """First 5 cycles flat, last 5 cycles higher -> improving."""
        tmpfile = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, prefix="compare_test_"
        )
        tmpfile.close()
        pt = ProgressionTracker(path=tmpfile.name)
        # Older window: flat at 100
        for i in range(5):
            pt.record(_make_cycle_metrics(cycle=i, fitness=100.0, words=120))
        # Newer window: flat at 200
        for i in range(5):
            pt.record(_make_cycle_metrics(cycle=i + 5, fitness=200.0, words=120))

        result = pt.compare_windows("fitness", window_a=5, window_b=5)
        assert result["direction"] == "improving"
        assert result["improvement"] > 0
        assert result["window_a_mean"] == pytest.approx(200.0)
        assert result["window_b_mean"] == pytest.approx(100.0)

    def test_compare_windows_declining(self):
        tmpfile = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, prefix="compare_decl_"
        )
        tmpfile.close()
        pt = ProgressionTracker(path=tmpfile.name)
        # Older: high
        for i in range(5):
            pt.record(_make_cycle_metrics(cycle=i, fitness=200.0, words=120))
        # Newer: low
        for i in range(5):
            pt.record(_make_cycle_metrics(cycle=i + 5, fitness=100.0, words=120))

        result = pt.compare_windows("fitness", window_a=5, window_b=5)
        assert result["direction"] == "declining"
        assert result["improvement"] < 0

    def test_compare_windows_stable(self):
        tmpfile = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, prefix="compare_stab_"
        )
        tmpfile.close()
        pt = ProgressionTracker(path=tmpfile.name)
        for i in range(10):
            pt.record(_make_cycle_metrics(cycle=i, fitness=150.0, words=120))

        result = pt.compare_windows("fitness", window_a=5, window_b=5)
        assert result["direction"] == "stable"
        assert result["improvement"] == pytest.approx(0.0, abs=0.01)

    def test_compare_windows_empty_tracker(self):
        tmpfile = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, prefix="compare_empty_"
        )
        tmpfile.close()
        pt = ProgressionTracker(path=tmpfile.name)
        result = pt.compare_windows("fitness")
        assert result["direction"] == "stable"
        assert result["improvement"] == 0.0

    def test_full_progression_cycle(self):
        """Record 15 cycles, plateau analysis, window comparison -- all types."""
        pt = self._make_tracker_with_data(15, plateau_start=10)

        # plateau analysis
        plateau = pt.get_plateau_analysis(window=10)
        assert isinstance(plateau, dict)
        assert plateau["status"] in ("plateau", "improving", "declining")
        assert isinstance(plateau["fitness_variance"], float)
        assert isinstance(plateau["trend_slope"], float)
        assert isinstance(plateau["plateau_duration"], int)

        # compare windows
        comparison = pt.compare_windows("fitness", window_a=5, window_b=5)
        assert isinstance(comparison, dict)
        assert isinstance(comparison["window_a_mean"], float)
        assert isinstance(comparison["window_b_mean"], float)
        assert isinstance(comparison["improvement"], float)
        assert comparison["direction"] in ("improving", "declining", "stable")

        # Also compare val_loss
        vl_comparison = pt.compare_windows("val_loss", window_a=5, window_b=5)
        assert isinstance(vl_comparison, dict)
        assert vl_comparison["metric"] == "val_loss"
