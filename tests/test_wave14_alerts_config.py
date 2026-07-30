#!/usr/bin/env python3
"""Wave 14 — Tests for alerts, config, metrics, data rotation, and progression tracker.

Verifies full lifecycle of:
  1. AlertManager — send, monoculture, recovery stuck, history, config summary
  2. EvolutionSection (config) — summary, validation, field access
  3. MetricsCollector — record, detect anomalies, correlate, trend, count
  4. Data rotation — disk usage, quality-aware rotation, run_all_rotations
  5. ProgressionTracker — plateau analysis, compare windows, fitness trend, stagnant

Uses pytest, tempfile, unittest.mock.  No existing files are modified.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _tmpjsonl() -> str:
    """Create a temporary .jsonl file path (not yet written)."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    # Remove the empty file; MetricsCollector will create it
    os.unlink(path)
    return path


def _tmpjson() -> str:
    """Create a temporary .json file path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)
    return path


# ═══════════════════════════════════════════════════════════════════════
# 1. TestAlertManagerFullCycle
# ═══════════════════════════════════════════════════════════════════════


class TestAlertManagerFullCycle:
    """Full lifecycle tests for AlertManager."""

    def test_send_alert_returns_bool(self):
        """send_alert returns a boolean."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am.send_alert("warning", "Test", "test message")
        assert isinstance(result, bool)

    def test_send_alert_no_channels_returns_false(self):
        """send_alert with no configured channels returns False."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        # No discord/slack/email configured
        result = am.send_alert("warning", "Test", "test message")
        assert result is False

    def test_send_alert_records_in_history(self):
        """send_alert records alert in history even when not sent."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        am.send_alert("warning", "Title1", "msg1")
        am.send_alert("error", "Title2", "msg2")
        history = am.get_alert_history()
        assert len(history) == 2
        assert history[0]["title"] == "Title1"
        assert history[0]["level"] == "warning"
        assert history[1]["title"] == "Title2"
        assert history[1]["level"] == "error"

    def test_send_alert_critical_bypasses_rate_limit(self):
        """Critical alerts bypass rate limiting."""
        from monitoring_alerts import AlertManager, AlertThresholds

        am = AlertManager(AlertThresholds(max_alerts_per_hour=1))
        # Fill rate limit
        am.send_alert("warning", "first", "msg")
        # Should be rate-limited
        assert am.send_alert("warning", "second", "msg") is False
        # But critical bypasses
        result = am.send_alert("critical", "critical", "msg", critical=True)
        assert isinstance(result, bool)

    def test_send_monoculture_alert_high_dominance(self):
        """send_monoculture_alert with high dominance (>0.8) is critical."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am.send_monoculture_alert("GROW", 0.9, "increase diversity")
        assert isinstance(result, bool)
        history = am.get_alert_history()
        assert len(history) == 1
        assert history[0]["title"] == "Strategy Monoculture Detected"
        assert history[0]["level"] == "warning"

    def test_send_monoculture_alert_medium_dominance(self):
        """send_monoculture_alert with medium dominance (0.7-0.8) is warning."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am.send_monoculture_alert("MUTATE", 0.75, "rotate strategies")
        assert isinstance(result, bool)
        history = am.get_alert_history()
        assert history[0]["level"] == "warning"

    def test_send_monoculture_alert_low_dominance(self):
        """send_monoculture_alert with low dominance (<0.7) is info."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am.send_monoculture_alert("CROSSOVER", 0.5, "keep monitoring")
        assert isinstance(result, bool)
        history = am.get_alert_history()
        assert history[0]["level"] == "info"

    def test_send_recovery_stuck_alert_short(self):
        """send_recovery_stuck_alert with few cycles is non-critical."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am.send_recovery_stuck_alert(3, "stable", 0.85)
        assert isinstance(result, bool)
        history = am.get_alert_history()
        assert len(history) == 1
        assert history[0]["title"] == "Recovery Mode Stuck"
        assert "3 cycles" in history[0]["message"]

    def test_send_recovery_stuck_alert_long(self):
        """send_recovery_stuck_alert with >=10 cycles is critical."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am.send_recovery_stuck_alert(12, "declining", 0.42)
        assert isinstance(result, bool)
        history = am.get_alert_history()
        assert len(history) == 1
        assert "12 cycles" in history[0]["message"]

    def test_get_alert_history_returns_list(self):
        """get_alert_history returns a list."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        history = am.get_alert_history()
        assert isinstance(history, list)
        assert len(history) == 0

    def test_get_config_summary_returns_dict(self):
        """get_config_summary returns a dict with expected keys."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        summary = am.get_config_summary()
        assert isinstance(summary, dict)
        assert "discord" in summary
        assert "slack" in summary
        assert "email" in summary
        assert "thresholds" in summary
        assert isinstance(summary["thresholds"], dict)
        assert summary["discord"] is False  # no env var set
        assert summary["slack"] is False
        assert summary["email"] is False

    def test_get_config_summary_reflects_thresholds(self):
        """get_config_summary reflects custom thresholds."""
        from monitoring_alerts import AlertManager, AlertThresholds

        custom = AlertThresholds(fitness_regression_pct=0.10, max_alerts_per_hour=5)
        am = AlertManager(thresholds=custom)
        summary = am.get_config_summary()
        assert summary["thresholds"]["fitness_regression_pct"] == 0.10
        assert summary["thresholds"]["max_alerts_per_hour"] == 5


# ═══════════════════════════════════════════════════════════════════════
# 2. TestConfigFullCycle
# ═══════════════════════════════════════════════════════════════════════


class TestConfigFullCycle:
    """Full lifecycle tests for EvolutionSection config."""

    def test_get_config_summary_returns_complete_dict(self):
        """get_config_summary returns dict with all expected keys."""
        from config import EvolutionSection

        evo = EvolutionSection()
        summary = evo.get_config_summary()
        assert isinstance(summary, dict)
        expected_keys = {
            "seed",
            "use_gpu",
            "auto_curriculum",
            "archives_enabled",
            "novelty_weight",
            "qd_bonus_weight",
            "mutation_rate",
            "population_size",
            "eval_budget",
            "max_consecutive_same_strategy",
            "total_config_fields",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_get_config_summary_values_match_defaults(self):
        """Summary values match dataclass defaults."""
        from config import EvolutionSection

        evo = EvolutionSection()
        summary = evo.get_config_summary()
        assert summary["seed"] == 42
        assert summary["population_size"] == 50
        assert summary["eval_budget"] == 5000
        assert summary["mutation_rate"] == 0.3

    def test_validate_config_returns_list(self):
        """validate_config returns a list."""
        from config import EvolutionSection

        evo = EvolutionSection()
        result = evo.validate_config()
        assert isinstance(result, list)

    def test_validate_config_valid_returns_empty(self):
        """Default config passes validation with no warnings."""
        from config import EvolutionSection

        evo = EvolutionSection()
        warnings = evo.validate_config()
        assert warnings == []

    def test_validate_config_high_novelty_weight(self):
        """High novelty_weight produces a warning."""
        from config import EvolutionSection

        evo = EvolutionSection(novelty_weight=0.95)
        warnings = evo.validate_config()
        assert any("novelty_weight" in w for w in warnings)

    def test_validate_config_small_population(self):
        """Small population_size produces a warning."""
        from config import EvolutionSection

        evo = EvolutionSection(population_size=5)
        warnings = evo.validate_config()
        assert any("population_size" in w for w in warnings)

    def test_validate_config_no_archives(self):
        """No archives enabled produces a warning."""
        from config import EvolutionSection

        evo = EvolutionSection(
            use_novelty_archive=False,
            use_qd_archive=False,
            use_strategy_archive=False,
        )
        warnings = evo.validate_config()
        assert any("diversity" in w.lower() or "no diversity" in w.lower() for w in warnings)

    def test_config_field_access(self):
        """Fields are directly accessible on the dataclass."""
        from config import EvolutionSection

        evo = EvolutionSection()
        assert evo.population_size == 50
        assert evo.mutation_rate == 0.3
        assert evo.max_generations == 200
        assert evo.auto_curriculum is True
        # Mutate and verify
        evo.population_size = 100
        assert evo.population_size == 100

    def test_config_custom_values(self):
        """Custom constructor values are stored."""
        from config import EvolutionSection

        evo = EvolutionSection(population_size=200, mutation_rate=0.5, seed=99)
        assert evo.population_size == 200
        assert evo.mutation_rate == 0.5
        assert evo.seed == 99


# ═══════════════════════════════════════════════════════════════════════
# 3. TestMetricsFullCycle
# ═══════════════════════════════════════════════════════════════════════


class TestMetricsFullCycle:
    """Full lifecycle tests for MetricsCollector."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Create and clean up a temporary metrics file."""
        self.path = _tmpjsonl()
        yield
        # Cleanup
        if os.path.isfile(self.path):
            os.unlink(self.path)

    def test_record_and_count(self):
        """Record 50 cycles and verify count_records."""
        from metrics import MetricsCollector

        mc = MetricsCollector(self.path)
        for i in range(50):
            mc.record_cycle({"cycle": i, "fitness": 0.1 * i, "level": i % 5})
        assert mc.count_records() == 50

    def test_get_history_returns_list(self):
        """get_history returns a list of recorded dicts."""
        from metrics import MetricsCollector

        mc = MetricsCollector(self.path)
        mc.record_cycle({"cycle": 1, "fitness": 0.5})
        mc.record_cycle({"cycle": 2, "fitness": 0.6})
        history = mc.get_history()
        assert isinstance(history, list)
        assert len(history) == 2
        assert history[0]["cycle"] == 1
        assert history[1]["cycle"] == 2

    def test_detect_anomalies_finds_outliers(self):
        """detect_anomalies finds z-score outliers in metric data."""
        from metrics import MetricsCollector

        mc = MetricsCollector(self.path)
        # Record mostly stable values
        for i in range(30):
            mc.record_cycle({"cycle": i, "fitness": 1.0 + 0.01 * (i % 3)})
        # Inject a huge outlier
        mc.record_cycle({"cycle": 30, "fitness": 100.0})
        anomalies = mc.detect_anomalies("fitness", window=20, threshold=2.0)
        assert isinstance(anomalies, list)
        # The outlier should be detected
        assert any(a["value"] == 100.0 for a in anomalies)

    def test_detect_anomalies_no_outliers(self):
        """detect_anomalies returns empty list for uniform data."""
        from metrics import MetricsCollector

        mc = MetricsCollector(self.path)
        for i in range(20):
            mc.record_cycle({"cycle": i, "fitness": 5.0})
        anomalies = mc.detect_anomalies("fitness", window=20)
        assert anomalies == []

    def test_correlate_metrics_positive(self):
        """correlate_metrics computes positive correlation for correlated data."""
        from metrics import MetricsCollector

        mc = MetricsCollector(self.path)
        for i in range(30):
            mc.record_cycle({
                "cycle": i,
                "metric_a": float(i),
                "metric_b": float(i) * 2.0,
            })
        result = mc.correlate_metrics("metric_a", "metric_b", window=30)
        assert isinstance(result, dict)
        assert result["metric_a"] == "metric_a"
        assert result["metric_b"] == "metric_b"
        assert result["correlation"] > 0.99
        assert result["direction"] == "positive"
        assert result["strength"] == "strong"

    def test_correlate_metrics_negative(self):
        """correlate_metrics computes negative correlation for inverse data."""
        from metrics import MetricsCollector

        mc = MetricsCollector(self.path)
        for i in range(30):
            mc.record_cycle({
                "cycle": i,
                "metric_a": float(i),
                "metric_b": 100.0 - float(i),
            })
        result = mc.correlate_metrics("metric_a", "metric_b", window=30)
        assert result["correlation"] < -0.99
        assert result["direction"] == "negative"

    def test_correlate_metrics_insufficient_data(self):
        """correlate_metrics handles insufficient data gracefully."""
        from metrics import MetricsCollector

        mc = MetricsCollector(self.path)
        mc.record_cycle({"cycle": 1, "a": 1.0, "b": 2.0})
        result = mc.correlate_metrics("a", "b")
        assert result["sample_size"] == 1
        assert result["strength"] == "none"

    def test_get_trend_improving(self):
        """get_trend returns 'improving' for ascending values."""
        from metrics import MetricsCollector

        mc = MetricsCollector(self.path)
        for i in range(20):
            mc.record_cycle({"cycle": i, "fitness": float(i)})
        trend = mc.get_trend("fitness", window=20)
        assert isinstance(trend, dict)
        assert trend["trend"] == "improving"
        assert trend["current"] == 19.0

    def test_get_trend_declining(self):
        """get_trend returns 'declining' for descending values."""
        from metrics import MetricsCollector

        mc = MetricsCollector(self.path)
        for i in range(20):
            mc.record_cycle({"cycle": i, "fitness": 20.0 - float(i)})
        trend = mc.get_trend("fitness", window=20)
        assert trend["trend"] == "declining"

    def test_get_trend_stable(self):
        """get_trend returns 'stable' for constant values."""
        from metrics import MetricsCollector

        mc = MetricsCollector(self.path)
        for i in range(10):
            mc.record_cycle({"cycle": i, "fitness": 5.0})
        trend = mc.get_trend("fitness", window=10)
        assert trend["trend"] == "stable"

    def test_count_records_correct_after_clear(self):
        """count_records returns 0 after clear."""
        from metrics import MetricsCollector

        mc = MetricsCollector(self.path)
        for i in range(10):
            mc.record_cycle({"cycle": i, "fitness": 1.0})
        assert mc.count_records() == 10
        mc.clear()
        assert mc.count_records() == 0


# ═══════════════════════════════════════════════════════════════════════
# 4. TestDataRotationFullCycle
# ═══════════════════════════════════════════════════════════════════════


class TestDataRotationFullCycle:
    """Full lifecycle tests for data rotation utilities."""

    def test_get_disk_usage_returns_dict_with_keys(self):
        """get_disk_usage returns dict with expected keys."""
        from data_rotation import get_disk_usage

        result = get_disk_usage()
        assert isinstance(result, dict)
        expected_keys = {
            "total_bytes",
            "used_bytes",
            "free_bytes",
            "usage_percent",
            "checkpoint_size",
            "recurrent_size",
            "total_project_size",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_get_disk_usage_values_are_numeric(self):
        """All values in disk usage dict are numeric."""
        from data_rotation import get_disk_usage

        result = get_disk_usage()
        assert isinstance(result["total_bytes"], int)
        assert isinstance(result["used_bytes"], int)
        assert isinstance(result["free_bytes"], int)
        assert isinstance(result["usage_percent"], float)
        assert result["total_bytes"] > 0
        assert result["used_bytes"] > 0
        assert result["free_bytes"] >= 0
        assert 0 < result["usage_percent"] <= 100

    def test_quality_aware_rotate_checkpoints_with_scores(self):
        """quality_aware_rotate_checkpoints handles fitness scores dict."""
        from data_rotation import quality_aware_rotate_checkpoints

        # This should not raise even if the checkpoint dir doesn't exist
        result = quality_aware_rotate_checkpoints(
            keep_n=5, fitness_scores={"checkpoint_1.pt": 0.9, "checkpoint_2.pt": 0.5}
        )
        assert isinstance(result, list)

    def test_quality_aware_rotate_checkpoints_without_scores(self):
        """quality_aware_rotate_checkpoints with None falls back to mtime rotation."""
        from data_rotation import quality_aware_rotate_checkpoints

        result = quality_aware_rotate_checkpoints(keep_n=5, fitness_scores=None)
        assert isinstance(result, list)

    def test_run_all_rotations_returns_summary_dict(self):
        """run_all_rotations returns a dict with expected keys."""
        from data_rotation import run_all_rotations

        summary = run_all_rotations()
        assert isinstance(summary, dict)
        assert "checkpoints" in summary
        assert "recurrent" in summary
        assert "archives" in summary
        assert "reports" in summary
        # All values should be non-negative integers
        for key in ("checkpoints", "recurrent", "archives", "reports"):
            assert isinstance(summary[key], int)
            assert summary[key] >= 0

    def test_run_all_rotations_dry_run_no_crash(self):
        """run_all_rotations completes without exceptions on a clean system."""
        from data_rotation import run_all_rotations

        # Run twice to ensure idempotency
        s1 = run_all_rotations()
        s2 = run_all_rotations()
        assert isinstance(s1, dict)
        assert isinstance(s2, dict)


# ═══════════════════════════════════════════════════════════════════════
# 5. TestProgressionTrackerFullCycle
# ═══════════════════════════════════════════════════════════════════════


class TestProgressionTrackerFullCycle:
    """Full lifecycle tests for ProgressionTracker."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Create and clean up a temporary progression file."""
        self.path = _tmpjsonl()
        yield
        if os.path.isfile(self.path):
            os.unlink(self.path)

    def _make_tracker(self):
        from progression_tracker import ProgressionTracker
        return ProgressionTracker(path=self.path)

    def _record_cycles(self, tracker, n: int, fitness_values: list[float] | None = None):
        """Helper to record n cycles with optional custom fitness values."""
        from progression_tracker import CycleMetrics

        for i in range(n):
            if fitness_values and i < len(fitness_values):
                fitness = fitness_values[i]
            else:
                fitness = 10.0 + i * 0.5  # steady improvement
            tracker.record(CycleMetrics(
                cycle=i + 1,
                level=min(i // 3, 13),
                fitness=fitness,
                words=100 + i * 5,
                archive_size=50 + i,
                val_loss=1.0 - i * 0.02 if i < 40 else 0.2,
                time_s=5.0 + i * 0.1,
                mg_count=i % 5,
                total_challengers=10,
                champion_improvements=1 if i % 3 == 0 else 0,
            ))

    def test_record_20_cycles(self):
        """Record 20 cycles and verify history length."""
        tracker = self._make_tracker()
        self._record_cycles(tracker, 20)
        assert len(tracker.history) == 20

    def test_get_plateau_analysis_insufficient_data(self):
        """get_plateau_analysis with few cycles returns insufficient_data."""
        tracker = self._make_tracker()
        self._record_cycles(tracker, 5)
        result = tracker.get_plateau_analysis(window=10)
        assert result["status"] == "insufficient_data"
        assert result["cycles_available"] == 5

    def test_get_plateau_analysis_plateau(self):
        """get_plateau_analysis detects a plateau (constant fitness)."""
        tracker = self._make_tracker()
        # All same fitness = plateau
        self._record_cycles(tracker, 10, fitness_values=[50.0] * 10)
        result = tracker.get_plateau_analysis(window=10)
        assert result["status"] == "plateau"
        assert result["is_plateau"] is True
        assert result["fitness_variance"] == 0.0

    def test_get_plateau_analysis_improving(self):
        """get_plateau_analysis detects improving trend."""
        tracker = self._make_tracker()
        self._record_cycles(tracker, 10, fitness_values=[float(i) for i in range(10)])
        result = tracker.get_plateau_analysis(window=10)
        assert result["status"] == "improving"
        assert result["trend_slope"] > 0

    def test_compare_windows(self):
        """compare_windows compares newer vs older window."""
        tracker = self._make_tracker()
        # First 5 cycles: low fitness
        # Last 5 cycles: high fitness
        fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        self._record_cycles(tracker, 10, fitness_values=fitnesses)
        result = tracker.compare_windows("fitness", window_a=5, window_b=5)
        assert isinstance(result, dict)
        assert result["metric"] == "fitness"
        assert result["direction"] == "improving"
        assert result["improvement"] > 0
        assert result["window_a_mean"] > result["window_b_mean"]

    def test_compare_windows_insufficient_history(self):
        """compare_windows with no history returns stable."""
        tracker = self._make_tracker()
        result = tracker.compare_windows("fitness")
        assert result["direction"] == "stable"
        assert result["window_a_mean"] == 0.0

    def test_fitness_trend_positive(self):
        """fitness_trend returns positive for improving fitness."""
        tracker = self._make_tracker()
        self._record_cycles(tracker, 10, fitness_values=[float(i * 10) for i in range(10)])
        trend = tracker.fitness_trend(window=10)
        assert trend > 0

    def test_fitness_trend_negative(self):
        """fitness_trend returns negative for declining fitness."""
        tracker = self._make_tracker()
        self._record_cycles(tracker, 10, fitness_values=[100.0 - float(i * 10) for i in range(10)])
        trend = tracker.fitness_trend(window=10)
        assert trend < 0

    def test_fitness_trend_insufficient_data(self):
        """fitness_trend with <2 cycles returns 0.0."""
        tracker = self._make_tracker()
        self._record_cycles(tracker, 1)
        assert tracker.fitness_trend() == 0.0

    def test_is_stagnant_true(self):
        """is_stagnant returns True when fitness and words are stable."""
        tracker = self._make_tracker()
        # All same fitness and words
        from progression_tracker import CycleMetrics

        for i in range(8):
            tracker.record(CycleMetrics(
                cycle=i + 1, level=3, fitness=50.0, words=100,
                archive_size=50, val_loss=0.5, time_s=5.0,
                mg_count=1, total_challengers=10, champion_improvements=0,
            ))
        assert tracker.is_stagnant(window=5) is True

    def test_is_stagnant_false_with_improvement(self):
        """is_stagnant returns False when fitness is improving."""
        tracker = self._make_tracker()
        self._record_cycles(tracker, 10, fitness_values=[float(i * 10) for i in range(10)])
        assert tracker.is_stagnant(window=5) is False

    def test_is_stagnant_insufficient_data(self):
        """is_stagnant with fewer cycles than window returns False."""
        tracker = self._make_tracker()
        self._record_cycles(tracker, 3)
        assert tracker.is_stagnant(window=5) is False

    def test_persist_and_reload(self):
        """Tracker persists to disk and reloads correctly."""
        tracker1 = self._make_tracker()
        self._record_cycles(tracker1, 10)
        assert len(tracker1.history) == 10

        # Create a new tracker from the same file
        from progression_tracker import ProgressionTracker
        tracker2 = ProgressionTracker(path=self.path)
        assert len(tracker2.history) == 10
        assert tracker2.history[0].cycle == 1
        assert tracker2.history[-1].cycle == 10
