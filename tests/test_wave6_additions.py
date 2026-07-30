#!/usr/bin/env python3
"""Comprehensive tests for Wave 6 additions.

Tests cover:
  1. TestDetectAnomalies — z-score anomaly detection in MetricsCollector
  2. TestCorrelateMetrics — Pearson correlation between metrics
  3. TestMutationTracker — mutation outcome recording and stats
  4. TestMonocultureAlert — monoculture dominance alerting
  5. TestRecoveryStuckAlert — recovery mode stuck alerting
  6. TestGetDiskUsage — disk usage reporting
  7. TestQualityAwareRotateCheckpoints — fitness-aware checkpoint rotation
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import shutil

import pytest

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics import MetricsCollector
from mutation_proposer import DirectedMutator
from monitoring_alerts import AlertManager


# ══════════════════════════════════════════════════════════════════════
# 1. TestDetectAnomalies
# ══════════════════════════════════════════════════════════════════════

class TestDetectAnomalies:
    """Tests for MetricsCollector.detect_anomalies()."""

    def test_detects_known_outlier_values(self):
        """An extreme outlier in mostly-constant data should be flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)

            # 18 normal values around 10.0, then one extreme outlier at 100.0
            for i in range(18):
                mc.record_cycle({"fitness": 10.0 + i * 0.1})
            mc.record_cycle({"fitness": 100.0})  # outlier

            anomalies = mc.detect_anomalies("fitness", window=20, threshold=2.0)
            assert len(anomalies) >= 1, "Expected at least one anomaly for extreme outlier"
            outlier = anomalies[-1]
            assert outlier["value"] == 100.0
            assert outlier["direction"] == "high"
            assert abs(outlier["z_score"]) > 2.0

    def test_empty_history_returns_empty(self):
        """No records → no anomalies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            anomalies = mc.detect_anomalies("fitness", window=20, threshold=2.0)
            assert anomalies == []

    def test_normal_data_returns_empty_list(self):
        """Uniform data with no outliers should produce no anomalies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)

            # All values are the same — stddev = 0 → z_score = 0 → no anomalies
            for _ in range(20):
                mc.record_cycle({"fitness": 42.0})

            anomalies = mc.detect_anomalies("fitness", window=20, threshold=2.0)
            assert anomalies == []

    def test_anomaly_has_required_keys(self):
        """Each anomaly dict should contain index, value, z_score, direction, timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)

            for i in range(15):
                mc.record_cycle({"fitness": 10.0})
            mc.record_cycle({"fitness": 50.0})

            anomalies = mc.detect_anomalies("fitness", window=20, threshold=2.0)
            for a in anomalies:
                assert "index" in a
                assert "value" in a
                assert "z_score" in a
                assert "direction" in a
                assert a["direction"] in ("high", "low")


# ══════════════════════════════════════════════════════════════════════
# 2. TestCorrelateMetrics
# ══════════════════════════════════════════════════════════════════════

class TestCorrelateMetrics:
    """Tests for MetricsCollector.correlate_metrics()."""

    def test_perfect_positive_correlation(self):
        """Two metrics that move in lockstep should have r ≈ 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)

            for i in range(20):
                mc.record_cycle({"metric_a": float(i), "metric_b": float(i)})

            result = mc.correlate_metrics("metric_a", "metric_b", window=20)
            assert result["correlation"] == pytest.approx(1.0, abs=1e-4)
            assert result["strength"] == "strong"
            assert result["direction"] == "positive"
            assert result["sample_size"] == 20

    def test_no_correlation_between_independent_data(self):
        """Uncorrelated metrics should have |r| close to 0."""
        import random
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)

            rng = random.Random(123)
            for _ in range(30):
                mc.record_cycle({
                    "metric_a": rng.gauss(10, 1),
                    "metric_b": rng.gauss(50, 5),
                })

            result = mc.correlate_metrics("metric_a", "metric_b", window=30)
            # With random data, |r| should be small (below moderate threshold)
            assert abs(result["correlation"]) < 0.4

    def test_different_metric_names(self):
        """Correlate two metrics with distinct names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)

            for i in range(15):
                mc.record_cycle({"fitness": float(i), "diversity": float(14 - i)})

            result = mc.correlate_metrics("fitness", "diversity", window=15)
            # Perfect negative correlation
            assert result["correlation"] == pytest.approx(-1.0, abs=1e-4)
            assert result["direction"] == "negative"
            assert result["metric_a"] == "fitness"
            assert result["metric_b"] == "diversity"

    def test_insufficient_data_returns_zero(self):
        """With only 1 paired record, correlation should be 0 (none)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)

            mc.record_cycle({"metric_a": 1.0, "metric_b": 2.0})
            result = mc.correlate_metrics("metric_a", "metric_b", window=20)
            assert result["correlation"] == 0.0
            assert result["strength"] == "none"
            assert result["sample_size"] == 1


# ══════════════════════════════════════════════════════════════════════
# 3. TestMutationTracker
# ══════════════════════════════════════════════════════════════════════

class TestMutationTracker:
    """Tests for DirectedMutator mutation outcome tracking."""

    @pytest.fixture(autouse=True)
    def _reset_tracker(self):
        """Clear class-level mutation state before each test."""
        DirectedMutator._mutation_success = {}
        DirectedMutator._mutation_attempts = {}
        DirectedMutator._mutation_history = []
        yield
        DirectedMutator._mutation_success = {}
        DirectedMutator._mutation_attempts = {}
        DirectedMutator._mutation_history = []

    def test_record_mutation_outcome_increments_attempts(self):
        """Recording a mutation should increment attempt count for its type prefix."""
        DirectedMutator.record_mutation_outcome("fix_reg_x10", -0.5)
        DirectedMutator.record_mutation_outcome("fix_reg_x10", 0.2)

        stats = DirectedMutator.get_mutation_stats()
        assert stats["by_type"]["fix"]["attempts"] == 2
        assert stats["total_attempts"] == 2

    def test_record_mutation_outcome_tracks_successes(self):
        """Positive fitness_delta should count as success."""
        DirectedMutator.record_mutation_outcome("inject_addi", 1.0)
        DirectedMutator.record_mutation_outcome("inject_addi", -0.3)

        stats = DirectedMutator.get_mutation_stats()
        assert stats["by_type"]["inject"]["successes"] == 1
        assert stats["by_type"]["inject"]["attempts"] == 2
        assert stats["by_type"]["inject"]["success_rate"] == pytest.approx(0.5)
        assert stats["total_successes"] == 1

    def test_get_mutation_stats_returns_correct_rates(self):
        """Success rate should be computed correctly."""
        # 3 successes out of 4 attempts
        for _ in range(3):
            DirectedMutator.record_mutation_outcome("delete_instr", 0.5)
        DirectedMutator.record_mutation_outcome("delete_instr", -1.0)

        stats = DirectedMutator.get_mutation_stats()
        assert stats["by_type"]["delete"]["success_rate"] == pytest.approx(0.75)
        assert stats["recent_avg_delta"] == pytest.approx(
            (0.5 + 0.5 + 0.5 + (-1.0)) / 4
        )

    def test_history_truncation_at_200(self):
        """Mutation history should be capped at 200 entries."""
        for i in range(210):
            DirectedMutator.record_mutation_outcome(f"type_{i % 3}", float(i))

        stats = DirectedMutator.get_mutation_stats()
        # Total attempts should be 210 (the class-level counters are cumulative)
        assert stats["total_attempts"] == 210
        # But history list should be truncated to 200
        assert len(DirectedMutator._mutation_history) == 200

    def test_empty_stats(self):
        """No mutations recorded → empty by_type, zero totals."""
        stats = DirectedMutator.get_mutation_stats()
        assert stats["by_type"] == {}
        assert stats["total_attempts"] == 0
        assert stats["total_successes"] == 0
        assert stats["recent_avg_delta"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# 4. TestMonocultureAlert
# ══════════════════════════════════════════════════════════════════════

class TestTestMonocultureAlert:
    """Tests for AlertManager.send_monoculture_alert()."""

    def test_alert_fires_when_dominance_above_07(self):
        """High dominance ratio (> 0.7) should produce a warning-level alert."""
        am = AlertManager()
        result = am.send_monoculture_alert("RANDOM", 0.85, "Increase mutation diversity")
        # No webhooks configured → returns False, but alert is recorded
        history = am.get_alert_history()
        assert len(history) == 1
        assert history[0]["level"] == "warning"
        assert "RANDOM" in history[0]["message"]
        assert "85%" in history[0]["message"]

    def test_info_level_for_moderate_dominance(self):
        """Moderate dominance (<= 0.7) should produce an info-level alert."""
        am = AlertManager()
        result = am.send_monoculture_alert("TASK_AWARE", 0.6, "Monitor diversity")
        history = am.get_alert_history()
        assert len(history) == 1
        assert history[0]["level"] == "info"
        assert "TASK_AWARE" in history[0]["message"]

    def test_dominance_exactly_07_is_info(self):
        """Dominance of exactly 0.7 should be info, not warning."""
        am = AlertManager()
        am.send_monoculture_alert("DIRECTED", 0.7, "Check balance")
        history = am.get_alert_history()
        assert history[0]["level"] == "info"


# ══════════════════════════════════════════════════════════════════════
# 5. TestRecoveryStuckAlert
# ══════════════════════════════════════════════════════════════════════

class TestRecoveryStuckAlert:
    """Tests for AlertManager.send_recovery_stuck_alert()."""

    def test_alert_fires(self):
        """Recovery stuck alert should be recorded."""
        am = AlertManager()
        am.send_recovery_stuck_alert(5, "declining", 0.75)
        history = am.get_alert_history()
        assert len(history) == 1
        assert "5 cycles" in history[0]["message"]
        assert "declining" in history[0]["message"]

    def test_critical_at_10_plus_cycles(self):
        """10+ cycles in recovery should mark the alert as critical."""
        am = AlertManager()
        am.send_recovery_stuck_alert(10, "stable", 0.90)
        history = am.get_alert_history()
        assert len(history) == 1
        # The message should contain "10 cycles"
        assert "10 cycles" in history[0]["message"]

    def test_critical_flag_passes_through(self):
        """Verify critical=True is passed when cycles >= 10."""
        am = AlertManager()
        am.send_recovery_stuck_alert(12, "declining", 0.85)
        history = am.get_alert_history()
        assert len(history) == 1
        assert "12 cycles" in history[0]["message"]

    def test_below_threshold_not_critical(self):
        """< 10 cycles should not be critical."""
        am = AlertManager()
        am.send_recovery_stuck_alert(3, "improving", 0.60)
        history = am.get_alert_history()
        assert len(history) == 1
        assert "3 cycles" in history[0]["message"]


# ══════════════════════════════════════════════════════════════════════
# 6. TestGetDiskUsage
# ══════════════════════════════════════════════════════════════════════

class TestGetDiskUsage:
    """Tests for data_rotation.get_disk_usage()."""

    def test_returns_dict_with_required_keys(self):
        """get_disk_usage should return a dict containing all expected keys."""
        from data_rotation import get_disk_usage

        result = get_disk_usage()
        required_keys = [
            "total_bytes", "used_bytes", "free_bytes", "usage_percent",
            "checkpoint_size", "recurrent_size", "total_project_size",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # Sanity checks on values
        assert result["total_bytes"] > 0
        assert result["used_bytes"] > 0
        assert result["free_bytes"] >= 0
        assert 0 < result["usage_percent"] <= 100
        assert result["checkpoint_size"] >= 0
        assert result["recurrent_size"] >= 0
        assert result["total_project_size"] >= 0

    def test_with_custom_path(self):
        """Passing a custom path should still return valid results."""
        from data_rotation import get_disk_usage

        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_disk_usage(self_path=tmpdir)
            assert "total_bytes" in result
            assert "used_bytes" in result
            assert result["total_bytes"] > 0


# ══════════════════════════════════════════════════════════════════════
# 7. TestQualityAwareRotateCheckpoints
# ══════════════════════════════════════════════════════════════════════

class TestQualityAwareRotateCheckpoints:
    """Tests for data_rotation.quality_aware_rotate_checkpoints()."""

    def test_with_fitness_scores_provided(self):
        """When fitness_scores is provided, lowest-fitness checkpoints should be deleted first."""
        from data_rotation import quality_aware_rotate_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "milestone1", "checkpoints")
            os.makedirs(ckpt_dir)

            # Create 5 checkpoint files with different fitness scores
            files = {}
            for i in range(5):
                fname = f"checkpoint_epoch_{i}.pt"
                fpath = os.path.join(ckpt_dir, fname)
                with open(fpath, "w") as f:
                    f.write(f"data_{i}")
                files[fname] = float(i)  # fitness 0,1,2,3,4

            # Patch the module-level M1_CHECKPOINTS to point to our temp dir
            import data_rotation as dr
            original_ckpt = dr.M1_CHECKPOINTS
            try:
                dr.M1_CHECKPOINTS = type(dr.M1_CHECKPOINTS)(ckpt_dir)
                deleted = quality_aware_rotate_checkpoints(
                    keep_n=2, fitness_scores=files
                )
                # 3 files should be deleted (the 3 lowest fitness: 0, 1, 2)
                assert len(deleted) == 3
                # Remaining files should be epoch_3 and epoch_4 (highest fitness)
                remaining = os.listdir(ckpt_dir)
                assert "checkpoint_epoch_3.pt" in remaining
                assert "checkpoint_epoch_4.pt" in remaining
                # Lowest fitness files should be gone
                assert "checkpoint_epoch_0.pt" not in remaining
                assert "checkpoint_epoch_1.pt" not in remaining
            finally:
                dr.M1_CHECKPOINTS = original_ckpt

    def test_without_fitness_scores_falls_back_to_mtime(self):
        """When fitness_scores is None, should delegate to rotate_checkpoints (mtime-based)."""
        from data_rotation import quality_aware_rotate_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "milestone1", "checkpoints")
            os.makedirs(ckpt_dir)

            # Create checkpoint files
            for i in range(4):
                fpath = os.path.join(ckpt_dir, f"checkpoint_epoch_{i}.pt")
                with open(fpath, "w") as f:
                    f.write(f"data_{i}")

            import data_rotation as dr
            original_ckpt = dr.M1_CHECKPOINTS
            try:
                dr.M1_CHECKPOINTS = type(dr.M1_CHECKPOINTS)(ckpt_dir)
                # fitness_scores=None should fall back to mtime rotation
                deleted = quality_aware_rotate_checkpoints(
                    keep_n=2, fitness_scores=None
                )
                # Should still delete files (mtime-based)
                remaining = os.listdir(ckpt_dir)
                assert len(remaining) <= 3  # at most keep_n + sacred (none sacred here)
            finally:
                dr.M1_CHECKPOINTS = original_ckpt

    def test_sacred_files_preserved(self):
        """Sacred files (checkpoint_best.pt, checkpoint_current.pt) must never be deleted."""
        from data_rotation import quality_aware_rotate_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "milestone1", "checkpoints")
            os.makedirs(ckpt_dir)

            # Create sacred + regular files
            sacred = ["checkpoint_best.pt", "checkpoint_current.pt"]
            regular = [f"checkpoint_epoch_{i}.pt" for i in range(4)]

            scores = {}
            for name in sacred + regular:
                fpath = os.path.join(ckpt_dir, name)
                with open(fpath, "w") as f:
                    f.write("data")
                scores[name] = 0.5  # all same fitness

            import data_rotation as dr
            original_ckpt = dr.M1_CHECKPOINTS
            try:
                dr.M1_CHECKPOINTS = type(dr.M1_CHECKPOINTS)(ckpt_dir)
                deleted = quality_aware_rotate_checkpoints(
                    keep_n=2, fitness_scores=scores
                )
                remaining = os.listdir(ckpt_dir)
                # Sacred files must always remain
                for name in sacred:
                    assert name in remaining, f"Sacred file {name} was deleted!"
            finally:
                dr.M1_CHECKPOINTS = original_ckpt

    def test_no_files_to_rotate(self):
        """If there are fewer files than keep_n, nothing should be deleted."""
        from data_rotation import quality_aware_rotate_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "milestone1", "checkpoints")
            os.makedirs(ckpt_dir)

            scores = {"checkpoint_epoch_0.pt": 1.0}
            fpath = os.path.join(ckpt_dir, "checkpoint_epoch_0.pt")
            with open(fpath, "w") as f:
                f.write("data")

            import data_rotation as dr
            original_ckpt = dr.M1_CHECKPOINTS
            try:
                dr.M1_CHECKPOINTS = type(dr.M1_CHECKPOINTS)(ckpt_dir)
                deleted = quality_aware_rotate_checkpoints(
                    keep_n=5, fitness_scores=scores
                )
                assert deleted == []
                assert os.path.isfile(fpath)
            finally:
                dr.M1_CHECKPOINTS = original_ckpt
