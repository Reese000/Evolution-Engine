#!/usr/bin/env python3
"""Comprehensive tests for Wave 7 additions.

Tests cover:
  1. TestMonocultureAlert — monoculture dominance alerting with mock verification
  2. TestRecoveryStuckAlert — recovery mode stuck alerting with mock verification
  3. TestPlateauAnalysis — plateau detection via MetricsCollector.get_trend
  4. TestCompareWindows — improving / declining / stable metric trend detection
  5. TestGetDiskUsage — disk usage dict structure and value validation
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from metrics import MetricsCollector

# Path setup
from monitoring_alerts import AlertManager

# ══════════════════════════════════════════════════════════════════════
# 1. TestMonocultureAlert
# ══════════════════════════════════════════════════════════════════════


class TestMonocultureAlert:
    """Tests for AlertManager.send_monoculture_alert() with mock verification."""

    def test_alert_fires_when_dominance_above_07(self):
        """High dominance (> 0.7) should invoke send_alert with 'warning' level."""
        am = AlertManager()
        with patch.object(am, "send_alert", return_value=True) as mock_send:
            am.send_monoculture_alert("RANDOM", 0.85, "Increase mutation diversity")
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[1]["level"] == "warning" or call_args[0][0] == "warning"
            # Should NOT be critical (0.85 < 0.8 threshold for critical)
            # Actually 0.85 > 0.8 so it IS critical
            assert call_args[1].get("critical", False) is True

    def test_info_level_for_moderate_dominance(self):
        """Moderate dominance (0.5–0.7) should use 'info' level, not critical."""
        am = AlertManager()
        with patch.object(am, "send_alert", return_value=True) as mock_send:
            am.send_monoculture_alert("TASK_AWARE", 0.6, "Monitor diversity")
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[1]["level"] == "info" or call_args[0][0] == "info"
            assert call_args[1].get("critical", False) is False

    def test_critical_at_above_08(self):
        """Dominance > 0.8 should set critical=True."""
        am = AlertManager()
        with patch.object(am, "send_alert", return_value=True) as mock_send:
            am.send_monoculture_alert("DIRECTED", 0.92, "Force diversity injection")
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[1].get("critical", False) is True

    def test_warning_at_boundary_071(self):
        """Dominance of 0.71 should be warning but not critical."""
        am = AlertManager()
        with patch.object(am, "send_alert", return_value=True) as mock_send:
            am.send_monoculture_alert("STRAT_X", 0.71, "Watch closely")
            call_args = mock_send.call_args
            assert call_args[1]["level"] == "warning" or call_args[0][0] == "warning"
            assert call_args[1].get("critical", False) is False

    def test_message_contains_strategy_and_ratio(self):
        """Alert message should include the strategy name and dominance percentage."""
        am = AlertManager()
        am.send_monoculture_alert("MUTATION_ONLY", 0.75, "Diversify strategies")
        history = am.get_alert_history()
        assert len(history) == 1
        assert "MUTATION_ONLY" in history[0]["message"]
        assert "75%" in history[0]["message"]


# ══════════════════════════════════════════════════════════════════════
# 2. TestRecoveryStuckAlert
# ══════════════════════════════════════════════════════════════════════


class TestRecoveryStuckAlert:
    """Tests for AlertManager.send_recovery_stuck_alert() with mock verification."""

    def test_alert_fires_with_correct_message(self):
        """Recovery stuck alert should record correct cycle count and trend."""
        am = AlertManager()
        with patch.object(am, "send_alert", return_value=True) as mock_send:
            am.send_recovery_stuck_alert(5, "declining", 0.75)
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            msg = call_args[1]["message"]
            assert "5 cycles" in msg
            assert "declining" in msg
            assert "0.75" in msg

    def test_critical_at_10_plus_cycles(self):
        """10+ cycles in recovery should mark the alert as critical."""
        am = AlertManager()
        with patch.object(am, "send_alert", return_value=True) as mock_send:
            am.send_recovery_stuck_alert(10, "stable", 0.90)
            call_args = mock_send.call_args
            assert call_args[1].get("critical", False) is True

    def test_critical_at_15_cycles(self):
        """15 cycles should also be critical."""
        am = AlertManager()
        with patch.object(am, "send_alert", return_value=True) as mock_send:
            am.send_recovery_stuck_alert(15, "declining", 0.85)
            call_args = mock_send.call_args
            assert call_args[1].get("critical", False) is True
            assert "15 cycles" in call_args[1]["message"]

    def test_not_critical_below_10(self):
        """9 cycles should not be critical."""
        am = AlertManager()
        with patch.object(am, "send_alert", return_value=True) as mock_send:
            am.send_recovery_stuck_alert(9, "declining", 0.60)
            call_args = mock_send.call_args
            assert call_args[1].get("critical", False) is False

    def test_always_warning_level(self):
        """Recovery stuck should always use warning level."""
        am = AlertManager()
        with patch.object(am, "send_alert", return_value=True) as mock_send:
            am.send_recovery_stuck_alert(12, "stable", 0.95)
            call_args = mock_send.call_args
            assert call_args[1]["level"] == "warning" or call_args[0][0] == "warning"

    def test_confidence_in_message(self):
        """Confidence value should appear in the alert message."""
        am = AlertManager()
        am.send_recovery_stuck_alert(3, "improving", 0.42)
        history = am.get_alert_history()
        assert "0.42" in history[0]["message"]


# ══════════════════════════════════════════════════════════════════════
# 3. TestPlateauAnalysis
# ══════════════════════════════════════════════════════════════════════


class TestPlateauAnalysis:
    """Tests for plateau detection via MetricsCollector.get_trend()."""

    def test_plateau_data_detected_as_stable(self):
        """Fitness values that barely change should be classified as 'stable'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            # Record 10 values all very close to 50.0
            for i in range(10):
                mc.record_cycle({"fitness": 50.0 + (i % 2) * 0.001})
            trend = mc.get_trend("fitness", window=10)
            assert trend["trend"] == "stable"
            assert trend["window_size"] == 10

    def test_improving_data_detected(self):
        """Steadily increasing fitness should be classified as 'improving'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            for i in range(12):
                mc.record_cycle({"fitness": 10.0 + i * 5.0})
            trend = mc.get_trend("fitness", window=12)
            assert trend["trend"] == "improving"
            assert trend["current"] == 65.0

    def test_insufficient_data_single_record(self):
        """A single data point should return 'stable' (not enough to judge trend)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            mc.record_cycle({"fitness": 42.0})
            trend = mc.get_trend("fitness", window=10)
            assert trend["trend"] == "stable"
            assert trend["window_size"] == 1
            assert trend["current"] == 42.0

    def test_empty_data_returns_unknown(self):
        """No records at all should return trend='unknown'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            trend = mc.get_trend("fitness", window=10)
            assert trend["trend"] == "unknown"
            assert trend["window_size"] == 0
            assert trend["current"] is None

    def test_declining_plateau_detected(self):
        """Consistently decreasing values should be classified as 'declining'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            for i in range(12):
                mc.record_cycle({"fitness": 100.0 - i * 3.0})
            trend = mc.get_trend("fitness", window=12)
            assert trend["trend"] == "declining"

    def test_plateau_min_max_range_tight(self):
        """Plateau data should have a very tight min/max range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            for i in range(10):
                mc.record_cycle({"fitness": 75.0 + i * 0.01})
            trend = mc.get_trend("fitness", window=10)
            assert trend["trend"] == "stable"
            assert trend["max"] - trend["min"] < 0.2


# ══════════════════════════════════════════════════════════════════════
# 4. TestCompareWindows
# ══════════════════════════════════════════════════════════════════════


class TestCompareWindows:
    """Tests for metric trend comparison (improving / declining / stable)
    using MetricsCollector.get_trend with different data patterns."""

    def test_improving_metric(self):
        """A metric with strong upward trajectory should be 'improving'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            # First half low, second half high → improving
            for i in range(10):
                mc.record_cycle({"score": 1.0})
            for i in range(10):
                mc.record_cycle({"score": 9.0})
            trend = mc.get_trend("score", window=20)
            assert trend["trend"] == "improving"
            assert trend["avg"] == pytest.approx(5.0, abs=0.1)

    def test_declining_metric(self):
        """A metric with strong downward trajectory should be 'declining'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            # First half high, second half low → declining
            for i in range(10):
                mc.record_cycle({"score": 9.0})
            for i in range(10):
                mc.record_cycle({"score": 1.0})
            trend = mc.get_trend("score", window=20)
            assert trend["trend"] == "declining"

    def test_stable_metric(self):
        """A metric that stays flat should be 'stable'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            for i in range(10):
                mc.record_cycle({"score": 50.0})
            trend = mc.get_trend("score", window=10)
            assert trend["trend"] == "stable"
            assert trend["current"] == 50.0
            assert trend["min"] == 50.0
            assert trend["max"] == 50.0

    def test_two_point_improving(self):
        """With only 2 data points, the last > first should be 'improving'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            mc.record_cycle({"metric": 10.0})
            mc.record_cycle({"metric": 20.0})
            trend = mc.get_trend("metric", window=10)
            assert trend["trend"] == "improving"
            assert trend["window_size"] == 2

    def test_two_point_stable(self):
        """With 2 equal data points, trend should be 'stable'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            mc.record_cycle({"metric": 50.0})
            mc.record_cycle({"metric": 50.0})
            trend = mc.get_trend("metric", window=10)
            assert trend["trend"] == "stable"

    def test_window_size_respected(self):
        """get_trend should only consider the most recent 'window' records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            # First 10 records: declining (100 → 55)
            for i in range(10):
                mc.record_cycle({"metric": 100.0 - i * 5.0})
            # Next 10 records: stable at 50.0
            for i in range(10):
                mc.record_cycle({"metric": 50.0})
            # With window=10, only the stable tail is seen
            trend = mc.get_trend("metric", window=10)
            assert trend["trend"] == "stable"
            assert trend["current"] == 50.0


# ══════════════════════════════════════════════════════════════════════
# 5. TestGetDiskUsage
# ══════════════════════════════════════════════════════════════════════


class TestGetDiskUsage:
    """Tests for data_rotation.get_disk_usage()."""

    def test_returns_dict_with_required_keys(self):
        """get_disk_usage should return a dict with all expected keys."""
        from data_rotation import get_disk_usage

        result = get_disk_usage()
        required_keys = [
            "total_bytes",
            "used_bytes",
            "free_bytes",
            "usage_percent",
            "checkpoint_size",
            "recurrent_size",
            "total_project_size",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_total_is_used_plus_free(self):
        """total_bytes should equal used_bytes + free_bytes."""
        from data_rotation import get_disk_usage

        result = get_disk_usage()
        assert result["total_bytes"] == result["used_bytes"] + result["free_bytes"]

    def test_usage_percent_calculation(self):
        """usage_percent should equal used/total * 100."""
        from data_rotation import get_disk_usage

        result = get_disk_usage()
        expected_pct = (result["used_bytes"] / result["total_bytes"]) * 100
        assert result["usage_percent"] == pytest.approx(expected_pct, rel=1e-6)

    def test_total_bytes_positive(self):
        """total_bytes should be a positive integer."""
        from data_rotation import get_disk_usage

        result = get_disk_usage()
        assert isinstance(result["total_bytes"], int)
        assert result["total_bytes"] > 0

    def test_with_custom_path(self):
        """Passing a custom path should still return valid results."""
        from data_rotation import get_disk_usage

        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_disk_usage(self_path=tmpdir)
            assert "total_bytes" in result
            assert "used_bytes" in result
            assert "free_bytes" in result
            assert result["total_bytes"] > 0

    def test_non_negative_sizes(self):
        """All size fields should be non-negative."""
        from data_rotation import get_disk_usage

        result = get_disk_usage()
        assert result["checkpoint_size"] >= 0
        assert result["recurrent_size"] >= 0
        assert result["total_project_size"] >= 0
