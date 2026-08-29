#!/usr/bin/env python3
"""Tests for monitoring_alerts — alert trigger logic and rate limiting.

Relevant to long-term unattended operation: alerts must fire on the right
conditions (regression/OOM/crash-loop/stale-health) and be rate-limited so a
failure storm can't spam channels. Tests avoid network by leaving webhook URLs
unset and asserting on the in-memory alert log / rate-limit gate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring_alerts import AlertManager, AlertThresholds


def _mgr(**threshold_kwargs) -> AlertManager:
    am = AlertManager(AlertThresholds(**threshold_kwargs)) if threshold_kwargs else AlertManager()
    # Defensive: guarantee no channels so nothing hits the network.
    am._discord_url = ""
    am._slack_url = ""
    am._email_to = ""
    return am


class TestRateLimit:
    def test_gate_allows_up_to_limit_then_blocks(self):
        am = _mgr(max_alerts_per_hour=3)
        assert am._check_rate_limit() is True
        assert am._check_rate_limit() is True
        assert am._check_rate_limit() is True
        assert am._check_rate_limit() is False

    def test_send_alert_records_rate_limited_message(self):
        am = _mgr(max_alerts_per_hour=2)
        am.send_alert("warning", "t1", "m1")
        am.send_alert("warning", "t2", "m2")
        am.send_alert("warning", "t3", "m3")  # over limit
        history = am.get_alert_history()
        assert len(history) == 3
        assert "[RATE LIMITED]" in history[-1]["message"]

    def test_critical_bypasses_rate_limit(self):
        am = _mgr(max_alerts_per_hour=1)
        am.send_alert("warning", "t1", "m1")
        am.send_alert("warning", "t2", "m2")  # rate limited
        am.send_alert("critical", "t3", "m3", critical=True)  # bypass
        history = am.get_alert_history()
        # The critical message is recorded normally (not rate-limited).
        assert "[RATE LIMITED]" not in history[-1]["message"]
        assert history[-1]["level"] == "critical"


class TestFitnessRegression:
    def test_no_alert_below_threshold(self):
        am = _mgr()
        assert am.send_fitness_regression(current=95.0, previous_avg=100.0) is False
        assert am.get_alert_history() == []

    def test_no_alert_when_previous_avg_nonpositive(self):
        am = _mgr()
        assert am.send_fitness_regression(current=10.0, previous_avg=0.0) is False
        assert am.get_alert_history() == []

    def test_alert_recorded_above_threshold(self):
        am = _mgr()
        am.send_fitness_regression(current=70.0, previous_avg=100.0)  # 30% drop
        history = am.get_alert_history()
        assert len(history) == 1
        assert history[0]["title"] == "Fitness Regression"


class TestSeverityMapping:
    def test_oom_severity_escalates(self):
        am = _mgr(oom_per_cycle_limit=3)
        am.send_oom_alert(cycle=1, population=50, consecutive=1)
        assert am.get_alert_history()[-1]["level"] == "warning"
        am.send_oom_alert(cycle=1, population=50, consecutive=2)
        assert am.get_alert_history()[-1]["level"] == "error"
        am.send_oom_alert(cycle=1, population=50, consecutive=3)
        assert am.get_alert_history()[-1]["level"] == "critical"

    def test_crash_loop_severity(self):
        am = _mgr(crash_loop_consecutive=5)
        am.send_crash_loop_alert(consecutive_errors=3, cycle=1)
        assert am.get_alert_history()[-1]["level"] == "warning"
        am.send_crash_loop_alert(consecutive_errors=5, cycle=1)
        assert am.get_alert_history()[-1]["level"] == "error"
        am.send_crash_loop_alert(consecutive_errors=10, cycle=1)
        assert am.get_alert_history()[-1]["level"] == "critical"


class TestHealthStaleAndConfig:
    def test_health_stale_threshold(self):
        am = _mgr(health_stale_seconds=300.0)
        assert am.send_health_stale_alert(last_update_seconds=100.0) is False
        assert am.get_alert_history() == []
        am.send_health_stale_alert(last_update_seconds=600.0)
        assert am.get_alert_history()[-1]["title"] == "Health Monitor Stale"

    def test_config_summary_reports_no_channels(self):
        am = _mgr()
        summary = am.get_config_summary()
        assert summary["discord"] is False
        assert summary["slack"] is False
        assert summary["email"] is False
