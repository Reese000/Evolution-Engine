#!/usr/bin/env python3
"""monitoring_alerts.py — Alert system for Evolution Engine failures.

Supports:
  - Discord webhook alerts
  - Slack webhook alerts
  - Email alerts (SMTP)
  - Configurable thresholds for fitness regression, OOM, crash loops

Usage:
    from monitoring_alerts import AlertManager
    alert_mgr = AlertManager()
    alert_mgr.send_fitness_regression(current=0.3, previous=0.8)
    alert_mgr.send_oom_alert(cycle=5, population=50)
    alert_mgr.send_crash_loop_alert(consecutive_errors=10, cycle=15)
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# ── Alert Thresholds ─────────────────────────────────────────────────


@dataclass
class AlertThresholds:
    """Configurable thresholds for triggering alerts."""

    fitness_regression_pct: float = 0.20  # 20% drop from recent avg
    oom_per_cycle_limit: int = 3  # Max OOM events per cycle
    crash_loop_consecutive: int = 5  # Consecutive errors before alert
    crash_loop_window_minutes: float = 30.0  # Time window for crash detection
    max_alerts_per_hour: int = 10  # Rate limit
    health_stale_seconds: float = 300.0  # 5 min without health update


# ── Alert Manager ─────────────────────────────────────────────────────


class AlertManager:
    """Manages alert delivery via webhooks and email.

    Configuration via environment variables:
      - EE_DISCORD_WEBHOOK: Discord webhook URL
      - EE_SLACK_WEBHOOK: Slack webhook URL
      - EE_ALERT_EMAIL_TO: Email recipient
      - EE_ALERT_EMAIL_SMTP: SMTP server (host:port)
      - EE_ALERT_EMAIL_USER: SMTP username
      - EE_ALERT_EMAIL_PASS: SMTP password
      - EE_ALERT_EMAIL_FROM: Sender email address
    """

    def __init__(self, thresholds: AlertThresholds | None = None) -> None:
        self.thresholds = thresholds or AlertThresholds()
        self._discord_url: str = os.environ.get("EE_DISCORD_WEBHOOK", "")
        self._slack_url: str = os.environ.get("EE_SLACK_WEBHOOK", "")
        self._email_to: str = os.environ.get("EE_ALERT_EMAIL_TO", "")
        self._email_smtp: str = os.environ.get("EE_ALERT_EMAIL_SMTP", "")
        self._email_user: str = os.environ.get("EE_ALERT_EMAIL_USER", "")
        self._email_pass: str = os.environ.get("EE_ALERT_EMAIL_PASS", "")
        self._email_from: str = os.environ.get("EE_ALERT_EMAIL_FROM", "")
        self._alert_log: list[dict[str, Any]] = []
        self._alert_times: list[float] = []

    def _check_rate_limit(self) -> bool:
        """Return True if we can send an alert (not rate-limited)."""
        import time

        now = time.time()
        cutoff = now - 3600.0  # 1 hour window
        self._alert_times = [t for t in self._alert_times if t > cutoff]
        if len(self._alert_times) >= self.thresholds.max_alerts_per_hour:
            return False
        self._alert_times.append(now)
        return True

    def _record_alert(self, level: str, title: str, message: str) -> None:
        """Record alert in memory log."""
        self._alert_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "title": title,
                "message": message,
            }
        )

    def send_alert(self, level: str, title: str, message: str, critical: bool = False) -> bool:
        """Send alert to all configured channels.

        Args:
            level: Alert severity ("warning", "error", "critical")
            title: Short alert title
            message: Detailed alert message
            critical: If True, bypass rate limiting

        Returns:
            True if at least one alert was sent successfully.
        """
        if not critical and not self._check_rate_limit():
            self._record_alert(level, title, f"[RATE LIMITED] {message}")
            return False

        sent = False

        # Discord
        if self._discord_url:
            sent |= self._send_discord(level, title, message)

        # Slack
        if self._slack_url:
            sent |= self._send_slack(level, title, message)

        # Email (critical only)
        if self._email_to and self._email_smtp and critical:
            sent |= self._send_email(level, title, message)

        self._record_alert(level, title, message)
        return sent

    def _send_discord(self, level: str, title: str, message: str) -> bool:
        """Send alert via Discord webhook."""
        color_map = {
            "warning": 0xFFAA00,
            "error": 0xFF4444,
            "critical": 0xFF0000,
        }
        embed = {
            "title": f"🔔 Evolution Engine — {title}",
            "description": message,
            "color": color_map.get(level, 0x808080),
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": f"Severity: {level.upper()}"},
        }
        payload = json.dumps({"embeds": [embed]}).encode("utf-8")
        return self._http_post(self._discord_url, payload)

    def _send_slack(self, level: str, title: str, message: str) -> bool:
        """Send alert via Slack webhook."""
        emoji_map = {
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨",
        }
        emoji = emoji_map.get(level, "📢")
        text = f"{emoji} *{title}*\n{message}\n_Severity: {level.upper()}_"
        payload = json.dumps({"text": text}).encode("utf-8")
        return self._http_post(self._slack_url, payload)

    def _send_email(self, level: str, title: str, message: str) -> bool:
        """Send alert via SMTP email."""
        try:
            host, port = self._email_smtp.split(":")
            port = int(port)
        except (ValueError, AttributeError):
            return False

        try:
            body = (
                f"Subject: [{level.upper()}] Evolution Engine — {title}\n"
                f"From: {self._email_from or 'evo-alerts@localhost'}\n"
                f"To: {self._email_to}\n"
                f"Content-Type: text/plain; charset=utf-8\n"
                f"\n"
                f"Evolution Engine Alert\n"
                f"{'=' * 50}\n"
                f"Level: {level.upper()}\n"
                f"Title: {title}\n"
                f"Time: {datetime.now().isoformat()}\n"
                f"\n"
                f"{message}\n"
            )

            context = ssl.create_default_context()
            with smtplib.SMTP(host, port, timeout=10) as server:
                server.starttls(context=context)
                if self._email_user and self._email_pass:
                    server.login(self._email_user, self._email_pass)
                server.sendmail(
                    self._email_from or "evo-alerts@localhost",
                    self._email_to,
                    body.encode("utf-8"),
                )
            return True
        except (smtplib.SMTPException, OSError, ssl.SSLError):
            return False

    def _http_post(self, url: str, payload: bytes) -> bool:
        """POST JSON payload to a webhook URL."""
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in (200, 204)
        except (urllib.error.URLError, OSError, ValueError):
            return False

    # ── Convenience alert methods ──────────────────────────────────

    def send_fitness_regression(self, current: float, previous_avg: float, cycle: int = 0) -> bool:
        """Alert when fitness drops significantly from recent average."""
        if previous_avg <= 0:
            return False
        drop_pct = (previous_avg - current) / previous_avg
        if drop_pct < self.thresholds.fitness_regression_pct:
            return False
        return self.send_alert(
            "warning",
            "Fitness Regression",
            f"Fitness dropped {drop_pct:.0%}: {current:.3f} " f"(avg was {previous_avg:.3f}) at cycle {cycle}",
        )

    def send_oom_alert(self, cycle: int, population: int, consecutive: int = 1) -> bool:
        """Alert on out-of-memory errors."""
        if consecutive >= self.thresholds.oom_per_cycle_limit:
            level = "critical"
        elif consecutive >= 2:
            level = "error"
        else:
            level = "warning"
        return self.send_alert(
            level,
            "Out of Memory",
            f"OOM in cycle {cycle}, population={population}, " f"consecutive OOMs={consecutive}",
            critical=(consecutive >= self.thresholds.oom_per_cycle_limit),
        )

    def send_crash_loop_alert(self, consecutive_errors: int, cycle: int) -> bool:
        """Alert on repeated crashes."""
        level = (
            "critical"
            if consecutive_errors >= self.thresholds.crash_loop_consecutive * 2
            else "error" if consecutive_errors >= self.thresholds.crash_loop_consecutive else "warning"
        )
        return self.send_alert(
            level,
            "Crash Loop Detected",
            f"{consecutive_errors} consecutive errors in cycle {cycle}",
            critical=(consecutive_errors >= self.thresholds.crash_loop_consecutive * 2),
        )

    def send_health_stale_alert(self, last_update_seconds: float) -> bool:
        """Alert when health.json hasn't been updated recently."""
        if last_update_seconds < self.thresholds.health_stale_seconds:
            return False
        return self.send_alert(
            "warning",
            "Health Monitor Stale",
            f"health.json last updated {last_update_seconds:.0f}s ago "
            f"(threshold: {self.thresholds.health_stale_seconds:.0f}s)",
        )

    def send_retrain_failure_alert(self, cycle: int, attempts: int, reason: str) -> bool:
        """Alert on retrain failure after exhaustion."""
        return self.send_alert(
            "error",
            "Retrain Failed",
            f"Retrain failed after {attempts} attempts in cycle {cycle}: {reason}",
        )

    def send_monoculture_alert(self, dominant_strategy: str, dominance_ratio: float, recommendation: str) -> bool:
        """Alert when a single strategy dominates the population."""
        level = "warning" if dominance_ratio > 0.7 else "info"
        return self.send_alert(
            level=level,
            title="Strategy Monoculture Detected",
            message=f"{dominant_strategy} at {dominance_ratio:.0%} dominance. Recommendation: {recommendation}",
            critical=(dominance_ratio > 0.8),
        )

    def send_recovery_stuck_alert(self, cycles_in_recovery: int, fitness_trend: str, confidence: float) -> bool:
        """Alert when recovery mode is stuck and not progressing."""
        return self.send_alert(
            level="warning",
            title="Recovery Mode Stuck",
            message=f"Recovery active for {cycles_in_recovery} cycles. Trend: {fitness_trend}, confidence: {confidence:.2f}",
            critical=(cycles_in_recovery >= 10),
        )

    # ── Introspection ──────────────────────────────────────────────

    def get_alert_history(self) -> list[dict[str, Any]]:
        """Return list of all alerts sent this session."""
        return list(self._alert_log)

    def get_config_summary(self) -> dict[str, Any]:
        """Return a summary of configured alert channels."""
        return {
            "discord": bool(self._discord_url),
            "slack": bool(self._slack_url),
            "email": bool(self._email_to),
            "thresholds": {
                "fitness_regression_pct": self.thresholds.fitness_regression_pct,
                "oom_per_cycle_limit": self.thresholds.oom_per_cycle_limit,
                "crash_loop_consecutive": self.thresholds.crash_loop_consecutive,
                "max_alerts_per_hour": self.thresholds.max_alerts_per_hour,
            },
        }
