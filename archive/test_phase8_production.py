#!/usr/bin/env python3
"""Tests for Phase 8 (Production Hardening).

Validates:
  - Dockerfile: syntax and key directives present
  - docker-compose.yml: valid YAML, required services/sections
  - monitoring_alerts.py: AlertManager, thresholds, webhook config, rate limiting
  - log_rotation.py: CompressedRotatingFileHandler, component loggers, compression
  - deploy.sh / stop.sh / status.sh: existence, executability, key logic
"""

import gzip
import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

try:
    import log_rotation  # noqa: F401
    _has_log_rotation = True
except ImportError:
    _has_log_rotation = False

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════════════
# Dockerfile tests
# ══════════════════════════════════════════════════════════════════════


class TestDockerfile(unittest.TestCase):
    """Tests for Dockerfile correctness."""

    def setUp(self):
        self.dockerfile_path = os.path.join(str(PROJECT_ROOT), "Dockerfile")
        self._content = None
        if os.path.isfile(self.dockerfile_path):
            with open(self.dockerfile_path, encoding="utf-8") as f:
                self._content = f.read()

    @property
    def content(self) -> str:
        return self._content or ""

    def test_dockerfile_exists(self):
        self.assertTrue(os.path.isfile(self.dockerfile_path))

    def test_uses_python_311(self):
        self.assertIn("python:3.11", self.content)

    def test_has_cuda_arg(self):
        self.assertIn("ARG CUDA", self.content)

    def test_has_cuda_base_image(self):
        self.assertIn("nvidia/cuda", self.content)

    def test_has_healthcheck(self):
        self.assertIn("HEALTHCHECK", self.content)

    def test_has_entrypoint(self):
        self.assertIn("ENTRYPOINT", self.content)

    def test_has_user(self):
        self.assertIn("USER", self.content)

    def test_installs_requirements(self):
        self.assertIn("requirements.txt", self.content)

    def test_has_copy_instructions(self):
        self.assertIn("COPY", self.content)


# ══════════════════════════════════════════════════════════════════════
# docker-compose.yml tests
# ══════════════════════════════════════════════════════════════════════


class TestDockerCompose(unittest.TestCase):
    """Tests for docker-compose.yml correctness."""

    def setUp(self):
        self.compose_path = os.path.join(str(PROJECT_ROOT), "docker-compose.yml")
        self._content = None
        if os.path.isfile(self.compose_path):
            with open(self.compose_path, encoding="utf-8") as f:
                self._content = f.read()

    @property
    def content(self) -> str:
        return self._content or ""

    def test_compose_exists(self):
        self.assertTrue(os.path.isfile(self.compose_path))

    def test_has_services(self):
        self.assertIn("services:", self.content)

    def test_has_evolution_engine_service(self):
        self.assertIn("evolution-engine", self.content)

    def test_has_volumes(self):
        self.assertIn("volumes:", self.content)

    def test_has_environment(self):
        self.assertIn("environment:", self.content)

    def test_has_healthcheck(self):
        self.assertIn("healthcheck:", self.content)

    def test_has_logging_config(self):
        self.assertIn("logging:", self.content)

    def test_has_deploy_resources(self):
        self.assertIn("deploy:", self.content)

    def test_has_webhook_env_vars(self):
        self.assertIn("EE_DISCORD_WEBHOOK", self.content)
        self.assertIn("EE_SLACK_WEBHOOK", self.content)


# ══════════════════════════════════════════════════════════════════════
# monitoring_alerts.py tests
# ══════════════════════════════════════════════════════════════════════


class TestMonitoringAlerts(unittest.TestCase):
    """Tests for AlertManager class."""

    def test_import(self):
        from monitoring_alerts import AlertManager, AlertThresholds

        self.assertTrue(callable(AlertManager))
        self.assertTrue(callable(AlertThresholds))

    def test_default_thresholds(self):
        from monitoring_alerts import AlertThresholds

        t = AlertThresholds()
        self.assertGreater(t.fitness_regression_pct, 0)
        self.assertGreater(t.oom_per_cycle_limit, 0)
        self.assertGreater(t.crash_loop_consecutive, 0)
        self.assertGreater(t.max_alerts_per_hour, 0)
        self.assertGreater(t.health_stale_seconds, 0)

    def test_custom_thresholds(self):
        from monitoring_alerts import AlertThresholds

        t = AlertThresholds(fitness_regression_pct=0.30, crash_loop_consecutive=3)
        self.assertAlmostEqual(t.fitness_regression_pct, 0.30)
        self.assertEqual(t.crash_loop_consecutive, 3)

    def test_alert_manager_init(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        self.assertIsNotNone(am.thresholds)
        self.assertIsInstance(am.get_alert_history(), list)
        self.assertEqual(len(am.get_alert_history()), 0)

    def test_config_summary(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        summary = am.get_config_summary()
        self.assertIn("discord", summary)
        self.assertIn("slack", summary)
        self.assertIn("email", summary)
        self.assertIn("thresholds", summary)
        # Without env vars, all should be False
        self.assertFalse(summary["discord"])
        self.assertFalse(summary["slack"])
        self.assertFalse(summary["email"])

    def test_record_alert(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        am._record_alert("warning", "Test", "message")
        history = am.get_alert_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["level"], "warning")
        self.assertEqual(history[0]["title"], "Test")

    def test_rate_limiting(self):
        from monitoring_alerts import AlertManager, AlertThresholds

        am = AlertManager(AlertThresholds(max_alerts_per_hour=3))
        # First 3 should pass
        for _ in range(3):
            self.assertTrue(am._check_rate_limit())
        # 4th should be rate-limited
        self.assertFalse(am._check_rate_limit())

    def test_fitness_regression_below_threshold(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        # No regression (current > previous)
        result = am.send_fitness_regression(current=1.0, previous_avg=0.5, cycle=1)
        self.assertFalse(result)

    def test_fitness_regression_no_previous(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am.send_fitness_regression(current=0.1, previous_avg=0.0, cycle=1)
        self.assertFalse(result)

    def test_send_alert_no_channels(self):
        """With no webhook configured, send_alert returns False but records."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am.send_alert("warning", "Test", "msg")
        self.assertFalse(result)
        self.assertGreater(len(am.get_alert_history()), 0)

    def test_oom_alert_records(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        am.send_oom_alert(cycle=1, population=50)
        history = am.get_alert_history()
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0]["title"], "Out of Memory")

    def test_crash_loop_alert_records(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        am.send_crash_loop_alert(consecutive_errors=3, cycle=1)
        history = am.get_alert_history()
        self.assertGreater(len(history), 0)

    def test_retrain_failure_alert(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        am.send_retrain_failure_alert(cycle=5, attempts=3, reason="NaN loss")
        history = am.get_alert_history()
        self.assertGreater(len(history), 0)
        self.assertIn("NaN loss", history[0]["message"])

    def test_health_stale_alert_below_threshold(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am.send_health_stale_alert(last_update_seconds=60)
        self.assertFalse(result)

    def test_health_stale_alert_above_threshold(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        am.send_health_stale_alert(last_update_seconds=600)
        history = am.get_alert_history()
        self.assertGreater(len(history), 0)

    def test_http_post_failure(self):
        """_http_post should return False on invalid URL."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am._http_post("http://localhost:99999/fake", b"{}")
        self.assertFalse(result)


# ══════════════════════════════════════════════════════════════════════
# log_rotation.py tests
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _has_log_rotation, reason="Module archived 2026-07-29")
class TestLogRotation(unittest.TestCase):
    """Tests for production log rotation."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._log_dir = os.path.join(self._tmpdir, "logs")

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_import(self):
        from log_rotation import compress_old_logs, get_component_logger, get_log_stats, setup_production_logging

        self.assertTrue(callable(setup_production_logging))
        self.assertTrue(callable(get_component_logger))
        self.assertTrue(callable(compress_old_logs))
        self.assertTrue(callable(get_log_stats))

    def test_component_names(self):
        from log_rotation import COMPONENT_NAMES

        self.assertIn("evolution", COMPONENT_NAMES)
        self.assertIn("retrain", COMPONENT_NAMES)
        self.assertIn("errors", COMPONENT_NAMES)
        self.assertIn("alerts", COMPONENT_NAMES)
        self.assertIn("monitoring", COMPONENT_NAMES)

    def test_setup_creates_log_dir(self):
        from log_rotation import setup_production_logging

        setup_production_logging(log_dir=self._log_dir)
        self.assertTrue(os.path.isdir(self._log_dir))

    def test_setup_creates_component_files(self):
        from log_rotation import setup_production_logging

        setup_production_logging(log_dir=self._log_dir)
        for comp in ("evolution", "retrain", "errors", "alerts", "monitoring"):
            log_path = os.path.join(self._log_dir, f"{comp}.log")
            self.assertTrue(os.path.isfile(log_path), f"Missing {comp}.log")

    def test_setup_creates_root_log(self):
        from log_rotation import setup_production_logging

        setup_production_logging(log_dir=self._log_dir)
        root_log = os.path.join(self._log_dir, "evolution.log")
        self.assertTrue(os.path.isfile(root_log))

    def test_component_logger_writes_to_file(self):
        from log_rotation import get_component_logger, setup_production_logging

        setup_production_logging(log_dir=self._log_dir)
        logger = get_component_logger("evolution")
        logger.info("Test evolution message")
        # Flush handlers
        for h in logger.handlers:
            h.flush()
        log_path = os.path.join(self._log_dir, "evolution.log")
        with open(log_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Test evolution message", content)

    def test_errors_logger_filters_level(self):
        from log_rotation import get_component_logger, setup_production_logging

        setup_production_logging(log_dir=self._log_dir)
        logger = get_component_logger("errors")
        logger.debug("This should not appear")
        logger.error("This should appear")
        for h in logger.handlers:
            h.flush()
        log_path = os.path.join(self._log_dir, "errors.log")
        with open(log_path, encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("should not appear", content)
        self.assertIn("should appear", content)

    def test_compress_old_logs(self):
        from log_rotation import compress_old_logs

        # Create an old log file
        os.makedirs(self._log_dir, exist_ok=True)
        old_log = os.path.join(self._log_dir, "old.log")
        with open(old_log, "w", encoding="utf-8") as f:
            f.write("old log content\n" * 100)
        # Make it "old" by setting mtime
        old_time = time.time() - (31 * 86400)
        os.utime(old_log, (old_time, old_time))

        compressed = compress_old_logs(self._log_dir, max_age_days=30)
        self.assertEqual(len(compressed), 1)
        self.assertFalse(os.path.isfile(old_log))
        self.assertTrue(os.path.isfile(old_log + ".gz"))

        # Verify content
        with gzip.open(old_log + ".gz", "rt", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("old log content", content)

    def test_compress_skips_recent(self):
        from log_rotation import compress_old_logs

        os.makedirs(self._log_dir, exist_ok=True)
        recent_log = os.path.join(self._log_dir, "recent.log")
        with open(recent_log, "w", encoding="utf-8") as f:
            f.write("recent content\n")
        compressed = compress_old_logs(self._log_dir, max_age_days=30)
        self.assertEqual(len(compressed), 0)
        self.assertTrue(os.path.isfile(recent_log))

    def test_get_log_stats_empty(self):
        from log_rotation import get_log_stats

        stats = get_log_stats("/nonexistent/path")
        self.assertEqual(stats["total_files"], 0)
        self.assertEqual(stats["total_size_mb"], 0.0)

    def test_get_log_stats_populated(self):
        from log_rotation import get_log_stats

        os.makedirs(self._log_dir, exist_ok=True)
        for name in ("evolution.log", "retrain.log", "errors.log"):
            with open(os.path.join(self._log_dir, name), "w", encoding="utf-8") as f:
                f.write("test line\n" * 1000)  # Enough to register in MB
        stats = get_log_stats(self._log_dir)
        self.assertEqual(stats["total_files"], 3)
        self.assertGreater(stats["total_size_mb"], 0)
        self.assertIn("evolution", stats["component_files"])

    def test_get_log_stats_with_compressed(self):
        from log_rotation import get_log_stats

        os.makedirs(self._log_dir, exist_ok=True)
        # Create a .gz file manually
        gz_path = os.path.join(self._log_dir, "old.log.gz")
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            f.write("compressed content\n")
        stats = get_log_stats(self._log_dir)
        self.assertEqual(stats["compressed_count"], 1)

    def test_compressed_rotating_handler_compresses(self):
        from log_rotation import CompressedRotatingFileHandler

        log_file = os.path.join(self._log_dir, "test.log")
        os.makedirs(self._log_dir, exist_ok=True)
        handler = CompressedRotatingFileHandler(
            log_file,
            maxBytes=100,
            backupCount=3,
        )
        logger = logging.getLogger("test_rotating")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # Write enough to trigger rotation
        for i in range(20):
            logger.info(f"Test line {i}: {'x' * 20}")
        handler.flush()

        # Check that rotation happened and compression occurred
        files = os.listdir(self._log_dir)
        gz_files = [f for f in files if f.endswith(".gz")]
        # At least one gz file should exist after rotation
        self.assertTrue(
            len(gz_files) > 0 or len([f for f in files if f.endswith(".log1")]) > 0, f"Expected rotation, got: {files}"
        )

        logger.removeHandler(handler)
        handler.close()


# ══════════════════════════════════════════════════════════════════════
# Deployment scripts tests
# ══════════════════════════════════════════════════════════════════════


class TestDeploymentScripts(unittest.TestCase):
    """Tests for deploy.sh, stop.sh, status.sh."""

    def test_deploy_sh_exists(self):
        path = os.path.join(str(PROJECT_ROOT), "deploy.sh")
        self.assertTrue(os.path.isfile(path))

    def test_stop_sh_exists(self):
        path = os.path.join(str(PROJECT_ROOT), "stop.sh")
        self.assertTrue(os.path.isfile(path))

    def test_status_sh_exists(self):
        path = os.path.join(str(PROJECT_ROOT), "status.sh")
        self.assertTrue(os.path.isfile(path))

    def test_deploy_sh_has_shebang(self):
        with open(os.path.join(str(PROJECT_ROOT), "deploy.sh"), encoding="utf-8") as f:
            first_line = f.readline()
        self.assertTrue(first_line.startswith("#!/"))

    def test_stop_sh_has_shebang(self):
        with open(os.path.join(str(PROJECT_ROOT), "stop.sh"), encoding="utf-8") as f:
            first_line = f.readline()
        self.assertTrue(first_line.startswith("#!/"))

    def test_status_sh_has_shebang(self):
        with open(os.path.join(str(PROJECT_ROOT), "status.sh"), encoding="utf-8") as f:
            first_line = f.readline()
        self.assertTrue(first_line.startswith("#!/"))

    def test_deploy_sh_has_docker_support(self):
        with open(os.path.join(str(PROJECT_ROOT), "deploy.sh"), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("docker", content.lower())

    def test_deploy_sh_has_dry_run(self):
        with open(os.path.join(str(PROJECT_ROOT), "deploy.sh"), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("dry-run", content)

    def test_stop_sh_has_force_mode(self):
        with open(os.path.join(str(PROJECT_ROOT), "stop.sh"), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("--force", content)
        self.assertIn("SIGTERM", content)

    def test_status_sh_has_json_mode(self):
        with open(os.path.join(str(PROJECT_ROOT), "status.sh"), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("--json", content)
        self.assertIn("health.json", content)

    def test_deploy_sh_references_config(self):
        with open(os.path.join(str(PROJECT_ROOT), "deploy.sh"), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("engine_config.json", content)
        self.assertIn("autonomous_runner.py", content)

    def test_stop_sh_has_pid_file(self):
        with open(os.path.join(str(PROJECT_ROOT), "stop.sh"), encoding="utf-8") as f:
            content = f.read()
        self.assertIn(".runner.pid", content)


# ══════════════════════════════════════════════════════════════════════
# Integration: AlertManager thresholds + autonomous_runner awareness
# ══════════════════════════════════════════════════════════════════════


class TestAlertIntegration(unittest.TestCase):
    """Integration tests for alert thresholds and autonomous_runner."""

    def test_fitness_regression_detection(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        # 50% drop should trigger
        am.send_fitness_regression(current=0.4, previous_avg=0.8, cycle=5)
        self.assertGreater(len(am.get_alert_history()), 0)

    def test_oom_consecutive_critical(self):
        from monitoring_alerts import AlertManager, AlertThresholds

        am = AlertManager(AlertThresholds(oom_per_cycle_limit=2))
        am.send_oom_alert(cycle=1, population=50, consecutive=3)
        history = am.get_alert_history()
        self.assertEqual(history[0]["level"], "critical")

    def test_crash_loop_critical_level(self):
        from monitoring_alerts import AlertManager, AlertThresholds

        am = AlertManager(AlertThresholds(crash_loop_consecutive=3))
        am.send_crash_loop_alert(consecutive_errors=10, cycle=1)
        history = am.get_alert_history()
        self.assertEqual(history[0]["level"], "critical")

    def test_multiple_alerts_tracked(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        am._record_alert("warning", "A", "msg1")
        am._record_alert("error", "B", "msg2")
        am._record_alert("critical", "C", "msg3")
        history = am.get_alert_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["title"], "A")
        self.assertEqual(history[2]["title"], "C")


if __name__ == "__main__":
    unittest.main()
