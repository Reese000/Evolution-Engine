#!/usr/bin/env python3
"""Tests for Phase 3 (Structured Logging + Metrics) and Phase 6 (Unified Config).

Validates:
  - logging_config.py: setup_logging, get_logger, colored/plain formatters
  - metrics.py: MetricsCollector record, get_history, get_trend
  - config.py: Config loading, env overrides, validation, CLI args
  - autonomous_runner.py: Integration with unified Config
"""

import logging
import os
import tempfile
import unittest
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ══════════════════════════════════════════════════════════════════════
# Phase 3.1: Logging Config
# ══════════════════════════════════════════════════════════════════════


class TestLoggingConfig(unittest.TestCase):
    """Tests for logging_config.py module."""

    def test_import_logging_config(self):
        from logging_config import get_logger, setup_logging

        self.assertTrue(callable(setup_logging))
        self.assertTrue(callable(get_logger))

    def test_get_logger_returns_logger(self):
        from logging_config import get_logger

        logger = get_logger("TestComponent")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "TestComponent")

    def test_setup_logging_adds_handlers(self):
        from logging_config import setup_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir)
            root = logging.getLogger()
            handler_types = [type(h).__name__ for h in root.handlers]
            # Should have at least a StreamHandler and a RotatingFileHandler
            self.assertTrue(any("Stream" in h for h in handler_types), f"Expected StreamHandler, got: {handler_types}")

    def test_setup_logging_idempotent(self):
        from logging_config import setup_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir)
            count_before = len(logging.getLogger().handlers)
            setup_logging(log_dir=tmpdir)
            count_after = len(logging.getLogger().handlers)
            # Should not add duplicate handlers on second call
            self.assertEqual(count_before, count_after)

    def test_log_file_created(self):
        from logging_config import setup_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir)
            log_path = os.path.join(tmpdir, "evolution.log")
            try:
                # Write a test message
                logger = logging.getLogger("TestFileOutput")
                logger.info("test message")
                # Force flush
                for h in logging.getLogger().handlers:
                    h.flush()
                self.assertTrue(os.path.isfile(log_path))
            finally:
                # Close and detach any file handlers pointing into tmpdir so
                # Windows can delete the TemporaryDirectory on context exit.
                root = logging.getLogger()
                for h in list(root.handlers):
                    base = getattr(h, "baseFilename", None)
                    if base and os.path.dirname(base) == os.path.normpath(tmpdir):
                        h.close()
                        root.removeHandler(h)


# ══════════════════════════════════════════════════════════════════════
# Phase 3.3: Metrics Collector
# ══════════════════════════════════════════════════════════════════════


class TestMetricsCollector(unittest.TestCase):
    """Tests for metrics.py MetricsCollector class."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.jsonl_path = os.path.join(self._tmpdir, "test_cycles.jsonl")

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_import_metrics(self):
        from metrics import MetricsCollector

        self.assertTrue(hasattr(MetricsCollector, "record_cycle"))
        self.assertTrue(hasattr(MetricsCollector, "get_history"))
        self.assertTrue(hasattr(MetricsCollector, "get_trend"))

    def test_record_and_read(self):
        from metrics import MetricsCollector

        mc = MetricsCollector(self.jsonl_path)
        mc.record_cycle({"cycle": 1, "fitness": 0.5, "level": 2})
        mc.record_cycle({"cycle": 2, "fitness": 0.7, "level": 3})

        history = mc.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["cycle"], 1)
        self.assertAlmostEqual(history[0]["fitness"], 0.5)
        self.assertEqual(history[1]["cycle"], 2)

    def test_record_adds_timestamp(self):
        from metrics import MetricsCollector

        mc = MetricsCollector(self.jsonl_path)
        mc.record_cycle({"cycle": 1})
        history = mc.get_history()
        self.assertIn("timestamp", history[0])

    def test_get_history_last_n(self):
        from metrics import MetricsCollector

        mc = MetricsCollector(self.jsonl_path)
        for i in range(10):
            mc.record_cycle({"cycle": i + 1, "fitness": float(i)})
        last_3 = mc.get_history(last_n=3)
        self.assertEqual(len(last_3), 3)
        self.assertEqual(last_3[0]["cycle"], 8)
        self.assertEqual(last_3[2]["cycle"], 10)

    def test_get_history_empty(self):
        from metrics import MetricsCollector

        mc = MetricsCollector(self.jsonl_path)
        history = mc.get_history()
        self.assertEqual(history, [])

    def test_get_trend_empty(self):
        from metrics import MetricsCollector

        mc = MetricsCollector(self.jsonl_path)
        trend = mc.get_trend("fitness")
        self.assertEqual(trend["trend"], "unknown")
        self.assertIsNone(trend["current"])

    def test_get_trend_improving(self):
        from metrics import MetricsCollector

        mc = MetricsCollector(self.jsonl_path)
        for i in range(10):
            mc.record_cycle({"cycle": i + 1, "fitness": float(i) * 10.0})
        trend = mc.get_trend("fitness", window=10)
        self.assertEqual(trend["trend"], "improving")
        self.assertEqual(trend["current"], 90.0)
        self.assertGreater(trend["avg"], 0)

    def test_get_trend_stable(self):
        from metrics import MetricsCollector

        mc = MetricsCollector(self.jsonl_path)
        for i in range(10):
            mc.record_cycle({"cycle": i + 1, "fitness": 50.0})
        trend = mc.get_trend("fitness", window=10)
        self.assertEqual(trend["trend"], "stable")
        self.assertAlmostEqual(trend["avg"], 50.0)

    def test_count_records(self):
        from metrics import MetricsCollector

        mc = MetricsCollector(self.jsonl_path)
        self.assertEqual(mc.count_records(), 0)
        mc.record_cycle({"cycle": 1})
        mc.record_cycle({"cycle": 2})
        self.assertEqual(mc.count_records(), 2)

    def test_clear(self):
        from metrics import MetricsCollector

        mc = MetricsCollector(self.jsonl_path)
        mc.record_cycle({"cycle": 1})
        self.assertTrue(os.path.isfile(self.jsonl_path))
        mc.clear()
        self.assertFalse(os.path.isfile(self.jsonl_path))


# ══════════════════════════════════════════════════════════════════════
# Phase 6.1: Unified Configuration
# ══════════════════════════════════════════════════════════════════════


class TestConfig(unittest.TestCase):
    """Tests for config.py Config class."""

    def test_import_config(self):
        from config import add_config_args, load_config

        self.assertTrue(callable(load_config))
        self.assertTrue(callable(add_config_args))

    def test_config_defaults(self):
        from config import Config

        cfg = Config()
        self.assertEqual(cfg.evolution.population_size, 50)
        # max_cycles is null in engine_config.json (no-limit by default)
        self.assertIsNone(cfg.autonomous_runner.max_cycles)
        self.assertAlmostEqual(cfg.retrain.retrain_lr, 3e-5)

    def test_config_loads_json(self):
        from config import Config

        cfg = Config()  # loads config/engine_config.json
        # Verify it loaded the file (pop=50 matches the JSON)
        self.assertEqual(cfg.evolution.population_size, 50)
        self.assertEqual(cfg.retrain.retrain_model_size, "medium")

    def test_config_env_override(self):
        from config import Config

        os.environ["EE_POPULATION_SIZE"] = "123"
        try:
            cfg = Config()
            self.assertEqual(cfg.evolution.population_size, 123)
        finally:
            del os.environ["EE_POPULATION_SIZE"]

    def test_config_env_override_float(self):
        from config import Config

        os.environ["EE_RETRAIN_LR"] = "0.001"
        try:
            cfg = Config()
            self.assertAlmostEqual(cfg.retrain.retrain_lr, 0.001)
        finally:
            del os.environ["EE_RETRAIN_LR"]

    def test_config_env_override_bool(self):
        from config import Config

        os.environ["EE_USE_GPU"] = "true"
        try:
            cfg = Config()
            self.assertTrue(cfg.evolution.use_gpu)
        finally:
            del os.environ["EE_USE_GPU"]

    def test_config_validation_bad_population(self):
        from config import Config

        os.environ["EE_POPULATION_SIZE"] = "-5"
        try:
            with self.assertRaises(ValueError):
                Config()
        finally:
            del os.environ["EE_POPULATION_SIZE"]

    def test_config_validation_bad_lr(self):
        from config import Config

        os.environ["EE_RETRAIN_LR"] = "-1"
        try:
            with self.assertRaises(ValueError):
                Config()
        finally:
            del os.environ["EE_RETRAIN_LR"]

    def test_config_validation_bad_max_cycles(self):
        from config import Config

        os.environ["EE_MAX_CYCLES"] = "0"
        try:
            with self.assertRaises(ValueError):
                Config()
        finally:
            del os.environ["EE_MAX_CYCLES"]

    def test_config_to_dict(self):
        from config import Config

        cfg = Config()
        d = cfg.to_dict()
        self.assertIn("evolution", d)
        self.assertIn("retrain", d)
        self.assertIn("meta_controller", d)
        self.assertIn("autonomous_runner", d)
        self.assertEqual(d["evolution"]["population_size"], 50)

    def test_load_config_with_overrides(self):
        from config import load_config

        cfg = load_config(max_cycles=50, target_fitness=100.0)
        self.assertEqual(cfg.autonomous_runner.max_cycles, 50)
        self.assertAlmostEqual(cfg.autonomous_runner.target_fitness, 100.0)

    def test_add_config_args(self):
        import argparse

        from config import add_config_args

        parser = argparse.ArgumentParser()
        add_config_args(parser)
        args = parser.parse_args(["--max-cycles", "25", "--target-fitness", "50.0"])
        self.assertEqual(args.max_cycles, 25)
        self.assertAlmostEqual(args.target_fitness, 50.0)


# ══════════════════════════════════════════════════════════════════════
# Phase 6.2: AutonomousRunner Config Integration
# ══════════════════════════════════════════════════════════════════════


class TestAutonomousRunnerConfigIntegration(unittest.TestCase):
    """Tests that AutonomousRunner uses unified Config correctly."""

    def test_runner_uses_engine_config(self):
        from autonomous_runner import AutonomousRunner
        from config import Config

        ec = Config()
        runner = AutonomousRunner(engine_config=ec)
        self.assertEqual(runner._config["population_size"], 50)

    def test_runner_config_dict_backward_compat(self):
        from autonomous_runner import AutonomousRunner
        from config import Config

        ec = Config()
        runner = AutonomousRunner(engine_config=ec)
        # Dict-style access (backward compat) should work
        self.assertIn("max_cycles", runner._config)
        self.assertIn("target_fitness", runner._config)
        self.assertIn("population_size", runner._config)

    def test_runner_config_override_cli(self):
        from autonomous_runner import AutonomousRunner
        from config import Config

        ec = Config()
        runner = AutonomousRunner(engine_config=ec)
        runner._config["max_cycles"] = 5
        self.assertEqual(runner._config["max_cycles"], 5)

    def test_runner_config_target_override(self):
        from autonomous_runner import AutonomousRunner
        from config import Config

        ec = Config()
        runner = AutonomousRunner(engine_config=ec)
        runner._config["target_fitness"] = 100.0
        self.assertAlmostEqual(runner._config["target_fitness"], 100.0)

    def test_runner_retrain_config_from_engine(self):
        from autonomous_runner import AutonomousRunner
        from config import Config

        ec = Config()
        runner = AutonomousRunner(engine_config=ec)
        # Retrain config should come from EngineConfig
        self.assertAlmostEqual(runner._retrain_config.retrain_lr, 3e-5)
        self.assertEqual(runner._retrain_config.retrain_model_size, "medium")


if __name__ == "__main__":
    unittest.main()
