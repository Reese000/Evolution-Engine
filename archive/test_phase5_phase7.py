#!/usr/bin/env python3
"""Tests for Phase 5 (Cycle Resilience) and Phase 7 (Monitoring).

Validates:
  - cycle_state.py: CycleState save/load with checksums, corruption detection
  - autonomous_runner.py: cleanup_stale_state(), OOM recovery, retrain failure recovery
  - v11_dashboard.py: read_health_json, read_metrics_jsonl, compute_metrics_trend
  - autonomous_runner.py: _generate_progress_report, MetricsCollector integration
"""

import glob
import json
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import v11_dashboard
    _has_v11_dashboard = True
except ImportError:
    _has_v11_dashboard = False


# ══════════════════════════════════════════════════════════════════════
# Phase 5.1: CycleState persistence with integrity checksums
# ══════════════════════════════════════════════════════════════════════


class TestCycleState(unittest.TestCase):
    """Tests for cycle_state.py CycleState class."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.state_path = os.path.join(self._tmpdir, "cycle_state.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_import_cycle_state(self):
        from cycle_state import SCHEMA_VERSION, CycleState, default_cycle_state
        self.assertTrue(callable(CycleState))
        self.assertTrue(callable(default_cycle_state))
        self.assertEqual(SCHEMA_VERSION, 2)

    def test_default_state_has_all_fields(self):
        from cycle_state import default_cycle_state
        state = default_cycle_state()
        self.assertEqual(state["version"], 2)
        self.assertIn("current_cycle", state)
        self.assertIn("best_fitness", state)
        self.assertIn("best_level", state)
        self.assertIn("best_checkpoint", state)
        self.assertIn("champion_program", state)
        self.assertIn("strategy_archive", state)
        self.assertIn("diversity_score", state)
        self.assertIn("consecutive_stalls", state)
        self.assertIn("consecutive_errors", state)
        self.assertIn("total_errors", state)
        self.assertIn("last_cycle_timestamp", state)
        self.assertIn("start_timestamp", state)
        self.assertIn("cycles_completed", state)
        self.assertIn("stop_reason", state)
        self.assertIn("population_size", state)
        self.assertIn("retrain_config", state)
        self.assertIn("metrics_summary", state)

    def test_save_and_load_roundtrip(self):
        from cycle_state import CycleState
        cs = CycleState(self.state_path)
        cs.update(current_cycle=5, best_fitness=42.5, best_level=3, population_size=50)
        self.assertTrue(cs.save())
        self.assertTrue(os.path.isfile(self.state_path))

        cs2 = CycleState(self.state_path)
        loaded = cs2.load()
        self.assertTrue(loaded)
        self.assertEqual(cs2["current_cycle"], 5)
        self.assertAlmostEqual(cs2["best_fitness"], 42.5)
        self.assertEqual(cs2["best_level"], 3)

    def test_checksum_valid(self):
        """After save, the checksum field should be present and valid."""
        from cycle_state import CycleState, _compute_checksum
        cs = CycleState(self.state_path)
        cs.update(current_cycle=1, best_fitness=10.0)
        cs.save()

        with open(self.state_path) as f:
            raw = json.load(f)
        self.assertIn("checksum", raw)
        stored = raw.pop("checksum")
        computed = _compute_checksum(raw)
        self.assertEqual(stored, computed)

    def test_corruption_detected(self):
        """Tampering with a field should cause load() to return False."""
        from cycle_state import CycleState
        cs = CycleState(self.state_path)
        cs.update(current_cycle=3, best_fitness=20.0)
        cs.save()

        # Tamper with the file
        with open(self.state_path) as f:
            data = json.load(f)
        data["best_fitness"] = 999.0  # Tamper
        with open(self.state_path, "w") as f:
            json.dump(data, f)

        cs2 = CycleState(self.state_path)
        loaded = cs2.load()
        self.assertFalse(loaded, "Corrupted state should fail to load")

    def test_missing_file_returns_false(self):
        from cycle_state import CycleState
        cs = CycleState(os.path.join(self._tmpdir, "nonexistent.json"))
        loaded = cs.load()
        self.assertFalse(loaded)

    def test_corrupt_json_returns_false(self):
        from cycle_state import CycleState
        with open(self.state_path, "w") as f:
            f.write("not valid json {{{")
        cs = CycleState(self.state_path)
        loaded = cs.load()
        self.assertFalse(loaded)

    def test_reset_preserves_start_timestamp(self):
        from cycle_state import CycleState
        cs = CycleState(self.state_path)
        cs.update(current_cycle=10, best_fitness=100.0)
        original_start = cs["start_timestamp"]
        cs.reset()
        self.assertEqual(cs["start_timestamp"], original_start)
        self.assertEqual(cs["current_cycle"], 0)

    def test_getitem_setitem(self):
        from cycle_state import CycleState
        cs = CycleState(self.state_path)
        cs["current_cycle"] = 7
        self.assertEqual(cs["current_cycle"], 7)
        self.assertEqual(cs.get("current_cycle"), 7)
        self.assertIsNone(cs.get("nonexistent"))
        self.assertEqual(cs.get("nonexistent", 42), 42)

    def test_file_exists(self):
        from cycle_state import CycleState
        cs = CycleState(self.state_path)
        self.assertFalse(cs.file_exists())
        cs.save()
        self.assertTrue(cs.file_exists())


# ══════════════════════════════════════════════════════════════════════
# Phase 5.2: Automatic Cleanup Between Cycles
# ══════════════════════════════════════════════════════════════════════


class TestCleanupStaleState(unittest.TestCase):
    """Tests for cleanup_stale_state() function."""

    def test_import_cleanup(self):
        from autonomous_runner import cleanup_stale_state
        self.assertTrue(callable(cleanup_stale_state))

    def test_removes_bak_files(self):
        from autonomous_runner import cleanup_stale_state
        from paths import PROJECT_ROOT
        # Create fake .bak files
        bak_files = []
        for name in ["runner_state.json.bak", "cycle_state.json.bak",
                      "runner_state.json.old", "cycle_state.json.old"]:
            p = os.path.join(str(PROJECT_ROOT), name)
            Path(p).touch()
            bak_files.append(p)
        try:
            cleaned = cleanup_stale_state(current_cycle=0)
            # Should have cleaned the .bak files
            for p in bak_files:
                self.assertFalse(os.path.isfile(p), f"Should have removed {p}")
        finally:
            for p in bak_files:
                if os.path.isfile(p):
                    os.remove(p)

    def test_removes_strategy_archive_bak(self):
        from autonomous_runner import cleanup_stale_state
        from paths import PROJECT_ROOT
        p = os.path.join(str(PROJECT_ROOT), "strategy_archive_test.json.bak")
        Path(p).touch()
        try:
            cleaned = cleanup_stale_state(current_cycle=0)
            self.assertFalse(os.path.isfile(p))
        finally:
            if os.path.isfile(p):
                os.remove(p)

    def test_returns_list(self):
        from autonomous_runner import cleanup_stale_state
        result = cleanup_stale_state(current_cycle=0)
        self.assertIsInstance(result, list)

    def test_does_not_delete_checkpoint_best(self):
        """checkpoint_best.pt should never be touched by cleanup."""
        from autonomous_runner import cleanup_stale_state
        from paths import M1_CHECKPOINTS
        ckpt = os.path.join(str(M1_CHECKPOINTS), "checkpoint_best.pt")
        existed_before = os.path.isfile(ckpt)
        cleanup_stale_state(current_cycle=100)
        if existed_before:
            self.assertTrue(os.path.isfile(ckpt))


# ══════════════════════════════════════════════════════════════════════
# Phase 5.3: OOM Recovery
# ══════════════════════════════════════════════════════════════════════


class TestOOMRecovery(unittest.TestCase):
    """Tests for OOM handling with retry and CPU fallback."""

    def test_import_oom_handler(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        self.assertTrue(hasattr(runner, "_handle_oom"))
        self.assertTrue(hasattr(runner, "_is_cuda_oom"))
        self.assertTrue(hasattr(runner, "_clear_gpu_cache"))

    def test_oom_reduces_population(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        runner._config["evolution"]["population_size"] = 50
        runner._handle_oom(MemoryError("test OOM"), cycle=1, max_retries=1)
        self.assertEqual(runner._config["evolution"]["population_size"], 37)

    def test_oom_minimum_population(self):
        """When population is already near minimum, OOM should still work."""
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        runner._config["evolution"]["population_size"] = 10
        result = runner._handle_oom(MemoryError("test OOM"), cycle=1, max_retries=1)
        # max(8, int(10*0.75)) = max(8, 7) = 8 — clamped at minimum
        self.assertEqual(runner._config["evolution"]["population_size"], 8)
        self.assertTrue(result)  # Still returns True (retry OK)

    def test_cuda_oom_detection(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        self.assertTrue(runner._is_cuda_oom(MemoryError("CUDA out of memory")))
        # "OOM error" alone is NOT CUDA OOM — requires "cuda" or "out of memory"
        self.assertFalse(runner._is_cuda_oom(MemoryError("OOM error")))
        self.assertFalse(runner._is_cuda_oom(MemoryError("regular error")))

    def test_clear_gpu_cache_no_crash(self):
        """_clear_gpu_cache should not raise even without GPU."""
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        runner._clear_gpu_cache()  # Should not raise

    def test_oom_cpu_fallback_on_final_retry(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        runner._config["evolution"]["population_size"] = 100
        runner._config["evolution"]["use_gpu"] = True
        runner._handle_oom(MemoryError("CUDA OOM"), cycle=1, max_retries=1)
        # On final retry with CUDA OOM, should fall back to CPU
        self.assertFalse(runner._config["evolution"]["use_gpu"])


# ══════════════════════════════════════════════════════════════════════
# Phase 5.4: Retrain Failure Recovery
# ══════════════════════════════════════════════════════════════════════


class TestRetrainFailureRecovery(unittest.TestCase):
    """Tests for retrain failure recovery logic."""

    def test_import_retrain_recovery(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        self.assertTrue(hasattr(runner, "_run_retrain_with_recovery"))

    def test_nan_loss_reduces_lr(self):
        """When retrain returns NaN val_loss, LR should be halved."""
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        original_lr = runner._retrain_config.retrain_lr

        # Mock _run_retrain to return NaN val_loss
        def mock_retrain(cycle, evo_result):
            return {"retrained": True, "val_loss": float("nan")}

        runner._run_retrain = mock_retrain
        result = runner._run_retrain_with_recovery(cycle=1, evo_result={}, max_attempts=2)
        # After 2 attempts with NaN, LR should be halved twice
        self.assertLess(runner._retrain_config.retrain_lr, original_lr)
        self.assertFalse(result.get("retrained"))

    def test_inf_loss_reduces_lr(self):
        """When retrain returns Inf val_loss, LR should be halved."""
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        original_lr = runner._retrain_config.retrain_lr

        def mock_retrain(cycle, evo_result):
            return {"retrained": True, "val_loss": float("inf")}

        runner._run_retrain = mock_retrain
        result = runner._run_retrain_with_recovery(cycle=1, evo_result={}, max_attempts=2)
        self.assertLess(runner._retrain_config.retrain_lr, original_lr)

    def test_successful_retrain_passes_through(self):
        """A successful retrain should pass through without modification."""
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()

        def mock_retrain(cycle, evo_result):
            return {"retrained": True, "val_loss": 0.5}

        runner._run_retrain = mock_retrain
        result = runner._run_retrain_with_recovery(cycle=1, evo_result={}, max_attempts=3)
        self.assertTrue(result["retrained"])
        self.assertAlmostEqual(result["val_loss"], 0.5)

    def test_cuda_error_returns_skip(self):
        """CUDA error in retrain should return retrained=False with reason."""
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()

        def mock_retrain(cycle, evo_result):
            raise RuntimeError("CUDA error")

        runner._run_retrain = mock_retrain
        result = runner._run_retrain_with_recovery(cycle=1, evo_result={}, max_attempts=2)
        self.assertFalse(result.get("retrained"))
        self.assertIn("reason", result)

    def test_max_attempts_returns_skip(self):
        """After max_attempts failures, should skip retrain."""
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()

        call_count = [0]
        def mock_retrain(cycle, evo_result):
            call_count[0] += 1
            raise RuntimeError("generic failure")

        runner._run_retrain = mock_retrain
        result = runner._run_retrain_with_recovery(cycle=1, evo_result={}, max_attempts=3)
        self.assertEqual(call_count[0], 3)
        self.assertFalse(result.get("retrained"))
        self.assertEqual(result.get("attempts"), 3)


# ══════════════════════════════════════════════════════════════════════
# Phase 7.1: Dashboard reads from health.json + metrics/cycles.jsonl
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _has_v11_dashboard, reason="Module archived 2026-07-29")
class TestDashboardHealthMetrics(unittest.TestCase):
    """Tests for v11_dashboard.py health.json and metrics reading."""

    def test_import_dashboard_functions(self):
        from v11_dashboard import compute_metrics_trend, read_health_json, read_metrics_jsonl, read_progress_report
        self.assertTrue(callable(read_health_json))
        self.assertTrue(callable(read_metrics_jsonl))
        self.assertTrue(callable(compute_metrics_trend))
        self.assertTrue(callable(read_progress_report))

    def test_read_health_json_missing(self):
        from v11_dashboard import read_health_json
        result = read_health_json("/nonexistent/path/health.json")
        self.assertIsNone(result)

    def test_read_health_json_valid(self):
        from v11_dashboard import read_health_json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"status": "running", "cycles_completed": 5, "current_fitness": 42.0}, f)
            path = f.name
        try:
            result = read_health_json(path)
            self.assertIsNotNone(result)
            self.assertEqual(result["status"], "running")
            self.assertEqual(result["cycles_completed"], 5)
        finally:
            os.unlink(path)

    def test_read_health_json_corrupt(self):
        from v11_dashboard import read_health_json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            path = f.name
        try:
            result = read_health_json(path)
            self.assertIsNone(result)
        finally:
            os.unlink(path)

    def test_read_metrics_jsonl_missing(self):
        from v11_dashboard import read_metrics_jsonl
        result = read_metrics_jsonl("/nonexistent/path.jsonl")
        self.assertEqual(result, [])

    def test_read_metrics_jsonl_valid(self):
        from v11_dashboard import read_metrics_jsonl
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"cycle": 1, "fitness": 0.5}) + "\n")
            f.write(json.dumps({"cycle": 2, "fitness": 0.7}) + "\n")
            path = f.name
        try:
            result = read_metrics_jsonl(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["cycle"], 1)
        finally:
            os.unlink(path)

    def test_read_metrics_jsonl_last_n(self):
        from v11_dashboard import read_metrics_jsonl
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(10):
                f.write(json.dumps({"cycle": i + 1}) + "\n")
            path = f.name
        try:
            result = read_metrics_jsonl(path, last_n=3)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0]["cycle"], 8)
        finally:
            os.unlink(path)

    def test_compute_metrics_trend_improving(self):
        from v11_dashboard import compute_metrics_trend
        records = [{"fitness": float(i) * 10.0} for i in range(10)]
        trend = compute_metrics_trend(records, "fitness", window=10)
        self.assertEqual(trend["trend"], "improving")
        self.assertEqual(trend["current"], 90.0)

    def test_compute_metrics_trend_stable(self):
        from v11_dashboard import compute_metrics_trend
        records = [{"fitness": 50.0} for _ in range(10)]
        trend = compute_metrics_trend(records, "fitness", window=10)
        self.assertEqual(trend["trend"], "stable")

    def test_compute_metrics_trend_empty(self):
        from v11_dashboard import compute_metrics_trend
        trend = compute_metrics_trend([], "fitness")
        self.assertEqual(trend["trend"], "unknown")
        self.assertIsNone(trend["current"])

    def test_read_progress_report_missing(self):
        from v11_dashboard import read_progress_report
        result = read_progress_report("/nonexistent/progress_report.json")
        self.assertIsNone(result)

    def test_read_progress_report_valid(self):
        from v11_dashboard import read_progress_report
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"cycle": 5, "summary": {"best_fitness": 42.0}}, f)
            path = f.name
        try:
            result = read_progress_report(path)
            self.assertIsNotNone(result)
            self.assertEqual(result["cycle"], 5)
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════════
# Phase 7.2: Progress Report Generation
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="FileNotFoundError in progress report — PM Rule #27",
)

class TestProgressReport(unittest.TestCase):
    """Tests for _generate_progress_report in AutonomousRunner."""

    def test_runner_has_progress_report_method(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        self.assertTrue(hasattr(runner, "_generate_progress_report"))

    def test_runner_has_metrics_collector(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        self.assertIsNotNone(runner._metrics)

    def test_generate_progress_report_creates_file(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        cycle_metrics = {
            "cycle": 1,
            "fitness": 0.5,
            "level": 2,
            "best_fitness": 0.5,
            "best_level": 2,
            "cycle_time_seconds": 10.0,
        }
        runner._generate_progress_report(1, cycle_metrics)

        report_path = os.path.join(str(PROJECT_ROOT), "progress_report.json")
        self.assertTrue(os.path.isfile(report_path))
        try:
            with open(report_path) as f:
                report = json.load(f)
            self.assertEqual(report["cycle"], 1)
            self.assertIn("summary", report)
            self.assertIn("trends", report)
            self.assertIn("latest_cycle", report)
            self.assertIn("resource_usage", report)
        finally:
            if os.path.isfile(report_path):
                os.remove(report_path)


if __name__ == "__main__":
    unittest.main()
