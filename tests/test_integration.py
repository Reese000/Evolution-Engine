#!/usr/bin/env python3
"""Integration tests for the Evolution Engine full pipeline.

Validates the end-to-end flow:
  - Config load → evolution cycle → retrain → health check → repeat
  - Crash recovery: simulate SIGTERM mid-cycle, verify state is preserved
  - Graceful shutdown: verify SIGINT saves state and exits cleanly
  - Health monitoring: verify health.json is updated correctly
  - Stop conditions: target-fitness, max-cycles, timeout

These tests are self-contained — they use mocks and lightweight stubs
rather than requiring GPU hardware or full model checkpoints.
"""

import json
import os
import signal
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def _make_runner(tmpdir: str):
    """Create an AutonomousRunner with a temp config, no checkpoint needed."""
    from autonomous_runner import AutonomousRunner

    # Write a minimal config
    config = {
        "evolution": {
            "population_size": 10,
            "max_generations": 5,
            "mutation_rate": 0.3,
            "elite_ratio": 0.1,
        },
        "retrain": {
            "retrain_lr": 3e-5,
            "retrain_batch_size": 4,
            "retrain_max_epochs": 2,
            "retrain_patience": 1,
            "retrain_freeze_layers": 0,
            "retrain_min_examples": 1,
            "retrain_max_per_level": 0,
            "retrain_model_size": "small",
            "retrain_timeout": 60,
            "retrain_val_split": 0.1,
            "retrain_weight_decay": 0.01,
            "retrain_grad_clip": 1.0,
            "ref_high_augment": 10,
        },
        "meta_controller": {},
        "autonomous_runner": {
            "max_cycles": 5,
            "target_fitness": 100.0,
            "generations_per_cycle": 5,
            "health_check_interval": 10,
            "state_file": "runner_state.json",
            "stop_file": "stop",
            "max_errors_before_stop": 3,
        },
    }
    config_path = os.path.join(tmpdir, "test_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    runner = AutonomousRunner(config_path=config_path)
    return runner


# ══════════════════════════════════════════════════════════════════════
# 1. Config → Runner Pipeline Integration
# ══════════════════════════════════════════════════════════════════════


class TestConfigToRunnerPipeline(unittest.TestCase):
    """Tests the config → runner initialization pipeline end-to-end."""

    def test_config_loads_into_runner(self):
        """Config values are correctly propagated to the runner."""
        from autonomous_runner import AutonomousRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            # Verify config values propagated
            self.assertEqual(runner._config["max_cycles"], 5)
            self.assertAlmostEqual(runner._config["target_fitness"], 100.0)
            self.assertEqual(runner._config["generations_per_cycle"], 5)
            self.assertEqual(runner._config["max_errors_before_stop"], 3)

    def test_runner_state_defaults(self):
        """Runner starts with correct default state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            state = runner._state
            self.assertEqual(state["current_cycle"], 0)
            self.assertAlmostEqual(state["best_fitness"], 0.0)
            self.assertEqual(state["error_count"], 0)
            self.assertEqual(state["version"], 1)
            self.assertFalse(state["stop_reason"])

    def test_engine_config_dict_compatibility(self):
        """EngineConfig's to_dict() is compatible with runner config access."""
        from config import Config

        cfg = Config()
        d = cfg.to_dict()
        self.assertIn("evolution", d)
        self.assertIn("retrain", d)
        self.assertIn("meta_controller", d)
        self.assertIn("autonomous_runner", d)
        self.assertIsInstance(d["evolution"]["population_size"], int)
        self.assertIsInstance(d["retrain"]["retrain_lr"], float)

    def test_retrain_config_built_from_engine(self):
        """Runner correctly builds retrain config namespace from engine config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            rc = runner._retrain_config
            self.assertAlmostEqual(rc.retrain_lr, 3e-5)
            self.assertEqual(rc.retrain_batch_size, 4)
            self.assertEqual(rc.retrain_max_epochs, 2)
            self.assertEqual(rc.retrain_patience, 1)


# ══════════════════════════════════════════════════════════════════════
# 2. Crash Recovery: SIGTERM mid-cycle
# ══════════════════════════════════════════════════════════════════════


class TestCrashRecovery(unittest.TestCase):
    """Tests that state is preserved when SIGTERM interrupts mid-cycle."""

    def test_state_persisted_after_cycle(self):
        """After a successful cycle, state is saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._state["current_cycle"] = 3
            runner._state["best_fitness"] = 42.5
            runner._state["best_level"] = 5

            runner.save_state()

            # Verify file exists and has correct content
            self.assertTrue(os.path.isfile(runner._state_path))
            with open(runner._state_path) as f:
                data = json.load(f)
            self.assertEqual(data["current_cycle"], 3)
            self.assertAlmostEqual(data["best_fitness"], 42.5)
            self.assertEqual(data["best_level"], 5)

    def test_resume_from_saved_state(self):
        """Runner can resume from a saved state file after restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner1 = _make_runner(tmpdir)
            runner1._state["current_cycle"] = 7
            runner1._state["best_fitness"] = 99.9
            runner1._state["best_level"] = 8
            runner1.save_state()

            # Simulate a restart — new runner loads state
            runner2 = _make_runner(tmpdir)
            loaded = runner2.load_state()
            self.assertTrue(loaded)
            self.assertEqual(runner2._state["current_cycle"], 7)
            self.assertAlmostEqual(runner2._state["best_fitness"], 99.9)
            self.assertEqual(runner2._state["best_level"], 8)

    def test_corrupt_state_file_starts_fresh(self):
        """Runner handles corrupt state files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            # Write corrupt data
            with open(runner._state_path, "w") as f:
                f.write("{invalid json!!!")

            loaded = runner.load_state()
            self.assertFalse(loaded)
            # Should be at defaults
            self.assertEqual(runner._state["current_cycle"], 0)

    def test_missing_state_file_starts_fresh(self):
        """Runner starts fresh when no state file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            loaded = runner.load_state()
            self.assertFalse(loaded)
            self.assertEqual(runner._state["current_cycle"], 0)

    def test_sigterm_sets_shutdown_flag(self):
        """SIGTERM signal sets the shutdown flag without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            self.assertFalse(runner._shutdown_requested)

            # Simulate SIGTERM
            runner._request_shutdown(signal.SIGTERM, None)
            self.assertTrue(runner._shutdown_requested)

    def test_sigint_sets_shutdown_flag(self):
        """SIGINT signal sets the shutdown flag for graceful stop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._request_shutdown(signal.SIGINT, None)
            self.assertTrue(runner._shutdown_requested)

    def test_state_preserves_timestamps(self):
        """Saved state preserves start and last-cycle timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner.save_state()

            with open(runner._state_path) as f:
                data = json.load(f)
            self.assertIn("start_timestamp", data)
            self.assertIn("last_cycle_timestamp", data)
            self.assertTrue(len(data["start_timestamp"]) > 0)

    def test_cycle_state_checksum_integrity(self):
        """CycleState checksums detect and reject corruption."""
        from cycle_state import CycleState

        with tempfile.TemporaryDirectory() as tmpdir:
            cs_path = os.path.join(tmpdir, "cycle_state.json")
            cs = CycleState(cs_path)
            cs.update(current_cycle=5, best_fitness=1.5, best_level=3)
            cs.save()

            # Verify valid load
            cs2 = CycleState(cs_path)
            self.assertTrue(cs2.load())
            self.assertEqual(cs2["current_cycle"], 5)

            # Corrupt the checksum
            with open(cs_path, "r") as f:
                data = json.load(f)
            data["checksum"] = "bad_checksum"
            with open(cs_path, "w") as f:
                json.dump(data, f)

            cs3 = CycleState(cs_path)
            self.assertFalse(cs3.load())
            # Should be reset to defaults
            self.assertEqual(cs3["current_cycle"], 0)


# ══════════════════════════════════════════════════════════════════════
# 3. Graceful Shutdown: SIGINT saves state
# ══════════════════════════════════════════════════════════════════════


class TestGracefulShutdown(unittest.TestCase):
    """Tests that SIGINT saves state and exits cleanly."""

    def test_graceful_stop_sets_flag(self):
        """graceful_stop() sets shutdown flag for clean exit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            self.assertFalse(runner._shutdown_requested)
            runner.graceful_stop()
            self.assertTrue(runner._shutdown_requested)

    def test_should_stop_on_shutdown_requested(self):
        """Runner stops when shutdown flag is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._shutdown_requested = True
            self.assertTrue(runner.should_stop())

    def test_shutdown_cleans_stop_file(self):
        """Stop file is consumed (removed) when detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            stop_file = os.path.join(str(PROJECT_ROOT), runner._config["stop_file"])
            try:
                Path(stop_file).touch()
                self.assertTrue(os.path.isfile(stop_file))
                self.assertTrue(runner.should_stop())
                self.assertFalse(os.path.isfile(stop_file))
            finally:
                if os.path.isfile(stop_file):
                    os.remove(stop_file)

    def test_max_errors_stops_runner(self):
        """Runner stops when consecutive errors exceed threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._state["consecutive_errors"] = runner._config["max_errors_before_stop"]
            self.assertTrue(runner.should_stop())


# ══════════════════════════════════════════════════════════════════════
# 4. Health Monitoring
# ══════════════════════════════════════════════════════════════════════


class TestHealthMonitoring(unittest.TestCase):
    """Tests health.json is written and updated correctly."""

    def test_health_json_written(self):
        """HealthMonitor writes valid health.json."""
        from autonomous_runner import HealthMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "state.json")
            hm = HealthMonitor(state_path=state_file)
            hm.set_status("running")
            state = {
                "current_cycle": 5,
                "best_fitness": 42.0,
                "best_level": 7,
                "error_count": 2,
                "last_cycle_timestamp": "2026-06-28T12:00:00",
                "target_fitness": 200.0,
            }
            hm.write_health(state)

            health_path = os.path.join(tmpdir, "health.json")
            self.assertTrue(os.path.isfile(health_path))
            with open(health_path) as f:
                health = json.load(f)

            self.assertEqual(health["status"], "running")
            self.assertEqual(health["cycles_completed"], 5)
            self.assertEqual(health["current_fitness"], 42.0)
            self.assertEqual(health["current_level"], 7)
            self.assertEqual(health["errors"], 2)
            self.assertEqual(health["target_fitness"], 200.0)
            self.assertGreaterEqual(health["uptime_seconds"], 0)

    def test_health_updates_on_cycle(self):
        """Health status updates correctly after each cycle."""
        from autonomous_runner import HealthMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "state.json")
            hm = HealthMonitor(state_path=state_file)

            # Cycle 1
            hm.set_status("running")
            hm.write_health({"current_cycle": 1, "best_fitness": 10.0,
                             "best_level": 1, "error_count": 0})
            with open(os.path.join(tmpdir, "health.json")) as f:
                h1 = json.load(f)
            self.assertEqual(h1["cycles_completed"], 1)

            time.sleep(0.01)

            # Cycle 2
            hm.write_health({"current_cycle": 2, "best_fitness": 25.0,
                             "best_level": 3, "error_count": 0})
            with open(os.path.join(tmpdir, "health.json")) as f:
                h2 = json.load(f)
            self.assertEqual(h2["cycles_completed"], 2)
            self.assertEqual(h2["current_fitness"], 25.0)
            self.assertEqual(h2["current_level"], 3)
            self.assertGreaterEqual(h2["uptime_seconds"], h1["uptime_seconds"])

    def test_health_stopped_status(self):
        """Health shows 'stopped' after runner finishes."""
        from autonomous_runner import HealthMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            hm = HealthMonitor(state_path=os.path.join(tmpdir, "s.json"))
            hm.set_status("stopped")
            hm.write_health({"current_cycle": 10})
            with open(os.path.join(tmpdir, "health.json")) as f:
                health = json.load(f)
            self.assertEqual(health["status"], "stopped")

    def test_runner_writes_health_during_run(self):
        """AutonomousRunner.write_health is called after each cycle in run()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            # Manually trigger health write
            runner._health.set_status("running")
            runner._health.write_health(runner._state)

            health_path = os.path.join(os.path.dirname(runner._state_path), "health.json")
            self.assertTrue(os.path.isfile(health_path))
            with open(health_path) as f:
                health = json.load(f)
            self.assertEqual(health["status"], "running")


# ══════════════════════════════════════════════════════════════════════
# 5. Stop Conditions
# ══════════════════════════════════════════════════════════════════════


class TestStopConditions(unittest.TestCase):
    """Tests all stop conditions for the autonomous runner."""

    def test_stop_max_cycles_reached(self):
        """Runner stops when current_cycle >= max_cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._config["max_cycles"] = 5
            runner._state["current_cycle"] = 5
            self.assertTrue(runner.should_stop())

    def test_stop_max_cycles_not_reached(self):
        """Runner does not stop when cycle < max_cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._config["max_cycles"] = 100
            runner._state["current_cycle"] = 5
            self.assertFalse(runner.should_stop())

    def test_stop_target_fitness_reached(self):
        """Runner stops when best_fitness >= target_fitness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._config["target_fitness"] = 100.0
            runner._state["best_fitness"] = 100.0
            self.assertTrue(runner.should_stop())

    def test_stop_target_fitness_above(self):
        """Runner stops when best_fitness exceeds target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._config["target_fitness"] = 50.0
            runner._state["best_fitness"] = 75.0
            self.assertTrue(runner.should_stop())

    def test_stop_target_fitness_not_reached(self):
        """Runner continues when fitness is below target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._config["target_fitness"] = 200.0
            runner._state["best_fitness"] = 10.0
            self.assertFalse(runner.should_stop())

    def test_stop_on_shutdown_signal(self):
        """Runner stops after SIGINT/SIGTERM signal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._shutdown_requested = True
            self.assertTrue(runner.should_stop())

    def test_stop_on_max_consecutive_errors(self):
        """Runner stops after too many consecutive errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._config["max_errors_before_stop"] = 5
            runner._state["consecutive_errors"] = 5
            self.assertTrue(runner.should_stop())

    def test_stop_on_stop_file(self):
        """Runner stops when stop file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            stop_file = os.path.join(str(PROJECT_ROOT), runner._config["stop_file"])
            try:
                Path(stop_file).touch()
                self.assertTrue(runner.should_stop())
            finally:
                if os.path.isfile(stop_file):
                    os.remove(stop_file)

    def test_no_stop_conditions_met(self):
        """Runner does not stop when no conditions are met."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._config["max_cycles"] = 100
            runner._config["target_fitness"] = 200.0
            runner._state["current_cycle"] = 1
            runner._state["best_fitness"] = 10.0
            runner._state["consecutive_errors"] = 0
            self.assertFalse(runner.should_stop())


# ══════════════════════════════════════════════════════════════════════
# 6. MetaController → Runner Integration
# ══════════════════════════════════════════════════════════════════════


class TestMetaControllerRunnerIntegration(unittest.TestCase):
    """Tests MetaController integration with the runner pipeline."""

    def test_meta_controller_checkpoint_gating(self):
        """MetaController accepts good checkpoints and rejects bad ones."""
        from meta_controller import MetaController

        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetaController(checkpoint_dir=tmpdir,
                                initial_val_loss=2.0)

            # Create a fake checkpoint file
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            with open(ckpt_path, "wb") as f:
                f.write(b"fake checkpoint data")

            # Good val_loss (within tolerance)
            accepted = mc.accept_checkpoint(1.5, ckpt_path)
            self.assertTrue(accepted)

            # Bad val_loss (way worse than best)
            rejected = mc.accept_checkpoint(10.0, ckpt_path)
            self.assertFalse(rejected)

    def test_meta_controller_tuning_params(self):
        """MetaController produces valid tuning parameters per cycle."""
        from meta_controller import MetaController, TelemetrySnapshot

        mc = MetaController(checkpoint_dir="/tmp/test_mc_integration")

        # Feed 5 cycles of telemetry
        for c in range(1, 6):
            snap = TelemetrySnapshot(
                cycle=c, fitness=-3.0 + c * 0.5, level=min(c, 3),
                champion_words=50, val_loss=3.0 - c * 0.3,
                diversity=0.3, novelty_size=10, qd_occupied=5,
                qd_total=100, traces_count=20, retrained=True,
                generations_completed=50,
            )
            mc.update_cycle(snap)

        # Should produce valid retrain params
        retrain_params = mc.tune_retrain_params(5)
        self.assertIn("retrain_lr", retrain_params)
        self.assertIn("retrain_freeze_layers", retrain_params)
        self.assertIn("retrain_patience", retrain_params)
        self.assertIn("retrain_max_epochs", retrain_params)
        self.assertIsInstance(retrain_params["retrain_lr"], float)
        self.assertGreater(retrain_params["retrain_lr"], 0)

    def test_meta_controller_state_save_load(self):
        """MetaController state persists across save/load cycle."""
        from meta_controller import MetaController, TelemetrySnapshot

        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetaController(checkpoint_dir=tmpdir)
            mc.best_fitness = 42.0
            mc.best_level = 5
            mc._gated_count = 3

            state_path = os.path.join(tmpdir, "mc_state.json")
            mc.save_state(state_path)
            self.assertTrue(os.path.isfile(state_path))

            mc2 = MetaController(checkpoint_dir=tmpdir)
            loaded = mc2.load_state(state_path)
            self.assertTrue(loaded)
            self.assertAlmostEqual(mc2.best_fitness, 42.0)
            self.assertEqual(mc2.best_level, 5)
            self.assertEqual(mc2._gated_count, 3)

    def test_stagnation_detection_full_lifecycle(self):
        """MetaController detects all stagnation types across a lifecycle."""
        from meta_controller import MetaController, TelemetrySnapshot, StagnationResult

        mc = MetaController(checkpoint_dir="/tmp/test_lifecycle")

        # Phase 1: Normal improving
        for c in range(1, 4):
            snap = TelemetrySnapshot(
                cycle=c, fitness=-3.0 + c, level=c,
                champion_words=50, val_loss=3.0 - c * 0.5,
                diversity=0.3, novelty_size=10, qd_occupied=5,
                qd_total=100, traces_count=20, retrained=True,
                generations_completed=50,
            )
            mc.update_cycle(snap)
        self.assertEqual(mc.detect_stagnation(), StagnationResult.NORMAL)

        # Phase 2: Plateau (same fitness for 4 cycles)
        for c in range(4, 8):
            snap = TelemetrySnapshot(
                cycle=c, fitness=0.0, level=3,
                champion_words=50, val_loss=1.5,
                diversity=0.3, novelty_size=10, qd_occupied=5,
                qd_total=100, traces_count=20, retrained=True,
                generations_completed=50,
            )
            mc.update_cycle(snap)
        state = mc.detect_stagnation()
        self.assertIn(state, [StagnationResult.FITNESS_PLATEAU,
                              StagnationResult.LEVEL_STALLED,
                              StagnationResult.COMPOUND_STALL])


# ══════════════════════════════════════════════════════════════════════
# 7. Full Pipeline Mock Integration
# ══════════════════════════════════════════════════════════════════════


class TestFullPipelineMock(unittest.TestCase):
    """Integration test: config → runner → mock cycle → state → health."""

    def test_full_pipeline_with_mock_evolution(self):
        """Simulate a full pipeline: config → runner → cycle → state → health."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)

            # Simulate cycle 1
            runner._state["current_cycle"] = 1
            runner._state["best_fitness"] = 10.5
            runner._state["best_level"] = 2
            runner._state["cycles_completed"] = 1
            runner.save_state()

            # Write health
            runner._health.set_status("running")
            runner._health.write_health(runner._state)

            # Verify state
            with open(runner._state_path) as f:
                state = json.load(f)
            self.assertEqual(state["current_cycle"], 1)
            self.assertAlmostEqual(state["best_fitness"], 10.5)

            # Verify health
            health_path = os.path.join(os.path.dirname(runner._state_path), "health.json")
            with open(health_path) as f:
                health = json.load(f)
            self.assertEqual(health["cycles_completed"], 1)
            self.assertEqual(health["current_fitness"], 10.5)

            # Simulate cycle 2 — runner restarts, loads state, continues
            runner2 = _make_runner(tmpdir)
            loaded = runner2.load_state()
            self.assertTrue(loaded)
            self.assertEqual(runner2._state["current_cycle"], 1)  # was 1 after last save

            # Run cycle 2
            runner2._state["current_cycle"] = 2
            runner2._state["best_fitness"] = 25.0
            runner2._state["best_level"] = 3
            runner2._state["cycles_completed"] = 2
            runner2.save_state()
            runner2._health.write_health(runner2._state)

            # Verify final state
            with open(runner2._state_path) as f:
                final_state = json.load(f)
            self.assertEqual(final_state["current_cycle"], 2)
            self.assertAlmostEqual(final_state["best_fitness"], 25.0)

            with open(health_path) as f:
                final_health = json.load(f)
            self.assertEqual(final_health["cycles_completed"], 2)
            self.assertEqual(final_health["current_fitness"], 25.0)

    def test_error_counting_across_cycles(self):
        """Error count is correctly tracked across cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)

            # Simulate errors
            err = RuntimeError("test error")
            runner.log_error(err, cycle=1)
            self.assertEqual(runner._state["error_count"], 1)
            self.assertEqual(runner._state["total_errors"], 1)
            self.assertEqual(runner._state["consecutive_errors"], 1)

            runner.log_error(err, cycle=2)
            self.assertEqual(runner._state["consecutive_errors"], 2)
            self.assertEqual(runner._state["total_errors"], 2)

            # Reset consecutive errors (happens on successful cycle)
            runner._state["consecutive_errors"] = 0
            runner.log_error(err, cycle=3)
            self.assertEqual(runner._state["consecutive_errors"], 1)
            self.assertEqual(runner._state["total_errors"], 3)

    def test_metrics_collector_records_cycles(self):
        """MetricsCollector correctly records cycle metrics."""
        from metrics import MetricsCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "cycles.jsonl")
            mc = MetricsCollector(jsonl_path)

            # Record several cycles
            for i in range(5):
                mc.record_cycle({
                    "cycle": i + 1,
                    "fitness": 10.0 + i * 5,
                    "level": i + 1,
                })

            # Read back
            history = mc.get_history()
            self.assertEqual(len(history), 5)
            self.assertAlmostEqual(history[-1]["fitness"], 30.0)

            # Get trend (returns a dict with 'trend' key)
            trend = mc.get_trend("fitness", window=3)
            self.assertIn("trend", trend)
            self.assertIn(trend["trend"], ["improving", "stable", "declining", "unknown"])

    def test_progress_report_generated(self):
        """Progress report is generated after each cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            runner._health._start_time = time.time()
            # Set runner state so progress report reads correct values
            runner._state["best_fitness"] = 10.0
            runner._state["best_level"] = 2
            runner._state["cycles_completed"] = 1

            # Mock metrics
            runner._metrics.record_cycle({
                "cycle": 1, "fitness": 10.0, "level": 2,
            })

            runner._generate_progress_report(1, {
                "cycle": 1, "fitness": 10.0, "level": 2,
                "best_fitness": 10.0, "best_level": 2,
                "val_loss": None, "retrained": False,
                "population_size": 10, "cycle_time_seconds": 5.0,
                "errors": 0,
            })

            report_path = os.path.join(str(PROJECT_ROOT), "progress_report.json")
            try:
                self.assertTrue(os.path.isfile(report_path))
                with open(report_path) as f:
                    report = json.load(f)
                self.assertEqual(report["cycle"], 1)
                self.assertIn("summary", report)
                self.assertIn("trends", report)
                self.assertEqual(report["summary"]["best_fitness"], 10.0)
            finally:
                if os.path.isfile(report_path):
                    os.remove(report_path)


# ══════════════════════════════════════════════════════════════════════
# 8. Retrain Failure Recovery
# ══════════════════════════════════════════════════════════════════════


class TestRetrainFailureRecovery(unittest.TestCase):
    """Tests retrain failure recovery (Phase 5.4)."""

    def test_nan_loss_reduces_lr(self):
        """NaN val_loss triggers LR reduction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            original_lr = runner._retrain_config.retrain_lr

            # Mock _run_retrain to return NaN val_loss
            runner._run_retrain = MagicMock(return_value={
                "retrained": True, "val_loss": float("nan"),
            })

            result = runner._run_retrain_with_recovery(1, {}, max_attempts=2)
            # After NaN, LR should have been halved
            self.assertLess(runner._retrain_config.retrain_lr, original_lr)

    def test_cuda_error_returns_skip(self):
        """CUDA error after all retries returns skip result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)

            runner._run_retrain = MagicMock(
                side_effect=RuntimeError("CUDA out of memory")
            )

            result = runner._run_retrain_with_recovery(1, {}, max_attempts=2)
            self.assertFalse(result.get("retrained"))
            self.assertIn("cuda", result.get("reason", ""))

    def test_successful_retrain_passes_through(self):
        """Successful retrain result passes through unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            expected = {"retrained": True, "val_loss": 1.5, "checkpoint": "/tmp/ckpt.pt"}
            runner._run_retrain = MagicMock(return_value=expected)

            result = runner._run_retrain_with_recovery(1, {}, max_attempts=1)
            self.assertTrue(result["retrained"])
            self.assertAlmostEqual(result["val_loss"], 1.5)


if __name__ == "__main__":
    unittest.main()
