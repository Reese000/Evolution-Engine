#!/usr/bin/env python3
"""AutonomousRunner — Continuous evolution/retrain daemon.

Runs the evolution→retrain loop indefinitely until:
  - max_cycles reached
  - target_fitness achieved
  - stop file created (touch stop)
  - manual SIGINT/SIGTERM

Features:
  - Crash recovery: loads saved state on restart, resumes from last cycle
  - Graceful shutdown: completes current cycle before stopping
  - Health monitoring: writes health.json for external monitoring
  - Retrain integration: uses MetaController for checkpoint gating
  - OOM recovery: reduces population on GPU out-of-memory

Usage:
    python autonomous_runner.py                       # Run with defaults
    python autonomous_runner.py --max-cycles 10       # Override max cycles
    python autonomous_runner.py --target-fitness 200   # Stop at target
    python autonomous_runner.py --resume               # Resume from saved state
    python autonomous_runner.py --dry-run              # Validate config only
    python autonomous_runner.py --status               # Show health status
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Path setup ─────────────────────────────────────────────────────────

from paths import (
    PROJECT_ROOT,
    M1_CHECKPOINTS,
    M1_RECURRENT,
    MILESTONE1_DIR,
    RESULTS_FILE,
    META_CONTROLLER_STATE,
    ENGINE_CONFIG,
    ensure_project_on_path,
)

ensure_project_on_path()

# ── Logging ────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [Runner] {msg}", flush=True)


# ── Health Status ──────────────────────────────────────────────────────


class HealthMonitor:
    """Writes health.json for external monitoring tools."""

    def __init__(self, state_path: str = "runner_state.json"):
        self._state_path = state_path
        self._health_path = os.path.join(os.path.dirname(state_path) or ".", "health.json")
        self._start_time = time.time()
        self._status = "idle"

    def set_status(self, status: str) -> None:
        self._status = status

    def write_health(self, state: dict[str, Any]) -> None:
        health = {
            "status": self._status,
            "last_cycle": state.get("last_cycle_timestamp", ""),
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "cycles_completed": state.get("current_cycle", 0),
            "current_fitness": state.get("best_fitness", 0.0),
            "current_level": state.get("best_level", 0),
            "errors": state.get("error_count", 0),
            "target_fitness": state.get("target_fitness", None),
        }
        os.makedirs(os.path.dirname(self._health_path) or ".", exist_ok=True)
        with open(self._health_path, "w") as f:
            json.dump(health, f, indent=2)

    def show_status(self) -> None:
        """Print current health status from disk."""
        if not os.path.isfile(self._health_path):
            print("No health.json found. Runner has not been started.")
            return
        with open(self._health_path) as f:
            health = json.load(f)
        print("=" * 60)
        print("Evolution Engine — Health Status")
        print("=" * 60)
        for k, v in health.items():
            print(f"  {k:25s}: {v}")
        print("=" * 60)


# ── AutonomousRunner ───────────────────────────────────────────────────


class AutonomousRunner:
    """Runs the evolution/retrain loop indefinitely.

    State is persisted to runner_state.json after every cycle so the
    system can resume after crashes or restarts.
    """

    def __init__(self, config_path: str = "config/engine_config.json"):
        self._shutdown_requested = False
        self._health = HealthMonitor()

        # Load config
        self._config = self._load_config(config_path)

        # Runner state
        self._state = self._default_state()

        # Retrain config (set by load_state or defaults)
        self._retrain_config = self._build_retrain_config()

        # Resolve paths
        self._checkpoint = self._find_starting_checkpoint()
        self._results_file = str(RESULTS_FILE)

    # ── Config loading ─────────────────────────────────────────────

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load config from JSON file, falling back to defaults."""
        defaults = {
            "max_cycles": 100,
            "target_fitness": 200.0,
            "generations_per_cycle": 100,
            "health_check_interval": 60,
            "state_file": "runner_state.json",
            "stop_file": "stop",
            "max_errors_before_stop": 10,
            "evolution": {
                "population_size": 50,
                "auto_curriculum": True,
                "use_gpu": True,
            },
            "retrain": {
                "retrain_model_size": "xlarge",
            },
        }
        cfg = dict(defaults)
        full_path = os.path.join(PROJECT_ROOT, config_path)
        if os.path.isfile(full_path):
            try:
                with open(full_path) as f:
                    file_cfg = json.load(f)
                # Merge top-level runner config
                runner_cfg = file_cfg.get("autonomous_runner", {})
                cfg.update(runner_cfg)
                # Preserve sub-sections
                if "evolution" in file_cfg:
                    cfg["evolution"] = file_cfg["evolution"]
                if "retrain" in file_cfg:
                    cfg["retrain"] = file_cfg["retrain"]
                log(f"Loaded config from {config_path}")
            except (json.JSONDecodeError, OSError) as e:
                log(f"Warning: failed to load config ({e}), using defaults")
        else:
            log(f"No config file at {config_path}, using defaults")
        return cfg

    def _default_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_cycle": 0,
            "best_fitness": 0.0,
            "best_level": 0,
            "best_checkpoint": "",
            "error_count": 0,
            "total_errors": 0,
            "consecutive_errors": 0,
            "last_cycle_timestamp": "",
            "start_timestamp": datetime.now().isoformat(),
            "cycles_completed": 0,
            "stop_reason": None,
        }

    def _build_retrain_config(self) -> Any:
        """Build a config-like object for retrain_model()."""
        import types
        cfg = types.SimpleNamespace()
        rc = self._config.get("retrain", {})
        cfg.retrain_lr = rc.get("retrain_lr", 3e-5)
        cfg.retrain_freeze_layers = rc.get("retrain_freeze_layers", 0)
        cfg.retrain_max_per_level = rc.get("retrain_max_per_level", 0)
        cfg.retrain_model_size = rc.get("retrain_model_size", "xlarge")
        cfg.retrain_timeout = rc.get("retrain_timeout", 3600)
        cfg.retrain_patience = rc.get("retrain_patience", 5)
        cfg.retrain_max_epochs = rc.get("retrain_max_epochs", 50)
        cfg.retrain_grad_clip = rc.get("retrain_grad_clip", 1.0)
        cfg.retrain_batch_size = rc.get("retrain_batch_size", 8)
        cfg.retrain_weight_decay = rc.get("retrain_weight_decay", 0.01)
        cfg.retrain_min_examples = rc.get("retrain_min_examples", 5)
        cfg.retrain_val_split = rc.get("retrain_val_split", 0.15)
        cfg.retrain_adaptive_freeze = rc.get("retrain_adaptive_freeze", False)
        cfg.ref_high_augment = rc.get("ref_high_augment", 50)
        return cfg

    def _find_starting_checkpoint(self) -> str:
        """Locate the best available checkpoint."""
        candidates = [
            os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt"),
            os.path.join(M1_CHECKPOINTS, "model_final.pt"),
            os.path.join(M1_CHECKPOINTS, "model_45m.pt"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return ""

    # ── State persistence ──────────────────────────────────────────

    @property
    def _state_path(self) -> str:
        return os.path.join(PROJECT_ROOT, self._config["state_file"])

    def save_state(self) -> None:
        """Persist runner state to disk."""
        self._state["last_cycle_timestamp"] = datetime.now().isoformat()
        try:
            with open(self._state_path, "w") as f:
                json.dump(self._state, f, indent=2)
            log(f"State saved (cycle={self._state['current_cycle']}, "
                f"fitness={self._state['best_fitness']:.3f})")
        except (OSError, PermissionError) as e:
            log(f"Warning: failed to save state: {e}")

    def load_state(self) -> bool:
        """Resume from saved state on restart.

        Returns True if state was loaded successfully.
        """
        if not os.path.isfile(self._state_path):
            log("No saved state found — starting fresh")
            return False
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            # Validate version
            if data.get("version", 0) < 1:
                log("Warning: old state version, starting fresh")
                return False
            self._state.update(data)
            log(f"Resumed state: cycle={self._state['current_cycle']}, "
                f"fitness={self._state['best_fitness']:.3f}, "
                f"level={self._state['best_level']}")
            return True
        except (json.JSONDecodeError, OSError) as e:
            log(f"Warning: failed to load state ({e}), starting fresh")
            return False

    # ── Stop conditions ────────────────────────────────────────────

    def should_stop(self) -> bool:
        """Check all stop conditions."""
        if self._shutdown_requested:
            log("Shutdown requested (signal)")
            return True

        if self._state["current_cycle"] >= self._config["max_cycles"]:
            log(f"Max cycles reached ({self._config['max_cycles']})")
            return True

        if self._state["best_fitness"] >= self._config["target_fitness"]:
            log(f"Target fitness reached ({self._state['best_fitness']:.3f} >= "
                f"{self._config['target_fitness']})")
            return True

        stop_file = os.path.join(PROJECT_ROOT, self._config["stop_file"])
        if os.path.isfile(stop_file):
            log(f"Stop file detected: {stop_file}")
            os.remove(stop_file)
            return True

        if self._state["consecutive_errors"] >= self._config["max_errors_before_stop"]:
            log(f"Too many consecutive errors ({self._state['consecutive_errors']})")
            return True

        return False

    # ── Signal handling ────────────────────────────────────────────

    def _request_shutdown(self, signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM gracefully."""
        sig_name = signal.Signals(signum).name
        log(f"Received {sig_name} — will stop after current cycle")
        self._shutdown_requested = True

    # ── Main loop ──────────────────────────────────────────────────

    def run(self) -> None:
        """Main loop: cycle → retrain → cycle → ..."""
        # Install signal handlers
        signal.signal(signal.SIGINT, self._request_shutdown)
        signal.signal(signal.SIGTERM, self._request_shutdown)

        log("AutonomousRunner starting")
        log(f"  Max cycles:      {self._config['max_cycles']}")
        log(f"  Target fitness:  {self._config['target_fitness']}")
        log(f"  Generations:     {self._config['generations_per_cycle']}")
        log(f"  Checkpoint:      {self._checkpoint or 'none'}")

        self._health.set_status("running")

        while not self.should_stop():
            cycle = self._state["current_cycle"] + 1
            log(f"\n{'─' * 60}")
            log(f"CYCLE {cycle}/{self._config['max_cycles']}")
            log(f"{'─' * 60}")

            try:
                self.run_cycle(cycle)
                self._state["consecutive_errors"] = 0
            except KeyboardInterrupt:
                self.graceful_stop()
                return
            except MemoryError as e:
                self._handle_oom(e, cycle)
            except Exception as e:
                self.log_error(e, cycle)
                self.attempt_recovery()

            self._state["current_cycle"] = cycle
            self._state["cycles_completed"] = cycle
            self.save_state()
            self._health.write_health(self._state)

        self._health.set_status("stopped")
        self.save_state()
        log("AutonomousRunner stopped")

    def run_cycle(self, cycle: int) -> None:
        """Single evolution cycle + retrain.

        Delegates to run_multi_cycle functions for compatibility.
        """
        # Phase 1: Evolution
        evo_result = self._run_evolution(cycle)

        # Update best fitness
        fitness = evo_result.get("final_fitness", 0.0)
        level = evo_result.get("final_level", 0)
        if fitness > self._state["best_fitness"]:
            self._state["best_fitness"] = fitness
            self._state["best_level"] = level
            log(f"  New best: fitness={fitness:.3f}, level={level}")

        # Phase 2: Retrain (unless skipped)
        retrain_result = self._run_retrain(cycle, evo_result)

        # Record to results file
        result = {**evo_result, **retrain_result, "cycle": cycle}
        self._record_result(result)

    def _run_evolution(self, cycle: int) -> dict[str, Any]:
        """Run the evolution engine for one cycle."""
        from run_multi_cycle import run_evolution

        gens = self._config["generations_per_cycle"]
        return run_evolution(
            cycle=cycle,
            num_generations=gens,
            checkpoint_path=self._checkpoint,
        )

    def _run_retrain(self, cycle: int, evo_result: dict[str, Any]) -> dict[str, Any]:
        """Run retrain after evolution (with gating)."""
        from run_multi_cycle import retrain_model

        retrain_result = retrain_model(
            cycle=cycle,
            checkpoint_path=self._checkpoint,
            config=self._retrain_config,
            best_level=evo_result.get("final_level", 0),
        )

        # Checkpoint gating via MetaController
        if retrain_result.get("retrained"):
            try:
                from meta_controller import MetaController, parse_val_loss_from_retrain_log

                mc = MetaController(checkpoint_dir=str(M1_CHECKPOINTS))
                retrain_log = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")
                val_loss = parse_val_loss_from_retrain_log(retrain_log)
                if val_loss is not None:
                    new_ck = retrain_result.get("checkpoint", "")
                    if new_ck and os.path.isfile(new_ck):
                        accepted = mc.accept_checkpoint(val_loss, new_ck)
                        if accepted:
                            self._checkpoint = new_ck
                            log(f"  Checkpoint accepted (val_loss={val_loss:.4f})")
                        else:
                            log(f"  Checkpoint rejected (val_loss={val_loss:.4f})")
            except Exception as e:
                log(f"  Checkpoint gating failed: {e}")

        return retrain_result

    def _record_result(self, result: dict[str, Any]) -> None:
        """Append result to the multi-cycle results file."""
        try:
            with open(self._results_file, "a") as f:
                f.write(json.dumps(result) + "\n")
        except (OSError, PermissionError) as e:
            log(f"Warning: failed to record result: {e}")

    # ── Error handling ─────────────────────────────────────────────

    def log_error(self, error: Exception, cycle: int) -> None:
        """Log an error and update state."""
        self._state["error_count"] = self._state.get("error_count", 0) + 1
        self._state["total_errors"] = self._state.get("total_errors", 0) + 1
        self._state["consecutive_errors"] = self._state.get("consecutive_errors", 0) + 1
        log(f"  ERROR in cycle {cycle}: {type(error).__name__}: {error}")

    def _handle_oom(self, error: MemoryError, cycle: int) -> None:
        """Handle out-of-memory by reducing population size."""
        log(f"  OOM in cycle {cycle}: reducing population by 25%")
        self.log_error(error, cycle)
        # Reduce population in config
        evol = self._config.get("evolution", {})
        old_pop = evol.get("population_size", 50)
        new_pop = max(8, int(old_pop * 0.75))
        evol["population_size"] = new_pop
        log(f"  Population: {old_pop} → {new_pop}")
        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                log("  GPU cache cleared")
        except ImportError:
            pass

    def attempt_recovery(self) -> None:
        """Try to recover from errors: reload checkpoint."""
        log("  Attempting recovery...")
        self._checkpoint = self._find_starting_checkpoint()
        if self._checkpoint:
            log(f"  Recovery checkpoint: {self._checkpoint}")
        else:
            log("  No checkpoint available for recovery")

    def graceful_stop(self) -> None:
        """Complete current cycle and save state before exit."""
        log("Graceful stop requested — finishing current cycle...")
        self._shutdown_requested = True

    # ── Health ─────────────────────────────────────────────────────

    def get_health_status(self) -> dict[str, Any]:
        """Return current health: cycles_run, last_fitness, uptime, errors."""
        return {
            "cycles_run": self._state["current_cycle"],
            "best_fitness": self._state["best_fitness"],
            "best_level": self._state["best_level"],
            "error_count": self._state.get("total_errors", 0),
            "uptime_seconds": round(time.time() - self._health._start_time, 1),
        }


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Evolution Engine Runner"
    )
    parser.add_argument(
        "--config", type=str, default="config/engine_config.json",
        help="Config file path (default: config/engine_config.json)"
    )
    parser.add_argument(
        "--max-cycles", type=int, default=None,
        help="Override max cycles from config"
    )
    parser.add_argument(
        "--target-fitness", type=float, default=None,
        help="Override target fitness from config"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved state"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config without running"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current health status"
    )
    args = parser.parse_args()

    # Status mode
    if args.status:
        hm = HealthMonitor()
        hm.show_status()
        return

    # Create runner
    runner = AutonomousRunner(config_path=args.config)

    # Override config from CLI
    if args.max_cycles is not None:
        runner._config["max_cycles"] = args.max_cycles
    if args.target_fitness is not None:
        runner._config["target_fitness"] = args.target_fitness

    # Dry run mode
    if args.dry_run:
        print("Config validated OK:")
        for k, v in runner._config.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")
        print(f"Checkpoint: {runner._checkpoint or 'none'}")
        print(f"State file: {runner._state_path}")
        return

    # Resume mode
    if args.resume:
        runner.load_state()

    runner.run()


if __name__ == "__main__":
    main()
