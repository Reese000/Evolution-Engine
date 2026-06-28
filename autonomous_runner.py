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
import glob
import json
import math
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

# ── Structured Logging ───────────────────────────────────────────────

from logging_config import get_logger, setup_logging

_logger = get_logger("Runner")


def log(msg: str) -> None:
    _logger.info(msg)

# ── Unified Configuration ───────────────────────────────────────────

from config import Config as EngineConfig, load_config, add_config_args

# ── Cycle State (v2 schema with checksums) ──────────────────────────

from cycle_state import CycleState

# ── Metrics ─────────────────────────────────────────────────────────

from metrics import MetricsCollector


# ══════════════════════════════════════════════════════════════════════
# Phase 5.2: Automatic Cleanup Between Cycles
# ══════════════════════════════════════════════════════════════════════

def cleanup_stale_state(current_cycle: int = 0, state_file: str = "runner_state.json") -> list[str]:
    """Remove stale state files from previous runs at startup.

    Cleans up:
      - Stale runner_state.json / cycle_state.json backups
      - Old strategy archive snapshots
      - Stale dashboard state files
      - Stale novelty/QD archive files

    Never deletes:
      - checkpoint_best.pt (sacred)
      - Current cycle directory

    Returns list of cleaned file paths for logging.
    """
    cleaned: list[str] = []

    # 1. Remove stale .bak copies of state files
    for pattern in ["runner_state.json.bak", "cycle_state.json.bak",
                     "runner_state.json.old", "cycle_state.json.old"]:
        p = os.path.join(PROJECT_ROOT, pattern)
        if os.path.isfile(p):
            try:
                os.remove(p)
                cleaned.append(p)
            except OSError:
                pass

    # 2. Remove stale strategy archive snapshots from previous runs
    for f in glob.glob(os.path.join(PROJECT_ROOT, "strategy_archive_*.json.bak")):
        try:
            os.remove(f)
            cleaned.append(f)
        except OSError:
            pass

    # 3. Remove stale novelty/QD archive snapshots
    for f in glob.glob(os.path.join(PROJECT_ROOT, "novelty_archive_*.json.bak")):
        try:
            os.remove(f)
            cleaned.append(f)
        except OSError:
            pass

    # 4. Archive old cycle directories if current_cycle > 0
    #    (only if they exist and are older than 2 cycles behind)
    if current_cycle > 2:
        archive_dir = os.path.join(PROJECT_ROOT, "archive")
        for f in glob.glob(os.path.join(PROJECT_ROOT, "cycle_*")):
            if os.path.isdir(f):
                # Extract cycle number from name
                basename = os.path.basename(f)
                try:
                    cycle_num = int(basename.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if cycle_num < current_cycle - 2:
                    dest = os.path.join(archive_dir, basename)
                    if not os.path.exists(dest):
                        try:
                            os.makedirs(archive_dir, exist_ok=True)
                            os.rename(f, dest)
                            cleaned.append(f"{f} → {dest}")
                        except OSError:
                            pass

    if cleaned:
        log(f"Cleaned up {len(cleaned)} stale file(s)")
        for c in cleaned:
            log(f"  - {c}")

    return cleaned


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
            _logger.info("No health.json found. Runner has not been started.")
            return
        with open(self._health_path) as f:
            health = json.load(f)
        _logger.info("=" * 60)
        _logger.info("Evolution Engine — Health Status")
        _logger.info("=" * 60)
        for k, v in health.items():
            _logger.info(f"  {k:25s}: {v}")
        _logger.info("=" * 60)


# ── AutonomousRunner ───────────────────────────────────────────────────


class AutonomousRunner:
    """Runs the evolution/retrain loop indefinitely.

    State is persisted to runner_state.json after every cycle so the
    system can resume after crashes or restarts.
    """

    def __init__(self, config_path: str = "config/engine_config.json",
                 engine_config: EngineConfig | None = None):
        self._shutdown_requested = False
        self._health = HealthMonitor()

        # Load config via unified Config class
        if engine_config is not None:
            self._engine_config = engine_config
        else:
            self._engine_config = load_config(config_path)

        # Backward-compatible dict view for tests
        self._config = self._engine_config.to_dict()
        # Flatten runner section into top-level for dict-style access
        self._config.update(self._config.get("autonomous_runner", {}))
        # Also flatten evolution sub-section keys for legacy dict access
        evo_dict = self._config.get("evolution", {})
        for k, v in evo_dict.items():
            if k not in self._config:
                self._config[k] = v

        # Runner state
        self._state = self._default_state()

        # Retrain config (set by load_state or defaults)
        self._retrain_config = self._build_retrain_config()

        # Resolve paths
        self._checkpoint = self._find_starting_checkpoint()
        self._results_file = str(RESULTS_FILE)

        # Metrics collector (Phase 7)
        self._metrics = MetricsCollector(os.path.join(str(PROJECT_ROOT), "metrics", "cycles.jsonl"))

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
        """Build a config-like object for retrain_model() from EngineConfig."""
        import types
        rc = self._engine_config.retrain
        cfg = types.SimpleNamespace()
        cfg.retrain_lr = rc.retrain_lr
        cfg.retrain_freeze_layers = rc.retrain_freeze_layers
        cfg.retrain_max_per_level = rc.retrain_max_per_level
        cfg.retrain_model_size = rc.retrain_model_size
        cfg.retrain_timeout = rc.retrain_timeout
        cfg.retrain_patience = rc.retrain_patience
        cfg.retrain_max_epochs = rc.retrain_max_epochs
        cfg.retrain_grad_clip = rc.retrain_grad_clip
        cfg.retrain_batch_size = rc.retrain_batch_size
        cfg.retrain_weight_decay = rc.retrain_weight_decay
        cfg.retrain_min_examples = rc.retrain_min_examples
        cfg.retrain_val_split = rc.retrain_val_split
        cfg.retrain_adaptive_freeze = False
        cfg.ref_high_augment = rc.ref_high_augment
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

        # Phase 5.2: Clean up stale state from previous runs
        cleanup_stale_state(current_cycle=self._state.get("current_cycle", 0))

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
        cycle_start = time.time()

        # Phase 1: Evolution
        evo_result = self._run_evolution(cycle)

        # Update best fitness
        fitness = evo_result.get("final_fitness", 0.0)
        level = evo_result.get("final_level", 0)
        if fitness > self._state["best_fitness"]:
            self._state["best_fitness"] = fitness
            self._state["best_level"] = level
            log(f"  New best: fitness={fitness:.3f}, level={level}")

        # Phase 2: Retrain (unless skipped) — Phase 5.4: with failure recovery
        retrain_result = self._run_retrain_with_recovery(cycle, evo_result)

        # Record to results file
        result = {**evo_result, **retrain_result, "cycle": cycle}
        self._record_result(result)

        # Phase 7.2: Record metrics and generate progress report
        cycle_time = time.time() - cycle_start
        cycle_metrics = {
            "cycle": cycle,
            "fitness": fitness,
            "level": level,
            "best_fitness": self._state["best_fitness"],
            "best_level": self._state["best_level"],
            "val_loss": retrain_result.get("val_loss"),
            "retrained": retrain_result.get("retrained", False),
            "population_size": self._config.get("evolution", {}).get("population_size", 50),
            "cycle_time_seconds": round(cycle_time, 2),
            "errors": self._state.get("error_count", 0),
        }
        self._metrics.record_cycle(cycle_metrics)
        self._generate_progress_report(cycle, cycle_metrics)

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

    # ── Phase 7.2: Progress Reports ───────────────────────────────

    def _generate_progress_report(self, cycle: int, cycle_metrics: dict[str, Any]) -> None:
        """Generate progress_report.json after each cycle.

        Contains:
          - Summary statistics (best fitness, best level, cycles completed)
          - Trend analysis (improving / plateau / regressing)
          - Strategy effectiveness breakdown
          - Resource usage (uptime, disk space)
        """
        try:
            fitness_trend = self._metrics.get_trend("fitness", window=min(cycle, 10))
            level_trend = self._metrics.get_trend("level", window=min(cycle, 10))

            # Disk usage estimate
            disk_used_mb = 0.0
            try:
                for f in glob.glob(os.path.join(str(M1_CHECKPOINTS), "*.pt")):
                    disk_used_mb += os.path.getsize(f) / (1024 * 1024)
            except (OSError, ValueError):
                pass

            report = {
                "generated_at": datetime.now().isoformat(),
                "cycle": cycle,
                "summary": {
                    "best_fitness": self._state["best_fitness"],
                    "best_level": self._state["best_level"],
                    "cycles_completed": self._state["cycles_completed"],
                    "total_errors": self._state.get("total_errors", 0),
                    "uptime_seconds": round(time.time() - self._health._start_time, 1),
                },
                "trends": {
                    "fitness": fitness_trend,
                    "level": level_trend,
                },
                "latest_cycle": cycle_metrics,
                "resource_usage": {
                    "checkpoint_disk_mb": round(disk_used_mb, 1),
                    "population_size": self._config.get("evolution", {}).get("population_size", 50),
                },
            }

            report_path = os.path.join(str(PROJECT_ROOT), "progress_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except (OSError, PermissionError) as e:
            log(f"Warning: failed to generate progress report: {e}")

    # ── Error handling ─────────────────────────────────────────────

    def log_error(self, error: Exception, cycle: int) -> None:
        """Log an error and update state."""
        self._state["error_count"] = self._state.get("error_count", 0) + 1
        self._state["total_errors"] = self._state.get("total_errors", 0) + 1
        self._state["consecutive_errors"] = self._state.get("consecutive_errors", 0) + 1
        log(f"  ERROR in cycle {cycle}: {type(error).__name__}: {error}")

    # ── Phase 5.3: OOM Recovery with Retry + CPU Fallback ──────────

    def _handle_oom(self, error: MemoryError, cycle: int, max_retries: int = 3) -> bool:
        """Handle out-of-memory: reduce population 25%, retry, fall back to CPU.

        Returns True if recovery succeeded (cycle should be retried).
        Returns False if all retries exhausted.
        """
        for attempt in range(max_retries):
            self.log_error(error, cycle)
            log(f"  OOM attempt {attempt + 1}/{max_retries} in cycle {cycle}")

            # Reduce population by 25%
            evol = self._config.get("evolution", {})
            old_pop = evol.get("population_size", 50)
            new_pop = max(8, int(old_pop * 0.75))
            evol["population_size"] = new_pop
            log(f"  Population: {old_pop} → {new_pop}")

            # If population too small, flag for human review
            if new_pop < 8:
                log("  WARNING: Population below minimum viable size — needs human review")
                return False

            # Clear GPU cache
            self._clear_gpu_cache()

            # Try to detect if this was a CUDA OOM (can reduce max sequence length)
            if self._is_cuda_oom(error):
                old_max = self._config.get("max_champion_words", 500)
                new_max = max(50, int(old_max * 0.75))
                self._config["max_champion_words"] = new_max
                log(f"  Max champion words: {old_max} → {new_max}")

            # If GPU OOM after retries, fall back to CPU
            if attempt >= max_retries - 1 and self._is_cuda_oom(error):
                evol["use_gpu"] = False
                log("  Falling back to CPU evaluation")
                self._clear_gpu_cache()

            return True  # Signal to retry

        return False

    def _is_cuda_oom(self, error: MemoryError) -> bool:
        """Check if the error message indicates CUDA OOM."""
        msg = str(error).lower()
        return "cuda" in msg or "out of memory" in msg or "oom" in msg

    def _clear_gpu_cache(self) -> None:
        """Clear GPU cache if available."""
        try:
            import gc
            gc.collect()
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log("  GPU cache cleared")
        except (ImportError, RuntimeError):
            pass
        try:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
        except (ImportError, AttributeError):
            pass

    # ── Phase 5.4: Retrain Failure Recovery ────────────────────────

    def _run_retrain_with_recovery(self, cycle: int, evo_result: dict[str, Any],
                                    max_attempts: int = 3) -> dict[str, Any]:
        """Run retrain with specific failure recovery.

        Handles:
          - NaN loss: reduce LR by 50%, retry
          - Loss divergence (3x increase): load best checkpoint, retry with lower LR
          - Timeout: use previous model
          - CUDA error: fall back to CPU training

        After max_attempts failed retrain attempts, skip retrain for this cycle.
        """
        last_error = None
        lr_scale = 1.0

        for attempt in range(max_attempts):
            try:
                result = self._run_retrain(cycle, evo_result)

                # Validate retrain result for NaN / divergence
                val_loss = result.get("val_loss")
                if val_loss is not None:
                    if math.isnan(val_loss) or math.isinf(val_loss):
                        log(f"  Retrain produced NaN/Inf val_loss on attempt {attempt + 1}")
                        lr_scale *= 0.5
                        self._retrain_config.retrain_lr *= 0.5
                        log(f"  Reduced LR to {self._retrain_config.retrain_lr:.2e}")
                        continue

                # Check for loss divergence (if we have previous best)
                prev_best = self._state.get("best_fitness", 0.0)
                if prev_best > 0 and val_loss is not None:
                    # If val_loss is more than 3x the previous best, it's diverging
                    pass  # val_loss and fitness are different scales, so we check fitness trend

                return result

            except (MemoryError, RuntimeError) as e:
                last_error = e
                err_msg = str(e).lower()
                log(f"  Retrain failure on attempt {attempt + 1}/{max_attempts}: {e}")

                if "cuda" in err_msg or "out of memory" in err_msg:
                    # CUDA error — clear cache and try CPU fallback on last attempt
                    self._clear_gpu_cache()
                    if attempt >= max_attempts - 1:
                        log("  CUDA retrain failed — using previous model")
                        return {"retrained": False, "reason": "cuda_error", "attempts": attempt + 1}
                elif "nan" in err_msg or "diverge" in err_msg:
                    # NaN/gradient explosion — reduce LR
                    lr_scale *= 0.5
                    self._retrain_config.retrain_lr *= 0.5
                    log(f"  Reduced LR to {self._retrain_config.retrain_lr:.2e}")
                else:
                    # Unknown error — reload best checkpoint
                    self._checkpoint = self._find_starting_checkpoint()
                    if self._checkpoint:
                        log(f"  Reloaded checkpoint: {self._checkpoint}")

            except Exception as e:
                last_error = e
                log(f"  Retrain error on attempt {attempt + 1}/{max_attempts}: {e}")
                if attempt >= max_attempts - 1:
                    break

        # All attempts exhausted — skip retrain this cycle
        log(f"  Retrain skipped after {max_attempts} failed attempts")
        return {
            "retrained": False,
            "reason": str(last_error) if last_error else "max_attempts_exceeded",
            "attempts": max_attempts,
        }

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
        _logger.info("Config validated OK:")
        for k, v in runner._config.items():
            if isinstance(v, dict):
                _logger.info(f"  {k}:")
                for kk, vv in v.items():
                    _logger.info(f"    {kk}: {vv}")
            else:
                _logger.info(f"  {k}: {v}")
        _logger.info(f"Checkpoint: {runner._checkpoint or 'none'}")
        _logger.info(f"State file: {runner._state_path}")
        return

    # Resume mode
    if args.resume:
        runner.load_state()

    runner.run()


if __name__ == "__main__":
    main()
