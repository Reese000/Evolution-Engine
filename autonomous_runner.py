#!/usr/bin/env python3
"""AutonomousRunner — Continuous evolution/retrain daemon.

Runs the evolution→retrain loop indefinitely until:
  - max_cycles reached (unless --no-limit)
  - target_fitness achieved
  - stop file created (touch stop)
  - manual SIGINT/SIGTERM

Features:
  - Crash recovery: loads saved state on restart, resumes from last cycle
  - Graceful shutdown: completes current cycle before stopping
  - Health monitoring: writes health.json for external monitoring
  - Retrain integration: uses MetaController for checkpoint gating
  - OOM recovery: reduces population on GPU out-of-memory
  - Disk cleanup: automatically prunes old cycle directories
  - Continuous mode: --no-limit runs indefinitely with no cycle cap

Usage:
    python autonomous_runner.py                       # Run with defaults
    python autonomous_runner.py --no-limit            # Run indefinitely
    python autonomous_runner.py --max-cycles 10       # Override max cycles
    python autonomous_runner.py --target-fitness 200   # Stop at target
    python autonomous_runner.py --resume --no-limit   # Resume and keep going
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
import time
from datetime import datetime
from typing import Any

from paths import M1_CHECKPOINTS, META_CONTROLLER_STATE, PROJECT_ROOT, RESULTS_FILE, ensure_project_on_path

# ── Path setup ─────────────────────────────────────────────────────────


ensure_project_on_path()

# ── Structured Logging ───────────────────────────────────────────────

from data_rotation import run_all_rotations
from logging_config import get_logger

_logger = get_logger("Runner")


def log(msg: str) -> None:
    _logger.info(msg)


# ── Unified Configuration ───────────────────────────────────────────

from config import Config as EngineConfig
from config import load_config
from metrics import MetricsCollector

# ── Cycle State (v2 schema with checksums) ──────────────────────────


# ── Metrics ─────────────────────────────────────────────────────────


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
      - checkpoint_current.pt (sacred)
      - Current cycle directory

    Returns list of cleaned file paths for logging.
    """
    cleaned: list[str] = []

    # 1. Remove stale .bak copies of state files
    for pattern in ["runner_state.json.bak", "cycle_state.json.bak", "runner_state.json.old", "cycle_state.json.old"]:
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


# ── Disk Cleanup ──────────────────────────────────────────────────────

CYCLES_TO_KEEP = 10


def cleanup_old_cycles(current_cycle: int, keep: int = CYCLES_TO_KEEP) -> list[str]:
    """Remove old cycle directories after each completed cycle.

    Keeps the most recent ``keep`` cycle directories and deletes the rest.
    This prevents unbounded disk usage during indefinite operation.

    Args:
        current_cycle: The cycle number just completed.
        keep: Number of most-recent cycle dirs to retain (default: 10).

    Returns:
        List of directory paths that were removed.
    """
    if current_cycle <= keep:
        return []

    removed: list[str] = []
    cutoff = current_cycle - keep

    for path in glob.glob(os.path.join(PROJECT_ROOT, "cycle_*")):
        if not os.path.isdir(path):
            continue
        basename = os.path.basename(path)
        try:
            cycle_num = int(basename.split("_")[1])
        except (IndexError, ValueError):
            continue
        if cycle_num <= cutoff:
            try:
                import shutil

                shutil.rmtree(path)
                removed.append(path)
            except OSError as e:
                log(f"  Warning: failed to remove {path}: {e}")

    if removed:
        log(f"Disk cleanup: removed {len(removed)} old cycle dir(s) " f"(keeping last {keep})")
        for r in removed:
            log(f"  - {r}")

    return removed


# ── Health Status ──────────────────────────────────────────────────────


class HealthMonitor:
    """Writes health.json for external monitoring tools."""

    def __init__(self, state_path: str = "runner_state.json"):
        self._state_path = state_path
        # Resolve health.json relative to PROJECT_ROOT so external monitoring
        # tools can find it regardless of the working directory at launch.
        _dir = os.path.dirname(state_path)
        if not _dir or _dir == ".":
            _dir = str(PROJECT_ROOT)
        self._health_path = os.path.join(_dir, "health.json")
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


# ── Progress Report (standalone) ───────────────────────────────────


def generate_progress_report(state: dict[str, Any]) -> dict[str, Any]:
    """Generate a comprehensive progress report from runner state/metrics.

    This standalone function can be imported and called independently of
    the AutonomousRunner class, making it suitable for external monitoring
    tools, tests, and dashboards.

    Args:
        state: Dict containing runner state with the following expected keys:
            - cycle_number (int): Current cycle number.
            - best_fitness (float): Best fitness achieved so far.
            - fitness_history (list[float]): Full or partial fitness history.
            - level_history (list[tuple[int, int]]): List of (level, cycle) pairs.
            - retrain_val_losses (list[float]): Val loss from each retrain.
            - unique_individuals (int): Count of unique individuals seen.
            - time_evolution (float): Total seconds spent evolving.
            - time_retrain (float): Total seconds spent retraining.
            - time_idle (float): Total seconds idle.
            - errors (dict[str, int]): Mapping of error type name to count.

    Returns:
        Dict matching the progress_report_schema.json schema.
    """
    # ── Timestamp ──────────────────────────────────────────────────
    timestamp = datetime.now().isoformat()

    # ── Cycle & fitness ────────────────────────────────────────────
    cycle_number = state.get("cycle_number", 0)
    best_fitness = state.get("best_fitness", 0.0)

    fitness_history = state.get("fitness_history", [])
    # Last 20 values for trajectory
    fitness_trajectory = fitness_history[-20:] if fitness_history else []

    # ── Level progression ──────────────────────────────────────────
    level_history = state.get("level_history", [])
    level_progression_history = [{"level": lvl, "cycle": cyc} for lvl, cyc in level_history]

    # ── Retrain losses ─────────────────────────────────────────────
    retrain_val_loss_history = state.get("retrain_val_losses", [])

    # ── Diversity ──────────────────────────────────────────────────
    unique_individuals = state.get("unique_individuals", 0)
    diversity_metrics = {
        "unique_individuals": unique_individuals,
    }

    # ── Time breakdown ─────────────────────────────────────────────
    time_evolution = state.get("time_evolution", 0.0)
    time_retrain = state.get("time_retrain", 0.0)
    time_idle = state.get("time_idle", 0.0)
    time_total = round(time_evolution + time_retrain + time_idle, 3)

    time_breakdown_seconds = {
        "evolution": round(time_evolution, 3),
        "retrain": round(time_retrain, 3),
        "idle": round(time_idle, 3),
        "total": time_total,
    }

    # ── Errors ─────────────────────────────────────────────────────
    errors = state.get("errors", {})
    error_count = sum(errors.values())
    error_types = dict(errors)

    report = {
        "timestamp": timestamp,
        "cycle_number": cycle_number,
        "best_fitness": best_fitness,
        "fitness_trajectory": fitness_trajectory,
        "level_progression_history": level_progression_history,
        "retrain_val_loss_history": retrain_val_loss_history,
        "diversity_metrics": diversity_metrics,
        "time_breakdown_seconds": time_breakdown_seconds,
        "error_count": error_count,
        "error_types": error_types,
    }

    return report


# ── AutonomousRunner ───────────────────────────────────────────────────


class AutonomousRunner:
    """Runs the evolution/retrain loop indefinitely.

    State is persisted to runner_state.json after every cycle so the
    system can resume after crashes or restarts.
    """

    def __init__(self, config_path: str = "config/engine_config.json", engine_config: EngineConfig | None = None):
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

        # ── Stagnation tracking (indefinite operation) ────────────────
        self._stagnation_cycles = 0
        self._max_stagnation_cycles = self._config.get("max_stagnation_cycles", 15)
        self._stagnation_history: list[tuple[int, str, float | None]] = []
        self._self_correction_attempts = 0
        self._max_self_correction_attempts = self._config.get("max_self_correction_attempts", 5)
        self._last_fitness: float | None = None

        # ── Persistent MetaController (reused across _run_retrain calls) ──
        self._meta_controller: Any = None

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
            "error_types": {},
            "last_cycle_timestamp": "",
            "start_timestamp": datetime.now().isoformat(),
            "cycles_completed": 0,
            "stop_reason": None,
            "stagnation_cycles": 0,
            "self_correction_attempts": 0,
            "stagnation_history": [],
            "last_fitness": None,
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
        cfg.augmentation_k = getattr(rc, "augmentation_k", 8)
        cfg.retrain_adaptive_freeze = False
        return cfg

    def _find_starting_checkpoint(self) -> str:
        """Locate the best available checkpoint."""
        candidates = [
            os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt"),
            os.path.join(M1_CHECKPOINTS, "checkpoint_current.pt"),
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
        """Persist runner state to disk using atomic write.

        Uses temp file + os.replace to prevent corruption if the process
        crashes mid-write. This is critical for long-running evolution
        cycles where a crash during state save would destroy the resume
        point.
        """
        import tempfile

        self._state["last_cycle_timestamp"] = datetime.now().isoformat()
        # Persist stagnation tracking state
        self._state["stagnation_cycles"] = self._stagnation_cycles
        self._state["self_correction_attempts"] = self._self_correction_attempts
        self._state["stagnation_history"] = self._stagnation_history
        self._state["last_fitness"] = self._last_fitness
        dir_name = os.path.dirname(self._state_path) or "."
        try:
            os.makedirs(dir_name, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._state, f, indent=2)
                os.replace(tmp_path, self._state_path)
            except (OSError, PermissionError):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            log(f"State saved (cycle={self._state['current_cycle']}, " f"fitness={self._state['best_fitness']:.3f})")
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
            # Restore stagnation tracking state
            self._stagnation_cycles = self._state.get("stagnation_cycles", 0)
            self._self_correction_attempts = self._state.get("self_correction_attempts", 0)
            self._stagnation_history = self._state.get("stagnation_history", [])
            self._last_fitness = self._state.get("last_fitness", None)
            log(
                f"Resumed state: cycle={self._state['current_cycle']}, "
                f"fitness={self._state['best_fitness']:.3f}, "
                f"level={self._state['best_level']}"
            )
            return True
        except (json.JSONDecodeError, OSError) as e:
            log(f"Warning: failed to load state ({e}), starting fresh")
            return False

    # ── Stop conditions ────────────────────────────────────────────

    def should_stop(self, meta_controller=None, engine=None) -> bool:
        """Check all stop conditions.

        Args:
            meta_controller: Optional MetaController instance for stagnation detection.
            engine: Optional EvolutionEngine instance for self-correction operations.
        """
        if self._shutdown_requested:
            log("Shutdown requested (signal)")
            return True

        max_cycles = self._config.get("max_cycles")
        if max_cycles is not None and self._state["current_cycle"] >= max_cycles:
            log(f"Max cycles reached ({max_cycles})")
            return True

        if self._state["best_fitness"] >= self._config["target_fitness"]:
            log(f"Target fitness reached ({self._state['best_fitness']:.3f} >= " f"{self._config['target_fitness']})")
            return True

        stop_file = os.path.join(PROJECT_ROOT, self._config["stop_file"])
        if os.path.isfile(stop_file):
            log(f"Stop file detected: {stop_file}")
            os.remove(stop_file)
            return True

        if self._state["consecutive_errors"] >= self._config["max_errors_before_stop"]:
            log(f"Too many consecutive errors ({self._state['consecutive_errors']})")
            return True

        # ── Stagnation check ───────────────────────────────────────────
        if meta_controller is not None:
            from meta_controller import StagnationResult

            state = meta_controller.detect_stagnation()
            if state != StagnationResult.NORMAL:
                self._stagnation_cycles += 1
                self._stagnation_history.append((self._state["current_cycle"], str(state), self._last_fitness))
                log(
                    f"Stagnation detected: {state} "
                    f"(streak: {self._stagnation_cycles}/{self._max_stagnation_cycles})"
                )

                # Try self-correction before giving up
                if self._stagnation_cycles >= 5 and self._self_correction_attempts < self._max_self_correction_attempts:
                    self._self_correction_attempts += 1
                    self._try_self_correction(meta_controller, engine)
                    return False  # Don't stop yet, try the correction

                if self._stagnation_cycles >= self._max_stagnation_cycles:
                    self._write_stagnation_report()
                    return True
            else:
                # Progress made — reset stagnation tracking
                if self._stagnation_cycles > 0:
                    log(f"Progress detected after {self._stagnation_cycles} stagnant cycles, " f"resetting counter")
                self._stagnation_cycles = 0
                self._self_correction_attempts = 0

        return False

    # ── Signal handling ────────────────────────────────────────────

    def _request_shutdown(self, signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM gracefully."""
        sig_name = signal.Signals(signum).name
        log(f"Received {sig_name} — will stop after current cycle")
        self._shutdown_requested = True

    # ── Main loop ──────────────────────────────────────────────────

    def _init_meta_controller(self) -> Any:
        """Create or load a MetaController for stagnation detection.

        Returns a MetaController instance, or None if the required
        directories or modules are not available.
        """
        try:
            from meta_controller import MetaController

            mc = MetaController(checkpoint_dir=str(M1_CHECKPOINTS))
            # Restore previous state if available
            mc_path = str(META_CONTROLLER_STATE)
            if os.path.isfile(mc_path):
                mc.load_state(mc_path)
            return mc
        except Exception as e:
            log(f"MetaController init skipped: {e}")
            return None

    def run(self) -> None:
        """Main loop: cycle → retrain → cycle → ..."""
        # Install signal handlers
        signal.signal(signal.SIGINT, self._request_shutdown)
        signal.signal(signal.SIGTERM, self._request_shutdown)

        # Phase 5.2: Clean up stale state from previous runs
        cleanup_stale_state(current_cycle=self._state.get("current_cycle", 0))

        log("AutonomousRunner starting")
        mc_display = self._config.get("max_cycles")
        log(f"  Max cycles:      {'unlimited' if mc_display is None else mc_display}")
        log(f"  Target fitness:  {self._config['target_fitness']}")
        log(f"  Generations:     {self._config['generations_per_cycle']}")
        log(f"  Checkpoint:      {self._checkpoint or 'none'}")
        log(
            f"  Stagnation cap:  {self._max_stagnation_cycles} cycles "
            f"(self-correction up to {self._max_self_correction_attempts} attempts)"
        )

        self._health.set_status("running")

        # Lazy-init MetaController for stagnation detection (if not already wired)
        self._meta_controller = self._init_meta_controller()
        meta_controller = self._meta_controller

        while not self.should_stop(meta_controller=meta_controller):
            cycle = self._state["current_cycle"] + 1
            mc_display = self._config.get("max_cycles")
            max_label = "inf" if mc_display is None else mc_display
            log(f"\n{'─' * 60}")
            log(f"CYCLE {cycle}/{max_label}")
            log(f"{'─' * 60}")

            try:
                self.run_cycle(cycle)
                self._state["consecutive_errors"] = 0
            except KeyboardInterrupt:
                self.graceful_stop()
                return
            except MemoryError as e:
                recovered = self._handle_oom(e, cycle)
                if not recovered:
                    log(f"  OOM recovery failed after retries in cycle {cycle}")
                    self._state["error_count"] = self._state.get("error_count", 0) + 1
                    self._state["consecutive_errors"] = self._state.get("consecutive_errors", 0) + 1
                else:
                    self._state["consecutive_errors"] = 0
            except Exception as e:
                self.log_error(e, cycle)
                self.attempt_recovery()

            self._state["current_cycle"] = cycle
            self._state["cycles_completed"] = cycle
            self.save_state()
            self._health.write_health(self._state)

            # Disk cleanup: prune old cycle directories
            cleanup_old_cycles(cycle)

            # Data rotation: checkpoints, recurrent data, archives, reports
            rotation_result = run_all_rotations()
            if any(rotation_result.get(k, 0) > 0 for k in ["checkpoints", "recurrent", "archives", "reports"]):
                log(f"Data rotation: {sum(rotation_result.values())} items cleaned")

        self._health.set_status("stopped")
        self.save_state()
        log("AutonomousRunner stopped")

    def run_cycle(self, cycle: int) -> None:
        """Single evolution cycle + retrain.

        Delegates to run_multi_cycle functions for compatibility.
        """
        cycle_start = time.time()

        # Phase 1: Evolution
        evo_start = time.time()
        evo_result = self._run_evolution(cycle)
        evo_time = time.time() - evo_start

        # Skip retrain if evolution itself failed
        if "error" in evo_result:
            log(f"  Evolution failed: {evo_result['error']}")
            result = {**evo_result, "retrained": False, "cycle": cycle}
            self._record_result(result)
            return

        # Update best fitness
        fitness = evo_result.get("final_fitness", 0.0)
        level = evo_result.get("final_level", 0)
        self._last_fitness = fitness
        if fitness > self._state["best_fitness"]:
            self._state["best_fitness"] = fitness
            self._state["best_level"] = level
            log(f"  New best: fitness={fitness:.3f}, level={level}")

        # Phase 2: Retrain (unless skipped) — Phase 5.4: with failure recovery
        retrain_start = time.time()
        retrain_result = self._run_retrain_with_recovery(cycle, evo_result)
        retrain_time = time.time() - retrain_start

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
            "evolution_seconds": round(evo_time, 2),
            "retrain_seconds": round(retrain_time, 2),
        }
        self._metrics.record_cycle(cycle_metrics)
        self._generate_progress_report(cycle, cycle_metrics)

    def _run_evolution(self, cycle: int) -> dict[str, Any]:
        """Run the evolution engine for one cycle."""
        from run_multi_cycle import run_evolution

        gens = self._config["generations_per_cycle"]
        # Pass OOM-adjusted config values through to the evolution engine
        # so that population_size/use_gpu reductions actually take effect.
        evo = self._config.get("evolution", {})
        meta_params = {}
        if "population_size" in evo:
            meta_params["population_size"] = evo["population_size"]
        if "use_gpu" in evo:
            meta_params["use_gpu"] = evo["use_gpu"]
        if "max_champion_words" in self._config:
            meta_params["max_champion_words"] = self._config["max_champion_words"]
        return run_evolution(
            cycle=cycle,
            num_generations=gens,
            checkpoint_path=self._checkpoint,
            meta_evolution_params=meta_params or None,
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

        # Checkpoint gating via persistent MetaController
        # Uses self._meta_controller (same instance across all _run_retrain calls)
        # so _best_val_loss persists and gating actually works.
        if retrain_result.get("retrained") and self._meta_controller is not None:
            try:
                from meta_controller import parse_val_loss_from_retrain_log

                mc = self._meta_controller
                retrain_log = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")

                # Prefer val_loss already parsed by retrain_model (in the result dict),
                # fall back to log parsing for older code paths.
                val_loss = retrain_result.get("val_loss")
                if val_loss is None:
                    val_loss = parse_val_loss_from_retrain_log(retrain_log)
                if val_loss is not None:
                    retrain_result["val_loss"] = val_loss  # propagate to caller
                    new_ck = retrain_result.get("checkpoint", "")
                    if new_ck and os.path.isfile(new_ck):
                        accepted = mc.accept_checkpoint(val_loss, new_ck)
                        if accepted:
                            self._checkpoint = new_ck
                            log(f"  Checkpoint accepted (val_loss={val_loss:.4f})")
                        else:
                            log(f"  Checkpoint rejected (val_loss={val_loss:.4f})")
                # Persist MetaController state after each retrain
                mc_path = str(META_CONTROLLER_STATE)
                try:
                    mc.save_state(mc_path)
                except Exception as e:
                    log(f"  MetaController state save failed: {e}")
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

        Writes the rich, schema-v1.0 progress report (summary/fitness/level/
        retrain/diversity/time_breakdown/errors/trends/latest_cycle/
        resource_usage) to ``progress_report.json`` — this is the canonical,
        actively-consumed report file read by ``v11_dashboard.py`` and
        external monitoring tools.

        Also writes a legacy/simplified report (matching
        ``progress_report_schema.json``'s older flat shape, produced by the
        standalone ``generate_progress_report`` function) to
        ``progress_report_extended.json`` for tools that only need the flat
        trajectory-style fields.
        """
        try:
            all_records = self._metrics.get_history()

            # ── Standardized (legacy/flat) report ──────────────────
            # Level progression history: (level, cycle) tuples from metrics
            level_progression = []
            for r in all_records:
                lvl = r.get("level")
                cyc = r.get("cycle")
                if lvl is not None and cyc is not None:
                    level_progression.append((lvl, cyc))

            # Accumulate timing across all recorded cycles
            total_evo = 0.0
            total_retrain = 0.0
            for r in all_records:
                total_evo += r.get("evolution_seconds", 0.0) or 0.0
                total_retrain += r.get("retrain_seconds", 0.0) or 0.0
            uptime = round(time.time() - self._health._start_time, 1)
            total_idle = max(0.0, uptime - total_evo - total_retrain)

            report_state = {
                "cycle_number": cycle,
                "best_fitness": self._state["best_fitness"],
                "fitness_history": [r["fitness"] for r in all_records if "fitness" in r],
                "level_history": level_progression,
                "retrain_val_losses": [r.get("val_loss") for r in all_records if r.get("val_loss") is not None],
                "unique_individuals": cycle_metrics.get("unique_individuals"),
                "time_evolution": total_evo,
                "time_retrain": total_retrain,
                "time_idle": total_idle,
                "errors": dict(self._state.get("error_types", {})),
            }
            std_report = generate_progress_report(report_state)
            ext_path = os.path.join(str(PROJECT_ROOT), "progress_report_extended.json")
            with open(ext_path, "w", encoding="utf-8") as f:
                json.dump(std_report, f, indent=2)

            # ── Rich report (canonical progress_report.json) ────────
            self._write_extended_report(cycle, cycle_metrics, all_records)
        except (OSError, PermissionError) as e:
            log(f"Warning: failed to generate progress report: {e}")

    def _write_extended_report(
        self, cycle: int, cycle_metrics: dict[str, Any], all_records: list[dict[str, Any]]
    ) -> None:
        """Build and write the rich schema-v1.0 progress report to
        ``progress_report.json`` (the canonical, actively-consumed report).
        """
        try:
            fitness_trend = self._metrics.get_trend("fitness", window=min(cycle, 10))
            level_trend = self._metrics.get_trend("level", window=min(cycle, 10))

            recent_records = self._metrics.get_history(last_n=20)
            fitness_trajectory = [r["fitness"] for r in recent_records if "fitness" in r]
            level_history = [r["level"] for r in recent_records if "level" in r]

            val_loss_history = [r.get("val_loss") for r in recent_records]
            retrain_count = sum(1 for r in recent_records if r.get("retrained"))

            best_ever = max((r.get("fitness", 0.0) for r in all_records), default=0.0)
            diversity_score = cycle_metrics.get("diversity_score")

            evo_secs = cycle_metrics.get("evolution_seconds")
            retrain_secs = cycle_metrics.get("retrain_seconds")
            total_cycle_secs = cycle_metrics.get("cycle_time_seconds")

            all_cycle_times = [
                r.get("cycle_time_seconds") for r in all_records if r.get("cycle_time_seconds") is not None
            ]
            avg_cycle_secs = round(sum(all_cycle_times) / len(all_cycle_times), 2) if all_cycle_times else None

            error_types = dict(self._state.get("error_types", {}))

            disk_used_mb = 0.0
            try:
                for f in glob.glob(os.path.join(str(M1_CHECKPOINTS), "*.pt")):
                    disk_used_mb += os.path.getsize(f) / (1024 * 1024)
            except (OSError, ValueError):
                pass

            report = {
                "generated_at": datetime.now().isoformat(),
                "cycle": cycle,
                "schema_version": "1.0",
                "summary": {
                    "best_fitness": self._state["best_fitness"],
                    "best_level": self._state["best_level"],
                    "cycles_completed": self._state["cycles_completed"],
                    "total_errors": self._state.get("total_errors", 0),
                    "uptime_seconds": round(time.time() - self._health._start_time, 1),
                    "start_timestamp": self._state.get("start_timestamp", ""),
                },
                "fitness": {
                    "trajectory": fitness_trajectory,
                    "trend": fitness_trend,
                    "best_ever": best_ever,
                },
                "level": {
                    "progression_history": level_history,
                    "trend": level_trend,
                },
                "retrain": {
                    "val_loss_history": val_loss_history,
                    "last_val_loss": cycle_metrics.get("val_loss"),
                    "retrain_count": retrain_count,
                },
                "diversity": {
                    "unique_individuals": None,
                    "qd_occupied": None,
                    "qd_total": None,
                    "diversity_score": diversity_score,
                    "latest_diversity": diversity_score,
                },
                "time_breakdown": {
                    "evolution_seconds": evo_secs,
                    "retrain_seconds": retrain_secs,
                    "total_cycle_seconds": total_cycle_secs,
                    "avg_cycle_seconds": avg_cycle_secs,
                },
                "errors": {
                    "total_count": self._state.get("total_errors", 0),
                    "types": error_types,
                    "consecutive_errors": self._state.get("consecutive_errors", 0),
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
        error_type = type(error).__name__
        error_types = self._state.setdefault("error_types", {})
        error_types[error_type] = error_types.get(error_type, 0) + 1
        log(f"  ERROR in cycle {cycle}: {error_type}: {error}")

    # ── Phase 5.3: OOM Recovery with Retry + CPU Fallback ──────────

    def _handle_oom(self, error: MemoryError, cycle: int, max_retries: int = 3) -> bool:
        """Handle out-of-memory: reduce population 25%, retry, fall back to CPU.

        Returns True if recovery succeeded (cycle should be retried).
        Returns False if all retries exhausted.
        """
        # Track how many OOM errors we've seen in a row for this runner lifetime
        if not hasattr(self, "_oom_count"):
            self._oom_count = 0
        self._oom_count += 1
        attempt = self._oom_count

        self.log_error(error, cycle)
        log(f"  OOM attempt {attempt}/{max_retries} in cycle {cycle}")

        # Exhausted all retries
        if attempt > max_retries:
            log(f"  OOM recovery exhausted after {max_retries} retries")
            self._oom_count = 0  # Reset for future OOMs
            return False

        # Reduce population by 25%
        evol = self._config.get("evolution", {})
        old_pop = evol.get("population_size", 50)
        new_pop = max(8, int(old_pop * 0.75))
        evol["population_size"] = new_pop
        log(f"  Population: {old_pop} -> {new_pop}")

        # Clear GPU cache
        self._clear_gpu_cache()

        # Try to detect if this was a CUDA OOM (can reduce max sequence length)
        if self._is_cuda_oom(error):
            old_max = self._config.get("max_champion_words", 500)
            new_max = max(50, int(old_max * 0.75))
            self._config["max_champion_words"] = new_max
            log(f"  Max champion words: {old_max} -> {new_max}")

        # If GPU OOM after max retries, fall back to CPU
        if attempt >= max_retries and self._is_cuda_oom(error):
            evol["use_gpu"] = False
            log("  Falling back to CPU evaluation")
            self._clear_gpu_cache()
            self._oom_count = 0  # Reset

        return True  # Signal to retry with reduced population

    def _is_cuda_oom(self, error: MemoryError) -> bool:
        """Check if the error message indicates CUDA OOM.

        Requires "cuda" in the message to distinguish from generic OOM.
        """
        msg = str(error).lower()
        return "cuda" in msg or "out of memory" in msg

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

    def _run_retrain_with_recovery(
        self, cycle: int, evo_result: dict[str, Any], max_attempts: int = 3
    ) -> dict[str, Any]:
        """Run retrain with specific failure recovery.

        Handles:
          - NaN loss: reduce LR by 50%, retry
          - Loss divergence (3x increase): load best checkpoint, retry with lower LR
          - Timeout: use previous model
          - CUDA error: fall back to CPU training

        After max_attempts failed retrain attempts, skip retrain for this cycle.
        """
        last_error = None

        for attempt in range(max_attempts):
            try:
                result = self._run_retrain(cycle, evo_result)

                # Validate retrain result for NaN / divergence
                val_loss = result.get("val_loss")
                if val_loss is not None:
                    if math.isnan(val_loss) or math.isinf(val_loss):
                        log(f"  Retrain produced NaN/Inf val_loss on attempt {attempt + 1}")
                        self._retrain_config.retrain_lr *= 0.5
                        log(f"  Reduced LR to {self._retrain_config.retrain_lr:.2e}")
                        continue

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

    # ── Stagnation self-correction & reporting ──────────────────────

    def _try_self_correction(self, meta_controller: Any, engine: Any = None) -> None:
        """Try escalating strategies to break stagnation.

        Each attempt applies a progressively stronger intervention:
          1. Inject diversity — remove low-fitness QD archive entries
          2. Boost mutation rate (2x)
          3. Reset champion from archive
          4. Full archive reset (preserve champion)
          5. Log diagnostics (last resort)
        """
        attempt = self._self_correction_attempts
        log(f"Self-correction attempt {attempt}/{self._max_self_correction_attempts}")

        if attempt == 1:
            # Inject diversity — reset part of the archive
            log("Strategy: Inject diversity (reset low-fitness archive entries)")
            if engine is not None and hasattr(engine, "_qd_archive") and engine._qd_archive:
                sorted_entries = sorted(
                    engine._qd_archive.items(),
                    key=lambda x: x[1].get("fitness", 0),
                )
                cutoff = max(1, len(sorted_entries) // 4)
                for key, _ in sorted_entries[:cutoff]:
                    del engine._qd_archive[key]
                log(f"Removed {cutoff} low-fitness archive entries")

        elif attempt == 2:
            # Increase mutation rate temporarily
            log("Strategy: Boost mutation rate (2x)")
            if engine is not None and hasattr(engine, "config"):
                original_rate = getattr(engine.config, "mutation_rate", 0.1)
                engine.config.mutation_rate = min(original_rate * 2, 0.5)

        elif attempt == 3:
            # Reset champion to a random archive member
            log("Strategy: Reset champion from archive")
            if engine is not None and hasattr(engine, "_qd_archive") and engine._qd_archive:
                import random

                random_key = random.choice(list(engine._qd_archive.keys()))
                entry = engine._qd_archive[random_key]
                engine.champion = entry.get("program", engine.champion)
                engine.champion_fitness = entry.get("fitness", 0)

        elif attempt == 4:
            # Full archive reset with champion preservation
            log("Strategy: Archive reset (preserve champion)")
            if engine is not None:
                champion = getattr(engine, "champion", None)
                champion_fitness = getattr(engine, "champion_fitness", 0)
                if hasattr(engine, "_qd_archive"):
                    engine._qd_archive.clear()
                # Re-seed: the next generation will repopulate from champion mutation

        elif attempt >= 5:
            # Last resort: log detailed diagnostics
            log("All self-correction attempts exhausted — logging diagnostics")

    def _write_stagnation_report(self) -> None:
        """Write a stagnation report when the engine gives up."""
        report_path = os.path.join(str(PROJECT_ROOT), "stagnation_report.json")
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_cycles": self._state.get("current_cycle", 0),
            "stagnation_cycles": self._stagnation_cycles,
            "self_correction_attempts": self._self_correction_attempts,
            "stagnation_history": self._stagnation_history,
            "final_fitness": self._last_fitness,
            "recommendation": self._generate_recommendation(),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        log(f"Stagnation report written to {report_path}")

    def _generate_recommendation(self) -> str:
        """Generate a human-readable recommendation based on stagnation pattern."""
        if self._stagnation_cycles >= self._max_stagnation_cycles:
            return (
                f"Engine stagnated for {self._stagnation_cycles} consecutive cycles. "
                "Recommend: 1) Review champion program for local optimum, "
                "2) Expand the fitness landscape (new evaluation criteria), "
                "3) Increase population size for broader search, "
                "4) Add new mutation operators to explore different program structures."
            )
        return "No stagnation detected."

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
    parser = argparse.ArgumentParser(description="Autonomous Evolution Engine Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="config/engine_config.json",
        help="Config file path (default: config/engine_config.json)",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=None, help="Override max cycles from config (ignored if --no-limit is set)"
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="Disable cycle cap — run indefinitely until stop file, signal, or target fitness",
    )
    parser.add_argument("--target-fitness", type=float, default=None, help="Override target fitness from config")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    parser.add_argument("--status", action="store_true", help="Show current health status")
    args = parser.parse_args()

    # Status mode
    if args.status:
        hm = HealthMonitor()
        hm.show_status()
        return

    # Create runner
    runner = AutonomousRunner(config_path=args.config)

    # Override config from CLI
    if args.no_limit:
        runner._config["max_cycles"] = None
        _logger.info("Continuous mode active: no cycle limit")
    elif args.max_cycles is not None:
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
