#!/usr/bin/env python3
"""
Multi-Cycle Recurrent Feedback Runner.

Proves that the evolution -> retrain loop produces compounding self-improvement.

Flow per cycle:
  1. Run evolution (N generations) with current seed model
  2. Collect champion traces -> append to recurrent dataset
  3. Build training examples from recurrent dataset
  4. Retrain seed model on accumulated examples
  5. Replace checkpoint used by evolution engine
  6. Log metrics

Usage:
  python3 run_multi_cycle.py --cycles 5 --gens 100
"""

from __future__ import annotations

# R50: Enable faulthandler to capture faulting stack on ACCESS_VIOLATION
import faulthandler as _faulthandler

# v13: CUDA memory management for crash-resilient long runs
import logging
import os

_FAULT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_fault.log")
try:
    _fault_file = open(_FAULT_LOG, "w")  # noqa: SIM115
    _faulthandler.enable(file=_fault_file)
except (OSError, ValueError) as _e:
    logging.getLogger(__name__).debug("faulthandler enable failed: %s", _e)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
import argparse
import json
import math
import os
import re as _re
import shutil

# Pre-compiled regex patterns for hot-path parsing in the cycle loop.
# Avoids re-compilation on every call (once per cycle per retrain).
_VAL_LOSS_BT_RE = _re.compile(r"Val Loss:\s*([0-9]+\.[0-9]+)")
_EPOCH_RE = _re.compile(r"Epoch\s+(\d+)/")
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
M1_CHECKPOINTS = os.path.join(PROJECT_ROOT, "milestone1", "checkpoints")
M1_DIR = os.path.join(PROJECT_ROOT, "milestone1")
M1_RECURRENT = os.path.join(PROJECT_ROOT, "milestone1", "recurrent_data")  # legacy dir
M1_RECURRENT_DATA = os.path.join(M1_DIR, "recurrent_data.jsonl")  # canonical training path
RESULTS_FILE = os.path.join(PROJECT_ROOT, "multi_cycle_results.jsonl")

# PM Rule #17: Minimum elapsed seconds to credit a cycle as completed.
# Crash-truncated cycles (< MIN_CYCLE_SECONDS) are excluded from resume.
MIN_CYCLE_SECONDS = float(os.environ.get("EE_MIN_CYCLE_SECONDS", "30"))


def _is_valid_cycle_result(r: dict) -> bool:
    """PM Rule #17: A cycle result is only valid if it has no error and
    elapsed for at least MIN_CYCLE_SECONDS.  Crash-truncated partial
    cycles must not be credited as completed on resume."""
    if "error" in r:
        return False
    elapsed = r.get("elapsed_seconds", 0)
    if not isinstance(elapsed, (int, float)):
        return False
    if elapsed < MIN_CYCLE_SECONDS:
        phase_total = r.get("phase_total_seconds")
        if isinstance(phase_total, (int, float)) and phase_total >= MIN_CYCLE_SECONDS:
            return True
    return elapsed >= MIN_CYCLE_SECONDS


def _max_cycle_in_results(path: str = RESULTS_FILE) -> int:
    """Highest cycle number recorded in results (0 if empty or missing)."""
    if not os.path.isfile(path):
        return 0
    max_cycle = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                cycle = int(json.loads(line).get("cycle", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            max_cycle = max(max_cycle, cycle)
    return max_cycle


# Structured logging
from logging_config import get_logger as _get_logger

_logger_rmc = _get_logger("RunMultiCycle")


# Centralized path setup (replaces manual sys.path.insert hacks)
from paths import ensure_project_on_path

ensure_project_on_path()

# Prometheus metrics export (optional)
try:
    from metrics_export import create_metrics_app, stop_metrics_app, update_metrics

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

# MetaController -- dynamic parameterization orchestration
try:
    from meta_controller import MetaController, StagnationResult, TelemetrySnapshot, parse_val_loss_from_retrain_log

    _HAS_META_CONTROLLER = True
except ImportError:
    _HAS_META_CONTROLLER = False
    MetaController = None  # type: ignore
    TelemetrySnapshot = None  # type: ignore

# v11.5: BarrierTuner -- centralized barrier detection + self-tuning
try:
    from barrier_tuner import BarrierTuner, BarrierType, CycleMetrics

    _HAS_BARRIER_TUNER = True
except ImportError:
    _HAS_BARRIER_TUNER = False
    BarrierTuner = None  # type: ignore
    CycleMetrics = None  # type: ignore
    BarrierType = None  # type: ignore

# ProgressionTracker auto-tuning
try:
    from progression_tracker import ParameterActuator, TuningPolicy

    _HAS_TUNING_POLICY = True
except ImportError:
    _HAS_TUNING_POLICY = False
    TuningPolicy = None  # type: ignore
    ParameterActuator = None  # type: ignore


# Runtime GPU check with result caching (GPU availability doesn't change
# during a run, but avoids repeated torch.cuda imports on every cycle).
_has_gpu_cache: bool | None = None
_has_cupy_cache: bool | None = None


def _gpu_eval_enabled() -> bool:
    return os.environ.get("EE_USE_GPU", "1") != "0"


def has_gpu() -> bool:
    """PyTorch CUDA available for training (respects EE_USE_GPU=0)."""
    global _has_gpu_cache
    if _has_gpu_cache is not None:
        return _has_gpu_cache
    if not _gpu_eval_enabled():
        _has_gpu_cache = False
        return False
    try:
        import torch

        _has_gpu_cache = torch.cuda.is_available() and torch.cuda.device_count() > 0
    except (ImportError, AttributeError, RuntimeError):
        _has_gpu_cache = False
    return _has_gpu_cache


def has_cupy_cuda() -> bool:
    """CuPy CUDA available for GPU VM batch evaluation (respects EE_USE_GPU=0)."""
    global _has_cupy_cache
    if _has_cupy_cache is not None:
        return _has_cupy_cache
    if not _gpu_eval_enabled():
        _has_cupy_cache = False
        return False
    try:
        from gpu_vm_farm import _ensure_cupy

        _has_cupy_cache = bool(_ensure_cupy())
    except ImportError:
        _has_cupy_cache = False
    return _has_cupy_cache


# Default MetaController state path
META_CONTROLLER_STATE = os.path.join(SCRIPT_DIR, "meta_controller_state.json")
BEST_CHECKPOINT_PATH = os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt")


def load_checkpoint(path: str) -> dict:
    """Load a PyTorch checkpoint with SHA-256 integrity validation.

    Computes SHA-256 of the file on disk, loads the checkpoint via
    torch.load (weights_only=False for legacy checkpoints), then
    computes SHA-256 of the loaded tensor data and compares.
    Warns on mismatch but still returns the data (non-fatal for
    backward compatibility).
    """
    import hashlib

    import torch

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Compute file-level SHA-256
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    file_hash = sha256.hexdigest()
    log(f"  Checksum: {os.path.basename(path)} -> sha256:{file_hash[:16]}...")

    # Load checkpoint
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Verify loaded data integrity via tensor serialization
    data_hash = hashlib.sha256()
    for k in sorted(ckpt.keys()):
        v = ckpt[k]
        if hasattr(v, "numpy"):
            data_hash.update(v.numpy().tobytes())
        elif hasattr(v, "items"):
            for sk in sorted(v.keys()):
                sv = v[sk]
                if hasattr(sv, "numpy"):
                    data_hash.update(sv.numpy().tobytes())
                elif isinstance(sv, (int, float, str)):
                    data_hash.update(str(sv).encode())
    loaded_hash = data_hash.hexdigest()

    if loaded_hash != file_hash:
        log(
            f"  WARNING: Checksum mismatch for {os.path.basename(path)} "
            f"(file={file_hash[:16]}, data={loaded_hash[:16]})"
        )
    else:
        log(f"  Checksum verified: {os.path.basename(path)} OK")

    return ckpt


def log(msg: str) -> None:
    _logger_rmc.info(msg)


def _sync_champion_metrics_from_engine(engine: Any, metrics: dict[str, Any]) -> None:
    """Ensure persisted metrics match the engine champion after late promotions."""
    if engine.champion is None:
        return
    from evolution.fitness_helpers import (
        end_of_cycle_champion_fitness,
        ensure_champion_trace,
        terminal_certified_fitness_context,
    )

    with terminal_certified_fitness_context(engine):
        verified, verify_score = engine._verify_champion()
        champion_verify = max(
            float(verify_score),
            float(getattr(engine, "_champion_verify_score", 0.0)),
        )
        metrics["champion_verify_score"] = champion_verify
        metrics["champion_verified"] = champion_verify >= 1.0 - 1e-6
        metrics["champion_words"] = len(engine.champion)
        metrics["champion_program"] = [f"0x{w:08X}" for w in engine.champion]
        from evolution.fitness_helpers import certified_reporting_floor

        trace = ensure_champion_trace(engine)
        recomputed = end_of_cycle_champion_fitness(
            engine,
            engine.champion,
            trace,
            champion_verify,
        )
        if champion_verify < 0.99 - 1e-6:
            recomputed = max(recomputed, certified_reporting_floor(engine))
        engine.champion_fitness = recomputed
        metrics["final_fitness"] = recomputed
        if engine.champion is not None:
            from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
            from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS

            if (
                int(engine.champion_level) >= 30
                and 12 <= len(engine.champion) <= L30_COMPACT_MAX_WORDS
                and metrics.get("champion_verified")
            ):
                prom_raw = float(compact_l30_promotion_raw_fitness(list(engine.champion)))
                if prom_raw > 0.0:
                    metrics["promotion_raw_fitness"] = prom_raw
    metrics["verified_fitness"] = float(metrics["final_fitness"]) if metrics["champion_verified"] else 0.0


def run_evolution(
    cycle: int,
    num_generations: int,
    checkpoint_path: str,
    meta_evolution_params: dict[str, Any] | None = None,
    archive_data: dict[str, Any] | None = None,
    start_level: int = 0,
    meta_controller: Any = None,
) -> dict[str, Any]:
    """Run a single evolution cycle.

    Uses the given checkpoint as the seed model for program generation.
    If meta_evolution_params is provided, overrides EvolutionConfig values
    with dynamically tuned parameters from MetaController.
    If archive_data is provided (from a previous cycle), restores QD and
    novelty archives so cross-cycle knowledge is preserved.

    Returns a metrics dict.
    """
    log(f"  Cycle {cycle}: Running evolution ({num_generations} gens)...")
    if meta_evolution_params:
        log(f"  Cycle {cycle}: Meta-tuned evolution params: {meta_evolution_params}")

    # Paths already configured at module level via ensure_project_on_path()

    from evolution_engine import EvolutionConfig, EvolutionEngine

    # Set up paths for this cycle
    cycle_dir = os.path.join(PROJECT_ROOT, f"cycle_{cycle}")
    state_dir = os.path.join(cycle_dir, "dashboard_state")
    log_dir = os.path.join(cycle_dir, "logs")
    champion_dir = os.path.join(cycle_dir, "champions")
    try:
        os.makedirs(state_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(champion_dir, exist_ok=True)
    except (OSError, PermissionError) as e:
        log(f"  Cycle {cycle}: Failed to create directories: {e}")
        return {"cycle": cycle, "error": f"directory_creation_failed: {e}"}

    # Configure: aggressive curriculum advancement to see multi-level progression
    log(f"  Evolution starting at curriculum_level={start_level}")
    stall_threshold = int((meta_evolution_params or {}).get("curriculum_stall_threshold", 2))
    config = EvolutionConfig(
        max_generations=num_generations,
        population_size=50,
        curriculum_stall_threshold=stall_threshold,
        auto_curriculum=True,
        curriculum_level=start_level,
        checkpoint_interval=50,
        elite_ratio=0.1,
        dashboard_state_dir=state_dir,
        log_dir=log_dir,
        champion_dir=champion_dir,
        best_champion_path=os.path.join(PROJECT_ROOT, "milestone2", "best_champion"),
        recurrent_feedback_dir=M1_DIR,  # Write to milestone1/ (not subdirectory)
        novelty_weight=0.3,
        use_novelty_archive=True,
        use_qd_archive=True,
        qd_bonus_weight=0.1,
        use_gpu=os.environ.get("EE_USE_GPU", "1") != "0",  # Set EE_USE_GPU=0 to disable GPU eval
    )

    # Apply MetaController-tuned evolution params if provided
    if meta_evolution_params:
        for key, val in meta_evolution_params.items():
            if hasattr(config, key):
                setattr(config, key, val)

    # Let adaptive MG ratios work (L0-L5: 0.2, L6-L7: 0.5, L8+: 0.6)
    # Previously hardcoded to 0.2 which defeated the adaptive system.
    config.model_guided_ratio = None

    # Set the seed model path to our current cycle's checkpoint
    engine = EvolutionEngine(config=config)
    engine._cycle_number = cycle  # set for archive tracking

    from evolution.progress_controller import ProgressController

    engine.progress_controller = ProgressController(PROJECT_ROOT, stall_threshold=stall_threshold)
    engine.progress_controller.set_logger(log)
    engine.progress_controller.state.level = start_level
    if meta_controller is not None:
        engine.meta_controller = meta_controller

    # Override the seed model path to use our trained checkpoint
    engine._seed_model_path = os.path.relpath(checkpoint_path, SCRIPT_DIR)

    # v11.5: Apply external evolution params (from BarrierTuner) to engine
    if meta_evolution_params:
        # Filter to only evolution-relevant params
        evol_keys = {
            "mutation_rate",
            "mutation_rate_start",
            "mutation_rate_end",
            "use_mutation_scheduler",
            "adaptive_mutation_ceil",
            "diversity_threshold",
            "diversity_inject_ratio",
            "novelty_weight",
            "population_size",
            "elite_ratio",
            "champion_crossover_prob",
            "crossover_dedicated_prob",
            "max_consecutive_same_strategy",
            "qd_bonus_weight",
            "tournament_size",
            "crossover_rate",
        }
        engine_params = {k: v for k, v in meta_evolution_params.items() if k in evol_keys}
        if engine_params:
            engine.apply_external_params(engine_params)
        applied_mu = meta_evolution_params.get("mutation_rate")
        if applied_mu is not None:
            engine._applied_mutation_rate = float(applied_mu)

    # Load archive state from previous cycle (cross-cycle persistence)
    if archive_data is not None:
        engine.load_archive_state(
            qd_data=archive_data.get("qd"),
            novelty_data=archive_data.get("novelty"),
            strategy_data=archive_data.get("strategy"),
        )
        qd_size = engine.qd_archive.size if engine.qd_archive is not None else 0
        novel_size = engine.novelty_archive.size if engine.novelty_archive is not None else 0
        strat_size = engine.strategy_archive.total_records() if engine.strategy_archive is not None else 0
        log(
            f"  Cycle {cycle}: Loaded archives -- QD={qd_size} entries, "
            f"novelty={novel_size} entries, strategy={strat_size} records"
        )

    tick_path = os.path.join(PROJECT_ROOT, ".agents", "tick_state.json")
    try:
        with open(tick_path, encoding="utf-8") as f:
            tick = json.load(f)
        if tick.get("force_reference_champion") or tick.get("force_champion_reseed"):
            os.environ["EE_FORCE_REFERENCE_CHAMPION"] = "1"
        if (tick.get("force_register_repair") or tick.get("force_champion_reseed")) and not tick.get(
            "terminal_diversity_boost"
        ):
            os.environ["EE_REGISTER_REPAIR_BOOST"] = "1"
    except (OSError, json.JSONDecodeError):
        pass

    start = time.time()
    try:
        result = engine.run()
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        import traceback

        tb = traceback.format_exc()
        for line in tb.strip().split(chr(10)):
            log(f"  [TRACEBACK] {line}")
        log(f"  Cycle {cycle}: Evolution engine crashed: {e}")
        return {
            "cycle": cycle,
            "error": f"evolution_crash: {e}",
            "generations_completed": engine.current_generation,
            "final_fitness": engine.champion_fitness,
        }
    elapsed = time.time() - start

    vm_farm = getattr(engine, "vm_farm", None)
    gpu_batches = 0
    cpu_fallbacks = 0
    vm_use_gpu = False
    if vm_farm is not None:
        gpu_batches = getattr(vm_farm, "_gpu_eval_count", 0)
        cpu_fallbacks = getattr(vm_farm, "_gpu_fallback_count", 0)
        vm_use_gpu = bool(vm_farm.use_gpu)
        log(
            f"  Cycle {cycle}: VM eval device={'GPU' if vm_use_gpu else 'CPU'}, "
            f"gpu_batches={gpu_batches}, "
            f"cpu_fallbacks={cpu_fallbacks}"
        )

    # Collect metrics
    champion_traces = engine.recurrent_examples_count
    novelty_size = 0
    qd_occupied = 0
    qd_total = 0
    if engine.novelty_archive is not None:
        novelty_size = engine.novelty_archive.size
    if engine.qd_archive is not None:
        qd_occupied = engine.qd_archive.size
        qd_total = engine.qd_archive.num_bins**2

    strategy_records = 0
    strategy_diversity = 0.0
    strategy_dominant = None
    strategy_monoculture = False
    if engine.strategy_archive is not None:
        strategy_records = engine.strategy_archive.total_records()
        strategy_diversity = engine.strategy_archive.get_strategy_diversity(engine.champion_level)
        dom = engine.strategy_archive.get_dominance_report()
        strategy_dominant = dom.get("dominant_strategy")
        strategy_monoculture = bool(dom.get("is_monoculture", False))

    hist = getattr(engine, "best_fitness_history", None) or []
    fitness_start = hist[0] if hist else None
    fitness_end = hist[-1] if hist else None
    fitness_gain = (fitness_end - fitness_start) if fitness_start is not None and fitness_end is not None else 0.0

    metrics = {
        "cycle": cycle,
        "generations_completed": result.get("generations_completed", 0),
        "final_fitness": result.get("final_fitness", 0.0),
        "final_level": result.get("final_level", 0),
        "champion_words": result.get("champion_words", 0),
        # R50: Integrity gate — verified fitness and verify() result
        "verified_fitness": result.get("verified_fitness", 0.0),
        "champion_verified": result.get("champion_verified", False),
        "champion_verify_score": result.get("champion_verify_score", 0.0),
        "max_pop_verify": result.get("max_pop_verify", 0.0),
        # v13 Phase E.2: Include champion program for memorization diagnostic
        "champion_program": ([f"0x{w:08X}" for w in engine.champion] if engine.champion is not None else None),
        "elapsed_seconds": round(elapsed, 2),
        "champion_traces": champion_traces,
        "novelty_archive_size": novelty_size,
        "qd_occupied_cells": qd_occupied,
        "qd_total_cells": qd_total,
        "checkpoint": checkpoint_path,
        # PM Rule #18: task_fitness = fitness minus capped level_bonus
        "task_fitness": getattr(engine, "_task_fitness", 0.0),
        # Self-calibrate state (for cross-cycle persistence)
        "mutation_rate": getattr(engine.config, "mutation_rate", 0.3),
        "mutation_rate_applied": getattr(engine, "_applied_mutation_rate", None),
        "diversity_threshold": getattr(engine.config, "diversity_threshold", 0.3),
        "diversity_inject_ratio": getattr(engine.config, "diversity_inject_ratio", 0.1),
        # v11.5: Include diversity for BarrierTuner
        "diversity": engine._current_diversity,
        "max_challenger_words": int(getattr(engine, "_max_challenger_words_cycle", 0)),
        "vm_use_gpu": vm_use_gpu,
        "vm_gpu_batches": gpu_batches,
        "vm_cpu_fallbacks": cpu_fallbacks,
        "strategy_archive_records": strategy_records,
        "strategy_diversity": strategy_diversity,
        "strategy_dominant": strategy_dominant,
        "strategy_monoculture": strategy_monoculture,
        "fitness_start": fitness_start,
        "fitness_end": fitness_end,
        "fitness_gain": fitness_gain,
    }

    # EE-W5: additive per-generation phase split (VM eval vs orchestration).
    # Copied verbatim from the engine run summary; existing keys untouched.
    for _k, _v in result.items():
        if isinstance(_k, str) and _k.startswith("gen_phase_"):
            metrics[_k] = _v

    try:
        if hasattr(engine, "_enforce_l30_certified_champion"):
            engine._enforce_l30_certified_champion()
        elif hasattr(engine, "_maybe_restore_best_certified_champion"):
            engine._maybe_restore_best_certified_champion()
        from evolution.fitness_helpers import reporting_champion_fitness, terminal_certified_fitness_context

        with terminal_certified_fitness_context(engine):
            verified, verify_score = engine._verify_champion()
            champion_verify = max(
                float(verify_score),
                float(getattr(engine, "_champion_verify_score", 0.0)),
            )
            metrics["champion_verify_score"] = champion_verify
            metrics["champion_verified"] = champion_verify >= 1.0 - 1e-6
            if engine.champion is not None:
                from evolution.fitness_helpers import (
                    certified_reporting_floor,
                    end_of_cycle_champion_fitness,
                    ensure_champion_trace,
                )

                trace = ensure_champion_trace(engine)
                recomputed = end_of_cycle_champion_fitness(
                    engine,
                    engine.champion,
                    trace,
                    champion_verify,
                )
                if champion_verify < 0.99 - 1e-6:
                    recomputed = max(recomputed, certified_reporting_floor(engine))
                engine.champion_fitness = recomputed
                metrics["final_fitness"] = recomputed
        if engine.champion is not None:
            metrics["champion_words"] = len(engine.champion)
            metrics["champion_program"] = [f"0x{w:08X}" for w in engine.champion]
            if engine.champion is not None:
                from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
                from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS

                if (
                    int(engine.champion_level) >= 30
                    and 12 <= len(engine.champion) <= L30_COMPACT_MAX_WORDS
                    and metrics.get("champion_verified")
                ):
                    prom_raw = float(compact_l30_promotion_raw_fitness(list(engine.champion)))
                    if prom_raw > 0.0:
                        metrics["promotion_raw_fitness"] = prom_raw
        metrics["verified_fitness"] = float(metrics["final_fitness"]) if metrics["champion_verified"] else 0.0
    except (RuntimeError, TypeError, ValueError, AttributeError) as e:
        log(f"  Cycle {cycle}: Champion verify/fitness recompute failed: {e}")

    try:
        from evolution.champion_rollback import apply_l30_certified_metrics_if_degraded

        if apply_l30_certified_metrics_if_degraded(PROJECT_ROOT, engine, metrics):
            log(
                f"  Cycle {cycle}: Applied L30 certified metrics "
                f"(fit={metrics.get('final_fitness'):.1f}, "
                f"words={metrics.get('champion_words')}, "
                f"verify={metrics.get('champion_verify_score')})"
            )
    except (ImportError, RuntimeError, TypeError, ValueError, OSError) as e:
        log(f"  Cycle {cycle}: L30 certified metrics gate failed: {e}")

    try:
        from evolution.terminal_stagnation import (
            compact_l30_certified_champion_words,
            maybe_rearm_scan_floor_above_champion,
            reconcile_terminal_efficiency_compact_state,
        )

        certified_words = compact_l30_certified_champion_words(metrics, PROJECT_ROOT)
        if certified_words > 0 and int(metrics.get("final_level", 0)) >= int(
            getattr(getattr(engine, "config", None), "max_level", 30)
        ):
            reconcile_terminal_efficiency_compact_state(
                certified_words,
                int(cycle),
                heldout=float(metrics.get("champion_verify_score", 0.0)),
                verify_score=float(metrics.get("champion_verify_score", 0.0)),
            )
        champ_prog = getattr(engine, "champion", None)
        if (
            champ_prog
            and certified_words > 0
            and len(champ_prog) == certified_words
            and int(metrics.get("final_level", 0)) >= int(getattr(getattr(engine, "config", None), "max_level", 30))
        ):
            if maybe_rearm_scan_floor_above_champion(
                int(cycle),
                list(champ_prog),
                verify_score=float(metrics.get("champion_verify_score", 0.0)),
            ):
                log(f"  Cycle {cycle}: Re-armed scan floor at " f"{len(champ_prog)}w (champion outgrew shorter floor)")
    except (RuntimeError, TypeError, ValueError, AttributeError, ImportError) as e:
        log(f"  Cycle {cycle}: Scan floor re-arm failed: {e}")

    try:
        if hasattr(engine, "_maybe_promote_terminal_efficiency_champion"):
            prev_words = int(metrics.get("champion_words", 0))
            if engine._maybe_promote_terminal_efficiency_champion():
                _sync_champion_metrics_from_engine(engine, metrics)
                log(
                    f"  Cycle {cycle}: Post-certify terminal efficiency promote "
                    f"{prev_words}w -> {metrics['champion_words']}w "
                    f"(fit={metrics.get('final_fitness', 0):.3f})"
                )
    except (RuntimeError, TypeError, ValueError, AttributeError) as e:
        log(f"  Cycle {cycle}: Post-certify terminal efficiency promote failed: {e}")

    try:
        from evolution.terminal_stagnation import (
            compact_l30_certified_champion_words,
            reconcile_terminal_efficiency_compact_state,
        )

        certified_words = compact_l30_certified_champion_words(metrics, PROJECT_ROOT)
        if certified_words > 0:
            if reconcile_terminal_efficiency_compact_state(
                certified_words,
                int(cycle),
                heldout=float(metrics.get("champion_verify_score", 0.0)),
                verify_score=float(metrics.get("champion_verify_score", 0.0)),
            ):
                log(
                    f"  Cycle {cycle}: Reconciled terminal efficiency state to "
                    f"{certified_words}w certified champion"
                )
    except (RuntimeError, TypeError, ValueError, AttributeError, ImportError) as e:
        log(f"  Cycle {cycle}: Terminal efficiency reconcile failed: {e}")

    # Serialize archive state to sidecar (keeps jsonl lean)
    try:
        from evolution.archive_persistence import cap_strategy_archive_state, save_archive_sidecar
        from evolution.terminal_stagnation import terminal_certified_plateau

        archive_state = engine.save_archive_state()
        if terminal_certified_plateau(
            int(metrics.get("final_level", 0)),
            float(metrics.get("champion_verify_score", 0.0)),
            float(metrics.get("final_fitness", 0.0)),
        ):
            cap = int(os.environ.get("EE_TERMINAL_STRATEGY_ARCHIVE_CAP", "3200"))
            archive_state = cap_strategy_archive_state(archive_state, cap) or archive_state
        sidecar_path = save_archive_sidecar(PROJECT_ROOT, cycle, archive_state)
        metrics["archive_state_path"] = sidecar_path
    except (TypeError, ValueError, OSError) as e:
        log(f"  Cycle {cycle}: Failed to serialize archive state: {e}")

    try:
        from evolution.fitness_helpers import align_scan_floor_persisted_metrics

        align_scan_floor_persisted_metrics(engine, metrics)
    except (ImportError, RuntimeError, TypeError, ValueError, AttributeError):
        pass

    log(
        f"  Cycle {cycle} complete: gen={metrics['generations_completed']}, "
        f"fitness={metrics['final_fitness']:.3f}, level={metrics['final_level']}, "
        f"gain={fitness_gain:.1f}, "
        f"traces={champion_traces}, novelty={novelty_size}, "
        f"qd={qd_occupied}/{qd_total}, strategy={strategy_records}, "
        f"strat_div={strategy_diversity:.2f}, dominant={strategy_dominant}, "
        f"mono={strategy_monoculture}, {elapsed:.1f}s"
    )
    return metrics


def merge_archive_to_recurrent(cycle: int) -> int:
    """Merge ChampionArchive entries into recurrent_data.jsonl for training.

    The archive accumulates champion programs across cycles with rich metadata
    (level, fitness, cycle). This function exports archive entries into the
    training data file, deduplicating by champion_sha256.

    Returns the number of new entries added.
    """
    try:
        from champion_archive import ChampionArchive
    except ImportError:
        log(f"  Cycle {cycle}: ChampionArchive not available, skipping merge")
        return 0

    archive_path = os.path.join(PROJECT_ROOT, "champions", "archive.jsonl")
    if not os.path.isfile(archive_path):
        return 0

    archive = ChampionArchive(path=archive_path, max_size=1000)
    if archive.size == 0:
        return 0

    recurrent_data = M1_RECURRENT_DATA
    os.makedirs(M1_RECURRENT, exist_ok=True)

    # Load existing sha256 hashes to deduplicate
    existing_hashes: set = set()
    if os.path.isfile(recurrent_data):
        try:
            with open(recurrent_data) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        h = entry.get("champion_sha256", "")
                        if h:
                            existing_hashes.add(h)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except (OSError, PermissionError):
            pass  # non-fatal: archive dedup is best-effort

    # Append new archive entries (deduped by sha256); L17+ verified get extra weight
    added = 0
    from evolution.register_repair import merge_priority_weight

    with open(recurrent_data, "a") as f:
        for entry in archive.entries:
            if entry.sha256 in existing_hashes:
                continue
            weight = merge_priority_weight(
                int(entry.level),
                float(getattr(entry, "verify_score", 0.0) or 0.0),
                float(entry.fitness),
            )
            example = {
                "level": entry.level,
                "generation": entry.generation,
                "fitness": entry.fitness,
                "champion_sha256": entry.sha256,
                "champion_words": entry.instructions_count,
                "champion_program": entry.program_hex,
                "trace_status": entry.trace_status,
                "trace_instr_count": entry.instructions_count,
                "task_score": entry.task_score,
                "save_reason": "champion_archive",
                "cycle": entry.cycle,
                "training_weight": weight,
            }
            f.writelines(json.dumps(example) + "\n" for _ in range(weight))
            existing_hashes.add(entry.sha256)
            added += weight

    if added > 0:
        log(
            f"  Cycle {cycle}: Merged {added} archive champions into recurrent_data.jsonl "
            f"(archive={archive.size}, existing={len(existing_hashes) - added})"
        )
    return added


def _resolve_retrain_checkpoint(cycle: int, checkpoint_path: str) -> tuple[str, str | None]:
    """Use checkpoint_current.pt for retrain and optionally back up before training."""
    full_checkpoint = os.path.join(M1_CHECKPOINTS, "checkpoint_current.pt")
    backup_checkpoint = None
    if os.path.isfile(full_checkpoint):
        log(f"  Cycle {cycle}: Using best full checkpoint for retrain: {full_checkpoint}")
        checkpoint_path = full_checkpoint
        backup_checkpoint = full_checkpoint + ".pre_retrain_backup"
        try:
            shutil.copy2(full_checkpoint, backup_checkpoint)
            log(f"  Cycle {cycle}: Backed up full checkpoint to: {backup_checkpoint}")
        except (OSError, PermissionError) as e:
            log(f"  Cycle {cycle}: Warning: backup failed: {e}")
            backup_checkpoint = None
    else:
        log(f"  Cycle {cycle}: Using passed checkpoint for retrain: {checkpoint_path}")
    return checkpoint_path, backup_checkpoint


def _validate_retrain_eligibility(cycle: int, config: Any) -> tuple[dict[str, Any] | None, int]:
    """Return early skip dict when retrain cannot run, else (None, example_count)."""
    recurrent_data = M1_RECURRENT_DATA
    if not os.path.isfile(recurrent_data):
        log(f"  Cycle {cycle}: No recurrent data found -- skipping retrain")
        return {"retrained": False, "reason": "no_data"}, 0
    with open(recurrent_data) as f:
        example_count = sum(1 for line in f if line.strip())
    min_examples = 1
    if config is not None:
        min_examples = getattr(config, "retrain_min_examples", 1)
    if example_count < min_examples:
        log(f"  Cycle {cycle}: Only {example_count} examples (< {min_examples}) -- skipping retrain")
        return {"retrained": False, "reason": f"only_{example_count}_examples"}, example_count
    return None, example_count


def _kill_orphan_train_model_processes(cycle: int) -> None:
    """Terminate stale train_model.py processes that can deadlock CUDA."""
    import signal

    orphan_count = 0
    try:
        pids_str = subprocess.check_output(["pgrep", "-f", r"train_model\.py"], timeout=5).decode().strip()
        for old_pid_str in pids_str.split("\n"):
            old_pid = old_pid_str.strip()
            if not old_pid or old_pid == str(os.getpid()):
                continue
            try:
                pid_int = int(old_pid)
                os.kill(pid_int, signal.SIGTERM)
                exited = False
                for _wait in range(6):
                    try:
                        os.kill(pid_int, 0)
                        time.sleep(0.5)
                    except ProcessLookupError:
                        exited = True
                        break
                if not exited:
                    os.kill(pid_int, signal.SIGKILL)
                    log(f"  Cycle {cycle}: Orphan PID {pid_int} did not respond to SIGTERM -- used SIGKILL")
                else:
                    log(f"  Cycle {cycle}: Orphan PID {pid_int} exited gracefully after SIGTERM")
                orphan_count += 1
            except (ProcessLookupError, ValueError, OSError) as e:
                log(f"  Cycle {cycle}: Failed to clean up orphan PID {old_pid}: {e}")
    except subprocess.CalledProcessError:
        pass
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log(f"  Cycle {cycle}: Orphan cleanup check failed: {e}")
    if orphan_count:
        log(f"  Cycle {cycle}: Cleaned up {orphan_count} orphan process(es)")


def _cooldown_gpu_before_retrain(cycle: int) -> None:
    """Release GPU memory and cooldown CUDA driver before in-process retrain."""
    try:
        import gc as _gc

        import torch

        if torch.cuda.is_available():
            if "cupy" in sys.modules:
                try:
                    import cupy as cp

                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    log(f"  Cycle {cycle}: CuPy memory pool cleared")
                except (ImportError, RuntimeError, AttributeError):
                    pass
            _gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            time.sleep(3)
            log(f"  Cycle {cycle}: GPU reset complete")
    except (RuntimeError, ImportError, OSError):
        pass


@dataclass
class RetrainRunConfig:
    model_size: str
    lr: float | None
    adaptive_freeze: bool
    freeze: int
    compile_mode: Any
    wd: float | None
    grad_clip: float | None
    patience: int | None
    max_epochs: int | None
    val_split: float | None
    max_per_level: int | None
    grad_accum_steps: int
    lr_scheduler: str
    swa_start: int | None
    swa_lr: float | None
    effective_bs: int
    train_timeout: int


def _resolve_retrain_run_config(cycle: int, config: Any, best_level: int) -> RetrainRunConfig:
    model_size = getattr(config, "retrain_model_size", "xlarge") if config is not None else "xlarge"
    lr = getattr(config, "retrain_lr", None) if config is not None else None
    if lr is not None and lr > 1e-5:
        existing_ckpt = os.path.join(M1_CHECKPOINTS, "checkpoint_current.pt")
        if os.path.isfile(existing_ckpt):
            try:
                import torch as _torch

                _ckpt_data = _torch.load(existing_ckpt, map_location="cpu", weights_only=False)
                _existing_val = _ckpt_data.get("val_loss", 999)
                if _existing_val < 2.0:
                    lr = 1e-5
                    log(
                        f"  Cycle {cycle}: Capped LR from config to {lr} "
                        f"(existing val_loss={_existing_val:.4f} < 2.0)"
                    )
            except (OSError, KeyError, ValueError, RuntimeError):
                pass
    if lr is None and cycle >= 2:
        lr = 3e-5
        log(f"  Cycle {cycle}: Using adaptive fine-tuning LR=3e-5 (cycle>=2)")
    adaptive_freeze = getattr(config, "retrain_adaptive_freeze", False) if config is not None else False
    freeze = getattr(config, "retrain_freeze_layers", 0) if config is not None else 0
    if adaptive_freeze:
        log(f"  Cycle {cycle}: Adaptive freeze ENABLED — system will determine optimal freeze depth")
    elif freeze > 0:
        log(f"  Cycle {cycle}: Freezing {freeze} blocks (config override)")
    elif cycle >= 3 and best_level < 5:
        freeze = 2
        log(f"  Cycle {cycle}: Auto-freezing 2 blocks to prevent forgetting")
    elif cycle >= 3 and best_level >= 5:
        log(f"  Cycle {cycle}: Best level={best_level}, skipping freeze to learn L5+ patterns")
    compile_mode = None
    wd = getattr(config, "retrain_weight_decay", None) if config is not None else None
    if wd is not None and wd < 0:
        log(f"  Cycle {cycle}: WARNING retrain_weight_decay={wd} < 0, ignoring")
        wd = None
    grad_clip = getattr(config, "retrain_grad_clip", None) if config is not None else None
    if grad_clip is not None and grad_clip < 0:
        log(f"  Cycle {cycle}: WARNING retrain_grad_clip={grad_clip} < 0, ignoring")
        grad_clip = None
    patience = getattr(config, "retrain_patience", None) if config is not None else None
    if patience is not None and patience < 1:
        log(f"  Cycle {cycle}: WARNING retrain_patience={patience} < 1, ignoring")
        patience = None
    bs = getattr(config, "retrain_batch_size", 8) if config is not None else 8
    effective_bs = max(1, min(bs, 16))
    if effective_bs != bs:
        log(f"  Cycle {cycle}: Batch size {bs} capped to {effective_bs} (GPU memory limit)")
    max_epochs = getattr(config, "retrain_max_epochs", None) if config is not None else None
    if best_level >= 17:
        max_epochs = 12
        log(f"  Cycle {cycle}: L17+ retrain boost: max_epochs=12 (best_level={best_level})")
    elif best_level >= 5:
        max_epochs = 8
        log(f"  Cycle {cycle}: Level-based epoch boost: capped at 8 (best_level={best_level})")
    val_split = getattr(config, "retrain_val_split", None) if config is not None else None
    if val_split is None:
        val_split = 0.15
    max_per_level = getattr(config, "retrain_max_per_level", None) if config is not None else None
    if max_per_level is None and cycle >= 2:
        max_per_level = 500
    if best_level >= 17 and (max_per_level is None or max_per_level < 800):
        max_per_level = 800
        log(f"  Cycle {cycle}: L17+ retrain boost: max_per_level=800")
    grad_accum_steps = getattr(config, "retrain_grad_accum_steps", 2) if config is not None else 2
    lr_scheduler = getattr(config, "retrain_lr_scheduler", "plateau") if config is not None else "plateau"
    swa_start = getattr(config, "retrain_swa_start", None) if config is not None else None
    swa_lr = getattr(config, "retrain_swa_lr", None) if config is not None else None
    train_timeout = getattr(config, "retrain_timeout", 3600) if config is not None else 3600
    return RetrainRunConfig(
        model_size=model_size,
        lr=lr,
        adaptive_freeze=adaptive_freeze,
        freeze=freeze,
        compile_mode=compile_mode,
        wd=wd,
        grad_clip=grad_clip,
        patience=patience,
        max_epochs=max_epochs,
        val_split=val_split,
        max_per_level=max_per_level,
        grad_accum_steps=grad_accum_steps,
        lr_scheduler=lr_scheduler,
        swa_start=swa_start,
        swa_lr=swa_lr,
        effective_bs=effective_bs,
        train_timeout=train_timeout,
    )


def _build_retrain_subprocess_cmd(
    cycle: int,
    checkpoint_path: str,
    recurrent_data: str,
    train_script: str,
    run_cfg: RetrainRunConfig,
    best_level: int,
) -> list[str]:
    import sys as _sys

    cuda_python = _sys.executable or "python"
    cmd = [
        cuda_python,
        train_script,
        "--resume",
        checkpoint_path,
        "--recurrent-data",
        recurrent_data,
        "--model-size",
        run_cfg.model_size,
    ]
    if run_cfg.lr is not None:
        cmd.extend(["--lr", str(run_cfg.lr)])
    if run_cfg.adaptive_freeze:
        cmd.extend(["--adaptive-freeze"])
    elif run_cfg.freeze > 0:
        cmd.extend(["--freeze-layers", str(run_cfg.freeze)])
    if run_cfg.wd is not None:
        cmd.extend(["--weight-decay", str(run_cfg.wd)])
        log(f"  Cycle {cycle}: Weight decay={run_cfg.wd} (config override)")
    if run_cfg.grad_clip is not None:
        cmd.extend(["--grad-clip", str(run_cfg.grad_clip)])
        log(f"  Cycle {cycle}: Grad clip={run_cfg.grad_clip} (config override)")
    if run_cfg.patience is not None:
        cmd.extend(["--patience", str(run_cfg.patience)])
        log(f"  Cycle {cycle}: Patience={run_cfg.patience} (config override)")
    cmd.extend(["--batch-size", str(run_cfg.effective_bs)])
    if run_cfg.max_epochs is not None:
        cmd.extend(["--max-epochs", str(run_cfg.max_epochs)])
    if run_cfg.val_split is not None:
        cmd.extend(["--val-split", str(run_cfg.val_split)])
    if run_cfg.max_per_level is not None:
        cmd.extend(["--max-per-level", str(run_cfg.max_per_level)])
    if run_cfg.grad_accum_steps > 1:
        cmd.extend(["--grad-accum-steps", str(run_cfg.grad_accum_steps)])
        log(f"  Cycle {cycle}: Gradient accumulation: {run_cfg.grad_accum_steps} steps")
    if run_cfg.lr_scheduler != "plateau":
        cmd.extend(["--lr-scheduler", run_cfg.lr_scheduler])
        log(f"  Cycle {cycle}: LR scheduler: {run_cfg.lr_scheduler}")
    if run_cfg.swa_start is not None:
        cmd.extend(["--swa-start", str(run_cfg.swa_start)])
        if run_cfg.swa_lr is not None:
            cmd.extend(["--swa-lr", str(run_cfg.swa_lr)])
        log(f"  Cycle {cycle}: SWA averaging from epoch {run_cfg.swa_start}")
    return cmd


def _build_retrain_train_kwargs(
    checkpoint_path: str,
    recurrent_data: str,
    run_cfg: RetrainRunConfig,
    config: Any,
) -> dict[str, Any]:
    train_kwargs: dict[str, Any] = {
        "rebuild_dataset": False,
        "resume_path": checkpoint_path,
        "recurrent_data_path": recurrent_data,
        "model_size": run_cfg.model_size,
    }
    if run_cfg.lr:
        train_kwargs["lr_override"] = run_cfg.lr
    if run_cfg.freeze > 0:
        train_kwargs["freeze_layers"] = run_cfg.freeze
    if run_cfg.adaptive_freeze:
        train_kwargs["adaptive_freeze"] = True
        train_kwargs["freeze_layers"] = 0
    if run_cfg.effective_bs:
        train_kwargs["batch_size"] = run_cfg.effective_bs
    if run_cfg.wd:
        train_kwargs["weight_decay"] = run_cfg.wd
    if run_cfg.patience:
        train_kwargs["patience"] = run_cfg.patience
    if run_cfg.grad_clip:
        train_kwargs["grad_clip"] = run_cfg.grad_clip
    if run_cfg.max_epochs:
        train_kwargs["max_epochs"] = run_cfg.max_epochs
    if run_cfg.val_split:
        train_kwargs["val_split"] = run_cfg.val_split
    if run_cfg.max_per_level:
        train_kwargs["max_per_level"] = run_cfg.max_per_level
    if run_cfg.grad_accum_steps > 1:
        train_kwargs["grad_accum_steps"] = run_cfg.grad_accum_steps
    if run_cfg.lr_scheduler != "plateau":
        train_kwargs["lr_scheduler"] = run_cfg.lr_scheduler
    if run_cfg.swa_start is not None:
        train_kwargs["swa_start"] = run_cfg.swa_start
        if run_cfg.swa_lr is not None:
            train_kwargs["swa_lr"] = run_cfg.swa_lr
    aug_k = getattr(config, "augmentation_k", 8) if config is not None else 8
    train_kwargs["augmentation_k"] = aug_k
    use_amp = not getattr(config, "no_amp", False) if config is not None else True
    train_kwargs["amp"] = use_amp
    train_kwargs["compile_mode"] = run_cfg.compile_mode
    return train_kwargs


def _cleanup_cuda_after_in_process_retrain() -> None:
    try:
        import gc as _gc

        import torch as _torch

        _gc.collect()
        _torch.cuda.empty_cache()
        _torch.cuda.reset_peak_memory_stats()
    except (RuntimeError, ImportError, AttributeError):
        pass


def _retry_gpu_retrain_after_cuda_error(
    cycle: int,
    train_mod: Any,
    train_kwargs: dict[str, Any],
    compile_mode: Any,
    cuda_err_str: str,
) -> bool:
    cudagraph_err = "CUDAGraph" in cuda_err_str or "cudagraph" in cuda_err_str.lower()
    _is_cuda_alloc = "cuda" in cuda_err_str.lower() or "cudacaching" in cuda_err_str.lower()
    _is_cuda_internal = "INTERNAL ASSERT FAILED" in cuda_err_str or "handles_" in cuda_err_str
    is_device_not_ready = "device not ready" in cuda_err_str.lower()
    is_cudagraph = cudagraph_err and compile_mode is not None
    if is_cudagraph and not is_device_not_ready:
        log(f"  Cycle {cycle}: CUDAGraphs error with compile_mode={compile_mode} -- retrying without compile")
        train_kwargs["compile_mode"] = None
        try:
            train_mod.train(**train_kwargs)
            return True
        except RuntimeError as retry_err:
            cuda_err_str = str(retry_err)
            log(f"  Cycle {cycle}: CUDAGraph retry also failed: {retry_err}")
    retries_left = 3
    while retries_left > 0:
        retries_left -= 1
        cooldown = 10 - retries_left * 3
        log(
            f"  Cycle {cycle}: GPU error ({cuda_err_str[:80]}...) — resetting GPU "
            f"(retries left={retries_left}, cooldown={cooldown}s)"
        )
        import gc as _gc

        import torch

        _gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        time.sleep(cooldown)
        try:
            train_mod.train(**train_kwargs)
            log(f"  Cycle {cycle}: GPU retrain succeeded after cooldown")
            return True
        except RuntimeError as retry_err:
            cuda_err_str = str(retry_err)
            if retries_left == 0:
                log(f"  Cycle {cycle}: GPU error persists after 3 retries — falling back to CPU")
    return False


def _run_cpu_retrain_fallback(
    cycle: int,
    train_mod: Any,
    train_kwargs: dict[str, Any],
    retrain_log: str,
) -> tuple[bool, dict[str, Any] | None]:
    log(f"  Cycle {cycle}: GPU retrain failed — falling back to CPU training")
    try:
        cpu_log = open(retrain_log + ".cpu", "w", buffering=1, encoding="utf-8")  # noqa: SIM115
        old_stdout2 = sys.stdout
        old_stderr2 = sys.stderr
        sys.stdout = cpu_log
        sys.stderr = cpu_log
        try:
            train_kwargs["amp"] = False
            train_mod.train(**train_kwargs)
            log(f"  Cycle {cycle}: CPU retrain completed successfully")
            return True, None
        except (RuntimeError, ValueError, TypeError) as cpu_err:
            log(f"  Cycle {cycle}: CPU retrain also failed: {cpu_err}")
            return False, {"retrained": False, "reason": f"cpu_fallback_error: {cpu_err}"}
        finally:
            sys.stdout = old_stdout2
            sys.stderr = old_stderr2
            cpu_log.close()
    except (OSError, ValueError, ImportError) as fallback_err:
        log(f"  Cycle {cycle}: CPU fallback setup failed: {fallback_err}")
        return False, {"retrained": False, "reason": f"cpu_fallback_setup: {fallback_err}"}


def _execute_in_process_retrain(
    cycle: int,
    train_mod: Any,
    train_kwargs: dict[str, Any],
    retrain_log: str,
    compile_mode: Any,
) -> tuple[bool, dict[str, Any] | None]:
    """Run in-process train() with log capture, CUDA retries, and CPU fallback."""
    log(f"  Cycle {cycle}: Launching subprocess retrain (fresh CUDA context)...")
    sys.stdout.flush()
    sys.stderr.flush()
    log_f = open(retrain_log, "w", buffering=1, encoding="utf-8")  # noqa: SIM115
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = log_f
    sys.stderr = log_f
    retrain_ok = False
    fatal_result: dict[str, Any] | None = None
    try:
        log(f"  Cycle {cycle}: Starting in-process training...")
        train_mod.train(**train_kwargs)
        retrain_ok = True
        log(f"  Cycle {cycle}: In-process training completed successfully")
        _cleanup_cuda_after_in_process_retrain()
    except RuntimeError as cuda_err:
        cuda_err_str = str(cuda_err)
        cudagraph_err = "CUDAGraph" in cuda_err_str or "cudagraph" in cuda_err_str.lower()
        is_cuda_alloc = "cuda" in cuda_err_str.lower() or "cudacaching" in cuda_err_str.lower()
        is_cuda_internal = "INTERNAL ASSERT FAILED" in cuda_err_str or "handles_" in cuda_err_str
        is_device_not_ready = "device not ready" in cuda_err_str.lower()
        is_cudagraph = cudagraph_err and compile_mode is not None
        is_retryable = is_cuda_alloc or is_cuda_internal or is_device_not_ready or is_cudagraph
        if is_retryable:
            if _retry_gpu_retrain_after_cuda_error(cycle, train_mod, train_kwargs, compile_mode, cuda_err_str):
                retrain_ok = True
                _cleanup_cuda_after_in_process_retrain()
        else:
            log(f"  Cycle {cycle}: Training failed with non-retryable error: {cuda_err}")
            fatal_result = {"retrained": False, "reason": f"fatal_error: {cuda_err}"}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_f.close()
    if not retrain_ok and fatal_result is None:
        cpu_ok, cpu_result = _run_cpu_retrain_fallback(cycle, train_mod, train_kwargs, retrain_log)
        if cpu_ok:
            retrain_ok = True
        elif cpu_result is not None:
            fatal_result = cpu_result
    return retrain_ok, fatal_result


def _resolve_retrain_output_checkpoint(cycle: int) -> str:
    cycle_ckpt = os.path.join(M1_CHECKPOINTS, f"checkpoint_cycle_{cycle}.pt")
    best_ckpt = os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt")
    final_ckpt = os.path.join(M1_CHECKPOINTS, "model_final.pt")
    current_ckpt = os.path.join(M1_CHECKPOINTS, "checkpoint_current.pt")
    new_checkpoint = current_ckpt
    source_ckpt = None
    if os.path.isfile(cycle_ckpt):
        source_ckpt = cycle_ckpt
        new_checkpoint = cycle_ckpt
        log(
            f"  Cycle {cycle}: Using cycle checkpoint {os.path.basename(cycle_ckpt)} "
            f"({os.path.getsize(cycle_ckpt) / 1e6:.1f}MB, with optimizer state)"
        )
    elif os.path.isfile(best_ckpt):
        source_ckpt = best_ckpt
        new_checkpoint = best_ckpt
        log(
            f"  Cycle {cycle}: Using checkpoint_best.pt "
            f"({os.path.getsize(best_ckpt) / 1e6:.1f}MB, with optimizer state)"
        )
    elif os.path.isfile(final_ckpt):
        source_ckpt = final_ckpt
        new_checkpoint = final_ckpt
        log(
            f"  Cycle {cycle}: Falling back to model_final.pt "
            f"({os.path.getsize(final_ckpt) / 1e6:.1f}MB, no optimizer state)"
        )
    if source_ckpt and os.path.isfile(source_ckpt):
        try:
            shutil.copy2(source_ckpt, current_ckpt)
            log(
                f"  Cycle {cycle}: Full checkpoint preserved: {current_ckpt} "
                f"({os.path.getsize(current_ckpt) / 1e6:.1f}MB)"
            )
        except (OSError, PermissionError) as e:
            log(f"  Cycle {cycle}: Failed to preserve full checkpoint: {e}")
    return new_checkpoint


def _parse_retrain_val_loss(retrain_log: str, new_checkpoint: str) -> float | None:
    parsed_val_loss = None
    try:
        parsed_val_loss = parse_val_loss_from_retrain_log(retrain_log)
    except (OSError, ValueError, KeyError):
        pass
    if parsed_val_loss is None and os.path.isfile(new_checkpoint):
        try:
            import torch as _vtorch

            _ckpt = _vtorch.load(new_checkpoint, map_location="cpu", weights_only=False)
            _ckpt_vl = _ckpt.get("val_loss")
            if _ckpt_vl is not None and isinstance(_ckpt_vl, (int, float)) and _ckpt_vl > 0:
                parsed_val_loss = float(_ckpt_vl)
        except (OSError, KeyError, ValueError, RuntimeError):
            pass
    return parsed_val_loss


def retrain_model(cycle: int, checkpoint_path: str, config: Any = None, best_level: int = 0) -> dict[str, Any]:
    """Retrain the seed model on accumulated recurrent data.

    Accepts an optional EvolutionConfig to override retrain hyperparameters.
    Uses adaptive defaults: cycle 1 gets full LR, cycles 2+ get lower LR
    and frozen early layers to prevent catastrophic forgetting.

    When best_level >= 5 (champion has reached L5+ levels), auto-freeze is
    disabled to allow the model to learn higher-level instruction patterns
    instead of protecting lower-level knowledge.

    Returns training metrics (loss, epochs, etc.).
    """
    log(f"  Cycle {cycle}: Retraining model...")

    checkpoint_path, backup_checkpoint = _resolve_retrain_checkpoint(cycle, checkpoint_path)
    skip_result, example_count = _validate_retrain_eligibility(cycle, config)
    if skip_result is not None:
        return skip_result

    recurrent_data = M1_RECURRENT_DATA
    log(f"  Cycle {cycle}: Retraining on {example_count} examples, using checkpoint: {checkpoint_path}")

    device = "cuda" if has_gpu() else "cpu"
    log(f"  Cycle {cycle}: Training device: {device.upper()}")

    train_script = os.path.join(SCRIPT_DIR, "milestone1", "train_model.py")
    retrain_log = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")

    run_cfg = _resolve_retrain_run_config(cycle, config, best_level)
    cmd = _build_retrain_subprocess_cmd(cycle, checkpoint_path, recurrent_data, train_script, run_cfg, best_level)
    log_cmd = " ".join(cmd[:6]) + " [...]"
    log(f"  Cycle {cycle}: Command: {log_cmd}")
    log(f"  Cycle {cycle}: Retrain timeout: {run_cfg.train_timeout}s")

    _kill_orphan_train_model_processes(cycle)

    start = time.time()
    try:
        _cooldown_gpu_before_retrain(cycle)

        # -- In-process training (Hermes terminal kills subprocesses) --
        #   The Hermes terminal tool sends SIGTERM to any child process
        #   running >~15-20s, regardless of shell=True, start_new_session,
        #   double-fork, or os.system().  The ONLY reliable approach is to
        #   import train_model.train() and call it directly in the same
        #   process.  This avoids child-process signal issues entirely.
        import importlib

        train_mod = importlib.import_module("train_model")
        os.makedirs(os.path.dirname(retrain_log), exist_ok=True)

        train_kwargs = _build_retrain_train_kwargs(checkpoint_path, recurrent_data, run_cfg, config)
        compile_mode = run_cfg.compile_mode
        use_amp = train_kwargs.get("amp", True)
        log(f"  Cycle {cycle}: AMP={'disabled' if not use_amp else 'enabled'}, " f"torch.compile={compile_mode}")

        train_kwargs["output_path"] = os.path.join(M1_CHECKPOINTS, f"checkpoint_cycle_{cycle}.pt")
        retrain_ok, fatal_result = _execute_in_process_retrain(
            cycle, train_mod, train_kwargs, retrain_log, compile_mode
        )
        if fatal_result is not None:
            return fatal_result
        if not retrain_ok:
            return {"retrained": False, "reason": "retrain_failed"}

    except FileNotFoundError:
        log(f"  Cycle {cycle}: train_model module not found")
        return {"retrained": False, "reason": "train_module_not_found"}
    except ImportError as e:
        log(f"  Cycle {cycle}: Failed to import train_model: {e}")
        return {"retrained": False, "reason": f"import_error: {e}"}
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        log(f"  Cycle {cycle}: Retrain failed: {type(e).__name__}: {e}")
        import traceback

        log(f"  Traceback: {''.join(traceback.format_tb(e.__traceback__))}")
        return {"retrained": False, "reason": f"exception: {e}"}
    elapsed = time.time() - start
    log(f"  Cycle {cycle}: Retrain completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    new_checkpoint = _resolve_retrain_output_checkpoint(cycle)
    parsed_val_loss = _parse_retrain_val_loss(retrain_log, new_checkpoint)

    return {
        "retrained": True,
        "checkpoint": new_checkpoint,
        "elapsed_seconds": round(elapsed, 2),
        "example_count": example_count,
        "exit_code": 0,
        "backup_checkpoint": backup_checkpoint,
        "val_loss": parsed_val_loss,
    }


def cycle_resume_state(cycle: int) -> dict[str, Any]:
    """Check if a cycle has already been completed (resume support).

    R24: Reads the results file once, tracking both the completed set
    and the full results list in a single pass. Stops reading as soon
    as the target cycle is found (early termination for large files).

    R47: Also extracts the highest champion level and fitness from the
    results file, so --resume can restore curriculum state across
    subprocess restarts (prevents level regression L17ΓåÆL3).
    """
    results = []
    completed_cycles: set[int] = set()
    found = False
    best_level = 0
    best_fitness = 0.0
    last_fitness = 0.0
    last_level = 0
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        results.append(r)
                        c = r.get("cycle")
                        if c is not None:
                            # PM Rule #17: Only credit cycles that completed
                            # successfully (no error, elapsed >= MIN_CYCLE_SECONDS)
                            if _is_valid_cycle_result(r):
                                completed_cycles.add(c)
                            if c == cycle:
                                found = _is_valid_cycle_result(r)
                        # Track highest level and fitness across all cycles
                        rl = r.get("final_level", 0)
                        rf = r.get("final_fitness", 0.0)
                        best_level = max(best_level, rl)
                        best_fitness = max(best_fitness, rf)
                        last_fitness = rf
                        last_level = rl
                    except json.JSONDecodeError:
                        continue
    return {
        "completed": found or cycle in completed_cycles,
        "all_results": results,
        "best_level": best_level,
        "best_fitness": best_fitness,
        "last_level": last_level,
        "last_fitness": last_fitness,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a summary table of all cycles."""
    log("")
    log("=" * 90)
    log("MULTI-CYCLE RECURRENT FEEDBACK -- SUMMARY")
    log("=" * 90)

    header = f"{'Cycle':>6} | {'Fitness':>8} | {'Lvl':>4} | {'Words':>6} | {'Gens':>5} | {'Traces':>7} | {'Novelty':>8} | {'QD%':>5} | {'Retrain':>8} | {'Time':>7}"
    sep = "-" * 90
    log(header)
    log(sep)

    for r in results:
        retrain_str = "OK" if r.get("retrained") else "--"
        qd_pct = (r.get("qd_occupied_cells", 0) / max(r.get("qd_total_cells", 1), 1)) * 100
        log(
            f"{r.get('cycle', 0):>6} | "
            f"{r.get('final_fitness', 0):>8.3f} | "
            f"{r.get('final_level', 0):>4} | "
            f"{r.get('champion_words', 0):>6} | "
            f"{r.get('generations_completed', 0):>5} | "
            f"{r.get('champion_traces', 0):>7} | "
            f"{r.get('novelty_archive_size', 0):>8} | "
            f"{qd_pct:>4.0f}% | "
            f"{retrain_str:>8} | "
            f"{r.get('elapsed_seconds', 0):>6.0f}s"
        )

    # Trend analysis
    if len(results) >= 3:
        first = results[0]
        last = results[-1]
        fit_change = last.get("final_fitness", 0) - first.get("final_fitness", 0)
        lvl_change = last.get("final_level", 0) - first.get("final_level", 0)
        trace_change = last.get("champion_traces", 0) - first.get("champion_traces", 0)
        novelty_change = last.get("novelty_archive_size", 0) - first.get("novelty_archive_size", 0)

        log(sep)
        log(f"Trend (C{first.get('cycle', 0)} -> C{last.get('cycle', 0)}):")
        log(
            f"  Fitness:    {first.get('final_fitness', 0):.3f} -> {last.get('final_fitness', 0):.3f} ({fit_change:+.3f})"
        )
        log(f"  Level:      {first.get('final_level', 0)} -> {last.get('final_level', 0)} ({lvl_change:+d})")
        log(f"  Traces:     {first.get('champion_traces', 0)} -> {last.get('champion_traces', 0)} ({trace_change:+d})")
        log(
            f"  Novelty:    {first.get('novelty_archive_size', 0)} -> {last.get('novelty_archive_size', 0)} ({novelty_change:+d})"
        )
    log("=" * 90)


def _make_parser() -> argparse.ArgumentParser:
    """Create the argument parser (exposed for testing)."""
    parser = argparse.ArgumentParser(description="Multi-Cycle Recurrent Feedback Runner")
    parser.add_argument("--cycles", type=int, default=5, help="Number of evolution->retrain cycles (default: 5)")
    parser.add_argument("--gens", type=int, default=100, help="Generations per evolution run (default: 100)")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed cycle")
    parser.add_argument("--clean-state", action="store_true", help="Delete saved state files before starting")
    parser.add_argument("--no-retrain", action="store_true", help="Skip retraining step (just run evolution cycles)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Starting checkpoint (default: model_45m.pt)")
    # Retrain parameterization CLI args
    parser.add_argument("--retrain-lr", type=float, default=None, help="Override retrain learning rate")
    parser.add_argument(
        "--retrain-freeze-layers", type=int, default=0, help="Freeze N transformer blocks during retrain (0=no freeze)"
    )
    parser.add_argument(
        "--retrain-adaptive-freeze",
        action="store_true",
        default=False,
        help="Enable adaptive freeze during retrain ΓÇö dynamically freezes converged blocks "
        "based on gradient norms instead of static freeze_layers",
    )
    parser.add_argument(
        "--retrain-max-per-level", type=int, default=None, help="Max recurrent examples per curriculum level"
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        help="Cap curriculum level for bisection experiments (e.g. --max-level 25)",
    )
    parser.add_argument(
        "--retrain-model-size",
        type=str,
        default="xlarge",
        choices=["small", "medium", "large", "xlarge"],
        help="Model size for retrain",
    )
    parser.add_argument("--retrain-timeout", type=int, default=3600, help="Timeout per retrain subprocess (seconds)")
    parser.add_argument("--retrain-patience", type=int, default=None, help="Early stopping patience for retrain")
    parser.add_argument("--retrain-max-epochs", type=int, default=None, help="Max training epochs per retrain")
    parser.add_argument("--retrain-grad-clip", type=float, default=None, help="Gradient clipping norm for retrain")
    parser.add_argument("--retrain-batch-size", type=int, default=None, help="Batch size for retrain")
    parser.add_argument("--retrain-weight-decay", type=float, default=None, help="Weight decay for retrain optimizer")
    # v24: Gradient accumulation (Pivot 1)
    parser.add_argument(
        "--retrain-grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for retrain (default: 1, effective batch=batch_size ├ù N)",
    )
    # v24: LR scheduler (Pivot 2)
    parser.add_argument(
        "--retrain-lr-scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine-warm"],
        help="LR scheduler for retrain: 'plateau' (ReduceLROnPlateau) or 'cosine-warm' (CosineAnnealingWarmRestarts)",
    )
    # v24: SWA (Pivot 4)
    parser.add_argument(
        "--retrain-swa-start", type=int, default=None, help="Start SWA averaging at this epoch during retrain (e.g. 30)"
    )
    parser.add_argument(
        "--retrain-swa-lr", type=float, default=None, help="Learning rate for SWA epochs during retrain"
    )
    parser.add_argument(
        "--force-retrain-interval",
        type=int,
        default=3,
        help="Force retrain every N cycles even if champion unchanged (0=disable, default: 3)",
    )
    parser.add_argument(
        "--skip-retrain-if-no-new-data",
        action="store_true",
        default=False,
        help="Skip retrain if recurrent_data.jsonl hasn't changed since last retrain (mtime check)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        default=False,
        help="Disable AMP (mixed precision) for retrain ΓÇö use FP32 on CUDA. Fixes WSL2 CUDA deadlock.",
    )
    return parser


def _build_retrain_config(args: argparse.Namespace) -> Any:
    """Build a minimal config-like object from CLI args for retrain params.

    Returns None if no overrides are specified (backward compatible).
    """
    # Check if any retrain override was provided
    retrain_args = [
        args.retrain_lr,
        args.retrain_freeze_layers,
        args.retrain_max_per_level,
        args.retrain_model_size,
        args.retrain_timeout,
        args.retrain_patience,
        args.retrain_max_epochs,
        args.retrain_grad_clip,
        args.retrain_batch_size,
        args.retrain_weight_decay,
    ]
    if not args.retrain_adaptive_freeze and all(
        v is None or v == 0 or v == "xlarge" or v == 3600 for v in retrain_args
    ):
        return None  # No overrides -- backward compatible

    # Create a simple namespace for retrain params
    import types

    cfg = types.SimpleNamespace()
    cfg.retrain_lr = args.retrain_lr
    cfg.retrain_freeze_layers = args.retrain_freeze_layers
    cfg.retrain_max_per_level = args.retrain_max_per_level
    cfg.retrain_model_size = args.retrain_model_size
    cfg.retrain_timeout = args.retrain_timeout
    cfg.retrain_patience = args.retrain_patience
    cfg.retrain_max_epochs = args.retrain_max_epochs
    cfg.retrain_grad_clip = args.retrain_grad_clip
    cfg.retrain_batch_size = args.retrain_batch_size
    cfg.retrain_weight_decay = args.retrain_weight_decay
    cfg.retrain_min_examples = 1  # default (training adds 51K synthetic + ref data anyway)
    cfg.retrain_val_split = None
    cfg.retrain_adaptive_freeze = args.retrain_adaptive_freeze
    # v24: Propagate pivot args to config
    cfg.retrain_swa_start = args.retrain_swa_start
    cfg.retrain_swa_lr = getattr(args, "retrain_swa_lr", None)
    cfg.retrain_grad_accum_steps = args.retrain_grad_accum_steps
    cfg.retrain_lr_scheduler = args.retrain_lr_scheduler
    cfg.no_amp = getattr(args, "no_amp", False)
    return cfg


@dataclass
class MultiCycleContext:
    """Mutable state for a multi-cycle run (setup + per-cycle loop)."""

    args: argparse.Namespace
    checkpoint: str
    retrain_config: Any
    completed_cycles: set[int] = field(default_factory=set)
    all_results: list[dict[str, Any]] = field(default_factory=list)
    resume_best_level: int = 0
    meta_controller: Any = None
    barrier_tuner: Any = None
    prometheus_server: Any = None
    tuning_policy: Any = None
    tuning_actuator: Any = None
    last_archive_params: dict[str, Any] = field(default_factory=dict)
    last_retrain_epochs: int = 0
    last_champion_fitness: float = float("-inf")
    last_champion_level: int = 0
    last_champion_words: int = 0
    cycles_since_last_retrain: int = 0
    post_retrain_boost_cycles: int = 0
    prev_cycle_metrics: dict[str, Any] = field(default_factory=dict)
    prev_archive_state: dict[str, Any] | None = None
    progress_controller: Any = None
    skip_evolution: bool = False
    cycle_start: int = 1


def _validate_main_args(args: argparse.Namespace) -> None:
    if args.cycles < 1:
        log("FATAL: --cycles must be >= 1")
        sys.exit(1)
    if args.gens < 1:
        log("FATAL: --gens must be >= 1")
        sys.exit(1)
    if args.gens > 10000:
        log("FATAL: --gens too large (max 10000)")
        sys.exit(1)
    max_cycles = 100
    if args.cycles > max_cycles:
        log(f"FATAL: --cycles too large (max {max_cycles})")
        sys.exit(1)


def _log_startup_banner(args: argparse.Namespace) -> None:
    if _gpu_eval_enabled():
        os.environ.setdefault("EE_TRAIN_DEVICE", "cuda")
    log("Multi-Cycle Recurrent Feedback Runner")
    log(f"  Cycles:        {args.cycles}")
    log(f"  Gen/cycle:     {args.gens}")
    log(f"  EE_USE_GPU:    {os.environ.get('EE_USE_GPU', '1')}")
    log(f"  EE_TRAIN_DEVICE: {os.environ.get('EE_TRAIN_DEVICE', 'cuda')}")
    log(f"  PyTorch CUDA (train): {has_gpu()}")
    log(f"  CuPy CUDA (VM eval):  {has_cupy_cuda()}")
    try:
        from gpu_vm_farm import get_gpu_vram_info

        vram = get_gpu_vram_info()
        if vram.get("available"):
            log(
                f"  GPU device:    {vram.get('device_name', 'N/A')} "
                f"({vram.get('used_mb', 0):.0f}/{vram.get('total_mb', 0):.0f} MB VRAM used)"
            )
    except ImportError:
        pass
    log(f"  PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")
    log("  Torch compile:  disabled (CUDA 13.0 stability)")
    log("  Memory cleanup: torch.cuda.empty_cache() each cycle")
    log(f"  CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING', '0')}")
    log(f"  Resume:        {args.resume}")
    log(f"  Starting ckpt: {args.checkpoint or 'checkpoint_current.pt'}")


def _resolve_start_checkpoint(args: argparse.Namespace) -> str:
    candidates: list[str] = []
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = [
            os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt"),
            os.path.join(M1_CHECKPOINTS, "checkpoint_current.pt"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                checkpoint = c
                break
    if not checkpoint or not os.path.isfile(checkpoint):
        log(f"FATAL: Checkpoint not found. Checked: {candidates}")
        sys.exit(1)
    return checkpoint


def _log_retrain_config(args: argparse.Namespace, retrain_config: Any, checkpoint: str) -> None:
    if retrain_config is not None:
        log(
            f"  Retrain overrides: LR={args.retrain_lr}, freeze={args.retrain_freeze_layers}, "
            f"adaptive_freeze={args.retrain_adaptive_freeze}, "
            f"per-level={args.retrain_max_per_level}, batch={args.retrain_batch_size}, "
            f"patience={args.retrain_patience}, clip={args.retrain_grad_clip}, "
            f"epochs={args.retrain_max_epochs}, decay={args.retrain_weight_decay}, "
            f"size={args.retrain_model_size}, timeout={args.retrain_timeout}s"
        )
    else:
        log("  Retrain overrides: None (adaptive defaults by cycle)")
    log(f"  Using: {checkpoint} ({os.path.getsize(checkpoint) / 1e6:.1f}MB)")


def _apply_resume_state(ctx: MultiCycleContext) -> None:
    if not ctx.args.resume or not os.path.isfile(RESULTS_FILE):
        ctx.cycle_start = 1
        return
    try:
        from scripts.dedupe_cycle_results import dedupe

        dedupe(RESULTS_FILE)
    except Exception as exc:  # noqa: BLE001
        log(f"  Resume dedupe skipped: {exc!r}")
    from evolution.archive_persistence import load_archive_sidecar
    from evolution.progress_controller import ProgressController

    last_retrained_ckpt = None
    last_retrain_cycle = -1
    valid_for_sustained: list[dict[str, Any]] = []
    with open(RESULTS_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if _is_valid_cycle_result(r):
                    ctx.completed_cycles.add(r.get("cycle", -1))
                    valid_for_sustained.append(r)
                    if r.get("retrained"):
                        last_retrain_cycle = int(r.get("cycle", -1))
                ctx.all_results.append(r)
                if r.get("retrained") and r.get("checkpoint"):
                    last_retrained_ckpt = r["checkpoint"]
            except json.JSONDecodeError:
                continue
    ctx.resume_best_level = ProgressController.sustained_resume_level(valid_for_sustained)
    ctx.prev_archive_state = load_archive_sidecar(PROJECT_ROOT)
    if ctx.prev_archive_state:
        from evolution.archive_persistence import cap_strategy_archive_state
        from evolution.terminal_stagnation import terminal_certified_plateau

        last = valid_for_sustained[-1] if valid_for_sustained else {}
        if terminal_certified_plateau(
            int(last.get("final_level", 0)),
            float(last.get("champion_verify_score", 0.0)),
            float(last.get("final_fitness", 0.0)),
        ):
            cap = int(os.environ.get("EE_TERMINAL_STRATEGY_ARCHIVE_CAP", "3200"))
            ctx.prev_archive_state = cap_strategy_archive_state(ctx.prev_archive_state, cap) or ctx.prev_archive_state
    log(f"  Resuming: {len(ctx.completed_cycles)} cycles already completed")
    if valid_for_sustained:
        last = valid_for_sustained[-1]
        carried = ProgressController.carry_forward_start_level(
            int(last.get("final_level", 0)),
            float(last.get("champion_verify_score", 0.0)),
            ctx.resume_best_level,
        )
        ctx.resume_best_level = max(ctx.resume_best_level, carried)
        ctx.last_champion_fitness = float(last.get("final_fitness", 0.0))
        ctx.last_champion_level = int(last.get("final_level", 0))
        ctx.last_champion_words = int(last.get("champion_words", 0))
    if ctx.resume_best_level > 0:
        log(f"  Resume sustained best level: L{ctx.resume_best_level}")
    if ctx.prev_archive_state:
        log("  Resume archive sidecar loaded")
    if last_retrained_ckpt and os.path.isfile(last_retrained_ckpt):
        ctx.checkpoint = last_retrained_ckpt
        log(f"  Resume checkpoint: {last_retrained_ckpt} ({os.path.getsize(last_retrained_ckpt) / 1e6:.1f}MB)")
    ctx.cycle_start = _max_cycle_in_results(RESULTS_FILE) + 1
    log(f"  Global cycle counter: next cycle {ctx.cycle_start}")
    if last_retrain_cycle >= 0 and valid_for_sustained:
        latest = int(valid_for_sustained[-1].get("cycle", 0))
        ctx.cycles_since_last_retrain = max(0, latest - last_retrain_cycle)
        log(
            f"  Resume cycles_since_last_retrain: {ctx.cycles_since_last_retrain} "
            f"(last retrain C{last_retrain_cycle})"
        )


def _clean_state_if_requested(args: argparse.Namespace) -> None:
    if not args.clean_state:
        return
    import glob

    clean_dirs = {os.getcwd(), PROJECT_ROOT}
    patterns = [
        "meta_controller_state.json",
        "strategy_archive_gen*.json",
        "novelty_archive_gen*.json",
        "barrier_tuner_state.json",
    ]
    for clean_dir in sorted(clean_dirs):
        for pat in patterns:
            for f in glob.glob(os.path.join(clean_dir, pat)):
                try:
                    os.remove(f)
                    _logger_rmc.info(f"[CleanState] Removed {f}")
                except (PermissionError, OSError) as e:
                    _logger_rmc.warning(f"[CleanState] Could not remove {f}: {e}")


def _init_meta_controller() -> Any:
    if not _HAS_META_CONTROLLER:
        log("  MetaController: Not available -- running without dynamic tuning")
        return None
    meta_controller = MetaController(
        checkpoint_dir=M1_CHECKPOINTS,
        best_checkpoint_name="checkpoint_best.pt",
        log_file=os.path.join(PROJECT_ROOT, "meta_controller.log"),
    )
    if os.path.isfile(META_CONTROLLER_STATE):
        if meta_controller.load_state(META_CONTROLLER_STATE):
            log(f"  MetaController: Loaded state from {META_CONTROLLER_STATE}")
            log(
                f"  MetaController: best_val_loss={meta_controller._best_val_loss:.4f}, "
                f"best_fitness={meta_controller.best_fitness:.3f}"
            )
        else:
            log("  MetaController: No saved state -- starting fresh")
    if os.path.isfile(BEST_CHECKPOINT_PATH):
        try:
            ckpt_size = os.path.getsize(BEST_CHECKPOINT_PATH)
            log(f"  MetaController: Best checkpoint exists ({ckpt_size / 1e6:.1f}MB)")
        except OSError:
            pass
    return meta_controller


def _init_barrier_tuner(args: argparse.Namespace) -> Any:
    if not _HAS_BARRIER_TUNER:
        return None
    model_layers = 8
    try:

        ckpt_path = os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt")
        if os.path.isfile(ckpt_path):
            ckpt = load_checkpoint(ckpt_path)
            if isinstance(ckpt, dict):
                model_layers = int(ckpt.get("n_layers") or ckpt.get("config", {}).get("n_layers", 8))
    except (OSError, KeyError, ValueError, RuntimeError):
        pass  # non-fatal: will use default model_layers value
    barrier_tuner_state = os.path.join(SCRIPT_DIR, "barrier_tuner_state.json")
    barrier_tuner = BarrierTuner(state_path=barrier_tuner_state, model_n_layers=model_layers)
    if args.resume and os.path.isfile(barrier_tuner_state):
        if barrier_tuner.load_state(barrier_tuner_state):
            log(f"  BarrierTuner: Loaded state from {barrier_tuner_state}")
        else:
            log("  BarrierTuner: No saved state -- starting fresh")
    else:
        log("  BarrierTuner: Initialized (fresh)")
    log(
        f"  BarrierTuner: level_stall={barrier_tuner.level_stall_cycles}cy, "
        f"fitness_plateau={barrier_tuner.fitness_plateau_cycles}cy, "
        f"diversity_collapse<{barrier_tuner.diversity_collapse_threshold}"
    )
    return barrier_tuner


def _init_prometheus_exporter() -> Any:
    if not _HAS_PROMETHEUS:
        return None
    try:
        prometheus_port = int(os.environ.get("EE_PROMETHEUS_PORT", "9100"))
        server = create_metrics_app(port=prometheus_port)
        log(f"  Prometheus metrics exporter: http://0.0.0.0:{prometheus_port}/metrics")
        return server
    except (OSError, ValueError, ImportError, RuntimeError) as e:
        log(f"  Prometheus exporter failed to start: {e}")
        return None


def _init_tuning_policy() -> tuple[Any, Any]:
    if not _HAS_TUNING_POLICY:
        log("  TuningPolicy: Not available -- running without auto-tuning")
        return None, None
    log("  TuningPolicy: Initialized (cooldown=2 cycles)")
    return TuningPolicy(cooldown=2), ParameterActuator()


def _prepare_multi_cycle_context(args: argparse.Namespace) -> MultiCycleContext:
    _validate_main_args(args)
    _log_startup_banner(args)
    checkpoint = _resolve_start_checkpoint(args)
    retrain_config = _build_retrain_config(args)
    _log_retrain_config(args, retrain_config, checkpoint)
    ctx = MultiCycleContext(
        args=args,
        checkpoint=checkpoint,
        retrain_config=retrain_config,
    )
    _apply_resume_state(ctx)
    log(f"  Checkpoint: {ctx.checkpoint} ({os.path.getsize(ctx.checkpoint) / 1e6:.1f}MB)")
    _clean_state_if_requested(args)
    ctx.meta_controller = _init_meta_controller()
    ctx.barrier_tuner = _init_barrier_tuner(args)
    ctx.prometheus_server = _init_prometheus_exporter()
    ctx.tuning_policy, ctx.tuning_actuator = _init_tuning_policy()
    if ctx.resume_best_level > 0:
        ctx.last_champion_level = ctx.resume_best_level
        log(f"  R47: Resuming at level L{ctx.last_champion_level} (from results file)")
    from evolution.progress_controller import ProgressController

    ctx.progress_controller = ProgressController(PROJECT_ROOT, stall_threshold=2)
    ctx.progress_controller.set_logger(log)
    return ctx


def _run_cycles(ctx: MultiCycleContext) -> None:
    from run_multi_cycle_loop import execute_cycles

    execute_cycles(ctx)


def _finalize_multi_cycle_run(all_results: list[dict[str, Any]], prometheus_server: Any) -> None:
    print_summary(all_results)
    summary_path = os.path.join(PROJECT_ROOT, "multi_cycle_summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"\nSummary saved: {summary_path}")
    except (OSError, PermissionError) as e:
        log(f"Summary save failed: {e}")
    if prometheus_server is not None:
        try:
            stop_metrics_app(prometheus_server)
            log("  Prometheus exporter stopped")
        except (OSError, RuntimeError, ImportError):
            pass  # non-fatal: exporter cleanup is best-effort
    log("Done.")


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    try:
        ctx = _prepare_multi_cycle_context(args)
        _run_cycles(ctx)
        _finalize_multi_cycle_run(ctx.all_results, ctx.prometheus_server)
    except Exception as exc:
        crash_log = os.path.join(PROJECT_ROOT, ".agents", "batch_crash.log")
        os.makedirs(os.path.dirname(crash_log), exist_ok=True)
        import traceback

        with open(crash_log, "a", encoding="utf-8") as f:
            f.write(f"\n--- {datetime.now(timezone.utc).isoformat()} ---\n")
            traceback.print_exc(file=f)
        log(f"FATAL: batch crashed: {exc!r} (see {crash_log})")
        raise


# ----------------------------------------------------------------------
# v13 Phase D: Training visibility hooks
# ----------------------------------------------------------------------


def compute_forgetting_curve(
    champion: list[int],
    champion_level: int,
    vm_farm: Any,
    budget: int = 5000,
) -> dict[int, float]:
    """Evaluate the champion on all previously solved levels.

    Tests the champion's performance on levels 0 through champion_level.
    If a level that was previously solved (raw_score >= 140.0) now scores
    significantly lower, the model may have forgotten that capability.

    Returns a dict mapping level -> raw_score (0.0 = complete failure).
    """
    from tasks_rv32i import get_task_level

    scores: dict[int, float] = {}
    for level in range(champion_level + 1):
        try:
            task = get_task_level(level)
            inst = task(seed=42).generate_instance()
            inputs = inst.get("inputs", {})
            expected_regs = inst.get("expected_output", {}).get("registers", {})
            result = vm_farm.evaluate(champion, inputs, budget=budget)
            status = getattr(result, "success", True)
            getattr(result, "error", None)
            final_regs = getattr(result, "registers", {})
            instr_count = getattr(result, "cycles", 0)

            # Crash penalty
            if not status:
                scores[level] = -50.0
                continue

            # Distance-weighted register scoring (matches main fitness function)
            total = len(expected_regs)
            if total == 0:
                reg_match = 1.0
            else:
                total_score = 0.0
                for reg_idx, expected_val in expected_regs.items():
                    actual_val = final_regs.get(reg_idx, 0)
                    diff = abs(actual_val - expected_val)
                    max_diff = max(1, abs(expected_val))
                    total_score += 1.0 / (1.0 + diff / max_diff)
                reg_match = total_score / total

            bloat = max(0, (instr_count - 10) * 0.01)
            raw_score = reg_match * 100.0 - bloat
            if reg_match == 1.0:
                raw_score += 50.0
            scores[level] = raw_score
        except (TypeError, ValueError, ZeroDivisionError):
            scores[level] = 0.0

    # Log summary of forgetting curve
    perfect = sum(1 for s in scores.values() if s >= 140.0)
    partial = sum(1 for s in scores.values() if 0 < s < 140.0)
    failed = sum(1 for s in scores.values() if s == 0.0)
    from logging import info

    info(
        f"[ForgettingCurve] L0-L{champion_level}: {perfect} perfect, {partial} partial, {failed} failed"
    )  # noqa: LOG015
    return scores


def compute_weight_delta(
    before_path: str,
    after_path: str,
) -> dict[str, float]:
    """Compute per-layer L2 norms of weight deltas between two checkpoints.

    Returns a dict mapping parameter key -> L2 norm of (after - before).
    Empty dict if either checkpoint can't be loaded.
    """
    try:
        import logging

        import torch

        if not os.path.isfile(before_path):
            logging.warning(f"[WeightDelta] before_path not found: {before_path}")  # noqa: LOG015
            return {}
        if not os.path.isfile(after_path):
            logging.warning(f"[WeightDelta] after_path not found: {after_path}")  # noqa: LOG015
            return {}

        before = torch.load(before_path, map_location="cpu", weights_only=False)
        after = torch.load(after_path, map_location="cpu", weights_only=False)

        b_sd = before.get("model_state_dict", before)
        a_sd = after.get("model_state_dict", after)

        deltas: dict[str, float] = {}
        for key in b_sd:
            if key in a_sd and b_sd[key].shape == a_sd[key].shape:
                diff = (a_sd[key] - b_sd[key]).float()
                norm = diff.norm().item()
                if not math.isnan(norm):  # skip nan (from -inf - -inf comparisons)
                    deltas[key] = round(norm, 4)

        if deltas:
            max_key = max(deltas, key=lambda k: deltas[k])
            logging.info(f"[WeightDelta] {len(deltas)} params, max={deltas[max_key]:.4f} ({max_key})")  # noqa: LOG015
        return deltas
    except (OSError, KeyError, ValueError, RuntimeError):
        return {}


# ΓöÇΓöÇ Phase J: Stratified memorization sampling (extracted for testability) ΓöÇΓöÇ


def _stratified_sample_mem_examples(
    mem_examples: list[dict],
    num_samples: int,
    prefix_instructions: int = 2,
    rng=None,
) -> list[dict]:
    """Stratified sample champion programs by level.

    Ensures every level present in the data is represented in the sample.
    Takes min(min_per_level, count) from each level, then fills remaining
    quota with shuffled entries from oversampled levels.

    Args:
        mem_examples: List of champion example dicts (must have 'level' key).
        num_samples: Maximum total number of examples to return.
        prefix_instructions: Used to compute min_per_level = max(3, pi + 1).
        rng: Random number generator (defaults to Random(42)).

    Returns:
        List of sampled examples. May be shorter than num_samples if
        total available data is limited.
    """
    import logging
    import random

    if rng is None:
        rng = random.Random(42)

    if not mem_examples:
        logging.warning("[MemGen] No mem_examples found for stratified sampling")  # noqa: LOG015
        return []

    # Group by level, safely handling missing/sketchy keys
    level_groups: dict[int, list[dict]] = {}
    for ex in mem_examples:
        if not isinstance(ex, dict):
            continue
        lev = ex.get("level", 0)
        if not isinstance(lev, (int, float)):
            lev = 0
        level_groups.setdefault(int(lev), []).append(ex)

    if not level_groups:
        logging.warning("[MemGen] No valid level-grouped examples found")  # noqa: LOG015
        return []

    # Log level distribution before stratification
    level_counts = {k: len(v) for k, v in sorted(level_groups.items())}
    logging.info(  # noqa: LOG015
        f"[MemGen] Level distribution before stratification: {level_counts} ({sum(level_counts.values())} total)"
    )

    min_per_level = max(3, prefix_instructions + 1)
    stratified: list[dict] = []
    for lev in sorted(level_groups.keys()):
        group = level_groups[lev]
        rng.shuffle(group)
        take = min(min_per_level, len(group))
        stratified.extend(group[:take])

    # Log level distribution after stratification
    stratified_counts: dict[int, int] = {}
    for ex in stratified:
        lev = int(ex.get("level", 0))
        stratified_counts[lev] = stratified_counts.get(lev, 0) + 1
    logging.info(
        f"[MemGen] Level distribution after stratified sample: {stratified_counts} ({len(stratified)} taken)"
    )  # noqa: LOG015

    # Fill remaining quota with shuffled remaining examples
    # R16: Use id-based set for O(1) membership check (avoids O(N*M) list scan)
    stratified_ids = {id(ex) for ex in stratified}
    remaining = [ex for ex in mem_examples if id(ex) not in stratified_ids]
    rng.shuffle(remaining)
    filler = remaining[: max(0, num_samples - len(stratified))]
    result = stratified + filler

    logging.info(  # noqa: LOG015
        f"[MemGen] Final sample: {len(result)} examples "
        f"({len(stratified)} stratified + {len(filler)} filler) "
        f"covering {len(stratified_counts)} levels"
    )
    return result


def compute_memorization_generalization(
    checkpoint_path: str,
    recurrent_data_path: str,
    num_samples: int = 50,
    prefix_instructions: int = 2,
) -> dict[str, Any]:
    """Compute memorization vs generalization scores for a retrained model.

    **Memorization**: Load champion programs from the recurrent training data,
    take the first N instructions as a prefix, generate completions, and measure
    how well they match the actual champion continuation (next-instruction accuracy).
    Champions are sampled randomly (seeded RNG) from the training data, limited
    to ``num_samples`` to bound compute time.

    **Generalization**: Use reference solutions from L8-L13 loaded from
    ``tasks_rv32i.reference_solution()`` (verified ground-truth solutions for
    each curriculum level). For each reference solution, multiple prefix variants
    are generated at different instruction offsets to increase statistical power:
    prefix at positions 0, 1, 2, ..., up to len(ref) - prefix_instructions - 1.
    Each variant tests the model's ability to complete unseen prefixes from
    correct programs.

    Returns a dict with:
    - ``memorization_score``: accuracy on champion prefixes
    - ``generalization_score``: accuracy on reference prefixes
    - ``memorization_examples``: number of champion examples evaluated
    - ``generalization_examples``: number of reference examples evaluated
    - ``per_level_memorization``: dict mapping level -> accuracy
    - ``per_level_generalization``: dict mapping level -> accuracy
    """
    import json
    import logging

    # Validate inputs
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if prefix_instructions < 1:
        raise ValueError(f"prefix_instructions must be >= 1, got {prefix_instructions}")

    # Internal scoring function -- separated for testability
    def _score_completions(
        examples: list[dict],
        prefix_count: int,
        generator: Any,
        is_memorization: bool,
    ) -> dict[int, list[float]]:
        """Score a list of examples for next-instruction prediction accuracy.

        Each example must have keys ``prefix`` (list of hex strings),
        ``expected`` (single hex string), and ``level`` (int).
        Uses generator.complete() to produce completions and computes
        the fraction of completions whose first instruction matches expected.

        Returns dict mapping level -> list of per-example accuracy scores.
        """
        scores: dict[int, list[float]] = {}
        for ex in examples:
            level = ex["level"]
            prefix = ex["prefix"]
            expected = ex["expected"]

            try:
                completions = generator.complete(
                    prefix,
                    prefix_count=len(prefix),
                    num_samples=3,
                    temperature=0.5,
                    top_k=20,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )
                match = 0
                for prog in completions:
                    if len(prog) >= 1 and prog[0] == expected:
                        match += 1
                accuracy = match / max(len(completions), 1)
            except (TypeError, ValueError, KeyError):
                accuracy = 0.0

            scores.setdefault(level, []).append(accuracy)
        return scores

    try:
        from mutation_proposer import _encode_inst
        from prefix_generator import PrefixGenerator as PG
        from tasks_rv32i import LEVELS, reference_solution
    except ImportError as e:
        logging.warning(f"[MemGen] Missing dependencies ({e}) -- skipping diagnostic")  # noqa: LOG015
        return {}

    if not os.path.isfile(checkpoint_path):
        logging.warning(f"[MemGen] Checkpoint not found: {checkpoint_path}")  # noqa: LOG015
        return {}
    if not os.path.isfile(recurrent_data_path):
        logging.warning(f"[MemGen] No recurrent data at {recurrent_data_path}")  # noqa: LOG015
        return {}

    try:
        gen = PG(checkpoint_path, temperature=0.5, max_new_tokens=32, verbose=False)
    except (OSError, ValueError, RuntimeError, ImportError) as e:
        logging.warning(f"[MemGen] Failed to load PrefixGenerator: {e}")  # noqa: LOG015
        return {}

    # -- Memorization: sample champion programs from training data -----
    mem_examples: list[dict] = []
    try:
        with open(recurrent_data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                prog_hex = ex.get("champion_program", [])
                if isinstance(prog_hex, list) and len(prog_hex) > prefix_instructions:
                    mem_examples.append(ex)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        logging.warning(f"[MemGen] Failed to read recurrent data: {e}")  # noqa: LOG015
        return {}

    # Shuffle and limit
    import random

    rng = random.Random(42)

    # Use stratified sampling (extracted function)
    mem_examples = _stratified_sample_mem_examples(
        mem_examples,
        num_samples,
        prefix_instructions,
        rng,
    )
    if not mem_examples:
        return {}

    # Build memorization test examples (prefix + expected)
    mem_tests: list[dict] = []
    for ex in mem_examples:
        level = ex.get("level", 0)
        prog_hex = ex["champion_program"]
        prefix = prog_hex[:prefix_instructions]
        if len(prog_hex) > prefix_instructions:
            mem_tests.append(
                {
                    "level": level,
                    "prefix": prefix,
                    "expected": prog_hex[prefix_instructions],
                }
            )

    # -- Generalization: reference solutions with multiple prefix offsets -
    gen_tests: list[dict] = []
    for level in range(8, max(LEVELS.keys()) + 1):
        try:
            ref_insts = reference_solution(level)
            ref_words = []
            for inst in ref_insts:
                word = _encode_inst(inst)
                if word is not None:
                    ref_words.append(word)
            ref_words.append(0x00100073)  # EBREAK
            ref_hex = [f"0x{w:08X}" for w in ref_words]

            # Generate multiple prefix variants at different offsets
            max_offset = max(0, len(ref_hex) - prefix_instructions - 1)
            for offset in range(max_offset + 1):
                prefix = ref_hex[offset : offset + prefix_instructions]
                expected = ref_hex[offset + prefix_instructions]
                gen_tests.append(
                    {
                        "level": level,
                        "prefix": prefix,
                        "expected": expected,
                    }
                )
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            logging.warning(f"[MemGen] Failed to build ref example for L{level}: {exc}")  # noqa: LOG015
            continue

    if not gen_tests:
        logging.warning("[MemGen] No generalization examples generated")  # noqa: LOG015
        return {}

    # -- Score ----------------------------------------------------------
    mem_scores = _score_completions(mem_tests, prefix_instructions, gen, is_memorization=True)
    gen_scores = _score_completions(gen_tests, prefix_instructions, gen, is_memorization=False)

    # -- Aggregate ------------------------------------------------------
    def avg(vals: list[float]) -> float:
        return sum(vals) / max(len(vals), 1) if vals else 0.0

    result: dict[str, Any] = {
        "memorization_score": round(avg([s for scores in mem_scores.values() for s in scores]), 4),
        "generalization_score": round(avg([s for scores in gen_scores.values() for s in scores]), 4),
        "memorization_examples": len(mem_tests),
        "generalization_examples": len(gen_tests),
        "per_level_memorization": {k: round(avg(v), 4) for k, v in sorted(mem_scores.items())},
        "per_level_generalization": {k: round(avg(v), 4) for k, v in sorted(gen_scores.items())},
        "prefix_instructions": prefix_instructions,
    }

    logging.info(  # noqa: LOG015
        f"[MemGen] Mem={result['memorization_score']:.3f} "
        f"({result['memorization_examples']} ex), "
        f"Gen={result['generalization_score']:.3f} "
        f"({result['generalization_examples']} ex, "
        f"{len({e['level'] for e in gen_tests})} levels)"
    )
    return result


if __name__ == "__main__":
    main()
