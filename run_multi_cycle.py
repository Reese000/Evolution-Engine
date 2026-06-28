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

# v13: CUDA memory management for crash-resilient long runs
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
M1_CHECKPOINTS = os.path.join(PROJECT_ROOT, "milestone1", "checkpoints")
M1_RECURRENT = os.path.join(PROJECT_ROOT, "milestone1", "recurrent_data")
M1_DIR = os.path.join(PROJECT_ROOT, "milestone1")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "multi_cycle_results.jsonl")

# Structured logging
from logging_config import get_logger as _get_logger
_logger_rmc = _get_logger("RunMultiCycle")

# v11: Keep this many recent cycle checkpoints + checkpoint_best.pt
_KEEP_CYCLE_CHECKPOINTS: int = 5

# Ensure script dir is on path for imports
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, M1_DIR)
sys.path.insert(0, "/root/milestone0/logit-masking")
sys.path.insert(0, "/root/milestone0/vm-farm")
sys.path.insert(0, "/root/milestone0/gpu-vm")
sys.path.insert(0, "/root/milestone3")
sys.path.insert(0, "/root/milestone0/logit-masking")

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


# Runtime GPU check (not cached at module load -- fixes stale detection)
def has_gpu() -> bool:
    import torch

    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


# Default MetaController state path
META_CONTROLLER_STATE = os.path.join(SCRIPT_DIR, "meta_controller_state.json")
BEST_CHECKPOINT_PATH = os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt")


def log(msg: str) -> None:
    _logger_rmc.info(msg)


def run_evolution(
    cycle: int, num_generations: int, checkpoint_path: str, meta_evolution_params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run a single evolution cycle.

    Uses the given checkpoint as the seed model for program generation.
    If meta_evolution_params is provided, overrides EvolutionConfig values
    with dynamically tuned parameters from MetaController.

    Returns a metrics dict.
    """
    log(f"  Cycle {cycle}: Running evolution ({num_generations} gens)...")
    if meta_evolution_params:
        log(f"  Cycle {cycle}: Meta-tuned evolution params: {meta_evolution_params}")

    # Ensure path setup for evolution_engine import
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    import_paths = [
        os.path.join(PROJECT_ROOT, "milestone0", "logit-masking"),
        os.path.join(PROJECT_ROOT, "milestone0", "vm-farm"),
        os.path.join(PROJECT_ROOT, "milestone3"),
        M1_DIR,
    ]
    for p in import_paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

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
    config = EvolutionConfig(
        max_generations=num_generations,
        population_size=50,
        curriculum_stall_threshold=3,
        auto_curriculum=True,
        checkpoint_interval=50,
        elite_ratio=0.1,
        dashboard_state_dir=state_dir,
        log_dir=log_dir,
        champion_dir=champion_dir,
        best_champion_path=os.path.join(PROJECT_ROOT, "milestone2", "best_champion"),
        recurrent_feedback_dir=M1_RECURRENT,
        novelty_weight=0.3,
        use_novelty_archive=True,
        use_qd_archive=True,
        qd_bonus_weight=0.1,
        use_gpu=True,  # GPU VM farm for fast evolution; CuPy cleanup before retrain
    )

    # Apply MetaController-tuned evolution params if provided
    if meta_evolution_params:
        for key, val in meta_evolution_params.items():
            if hasattr(config, key):
                setattr(config, key, val)

    # Re-enable model-guided generation (PrefixGenerator now on GPU)
    # 20% of challengers come from model, 80% from mutation
    config.model_guided_ratio = 0.2

    # Set the seed model path to our current cycle's checkpoint
    engine = EvolutionEngine(config=config)

    # Override the seed model path to use our trained checkpoint
    engine._seed_model_path = os.path.relpath(checkpoint_path, SCRIPT_DIR)

    # v11.5: Apply external evolution params (from BarrierTuner) to engine
    if meta_evolution_params:
        # Filter to only evolution-relevant params
        evol_keys = {
            "mutation_rate",
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

    start = time.time()
    try:
        result = engine.run()
    except Exception as e:
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

    # Collect metrics
    champion_traces = engine.recurrent_examples_count
    novelty_size = 0
    qd_occupied = 0
    qd_total = 0
    if engine.novelty_archive is not None:
        novelty_size = engine.novelty_archive.size
    if engine.qd_archive is not None:
        qd_occupied = sum(1 for row in engine.qd_archive._grid for cell in row if cell is not None)
        qd_total = engine.qd_archive.num_bins**2

    metrics = {
        "cycle": cycle,
        "generations_completed": result.get("generations_completed", 0),
        "final_fitness": result.get("final_fitness", 0.0),
        "final_level": result.get("final_level", 0),
        "champion_words": result.get("champion_words", 0),
        # v13 Phase E.2: Include champion program for memorization diagnostic
        "champion_program": ([f"0x{w:08X}" for w in engine.champion] if engine.champion is not None else None),
        "elapsed_seconds": round(elapsed, 2),
        "champion_traces": champion_traces,
        "novelty_archive_size": novelty_size,
        "qd_occupied_cells": qd_occupied,
        "qd_total_cells": qd_total,
        "checkpoint": checkpoint_path,
        # Self-calibrate state (for cross-cycle persistence)
        "mutation_rate": getattr(engine.config, "mutation_rate", 0.3),
        "diversity_threshold": getattr(engine.config, "diversity_threshold", 0.3),
        # v11.5: Include diversity for BarrierTuner
        "diversity": engine._current_diversity,
    }

    log(
        f"  Cycle {cycle} complete: gen={metrics['generations_completed']}, "
        f"fitness={metrics['final_fitness']:.3f}, level={metrics['final_level']}, "
        f"traces={champion_traces}, novelty={novelty_size}, "
        f"qd={qd_occupied}/{qd_total}, {elapsed:.1f}s"
    )
    return metrics


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

    # -- Always retrain from the best checkpoint (with optimizer state) --
    # The evolution engine uses a stripped checkpoint (model_state_dict only).
    # For retrain, we want the full checkpoint with optimizer to preserve Adam momentum.
    # Always use checkpoint_best_full.pt to avoid chaining rejected retrain outputs.
    full_checkpoint = os.path.join(M1_CHECKPOINTS, "checkpoint_best_full.pt")
    backup_checkpoint = None
    if os.path.isfile(full_checkpoint):
        log(f"  Cycle {cycle}: Using best full checkpoint for retrain: {full_checkpoint}")
        checkpoint_path = full_checkpoint
        # Backup so we can restore on gate rejection
        backup_checkpoint = full_checkpoint + ".pre_retrain_backup"
        try:
            shutil.copy2(full_checkpoint, backup_checkpoint)
            log(f"  Cycle {cycle}: Backed up full checkpoint to: {backup_checkpoint}")
        except (OSError, PermissionError) as e:
            log(f"  Cycle {cycle}: Warning: backup failed: {e}")
            backup_checkpoint = None
    else:
        log(f"  Cycle {cycle}: Using passed checkpoint for retrain: {checkpoint_path}")

    recurrent_data = os.path.join(M1_RECURRENT, "recurrent_data.jsonl")
    if not os.path.isfile(recurrent_data):
        log(f"  Cycle {cycle}: No recurrent data found -- skipping retrain")
        return {"retrained": False, "reason": "no_data"}

    # Check there are enough examples
    with open(recurrent_data) as f:
        example_count = sum(1 for line in f if line.strip())
    min_examples = 1
    if config is not None:
        min_examples = getattr(config, "retrain_min_examples", 1)
    if example_count < min_examples:
        log(f"  Cycle {cycle}: Only {example_count} examples (< {min_examples}) -- skipping retrain")
        return {"retrained": False, "reason": f"only_{example_count}_examples"}

    # -- Inject reference solutions into recurrent data (v12.5) ---------
    # The model learns correct program patterns for high-level tasks that
    # the champion's negative-fitness traces cannot teach.  Reference solutions
    # from tasks_rv32i provide verified-correct programs for levels L8-L13.
    try:
        import hashlib

        from mutation_proposer import _encode_inst
        from tasks_rv32i import reference_solution

        from tasks_rv32i import LEVELS as _LEVELS
        max_level = max(_LEVELS.keys()) if _LEVELS else 11
        ref_levels = list(range(8, max_level + 1))  # L8-L{max} (only existing levels)
        ref_added = 0
        ref_examples = []

        for level in ref_levels:
            try:
                # Get reference solution instruction dicts
                ref_insts = reference_solution(level)
                # Encode to raw 32-bit instruction words
                ref_words = []
                for inst in ref_insts:
                    word = _encode_inst(inst)
                    if word is not None:
                        ref_words.append(word)
                # Append EBREAK terminator
                ref_words.append(0x00100073)

                if len(ref_words) <= 1:
                    continue

                # Compute SHA256 hash
                prog_bytes = b"".join(w.to_bytes(4, "little") for w in ref_words)
                prog_sha256 = hashlib.sha256(prog_bytes).hexdigest()

                ref_example = {
                    "level": level,
                    "generation": -1,  # synthetic -- not from evolution
                    "fitness": 1.0,  # reference solutions are correct
                    "champion_sha256": prog_sha256,
                    "champion_words": len(ref_words),
                    "champion_program": [f"0x{w:08X}" for w in ref_words],
                    "trace_status": "ok",
                    "trace_instr_count": len(ref_words),
                    "save_reason": "reference_solution",
                    "stalled_count": 0,
                }
                ref_examples.append(ref_example)
                ref_added += 1
            except Exception as exc:
                log(f"  Cycle {cycle}: Failed to generate reference solution L{level}: {exc}")

        if ref_examples:
            # Append reference examples to recurrent data
            with open(recurrent_data, "a") as f:
                for ex in ref_examples:
                    f.write(json.dumps(ex) + "\n")
            example_count += len(ref_examples)
            log(
                f"  Cycle {cycle}: Injected {len(ref_examples)} reference solutions "
                f"(L{ref_levels[0]}-L{ref_levels[-1]}) into retrain data "
                f"({example_count} total examples)"
            )
    except ImportError as exc:
        log(f"  Cycle {cycle}: Cannot inject reference solutions: {exc}")
    except Exception as exc:
        log(f"  Cycle {cycle}: Reference injection failed: {exc}")

    # Train via subprocess to isolate GPU memory
    log(f"  Cycle {cycle}: Retraining on {example_count} examples, using checkpoint: {checkpoint_path}")

    device = "cuda" if has_gpu() else "cpu"
    log(f"  Cycle {cycle}: Training device: {device.upper()}")

    train_script = os.path.join(SCRIPT_DIR, "..", "milestone1", "train_model.py")
    retrain_log = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")

    # -- Build command with parameterized retrain hyperparameters ------
    import sys as _sys
    cuda_python = _sys.executable or "python"
    cmd = [
        cuda_python,
        train_script,
        "--resume",
        checkpoint_path,
        "--recurrent-data",
        recurrent_data,
    ]

    # Default model size: xlarge (~115M params: 16L, d=768, d_ff=2048)
    model_size = "xlarge"
    if config is not None:
        model_size = getattr(config, "retrain_model_size", "xlarge")
    cmd.extend(["--model-size", model_size])

    # -- Learning rate --
    # Use config override if set, otherwise auto-adapt by cycle
    lr = None
    if config is not None:
        lr = getattr(config, "retrain_lr", None)
    # Safety: if we have a known good checkpoint (val_loss < 2.0), cap LR to 1e-5
    # to prevent catastrophic forgetting of already-learned patterns.
    # LR=3e-4 is appropriate for initial training but destroys a fine-tuned model
    # (val_loss jumps from 1.36 → 2.15 in epoch 1).
    if lr is not None and lr > 1e-5:
        # Check if existing checkpoint has val_loss < 2.0 (known good)
        existing_ckpt = os.path.join(M1_CHECKPOINTS, "checkpoint_best_full.pt")
        if os.path.isfile(existing_ckpt):
            try:
                import torch as _torch
                _ckpt_data = _torch.load(existing_ckpt, map_location="cpu", weights_only=False)
                _existing_val = _ckpt_data.get("val_loss", 999)
                if _existing_val < 2.0:
                    lr = 1e-5
                    log(f"  Cycle {cycle}: Capped LR from config to {lr} (existing val_loss={_existing_val:.4f} < 2.0)")
            except Exception:
                pass
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    elif cycle >= 2:
        # Low LR for later cycles to prevent catastrophic forgetting
        cmd.extend(["--lr", "3e-5"])
        log(f"  Cycle {cycle}: Using adaptive fine-tuning LR=3e-5 (cycle>=2)")

    # -- Adaptive freeze (overrides static freeze) --
    adaptive_freeze = False
    if config is not None:
        adaptive_freeze = getattr(config, "retrain_adaptive_freeze", False)
    if adaptive_freeze:
        cmd.extend(["--adaptive-freeze"])
        log(f"  Cycle {cycle}: Adaptive freeze ENABLED — system will determine optimal freeze depth")
    # -- Freeze layers (only when adaptive freeze is OFF) --
    freeze = 0
    if config is not None:
        freeze = getattr(config, "retrain_freeze_layers", 0)
    if adaptive_freeze:
        pass  # Skip static freeze
    elif freeze > 0:
        cmd.extend(["--freeze-layers", str(freeze)])
        log(f"  Cycle {cycle}: Freezing {freeze} blocks (config override)")
    elif cycle >= 3 and best_level < 5:
        # Auto-freeze 2 layers for cycles 3+ to protect learned features,
        # but ONLY if champion never reached L5+ (where new patterns matter more)
        cmd.extend(["--freeze-layers", "2"])
        log(f"  Cycle {cycle}: Auto-freezing 2 blocks to prevent forgetting")
    elif cycle >= 3 and best_level >= 5:
        # Champion reached L5+ -- no freeze, let model learn higher-level patterns
        log(f"  Cycle {cycle}: Best level={best_level}, skipping freeze to learn L5+ patterns")

    # -- AMP + compile enabled for faster training --
    # In-process training path (below) uses amp=True and compile_mode by default.
    # The subprocess cmd (above) is retained for backward compat but not used
    # in the in-process training path that is actually executed.
    # v13: compile disabled by default due to SIGSEGV on CUDA 13.0
    compile_mode = None

    # -- Weight decay (v13: configurable) --
    if config is not None:
        wd = getattr(config, "retrain_weight_decay", None)
        if wd is not None:
            if wd >= 0:
                cmd.extend(["--weight-decay", str(wd)])
                log(f"  Cycle {cycle}: Weight decay={wd} (config override)")
            else:
                log(f"  Cycle {cycle}: WARNING retrain_weight_decay={wd} < 0, ignoring")

    # -- Gradient clipping (v13: configurable) --
    if config is not None:
        gc = getattr(config, "retrain_grad_clip", None)
        if gc is not None:
            if gc >= 0:
                cmd.extend(["--grad-clip", str(gc)])
                log(f"  Cycle {cycle}: Grad clip={gc} (config override)")
            else:
                log(f"  Cycle {cycle}: WARNING retrain_grad_clip={gc} < 0, ignoring")

    # -- Patience (v13: configurable) --
    if config is not None:
        pat = getattr(config, "retrain_patience", None)
        if pat is not None:
            if pat >= 1:
                cmd.extend(["--patience", str(pat)])
                log(f"  Cycle {cycle}: Patience={pat} (config override)")
            else:
                log(f"  Cycle {cycle}: WARNING retrain_patience={pat} < 1, ignoring")

    # -- High-level reference augmentation (L11-L13 focus) --------
    ref_high = getattr(config, "ref_high_augment", None) if config is not None else None
    if ref_high is not None and ref_high > 0:
        cmd.extend(["--ref-high-augment", str(ref_high)])
        log(f"  Cycle {cycle}: Adding {ref_high} high-level reference augmentations")

    # -- Batch size (RTX 4070 Ti 12GB) --
    # 115M param model with batch 32 causes 27GB total (12GB dedicated + 15GB
    # shared/system RAM offload) — kills performance. Use batch 8 + grad accum 4
    # for effective batch 32 while staying within 12GB dedicated VRAM.
    bs = getattr(config, "retrain_batch_size", 8) if config is not None else 8
    effective_bs = max(1, min(bs, 8))
    if effective_bs != bs:
        log(f"  Cycle {cycle}: Batch size {bs} capped to {effective_bs} (GPU memory limit)")
    cmd.extend(["--batch-size", str(effective_bs)])

    # -- Max epochs --
    me = getattr(config, "retrain_max_epochs", None) if config is not None else None

    # -- Level-based epoch boost for L5+ champions --\n    # When champion reached L5+, increase max_epochs to ensure the model\n    # has enough gradient updates to learn higher-level instruction patterns.
    # Overrides MetaController budget to ensure sufficient training for L5+ tasks.
    # NOTE: Applied to BOTH cmd (CLI fallback) AND train_kwargs (actual path).
    # v19: Capped at 55 total — prevents overfitting from excessive epochs.
    if best_level >= 5:
        base_max_epochs = me if me is not None else 35
        level_boost = min((best_level - 4) * 3, 20)  # L5=+3, L6=+6, ..., L13=+20
        boosted_epochs = min(base_max_epochs + level_boost, 55)
        me = boosted_epochs  # Override for train_kwargs (in-process path)
        cmd.extend(["--max-epochs", str(boosted_epochs)])
        log(
            f"  Cycle {cycle}: Level-based epoch boost: {base_max_epochs} -> {boosted_epochs} (best_level={best_level})"
        )
    elif me is not None:
        cmd.extend(["--max-epochs", str(me)])

    # -- Validation split --
    vs = getattr(config, "retrain_val_split", None) if config is not None else None
    if vs is not None:
        cmd.extend(["--val-split", str(vs)])
    elif vs is None:
        # v19: Default to 15% validation for more reliable val_loss signal
        vs = 0.15
        cmd.extend(["--val-split", str(vs)])

    # -- Max per level (replay buffer diversity) --
    mpl = getattr(config, "retrain_max_per_level", None) if config is not None else None
    if mpl is not None:
        cmd.extend(["--max-per-level", str(mpl)])
    elif cycle >= 2:
        # v19: Higher per-level cap (500) to keep more diverse data
        mpl = 500
        cmd.extend(["--max-per-level", str(mpl)])

    # v24: Gradient accumulation (Pivot 1)
    # Force grad_accum=4 when batch=8 to maintain effective batch 32
    # while keeping VRAM within 12GB dedicated.
    ga_steps = getattr(config, "retrain_grad_accum_steps", 4) if config is not None else 4
    if ga_steps > 1:
        cmd.extend(["--grad-accum-steps", str(ga_steps)])
        log(f"  Cycle {cycle}: Gradient accumulation: {ga_steps} steps")

    # v24: LR scheduler (Pivot 2)
    lr_sched = getattr(config, "retrain_lr_scheduler", "plateau") if config is not None else "plateau"
    if lr_sched != "plateau":
        cmd.extend(["--lr-scheduler", lr_sched])
        log(f"  Cycle {cycle}: LR scheduler: {lr_sched}")

    # v24: SWA (Pivot 4)
    swa_start = getattr(config, "retrain_swa_start", None) if config is not None else None
    if swa_start is not None:
        cmd.extend(["--swa-start", str(swa_start)])
        swa_lr = getattr(config, "retrain_swa_lr", None) if config is not None else None
        if swa_lr is not None:
            cmd.extend(["--swa-lr", str(swa_lr)])
        log(f"  Cycle {cycle}: SWA averaging from epoch {swa_start}")

    log_cmd = " ".join(cmd[:6]) + " [...]"
    log(f"  Cycle {cycle}: Command: {log_cmd}")

    # -- Timeout --
    train_timeout = 3600  # default 1 hour
    if config is not None:
        train_timeout = getattr(config, "retrain_timeout", 3600)
    log(f"  Cycle {cycle}: Retrain timeout: {train_timeout}s")

    # Kill any orphaned train_model processes before starting.
    # Prevents CUDA initialization deadlocks when a stale process from
    # a previous run holds GPU resources (observed on WSL2).
    # Uses SIGTERM first with a 3-second grace period, then SIGKILL.
    import signal

    import torch

    orphan_count = 0
    try:
        # Match only train_model.py scripts (not grep, not shell wrappers)
        pids_str = subprocess.check_output(["pgrep", "-f", r"train_model\.py"], timeout=5).decode().strip()
        for old_pid_str in pids_str.split("\n"):
            old_pid = old_pid_str.strip()
            if not old_pid or old_pid == str(os.getpid()):
                continue
            try:
                pid_int = int(old_pid)
                # SIGTERM first -- give it a chance to clean up
                os.kill(pid_int, signal.SIGTERM)
                # Wait 3 seconds for graceful exit
                try:
                    os.waitpid(pid_int, os.WNOHANG)
                    # Check if still alive
                    time.sleep(3)
                    os.kill(pid_int, signal.SIGKILL)
                    log(f"  Cycle {cycle}: Orphan PID {pid_int} did not respond to SIGTERM -- used SIGKILL")
                except ProcessLookupError:
                    log(f"  Cycle {cycle}: Orphan PID {pid_int} exited gracefully after SIGTERM")
                orphan_count += 1
            except (ProcessLookupError, ValueError, OSError) as e:
                log(f"  Cycle {cycle}: Failed to clean up orphan PID {old_pid}: {e}")
    except subprocess.CalledProcessError:
        # pgrep exit code 1 means no matches -- no orphans, which is normal
        pass
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log(f"  Cycle {cycle}: Orphan cleanup check failed: {e}")
    if orphan_count:
        log(f"  Cycle {cycle}: Cleaned up {orphan_count} orphan process(es)")

    start = time.time()
    try:
        # -- GPU cooldown + full reset before retrain ------------------
        # The evolution engine leaves the GPU state fragmented after
        # hundreds of generations.  A full GPU reset prevents both
        # 'device not ready' and 'CUDACachingAllocator assertion failed'.
        try:
            import gc as _gc  # underscore avoids shadowing grad_clip var

            import torch

            if torch.cuda.is_available():
                # Step 0: Clean up CuPy context (GPU VM farm uses CuPy)
                # Only clear if CuPy was actually imported (avoids grabbing 8GB)
                if "cupy" in sys.modules:
                    try:
                        import cupy as cp
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                        log(f"  Cycle {cycle}: CuPy memory pool cleared")
                    except (ImportError, Exception):
                        pass
                # Step 1: Collect all Python objects to release GPU tensor refs
                _gc.collect()
                # Step 2: Synchronize and drain pending CUDA operations
                torch.cuda.synchronize()
                # Step 3: Empty all cached allocator memory
                torch.cuda.empty_cache()
                # Step 4: Reset peak stats (triggers allocator reset)
                torch.cuda.reset_peak_memory_stats()
                # Step 5: Extended cooldown for CUDA driver to settle
                # 3s cooldown after CuPy+PyTorch cleanup to let the driver
                # fully reclaim CUDA contexts from the evolution engine.
                time.sleep(3)
                log(f"  Cycle {cycle}: GPU reset complete")
        except Exception:
            pass

        # -- In-process training (Hermes terminal kills subprocesses) --
        #   The Hermes terminal tool sends SIGTERM to any child process
        #   running >~15-20s, regardless of shell=True, start_new_session,
        #   double-fork, or os.system().  The ONLY reliable approach is to
        #   import train_model.train() and call it directly in the same
        #   process.  This avoids child-process signal issues entirely.
        import importlib

        train_mod = importlib.import_module("train_model")
        os.makedirs(os.path.dirname(retrain_log), exist_ok=True)

        # Map CLI args to train() kwargs
        train_kwargs = dict(
            rebuild_dataset=False,
            resume_path=checkpoint_path,
            recurrent_data_path=recurrent_data,
        )
        model_size = getattr(config, "retrain_model_size", "xlarge") if config is not None else "xlarge"
        train_kwargs["model_size"] = model_size
        if lr:
            train_kwargs["lr_override"] = lr
        if freeze > 0:
            train_kwargs["freeze_layers"] = freeze
        if adaptive_freeze:
            train_kwargs["adaptive_freeze"] = True
            train_kwargs["freeze_layers"] = 0  # Ensure no static freeze
        if effective_bs:
            train_kwargs["batch_size"] = effective_bs
        if wd:
            train_kwargs["weight_decay"] = wd
        if pat:
            train_kwargs["patience"] = pat
        if gc:
            train_kwargs["grad_clip"] = gc
        if me:
            train_kwargs["max_epochs"] = me
        if vs:
            train_kwargs["val_split"] = vs
        if mpl:
            train_kwargs["max_per_level"] = mpl
        if ref_high is not None and ref_high > 0:
            train_kwargs["ref_high_augment"] = ref_high
        # v24: Gradient accumulation (Pivot 1)
        if ga_steps > 1:
            train_kwargs["grad_accum_steps"] = ga_steps
        # v24: LR scheduler (Pivot 2)
        if lr_sched != "plateau":
            train_kwargs["lr_scheduler"] = lr_sched
        # v24: SWA (Pivot 4)
        if swa_start is not None:
            train_kwargs["swa_start"] = swa_start
            if swa_lr is not None:
                train_kwargs["swa_lr"] = swa_lr

        # -- AMP + compile (speed improvements from fast-training pipeline) --
        # AMP (mixed precision) provides 1.5-2x training speedup on RTX 4070 Ti.
        # torch.compile provides 30-50% additional speedup.
        # Both are wrapped in try/except to gracefully fall back on failure.
        train_kwargs["amp"] = True
        train_kwargs["compile_mode"] = compile_mode
        log(f"  Cycle {cycle}: AMP enabled, torch.compile={compile_mode}")

        # Log subprocess launch BEFORE stdout redirection so it appears in main log
        log(f"  Cycle {cycle}: Launching subprocess retrain (fresh CUDA context)...")
        sys.stdout.flush()
        sys.stderr.flush()

        # Redirect stdout to capture training output
        log_f = open(retrain_log, "w", buffering=1, encoding="utf-8")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = log_f
        sys.stderr = log_f

        # v26: Skip subprocess entirely — go straight to in-process training.
        # The v22/v23 proven path: train_mod.train() directly with GPU reset.
        # Subprocess retrain was added in v24 but has NEVER succeeded on WSL2 CUDA 13.0.
        # The subprocess attempts burn 12+ min per cycle and corrupt GPU state.
        retrain_ok = False
        try:
            # NOTE: GPU reset was already performed above (lines 596-616).
            # v22/v23 proved a single reset before training is sufficient.
            # Double-resetting causes WSL2 CUDA 13.0 "device not ready" errors.
            log(f"  Cycle {cycle}: Running in-process training (v22 path)")
            train_mod.train(**train_kwargs)
            retrain_ok = True
            log(f"  Cycle {cycle}: In-process training completed")
        except RuntimeError as cuda_err:
            cuda_err_str = str(cuda_err)
            cudagraph_err = "CUDAGraph" in cuda_err_str or "cudagraph" in cuda_err_str.lower()
            retrain_ok = False  # Initialize before any retry logic

            # v18: Retry ALL CUDA transient errors with extended GPU cooldown + escalation
            # Catches: "device not ready", CUDACachingAllocator assertions, CUDAGraph errors
            # Retries 3× with escalating cooldown + full GPU reset between attempts.
            is_cuda_alloc = "cuda" in cuda_err_str.lower() or "cudacaching" in cuda_err_str.lower()
            is_cuda_internal = "INTERNAL ASSERT FAILED" in cuda_err_str or "handles_" in cuda_err_str
            is_device_not_ready = "device not ready" in cuda_err_str.lower()
            is_cudagraph = cudagraph_err and compile_mode is not None
            is_retryable = is_cuda_alloc or is_cuda_internal or is_device_not_ready or is_cudagraph
            if is_retryable:
                # CUDAGraph: first try disabling compile
                if is_cudagraph and not is_device_not_ready:
                    log(f"  Cycle {cycle}: CUDAGraphs error with compile_mode={compile_mode} -- retrying without compile")
                    train_kwargs["compile_mode"] = None
                    try:
                        train_mod.train(**train_kwargs)
                        retrain_ok = True
                    except RuntimeError as retry_err:
                        cuda_err_str = str(retry_err)  # fall through to GPU retry loop below
                        log(f"  Cycle {cycle}: CUDAGraph retry also failed: {retry_err}")
                # Retry loop with escalating GPU reset + cooldown
                if not retrain_ok:
                    retries_left = 3
                    while retries_left > 0:
                        retries_left -= 1
                        cooldown = 10 - retries_left * 3  # 10s, 7s, 4s
                        log(f"  Cycle {cycle}: GPU error ({cuda_err_str[:80]}...) — resetting GPU (retries left={retries_left}, cooldown={cooldown}s)")
                        import gc as _gc
                        import torch
                        _gc.collect()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        time.sleep(cooldown)
                        try:
                            train_mod.train(**train_kwargs)
                            retrain_ok = True
                            break
                        except RuntimeError as retry_err:
                            cuda_err_str = str(retry_err)
                            if retries_left == 0:
                                log(f"  Cycle {cycle}: GPU error persists after 3 retries — falling back to CPU")
                                break  # fall through to CPU fallback below
                if retrain_ok:
                    log(f"  Cycle {cycle}: GPU retrain succeeded after cooldown")
            else:
                log(f"  Cycle {cycle}: Training failed with non-retryable error: {cuda_err}")
                return {"retrained": False, "reason": f"fatal_error: {cuda_err}"}
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log_f.close()

        # -- CPU fallback when GPU retrain fails --------------------------
        # After CUDA corruption from evolution prefix generation, the GPU
        # may be unusable for training. CPU training is ~10x slower but
        # still produces valid checkpoints that keep the loop alive.
        if not retrain_ok:
            log(f"  Cycle {cycle}: GPU retrain failed — falling back to CPU training")
            try:
                cpu_log = open(retrain_log + ".cpu", "w", buffering=1, encoding="utf-8")
                old_stdout2 = sys.stdout
                old_stderr2 = sys.stderr
                sys.stdout = cpu_log
                sys.stderr = cpu_log
                try:
                    train_kwargs["amp"] = False
                    train_mod.train(**train_kwargs)
                    retrain_ok = True
                    log(f"  Cycle {cycle}: CPU retrain completed successfully")
                except Exception as cpu_err:
                    log(f"  Cycle {cycle}: CPU retrain also failed: {cpu_err}")
                    return {"retrained": False, "reason": f"cpu_fallback_error: {cpu_err}"}
                finally:
                    sys.stdout = old_stdout2
                    sys.stderr = old_stderr2
                    cpu_log.close()
            except Exception as fallback_err:
                log(f"  Cycle {cycle}: CPU fallback setup failed: {fallback_err}")
                return {"retrained": False, "reason": f"cpu_fallback_setup: {fallback_err}"}

    except FileNotFoundError:
        log(f"  Cycle {cycle}: train_model module not found")
        return {"retrained": False, "reason": "train_module_not_found"}
    except ImportError as e:
        log(f"  Cycle {cycle}: Failed to import train_model: {e}")
        return {"retrained": False, "reason": f"import_error: {e}"}
    except Exception as e:
        log(f"  Cycle {cycle}: Retrain failed: {type(e).__name__}: {e}")
        import traceback

        log(f"  Traceback: {''.join(traceback.format_tb(e.__traceback__))}")
        return {"retrained": False, "reason": f"exception: {e}"}
    elapsed = time.time() - start

    # The training saves model_final.pt as the final output.
    best_ckpt = os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt")
    final_ckpt = os.path.join(M1_CHECKPOINTS, "model_final.pt")

    new_checkpoint = final_ckpt  # fallback

    # Prefer checkpoint_best.pt (best val loss + optimizer state) over model_final.pt (last epoch, no optimizer).
    # Preserving optimizer state across cycles enables efficient continued training
    # instead of rebuilding Adam momentum from scratch each retrain.
    if os.path.isfile(best_ckpt):
        new_checkpoint = best_ckpt
        log(
            f"  Cycle {cycle}: Using checkpoint_best.pt "
            f"({os.path.getsize(best_ckpt) / 1e6:.1f}MB, with optimizer state)"
        )
    elif os.path.isfile(final_ckpt):
        new_checkpoint = final_ckpt
        log(
            f"  Cycle {cycle}: Falling back to model_final.pt "
            f"({os.path.getsize(final_ckpt) / 1e6:.1f}MB, no optimizer state)"
        )

    # -- Preserve full checkpoint + strip for evolution -------------
    # After retrain, checkpoint_best.pt has optimizer state (full, ~524MB).
    # Save full version as checkpoint_best_full.pt for next retrain,
    # then strip checkpoint_best.pt to model_state_dict only for evolution/PrefixGenerator.
    if os.path.isfile(best_ckpt):
        full_ckpt_path = os.path.join(M1_CHECKPOINTS, "checkpoint_best_full.pt")
        try:
            shutil.copy2(best_ckpt, full_ckpt_path)
            log(
                f"  Cycle {cycle}: Full checkpoint preserved: {full_ckpt_path} "
                f"({os.path.getsize(full_ckpt_path) / 1e6:.1f}MB)"
            )
        except (OSError, PermissionError) as e:
            log(f"  Cycle {cycle}: Failed to preserve full checkpoint: {e}")

        # v15: Save cycle-specific full checkpoint for chaining.
        # This creates a chain of checkpoints so each cycle can resume from
        # the previous cycle's retrained model instead of always starting
        # from the C11 baseline.
        cycle_full_ckpt = os.path.join(M1_CHECKPOINTS, f"checkpoint_full_cycle_{cycle}.pt")
        try:
            shutil.copy2(best_ckpt, cycle_full_ckpt)
            log(
                f"  Cycle {cycle}: Chained full checkpoint saved: {cycle_full_ckpt} "
                f"({os.path.getsize(cycle_full_ckpt) / 1e6:.1f}MB)"
            )
        except (OSError, PermissionError) as e:
            log(f"  Cycle {cycle}: Failed to save chained checkpoint: {e}")

        # Strip checkpoint_best.pt for evolution engine (model_state_dict only)
        try:
            import torch

            ckpt_full = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            sd = ckpt_full.get("model_state_dict", ckpt_full)
            # Strip _orig_mod prefix from torch.compile
            clean_sd = {}
            for k, v in sd.items():
                key = k
                if key.startswith("_orig_mod."):
                    key = key[11:]
                clean_sd[key] = v
            # Read actual architecture from the saved training config
            train_cfg = ckpt_full.get("config", {})
            # Save proper config dict so train_model.py can load it
            strip_cfg = {
                "vocab_size": train_cfg.get("vocab_size", 2048),
                "d_model": train_cfg.get("d_model", 512),
                "n_layers": train_cfg.get("n_layers", 14),
                "n_heads": train_cfg.get("n_heads", 8),
                "d_ff": train_cfg.get("d_ff", 1365),
                "max_seq_len": train_cfg.get("max_seq_len", 2048),
            }
            # Infer from weights if config is empty
            if not train_cfg:
                qkv_keys = [k for k in clean_sd if "blocks.0.attn.qkv.weight" in k]
                if qkv_keys:
                    strip_cfg["d_model"] = clean_sd[qkv_keys[0]].shape[1]
                blk_keys = [k for k in clean_sd if k.startswith("blocks.") and "ln1.weight" in k]
                if blk_keys:
                    strip_cfg["n_layers"] = len(blk_keys)
            torch.save(
                {
                    "model_state_dict": clean_sd,
                    "config": strip_cfg,
                    "val_loss": ckpt_full.get("val_loss"),
                },
                best_ckpt,
            )
            log(f"  Cycle {cycle}: Stripped checkpoint for evolution: {os.path.getsize(best_ckpt) / 1e6:.1f}MB ({strip_cfg['n_layers']}L/d={strip_cfg['d_model']})")
        except Exception as e:
            log(f"  Cycle {cycle}: Failed to strip checkpoint: {e}")

    return {
        "retrained": True,
        "checkpoint": new_checkpoint,
        "elapsed_seconds": round(elapsed, 2),
        "example_count": example_count,
        "exit_code": 0,
        "backup_checkpoint": backup_checkpoint,
    }


def cycle_resume_state(cycle: int) -> dict[str, Any]:
    """Check if a cycle has already been completed (resume support)."""
    results = []
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    completed_cycles = {r.get("cycle") for r in results}
    return {"completed": cycle in completed_cycles, "all_results": results}


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
        "--retrain-adaptive-freeze", action="store_true", default=False,
        help="Enable adaptive freeze during retrain — dynamically freezes converged blocks "
             "based on gradient norms instead of static freeze_layers",
    )
    parser.add_argument(
        "--retrain-max-per-level", type=int, default=None, help="Max recurrent examples per curriculum level"
    )
    parser.add_argument(
        "--retrain-model-size", type=str, default="xlarge", choices=["small", "medium", "large", "xlarge"], help="Model size for retrain"
    )
    parser.add_argument("--retrain-timeout", type=int, default=3600, help="Timeout per retrain subprocess (seconds)")
    parser.add_argument("--retrain-patience", type=int, default=None, help="Early stopping patience for retrain")
    parser.add_argument("--retrain-max-epochs", type=int, default=None, help="Max training epochs per retrain")
    parser.add_argument("--retrain-grad-clip", type=float, default=None, help="Gradient clipping norm for retrain")
    parser.add_argument("--retrain-batch-size", type=int, default=None, help="Batch size for retrain")
    parser.add_argument("--retrain-weight-decay", type=float, default=None, help="Weight decay for retrain optimizer")
    # v24: Gradient accumulation (Pivot 1)
    parser.add_argument(
        "--retrain-grad-accum-steps", type=int, default=1,
        help="Gradient accumulation steps for retrain (default: 1, effective batch=batch_size × N)"
    )
    # v24: LR scheduler (Pivot 2)
    parser.add_argument(
        "--retrain-lr-scheduler", type=str, default="plateau",
        choices=["plateau", "cosine-warm"],
        help="LR scheduler for retrain: 'plateau' (ReduceLROnPlateau) or 'cosine-warm' (CosineAnnealingWarmRestarts)"
    )
    # v24: SWA (Pivot 4)
    parser.add_argument(
        "--retrain-swa-start", type=int, default=None,
        help="Start SWA averaging at this epoch during retrain (e.g. 30)"
    )
    parser.add_argument(
        "--retrain-swa-lr", type=float, default=None,
        help="Learning rate for SWA epochs during retrain"
    )
    parser.add_argument(
        "--force-retrain-interval",
        type=int,
        default=3,
        help="Force retrain every N cycles even if champion unchanged (0=disable, default: 3)",
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
    if not args.retrain_adaptive_freeze and all(v is None or v == 0 or v == "xlarge" or v == 3600 for v in retrain_args):
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
    cfg.retrain_grad_accum_steps = args.retrain_grad_accum_steps
    cfg.retrain_lr_scheduler = args.retrain_lr_scheduler
    return cfg


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    # Input validation
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

    log("Multi-Cycle Recurrent Feedback Runner")
    log(f"  Cycles:        {args.cycles}")
    log(f"  Gen/cycle:     {args.gens}")
    log(f"  GPU available: {has_gpu()}")
    log(f"  PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")
    log("  Torch compile:  disabled (CUDA 13.0 stability)")
    log("  Memory cleanup: torch.cuda.empty_cache() each cycle")
    log(f"  CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING', '0')}")
    log(f"  Resume:        {args.resume}")
    log(f"  Starting ckpt: {args.checkpoint or 'model_45m.pt'}")

    # Determine starting checkpoint
    candidates = []
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = [
            os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt"),  # v11: best from previous run
            os.path.join(M1_CHECKPOINTS, "model_final.pt"),
            os.path.join(M1_CHECKPOINTS, "model_45m.pt"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                checkpoint = c
                break
    if not checkpoint or not os.path.isfile(checkpoint):
        log(f"FATAL: Checkpoint not found. Checked: {candidates}")
        sys.exit(1)

    # Build retrain config (or None for backward-compatible defaults)
    retrain_config = _build_retrain_config(args)
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

    # Find completed cycles (resume support)
    completed_cycles = set()
    all_results = []
    last_retrained_ckpt = None
    if args.resume and os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        completed_cycles.add(r.get("cycle", -1))
                        all_results.append(r)
                        # Track the last successfully retrained checkpoint
                        if r.get("retrained") and r.get("checkpoint"):
                            last_retrained_ckpt = r["checkpoint"]
                    except json.JSONDecodeError:
                        continue
        log(f"  Resuming: {len(completed_cycles)} cycles already completed")
        # Use last retrained checkpoint instead of initial model
        if last_retrained_ckpt and os.path.isfile(last_retrained_ckpt):
            checkpoint = last_retrained_ckpt
            log(f"  Resume checkpoint: {last_retrained_ckpt} ({os.path.getsize(last_retrained_ckpt) / 1e6:.1f}MB)")

    # Save initial checkpoint info
    init_size_mb = os.path.getsize(checkpoint) / 1e6
    log(f"  Checkpoint: {checkpoint} ({init_size_mb:.1f}MB)")

    # -- Clean state (remove saved state files) -------------------------
    if args.clean_state:
        import glob

        # Default paths: current working directory + common project roots
        clean_dirs = {os.getcwd(), "/root/milestone2", "/root/"}
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

    # -- MetaController initialization --------------------------------
    meta_controller = None
    if _HAS_META_CONTROLLER:
        meta_controller = MetaController(
            checkpoint_dir=M1_CHECKPOINTS,
            best_checkpoint_name="checkpoint_best.pt",
            log_file=os.path.join(PROJECT_ROOT, "meta_controller.log"),
        )
        # Load state if available (always, not just on --resume, to preserve gating history)
        if os.path.isfile(META_CONTROLLER_STATE):
            if meta_controller.load_state(META_CONTROLLER_STATE):
                log(f"  MetaController: Loaded state from {META_CONTROLLER_STATE}")
                log(
                    f"  MetaController: best_val_loss={meta_controller._best_val_loss:.4f}, "
                    f"best_fitness={meta_controller.best_fitness:.3f}"
                )
            else:
                log("  MetaController: No saved state -- starting fresh")
        # Initialize best_val_loss from existing checkpoint_best.pt if available
        if os.path.isfile(BEST_CHECKPOINT_PATH):
            try:
                ckpt_size = os.path.getsize(BEST_CHECKPOINT_PATH)
                log(f"  MetaController: Best checkpoint exists ({ckpt_size / 1e6:.1f}MB)")
            except OSError:
                pass
    else:
        log("  MetaController: Not available -- running without dynamic tuning")

    # -- v11.5: BarrierTuner initialization ----------------------------
    barrier_tuner = None
    BARRIER_TUNER_STATE = os.path.join(SCRIPT_DIR, "barrier_tuner_state.json")
    if _HAS_BARRIER_TUNER:
        # Detect model size from checkpoint state dict if available
        model_layers = 8  # default: small model
        try:
            import torch

            ckpt_path = os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt")
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    model_layers = int(ckpt.get("n_layers") or ckpt.get("config", {}).get("n_layers", 8))
        except Exception:
            pass
        barrier_tuner = BarrierTuner(state_path=BARRIER_TUNER_STATE, model_n_layers=model_layers)
        if args.resume and os.path.isfile(BARRIER_TUNER_STATE):
            if barrier_tuner.load_state(BARRIER_TUNER_STATE):
                log(f"  BarrierTuner: Loaded state from {BARRIER_TUNER_STATE}")
            else:
                log("  BarrierTuner: No saved state -- starting fresh")
        else:
            log("  BarrierTuner: Initialized (fresh)")

        # Log BarrierTuner startup
        log(
            f"  BarrierTuner: level_stall={barrier_tuner.level_stall_cycles}cy, "
            f"fitness_plateau={barrier_tuner.fitness_plateau_cycles}cy, "
            f"diversity_collapse<{barrier_tuner.diversity_collapse_threshold}"
        )

    # -- StrategyArchive tuning (persist archive params across cycles) -
    last_archive_params: dict[str, Any] = {}

    # -- v11.5: Adaptive retrain budget tracking -----------------------
    # Track actual epochs used per cycle to cap next cycle intelligently
    last_retrain_epochs: int = 0
    last_champion_fitness: float = float("-inf")
    last_champion_level: int = 0
    last_champion_words: int = 0

    # -- Stuck-champion recovery counters ------------------------------
    cycles_since_last_retrain: int = 0

    # -- Post-retrain diversity boost ----------------------------------
    # After retrain, the model's output distribution narrows. The next
    # evolution cycle needs higher diversity to rediscover good programs.
    post_retrain_boost_cycles: int = 0  # >0 means boost active for this many more cycles

    # Main loop
    prev_cycle_metrics: dict[str, Any] = {}
    for cycle in range(1, args.cycles + 1):
        log(f"\n{'-' * 60}")
        log(f"CYCLE {cycle}/{args.cycles}")
        log(f"{'-' * 60}")

        if args.resume and cycle in completed_cycles:
            log(f"  Cycle {cycle} already completed -- skipping")
            continue

        # Reset skip flag each cycle (set by RESET-AND-RELEARN from previous cycle)
        skip_evolution = False

        # -- v13: GPU memory defragmentation at cycle boundaries ---
        # Prevents CUDA segfault from accumulated memory fragmentation
        try:
            import gc
            import torch

            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated() / 1024**2
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after = torch.cuda.memory_allocated() / 1024**2
                freed = before - after
                if freed > 50:
                    log(
                        f"  Cycle {cycle}: GPU memory defrag: freed {freed:.0f}MB "
                        f"(was {before:.0f}MB, now {after:.0f}MB)"
                    )
        except Exception as e:
            log(f"  Cycle {cycle}: GPU memory cleanup skipped ({e})")

        # -- Get self-tuning params for this cycle ------------------
        evo_meta_params: dict[str, Any] = {}
        retrain_meta_params: dict[str, Any] = {}
        archive_params: dict[str, Any] = {}

        # v11.5: BarrierTuner provides primary tuning
        if barrier_tuner is not None:
            bt_retrain = barrier_tuner.tune_retrain_params()
            bt_evolution = barrier_tuner.tune_evolution_params()
            bt_archive = barrier_tuner.tune_archive_params()

            # Merge into meta params (BarrierTuner takes priority)
            evo_meta_params.update(bt_evolution)
            retrain_meta_params.update(bt_retrain)
            archive_params.update(bt_archive)

            # Dynamic generation count override
            gen_count_override = bt_archive.get("max_generations", args.gens)
            if gen_count_override != args.gens:
                log(f"  Cycle {cycle}: BarrierTuner overrides gens: {args.gens} -> {gen_count_override}")
        else:
            gen_count_override = args.gens

        # MetaController supplements (backward compatible)
        if meta_controller is not None:
            mc_evo = meta_controller.tune_evolution_params(cycle)
            mc_retrain = meta_controller.tune_retrain_params(cycle)

            # Only apply MetaController params where BarrierTuner hasn't set them
            for k, v in mc_evo.items():
                if k not in evo_meta_params:
                    evo_meta_params[k] = v
            for k, v in mc_retrain.items():
                if k not in retrain_meta_params:
                    retrain_meta_params[k] = v

            # v11 P3.2: Trigger meta-evolution if MC requests it
            if mc_retrain.get("meta_evolution_requested"):
                log(f"  Cycle {cycle}: *** META-EVOLUTION TRIGGERED ***")
                log(f"  Cycle {cycle}: Running lightweight meta-evolution...")
                try:
                    from meta_evolution import MetaEvolutionConfig, MetaEvolutionEngine

                    me_config = MetaEvolutionConfig(
                        meta_population_size=4,
                        meta_generations=2,
                        base_generations=20,
                        seed=cycle,
                    )
                    me_engine = MetaEvolutionEngine(meta_config=me_config)
                    me_summary = me_engine.run()
                    best_cfg = me_engine.best_config()
                    log(
                        f"  Cycle {cycle}: Meta-evolution found: pop={best_cfg.population_size} "
                        f"mu={best_cfg.mutation_rate:.3f} cx={best_cfg.crossover_rate:.3f}"
                    )
                    evo_meta_params.update(
                        {
                            "mutation_rate": best_cfg.mutation_rate,
                            "crossover_rate": best_cfg.crossover_rate,
                            "elite_ratio": best_cfg.elite_ratio,
                            "population_size": best_cfg.population_size,
                        }
                    )
                except Exception as e:
                    log(f"  Cycle {cycle}: Meta-evolution failed: {e}")

        # Pass previous cycle's self-calibrate state for continuity
        if cycle > 1 and prev_cycle_metrics:
            prev_mu = prev_cycle_metrics.get("mutation_rate")
            prev_dt = prev_cycle_metrics.get("diversity_threshold")
            if prev_mu is not None:
                evo_meta_params.setdefault("mutation_rate", prev_mu)
                log(f"  Cycle {cycle}: Using prev cycle's mutation_rate={prev_mu:.3f}")
            if prev_dt is not None:
                evo_meta_params.setdefault("diversity_threshold", prev_dt)
                log(f"  Cycle {cycle}: Using prev cycle's diversity_threshold={prev_dt:.3f}")

        # -- Post-retrain diversity boost ----------------------------
        # After retrain, the model produces conservative programs. Boost
        # exploration params for the next cycle to speed recovery.
        if post_retrain_boost_cycles > 0:
            boost_novelty = max(evo_meta_params.get("novelty_weight", 0.3), 0.6)
            boost_diversity = max(evo_meta_params.get("diversity_threshold", 0.3), 0.5)
            boost_mutation = max(evo_meta_params.get("mutation_rate", 0.3), 0.7)
            evo_meta_params["novelty_weight"] = boost_novelty
            evo_meta_params["diversity_threshold"] = boost_diversity
            evo_meta_params["mutation_rate"] = boost_mutation
            evo_meta_params["diversity_inject_ratio"] = 0.4
            log(
                f"  Cycle {cycle}: Post-retrain diversity boost ACTIVE "
                f"(novelty={boost_novelty}, div={boost_diversity}, mu={boost_mutation}, "
                f"remaining={post_retrain_boost_cycles})"
            )
            post_retrain_boost_cycles -= 1

        # Log tuned params
        if evo_meta_params:
            log(f"  Cycle {cycle}: Tuned evolution: {evo_meta_params}")
        if retrain_meta_params:
            log(f"  Cycle {cycle}: Tuned retrain: {retrain_meta_params}")
        if archive_params:
            log(f"  Cycle {cycle}: Tuned archive: {archive_params}")

        # Inherit last cycle's archive params if none provided this cycle
        if not archive_params and last_archive_params:
            archive_params = dict(last_archive_params)

        # Phase 1: Run evolution (with dynamic gen count and archive params)
        # Merge archive params into evolution params
        # Take pre-cycle checkpoint snapshot (for LEVEL_REGRESSION rollback)
        if skip_evolution:
            log(f"  Cycle {cycle}: SKIPPING evolution (RESET-AND-RELEARN from previous cycle)")
            evo_result = {
                "cycle": cycle,
                "generations_completed": 0,
                "final_fitness": 0.0,
                "final_level": 0,
                "champion_words": 0,
                "elapsed_seconds": 0.0,
                "champion_traces": 0,
                "novelty_archive_size": 0,
                "qd_occupied_cells": 0,
                "qd_total_cells": 0,
                "checkpoint": checkpoint,
                "mutation_rate": 0.5,
                "diversity_threshold": 0.3,
                "diversity": 0.3,
            }
            skip_evolution = False  # consumed
        else:
            if meta_controller is not None:
                if hasattr(meta_controller, 'snapshot_before_cycle'):
                    meta_controller.snapshot_before_cycle(cycle, checkpoint)
            evo_with_archive = dict(evo_meta_params)
            evo_with_archive.update(archive_params)
            evo_result = run_evolution(cycle, gen_count_override, checkpoint, meta_evolution_params=evo_with_archive)

        # Save self-calibrate state for next cycle
        prev_cycle_metrics = evo_result

        # Snapshot checkpoint after evolution if this is a new best level
        # (before retrain -- captures the model that achieved this level)
        current_level = evo_result.get("final_level", 0)
        if meta_controller is not None and current_level > meta_controller.best_level:
            if hasattr(meta_controller, 'snapshot_before_cycle'):
                meta_controller.snapshot_before_cycle(cycle, checkpoint)
                meta_controller._log(f"Best-level snapshot: C{cycle} reached L{current_level}")

        # Phase 2: Retrain (unless --no-retrain)
        retrain_result: dict[str, Any] = {"retrained": False, "reason": "skipped"}
        if not args.no_retrain:
            # -- v11.5: Retrain skip -- skip if champion hasn't improved -----
            current_fitness = evo_result.get("final_fitness", 0.0)
            current_level = evo_result.get("final_level", 0)
            current_words = evo_result.get("champion_words", 0)
            should_retrain = True

            # Skip retrain if champion hasn't meaningfully improved and we're past initial cycles
            if cycle >= 2 and last_retrain_epochs > 0:
                # v13: 5% relative threshold with 0.01 floor to prevent
                # zero threshold when fitness=0. For fitness=100, threshold=5.0.
                # For fitness=0.5, threshold=0.025 (capped min 0.01).
                fitness_threshold = max(abs(last_champion_fitness) * 0.05, 0.01)
                fitness_same = abs(current_fitness - last_champion_fitness) < fitness_threshold
                level_same = current_level == last_champion_level
                words_same = current_words == last_champion_words
                if fitness_same and level_same and words_same:
                    # -- Force retrain interval override --
                    if args.force_retrain_interval > 0:
                        cycles_since_last_retrain += 1
                        if cycles_since_last_retrain >= args.force_retrain_interval:
                            log(
                                f"  Cycle {cycle}: Force retrain after {cycles_since_last_retrain} "
                                f"cycles without champion change"
                            )
                            should_retrain = True
                            cycles_since_last_retrain = 0
                    if not should_retrain:
                        log(
                            f"  Cycle {cycle}: Champion unchanged since last retrain (F={current_fitness:.3f} "
                            f"L{current_level} W={current_words}) -- skipping retrain"
                        )
                        # Still use the previous checkpoint for next cycle
                        retrain_result = {"retrained": False, "reason": "champion_unchanged"}
                        # Record retrain as successful (keep existing checkpoint)
                        retrain_result["checkpoint"] = checkpoint

            # -- v11.5: Adaptive retrain budget --------------------------
            # If last retrain early-stopped at E epochs, cap next at E*1.5
            # This prevents wasteful full-length training when less is needed
            if last_retrain_epochs > 0 and should_retrain:
                adaptive_max_epochs = min(max(int(last_retrain_epochs * 1.5), 30), 50)
                current_max = retrain_meta_params.get("retrain_max_epochs", 50)
                if adaptive_max_epochs < current_max:
                    retrain_meta_params["retrain_max_epochs"] = adaptive_max_epochs
                    log(
                        f"  Cycle {cycle}: Adaptive retrain budget: "
                        f"{current_max} -> {adaptive_max_epochs} epochs "
                        f"(last cycle used {last_retrain_epochs})"
                    )

            if should_retrain:
                # -- v11.5: Acceleration defaults -----------------------
                # Larger batch (48 vs 32) = 50% fewer backward passes
                # v19: Larger val_split (0.15 vs 0.05) = more reliable val_loss signal
                retrain_meta_params.setdefault("retrain_batch_size", 8)
                retrain_meta_params.setdefault("retrain_val_split", 0.15)

                # Merge MetaController retrain params with CLI overrides
                # (CLI overrides take precedence)
                merged_retrain_config = retrain_config
                if retrain_meta_params and _HAS_META_CONTROLLER:
                    import types

                    mc_cfg = types.SimpleNamespace()
                    mc_cfg.retrain_lr = retrain_meta_params.get("retrain_lr")
                    mc_cfg.retrain_freeze_layers = retrain_meta_params.get("retrain_freeze_layers", 0)
                    mc_cfg.retrain_max_per_level = retrain_meta_params.get("retrain_max_per_level")
                    mc_cfg.retrain_patience = retrain_meta_params.get("retrain_patience")
                    mc_cfg.retrain_max_epochs = retrain_meta_params.get("retrain_max_epochs")
                    mc_cfg.retrain_model_size = "xlarge"
                    mc_cfg.retrain_timeout = 3600
                    mc_cfg.retrain_min_examples = 1
                    mc_cfg.retrain_val_split = retrain_meta_params.get("retrain_val_split")
                    mc_cfg.retrain_grad_clip = None
                    mc_cfg.retrain_batch_size = retrain_meta_params.get("retrain_batch_size")
                    mc_cfg.retrain_weight_decay = None
                    mc_cfg.ref_high_augment = retrain_meta_params.get("ref_high_augment")
                    mc_cfg.retrain_adaptive_freeze = False  # Until CLI overrides

                    # CLI overrides win
                    if retrain_config is not None:
                        # Boolean flags need special handling
                        if getattr(retrain_config, "retrain_adaptive_freeze", False):
                            mc_cfg.retrain_adaptive_freeze = True
                        for attr in [
                            "retrain_lr",
                            "retrain_freeze_layers",
                            "retrain_max_per_level",
                            "retrain_patience",
                            "retrain_max_epochs",
                            "retrain_grad_clip",
                            "retrain_batch_size",
                            "retrain_weight_decay",
                            "retrain_adaptive_freeze",
                            "retrain_swa_start",
                            "retrain_grad_accum_steps",
                            "retrain_lr_scheduler",
                        ]:
                            cli_val = getattr(retrain_config, attr, None)
                            if cli_val is not None:
                                setattr(mc_cfg, attr, cli_val)

                    merged_retrain_config = mc_cfg

                retrain_result = retrain_model(
                    cycle, checkpoint, config=merged_retrain_config, best_level=current_level
                )
                # Reset force-retrain timer on any retrain attempt
                cycles_since_last_retrain = 0

            # -- Checkpoint gating ---------------------------------
            if meta_controller is not None and retrain_result.get("retrained"):
                new_ck = retrain_result.get("checkpoint")
                if new_ck and os.path.isfile(new_ck):
                    # Parse val_loss from retrain log
                    retrain_log = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")
                    val_loss = parse_val_loss_from_retrain_log(retrain_log)

                    if val_loss is not None:
                        accepted = meta_controller.accept_checkpoint(val_loss, new_ck)
                        # v13: Persist val_loss in results for cross-cycle analysis
                        retrain_result["val_loss"] = val_loss
                        if accepted:
                            log(f"  Cycle {cycle}: Checkpoint ACCEPTED (val_loss={val_loss:.4f})")
                            # v24 Pivot 5: Reset consecutive rejection counter on acceptance
                            if hasattr(meta_controller, '_consecutive_rejections'):
                                meta_controller._consecutive_rejections = 0
                            # After a successful retrain, boost diversity on the next cycle
                            # to help the model rediscover good programs faster.
                            post_retrain_boost_cycles = 1
                            cycle_ckpt = os.path.join(M1_CHECKPOINTS, f"model_cycle{cycle}.pt")
                            try:
                                shutil.copy2(new_ck, cycle_ckpt)
                                checkpoint = cycle_ckpt
                                log(f"  Cycle {cycle}: Next cycle will use: {checkpoint}")
                            except (OSError, PermissionError) as e:
                                log(f"  Cycle {cycle}: Checkpoint copy failed: {e}")
                        else:
                            log(
                                f"  Cycle {cycle}: Checkpoint REJECTED (val_loss={val_loss:.4f} > "
                                f"threshold={meta_controller._best_val_loss * meta_controller.checkpoint_acceptance_tolerance:.4f})"
                            )
                            # Restore checkpoint_best_full.pt from pre-retrain backup
                            # (retrain_model() always overwrites it during training)
                            _bkp = retrain_result.get("backup_checkpoint")
                            if _bkp and os.path.isfile(_bkp):
                                full_ckpt = os.path.join(M1_CHECKPOINTS, "checkpoint_best_full.pt")
                                try:
                                    shutil.copy2(_bkp, full_ckpt)
                                    log(f"  Cycle {cycle}: Restored checkpoint_best_full.pt from backup")
                                except (OSError, PermissionError) as e:
                                    log(f"  Cycle {cycle}: Warning: backup restore failed: {e}")
                            log(
                                f"  Cycle {cycle}: Falling back to pre-cycle snapshot "
                                f"(checkpoint_best.pt was overwritten by retrain)"
                            )
                            # Roll back using pre-cycle snapshot (checkpoint_best.pt was
                            # overwritten by retrain -- don't copy it over itself)
                            if meta_controller.execute_rollback(checkpoint, use_snapshot=True) is not None:
                                # Keep the original checkpoint path -- it was restored from snapshot.
                                # Do NOT use get_best_checkpoint() -- it returns checkpoint_best.pt
                                # which was overwritten by retrain with rejected weights.
                                pass

                            # v24 Pivot 5: Reset-and-Relearn on stagnation
                            # Track consecutive rejections; after 3, trigger a hard reset
                            # back to the best-ever checkpoint with a full retrain.
                            if not hasattr(meta_controller, '_consecutive_rejections'):
                                meta_controller._consecutive_rejections = 0
                            meta_controller._consecutive_rejections += 1
                            if meta_controller._consecutive_rejections >= 3:
                                log(
                                    f"  Cycle {cycle}: *** RESET-AND-RELEARN TRIGGERED *** "
                                    f"({meta_controller._consecutive_rejections} consecutive rejections)"
                                )
                                # Roll back to the absolute best checkpoint ever
                                best_ever_ckpt = os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt")
                                if os.path.isfile(best_ever_ckpt):
                                    # Force retrain on all accumulated data from best checkpoint
                                    log(f"  Cycle {cycle}: Reset to best-ever checkpoint: {best_ever_ckpt}")
                                    checkpoint = best_ever_ckpt
                                    meta_controller._consecutive_rejections = 0
                                    # Skip evolution this cycle — go straight to retrain
                                    skip_evolution = True
                                else:
                                    log(f"  Cycle {cycle}: Reset failed — best checkpoint not found")
                    else:
                        # v11: Log diagnostics when val_loss can't be parsed
                        retrain_log_path = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")
                        retrain_reason = retrain_result.get("reason", "unknown")
                        log(f"  Cycle {cycle}: Could not parse val_loss (reason: {retrain_reason})")
                        if os.path.isfile(retrain_log_path):
                            log(f"  Cycle {cycle}: Retrain log exists ({os.path.getsize(retrain_log_path)} bytes)")
                            try:
                                with open(retrain_log_path, encoding="utf-8", errors="replace") as rf:
                                    last_lines = rf.readlines()[-5:]
                                for rl in last_lines:
                                    log(f"  Cycle {cycle}: [retrain log] {rl.rstrip()}")
                            except Exception:
                                pass
                            # Retry: recreate retrain with doubled patience
                            if retrain_result.get("retrained") is False:
                                log(f"  Cycle {cycle}: Retrain failed -- not accepting checkpoint")
                        else:
                            log(f"  Cycle {cycle}: Retrain log NOT FOUND at {retrain_log_path}")

                        # v11 P1.2: Retry failed retrain with doubled patience
                        if retrain_result.get("retrained") is False and "train_module" not in retrain_reason:
                            log(f"  Cycle {cycle}: Retrying retrain with doubled patience...")
                            retry_config = dict(merged_retrain_config) if merged_retrain_config else {}
                            retry_config["retrain_patience"] = retry_config.get("retrain_patience", 5) * 2
                            retry_config["retrain_max_epochs"] = min(
                                retry_config.get("retrain_max_epochs", 50) * 2, 100
                            )
                            from types import SimpleNamespace

                            retry_cfg = SimpleNamespace(**retry_config)
                            retry_result = retrain_model(cycle, checkpoint, config=retry_cfg, best_level=current_level)
                            if retry_result.get("retrained"):
                                retrain_log2 = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")
                                val_loss2 = parse_val_loss_from_retrain_log(retrain_log2)
                                if val_loss2 is not None:
                                    log(f"  Cycle {cycle}: Retry succeeded! val_loss={val_loss2:.4f}")
                                    new_ck = retry_result.get("checkpoint")
                                    if new_ck and os.path.isfile(new_ck):
                                        meta_controller.accept_checkpoint(val_loss2, new_ck)
                                        cycle_ckpt = os.path.join(M1_CHECKPOINTS, f"model_cycle{cycle}.pt")
                                        shutil.copy2(new_ck, cycle_ckpt)
                                        checkpoint = cycle_ckpt
                                else:
                                    log(f"  Cycle {cycle}: Retry also failed to produce val_loss")
                            else:
                                reason2 = retry_result.get("reason", "unknown")
                                log(f"  Cycle {cycle}: Retry failed: {reason2}")

                        # Accept checkpoint unconditionally as fallback (existing behavior)
                        cycle_ckpt = os.path.join(M1_CHECKPOINTS, f"model_cycle{cycle}.pt")
                        try:
                            shutil.copy2(new_ck, cycle_ckpt)
                            checkpoint = cycle_ckpt
                        except (OSError, PermissionError) as e:
                            log(f"  Cycle {cycle}: Checkpoint copy failed: {e}")
                elif new_ck:
                    log(f"  Cycle {cycle}: Retrain checkpoint not found at {new_ck}")

            elif retrain_result.get("retrained") and meta_controller is None:
                # No MetaController -- original behavior
                new_ck = retrain_result.get("checkpoint")
                if new_ck and os.path.isfile(new_ck):
                    try:
                        cycle_ckpt = os.path.join(M1_CHECKPOINTS, f"model_cycle{cycle}.pt")
                        shutil.copy2(new_ck, cycle_ckpt)
                        checkpoint = cycle_ckpt
                        log(f"  Cycle {cycle}: Next cycle will use: {checkpoint}")
                    except (OSError, PermissionError) as e:
                        log(f"  Cycle {cycle}: Checkpoint copy failed: {e}")

        # -- Build TelemetrySnapshot and update MetaController ------
        if meta_controller is not None:
            diversity_proxy = 0.3  # default
            if evo_result.get("novelty_archive_size", 0) > 0:
                qd_total = max(evo_result.get("qd_total_cells", 100), 1)
                diversity_proxy = min(1.0, evo_result.get("qd_occupied_cells", 0) / qd_total)

            snapshot = TelemetrySnapshot(
                cycle=cycle,
                fitness=evo_result.get("final_fitness", 0.0),
                level=evo_result.get("final_level", 0),
                champion_words=evo_result.get("champion_words", 0),
                val_loss=None,  # populated below if available
                diversity=diversity_proxy,
                novelty_size=evo_result.get("novelty_archive_size", 0),
                qd_occupied=evo_result.get("qd_occupied_cells", 0),
                qd_total=evo_result.get("qd_total_cells", 100),
                traces_count=evo_result.get("champion_traces", 0),
                retrained=retrain_result.get("retrained", False),
                generations_completed=evo_result.get("generations_completed", 0),
            )

            # Try to include val_loss
            if retrain_result.get("retrained"):
                retrain_log = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")
                val_loss = parse_val_loss_from_retrain_log(retrain_log)
                if val_loss is not None:
                    snapshot.val_loss = val_loss
                else:
                    log(f"  Cycle {cycle}: [parser debug] val_loss=None, log={retrain_log}, exists={os.path.isfile(retrain_log)}")

            meta_controller.update_cycle(snapshot)

            # -- Check for rollback --------------------------------
            if meta_controller.should_rollback():
                log(f"  Cycle {cycle}: *** ROLLBACK TRIGGERED ***")
                log(f"  Cycle {cycle}: Rolling back to best checkpoint...")
                # LEVEL_REGRESSION: use_snapshot=True to restore pre-retrain model
                # (MetaController version check — use_snapshot may not exist)
                use_snap = meta_controller.detect_stagnation().value == "level_regression"
                rolled = meta_controller.execute_rollback(checkpoint)
                if rolled is not None:
                    # Use the rolled-back checkpoint directly (may be a pre-cycle snapshot)
                    checkpoint = rolled
                    log(f"  Cycle {cycle}: Rollback complete. Using: {checkpoint}")
                    # Mark this cycle as rollback in results
                    retrain_result["rollback_performed"] = True
                    # Notify BarrierTuner so stagnation tracking resets
                    if barrier_tuner is not None:
                        barrier_tuner.notify_rollback()
                        log(f"  Cycle {cycle}: BarrierTuner: stagnation reset after rollback")
                else:
                    log(f"  Cycle {cycle}: Rollback FAILED -- continuing with current checkpoint")

            # -- Save MetaController state -------------------------
            try:
                meta_controller.save_state(META_CONTROLLER_STATE)
            except Exception as e:
                log(f"  Cycle {cycle}: Failed to save MetaController state: {e}")

        # -- v11.5: Update BarrierTuner with cycle metrics ---------
        if barrier_tuner is not None:
            # Extract val_loss from retrain if available
            val_loss_val: float | None = None
            if retrain_result.get("retrained"):
                rl = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")
                try:
                    import re

                    with open(rl, encoding="utf-8", errors="replace") as f:
                        for line in f:
                            m = re.search(r"Val Loss:\s*([0-9]+\.[0-9]+)", line)
                            if m:
                                val_loss_val = float(m.group(1))
                except OSError:
                    pass

            bt_metrics = CycleMetrics(
                cycle=cycle,
                fitness=evo_result.get("final_fitness", 0.0),
                level=evo_result.get("final_level", 0),
                champion_words=evo_result.get("champion_words", 0),
                diversity=evo_result.get("final_fitness", 0.0),  # placeholder
                qd_occupied=evo_result.get("qd_occupied_cells", 0),
                qd_total=max(evo_result.get("qd_total_cells", 100), 1),
                novelty_size=evo_result.get("novelty_archive_size", 0),
                traces_count=evo_result.get("champion_traces", 0),
                retrained=retrain_result.get("retrained", False),
                val_loss=val_loss_val,
                generations_completed=evo_result.get("generations_completed", 0),
                mutation_rate=evo_result.get("mutation_rate", 0.5),
            )
            # Use diversity from evolution engine if available
            bt_metrics.diversity = evo_result.get("diversity", 0.3)

            barrier_info = barrier_tuner.update(bt_metrics)
            if barrier_info.barrier != BarrierType.NORMAL if BarrierType else True:
                log(
                    f"  Cycle {cycle}: BarrierTuner detected: "
                    f"{barrier_info.barrier.value} "
                    f"(x{barrier_info.consecutive_count}, "
                    f"severity={barrier_info.severity})"
                )
            last_archive_params = archive_params  # Save for next cycle

            # -- v11.5: Adaptive retrain budget tracking -------------
            if retrain_result.get("retrained"):
                # Extract actual epochs used from retrain log
                rl = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")
                try:
                    with open(rl, encoding="utf-8", errors="replace") as f:
                        log_content = f.read()
                    # Find last epoch number
                    epoch_matches = re.findall(r"Epoch\s+(\d+)/", log_content)
                    if epoch_matches:
                        # The last epoch logged tells how many were actually trained
                        actual_epochs = int(epoch_matches[-1])
                        # The "use" includes the epoch that ran
                        last_retrain_epochs = actual_epochs + 1
                        log(f"  Cycle {cycle}: Retrain used {last_retrain_epochs} epochs")
                except OSError:
                    pass

            # Update champion tracking for skip logic
            last_champion_fitness = evo_result.get("final_fitness", 0.0)
            last_champion_level = evo_result.get("final_level", 0)
            last_champion_words = evo_result.get("champion_words", 0)

            # -- v11.6: Auto-termination check ---------------------
            if barrier_tuner is not None and barrier_tuner.should_terminate():
                reason = barrier_tuner.get_termination_reason()
                log(f"  Cycle {cycle}: {reason}")
                log(f"  Cycle {cycle}: Stopping run -- no meaningful progress for extended period")
                # Break out of main loop
                break

        # Merge results
        if "error" in evo_result:
            log(f"  Cycle {cycle}: Evolution had errors -- see above")
        combined = {**evo_result, **retrain_result}

        # -- Phase D+E.2: Run training visibility diagnostics ------
        champion_prog = evo_result.get("champion_program")
        champion_level = evo_result.get("final_level", 0)
        champion_int: list[int] | None = None
        if champion_prog:
            try:
                champion_int = [int(w, 16) for w in champion_prog]
            except (ValueError, TypeError):
                pass
        retrain_ckpt = None
        if checkpoint and os.path.isfile(checkpoint):
            retrain_ckpt = checkpoint
        elif os.path.isfile(os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt")):
            retrain_ckpt = os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt")
        if retrain_ckpt:
            # Skip MemGen diagnostics — they use PrefixGenerator on CPU (~500s)
            diagnostics = None
            if diagnostics:
                combined["training_visibility"] = diagnostics

            # Phase L: Pass memorization data to BarrierTuner for next cycle
            if barrier_tuner is not None and barrier_tuner.history:
                mem_gen = diagnostics.get("memorization_generalization", {}) if diagnostics else {}
                mem_score = mem_gen.get("memorization_score")
                gen_score = mem_gen.get("generalization_score")
                if mem_score is not None or gen_score is not None:
                    last_bt = barrier_tuner.history[-1]
                    if mem_score is not None and isinstance(mem_score, (int, float)) and mem_score >= 0:
                        last_bt.memorization_score = mem_score
                        log(f"  Cycle {cycle}: BarrierTuner memorization_score={mem_score:.3f}")
                    if gen_score is not None and isinstance(gen_score, (int, float)) and gen_score >= 0:
                        last_bt.generalization_score = gen_score
                        log(f"  Cycle {cycle}: BarrierTuner generalization_score={gen_score:.3f}")
                    # Live diagnostic: warn on high mem_ratio (does not block retrain)
                    if mem_score is not None and gen_score is not None and gen_score > 0:
                        live_ratio = mem_score / gen_score
                        if live_ratio > 2.5:
                            log(
                                f"  ⚠️ Cycle {cycle}: HIGH MEM_RATIO={live_ratio:.1f}x (mem={mem_score:.3f}, gen={gen_score:.3f})"
                            )
                        elif live_ratio > 1.5:
                            log(
                                f"  ⚡ Cycle {cycle}: Elevated mem_ratio={live_ratio:.1f}x (mem={mem_score:.3f}, gen={gen_score:.3f})"
                            )
                else:
                    log(f"  Cycle {cycle}: BarrierTuner no mem/gen data available")

        all_results.append(combined)

        # -- Aggressive GPU model destruction at cycle boundary -----
        # The evolution engine's PrefixGenerator and the retrain model
        # both hold tensor references on CUDA. Simply calling empty_cache()
        # doesn't free them because the references still exist. Destroy
        # them explicitly so empty_cache can reclaim the memory.
        # This is the root cause of cycles 12+ all failing with CUDA
        # caching allocator fragmentation on the 12GB RTX 4070 Ti.
        try:
            import gc
            import torch
            if torch.cuda.is_available():
                # Force GC of any Python objects holding CUDA tensors
                gc.collect()
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated() / 1024**2
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                after = torch.cuda.memory_allocated() / 1024**2
                freed = before - after
                if freed > 50:
                    log(
                        f"  Cycle {cycle}: GPU memory cleanup: "
                        f"{before:.0f}MB -> {after:.0f}MB "
                        f"(freed {freed:.0f}MB)"
                    )
        except Exception as e:
            log(f"  Cycle {cycle}: GPU cleanup skipped ({e})")

        # v11: Cleanup stale checkpoints (keep last N + best)
        _cleanup_stale_checkpoints(cycle)

        # Append to results file
        try:
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(combined) + "\n")
            log(f"  Cycle {cycle}: Results saved to {RESULTS_FILE}")
        except (OSError, PermissionError) as e:
            log(f"  Cycle {cycle}: Failed to write results: {e}")

    # Print final summary
    print_summary(all_results)

    # Save to JSON
    summary_path = os.path.join(PROJECT_ROOT, "multi_cycle_summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"\nSummary saved: {summary_path}")
    except (OSError, PermissionError) as e:
        log(f"Summary save failed: {e}")
    log("Done.")


def _cleanup_stale_checkpoints(current_cycle, keep=5):
    import glob
    import os

    ckpt_dir = "/root/milestone2/../milestone1/checkpoints"
    # Use arithmetic range instead of file-sorting to prevent the bug where
    # OLD high-numbered checkpoints from a previous run displace NEW
    # low-numbered checkpoints from the current run.
    keep_set = set(range(max(1, current_cycle - keep + 1), current_cycle + 1))
    deleted = 0
    for f in glob.glob(os.path.join(ckpt_dir, "model_cycle*.pt")):
        try:
            n = int(os.path.basename(f).replace("model_cycle", "").replace(".pt", ""))
        except ValueError:
            continue
        if n not in keep_set:
            try:
                sz = os.path.getsize(f)
                os.remove(f)
                deleted += 1
                if deleted <= 3:
                    log(f"  [Cleanup] Deleted: {os.path.basename(f)} ({sz / 1e6:.0f}MB)")
            except (OSError, PermissionError):
                pass
    if deleted > 3:
        log(f"  [Cleanup] Deleted {deleted - 3} more stale checkpoints")

    # -- Cleanup *_full.pt files (cycle-chained full checkpoints) ------
    # Keep only the 2 most recent checkpoint_full_cycle_N.pt files.
    # These are ~524MB each with optimizer state, so discarding old ones
    # reclaims significant disk space.
    full_ckpt_files: list[tuple[int, str]] = []
    for f in glob.glob(os.path.join(ckpt_dir, "checkpoint_full_cycle_*.pt")):
        try:
            n = int(os.path.basename(f).replace("checkpoint_full_cycle_", "").replace(".pt", ""))
        except ValueError:
            continue
        full_ckpt_files.append((n, f))
    full_ckpt_files.sort(key=lambda x: x[0])  # ascending by cycle number
    if len(full_ckpt_files) > 2:
        for n, f in full_ckpt_files[:-2]:
            try:
                sz = os.path.getsize(f)
                os.remove(f)
                log(f"  [Cleanup] Deleted old full checkpoint: {os.path.basename(f)} ({sz / 1e6:.0f}MB)")
            except (OSError, PermissionError):
                pass


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
    for level in range(0, champion_level + 1):
        try:
            task = get_task_level(level)
            inputs = task.get("inputs", {})
            expected = task.get("expected_output", {}).get("registers_checked", {})
            result = vm_farm.evaluate(champion, inputs, budget=budget)
            status = getattr(result, "success", True)
            error = getattr(result, "error", None)
            trace = {
                "status": "ok" if status else ("crash" if error else "ok"),
                "final_regs": getattr(result, "registers", {}),
                "final_memory": getattr(result, "memory", {}),
                "instructions_count": getattr(result, "cycles", 0),
                "trap_cause": error or "",
            }
            # Compute raw task score (without novelty/QD blending)
            correct = 0
            total = len(expected)
            final_regs = trace.get("final_regs", {})
            for reg_idx, expected_val in expected.items():
                if final_regs.get(reg_idx, 0) == expected_val:
                    correct += 1
            reg_match = correct / total if total > 0 else 1.0
            bloat = max(0, (trace.get("instructions_count", 0) - 10) * 0.01)
            raw_score = reg_match * 100.0 - bloat
            if status and reg_match == 1.0:
                raw_score += 50.0
            scores[level] = raw_score
        except Exception:
            scores[level] = 0.0

    # Log summary of forgetting curve
    perfect = sum(1 for s in scores.values() if s >= 140.0)
    partial = sum(1 for s in scores.values() if 0 < s < 140.0)
    failed = sum(1 for s in scores.values() if s == 0.0)
    from logging import info

    info(f"[ForgettingCurve] L0-L{champion_level}: {perfect} perfect, {partial} partial, {failed} failed")
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
            logging.warning(f"[WeightDelta] before_path not found: {before_path}")
            return {}
        if not os.path.isfile(after_path):
            logging.warning(f"[WeightDelta] after_path not found: {after_path}")
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
                if norm == norm:  # skip nan (from -inf - -inf comparisons)
                    deltas[key] = round(norm, 4)

        if deltas:
            max_key = max(deltas, key=lambda k: deltas[k])
            logging.info(f"[WeightDelta] {len(deltas)} params, max={deltas[max_key]:.4f} ({max_key})")
        return deltas
    except Exception:
        return {}


# ── Phase J: Stratified memorization sampling (extracted for testability) ──


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
        logging.warning("[MemGen] No mem_examples found for stratified sampling")
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
        logging.warning("[MemGen] No valid level-grouped examples found")
        return []

    # Log level distribution before stratification
    level_counts = {k: len(v) for k, v in sorted(level_groups.items())}
    logging.info(
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
    logging.info(f"[MemGen] Level distribution after stratified sample: {stratified_counts} ({len(stratified)} taken)")

    # Fill remaining quota with shuffled remaining examples
    remaining = [ex for ex in mem_examples if ex not in stratified]
    rng.shuffle(remaining)
    filler = remaining[: max(0, num_samples - len(stratified))]
    result = stratified + filler

    logging.info(
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
            except Exception:
                accuracy = 0.0

            scores.setdefault(level, []).append(accuracy)
        return scores

    try:
        from mutation_proposer import _encode_inst
        from prefix_generator import PrefixGenerator as PG
        from tasks_rv32i import reference_solution
    except ImportError as e:
        logging.warning(f"[MemGen] Missing dependencies ({e}) -- skipping diagnostic")
        return {}

    if not os.path.isfile(checkpoint_path):
        logging.warning(f"[MemGen] Checkpoint not found: {checkpoint_path}")
        return {}
    if not os.path.isfile(recurrent_data_path):
        logging.warning(f"[MemGen] No recurrent data at {recurrent_data_path}")
        return {}

    try:
        gen = PG(checkpoint_path, temperature=0.5, max_new_tokens=32, verbose=False)
    except Exception as e:
        logging.warning(f"[MemGen] Failed to load PrefixGenerator: {e}")
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
    except Exception as e:
        logging.warning(f"[MemGen] Failed to read recurrent data: {e}")
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
        except Exception as exc:
            logging.warning(f"[MemGen] Failed to build ref example for L{level}: {exc}")
            continue

    if not gen_tests:
        logging.warning("[MemGen] No generalization examples generated")
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

    logging.info(
        f"[MemGen] Mem={result['memorization_score']:.3f} "
        f"({result['memorization_examples']} ex), "
        f"Gen={result['generalization_score']:.3f} "
        f"({result['generalization_examples']} ex, "
        f"{len({e['level'] for e in gen_tests})} levels)"
    )
    return result


# -- Phase D: Wire training visibility diagnostics into cycle loop -----
def _run_training_visibility_diagnostics(
    cycle: int,
    checkpoint: str,
    champion: list[int] | None,
    champion_level: int,
    recurrent_data_path: str,
) -> dict[str, Any]:
    """Run all training visibility diagnostics for a cycle.

    Calls compute_forgetting_curve, compute_weight_delta, and
    compute_memorization_generalization, returning their results as a dict.
    Silently handles failures (returns {} for each diagnostic on error).
    """
    diagnostics: dict[str, Any] = {}
    import logging

    log_fn = logging.info

    # -- Forgetting curve ------------------------------------------
    if champion is not None and champion_level >= 1:
        try:
            from evolution_engine import VMFarm
            from run_multi_cycle import compute_forgetting_curve

            vm = VMFarm(num_workers=1)
            fc = compute_forgetting_curve(champion, champion_level, vm)
            if fc:
                diagnostics["forgetting_curve"] = fc
                log_fn(f"  Cycle {cycle}: Forgetting curve: {len(fc)} levels tracked")
        except Exception as e:
            log_fn(f"  Cycle {cycle}: Forgetting curve skipped: {e}")

    # -- Weight delta ----------------------------------------------
    try:
        before_ckpt = os.path.join(os.path.dirname(checkpoint), "checkpoint_best.pt")
        after_ckpt = checkpoint
        wd = compute_weight_delta(before_ckpt, after_ckpt)
        if wd:
            diagnostics["weight_delta"] = wd
            max_key = max(wd, key=lambda k: wd[k])
            log_fn(f"  Cycle {cycle}: Weight delta: {len(wd)} params, max={wd[max_key]:.4f} ({max_key})")
    except Exception as e:
        log_fn(f"  Cycle {cycle}: Weight delta skipped: {e}")

    # -- Memorization vs Generalization (Phase E.2) ----------------
    try:
        mem_gen = compute_memorization_generalization(
            checkpoint,
            recurrent_data_path,
            num_samples=20,
            prefix_instructions=2,
        )
        if mem_gen:
            diagnostics["memorization_generalization"] = mem_gen
            log_fn(
                f"  Cycle {cycle}: Mem/Gen: "
                f"mem={mem_gen.get('memorization_score', '?'):.3f} "
                f"gen={mem_gen.get('generalization_score', '?'):.3f}"
            )
    except Exception as e:
        log_fn(f"  Cycle {cycle}: Mem/Gen skipped: {e}")

    # -- Gradient norms (Phase F) -------------------------------------
    try:
        retrain_log = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")
        from meta_controller import parse_grad_norms_from_retrain_log

        gn = parse_grad_norms_from_retrain_log(retrain_log)
        if gn:
            diagnostics["gradient_norms"] = gn
            # Log summary: mean and max
            norms_list = list(gn.values())
            mean_norm = sum(norms_list) / len(norms_list)
            max_norm = max(norms_list)
            max_key = max(gn, key=gn.get)
            log_fn(
                f"  Cycle {cycle}: Gradient norms: "
                f"mean={mean_norm:.6f}, max={max_norm:.6f} ({max_key}), "
                f"{len(gn)} params tracked"
            )
    except Exception as e:
        log_fn(f"  Cycle {cycle}: Gradient norms skipped: {e}")

    return diagnostics


if __name__ == "__main__":
    main()
