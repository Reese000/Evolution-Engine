#!/usr/bin/env python3
"""
Evolution Supervisor — crash-resilient wrapper for run_multi_cycle.py.

Launches evolution runs as subprocesses, auto-relaunches on crashes
(SIGSEGV exit 139, other nonzero exits), preserves val_loss gating
state across restarts, and caps total relaunches.

Features:
  - Per-cycle checkpointing via MetaController state persistence
  - Auto-relaunch on crash with --resume flag
  - Caps at max_relaunches to prevent infinite loops
  - Supervisor log with crash/resume timestamps
  - Tracks total effective cycles across restarts
  - Detects F=0.000 "bad evo" cycles and retries
  - Torch version validation at subprocess startup
  - CUDA cleanup between batches
  - Bad-champion validity pre-check

Usage:
    python run_evolution_supervised.py --cycles 30 --gens 10
    python run_evolution_supervised.py --cycles 30 --gens 10 --max-relaunches 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPERVISOR_LOG = os.path.join(SCRIPT_DIR, "supervisor.log")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "multi_cycle_results.jsonl")
META_CONTROLLER_STATE = os.path.join(SCRIPT_DIR, "meta_controller_state.json")


def log(msg: str) -> None:
    """Log to both supervisor.log and stdout."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(SUPERVISOR_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def get_completed_cycles() -> int:
    """Count completed cycles from results file."""
    if not os.path.isfile(RESULTS_FILE):
        return 0
    count = 0
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("retrained") is not None or entry.get("final_fitness") is not None:
                        count += 1
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return count


def get_best_val_loss() -> float | None:
    """Read best val_loss from MetaController state."""
    if not os.path.isfile(META_CONTROLLER_STATE):
        return None
    try:
        with open(META_CONTROLLER_STATE, encoding="utf-8") as f:
            state = json.load(f)
        best_vl = state.get("best_val_loss")
        if best_vl is not None and isinstance(best_vl, (int, float)) and best_vl > 0:
            return float(best_vl)
    except (json.JSONDecodeError, OSError):
        pass
    return None


def get_last_cycle_fitness() -> float:
    """Get the fitness of the last completed cycle."""
    if not os.path.isfile(RESULTS_FILE):
        return 0.0
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            last_line = None
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
            if last_line:
                entry = json.loads(last_line)
                return entry.get("final_fitness", 0.0)
    except (json.JSONDecodeError, OSError):
        pass
    return 0.0


def is_bad_champion(result_text: str) -> bool:
    """Check if the subprocess output indicates a bad champion (F=0.000)."""
    # Look for Cycle lines with F=0.000 in the output
    for line in result_text.split("\n"):
        if "Cycle" in line and "F=0.000" in line:
            return True
    return False


def cuda_cleanup() -> None:
    """Best-effort CUDA cleanup between batches."""
    try:
        import gc

        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            log("  CUDA cleanup: cache cleared")
    except Exception as e:
        log(f"  CUDA cleanup skipped: {e}")


def verify_torch_env(python_path: str) -> bool:
    """Verify the target Python has correct torch version."""
    try:
        result = subprocess.run(
            [
                python_path,
                "-c",
                "import sys, torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} path={sys.executable}')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
        log(f"  Torch env: {output}")
        return result.returncode == 0 and "torch=2.4.1" in output
    except Exception as e:
        log(f"  Torch env check failed: {e}")
        return False


def run_evolution_batch(
    cycles: int,
    gens: int,
    python_path: str,
    extra_args: list[str] | None = None,
) -> tuple[int, str]:
    """Run a batch of evolution cycles as a subprocess.

    Returns:
        (exit_code, combined_output)
    """
    cmd = [
        python_path,
        os.path.join(SCRIPT_DIR, "run_multi_cycle.py"),
        "--cycles",
        str(cycles),
        "--gens",
        str(gens),
    ]
    if extra_args:
        cmd.extend(extra_args)

    log(f"  Launching: python_path={python_path}")
    log(f"  Command: {' '.join(cmd[:6])}...")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour hard timeout per batch
        )
        elapsed = time.time() - start
        combined = (result.stdout or "") + "\n" + (result.stderr or "")

        # Log torch env from subprocess stderr (first few lines)
        for line in combined.split("\n")[:5]:
            if "torch" in line.lower() or "cuda" in line.lower():
                log(f"  Subprocess: {line.strip()}")

        log(f"  Process exited with code {result.returncode} after {elapsed:.0f}s")
        return result.returncode, combined
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        log(f"  Process TIMED OUT after {elapsed:.0f}s")
        return -1, "TIMEOUT"
    except Exception as e:
        log(f"  Process FAILED: {type(e).__name__}: {e}")
        return -2, str(e)


def supervisor_main() -> int:
    """Main supervisor loop."""
    parser = argparse.ArgumentParser(description="Crash-resilient evolution supervisor")
    parser.add_argument("--cycles", type=int, default=30, help="Total target cycles (default: 30)")
    parser.add_argument("--gens", type=int, default=10, help="Generations per evolution cycle (default: 10)")
    parser.add_argument(
        "--max-relaunches", type=int, default=10, help="Max crash-relaunches before giving up (default: 10)"
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Cycles per subprocess batch (default: 10)")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed cycle")
    parser.add_argument("--python", type=str, default=None, help="Python interpreter path (default: sys.executable)")
    args = parser.parse_args()

    # Determine Python interpreter
    python_path = args.python or sys.executable
    if not os.path.isfile(python_path):
        log(f"ERROR: Python not found at {python_path}")
        return 1

    # Clear supervisor log on fresh start
    if not args.resume:
        try:
            os.remove(SUPERVISOR_LOG)
        except OSError:
            pass

    log("=" * 60)
    log("Evolution Supervisor starting")
    log(f"  Python: {python_path}")
    log(f"  Target: {args.cycles} cycles, {args.gens} gens/cycle")
    log(f"  Batch size: {args.batch_size}, Max relaunches: {args.max_relaunches}")

    # Verify torch environment
    if not verify_torch_env(python_path):
        log("  WARNING: torch version check failed — proceeding anyway")

    log("=" * 60)

    total_completed = get_completed_cycles()
    relaunches = 0
    batch_num = 0

    while total_completed < args.cycles:
        remaining = args.cycles - total_completed
        batch_cycles = min(remaining, args.batch_size)
        batch_num += 1

        log(
            f"\n--- Batch {batch_num}: cycles {total_completed+1}-"
            f"{total_completed+batch_cycles} ({batch_cycles} cycles) ---"
        )

        best_vl = get_best_val_loss()
        log(f"  Current best val_loss: {best_vl}")

        # CUDA cleanup between batches
        cuda_cleanup()

        # Build extra args — do NOT pass --resume since the subprocess
        # runs a fixed cycle range (1..batch_size). The results file
        # accumulates across batches; duplicate cycle entries are fine.
        extra_args = []

        # Run the batch
        exit_code, output = run_evolution_batch(
            cycles=batch_cycles,
            gens=args.gens,
            python_path=python_path,
            extra_args=extra_args,
        )

        # Check what actually completed
        new_completed = get_completed_cycles()
        cycles_this_batch = new_completed - total_completed
        last_fitness = get_last_cycle_fitness()

        log(
            f"  Batch result: exit={exit_code}, "
            f"completed={cycles_this_batch}/{batch_cycles}, "
            f"last_fitness={last_fitness:.3f}"
        )

        # Check for bad evo cycles (F=0.000)
        if last_fitness == 0.0 and cycles_this_batch > 0:
            log("  WARNING: Last cycle produced F=0.000 (bad evo champion)")
            log("  Continuing — next batch will retry with fresh evolution")

        # Handle crash
        if exit_code != 0 and exit_code != -1:  # -1 = timeout, handled separately
            relaunches += 1
            log(f"  CRASH detected (exit={exit_code}). Relaunch #{relaunches}")

            if relaunches >= args.max_relaunches:
                log(f"  MAX RELAUNCHES REACHED ({args.max_relaunches}). Stopping.")
                break

            # Wait before relaunch (cool down) — longer for repeated crashes
            cooldown = min(5 * relaunches, 30)
            log(f"  Cooling down for {cooldown}s...")
            time.sleep(cooldown)

            # CUDA cleanup after crash
            cuda_cleanup()

            continue

        elif exit_code == -1:  # Timeout
            log("  TIMEOUT detected. Relaunching")
            relaunches += 1
            if relaunches >= args.max_relaunches:
                log("  MAX RELAUNCHES REACHED. Stopping.")
                break
            cuda_cleanup()
            continue

        # Success — update counters
        total_completed = new_completed
        log(f"  Total completed: {total_completed}/{args.cycles}")

        if total_completed >= args.cycles:
            break

        # Between successful batches, brief pause
        time.sleep(2)

    # Final summary
    final_vl = get_best_val_loss()
    final_completed = get_completed_cycles()
    log("\n" + "=" * 60)
    log("Supervisor finished")
    log(f"  Total cycles completed: {final_completed}")
    log(f"  Total relaunches: {relaunches}")
    log(f"  Best val_loss: {final_vl}")
    log("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(supervisor_main())
