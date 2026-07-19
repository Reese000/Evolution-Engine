#!/usr/bin/env python3
"""
Evolution Supervisor — crash-resilient wrapper for run_multi_cycle.py.

Launches evolution runs as subprocesses, auto-relaunches on crashes
(SIGSEGV exit 139 / ACCESS_VIOLATION 0xC0000005), preserves val_loss
gating state across restarts, and caps total relaunches.

Key design:
  - Each subprocess runs with --resume so it skips already-completed cycles
  - The results file accumulates across batches (never deleted)
  - meta_controller_state.json is NEVER deleted (preserves val_loss 0.0823)
  - On crash, CUDA cleanup + cooldown, then relaunch with --resume
  - Torch version validated at startup (must be 2.4.1+cu121)

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

# Required torch version for CUDA stability on RTX 4070 Ti
REQUIRED_TORCH_VERSION = "2.4.1"
REQUIRED_CUDA_VERSION = "12.1"


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
                    # Count entries that have retrained key OR final_fitness
                    if entry.get("retrained") is not None or entry.get("final_fitness") is not None:
                        count += 1
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return count


def get_completed_cycle_numbers() -> set[int]:
    """Get set of completed cycle numbers from results file."""
    if not os.path.isfile(RESULTS_FILE):
        return set()
    numbers = set()
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    c = entry.get("cycle")
                    if c is not None:
                        numbers.add(c)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return numbers


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


def cuda_cleanup() -> None:
    """Best-effort CUDA cleanup between batches.

    R49: Skip if CUDA not initialized — initializing CUDA in the supervisor
    process causes ACCESS_VIOLATION in subprocesses (torch 2.4.1+cu121 on
    Windows). The supervisor doesn't use CUDA; only subprocesses do.
    """
    try:
        import gc

        import torch

        # R49: Don't initialize CUDA in supervisor — it interferes with subprocesses
        if not torch.cuda.is_initialized():
            return
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        alloc = torch.cuda.memory_allocated() / 1024**2
        log(f"  CUDA cleanup: {alloc:.0f}MB allocated")
    except Exception as e:
        log(f"  CUDA cleanup skipped: {e}")


def verify_torch_env(python_path: str) -> bool:
    """Verify the target Python has correct torch version. Returns False if wrong."""
    check_script = (
        "import sys, torch; "
        "v = torch.__version__; c = torch.cuda.is_available(); "
        "cv = torch.version.cuda; print(f'torch={v} cuda={c} cuda_ver={cv} path={sys.executable}')"
    )
    try:
        result = subprocess.run(
            [python_path, "-c", check_script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout.strip()
        log(f"  Torch env: {output}")

        # Check torch version
        if f"torch={REQUIRED_TORCH_VERSION}" not in output:
            log(f"  FATAL: torch version != {REQUIRED_TORCH_VERSION} — aborting")
            return False
        if f"cuda_ver={REQUIRED_CUDA_VERSION}" not in output:
            log(f"  FATAL: CUDA version != {REQUIRED_CUDA_VERSION} — aborting")
            return False
        if "cuda=True" not in output:
            log("  FATAL: CUDA not available — aborting")
            return False

        return True
    except Exception as e:
        log(f"  FATAL: Torch env check failed: {e}")
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

    log(f"  Command: {' '.join(cmd)}")
    start = time.time()

    # Propagate EE_USE_GPU to subprocess for crash mitigation
    env = os.environ.copy()

    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour hard timeout per batch
            env=env,
        )
        elapsed = time.time() - start
        combined = (result.stdout or "") + "\n" + (result.stderr or "")

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
    parser.add_argument("--python", type=str, default=None, help="Python interpreter path (default: sys.executable)")
    args = parser.parse_args()

    # Determine Python interpreter
    python_path = args.python or sys.executable
    if not os.path.isfile(python_path):
        log(f"ERROR: Python not found at {python_path}")
        return 1

    # Clear supervisor log on fresh start (NOT results or meta_controller state)
    try:
        os.remove(SUPERVISOR_LOG)
    except OSError:
        pass

    log("=" * 60)
    log("Evolution Supervisor starting")
    log(f"  Python: {python_path}")
    log(f"  Target: {args.cycles} cycles, {args.gens} gens/cycle")
    log(f"  Batch size: {args.batch_size}, Max relaunches: {args.max_relaunches}")

    # Verify torch environment — ABORT if wrong
    if not verify_torch_env(python_path):
        log("ABORTING: torch version verification failed")
        return 1

    # R51: Pre-flight VRAM check — abort if not enough free VRAM
    # EE_VRAM_MIN_GB overrides (for experiments on dirty VRAM)
    vram_min = float(os.environ.get("EE_VRAM_MIN_GB", "9.0"))
    vram_ok, vram_msg = check_vram_free(min_gb=vram_min)
    if not vram_ok:
        log(f"ABORTING: {vram_msg}")
        log("Kill orphaned CUDA processes or reboot, then retry.")
        return 1
    log(f"  VRAM check: {vram_msg} (threshold={vram_min}GB)")

    # Kill any orphaned EE child processes from previous runs
    killed = kill_ee_subtree()
    if killed:
        log(f"  Killed {killed} orphaned EE process(es)")

    # Show preserved state
    best_vl = get_best_val_loss()
    completed = get_completed_cycles()
    log(f"  Preserved best val_loss: {best_vl}")
    log(f"  Already completed cycles: {completed}")
    log("=" * 60)

    total_completed = completed
    relaunches = 0
    batch_num = 0

    while total_completed < args.cycles:
        remaining = args.cycles - total_completed
        batch_cycles = min(remaining, args.batch_size)
        # The subprocess needs --cycles = total_completed + batch_cycles
        # so that --resume skips the first total_completed and runs the new ones.
        subprocess_cycles = total_completed + batch_cycles
        batch_num += 1

        log(
            f"\n--- Batch {batch_num}: requesting {batch_cycles} new cycles "
            f"(subprocess --cycles {subprocess_cycles} --resume) ---"
        )

        best_vl = get_best_val_loss()
        log(f"  Current best val_loss: {best_vl}")

        # CUDA cleanup between batches
        cuda_cleanup()

        # KEY: pass --resume so subprocess skips already-completed cycles
        # The results file accumulates; completed cycles get skipped.
        extra_args = ["--resume"]

        # Run the batch
        exit_code, output = run_evolution_batch(
            cycles=subprocess_cycles,
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
            f"new_cycles={cycles_this_batch}/{batch_cycles}, "
            f"total={new_completed}/{args.cycles}, "
            f"last_fitness={last_fitness:.3f}"
        )

        # Check for bad evo cycles (F=0.000)
        if last_fitness == 0.0 and cycles_this_batch > 0:
            log("  WARNING: Last cycle produced F=0.000 (bad evo champion)")

        # Handle crash
        if exit_code != 0 and exit_code != -1:
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


def check_vram_free(min_gb: float = 9.0) -> tuple[bool, str]:
    """Check if GPU has enough free VRAM. Returns (ok, message)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            free_mb, total_mb = [int(x.strip()) for x in result.stdout.strip().split(",")]
            free_gb = free_mb / 1024
            total_gb = total_mb / 1024
            if free_gb < min_gb:
                return (
                    False,
                    f"Only {free_gb:.1f}GB free of {total_gb:.1f}GB (need {min_gb}GB). Reboot or kill orphaned CUDA processes.",
                )
            return True, f"{free_gb:.1f}GB free of {total_gb:.1f}GB"
    except Exception as e:
        return False, f"Could not query GPU: {e}"
    return False, "Could not query GPU"


def kill_ee_subtree() -> int:
    """Kill any orphaned EE child processes. Returns count killed."""
    killed = 0
    try:
        # Find any python processes running EE scripts
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                'Get-CimInstance Win32_Process -Filter "'
                "CommandLine LIKE '%run_multi_cycle%' OR "
                "CommandLine LIKE '%run_evolution_supervised%' OR "
                "CommandLine LIKE '%augmentation_subprocess%'"
                '" | Select-Object -ExpandProperty ProcessId',
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for pid_str in result.stdout.strip().split():
                try:
                    pid = int(pid_str.strip())
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5)
                    log(f"  Killed orphaned EE process PID {pid}")
                    killed += 1
                except (ValueError, subprocess.TimeoutExpired):
                    pass
    except Exception:
        pass
    return killed


if __name__ == "__main__":
    sys.exit(supervisor_main())
