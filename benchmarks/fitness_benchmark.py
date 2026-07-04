#!/usr/bin/env python3
"""fitness_benchmark.py — Performance benchmark suite for the Evolution Engine.

Runs N independent evolution cycles with deterministic seeds, records fitness
curves, wall times, and convergence metrics, then aggregates statistics across
runs. Saves results to benchmarks/results.json and compares against any
previous baseline.

Usage:
    python3 benchmarks/fitness_benchmark.py --runs 3 --generations 50
    python3 benchmarks/fitness_benchmark.py --population-size 25 --runs 5
    python3 benchmarks/fitness_benchmark.py --generations 20 --runs 1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Path setup ──────────────────────────────────────────────────────────
# Ensure project root is on sys.path before any project imports.

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_BENCHMARKS_DIR = Path(__file__).resolve().parent
_RESULTS_FILE = _BENCHMARKS_DIR / "results.json"

# Per-run timeout: 30 minutes
_PER_RUN_TIMEOUT_SECONDS: int = 30 * 60


# ── Timeout helper ──────────────────────────────────────────────────────


class _TimeoutError(Exception):
    """Raised when a benchmark run exceeds the time limit."""


def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise _TimeoutError("Benchmark run exceeded time limit")


# ── Import project modules ──────────────────────────────────────────────


def _import_modules() -> tuple[Any, Any, Any]:
    """Import EvolutionEngine, EvolutionConfig, and VMFarm from the project.

    Returns:
        Tuple of (EvolutionEngine, EvolutionConfig, VMFarm).

    Raises:
        ImportError: If any required module cannot be imported.
    """
    try:
        from evolution_engine import EvolutionConfig, EvolutionEngine, VMFarm
    except ImportError as exc:
        raise ImportError(
            "Cannot import evolution_engine module. Ensure the project " f"is properly installed. Original error: {exc}"
        ) from exc

    return EvolutionEngine, EvolutionConfig, VMFarm


def _build_evolution_config(
    population_size: int,
    num_generations: int,
    seed: int,
) -> Any:
    """Build an EvolutionConfig from the project's Config class.

    Loads defaults from config/engine_config.json via the Config class,
    then applies benchmark-specific overrides (CPU-only, no I/O, etc.).

    Args:
        population_size: Number of individuals per generation.
        num_generations: Number of generations to evolve.
        seed: RNG seed for deterministic runs.

    Returns:
        EvolutionConfig instance ready for EvolutionEngine.
    """
    from config import Config as ConfigClass

    # Load config from default path (config/engine_config.json)
    cfg = ConfigClass()

    _, EvolutionConfig, _ = _import_modules()

    # Map Config.evolution section to EvolutionConfig dataclass fields.
    # We use the Config class values where applicable, then override
    # with benchmark-specific settings.
    evo_section = cfg.evolution

    evo_config = EvolutionConfig(
        # Core evolution params from Config class
        population_size=population_size,
        max_generations=num_generations,
        seed=seed,
        mutation_rate=evo_section.mutation_rate,
        crossover_rate=evo_section.crossover_rate,
        elite_ratio=evo_section.elite_ratio,
        tournament_size=evo_section.tournament_size,
        instruction_budget=evo_section.instruction_budget,
        eval_budget=evo_section.eval_budget,
        # Force CPU-only (no GPU)
        use_gpu=False,
        # Benchmark-specific: disable features that add overhead
        # without affecting core fitness evaluation
        auto_curriculum=False,
        curriculum_level=0,
        checkpoint_interval=0,  # no checkpointing overhead
        strategy_archive_path="",  # no file I/O
        champion_dir="",  # no champion loading
        log_dir="",  # no log file I/O
        dashboard_state_dir="",  # no dashboard state
        novelty_archive_path="",  # no archive file I/O
        qd_archive_path="",  # no archive file I/O
        use_novelty_archive=False,  # disable novelty archive
        use_qd_archive=False,  # disable QD archive
        use_strategy_archive=False,  # disable strategy archive
        use_secondary_tasks=False,  # disable secondary task eval
        use_circuit_breaker=False,  # disable circuit breaker
        recurrent_feedback_dir="",  # no recurrent feedback
        # Misc params from Config class
        stall_threshold=evo_section.stall_threshold,
        max_level=evo_section.max_level,
        max_champion_words=evo_section.max_champion_words,
        num_eval_seeds=evo_section.num_eval_seeds,
        diversity_threshold=evo_section.diversity_threshold,
        diversity_inject_ratio=evo_section.diversity_inject_ratio,
        champion_crossover_prob=evo_section.champion_crossover_prob,
        crossover_dedicated_prob=evo_section.crossover_dedicated_prob,
        max_consecutive_same_strategy=evo_section.max_consecutive_same_strategy,
        novelty_weight=evo_section.novelty_weight,
        qd_bonus_weight=evo_section.qd_bonus_weight,
    )

    return evo_config


# ── Single run execution ────────────────────────────────────────────────


def _run_single_benchmark(
    run_index: int,
    population_size: int,
    num_generations: int,
    seed: int,
    timeout_seconds: int = _PER_RUN_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Execute a single benchmark run and return structured results.

    Creates an EvolutionEngine with CPU-only settings, runs the core
    evolution loop for the specified number of generations, and records
    performance metrics.

    Args:
        run_index: Zero-based run index (for logging).
        population_size: Number of individuals per generation.
        num_generations: Number of generations to evolve.
        seed: RNG seed for deterministic reproduction.
        timeout_seconds: Maximum wall-clock seconds before timing out.

    Returns:
        Dict with keys: best_fitness, wall_time, convergence_generation,
        fitness_curve, secondary_task_scores, status, error.
    """
    EngineClass, _, VMFarmClass = _import_modules()

    result: dict[str, Any] = {
        "best_fitness": 0.0,
        "wall_time": 0.0,
        "convergence_generation": None,
        "fitness_curve": [],
        "secondary_task_scores": {},
        "status": "pending",
        "error": None,
    }

    # Set up per-run timeout via SIGALRM (Unix only; graceful skip on Windows)
    use_alarm = hasattr(signal, "SIGALRM") and os.name == "posix"
    old_handler = None

    print(f"  [Run {run_index}] seed={seed}, pop={population_size}, gens={num_generations}")

    try:
        # Apply timeout signal handler
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_seconds)

        # Build evolution config with CPU-only, no retrain, no external dirs
        evo_config = _build_evolution_config(
            population_size=population_size,
            num_generations=num_generations,
            seed=seed,
        )

        # Create engine: CPU-only VMFarm, no GPU
        vm_farm = VMFarmClass(num_workers=1, use_gpu=False)
        engine = EngineClass(config=evo_config, vm_farm=vm_farm)

        # Run the evolution loop
        start_time = time.time()
        summary = engine.run(num_generations=num_generations)
        wall_time = time.time() - start_time

        # Extract fitness curve from engine state
        fitness_curve = list(engine.best_fitness_history)
        # Pad curve if engine returned fewer entries than expected
        while len(fitness_curve) < num_generations:
            fitness_curve.append(fitness_curve[-1] if fitness_curve else 0.0)

        final_fitness = summary.get("final_fitness", 0.0)
        if not isinstance(final_fitness, (int, float)):
            final_fitness = 0.0

        # Compute convergence generation: first gen where fitness > 90% of final
        convergence_gen = None
        if final_fitness > 0:
            threshold = final_fitness * 0.9
            for i, f_val in enumerate(fitness_curve):
                if isinstance(f_val, (int, float)) and f_val >= threshold:
                    convergence_gen = i
                    break

        # Secondary task scores (instruction count, diversity, mutation rate)
        secondary_scores = {
            "instruction_count": summary.get("champion_words", 0),
            "code_diversity": summary.get("final_diversity", 0.0),
            "mutation_rate": evo_config.mutation_rate,
            "generations_stalled": summary.get("generations_stalled", 0),
            "total_improvements": summary.get("total_champion_improvements", 0),
        }

        result.update(
            {
                "best_fitness": final_fitness,
                "wall_time": round(wall_time, 3),
                "convergence_generation": convergence_gen,
                "fitness_curve": [round(f, 4) if isinstance(f, (int, float)) else 0.0 for f in fitness_curve],
                "secondary_task_scores": secondary_scores,
                "status": "ok",
                "error": summary.get("error"),
            }
        )

        print(
            f"  [Run {run_index}] completed: fitness={final_fitness:.2f}, "
            f"time={wall_time:.1f}s, convergence_gen={convergence_gen}"
        )

    except _TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Timed out after {timeout_seconds}s"
        print(f"  [Run {run_index}] TIMED OUT after {timeout_seconds}s")

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [Run {run_index}] ERROR: {exc}")
        traceback.print_exc()

    finally:
        # Cancel alarm and restore old handler
        if use_alarm:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    return result


# ── Statistics aggregation ───────────────────────────────────────────────


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean, std, min, max statistics across benchmark runs.

    Only considers runs with status == 'ok' for numeric aggregation.
    If all runs failed, returns zeroed stats with a note.

    Args:
        runs: List of run result dicts.

    Returns:
        Summary dict with mean_fitness, std_fitness, min_fitness, max_fitness,
        mean_wall_time, std_wall_time, mean_convergence_gen, runs_completed,
        runs_total, runs_failed.
    """
    total = len(runs)
    ok_runs = [r for r in runs if r["status"] == "ok"]

    if not ok_runs:
        return {
            "mean_fitness": 0.0,
            "std_fitness": 0.0,
            "min_fitness": 0.0,
            "max_fitness": 0.0,
            "mean_wall_time": 0.0,
            "std_wall_time": 0.0,
            "mean_convergence_gen": None,
            "convergence_std": None,
            "runs_completed": 0,
            "runs_total": total,
            "runs_failed": total,
            "note": "All runs failed or timed out",
        }

    fitness_values = [r["best_fitness"] for r in ok_runs]
    wall_times = [r["wall_time"] for r in ok_runs]
    convergence_gens = [r["convergence_generation"] for r in ok_runs if r["convergence_generation"] is not None]

    def _mean(vals: list[float | int]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals: list[float | int]) -> float:
        if len(vals) < 2:
            return 0.0
        m = _mean(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    def _min_val(vals: list[float | int]) -> float:
        return min(vals) if vals else 0.0

    def _max_val(vals: list[float | int]) -> float:
        return max(vals) if vals else 0.0

    conv_mean = _mean(convergence_gens) if convergence_gens else None
    conv_std = _std(convergence_gens) if len(convergence_gens) >= 2 else None

    return {
        "mean_fitness": round(_mean(fitness_values), 4),
        "std_fitness": round(_std(fitness_values), 4),
        "min_fitness": round(_min_val(fitness_values), 4),
        "max_fitness": round(_max_val(fitness_values), 4),
        "mean_wall_time": round(_mean(wall_times), 3),
        "std_wall_time": round(_std(wall_times), 3),
        "min_wall_time": round(_min_val(wall_times), 3),
        "max_wall_time": round(_max_val(wall_times), 3),
        "mean_convergence_gen": round(conv_mean, 1) if conv_mean is not None else None,
        "convergence_std": round(conv_std, 1) if conv_std is not None else None,
        "runs_completed": len(ok_runs),
        "runs_total": total,
        "runs_failed": total - len(ok_runs),
    }


# ── Baseline comparison ─────────────────────────────────────────────────


def _load_baseline() -> dict[str, Any] | None:
    """Load previous benchmark results for comparison.

    Returns:
        Previous results dict, or None if no baseline exists.
    """
    if not _RESULTS_FILE.is_file():
        return None
    try:
        with open(_RESULTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        # Validate it has expected structure
        if "summary" in data and "config" in data:
            return data
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def _compute_baseline_comparison(
    current_summary: dict[str, Any],
    current_config: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compute deltas between current and baseline benchmark results.

    Args:
        current_summary: Current run summary stats.
        current_config: Current run configuration.
        baseline: Previous baseline results dict.

    Returns:
        Comparison dict with absolute and percentage deltas.
    """
    prev_summary = baseline.get("summary", {})
    prev_config = baseline.get("config", {})

    def _delta(curr: float | None, prev: float | None) -> dict[str, Any]:
        if curr is None or prev is None:
            return {"current": curr, "previous": prev, "delta": None, "delta_pct": None}
        d = curr - prev
        pct = (d / abs(prev) * 100) if prev != 0 else None
        return {
            "current": round(curr, 4),
            "previous": round(prev, 4),
            "delta": round(d, 4),
            "delta_pct": round(pct, 2) if pct is not None else None,
        }

    return {
        "baseline_timestamp": baseline.get("timestamp", "unknown"),
        "fitness": _delta(current_summary.get("mean_fitness"), prev_summary.get("mean_fitness")),
        "wall_time": _delta(current_summary.get("mean_wall_time"), prev_summary.get("mean_wall_time")),
        "convergence_gen": _delta(
            current_summary.get("mean_convergence_gen"),
            prev_summary.get("mean_convergence_gen"),
        ),
        "config_changes": {
            k: {"current": current_config.get(k), "previous": prev_config.get(k)}
            for k in set(current_config) | set(prev_config)
            if current_config.get(k) != prev_config.get(k)
        },
    }


# ── Output formatting ───────────────────────────────────────────────────


def _print_summary_table(
    summary: dict[str, Any],
    runs: list[dict[str, Any]],
    config: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> None:
    """Print a formatted summary table to stdout.

    Args:
        summary: Aggregated run statistics.
        runs: Individual run results.
        config: Benchmark configuration used.
        comparison: Baseline comparison dict, or None.
    """
    print()
    print("=" * 70)
    print("  EVOLUTION ENGINE BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"  Population Size : {config.get('population_size', '?')}")
    print(f"  Generations     : {config.get('generations', '?')}")
    print(f"  Runs            : {summary['runs_completed']}/{summary['runs_total']} completed")
    if summary.get("runs_failed", 0) > 0:
        print(f"  Failed/Timeout  : {summary['runs_failed']}")
    print()
    print("  Fitness Statistics:")
    print(f"    Mean  : {summary['mean_fitness']:.4f}")
    print(f"    Std   : {summary['std_fitness']:.4f}")
    print(f"    Min   : {summary['min_fitness']:.4f}")
    print(f"    Max   : {summary['max_fitness']:.4f}")
    print()
    print("  Wall Time (seconds):")
    print(f"    Mean  : {summary['mean_wall_time']:.3f}s")
    print(f"    Std   : {summary['std_wall_time']:.3f}s")
    print(f"    Min   : {summary.get('min_wall_time', 0):.3f}s")
    print(f"    Max   : {summary.get('max_wall_time', 0):.3f}s")
    print()
    if summary.get("mean_convergence_gen") is not None:
        conv_std = summary.get("convergence_std")
        conv_std_str = f"{conv_std:.1f}" if conv_std is not None else "N/A"
        print(f"  Convergence Generation: {summary['mean_convergence_gen']:.1f} " f"(+/- {conv_std_str})")
    else:
        print("  Convergence Generation: N/A (fitness never reached 90% threshold)")
    print()

    # Per-run details
    print("  Per-Run Details:")
    print(f"  {'Run':>4} {'Status':>8} {'Fitness':>10} {'Time':>10} {'Conv Gen':>10}")
    print("  " + "-" * 46)
    for i, run in enumerate(runs):
        conv = str(run["convergence_generation"]) if run["convergence_generation"] is not None else "N/A"
        print(f"  {i:>4} {run['status']:>8} {run['best_fitness']:>10.4f} " f"{run['wall_time']:>9.3f}s {conv:>10}")
    print()

    # Baseline comparison
    if comparison is not None:
        print("  Baseline Comparison:")
        print(f"    Baseline from: {comparison.get('baseline_timestamp', 'unknown')}")
        for metric in ("fitness", "wall_time", "convergence_gen"):
            info = comparison.get(metric, {})
            prev = info.get("previous")
            curr = info.get("current")
            delta = info.get("delta")
            if delta is not None and prev is not None and curr is not None:
                direction = "+" if delta > 0 else ""
                pct_str = f" ({direction}{info['delta_pct']:.1f}%)" if info.get("delta_pct") is not None else ""
                print(f"    {metric:>16}: {prev:.4f} -> {curr:.4f} " f"(delta: {direction}{delta:.4f}{pct_str})")
            elif curr is not None:
                print(f"    {metric:>16}: {curr:.4f} (no previous baseline)")
            else:
                print(f"    {metric:>16}: N/A")

        # Config changes
        cfg_changes = comparison.get("config_changes", {})
        if cfg_changes:
            print("    Config Changes:")
            for key, vals in cfg_changes.items():
                print(f"      {key}: {vals.get('previous')} -> {vals.get('current')}")
        print()

    print("=" * 70)


# ── Main entry point ────────────────────────────────────────────────────


def main() -> int:
    """Main entry point for the benchmark suite.

    Parses CLI arguments, runs N independent evolution benchmarks,
    aggregates statistics, compares against baseline, and saves results.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Evolution Engine Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 benchmarks/fitness_benchmark.py\n"
            "  python3 benchmarks/fitness_benchmark.py --runs 5 --generations 100\n"
            "  python3 benchmarks/fitness_benchmark.py --population-size 25 --runs 1 --seed 99\n"
        ),
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Population size per generation (default: 50)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations per run (default: 50)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of independent benchmark runs (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for all runs (overrides default seeds 42, 123, 456)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_PER_RUN_TIMEOUT_SECONDS,
        help=f"Per-run timeout in seconds (default: {_PER_RUN_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.population_size < 2:
        print("Error: --population-size must be >= 2", file=sys.stderr)
        return 1
    if args.generations < 1:
        print("Error: --generations must be >= 1", file=sys.stderr)
        return 1
    if args.runs < 1:
        print("Error: --runs must be >= 1", file=sys.stderr)
        return 1

    # Deterministic seeds for reproducibility
    if args.seed is not None:
        seeds = [args.seed + i for i in range(args.runs)]
    else:
        default_seeds = [42, 123, 456]
        seeds = [default_seeds[i % len(default_seeds)] for i in range(args.runs)]

    config = {
        "population_size": args.population_size,
        "generations": args.generations,
        "runs": args.runs,
        "seed_base": seeds[0],
        "seeds": seeds,
        "timeout_seconds": args.timeout,
    }

    print("Evolution Engine Benchmark")
    print(f"  Pop: {args.population_size}, Generations: {args.generations}, Runs: {args.runs}")
    print(f"  Seeds: {seeds}")
    print(f"  Timeout per run: {args.timeout}s")
    print()

    # Load baseline for comparison
    baseline = _load_baseline()

    # Execute runs
    runs: list[dict[str, Any]] = []
    overall_start = time.time()

    for i in range(args.runs):
        run_result = _run_single_benchmark(
            run_index=i,
            population_size=args.population_size,
            num_generations=args.generations,
            seed=seeds[i],
            timeout_seconds=args.timeout,
        )
        runs.append(run_result)

    overall_time = time.time() - overall_start
    print(f"\nAll runs completed in {overall_time:.1f}s total")

    # Aggregate statistics
    summary = _aggregate_runs(runs)

    # Compute baseline comparison
    comparison = None
    if baseline is not None:
        comparison = _compute_baseline_comparison(summary, config, baseline)

    # Print summary table
    _print_summary_table(summary, runs, config, comparison)

    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "runs": runs,
        "summary": summary,
        "baseline_comparison": comparison,
    }

    try:
        with open(_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {_RESULTS_FILE}")
    except OSError as exc:
        print(f"Warning: Could not save results to {_RESULTS_FILE}: {exc}", file=sys.stderr)

    # Exit code: 0 if any run succeeded, 1 if all failed
    return 0 if summary["runs_completed"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
