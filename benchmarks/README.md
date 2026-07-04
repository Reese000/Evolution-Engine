# Evolution Engine Performance Benchmarks

Performance benchmark suite for the Evolution Engine's core evolution loop. Measures fitness convergence, wall-clock time, and program quality across multiple independent runs with deterministic seeds.

## Purpose

- Track evolution performance across code changes and refactors
- Detect performance regressions in the core fitness evaluation pipeline
- Provide reproducible benchmarks with fixed seeds
- Compare results across versions via baseline comparison

## Quick Start

From the project root (`Evolution Engine/`):

```bash
# Default: 3 runs of 50 generations with pop_size=50
python3 benchmarks/fitness_benchmark.py

# Quick smoke test: 1 run, 5 generations
python3 benchmarks/fitness_benchmark.py --runs 1 --generations 5

# Thorough benchmark: 5 runs of 100 generations
python3 benchmarks/fitness_benchmark.py --runs 5 --generations 100 --population-size 75
```

Or via the installed entry point (after `pip install -e .`):

```bash
benchmark --runs 3 --generations 50
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--population-size` | 50 | Number of individuals per generation (must be >= 2) |
| `--generations` | 50 | Number of generations per run (must be >= 1) |
| `--runs` | 3 | Number of independent benchmark runs |
| `--seed` | None | Base seed (overrides default seeds 42, 123, 456) |
| `--timeout` | 1800 | Per-run timeout in seconds (default: 30 minutes) |

## How It Works

1. **Configuration**: Loads defaults from `config/engine_config.json` via the `Config` class, then applies benchmark-specific overrides (CPU-only, no GPU, no file I/O, no optional archives).

2. **Evolution Loop**: Each run creates a fresh `EvolutionEngine` with the specified seed and runs for the configured number of generations.

3. **Metrics Collected Per Run**:
   - `best_fitness`: Final champion fitness
   - `wall_time`: Total execution time in seconds
   - `convergence_generation`: First generation where fitness reached >90% of final value
   - `fitness_curve`: List of best fitness per generation (for plotting)
   - `secondary_task_scores`: instruction_count, code_diversity, mutation_rate, generations_stalled, total_improvements

4. **Aggregation**: Mean, std, min, max computed across runs for fitness and wall time.

5. **Baseline Comparison**: If `benchmarks/results.json` exists from a previous run, the new results are compared against it with absolute and percentage deltas.

6. **Results Saved**: Output written to `benchmarks/results.json`.

## Interpreting Results

- **Mean Fitness**: Higher is better. Track this across versions to detect regressions.
- **Std Fitness**: Lower is better. High std means results vary significantly between seeds.
- **Mean Wall Time**: Execution time per run. Increases may indicate performance regressions.
- **Convergence Generation**: Lower means faster convergence. A value of None means fitness never reached 90% of final (possible stall).
- **Baseline Comparison Deltas**: Negative fitness delta = regression. Positive wall_time delta = slowdown.

## Comparing Across Versions

The benchmark automatically compares against the previous `results.json`:

1. Run the benchmark on the current code
2. Save or commit `benchmarks/results.json`
3. Make code changes
4. Run the benchmark again -- it will show deltas vs. the saved baseline
5. Commit the new `results.json` to track the evolution of performance

## Output Structure

The `results.json` file has this structure:

```json
{
  "timestamp": "2026-07-01T22:00:00+00:00",
  "config": {
    "population_size": 50,
    "generations": 50,
    "runs": 3,
    "seed_base": 42,
    "seeds": [42, 123, 456],
    "timeout_seconds": 1800
  },
  "runs": [
    {
      "best_fitness": 45.2,
      "wall_time": 12.3,
      "convergence_generation": 23,
      "fitness_curve": [0.0, 1.5, ...],
      "secondary_task_scores": {...},
      "status": "ok",
      "error": null
    }
  ],
  "summary": {
    "mean_fitness": 44.8,
    "std_fitness": 1.2,
    "min_fitness": 43.5,
    "max_fitness": 45.2,
    "mean_wall_time": 12.1,
    "std_wall_time": 0.3,
    "mean_convergence_gen": 25.0,
    "convergence_std": 2.0,
    "runs_completed": 3,
    "runs_total": 3,
    "runs_failed": 0
  },
  "baseline_comparison": null
}
```

## Design Decisions

- **CPU-only**: GPU acceleration is excluded from benchmarks since it depends on hardware and would not produce comparable results across machines.
- **No optional archives**: Novelty archive, QD archive, strategy archive, and secondary tasks are disabled to isolate the core evolution loop performance.
- **No checkpointing/checkpoint I/O**: File I/O overhead is excluded to focus on pure computation performance.
- **Deterministic seeds**: Fixed seeds (42, 123, 456) ensure reproducibility across runs and machines.
- **Per-run timeout**: 30-minute timeout prevents runaway benchmarks from hanging indefinitely.
