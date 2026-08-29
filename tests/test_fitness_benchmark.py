#!/usr/bin/env python3
"""Regression tests for benchmarks/fitness_benchmark.py.

Guards against the "wrong constructor call" failure noted in
ADVERSARIAL_AUDIT_2026_07_17.md (HIGH-5): the benchmark builds an
EvolutionConfig from many kwargs, and a drift between those kwargs and the
EvolutionConfig dataclass silently breaks the benchmark (which has no other
test coverage). These tests exercise the config builder and a full tiny run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.fitness_benchmark import _build_evolution_config, _run_single_benchmark


class TestBuildEvolutionConfig:
    """The benchmark's EvolutionConfig builder must stay in sync with the dataclass."""

    def test_builds_without_error(self):
        # A TypeError here is exactly the HIGH-5 regression (kwarg drift).
        cfg = _build_evolution_config(population_size=8, num_generations=4, seed=42)
        assert cfg is not None

    def test_benchmark_overrides_applied(self):
        cfg = _build_evolution_config(population_size=8, num_generations=4, seed=7)
        # Core params propagated from args.
        assert cfg.population_size == 8
        assert cfg.max_generations == 4
        assert cfg.seed == 7
        # Benchmark-specific overrides: CPU-only, no I/O, no archives.
        assert cfg.use_gpu is False
        assert cfg.auto_curriculum is False
        assert cfg.use_novelty_archive is False
        assert cfg.use_qd_archive is False
        assert cfg.use_strategy_archive is False
        assert cfg.use_secondary_tasks is False


class TestRunSingleBenchmark:
    """A single benchmark run should complete end-to-end on CPU."""

    def test_run_completes_ok(self):
        result = _run_single_benchmark(
            run_index=0,
            population_size=6,
            num_generations=3,
            seed=42,
            timeout_seconds=120,
        )
        assert result["status"] == "ok", f"benchmark run failed: {result.get('error')}"
        assert result["best_fitness"] > 0
        assert len(result["fitness_curve"]) == 3
        assert result["error"] is None
