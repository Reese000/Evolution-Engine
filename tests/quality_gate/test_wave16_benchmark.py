#!/usr/bin/env python3
"""Performance benchmark tests for the Evolution Engine.

Tests 4 subsystems for reasonable performance under moderate load.
Each test verifies that key operations complete within a time budget.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import Dict, List

import pytest

# Path setup — add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add tests/ directory so conftest fixtures are importable from subdirectories
_tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from meta_controller import MetaController, TelemetrySnapshot, StagnationResult
from strategy_archive import StrategyArchive
from metrics import MetricsCollector

# We'll import the engine factory from conftest
from conftest import make_engine_with_task, make_random_programs


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _snap(
    cycle: int,
    fitness: float,
    level: int = 0,
    val_loss=None,
    diversity: float = 0.5,
    champion_words: int = 10,
    retrained: bool = False,
) -> TelemetrySnapshot:
    """Create a TelemetrySnapshot with sensible defaults."""
    return TelemetrySnapshot(
        cycle=cycle,
        fitness=fitness,
        level=level,
        champion_words=champion_words,
        val_loss=val_loss,
        diversity=diversity,
        novelty_size=0,
        qd_occupied=0,
        qd_total=100,
        traces_count=0,
        retrained=retrained,
        generations_completed=50,
    )


STRATEGIES = ["RANDOM", "TASK_AWARE", "DIRECTED", "PERTURBATION_CAPPED"]


# ═══════════════════════════════════════════════════════════════════════
# 1. TestMetaControllerPerformance
# ═══════════════════════════════════════════════════════════════════════

class TestMetaControllerPerformance:
    """Benchmark MetaController stagnation detection and trend analysis."""

    def test_detect_stagnation_performance(self):
        """detect_stagnation should complete quickly with 100 telemetry entries."""
        with tempfile.TemporaryDirectory() as td:
            mc = MetaController(checkpoint_dir=td)

            # Add 100 telemetry entries
            for i in range(100):
                mc.update_cycle(_snap(
                    cycle=i,
                    fitness=1.0 + (i % 20) * 0.1,
                    level=i // 10,
                    diversity=0.5 + (i % 5) * 0.05,
                ))

            # Time detect_stagnation
            t0 = time.perf_counter()
            result = mc.detect_stagnation()
            elapsed = time.perf_counter() - t0

            assert isinstance(result, StagnationResult)
            assert elapsed < 1.0, f"detect_stagnation took {elapsed:.4f}s, expected < 1s"

    def test_get_cross_cycle_trends_performance(self):
        """get_cross_cycle_trends should complete quickly with 100 telemetry entries."""
        with tempfile.TemporaryDirectory() as td:
            mc = MetaController(checkpoint_dir=td)

            # Add 100 telemetry entries
            for i in range(100):
                mc.update_cycle(_snap(
                    cycle=i,
                    fitness=1.0 + i * 0.02,
                    level=i // 10,
                    diversity=0.5 + (i % 5) * 0.05,
                ))

            # Time get_cross_cycle_trends
            t0 = time.perf_counter()
            trends = mc.get_cross_cycle_trends()
            elapsed = time.perf_counter() - t0

            assert "fitness" in trends
            assert "val_loss" in trends
            assert "diversity" in trends
            assert elapsed < 1.0, f"get_cross_cycle_trends took {elapsed:.4f}s, expected < 1s"


# ═══════════════════════════════════════════════════════════════════════
# 2. TestStrategyArchivePerformance
# ═══════════════════════════════════════════════════════════════════════

class TestStrategyArchivePerformance:
    """Benchmark StrategyArchive recommendation and reporting."""

    def test_recommend_strategy_performance(self):
        """recommend_strategy should complete quickly with 500 recorded outcomes."""
        sa = StrategyArchive()

        # Record 500 outcomes across multiple strategies and levels
        import random
        rng = random.Random(42)
        for i in range(500):
            sa.record(
                strategy=rng.choice(STRATEGIES),
                level=rng.randint(0, 9),
                success=rng.random() < 0.4,
                generation=i // 50,
            )

        # Time recommend_strategy
        t0 = time.perf_counter()
        result = sa.recommend_strategy(level=3)
        elapsed = time.perf_counter() - t0

        assert isinstance(result, str)
        assert elapsed < 1.0, f"recommend_strategy took {elapsed:.4f}s, expected < 1s"

    def test_get_dominance_report_performance(self):
        """get_dominance_report should complete quickly with 500 recorded outcomes."""
        sa = StrategyArchive()

        import random
        rng = random.Random(42)
        for i in range(500):
            sa.record(
                strategy=rng.choice(STRATEGIES),
                level=rng.randint(0, 9),
                success=rng.random() < 0.4,
                generation=i // 50,
            )

        # Time get_dominance_report
        t0 = time.perf_counter()
        report = sa.get_dominance_report()
        elapsed = time.perf_counter() - t0

        assert "dominant_strategy" in report
        assert "dominance_ratio" in report
        assert "is_monoculture" in report
        assert elapsed < 1.0, f"get_dominance_report took {elapsed:.4f}s, expected < 1s"

    def test_per_level_effectiveness_performance(self):
        """per_level_effectiveness should complete quickly with 500 recorded outcomes."""
        sa = StrategyArchive()

        import random
        rng = random.Random(42)
        for i in range(500):
            sa.record(
                strategy=rng.choice(STRATEGIES),
                level=rng.randint(0, 9),
                success=rng.random() < 0.4,
                generation=i // 50,
            )

        # Time per_level_effectiveness for each strategy
        t0 = time.perf_counter()
        for strat in STRATEGIES:
            result = sa.per_level_effectiveness(strat)
            assert isinstance(result, dict)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"per_level_effectiveness took {elapsed:.4f}s, expected < 1s"


# ═══════════════════════════════════════════════════════════════════════
# 3. TestFitnessComputationPerformance
# ═══════════════════════════════════════════════════════════════════════

class TestFitnessComputationPerformance:
    """Benchmark EvolutionEngine._compute_fitness on 100 programs."""

    def test_compute_fitness_performance(self):
        """Average _compute_fitness should complete under 1ms per call."""
        engine = make_engine_with_task()

        # Generate 100 random programs
        programs = make_random_programs(100, length=5, seed=42)

        # Create matching traces (successful execution)
        traces = [
            {
                "status": "ok",
                "final_regs": {10: 42, 11: 7},
                "final_memory": {},
                "instructions_count": 5,
                "trap_cause": "",
            }
            for _ in range(100)
        ]

        # Time 100 _compute_fitness calls
        t0 = time.perf_counter()
        for prog, trace in zip(programs, traces):
            result = engine._compute_fitness(prog, trace)
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / 100) * 1000  # Convert to ms

        assert len(programs) == 100
        assert elapsed > 0
        assert avg_ms < 1.0, (
            f"Average _compute_fitness took {avg_ms:.4f}ms per call, expected < 1ms"
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. TestMetricsCollectorPerformance
# ═══════════════════════════════════════════════════════════════════════

class TestMetricsCollectorPerformance:
    """Benchmark MetricsCollector anomaly detection and correlation."""

    def test_detect_anomalies_performance(self):
        """detect_anomalies should complete quickly with 200 recorded cycles."""
        with tempfile.TemporaryDirectory() as td:
            metrics_path = os.path.join(td, "cycles.jsonl")
            mc = MetricsCollector(path=metrics_path)

            # Record 200 cycles
            import random
            rng = random.Random(42)
            for i in range(200):
                mc.record_cycle({
                    "cycle": i,
                    "fitness": 1.0 + rng.gauss(0, 0.1),
                    "level": rng.randint(0, 5),
                })

            # Time detect_anomalies
            t0 = time.perf_counter()
            anomalies = mc.detect_anomalies("fitness")
            elapsed = time.perf_counter() - t0

            assert isinstance(anomalies, list)
            assert elapsed < 1.0, f"detect_anomalies took {elapsed:.4f}s, expected < 1s"

    def test_correlate_metrics_performance(self):
        """correlate_metrics should complete quickly with 200 recorded cycles."""
        with tempfile.TemporaryDirectory() as td:
            metrics_path = os.path.join(td, "cycles.jsonl")
            mc = MetricsCollector(path=metrics_path)

            # Record 200 cycles
            import random
            rng = random.Random(42)
            for i in range(200):
                mc.record_cycle({
                    "cycle": i,
                    "fitness": 1.0 + rng.gauss(0, 0.1),
                    "level": rng.randint(0, 5),
                })

            # Time correlate_metrics
            t0 = time.perf_counter()
            result = mc.correlate_metrics("fitness", "level")
            elapsed = time.perf_counter() - t0

            assert "correlation" in result
            assert "strength" in result
            assert elapsed < 1.0, f"correlate_metrics took {elapsed:.4f}s, expected < 1s"
