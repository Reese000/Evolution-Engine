#!/usr/bin/env python3
"""Final performance benchmark tests for wave-20 new methods.

Three test classes:
1. TestMethodCallOverhead — time each of the 26 new methods once with
   minimal inputs; verify total overhead < 5 s.
2. TestArchiveScalability — 5000 StrategyArchive records + 200 ChampionArchive
   entries; time heavy query methods; verify total < 2 s.
3. TestMetricsScalability — 1000 recorded cycles; time detect_anomalies,
   correlate_metrics, get_trend; verify total < 3 s.
"""
from __future__ import annotations

import math
import os
import random
import sys
import tempfile
import time

import pytest

# ── Path setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champion_archive import ChampionArchive, ChampionEntry, _hash_program
from metrics import MetricsCollector
from strategy_archive import StrategyArchive

# ── Tiny helpers ────────────────────────────────────────────────────

STRATEGIES = ["RANDOM", "TASK_AWARE", "DIRECTED", "PERTURBATION_CAPPED"]


def _tiny_program(seed: int = 0) -> list[int]:
    """Return a 2-instruction RV32I program (minimal)."""
    rng = random.Random(seed)
    return [rng.getrandbits(32) & 0xFFFFFFFC for _ in range(2)]


def _populate_strategy_archive(sa: StrategyArchive, n: int = 5000, seed: int = 42) -> None:
    """Fill *sa* with *n* realistic records across multiple levels."""
    rng = random.Random(seed)
    for i in range(n):
        sa.record(
            strategy=rng.choice(STRATEGIES),
            level=rng.randint(0, 12),
            success=rng.random() > 0.4,
            generation=i,
        )
        sa.record_delta(rng.choice(STRATEGIES), rng.uniform(-10, 30))


def _populate_champion_archive(path: str, n: int = 200, seed: int = 99) -> ChampionArchive:
    """Return a ChampionArchive with *n* unique entries."""
    rng = random.Random(seed)
    ca = ChampionArchive(path=path, max_size=n + 50)
    for i in range(n):
        prog = [rng.getrandbits(32) & 0xFFFFFFFC for _ in range(rng.randint(2, 6))]
        ca.add(
            program=prog,
            level=rng.randint(0, 13),
            fitness=rng.uniform(50, 300),
            cycle=rng.randint(1, 20),
            generation=rng.randint(0, 50),
        )
    return ca


# ═══════════════════════════════════════════════════════════════════════
# 1. TestMethodCallOverhead
# ═══════════════════════════════════════════════════════════════════════


class TestMethodCallOverhead:
    """Time each new method once with minimal inputs; total must be < 5 s."""

    def test_total_overhead_under_5s(self) -> None:
        """Call every new method once and verify aggregate time."""
        total_time = 0.0
        tmpdir = tempfile.mkdtemp(prefix="bench_oh_")

        # --- StrategyArchive (15 methods) ---
        sa = StrategyArchive()

        def _timed(label: str, fn) -> None:
            nonlocal total_time
            t0 = time.perf_counter()
            result = fn()
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            return result

        _timed("SA.record", lambda: sa.record("RANDOM", 0, True, 0))
        _timed("SA.record_batch", lambda: sa.record_batch([
            {"strategy": "TASK_AWARE", "level": 1, "success": True, "gen": 0}
        ]))
        _timed("SA.success_rate", lambda: sa.success_rate("RANDOM"))
        _timed("SA.time_weighted_success_rate", lambda: sa.time_weighted_success_rate("RANDOM"))
        _timed("SA.exploration_bonus", lambda: sa.exploration_bonus("RANDOM", 0))
        _timed("SA.best_for_level", lambda: sa.best_for_level(0))
        _timed("SA.recommend_strategy", lambda: sa.recommend_strategy(0))
        _timed("SA.total_records", lambda: sa.total_records())
        _timed("SA.records_by_level", lambda: sa.records_by_level(0))
        _timed("SA.records_by_strategy", lambda: sa.records_by_strategy("RANDOM"))
        _timed("SA.get_strategy_history", lambda: sa.get_strategy_history("RANDOM"))
        _timed("SA.summary", lambda: sa.summary())
        _timed("SA.per_level_effectiveness", lambda: sa.per_level_effectiveness("RANDOM"))
        _timed("SA.get_strategy_diversity", lambda: sa.get_strategy_diversity())
        _timed("SA.get_dominance_report", lambda: sa.get_dominance_report())
        _timed("SA.get_archive_summary", lambda: sa.get_archive_summary())
        _timed("SA.record_delta", lambda: sa.record_delta("RANDOM", 5.0))
        _timed("SA.avg_delta", lambda: sa.avg_delta("RANDOM"))
        _timed("SA.decay_level_records", lambda: sa.decay_level_records(0, factor=0.5, rng=random.Random(1)))
        _timed("SA.clear", lambda: sa.clear())

        # --- ChampionArchive (5 methods) ---
        ca_path = os.path.join(tmpdir, "champ.jsonl")
        ca = ChampionArchive(path=ca_path, max_size=100)
        _prog1 = _tiny_program(1)
        _prog2 = _tiny_program(2)

        _timed("CA.add", lambda: ca.add(_prog1, level=1, fitness=100.0, cycle=1))
        _timed("CA.sample_training_batch", lambda: ca.sample_training_batch(batch_size=10))
        _timed("CA.sample_tiered_batch", lambda: ca.sample_tiered_batch(batch_size=10))
        _timed("CA.diversity_aware_select", lambda: ca.diversity_aware_select(level=1, n=1))
        _timed("CA.quality_weighted_prune", lambda: ca.quality_weighted_prune())
        _timed("CA.get_quality_distribution", lambda: ca.get_quality_distribution())
        _timed("CA.get_recency_distribution", lambda: ca.get_recency_distribution())
        _timed("CA.get_coverage_stats", lambda: ca.get_coverage_stats())
        _timed("CA.compute_sample_weights", lambda: ca.compute_sample_weights())
        _timed("CA.get_sample_weights_summary", lambda: ca.get_sample_weights_summary())
        _timed("CA.get_diversity_report", lambda: ca.get_diversity_report())

        # --- MetricsCollector (4 methods) ---
        mc_path = os.path.join(tmpdir, "metrics.jsonl")
        mc = MetricsCollector(path=mc_path)
        _timed("MC.record_cycle", lambda: mc.record_cycle({"cycle": 1, "fitness": 0.5}))
        _timed("MC.get_history", lambda: mc.get_history())
        _timed("MC.get_trend", lambda: mc.get_trend("fitness"))
        _timed("MC.count_records", lambda: mc.count_records())
        _timed("MC.detect_anomalies", lambda: mc.detect_anomalies("fitness"))
        _timed("MC.correlate_metrics", lambda: mc.correlate_metrics("fitness", "cycle"))
        _timed("MC.clear", lambda: mc.clear())

        print(f"\n  Total method-call overhead: {total_time:.4f}s (limit 5s)")
        assert total_time < 5.0, (
            f"Total overhead {total_time:.2f}s exceeds 5s budget"
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. TestArchiveScalability
# ═══════════════════════════════════════════════════════════════════════


class TestArchiveScalability:
    """5000 StrategyArchive records + 200 ChampionArchive entries;
    time heavy queries; total must be < 2 s."""

    def test_total_under_2s(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="bench_sc_")
        total_time = 0.0

        # --- Build StrategyArchive with 5000 records ---
        sa = StrategyArchive()
        t0 = time.perf_counter()
        _populate_strategy_archive(sa, n=5000)
        build_sa = time.perf_counter() - t0
        total_time += build_sa
        assert sa.total_records() == 5000

        def _timed(label: str, fn):
            nonlocal total_time
            t0 = time.perf_counter()
            result = fn()
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            return result

        _timed("SA.get_dominance_report", lambda: sa.get_dominance_report())
        _timed("SA.per_level_effectiveness", lambda: sa.per_level_effectiveness("RANDOM"))
        _timed("SA.time_weighted_success_rate(RANDOM)", lambda: sa.time_weighted_success_rate("RANDOM"))
        _timed("SA.time_weighted_success_rate(TASK_AWARE)", lambda: sa.time_weighted_success_rate("TASK_AWARE"))
        _timed("SA.summary", lambda: sa.summary())
        _timed("SA.get_archive_summary", lambda: sa.get_archive_summary())

        # --- Build ChampionArchive with 200 champions ---
        ca_path = os.path.join(tmpdir, "champ.jsonl")
        t0 = time.perf_counter()
        ca = _populate_champion_archive(ca_path, n=200)
        build_ca = time.perf_counter() - t0
        total_time += build_ca
        assert ca.size == 200

        _timed("CA.diversity_aware_select(level=5)", lambda: ca.diversity_aware_select(level=5, n=5))
        _timed("CA.diversity_aware_select(level=10)", lambda: ca.diversity_aware_select(level=10, n=5))
        _timed("CA.quality_weighted_prune", lambda: ca.quality_weighted_prune())
        _timed("CA.get_diversity_report", lambda: ca.get_diversity_report())
        _timed("CA.get_quality_distribution", lambda: ca.get_quality_distribution())

        print(f"\n  Total archive scalability time: {total_time:.4f}s (limit 2s)")
        assert total_time < 2.0, (
            f"Total archive time {total_time:.2f}s exceeds 2s budget"
        )


# ═══════════════════════════════════════════════════════════════════════
# 3. TestMetricsScalability
# ═══════════════════════════════════════════════════════════════════════


class TestMetricsScalability:
    """1000 recorded cycles; time detect_anomalies, correlate_metrics,
    get_trend; total must be < 3 s."""

    def test_total_under_3s(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="bench_mc_")
        mc_path = os.path.join(tmpdir, "metrics.jsonl")
        mc = MetricsCollector(path=mc_path)
        total_time = 0.0
        rng = random.Random(55)

        # Record 1000 cycles
        t0 = time.perf_counter()
        for i in range(1000):
            mc.record_cycle({
                "cycle": i,
                "fitness": 100.0 + rng.gauss(0, 10) + i * 0.05,
                "level": min(i // 80, 13),
                "diversity": rng.random(),
                "val_loss": max(0.01, 1.0 - i * 0.001 + rng.gauss(0, 0.05)),
            })
        build_mc = time.perf_counter() - t0
        total_time += build_mc
        assert mc.count_records() == 1000

        def _timed(label: str, fn):
            nonlocal total_time
            t0 = time.perf_counter()
            result = fn()
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            return result

        _timed("MC.detect_anomalies(fitness)", lambda: mc.detect_anomalies("fitness"))
        _timed("MC.detect_anomalies(level)", lambda: mc.detect_anomalies("level"))
        _timed("MC.correlate_metrics(fitness,level)", lambda: mc.correlate_metrics("fitness", "level"))
        _timed("MC.correlate_metrics(fitness,val_loss)", lambda: mc.correlate_metrics("fitness", "val_loss"))
        _timed("MC.get_trend(fitness)", lambda: mc.get_trend("fitness"))
        _timed("MC.get_trend(level)", lambda: mc.get_trend("level"))
        _timed("MC.get_trend(diversity)", lambda: mc.get_trend("diversity"))
        _timed("MC.get_history(last_n=100)", lambda: mc.get_history(last_n=100))

        print(f"\n  Total metrics scalability time: {total_time:.4f}s (limit 3s)")
        assert total_time < 3.0, (
            f"Total metrics time {total_time:.2f}s exceeds 3s budget"
        )
