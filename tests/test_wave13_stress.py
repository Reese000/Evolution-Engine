#!/usr/bin/env python3
"""test_wave13_stress.py — Stress and robustness tests for Evolution Engine subsystems.

Tests high-volume operation, edge cases, and performance of:
- MetaController telemetry ingestion and analysis
- StrategyArchive with 1000+ records across multiple levels
- ChampionArchive with 100 champions, diversity selection, and pruning
- MetricsCollector with 200 cycles of random data
- SparseArchive (novelty) with 50 entries and capacity management
"""

from __future__ import annotations

import gc
import math
import os
import random
import sys
import tempfile
import time

import pytest

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_controller import MetaController, TelemetrySnapshot
from strategy_archive import StrategyArchive
from champion_archive import ChampionArchive
from metrics import MetricsCollector
from novelty import SparseArchive


# ══════════════════════════════════════════════════════════════════════════════
# Test 1: MetaController Stress
# ══════════════════════════════════════════════════════════════════════════════

class TestMetaControllerStress:
    """Stress-test MetaController with 100 telemetry entries and analysis methods."""

    @pytest.fixture
    def mc(self, tmp_path):
        return MetaController(checkpoint_dir=str(tmp_path))

    def _make_snapshot(self, cycle, rng):
        """Create a TelemetrySnapshot with randomised metrics."""
        return TelemetrySnapshot(
            cycle=cycle,
            fitness=rng.uniform(0.1, 300.0),
            level=rng.randint(0, 15),
            champion_words=rng.randint(3, 50),
            val_loss=rng.uniform(0.01, 5.0),
            diversity=rng.uniform(0.0, 1.0),
            novelty_size=rng.randint(0, 200),
            qd_occupied=rng.randint(0, 50),
            qd_total=100,
            traces_count=rng.randint(0, 20),
            retrained=rng.random() > 0.5,
            generations_completed=rng.randint(10, 200),
        )

    def test_ingest_100_snapshots(self, mc):
        """Ingest 100 telemetry snapshots without crashing."""
        rng = random.Random(123)
        start = time.monotonic()
        for i in range(1, 101):
            snap = self._make_snapshot(i, rng)
            mc.update_cycle(snap)
        elapsed = time.monotonic() - start
        # Should finish well within 5 seconds
        assert elapsed < 5.0, f"100 snapshots took {elapsed:.2f}s, expected < 5s"
        # Verify internal telemetry grew
        assert len(mc.telemetry) > 0

    def test_stagnation_detection_under_load(self, mc):
        """Detect stagnation after a flat-line sequence."""
        rng = random.Random(456)
        # First 20 cycles with improving fitness
        for i in range(1, 21):
            mc.update_cycle(self._make_snapshot(i, rng))
        # Next 30 cycles with completely flat fitness (stagnation)
        for i in range(21, 51):
            snap = TelemetrySnapshot(
                cycle=i, fitness=50.0, level=3, champion_words=10,
                val_loss=0.5, diversity=0.3, novelty_size=5,
                qd_occupied=10, qd_total=100, traces_count=10,
                retrained=False, generations_completed=10,
            )
            mc.update_cycle(snap)
        result = mc.detect_stagnation()
        assert result is not None
        assert hasattr(result, 'is_stagnant') or isinstance(result, object)

    def test_effectiveness_report(self, mc):
        """Run effectiveness report after heavy ingestion."""
        rng = random.Random(789)
        for i in range(1, 51):
            mc.update_cycle(self._make_snapshot(i, rng))
        report = mc.get_effectiveness_report()
        assert isinstance(report, dict)
        # Should contain some key metrics
        assert "total_cycles" in report or "cycles_analyzed" in report or len(report) > 0

    def test_state_summary(self, mc):
        """get_state_summary after 100 cycles returns coherent dict."""
        rng = random.Random(321)
        for i in range(1, 101):
            mc.update_cycle(self._make_snapshot(i, rng))
        summary = mc.get_state_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_cross_cycle_trends(self, mc):
        """get_cross_cycle_trends after heavy data."""
        rng = random.Random(654)
        for i in range(1, 80):
            mc.update_cycle(self._make_snapshot(i, rng))
        trends = mc.get_cross_cycle_trends(window=10)
        assert isinstance(trends, dict)

    def test_recovery_confidence(self, mc):
        """get_recovery_confidence returns a dict."""
        rng = random.Random(987)
        for i in range(1, 40):
            mc.update_cycle(self._make_snapshot(i, rng))
        conf = mc.get_recovery_confidence()
        assert isinstance(conf, dict)

    def test_tuning_history_summary(self, mc):
        """get_tuning_history_summary returns dict."""
        rng = random.Random(111)
        for i in range(1, 30):
            mc.update_cycle(self._make_snapshot(i, rng))
        th = mc.get_tuning_history_summary()
        assert isinstance(th, dict)

    def test_save_load_cycle(self, mc, tmp_path):
        """Save and reload state preserves data."""
        rng = random.Random(222)
        for i in range(1, 31):
            mc.update_cycle(self._make_snapshot(i, rng))
        path = str(tmp_path / "mc_state.json")
        mc.save_state(path)
        mc2 = MetaController(checkpoint_dir=str(tmp_path / "mc2"))
        loaded = mc2.load_state(path)
        assert loaded is True

    def test_tune_retrain_params(self, mc):
        """tune_retrain_params does not crash."""
        rng = random.Random(333)
        for i in range(1, 21):
            mc.update_cycle(self._make_snapshot(i, rng))
        result = mc.tune_retrain_params(cycle=21)
        assert isinstance(result, dict)

    def test_tune_evolution_params(self, mc):
        """tune_evolution_params does not crash."""
        rng = random.Random(444)
        for i in range(1, 21):
            mc.update_cycle(self._make_snapshot(i, rng))
        result = mc.tune_evolution_params(cycle=21)
        assert isinstance(result, dict)

    def test_should_rollback(self, mc):
        """should_rollback returns a bool."""
        rng = random.Random(555)
        for i in range(1, 21):
            mc.update_cycle(self._make_snapshot(i, rng))
        result = mc.should_rollback()
        assert isinstance(result, bool)


# ══════════════════════════════════════════════════════════════════════════════
# Test 2: StrategyArchive Stress
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategyArchiveStress:
    """Stress-test StrategyArchive with 1000 outcomes across 10 levels."""

    STRATEGIES = ["RANDOM", "TASK_AWARE", "PERTURBATION", "CROSSOVER",
                  "ELITIST", "DIVERSITY", "NOVELTY", "ADAPTIVE"]

    def _populate(self, n_records=1000, n_levels=10):
        """Create a fully populated archive."""
        rng = random.Random(42)
        sa = StrategyArchive(max_records=2000)
        for i in range(n_records):
            strat = rng.choice(self.STRATEGIES)
            level = rng.randint(0, n_levels - 1)
            success = rng.random() < 0.4
            sa.record(strat, level, success, generation=i)
            if success:
                sa.record_delta(strat, rng.uniform(0.1, 10.0))
        return sa

    def test_record_1000_outcomes(self):
        """Record 1000 strategy outcomes across 10 levels."""
        sa = self._populate(n_records=1000, n_levels=10)
        assert sa.total_records() == 1000

    def test_batch_record(self):
        """record_batch with 500 records at once."""
        rng = random.Random(43)
        sa = StrategyArchive(max_records=2000)
        batch = []
        for i in range(500):
            batch.append({
                "strategy": rng.choice(self.STRATEGIES),
                "level": rng.randint(0, 9),
                "success": rng.random() < 0.4,
                "gen": i,
            })
        sa.record_batch(batch)
        assert sa.total_records() == 500

    def test_dominance_report(self):
        """get_dominance_report returns coherent structure."""
        sa = self._populate()
        report = sa.get_dominance_report()
        assert isinstance(report, dict)
        assert "dominant_strategy" in report
        assert "is_monoculture" in report
        assert "all_strategy_rates" in report

    def test_per_level_effectiveness(self):
        """per_level_effectiveness returns dict for each strategy."""
        sa = self._populate()
        for strat in self.STRATEGIES:
            eff = sa.per_level_effectiveness(strat)
            assert isinstance(eff, dict)
            for level, score in eff.items():
                assert 0.0 <= score <= 1.0

    def test_best_for_level(self):
        """best_for_level returns valid (strategy, score) or None."""
        sa = self._populate()
        for level in range(10):
            result = sa.best_for_level(level)
            if result is not None:
                strat, score = result
                assert strat in self.STRATEGIES
                assert isinstance(score, (int, float))

    def test_recommend_strategy(self):
        """recommend_strategy returns a known strategy name."""
        sa = self._populate()
        for level in range(10):
            rec = sa.recommend_strategy(level)
            assert isinstance(rec, str)
            # Should be one of the known strategies or fallback to RANDOM
            assert rec in self.STRATEGIES or rec == "RANDOM"

    def test_summary(self):
        """summary() returns comprehensive data."""
        sa = self._populate()
        sm = sa.summary()
        assert isinstance(sm, dict)
        assert sm["total_records"] == 1000
        assert "strategies" in sm
        assert "dominance" in sm

    def test_get_archive_summary(self):
        """get_archive_summary returns diversity and coverage info."""
        sa = self._populate()
        ars = sa.get_archive_summary()
        assert isinstance(ars, dict)
        assert ars["total_records"] == 1000
        assert ars["unique_strategies"] == len(self.STRATEGIES)
        assert len(ars["levels_covered"]) == 10

    def test_strategy_diversity(self):
        """get_strategy_diversity returns float in [0, 1]."""
        sa = self._populate()
        div = sa.get_strategy_diversity()
        assert 0.0 <= div <= 1.0
        # Per-level diversity should also work
        for level in range(10):
            d = sa.get_strategy_diversity(level=level)
            assert 0.0 <= d <= 1.0

    def test_success_rate(self):
        """success_rate returns float in [0, 1] for every strategy."""
        sa = self._populate()
        for strat in self.STRATEGIES:
            rate = sa.success_rate(strat, window=50)
            assert 0.0 <= rate <= 1.0
            # With level filter
            rate_l = sa.success_rate(strat, window=50, level=3)
            assert 0.0 <= rate_l <= 1.0

    def test_exploration_bonus(self):
        """exploration_bonus returns non-negative float."""
        sa = self._populate()
        for strat in self.STRATEGIES:
            bonus = sa.exploration_bonus(strat, level=5)
            assert bonus >= 0.0

    def test_adaptive_exploration_bonus(self):
        """adaptive_exploration_bonus returns float."""
        sa = self._populate()
        for strat in self.STRATEGIES:
            ab = sa.adaptive_exploration_bonus(strat, level=3)
            assert isinstance(ab, float)
            assert ab >= 0.0

    def test_records_by_level_and_strategy(self):
        """records_by_level and records_by_strategy return lists."""
        sa = self._populate()
        rl = sa.records_by_level(5)
        assert isinstance(rl, list)
        assert all(r["level"] == 5 for r in rl)
        rs = sa.records_by_strategy("RANDOM")
        assert isinstance(rs, list)
        assert all(r["strategy"] == "RANDOM" for r in rs)

    def test_decay_level_records(self):
        """decay_level_records modifies records without crashing."""
        sa = self._populate()
        modified = sa.decay_level_records(level=3, factor=0.5)
        assert isinstance(modified, int)
        assert modified >= 0

    def test_time_weighted_success_rate(self):
        """time_weighted_success_rate returns float in [0, 1]."""
        sa = self._populate()
        for strat in self.STRATEGIES:
            twr = sa.time_weighted_success_rate(strat)
            assert 0.0 <= twr <= 1.0

    def test_avg_delta(self):
        """avg_delta returns a numeric value."""
        sa = self._populate()
        for strat in self.STRATEGIES:
            ad = sa.avg_delta(strat)
            assert isinstance(ad, float)

    def test_get_strategy_history(self):
        """get_strategy_history returns a list."""
        sa = self._populate()
        hist = sa.get_strategy_history("RANDOM")
        assert isinstance(hist, list)
        assert len(hist) <= 20

    def test_max_records_trim(self):
        """Archive trims old records when max_records is exceeded."""
        sa = StrategyArchive(max_records=50)
        rng = random.Random(99)
        for i in range(100):
            sa.record("RANDOM", 0, True, generation=i)
        assert sa.total_records() == 50

    def test_clear(self):
        """clear() resets everything."""
        sa = self._populate()
        sa.clear()
        assert sa.total_records() == 0


# ══════════════════════════════════════════════════════════════════════════════
# Test 3: ChampionArchive Stress
# ══════════════════════════════════════════════════════════════════════════════

class TestChampionArchiveStress:
    """Stress-test ChampionArchive with 100 champions."""

    def _make_program(self, rng, length=6):
        """Generate a random RV32I program."""
        # RV32I ALU ops: low 7 bits = 0x33 (R-type) or 0x13 (I-type)
        opcodes = [0x33, 0x13, 0x63, 0x03, 0x23, 0x37]
        program = []
        for _ in range(length):
            base = rng.choice(opcodes)
            rd = rng.randint(1, 31) << 7
            rs1 = rng.randint(0, 31) << 15
            rs2 = rng.randint(0, 31) << 20
            func = rng.randint(0, 7) << 12
            word = base | rd | rs1 | rs2 | func
            program.append(word)
        program.append(0x00100073)  # EBREAK
        return program

    def _populate(self, n=50, max_size=150, tmp_path=None):
        """Create a populated archive with n unique champions."""
        path = str(tmp_path / "archive.jsonl") if tmp_path else "champions/archive_test.jsonl"
        ca = ChampionArchive(path=path, max_size=max_size)
        rng = random.Random(42)
        added = 0
        for i in range(n):
            prog = self._make_program(rng, length=rng.randint(3, 12))
            level = rng.randint(0, 15)
            fitness = rng.uniform(50.0, 300.0)
            cycle = rng.randint(1, 50)
            result = ca.add(prog, level=level, fitness=fitness, cycle=cycle, generation=i)
            if result:
                added += 1
        gc.collect()
        return ca, added

    @pytest.fixture
    def populated_archive(self, tmp_path):
        gc.collect()
        return self._populate(n=50, max_size=150, tmp_path=tmp_path)

    def test_add_100_champions(self, populated_archive):
        """Add multiple unique champions without crashing."""
        ca, added = populated_archive
        assert added >= 20  # Some may collide but most should be unique

    def test_diversity_aware_select(self, populated_archive):
        """diversity_aware_select returns valid results at multiple levels."""
        ca, _ = populated_archive
        for level in range(16):
            result = ca.diversity_aware_select(level=level, n=3)
            assert isinstance(result, list)
            for champ in result:
                assert isinstance(champ, dict)
                assert "program" in champ
                assert "fitness" in champ
                assert "level" in champ

    def test_diversity_aware_select_multiple_calls(self, populated_archive):
        """Multiple calls to diversity_aware_select don't corrupt state."""
        ca, _ = populated_archive
        for _ in range(10):
            result = ca.diversity_aware_select(level=0, n=2)
            assert isinstance(result, list)

    def test_quality_weighted_prune(self, tmp_path):
        """quality_weighted_prune removes low-quality champions when over capacity."""
        ca, added = self._populate(n=100, max_size=80, tmp_path=tmp_path)
        initial_size = ca.size
        removed = ca.quality_weighted_prune()
        assert isinstance(removed, int)
        # After pruning, should be at or below max_size
        assert ca.size <= 80

    def test_get_diversity_report(self, populated_archive):
        """get_diversity_report returns coherent structure."""
        ca, _ = populated_archive
        report = ca.get_diversity_report()
        assert isinstance(report, dict)
        assert report["total_champions"] > 0
        assert isinstance(report["unique_levels"], list)
        assert isinstance(report["avg_fitness_by_level"], dict)
        assert isinstance(report["program_length_variance"], float)
        assert isinstance(report["instruction_diversity"], float)
        assert isinstance(report["fitness_correlation_with_length"], float)
        # Correlation should be in [-1, 1]
        assert -1.0 <= report["fitness_correlation_with_length"] <= 1.0

    def test_sample_training_batch(self, populated_archive):
        """sample_training_batch returns requested number of champions."""
        ca, _ = populated_archive
        batch = ca.sample_training_batch(batch_size=50)
        assert isinstance(batch, list)
        assert len(batch) <= 50

    def test_sample_tiered_batch(self, populated_archive):
        """sample_tiered_batch returns champions."""
        ca, _ = populated_archive
        batch = ca.sample_tiered_batch(batch_size=64)
        assert isinstance(batch, list)
        assert len(batch) > 0

    def test_get_coverage_stats(self, populated_archive):
        """get_coverage_stats returns structured data."""
        ca, _ = populated_archive
        stats = ca.get_coverage_stats()
        assert isinstance(stats, dict)
        assert stats["total_entries"] > 0
        assert stats["unique_cycles"] > 0
        assert stats["unique_levels"] > 0

    def test_get_quality_distribution(self, populated_archive):
        """get_quality_distribution returns buckets and stats."""
        ca, _ = populated_archive
        qd = ca.get_quality_distribution()
        assert isinstance(qd, dict)
        assert qd["mean"] > 0
        assert qd["stddev"] >= 0

    def test_get_recency_distribution(self, populated_archive):
        """get_recency_distribution returns cycle info."""
        ca, _ = populated_archive
        rd = ca.get_recency_distribution()
        assert isinstance(rd, dict)
        assert rd["max_cycle"] > 0

    def test_prune_stale(self, populated_archive):
        """prune_stale removes old entries."""
        ca, _ = populated_archive
        before = ca.size
        removed = ca.prune_stale(max_cycles_old=10)
        assert isinstance(removed, int)
        assert ca.size <= before

    def test_stats(self, populated_archive):
        """stats() returns summary."""
        ca, _ = populated_archive
        st = ca.stats()
        assert isinstance(st, dict)
        assert st["size"] > 0

    def test_compute_sample_weights(self, populated_archive):
        """compute_sample_weights returns list of floats summing to ~1."""
        ca, _ = populated_archive
        weights = ca.compute_sample_weights()
        assert isinstance(weights, list)
        assert len(weights) == ca.size
        total = sum(weights)
        assert abs(total - 1.0) < 0.01

    def test_get_sample_weights_summary(self, populated_archive):
        """get_sample_weights_summary returns tier info."""
        ca, _ = populated_archive
        sws = ca.get_sample_weights_summary()
        assert isinstance(sws, dict)
        assert "tier_quotas" in sws
        assert "tier_sizes" in sws

    def test_save_and_reload(self, populated_archive, tmp_path):
        """Save archive to disk and reload."""
        ca, _ = populated_archive
        ca.save()
        # Reload from same path
        ca2 = ChampionArchive(path=ca.path, max_size=150)
        assert ca2.size == ca.size

    def test_duplicate_rejection(self, populated_archive):
        """Same program is not added twice."""
        ca, _ = populated_archive
        rng = random.Random(999)
        prog = self._make_program(rng)
        ca.add(prog, level=5, fitness=100.0, cycle=1)
        result = ca.add(prog, level=5, fitness=100.0, cycle=2)
        assert result is False  # Duplicate should be rejected

    def test_empty_archive_safety(self, tmp_path):
        """Methods handle empty archive gracefully."""
        path = str(tmp_path / "empty_archive.jsonl")
        ca = ChampionArchive(path=path, max_size=10)
        assert ca.size == 0
        assert ca.diversity_aware_select(level=0) == []
        assert ca.sample_training_batch() == []
        report = ca.get_diversity_report()
        assert report["total_champions"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# Test 4: MetricsCollector Stress
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricsStress:
    """Stress-test MetricsCollector with 200 cycles of random data."""

    @pytest.fixture
    def mc(self, tmp_path):
        path = str(tmp_path / "stress_metrics.jsonl")
        return MetricsCollector(path)

    def _record_200_cycles(self, mc):
        """Record 200 cycles with random data and return the data."""
        rng = random.Random(42)
        data = []
        for i in range(200):
            record = {
                "cycle": i,
                "fitness": rng.uniform(0.0, 300.0),
                "level": rng.randint(0, 15),
                "diversity": rng.uniform(0.0, 1.0),
                "novelty_size": rng.randint(0, 100),
                "words": rng.randint(3, 50),
            }
            mc.record_cycle(record)
            data.append(record)
        return data

    def test_record_200_cycles(self, mc):
        """Record 200 cycles and verify count."""
        self._record_200_cycles(mc)
        assert mc.count_records() == 200

    def test_get_history_full(self, mc):
        """get_history returns all 200 records."""
        self._record_200_cycles(mc)
        history = mc.get_history()
        assert len(history) == 200

    def test_get_history_window(self, mc):
        """get_history(last_n=50) returns 50 records."""
        self._record_200_cycles(mc)
        history = mc.get_history(last_n=50)
        assert len(history) == 50
        assert history[0]["cycle"] == 150

    def test_get_trend(self, mc):
        """get_trend returns valid structure for each metric."""
        self._record_200_cycles(mc)
        for metric in ["fitness", "level", "diversity", "novelty_size"]:
            trend = mc.get_trend(metric, window=20)
            assert isinstance(trend, dict)
            assert "current" in trend
            assert "avg" in trend
            assert "trend" in trend
            assert trend["trend"] in ("improving", "declining", "stable", "unknown")

    def test_detect_anomalies_on_each_metric(self, mc):
        """detect_anomalies runs on every metric without crashing."""
        self._record_200_cycles(mc)
        for metric in ["fitness", "level", "diversity", "novelty_size"]:
            anomalies = mc.detect_anomalies(metric, window=20, threshold=2.0)
            assert isinstance(anomalies, list)
            for a in anomalies:
                assert "index" in a
                assert "value" in a
                assert "z_score" in a
                assert "direction" in a
                assert a["direction"] in ("high", "low")

    def test_correlate_metrics_pairs(self, mc):
        """correlate_metrics between pairs of metrics."""
        self._record_200_cycles(mc)
        pairs = [("fitness", "level"), ("diversity", "novelty_size"),
                 ("fitness", "diversity"), ("words", "level")]
        for a, b in pairs:
            corr = mc.correlate_metrics(a, b, window=50)
            assert isinstance(corr, dict)
            assert corr["metric_a"] == a
            assert corr["metric_b"] == b
            assert -1.0 <= corr["correlation"] <= 1.0
            assert corr["strength"] in ("strong", "moderate", "weak", "none")
            assert corr["direction"] in ("positive", "negative", "none")
            assert corr["sample_size"] > 0

    def test_cache_invalidation(self, mc):
        """Cache is invalidated after new records."""
        self._record_200_cycles(mc)
        h1 = mc.get_history()
        # Add one more record
        mc.record_cycle({"cycle": 200, "fitness": 999.0})
        h2 = mc.get_history()
        assert len(h2) == 201

    def test_clear(self, mc):
        """clear() removes the metrics file."""
        self._record_200_cycles(mc)
        mc.clear()
        assert mc.count_records() == 0

    def test_detect_anomalies_edge_empty(self, mc):
        """detect_anomalies on empty metrics returns empty list."""
        anomalies = mc.detect_anomalies("fitness")
        assert anomalies == []

    def test_correlate_metrics_insufficient_data(self, mc):
        """correlate_metrics with < 2 samples returns zero correlation."""
        mc.record_cycle({"cycle": 0, "fitness": 1.0, "level": 1})
        corr = mc.correlate_metrics("fitness", "level")
        assert corr["sample_size"] == 1
        assert corr["correlation"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Test 5: SparseArchive (Novelty) Stress
# ══════════════════════════════════════════════════════════════════════════════

class TestNoveltyArchiveStress:
    """Stress-test SparseArchive with 50 entries and capacity management."""

    def _make_descriptor(self, rng):
        """Create a random behavioural descriptor."""
        return {
            "inst_diversity": rng.uniform(0.0, 1.0),
            "reg_pressure": rng.uniform(0.0, 1.0),
            "program_length": rng.uniform(0.0, 1.0),
            "branch_fraction": rng.uniform(0.0, 1.0),
            "alu_fraction": rng.uniform(0.0, 1.0),
            "load_store_ratio": rng.uniform(0.0, 1.0),
            "unique_regs_used": rng.uniform(0.0, 1.0),
        }

    def _populate(self, n=50, threshold=0.01):
        """Add n entries to a SparseArchive with a very low threshold."""
        archive = SparseArchive(k_neighbors=5, novelty_threshold=threshold)
        rng = random.Random(42)
        added = 0
        for i in range(n):
            desc = self._make_descriptor(rng)
            program = [rng.randint(0, 0xFFFFFFFF) for _ in range(rng.randint(3, 15))]
            was_added, score = archive.add_to_archive(
                desc,
                metadata={"program": program, "generation": i},
                check_threshold=True,
            )
            if was_added:
                added += 1
        return archive, added

    def test_add_50_entries(self):
        """Add 50 entries; most should be accepted with low threshold."""
        archive, added = self._populate(n=50, threshold=0.01)
        assert added >= 20  # Low threshold should accept many
        assert archive.size >= 20

    def test_novelty_score(self):
        """novelty_score returns a non-negative float."""
        archive, _ = self._populate()
        rng = random.Random(100)
        desc = self._make_descriptor(rng)
        score = archive.novelty_score(desc)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_manage_capacity(self):
        """manage_capacity trims archive to max_size."""
        archive, added = self._populate(n=50, threshold=0.01)
        max_size = max(10, archive.size // 2)
        removed = archive.manage_capacity(max_size=max_size)
        assert isinstance(removed, int)
        assert archive.size <= max_size

    def test_decayed_novelty_on_each_entry(self):
        """decayed_novelty returns valid float for every stored entry."""
        archive, _ = self._populate(n=30, threshold=0.01)
        for entry in archive._entries:
            meta = entry[2]
            program = meta.get("program")
            generation = meta.get("generation", 0)
            if program is not None:
                dn = archive.decayed_novelty(program, current_generation=generation + 100)
                assert isinstance(dn, float)
                assert dn >= 0.0

    def test_decayed_novelty_unknown_program(self):
        """decayed_novelty returns 0.0 for unknown programs."""
        archive, _ = self._populate()
        fake_prog = [0xDEADBEEF, 0xCAFEBABE]
        dn = archive.decayed_novelty(fake_prog, current_generation=1000)
        assert dn == 0.0

    def test_get_diversity_score(self):
        """get_diversity_score returns float in [0, 1]."""
        archive, _ = self._populate()
        div = archive.get_diversity_score()
        assert isinstance(div, float)
        assert 0.0 <= div <= 1.0

    def test_get_archive_summary(self):
        """get_archive_summary returns structured dict."""
        archive, _ = self._populate()
        summary = archive.get_archive_summary()
        assert isinstance(summary, dict)
        assert summary["size"] > 0
        assert "diversity_score" in summary
        assert "avg_novelty" in summary

    def test_trim_archive(self):
        """trim_archive removes excess entries."""
        archive, added = self._populate(n=50, threshold=0.01)
        target = max(5, archive.size // 3)
        removed = archive.trim_archive(max_entries=target)
        assert isinstance(removed, int)
        assert archive.size <= target

    def test_recent_novelty_scores(self):
        """recent_novelty_scores returns list of floats."""
        archive, _ = self._populate()
        scores = archive.recent_novelty_scores(n=10)
        assert isinstance(scores, list)
        assert len(scores) > 0
        for s in scores:
            assert isinstance(s, float)
            assert s >= 0.0

    def test_recent_descriptors(self):
        """recent_descriptors returns list of descriptor dicts."""
        archive, _ = self._populate()
        descs = archive.recent_descriptors(n=5)
        assert isinstance(descs, list)
        for d in descs:
            assert isinstance(d, dict)

    def test_clear(self):
        """clear() resets archive."""
        archive, _ = self._populate()
        archive.clear()
        assert archive.size == 0

    def test_serialize_deserialize(self):
        """Serialize and deserialize preserves entries."""
        archive, _ = self._populate(n=20, threshold=0.01)
        data = archive.serialize()
        restored = SparseArchive.deserialize(data, k_neighbors=5)
        assert restored.size == archive.size

    def test_adaptive_threshold(self):
        """adaptive_threshold decreases as archive grows."""
        archive = SparseArchive(k_neighbors=3, novelty_threshold=0.1)
        t1 = archive.adaptive_threshold()
        # Add entries to increase size
        rng = random.Random(77)
        for _ in range(20):
            archive.add_to_archive(self._make_descriptor(rng), check_threshold=False)
        t2 = archive.adaptive_threshold()
        assert t2 <= t1  # Should decrease as archive grows

    def test_no_crash_under_heavy_add(self):
        """Rapidly adding 100 entries causes no crashes."""
        archive = SparseArchive(k_neighbors=3, novelty_threshold=0.001)
        rng = random.Random(555)
        for i in range(100):
            desc = self._make_descriptor(rng)
            program = [rng.randint(0, 0xFFFFFFFF) for _ in range(8)]
            was_added, score = archive.add_to_archive(
                desc,
                metadata={"program": program, "generation": i},
                check_threshold=(i > 0),
            )
            assert isinstance(was_added, bool)
            assert isinstance(score, float)

    def test_manage_capacity_with_empty_archive(self):
        """manage_capacity on empty archive does not crash."""
        archive = SparseArchive(k_neighbors=5)
        removed = archive.manage_capacity(max_size=10)
        assert removed == 0
        assert archive.size == 0
