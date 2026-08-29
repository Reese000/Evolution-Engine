#!/usr/bin/env python3
"""test_wave15_final_comprehensive.py — Final comprehensive end-to-end tests.

Covers all new functionality added in wave 15 across 8 test classes:
  1. TestAllNewMethodsExist
  2. TestMetaControllerComplete
  3. TestStrategyArchiveComplete
  4. TestChampionArchiveComplete
  5. TestNoveltyComplete
  6. TestQDArchiveComplete
  7. TestMetricsComplete
  8. TestConfigComplete

Uses pytest, tempfile, random. No existing files modified.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
from typing import Dict, List, Optional

import pytest

# ── Path setup ────────────────────────────────────────────────────────
_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from champion_archive import FOUNDATIONAL_LEVELS, ChampionArchive, ChampionEntry
from config import (
    AutonomousRunnerSection,
    Config,
    EvolutionSection,
    MetaControllerSection,
    RetrainSection,
    load_config,
)
from meta_controller import (
    MetaController,
    StagnationResult,
    TelemetrySnapshot,
    TuningAction,
)
from metrics import MetricsCollector
from novelty import SparseArchive, extract_behavioral_descriptor
from qd_archive import MAPElitesArchive, QDArchive
from strategy_archive import StrategyArchive

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _snap(
    cycle: int = 1,
    fitness: float = 1.0,
    level: int = 0,
    val_loss: float | None = None,
    diversity: float = 0.5,
) -> TelemetrySnapshot:
    """Quick TelemetrySnapshot factory."""
    return TelemetrySnapshot(
        cycle=cycle,
        fitness=fitness,
        level=level,
        champion_words=10,
        val_loss=val_loss,
        diversity=diversity,
        novelty_size=0,
        qd_occupied=0,
        qd_total=100,
        traces_count=0,
        retrained=False,
        generations_completed=50,
    )


def _make_program(length: int = 5, seed: int = 42) -> list[int]:
    """Generate a deterministic RV32I-like program."""
    rng = random.Random(seed)
    # ALU opcodes: low bits 0x33, 0x13; BRANCH 0x63; LOAD 0x03; STORE 0x23; LUI 0x37
    opcodes = [0x33, 0x13, 0x63, 0x03, 0x23, 0x37]
    prog = []
    for _ in range(length):
        opcode = rng.choice(opcodes)
        rd = rng.randint(1, 31) << 7
        rs1 = rng.randint(0, 31) << 15
        imm = rng.randint(0, 127)
        word = opcode | rd | rs1 | imm
        prog.append(word)
    return prog


# ══════════════════════════════════════════════════════════════════════
# 1. TestAllNewMethodsExist — verify all 27+ new methods exist
# ══════════════════════════════════════════════════════════════════════

class TestAllNewMethodsExist:
    """Verify all new methods exist on their respective classes via hasattr."""

    # ── MetaController new methods ────────────────────────────────
    def test_mc_get_cross_cycle_trends(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "get_cross_cycle_trends")
        assert callable(mc.get_cross_cycle_trends)

    def test_mc_get_recovery_confidence(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "get_recovery_confidence")
        assert callable(mc.get_recovery_confidence)

    def test_mc_get_tuning_escalation_history(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "get_tuning_escalation_history")
        assert callable(mc.get_tuning_escalation_history)

    def test_mc_suggest_parameter_reset(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "suggest_parameter_reset")
        assert callable(mc.suggest_parameter_reset)

    def test_mc_get_effectiveness_score(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "get_effectiveness_score")
        assert callable(mc.get_effectiveness_score)

    def test_mc_get_effectiveness_report(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "get_effectiveness_report")
        assert callable(mc.get_effectiveness_report)

    def test_mc_should_rollback(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "should_rollback")
        assert callable(mc.should_rollback)

    def test_mc_detect_stagnation(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "detect_stagnation")
        assert callable(mc.detect_stagnation)

    def test_mc_tune_retrain_params(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "tune_retrain_params")
        assert callable(mc.tune_retrain_params)

    def test_mc_tune_evolution_params(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "tune_evolution_params")
        assert callable(mc.tune_evolution_params)

    def test_mc_accept_checkpoint(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "accept_checkpoint")
        assert callable(mc.accept_checkpoint)

    def test_mc_get_state_summary(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "get_state_summary")
        assert callable(mc.get_state_summary)

    def test_mc_save_state_load_state(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert hasattr(mc, "save_state")
        assert hasattr(mc, "load_state")
        assert callable(mc.save_state)
        assert callable(mc.load_state)

    # ── StrategyArchive new methods ───────────────────────────────
    def test_sa_per_level_effectiveness(self):
        sa = StrategyArchive()
        assert hasattr(sa, "per_level_effectiveness")
        assert callable(sa.per_level_effectiveness)

    def test_sa_adaptive_exploration_bonus(self):
        sa = StrategyArchive()
        assert hasattr(sa, "adaptive_exploration_bonus")
        assert callable(sa.adaptive_exploration_bonus)

    def test_sa_time_weighted_success_rate(self):
        sa = StrategyArchive()
        assert hasattr(sa, "time_weighted_success_rate")
        assert callable(sa.time_weighted_success_rate)

    def test_sa_get_dominance_report(self):
        sa = StrategyArchive()
        assert hasattr(sa, "get_dominance_report")
        assert callable(sa.get_dominance_report)

    def test_sa_summary(self):
        sa = StrategyArchive()
        assert hasattr(sa, "summary")
        assert callable(sa.summary)

    def test_sa_get_strategy_diversity(self):
        sa = StrategyArchive()
        assert hasattr(sa, "get_strategy_diversity")
        assert callable(sa.get_strategy_diversity)

    def test_sa_record_delta(self):
        sa = StrategyArchive()
        assert hasattr(sa, "record_delta")
        assert callable(sa.record_delta)

    def test_sa_avg_delta(self):
        sa = StrategyArchive()
        assert hasattr(sa, "avg_delta")
        assert callable(sa.avg_delta)

    def test_sa_get_archive_summary(self):
        sa = StrategyArchive()
        assert hasattr(sa, "get_archive_summary")
        assert callable(sa.get_archive_summary)

    # ── ChampionArchive new methods ───────────────────────────────
    def test_ca_diversity_aware_select(self):
        ca = ChampionArchive(path="/tmp/test_champions.jsonl", max_size=50)
        assert hasattr(ca, "diversity_aware_select")
        assert callable(ca.diversity_aware_select)

    def test_ca_quality_weighted_prune(self):
        ca = ChampionArchive(path="/tmp/test_champions.jsonl", max_size=50)
        assert hasattr(ca, "quality_weighted_prune")
        assert callable(ca.quality_weighted_prune)

    def test_ca_get_diversity_report(self):
        ca = ChampionArchive(path="/tmp/test_champions.jsonl", max_size=50)
        assert hasattr(ca, "get_diversity_report")
        assert callable(ca.get_diversity_report)

    def test_ca_sample_tiered_batch(self):
        ca = ChampionArchive(path="/tmp/test_champions.jsonl", max_size=50)
        assert hasattr(ca, "sample_tiered_batch")
        assert callable(ca.sample_tiered_batch)

    def test_ca_get_coverage_stats(self):
        ca = ChampionArchive(path="/tmp/test_champions.jsonl", max_size=50)
        assert hasattr(ca, "get_coverage_stats")
        assert callable(ca.get_coverage_stats)

    def test_ca_get_quality_distribution(self):
        ca = ChampionArchive(path="/tmp/test_champions.jsonl", max_size=50)
        assert hasattr(ca, "get_quality_distribution")
        assert callable(ca.get_quality_distribution)

    # ── SparseArchive new methods ─────────────────────────────────
    def test_sp_manage_capacity(self):
        sa = SparseArchive()
        assert hasattr(sa, "manage_capacity")
        assert callable(sa.manage_capacity)

    def test_sp_decayed_novelty(self):
        sa = SparseArchive()
        assert hasattr(sa, "decayed_novelty")
        assert callable(sa.decayed_novelty)

    def test_sp_novelty_score(self):
        sa = SparseArchive()
        assert hasattr(sa, "novelty_score")
        assert callable(sa.novelty_score)

    def test_sp_get_diversity_score(self):
        sa = SparseArchive()
        assert hasattr(sa, "get_diversity_score")
        assert callable(sa.get_diversity_score)

    def test_sp_get_archive_summary(self):
        sa = SparseArchive()
        assert hasattr(sa, "get_archive_summary")
        assert callable(sa.get_archive_summary)

    # ── MAPElitesArchive new methods ──────────────────────────────
    def test_qd_get_improvement_velocity(self):
        qd = MAPElitesArchive(bins=10)
        assert hasattr(qd, "get_improvement_velocity")
        assert callable(qd.get_improvement_velocity)

    def test_qd_get_illumination_diversity(self):
        qd = MAPElitesArchive(bins=10)
        assert hasattr(qd, "get_illumination_diversity")
        assert callable(qd.get_illumination_diversity)

    def test_qd_coverage(self):
        qd = MAPElitesArchive(bins=10)
        assert hasattr(qd, "coverage")
        assert callable(qd.coverage)

    def test_qd_get_archive_summary(self):
        qd = MAPElitesArchive(bins=10)
        assert hasattr(qd, "get_archive_summary")
        assert callable(qd.get_archive_summary)

    def test_qd_get_cell_stats(self):
        qd = MAPElitesArchive(bins=10)
        assert hasattr(qd, "get_cell_stats")
        assert callable(qd.get_cell_stats)

    def test_qd_get_fitness_distribution(self):
        qd = MAPElitesArchive(bins=10)
        assert hasattr(qd, "get_fitness_distribution")
        assert callable(qd.get_fitness_distribution)

    # ── MetricsCollector new methods ──────────────────────────────
    def test_mc_detect_anomalies(self):
        with tempfile.TemporaryDirectory() as td:
            mc = MetricsCollector(os.path.join(td, "m.jsonl"))
            assert hasattr(mc, "detect_anomalies")
            assert callable(mc.detect_anomalies)

    def test_mc_correlate_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            mc = MetricsCollector(os.path.join(td, "m.jsonl"))
            assert hasattr(mc, "correlate_metrics")
            assert callable(mc.correlate_metrics)

    def test_mc_get_trend(self):
        with tempfile.TemporaryDirectory() as td:
            mc = MetricsCollector(os.path.join(td, "m.jsonl"))
            assert hasattr(mc, "get_trend")
            assert callable(mc.get_trend)

    def test_mc_count_records(self):
        with tempfile.TemporaryDirectory() as td:
            mc = MetricsCollector(os.path.join(td, "m.jsonl"))
            assert hasattr(mc, "count_records")
            assert callable(mc.count_records)

    # ── Config new methods ────────────────────────────────────────
    def test_cfg_get_config_summary(self):
        evo = EvolutionSection()
        assert hasattr(evo, "get_config_summary")
        assert callable(evo.get_config_summary)

    def test_cfg_validate_config(self):
        evo = EvolutionSection()
        assert hasattr(evo, "validate_config")
        assert callable(evo.validate_config)


# ══════════════════════════════════════════════════════════════════════
# 2. TestMetaControllerComplete — full lifecycle
# ══════════════════════════════════════════════════════════════════════

class TestMetaControllerComplete:
    """Full MetaController lifecycle: create → add telemetry → detect → trend → score."""

    def test_full_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            mc = MetaController(checkpoint_dir=td)

            # ── Phase 1: Normal improvement ───────────────────────
            for c in range(1, 6):
                mc.update_cycle(_snap(
                    cycle=c,
                    fitness=1.0 + c * 0.5,
                    level=c,
                    val_loss=3.0 - c * 0.3,
                    diversity=0.5,
                ))

            # Should be NORMAL
            assert mc.detect_stagnation() == StagnationResult.NORMAL
            assert mc.best_fitness == 3.5
            assert mc.best_level == 5

            # ── Phase 2: Detect stagnation ────────────────────────
            for c in range(6, 12):
                mc.update_cycle(_snap(
                    cycle=c,
                    fitness=3.5,  # flat — no improvement
                    level=5,
                    val_loss=2.0,
                    diversity=0.5,
                ))

            # fitness_plateau cycles = 3, level_stall = 5
            result = mc.detect_stagnation()
            assert result in (
                StagnationResult.FITNESS_PLATEAU,
                StagnationResult.LEVEL_STALLED,
                StagnationResult.COMPOUND_STALL,
            )

            # ── Cross-cycle trends ────────────────────────────────
            trends = mc.get_cross_cycle_trends()
            assert "fitness" in trends
            assert "val_loss" in trends
            assert "diversity" in trends
            for metric in ("fitness", "val_loss", "diversity"):
                assert "direction" in trends[metric]
                assert "slope" in trends[metric]
                assert "confidence" in trends[metric]

            # ── Tuning escalation history ─────────────────────────
            history = mc.get_tuning_escalation_history()
            assert isinstance(history, list)

            # ── Suggest parameter reset ───────────────────────────
            suggestion = mc.suggest_parameter_reset()
            assert "should_reset" in suggestion
            assert "reason" in suggestion
            assert "suggested_params" in suggestion
            assert isinstance(suggestion["should_reset"], bool)

            # ── Effectiveness score ───────────────────────────────
            score = mc.get_effectiveness_score()
            assert 0.0 <= score <= 1.0

            # ── Recovery confidence ───────────────────────────────
            conf = mc.get_recovery_confidence()
            assert "is_recovering" in conf
            assert "cycles_in_recovery" in conf
            assert "fitness_trend_during_recovery" in conf
            assert "confidence" in conf
            assert "should_exit_recovery" in conf
            assert isinstance(conf["is_recovering"], bool)
            assert isinstance(conf["confidence"], float)

            # ── Should rollback check ─────────────────────────────
            # With fitness plateau and fitness not improving for > 2 cycles,
            # rollback may or may not trigger depending on effectiveness.
            rb = mc.should_rollback()
            assert isinstance(rb, bool)

            # ── State summary ─────────────────────────────────────
            summary = mc.get_state_summary()
            assert "best_fitness" in summary
            assert "best_level" in summary
            assert "recovery_mode" in summary
            assert "tuning_history" in summary

            # ── Effectiveness report ──────────────────────────────
            report = mc.get_effectiveness_report()
            assert "total_tuning_actions" in report
            assert "recovery_mode" in report
            assert "total_consecutive_stagnation" in report

            # ── Save/load state round-trip ────────────────────────
            state_path = os.path.join(td, "mc_state.json")
            mc.save_state(state_path)
            assert os.path.isfile(state_path)

            mc2 = MetaController(checkpoint_dir=td)
            loaded = mc2.load_state(state_path)
            assert loaded is True
            assert mc2.best_fitness == mc.best_fitness
            assert mc2.best_level == mc.best_level

    def test_empty_meta_controller(self):
        mc = MetaController(checkpoint_dir="/tmp")
        assert mc.detect_stagnation() == StagnationResult.NORMAL
        assert mc.get_effectiveness_score() == 0.5  # neutral
        assert mc.should_rollback() is False
        assert mc.get_cross_cycle_trends() == {
            "fitness": {"direction": "stable", "slope": 0.0, "confidence": 0.0},
            "val_loss": {"direction": "stable", "slope": 0.0, "confidence": 0.0},
            "diversity": {"direction": "stable", "slope": 0.0, "confidence": 0.0},
        }

    def test_detect_stagnation_caching(self):
        mc = MetaController(checkpoint_dir="/tmp")
        for c in range(1, 6):
            mc.update_cycle(_snap(cycle=c, fitness=1.0, level=0, diversity=0.5))
        r1 = mc.detect_stagnation()
        r2 = mc.detect_stagnation()
        assert r1 is r2  # cached result should be same enum instance


# ══════════════════════════════════════════════════════════════════════
# 3. TestStrategyArchiveComplete — full lifecycle
# ══════════════════════════════════════════════════════════════════════

class TestStrategyArchiveComplete:
    """Full StrategyArchive lifecycle: create → record → recommend → analyze."""

    def test_full_lifecycle(self):
        sa = StrategyArchive(max_records=200)
        rng = random.Random(42)
        strategies = ["RANDOM", "TASK_AWARE", "PERTURBATION", "CROSSOVER"]

        # Record outcomes across levels
        for gen in range(10):
            for lev in range(1, 4):
                strat = rng.choice(strategies)
                success = rng.random() > 0.4
                sa.record(strat, lev, success, generation=gen)

        sa.refresh_timestamp()

        # ── Recommend strategy ────────────────────────────────────
        rec = sa.recommend_strategy(level=1)
        assert isinstance(rec, str)
        assert rec in strategies or rec == "RANDOM"

        # ── Per-level effectiveness ───────────────────────────────
        for strat in strategies:
            eff = sa.per_level_effectiveness(strat)
            assert isinstance(eff, dict)
            for lev, score in eff.items():
                assert 0.0 <= score <= 1.0

        # ── Adaptive exploration bonus ────────────────────────────
        for strat in strategies:
            bonus = sa.adaptive_exploration_bonus(strat, level=1)
            assert 0.0 <= bonus <= 0.5  # base 0.1 * 2 = 0.2 max

        # ── Time-weighted success rate ────────────────────────────
        for strat in strategies:
            twr = sa.time_weighted_success_rate(strat)
            assert 0.0 <= twr <= 1.0

        # ── Dominance report ──────────────────────────────────────
        dr = sa.get_dominance_report()
        assert "dominant_strategy" in dr
        assert "dominance_ratio" in dr
        assert "is_monoculture" in dr
        assert "recommendation" in dr
        assert dr["recommendation"] in ("explore", "balanced")

        # ── Summary ───────────────────────────────────────────────
        s = sa.summary()
        assert "total_records" in s
        assert "strategies" in s
        assert "dominance" in s
        assert s["total_records"] == 30  # 10 gens * 3 levels

        # ── Archive summary ───────────────────────────────────────
        asum = sa.get_archive_summary()
        assert "total_records" in asum
        assert "unique_strategies" in asum
        assert "global_diversity" in asum
        assert "top_strategies" in asum

        # ── Strategy diversity ────────────────────────────────────
        div = sa.get_strategy_diversity()
        assert 0.0 <= div <= 1.0

        # ── Record delta and avg_delta ────────────────────────────
        sa.record_delta("RANDOM", 5.0)
        sa.record_delta("RANDOM", 3.0)
        avg = sa.avg_delta("RANDOM")
        assert abs(avg - 4.0) < 0.01
        assert sa.avg_delta("NONEXISTENT") == 0.0

    def test_batch_record(self):
        sa = StrategyArchive()
        records = [
            {"strategy": "A", "level": 1, "success": True, "gen": 0},
            {"strategy": "B", "level": 2, "success": False, "gen": 0},
            {"strategy": "A", "level": 1, "success": True, "gen": 1},
        ]
        sa.record_batch(records)
        assert sa.total_records() == 3
        rate = sa.success_rate("A", level=1)
        assert rate > 0.5

    def test_max_records_trim(self):
        sa = StrategyArchive(max_records=5)
        for i in range(10):
            sa.record("A", 1, True, generation=i)
        assert sa.total_records() <= 5


# ══════════════════════════════════════════════════════════════════════
# 4. TestChampionArchiveComplete — full lifecycle
# ══════════════════════════════════════════════════════════════════════

class TestChampionArchiveComplete:
    """Full ChampionArchive lifecycle: create → add → select → diversity → prune."""

    def _make_archive(self, td: str, max_size: int = 50) -> ChampionArchive:
        path = os.path.join(td, "archive.jsonl")
        return ChampionArchive(path=path, max_size=max_size)

    def _add_champions(self, ca: ChampionArchive, n: int = 10, start_seed: int = 0):
        """Add n unique champions."""
        for i in range(n):
            prog = _make_program(length=5 + i, seed=start_seed + i)
            ca.add(
                program=prog,
                level=i % 13,
                fitness=100.0 + i * 10,
                cycle=i,
            )

    def test_full_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            ca = self._make_archive(td, max_size=100)
            self._add_champions(ca, n=15)

            assert ca.size >= 10

            # ── Sample training batch ─────────────────────────────
            batch = ca.sample_training_batch(batch_size=5, rng=random.Random(42))
            assert len(batch) > 0
            assert len(batch) <= 5
            for entry in batch:
                assert isinstance(entry, ChampionEntry)

            # ── Tiered batch ──────────────────────────────────────
            tiered = ca.sample_tiered_batch(batch_size=10, rng=random.Random(42))
            assert len(tiered) > 0

            # ── Diversity-aware select ────────────────────────────
            level_to_select = 3
            selected = ca.diversity_aware_select(level=level_to_select, n=2)
            assert isinstance(selected, list)
            for entry in selected:
                assert "fitness" in entry
                assert "level" in entry

            # ── Diversity report ──────────────────────────────────
            dr = ca.get_diversity_report()
            assert "total_champions" in dr
            assert "unique_levels" in dr
            assert "avg_fitness_by_level" in dr
            assert "program_length_variance" in dr
            assert "instruction_diversity" in dr
            assert "fitness_correlation_with_length" in dr
            assert dr["total_champions"] > 0

            # ── Quality weighted prune ────────────────────────────
            # No pruning when under capacity
            pruned = ca.quality_weighted_prune()
            assert pruned == 0

            # ── Coverage stats ────────────────────────────────────
            cov = ca.get_coverage_stats()
            assert "total_entries" in cov
            assert "unique_cycles" in cov
            assert "foundational_count" in cov

            # ── Quality distribution ──────────────────────────────
            qd = ca.get_quality_distribution()
            assert "mean" in qd
            assert "median" in qd

            # ── Stats ─────────────────────────────────────────────
            st = ca.stats()
            assert "size" in st
            assert "max_fitness" in st

    def test_deduplication(self):
        with tempfile.TemporaryDirectory() as td:
            ca = self._make_archive(td)
            prog = _make_program(seed=100)
            added1 = ca.add(prog, level=5, fitness=100.0, cycle=1)
            added2 = ca.add(prog, level=5, fitness=100.0, cycle=1)
            assert added1 is True
            assert added2 is False  # duplicate

    def test_eviction(self):
        with tempfile.TemporaryDirectory() as td:
            ca = self._make_archive(td, max_size=5)
            for i in range(10):
                ca.add(
                    _make_program(seed=i * 100),
                    level=i % 13,
                    fitness=50.0 + i,
                    cycle=i,
                )
            assert ca.size <= 5  # eviction should have kept under max


# ══════════════════════════════════════════════════════════════════════
# 5. TestNoveltyComplete — SparseArchive lifecycle
# ══════════════════════════════════════════════════════════════════════

class TestNoveltyComplete:
    """SparseArchive: create → add → manage_capacity → decay → novelty_score."""

    def _make_desc(self, seed: int = 0) -> dict:
        rng = random.Random(seed)
        return {k: rng.random() for k in [
            "inst_diversity", "reg_pressure", "program_length",
            "branch_fraction", "alu_fraction", "load_store_ratio",
            "unique_regs_used",
        ]}

    def test_full_lifecycle(self):
        sa = SparseArchive(k_neighbors=3, novelty_threshold=0.1)

        # Add diverse entries
        for i in range(20):
            desc = self._make_desc(seed=i * 10)
            metadata = {"program": list(range(i, i + 5)), "generation": i}
            added, _score = sa.add_to_archive(desc, metadata, check_threshold=True)
            # First entry always added; later ones may be rejected by threshold
            if i == 0:
                assert added is True

        assert sa.size > 0

        # ── Novelty score ─────────────────────────────────────────
        query = self._make_desc(seed=999)
        ns = sa.novelty_score(query)
        assert ns >= 0.0

        # ── Capacity management ───────────────────────────────────
        removed = sa.manage_capacity(max_size=5)
        assert removed >= 0
        assert sa.size <= 5

        # ── Decayed novelty ───────────────────────────────────────
        # program [10,11,12,13,14] was added at generation 1
        prog = list(range(10, 15))
        dn = sa.decayed_novelty(prog, current_generation=10, half_life=50)
        assert dn >= 0.0

        # Unknown program should return 0.0
        dn_unknown = sa.decayed_novelty([999, 998], current_generation=10, half_life=50)
        assert dn_unknown == 0.0

        # ── Diversity score ───────────────────────────────────────
        div = sa.get_diversity_score()
        assert 0.0 <= div <= 1.0

        # ── Archive summary ───────────────────────────────────────
        summary = sa.get_archive_summary()
        assert "size" in summary
        assert "diversity_score" in summary
        assert "avg_novelty" in summary

        # ── Recent novelty scores ─────────────────────────────────
        recent = sa.recent_novelty_scores(n=3)
        assert isinstance(recent, list)

        # ── Serialization round-trip ──────────────────────────────
        data = sa.serialize()
        sa2 = SparseArchive.deserialize(data, k_neighbors=3)
        assert sa2.size == sa.size

    def test_adaptive_threshold(self):
        sa = SparseArchive()
        # Empty archive: adaptive threshold should be high
        t1 = sa.adaptive_threshold()
        assert t1 >= 0.05

        # Add entries to grow archive
        for i in range(100):
            sa.add_to_archive(self._make_desc(seed=i), check_threshold=False)
        t2 = sa.adaptive_threshold()
        assert t2 <= t1  # should decrease as archive grows

    def test_clear(self):
        sa = SparseArchive()
        sa.add_to_archive(self._make_desc(seed=1))
        assert sa.size == 1
        sa.clear()
        assert sa.size == 0


# ══════════════════════════════════════════════════════════════════════
# 6. TestQDArchiveComplete — MAPElitesArchive lifecycle
# ══════════════════════════════════════════════════════════════════════

class TestQDArchiveComplete:
    """MAPElitesArchive: create → add → velocity → illumination → coverage."""

    def test_full_lifecycle(self):
        qd = MAPElitesArchive(bins=10)
        rng = random.Random(42)

        # Add solutions across the grid
        for i in range(50):
            desc = [rng.uniform(-0.99, 0.99), rng.uniform(-0.99, 0.99)]
            prog = _make_program(length=5, seed=i)
            fitness = 50.0 + rng.random() * 100
            added = qd.add_solution(desc, fitness, prog)
            assert isinstance(added, bool)

        assert qd.size > 0

        # ── Coverage ──────────────────────────────────────────────
        cov = qd.coverage()
        assert 0.0 <= cov <= 1.0
        assert cov > 0.0  # should have some cells filled

        # ── Selection bias ────────────────────────────────────────
        sb = qd.selection_bias([0.0, 0.0])
        assert 0.0 <= sb <= 1.0

        # ── Improvement velocity ──────────────────────────────────
        vel = qd.get_improvement_velocity()
        assert "per_cell_velocity" in vel
        assert "avg_velocity" in vel
        assert "max_velocity" in vel
        assert "velocity_stddev" in vel
        assert len(vel["per_cell_velocity"]) > 0

        # ── Illumination diversity ────────────────────────────────
        illum = qd.get_illumination_diversity()
        assert "coverage_ratio" in illum
        assert "centroid_distance" in illum
        assert "uniformity" in illum
        assert "overall_diversity_score" in illum
        assert 0.0 <= illum["coverage_ratio"] <= 1.0
        assert illum["uniformity"] >= 0.0

        # ── Archive summary ───────────────────────────────────────
        summary = qd.get_archive_summary()
        assert "cells" in summary
        assert "fitness" in summary
        assert "rebins" in summary
        assert summary["cells"]["occupied"] > 0

        # ── Cell stats ────────────────────────────────────────────
        stats = qd.get_cell_stats()
        assert "total_cells" in stats
        assert "occupied" in stats
        assert "avg_fitness" in stats
        assert stats["occupied"] > 0

        # ── Fitness distribution ──────────────────────────────────
        fd = qd.get_fitness_distribution()
        assert "bins" in fd
        assert "mean" in fd
        assert fd["mean"] > 0

        # ── Grid solutions ────────────────────────────────────────
        gs = qd.grid_solutions()
        assert len(gs) > 0
        assert "fitness" in gs[0]

        # ── Serialization round-trip ──────────────────────────────
        data = qd.serialize()
        qd2 = MAPElitesArchive.deserialize(data)
        assert qd2.size == qd.size

    def test_empty_archive(self):
        qd = MAPElitesArchive(bins=5)
        assert qd.coverage() == 0.0
        assert qd.size == 0

        vel = qd.get_improvement_velocity()
        assert vel["per_cell_velocity"] == []
        assert vel["avg_velocity"] == 0.0

        illum = qd.get_illumination_diversity()
        assert illum["coverage_ratio"] == 0.0

        summary = qd.get_archive_summary()
        assert summary["cells"]["occupied"] == 0

    def test_replacement_only_if_better(self):
        qd = MAPElitesArchive(bins=5)
        desc = [0.1, 0.1]
        prog1 = _make_program(seed=1)
        prog2 = _make_program(seed=2)

        added1 = qd.add_solution(desc, 100.0, prog1)
        assert added1 is True
        assert qd.size == 1

        # Adding lower fitness to same cell → no change
        added2 = qd.add_solution(desc, 50.0, prog2)
        assert added2 is False
        assert qd.size == 1

        # Adding higher fitness to same cell → replacement
        added3 = qd.add_solution(desc, 200.0, prog2)
        assert added3 is True
        assert qd.size == 1

    def test_clear(self):
        qd = MAPElitesArchive(bins=5)
        qd.add_solution([0.0, 0.0], 100.0, _make_program(seed=1))
        assert qd.size == 1
        qd.clear()
        assert qd.size == 0


# ══════════════════════════════════════════════════════════════════════
# 7. TestMetricsComplete — MetricsCollector lifecycle
# ══════════════════════════════════════════════════════════════════════

class TestMetricsComplete:
    """MetricsCollector: create → record → trend → anomalies → correlate."""

    def test_full_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            mc = MetricsCollector(os.path.join(td, "cycles.jsonl"))

            # ── Record cycles ─────────────────────────────────────
            rng = random.Random(42)
            fitness_values = [50.0 + i * 2.5 + rng.gauss(0, 0.5) for i in range(20)]
            for i, f in enumerate(fitness_values):
                mc.record_cycle({
                    "cycle": i + 1,
                    "fitness": f,
                    "level": i % 5,
                    "val_loss": 2.0 - i * 0.05 + rng.gauss(0, 0.1),
                })

            # ── Count records ─────────────────────────────────────
            count = mc.count_records()
            assert count == 20

            # ── Get history ───────────────────────────────────────
            history = mc.get_history()
            assert len(history) == 20

            last_5 = mc.get_history(last_n=5)
            assert len(last_5) == 5

            # ── Get trend ─────────────────────────────────────────
            trend = mc.get_trend("fitness", window=10)
            assert "current" in trend
            assert "avg" in trend
            assert "min" in trend
            assert "max" in trend
            assert "trend" in trend
            assert trend["trend"] in ("improving", "declining", "stable", "unknown")
            assert trend["current"] is not None

            # ── Detect anomalies ──────────────────────────────────
            # Add an outlier
            mc.record_cycle({"cycle": 21, "fitness": 999.0, "level": 0})
            anomalies = mc.detect_anomalies("fitness", window=20, threshold=2.0)
            assert isinstance(anomalies, list)
            if anomalies:
                a = anomalies[0]
                assert "index" in a
                assert "value" in a
                assert "z_score" in a
                assert "direction" in a
                assert a["direction"] in ("high", "low")

            # ── Correlate metrics ─────────────────────────────────
            corr = mc.correlate_metrics("fitness", "val_loss", window=15)
            assert "metric_a" in corr
            assert "metric_b" in corr
            assert "correlation" in corr
            assert "strength" in corr
            assert "direction" in corr
            assert "sample_size" in corr
            assert -1.0 <= corr["correlation"] <= 1.0
            assert corr["strength"] in ("strong", "moderate", "weak", "none")

            # ── Clear ─────────────────────────────────────────────
            mc.clear()
            assert mc.count_records() == 0

    def test_empty_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            mc = MetricsCollector(os.path.join(td, "empty.jsonl"))
            assert mc.count_records() == 0
            assert mc.get_history() == []
            trend = mc.get_trend("fitness")
            assert trend["current"] is None
            assert trend["window_size"] == 0
            anomalies = mc.detect_anomalies("fitness")
            assert anomalies == []

    def test_correlate_insufficient_data(self):
        with tempfile.TemporaryDirectory() as td:
            mc = MetricsCollector(os.path.join(td, "corr.jsonl"))
            # Both metrics must be present in the record for correlation
            mc.record_cycle({"cycle": 1, "fitness": 1.0, "level": 3})
            corr = mc.correlate_metrics("fitness", "level")
            assert corr["sample_size"] == 1
            # With only 1 pair, correlation is 0.0 (undefined)
            assert corr["correlation"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# 8. TestConfigComplete — Config lifecycle
# ══════════════════════════════════════════════════════════════════════

class TestConfigComplete:
    """Config: create → get_config_summary → validate_config → fields accessible."""

    def test_full_lifecycle(self):
        # ── Create Config ─────────────────────────────────────────
        cfg = Config()

        # ── Verify all sections accessible ────────────────────────
        assert hasattr(cfg, "evolution")
        assert hasattr(cfg, "retrain")
        assert hasattr(cfg, "meta_controller")
        assert hasattr(cfg, "autonomous_runner")
        assert isinstance(cfg.evolution, EvolutionSection)
        assert isinstance(cfg.retrain, RetrainSection)
        assert isinstance(cfg.meta_controller, MetaControllerSection)
        assert isinstance(cfg.autonomous_runner, AutonomousRunnerSection)

        # ── EvolutionSection fields ───────────────────────────────
        evo = cfg.evolution
        assert evo.population_size > 0
        assert evo.max_generations > 0
        assert 0.0 <= evo.mutation_rate <= 1.0
        assert 0.0 <= evo.crossover_rate <= 1.0
        assert 0.0 <= evo.elite_ratio <= 1.0
        assert evo.seed > 0
        assert evo.max_level > 0
        assert evo.novelty_weight >= 0.0
        assert evo.qd_bonus_weight >= 0.0
        assert isinstance(evo.use_novelty_archive, bool)
        assert isinstance(evo.use_qd_archive, bool)
        assert isinstance(evo.use_strategy_archive, bool)
        assert isinstance(evo.use_mutation_scheduler, bool)
        assert isinstance(evo.use_temperature_annealing, bool)
        assert 0.0 <= evo.mutation_rate_start <= 1.0
        assert 0.0 <= evo.mutation_rate_end <= 1.0
        assert 0.0 <= evo.temperature_start <= 5.0
        assert 0.0 <= evo.temperature_end <= 5.0

        # ── RetrainSection fields ─────────────────────────────────
        retrain = cfg.retrain
        assert retrain.retrain_lr > 0
        assert retrain.retrain_batch_size > 0
        assert 0.0 <= retrain.retrain_val_split <= 1.0
        assert retrain.retrain_max_epochs > 0
        assert retrain.augmentation_k > 0

        # ── MetaControllerSection fields ──────────────────────────
        mc = cfg.meta_controller
        assert mc.lr_min > 0
        assert mc.lr_max > mc.lr_min
        assert mc.freeze_min >= 0
        assert mc.freeze_max > mc.freeze_min
        assert mc.patience_min >= 0
        assert mc.patience_max > mc.patience_min
        assert mc.fitness_plateau_cycles > 0
        assert mc.level_stall_cycles > 0
        assert 0.0 < mc.diversity_collapse_threshold < 1.0
        assert mc.val_loss_regression_ratio > 1.0

        # ── AutonomousRunnerSection fields ────────────────────────
        ar = cfg.autonomous_runner
        assert ar.max_cycles is None or ar.max_cycles > 0
        assert ar.target_fitness > 0
        assert ar.generations_per_cycle > 0
        assert ar.max_errors_before_stop > 0

        # ── get_config_summary ────────────────────────────────────
        summary = evo.get_config_summary()
        assert "seed" in summary
        assert "use_gpu" in summary
        assert "auto_curriculum" in summary
        assert "archives_enabled" in summary
        assert "novelty_weight" in summary
        assert "qd_bonus_weight" in summary
        assert "mutation_rate" in summary
        assert "population_size" in summary
        assert "eval_budget" in summary
        assert "max_consecutive_same_strategy" in summary
        assert "total_config_fields" in summary
        assert summary["total_config_fields"] > 20

        # ── validate_config — valid config should be empty ────────
        warnings = evo.validate_config()
        assert isinstance(warnings, list)
        # Default config should have no warnings
        assert len(warnings) == 0

        # ── validate_config — invalid config ──────────────────────
        bad = EvolutionSection()
        bad.novelty_weight = 0.95
        bad.population_size = 5
        bad.eval_budget = 50
        bad.max_consecutive_same_strategy = 2
        bad.use_novelty_archive = False
        bad.use_qd_archive = False
        bad.use_strategy_archive = False
        warnings_bad = bad.validate_config()
        assert len(warnings_bad) >= 3  # novelty_weight, population, eval_budget, strategy, consecutive

        # ── to_dict round-trip ────────────────────────────────────
        d = cfg.to_dict()
        assert "evolution" in d
        assert "retrain" in d
        assert "meta_controller" in d
        assert "autonomous_runner" in d

    def test_load_config_default(self):
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.evolution.population_size > 0

    def test_config_validation_raises(self):
        bad = Config.__new__(Config)
        bad.evolution = EvolutionSection(population_size=-1)
        bad.retrain = RetrainSection()
        bad.meta_controller = MetaControllerSection()
        bad.autonomous_runner = AutonomousRunnerSection()
        with pytest.raises(ValueError):
            bad._validate()
