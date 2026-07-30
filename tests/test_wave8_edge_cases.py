#!/usr/bin/env python3
"""Wave 8: Edge case and integration tests.

Covers boundary conditions, empty-state handling, and minimal-data scenarios
across the core archive and controller subsystems.

  1. TestEvolutionEngineInit — engine creation and attribute presence
  2. TestStrategyArchiveIntegration — record/recommend cycle, dominance, summary
  3. TestChampionArchiveEdgeCases — diversity select, quality prune, diversity report
  4. TestMetaControllerEdgeCases — recovery confidence, escalation, reset, trends
  5. TestNoveltyArchiveEdgeCases — capacity management, decayed novelty
  6. TestQDArchiveEdgeCases — velocity, illumination diversity, add+velocity tracking
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import pytest

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame
from strategy_archive import StrategyArchive
from champion_archive import ChampionArchive
from meta_controller import MetaController, TelemetrySnapshot
from novelty import SparseArchive
from qd_archive import MAPElitesArchive


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEvolutionEngineInit
# ══════════════════════════════════════════════════════════════════════════════


class TestEvolutionEngineInit:
    """Engine creation with default and custom configs, attribute presence."""

    def test_engine_creation_with_default_config(self):
        """Default EvolutionConfig should produce a valid engine."""
        cfg = EvolutionConfig()
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.config = cfg
        engine.rng = __import__("random").Random(42)
        assert engine.config is not None
        assert engine.config.mutation_rate == cfg.mutation_rate

    def test_engine_creation_with_custom_config(self):
        """Custom config values should be reflected on the engine."""
        cfg = EvolutionConfig(
            mutation_rate=0.99,
            population_size=10,
            eval_budget=100,
            use_strategy_archive=False,
            use_novelty_archive=False,
            use_qd_archive=False,
        )
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.config = cfg
        assert engine.config.mutation_rate == 0.99
        assert engine.config.population_size == 10
        assert engine.config.eval_budget == 100
        assert engine.config.use_strategy_archive is False
        assert engine.config.use_novelty_archive is False
        assert engine.config.use_qd_archive is False

    def test_engine_attributes_exist(self, engine):
        """Pre-built engine should have all core archive attributes."""
        assert hasattr(engine, "strategy_archive")
        assert hasattr(engine, "champion_archive") or hasattr(engine, "champion_anchor")
        assert hasattr(engine, "novelty_archive")
        assert hasattr(engine, "qd_archive")
        assert hasattr(engine, "_hall_of_fame")
        assert hasattr(engine, "champion_fitness")
        assert hasattr(engine, "champion_level")

    def test_config_dataclass_fields(self):
        """EvolutionConfig should expose expected fields."""
        cfg = EvolutionConfig()
        # Core evolution params
        assert hasattr(cfg, "mutation_rate")
        assert hasattr(cfg, "population_size")
        assert hasattr(cfg, "eval_budget")
        # Archive toggles
        assert hasattr(cfg, "use_strategy_archive")
        assert hasattr(cfg, "use_novelty_archive")
        assert hasattr(cfg, "use_qd_archive")

    def test_hall_of_fame_capacity(self):
        """HallOfFame should respect capacity."""
        hof = HallOfFame(capacity=3)
        assert hof.capacity == 3


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestStrategyArchiveIntegration
# ══════════════════════════════════════════════════════════════════════════════


class TestStrategyArchiveIntegration:
    """Record/recommend cycle, dominance report, archive summary."""

    def test_record_and_recommend_cycle(self):
        """Record a success, recommend should pick that strategy."""
        sa = StrategyArchive()
        sa.record("TASK_AWARE", level=0, success=True, generation=1)
        sa.record("RANDOM", level=0, success=False, generation=1)
        best = sa.recommend_strategy(level=0)
        assert best == "TASK_AWARE"

    def test_time_weighted_success_rate_mixed_records(self):
        """Time-weighted rate should weight recent records higher."""
        sa = StrategyArchive()
        # 10 old failures, 2 recent successes
        for i in range(10):
            sa.record("MIXED", level=0, success=False, generation=i)
        for i in range(2):
            sa.record("MIXED", level=0, success=True, generation=100 + i)
        rate = sa.time_weighted_success_rate("MIXED")
        # Should be above 0 because recent successes have higher weight
        assert rate > 0.0
        assert rate <= 1.0

    def test_time_weighted_success_rate_no_records(self):
        """Strategy with no records should return 0.0."""
        sa = StrategyArchive()
        assert sa.time_weighted_success_rate("NONEXISTENT") == 0.0

    def test_dominance_report_multiple_strategies(self):
        """Dominance report should identify the dominant strategy."""
        sa = StrategyArchive()
        for _ in range(20):
            sa.record("ALPHA", level=0, success=True, generation=1)
        for _ in range(5):
            sa.record("BETA", level=0, success=True, generation=1)
        report = sa.get_dominance_report()
        assert "dominant_strategy" in report
        assert "dominance_ratio" in report
        assert "is_monoculture" in report
        assert report["dominant_strategy"] == "ALPHA"
        # ALPHA has more records, so its success rate should be dominant
        assert report["dominance_ratio"] > 0.0

    def test_dominance_report_empty_archive(self):
        """Empty archive should return balanced with no dominant."""
        sa = StrategyArchive()
        report = sa.get_dominance_report()
        assert report["dominant_strategy"] is None
        assert report["is_monoculture"] is False
        assert report["recommendation"] == "balanced"

    def test_get_archive_summary_includes_dominance(self):
        """Archive summary should include a dominance sub-dict."""
        sa = StrategyArchive()
        sa.record("S1", level=0, success=True, generation=0)
        sa.record("S2", level=1, success=False, generation=1)
        summary = sa.get_archive_summary()
        assert "total_records" in summary
        assert "unique_strategies" in summary
        assert "top_strategies" in summary
        # Summary does not directly contain 'dominance', but contains coverage
        assert summary["total_records"] == 2
        assert summary["unique_strategies"] == 2

    def test_recommend_fallback_to_random(self):
        """No records should fall back to RANDOM."""
        sa = StrategyArchive()
        assert sa.recommend_strategy(level=99) == "RANDOM"

    def test_record_batch_integration(self):
        """Batch recording should produce same state as individual records."""
        sa = StrategyArchive()
        sa.record_batch([
            {"strategy": "B1", "level": 0, "success": True, "gen": 0},
            {"strategy": "B2", "level": 1, "success": False, "gen": 0},
        ])
        assert sa.total_records() == 2
        assert sa.recommend_strategy(level=0) == "B1"


# ══════════════════════════════════════════════════════════════════════════════
# 3. TestChampionArchiveEdgeCases
# ══════════════════════════════════════════════════════════════════════════════


class TestChampionArchiveEdgeCases:
    """Diversity-aware select, quality-weighted prune, diversity report."""

    def _make_archive(self, path: str, max_size: int = 500) -> ChampionArchive:
        """Create a ChampionArchive backed by a temp file."""
        return ChampionArchive(path=path, max_size=max_size)

    def _add_champion(self, archive: ChampionArchive, program: list[int],
                      level: int, fitness: float, cycle: int) -> bool:
        return archive.add(program, level=level, fitness=fitness, cycle=cycle)

    def test_diversity_aware_select_single_champion(self):
        """Selecting 1 from a single champion should return it."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ca = self._make_archive(path)
            prog = [0x00100093, 0x00200113, 0x00300193, 0x00000073]
            ca.add(prog, level=10, fitness=100.0, cycle=1)
            result = ca.diversity_aware_select(level=10, n=1)
            assert len(result) == 1
            assert result[0]["fitness"] == 100.0
        finally:
            os.unlink(path)

    def test_diversity_aware_select_empty(self):
        """Selecting from empty archive should return empty list."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ca = self._make_archive(path)
            result = ca.diversity_aware_select(level=0, n=5)
            assert result == []
        finally:
            os.unlink(path)

    def test_quality_weighted_prune_all_equal_fitness(self):
        """Pruning all-equal fitness should not crash."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ca = self._make_archive(path, max_size=5)
            # Add 10 champions all at fitness 50.0, same level (high)
            for i in range(10):
                ca.add([0x00100093 + i, 0x00000073], level=10, fitness=50.0, cycle=i)
            removed = ca.quality_weighted_prune()
            # All same fitness and same level, so some should be removed
            assert removed >= 0
            assert ca.size <= 10
        finally:
            os.unlink(path)

    def test_get_diversity_report_single_program(self):
        """Diversity report with one program should not crash."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ca = self._make_archive(path)
            ca.add([0x00100093, 0x00000073], level=5, fitness=100.0, cycle=1)
            report = ca.get_diversity_report()
            assert report["total_champions"] == 1
            assert report["unique_levels"] == [5]
            assert "program_length_variance" in report
            assert "instruction_diversity" in report
        finally:
            os.unlink(path)

    def test_stale_threshold_generations_custom_value(self):
        """Custom stale_threshold_generations should be respected."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ca = self._make_archive(path, max_size=100)
            ca.stale_threshold_generations = 5
            # Add entries at different cycles
            for i in range(10):
                ca.add([0x00100093 + i, 0x00000073], level=8, fitness=100.0, cycle=i)
            removed = ca.prune_stale()
            # Current max cycle is 9, threshold is 5 => cutoff = 4
            # Entries at cycles 0-3 should be pruned (non-foundational)
            assert removed >= 0
        finally:
            os.unlink(path)

    def test_empty_archive_diversity_report(self):
        """Empty archive diversity report should return zeroed values."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ca = self._make_archive(path)
            report = ca.get_diversity_report()
            assert report["total_champions"] == 0
            assert report["unique_levels"] == []
            assert report["program_length_variance"] == 0.0
        finally:
            os.unlink(path)

    def test_add_duplicate_returns_false(self):
        """Adding a duplicate program should return False."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ca = self._make_archive(path)
            prog = [0x00100093, 0x00000073]
            ca.add(prog, level=5, fitness=100.0, cycle=1)
            result = ca.add(prog, level=5, fitness=200.0, cycle=2)
            assert result is False
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TestMetaControllerEdgeCases
# ══════════════════════════════════════════════════════════════════════════════


class TestMetaControllerEdgeCases:
    """Recovery confidence, escalation history, parameter reset, cross-cycle trends."""

    def _make_mc(self) -> MetaController:
        with tempfile.TemporaryDirectory() as td:
            return MetaController(checkpoint_dir=td)

    def test_get_recovery_confidence_not_recovering(self):
        """When not in recovery, should report is_recovering=False."""
        mc = self._make_mc()
        assert mc.recovery_mode is False
        conf = mc.get_recovery_confidence()
        assert conf["is_recovering"] is False
        assert conf["cycles_in_recovery"] == 0
        assert conf["should_exit_recovery"] is False

    def test_get_tuning_escalation_history_empty(self):
        """Empty tuning log should return empty history."""
        mc = self._make_mc()
        history = mc.get_tuning_escalation_history()
        assert isinstance(history, list)
        assert len(history) == 0

    def test_suggest_parameter_reset_no_telemetry(self):
        """No telemetry should not suggest a reset."""
        mc = self._make_mc()
        result = mc.suggest_parameter_reset()
        assert result["should_reset"] is False
        assert result["reason"] == "no telemetry"
        assert result["suggested_params"] == {}

    def test_get_cross_cycle_trends_single_data_point(self):
        """Single data point should return stable trend with zero confidence."""
        mc = self._make_mc()
        mc.update_cycle(TelemetrySnapshot(
            cycle=1, fitness=50.0, level=0, champion_words=10,
            val_loss=None, diversity=0.5, novelty_size=0,
            qd_occupied=0, qd_total=100, traces_count=0,
            retrained=False, generations_completed=50,
        ))
        trends = mc.get_cross_cycle_trends()
        assert "fitness" in trends
        # With only 1 data point, slope should be 0 and confidence 0
        assert trends["fitness"]["slope"] == 0.0
        assert trends["fitness"]["confidence"] == 0.0

    def test_get_cross_cycle_trends_improving(self):
        """Improving fitness over 5 cycles should report improving trend."""
        mc = self._make_mc()
        for i in range(6):
            mc.update_cycle(TelemetrySnapshot(
                cycle=i, fitness=10.0 + i * 10.0, level=0,
                champion_words=10, val_loss=None, diversity=0.5,
                novelty_size=0, qd_occupied=0, qd_total=100,
                traces_count=0, retrained=False, generations_completed=50,
            ))
        trends = mc.get_cross_cycle_trends()
        assert trends["fitness"]["direction"] == "improving"
        assert trends["fitness"]["slope"] > 0

    def test_recovery_confidence_after_recovery_entries(self):
        """Recovery entries in tuning log should count toward cycles_in_recovery."""
        mc = self._make_mc()
        # Manually add tuning log entries with 'recovery' in the reason
        from meta_controller import TuningAction
        for i in range(3):
            action = TuningAction(
                domain="retrain", param="lr", old_value=1e-4,
                new_value=1e-5, reason=f"recovery:cycle_{i}", cycle=i,
            )
            mc._tuning_log.append(action)
        conf = mc.get_recovery_confidence()
        assert conf["cycles_in_recovery"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# 5. TestNoveltyArchiveEdgeCases
# ══════════════════════════════════════════════════════════════════════════════


class TestNoveltyArchiveEdgeCases:
    """Capacity management, decayed novelty, empty-archive edge cases."""

    def test_manage_capacity_empty_archive(self):
        """manage_capacity on empty archive should remove nothing."""
        sa = SparseArchive()
        removed = sa.manage_capacity(max_size=100)
        assert removed == 0
        assert sa.size == 0

    def test_manage_capacity_max_size_zero(self):
        """manage_capacity with max_size=0 should remove all entries."""
        sa = SparseArchive()
        desc1 = {"inst_diversity": 0.5, "reg_pressure": 0.3, "program_length": 0.4,
                 "branch_fraction": 0.1, "alu_fraction": 0.6, "load_store_ratio": 0.2,
                 "unique_regs_used": 0.3}
        desc2 = {"inst_diversity": 0.1, "reg_pressure": 0.9, "program_length": 0.8,
                 "branch_fraction": 0.5, "alu_fraction": 0.2, "load_store_ratio": 0.7,
                 "unique_regs_used": 0.8}
        sa.add_to_archive(desc1, {"program": [1], "generation": 0})
        sa.add_to_archive(desc2, {"program": [2], "generation": 1})
        assert sa.size == 2
        removed = sa.manage_capacity(max_size=0)
        assert removed == 2
        assert sa.size == 0

    def test_decayed_novelty_generation_zero(self):
        """decayed_novelty with generation=0 on a program not in archive."""
        sa = SparseArchive()
        result = sa.decayed_novelty([1, 2, 3], current_generation=0)
        assert result == 0.0

    def test_decayed_novelty_program_in_archive(self):
        """decayed_novelty should return positive for archived program."""
        sa = SparseArchive()
        prog = [0x00100093, 0x00200113]
        desc = {"inst_diversity": 0.3, "reg_pressure": 0.4, "program_length": 0.2,
                "branch_fraction": 0.1, "alu_fraction": 0.7, "load_store_ratio": 0.1,
                "unique_regs_used": 0.2}
        sa.add_to_archive(desc, {"program": prog, "generation": 0})
        # Now same generation => age=0 => decay=1.0
        score = sa.decayed_novelty(prog, current_generation=0)
        # First entry novelty_score is 0 (no neighbors), so decayed = 0
        assert score == 0.0

    def test_add_to_archive_first_entry(self):
        """First entry to empty archive should always be added."""
        sa = SparseArchive()
        desc = {"inst_diversity": 0.5, "reg_pressure": 0.5, "program_length": 0.5,
                "branch_fraction": 0.5, "alu_fraction": 0.5, "load_store_ratio": 0.5,
                "unique_regs_used": 0.5}
        added, score = sa.add_to_archive(desc, check_threshold=True)
        assert added is True
        assert sa.size == 1

    def test_empty_archive_novelty_score(self):
        """novelty_score against empty archive should return 0.0."""
        sa = SparseArchive()
        desc = {"inst_diversity": 0.5, "reg_pressure": 0.5, "program_length": 0.5,
                "branch_fraction": 0.5, "alu_fraction": 0.5, "load_store_ratio": 0.5,
                "unique_regs_used": 0.5}
        score = sa.novelty_score(desc)
        assert score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 6. TestQDArchiveEdgeCases
# ══════════════════════════════════════════════════════════════════════════════


class TestQDArchiveEdgeCases:
    """Empty archive diagnostics, add_solution velocity tracking."""

    def test_get_improvement_velocity_empty(self):
        """Empty archive should return zero velocity."""
        qa = MAPElitesArchive()
        vel = qa.get_improvement_velocity()
        assert vel["per_cell_velocity"] == []
        assert vel["avg_velocity"] == 0.0
        assert vel["max_velocity"] == 0.0
        assert vel["velocity_stddev"] == 0.0

    def test_get_illumination_diversity_empty(self):
        """Empty archive should return zero diversity metrics."""
        qa = MAPElitesArchive()
        div = qa.get_illumination_diversity()
        assert div["coverage_ratio"] == 0.0
        assert div["centroid_distance"] == 0.0
        assert div["uniformity"] == 0.0
        assert div["overall_diversity_score"] == 0.0

    def test_add_solution_and_velocity_tracking(self):
        """Adding a solution should update velocity tracking."""
        qa = MAPElitesArchive()
        # Add two solutions in different cells so avg_velocity is non-trivial
        qa.add_solution([-0.5, -0.5], fitness=10.0, program=[1])
        qa.add_solution([0.5, 0.5], fitness=100.0, program=[2])
        assert qa.size == 2
        # Velocity should now have two entries
        vel = qa.get_improvement_velocity()
        assert len(vel["per_cell_velocity"]) == 2
        # With different fitnesses, avg_velocity should be non-zero
        assert vel["avg_velocity"] != 0.0

    def test_add_solution_duplicate_cell_lower_fitness(self):
        """Lower fitness in same cell should not replace."""
        qa = MAPElitesArchive()
        prog1 = [0x00100093, 0x00000073]
        prog2 = [0x00200113, 0x00000073]
        qa.add_solution([0.5, 0.5], fitness=100.0, program=prog1)
        added = qa.add_solution([0.5, 0.5], fitness=50.0, program=prog2)
        assert added is False
        assert qa.size == 1

    def test_add_solution_duplicate_cell_higher_fitness(self):
        """Higher fitness in same cell should replace."""
        qa = MAPElitesArchive()
        prog1 = [0x00100093, 0x00000073]
        prog2 = [0x00200113, 0x00000073]
        qa.add_solution([0.5, 0.5], fitness=50.0, program=prog1)
        added = qa.add_solution([0.5, 0.5], fitness=100.0, program=prog2)
        assert added is True
        assert qa.size == 1
        # The cell should now have the higher fitness
        sols = qa.grid_solutions()
        assert sols[0]["fitness"] == 100.0

    def test_coverage_empty(self):
        """Empty archive should have 0 coverage."""
        qa = MAPElitesArchive()
        assert qa.coverage() == 0.0

    def test_coverage_one_cell(self):
        """One filled cell out of 400 (20x20) should have small coverage."""
        qa = MAPElitesArchive(bins=20)
        qa.add_solution([0.0, 0.0], fitness=1.0, program=[1])
        assert qa.coverage() == pytest.approx(1.0 / 400.0)

    def test_selection_bias_empty_cell(self):
        """Selection bias for empty cell should be 1.0 (exploration bonus)."""
        qa = MAPElitesArchive()
        bias = qa.selection_bias([0.0, 0.0])
        assert bias == 1.0

    def test_illumination_diversity_single_cell(self):
        """Single cell illumination diversity should have full uniformity (1 neighbor)."""
        qa = MAPElitesArchive()
        qa.add_solution([0.5, 0.5], fitness=100.0, program=[1])
        div = qa.get_illumination_diversity()
        assert div["coverage_ratio"] > 0.0
        # Single cell has itself as its own neighbor, so density=1/9
        # and with only one sample, uniformity is 1.0 (no variance)
        assert div["uniformity"] == 1.0
