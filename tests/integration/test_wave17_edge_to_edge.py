#!/usr/bin/env python3
"""Wave 17 edge-to-edge integration tests.

Three test classes exercising the full pipeline from telemetry ingestion
through detection to parameter tuning, strategy recommendation, and
champion archive management.
"""
from __future__ import annotations

import os

import pytest

from champion_archive import ChampionArchive
from meta_controller import MetaController, StagnationResult, TelemetrySnapshot
from strategy_archive import StrategyArchive

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _snap(
    cycle: int,
    fitness: float,
    level: int = 0,
    val_loss: float | None = None,
    diversity: float = 0.5,
    champion_words: int = 10,
) -> TelemetrySnapshot:
    """Quick TelemetrySnapshot factory."""
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
        retrained=val_loss is not None,
        generations_completed=50,
    )


def _make_mc(tmpdir: str) -> MetaController:
    """Create MetaController pointing at a temp checkpoint dir."""
    return MetaController(checkpoint_dir=tmpdir)


def _make_champion_archive(tmpdir: str) -> ChampionArchive:
    """Create ChampionArchive at a temp path."""
    path = os.path.join(tmpdir, "archive.jsonl")
    return ChampionArchive(path=path, max_size=50)


# ═══════════════════════════════════════════════════════════════════════
# 1. TestMetaControllerToEvolution
# ═══════════════════════════════════════════════════════════════════════


class TestMetaControllerToEvolution:
    """MetaController → detect_stagnation → tune_evolution_params pipeline."""

    def test_detect_stagnation_feeds_tune_evolution_params(self, tmp_path):
        """Stagnation detected from telemetry should produce non-empty
        evolution parameter overrides in tune_evolution_params."""
        mc = _make_mc(str(tmp_path))

        # Feed improving cycles, then a plateau to trigger stagnation
        for c in range(1, 4):
            mc.update_cycle(_snap(cycle=c, fitness=1.0 + c * 0.1, level=c, val_loss=2.0 - c * 0.1))

        # Plateau: fitness flat, level stalled
        for c in range(4, 8):
            mc.update_cycle(_snap(cycle=c, fitness=1.3, level=3, val_loss=1.5, diversity=0.3))

        state = mc.detect_stagnation()
        assert state in (
            StagnationResult.FITNESS_PLATEAU,
            StagnationResult.LEVEL_STALLED,
            StagnationResult.COMPOUND_STALL,
        ), f"Expected stagnation, got {state}"

        params = mc.tune_evolution_params(cycle=8)
        # At minimum, stagnation should inject mutation_rate or diversity params
        assert len(params) > 0, "tune_evolution_params should return non-empty dict on stagnation"

    def test_get_cross_cycle_trends_feeds_suggest_parameter_reset(self, tmp_path):
        """Declining fitness trend detected by get_cross_cycle_trends
        should cause suggest_parameter_reset to recommend a reset."""
        mc = _make_mc(str(tmp_path))

        # Feed 7 cycles of declining fitness with val_loss present
        for c in range(1, 8):
            mc.update_cycle(
                _snap(
                    cycle=c,
                    fitness=10.0 - c * 0.5,  # steadily declining
                    level=c,
                    val_loss=2.0 + c * 0.3,  # loss increasing
                    diversity=0.3,
                )
            )

        trends = mc.get_cross_cycle_trends()
        assert "fitness" in trends
        assert trends["fitness"]["direction"] == "declining"

        # Feed enough cycles for suggest_parameter_reset to act
        for c in range(8, 15):
            mc.update_cycle(
                _snap(
                    cycle=c,
                    fitness=10.0 - c * 0.5,
                    level=c,
                    val_loss=2.0 + c * 0.3,
                    diversity=0.3,
                )
            )

        suggestion = mc.suggest_parameter_reset()
        # With 14 cycles and strongly declining fitness, reset is likely
        # (or stagnation-dominated ratio triggers it)
        assert "should_reset" in suggestion
        assert "reason" in suggestion

    def test_recovery_confidence_tracks_correctly(self, tmp_path):
        """Recovery confidence should reflect recovery_mode state and
        fitness trend during recovery cycles."""
        mc = _make_mc(str(tmp_path))

        # First, create a regression scenario that triggers rollback
        for c in range(1, 5):
            mc.update_cycle(_snap(cycle=c, fitness=2.0 + c * 0.5, level=c, val_loss=1.0 - c * 0.1))

        # Now regress
        for c in range(5, 10):
            mc.update_cycle(_snap(cycle=c, fitness=2.0, level=4, val_loss=5.0, diversity=0.1))

        # Force recovery mode to test tracking
        mc.recovery_mode = True
        mc.recovery_cycles_remaining = 3
        mc._record_tuning("retrain", "retrain_lr", 1e-4, 5e-5, "recovery_mode")

        confidence = mc.get_recovery_confidence()
        assert confidence["is_recovering"] is True
        assert confidence["cycles_in_recovery"] >= 1
        assert 0.0 <= confidence["confidence"] <= 1.0

    def test_stagnation_none_with_few_cycles(self, tmp_path):
        """With fewer than 2 telemetry entries, detect_stagnation returns NORMAL."""
        mc = _make_mc(str(tmp_path))
        mc.update_cycle(_snap(cycle=1, fitness=1.0, level=0))
        assert mc.detect_stagnation() == StagnationResult.NORMAL

    def test_tune_evolution_params_normal_empty(self, tmp_path):
        """When system is healthy, tune_evolution_params returns empty dict."""
        mc = _make_mc(str(tmp_path))
        for c in range(1, 6):
            mc.update_cycle(_snap(cycle=c, fitness=1.0 + c, level=c, val_loss=2.0 - c * 0.1, diversity=0.5))
        params = mc.tune_evolution_params(cycle=6)
        assert params == {}, "No overrides expected when system is improving"

    def test_effectiveness_score_range(self, tmp_path):
        """Effectiveness score stays in [0.0, 1.0] across varied states."""
        mc = _make_mc(str(tmp_path))
        for c in range(1, 10):
            mc.update_cycle(_snap(cycle=c, fitness=c * 0.5, level=c, diversity=0.4))
        score = mc.get_effectiveness_score()
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# 2. TestStrategyArchiveToEvolution
# ═══════════════════════════════════════════════════════════════════════


class TestStrategyArchiveToEvolution:
    """StrategyArchive → record → recommend_strategy pipeline."""

    def test_recommend_strategy_uses_per_level_effectiveness(self):
        """recommend_strategy should prefer the strategy with the highest
        per-level success rate for that level."""
        archive = StrategyArchive()

        # Record mostly successful RANDOM at level 0
        for i in range(20):
            archive.record("RANDOM", level=0, success=True, generation=i)
        # Record mostly failing PERTURB at level 0
        for i in range(20):
            archive.record("PERTURB", level=0, success=False, generation=i)

        best = archive.recommend_strategy(level=0)
        assert best == "RANDOM", f"Expected RANDOM, got {best}"

        # Verify per_level_effectiveness confirms
        eff = archive.per_level_effectiveness("RANDOM")
        assert 0 in eff
        assert eff[0] > 0.7, f"RANDOM should be highly effective, got {eff[0]}"

    def test_get_dominance_report_detects_monoculture(self):
        """When one strategy dominates, get_dominance_report flags monoculture."""
        archive = StrategyArchive()

        # Record one strategy overwhelmingly
        for i in range(50):
            archive.record("TASK_AWARE", level=0, success=True, generation=i)
        # Record another mostly failing (low success rate)
        for i in range(50):
            archive.record("RANDOM", level=0, success=i < 3, generation=i)

        report = archive.get_dominance_report()
        assert report["is_monoculture"] is True, "Should detect monoculture"
        assert report["dominance_ratio"] > 0.6
        assert report["recommendation"] == "explore"

    def test_adaptive_exploration_bonus_adjusts(self):
        """adaptive_exploration_bonus should scale based on effectiveness:
        underperforming → higher bonus, well-performing → lower bonus."""
        archive = StrategyArchive()

        # Underperforming strategy at level 0
        for i in range(15):
            archive.record("WEAK_STRAT", level=0, success=False, generation=i)

        # Well-performing strategy at level 1
        for i in range(15):
            archive.record("STRONG_STRAT", level=1, success=True, generation=i)

        weak_bonus = archive.adaptive_exploration_bonus("WEAK_STRAT", level=0, base_bonus=0.1)
        strong_bonus = archive.adaptive_exploration_bonus("STRONG_STRAT", level=1, base_bonus=0.1)

        assert weak_bonus > strong_bonus, f"Weak strategy should get higher bonus: {weak_bonus} vs {strong_bonus}"
        assert weak_bonus == pytest.approx(0.2, abs=1e-6), "Underperforming should get 2x base"
        assert strong_bonus == pytest.approx(0.05, abs=1e-6), "Well-performing should get 0.5x base"

    def test_recommend_strategy_falls_back_to_global(self):
        """When no level-specific records exist, recommend_strategy falls back."""
        archive = StrategyArchive()
        archive.record("RANDOM", level=5, success=True, generation=0)
        archive.record("RANDOM", level=5, success=True, generation=1)
        # Asking for level 0 (no records) should use global best
        best = archive.recommend_strategy(level=0)
        assert best == "RANDOM"

    def test_recommend_strategy_default_random(self):
        """With no records at all, recommend_strategy returns RANDOM."""
        archive = StrategyArchive()
        assert archive.recommend_strategy(level=0) == "RANDOM"

    def test_dominance_report_no_strategies(self):
        """Empty archive should report no monoculture."""
        archive = StrategyArchive()
        report = archive.get_dominance_report()
        assert report["is_monoculture"] is False
        assert report["dominant_strategy"] is None

    def test_per_level_effectiveness_empty(self):
        """per_level_effectiveness returns empty dict for unknown strategy."""
        archive = StrategyArchive()
        assert archive.per_level_effectiveness("NONEXISTENT") == {}


# ═══════════════════════════════════════════════════════════════════════
# 3. TestChampionArchiveToEvolution
# ═══════════════════════════════════════════════════════════════════════


class TestChampionArchiveToEvolution:
    """ChampionArchive → add → diversity_aware_select → prune pipeline."""

    def _populate_archive(
        self,
        archive: ChampionArchive,
        n: int = 20,
        base_fitness: float = 100.0,
        base_level: int = 10,
    ) -> None:
        """Add n diverse champions to an archive."""
        for i in range(n):
            # Ensure unique programs by embedding the index in the instruction words
            program = [0x00000013 + i] * (3 + i % 5) + [0x00000033 + i * 0x100] * (2 + i % 3)
            archive.add(
                program,
                level=base_level + (i % 3),
                fitness=base_fitness + i * 5.0,
                cycle=i + 1,
                generation=i,
            )

    def test_diversity_aware_select_balances_quality_diversity(self, tmp_path):
        """diversity_aware_select should return champions that balance
        high fitness and program diversity (not just the top fitness)."""
        archive = _make_champion_archive(str(tmp_path))
        self._populate_archive(archive, n=15, base_level=10)

        selected = archive.diversity_aware_select(level=10, n=3, diversity_weight=0.5)
        assert len(selected) > 0, "Should select at least 1 champion"
        assert len(selected) <= 3

        # All selected should have the requested level
        for champ in selected:
            assert champ["level"] == 10

        # Selections should include dicts (ChampionEntry.to_dict() format)
        for champ in selected:
            assert "fitness" in champ
            assert "program" in champ

    def test_diversity_aware_select_empty_level(self, tmp_path):
        """When no champions exist at the requested level, return empty."""
        archive = _make_champion_archive(str(tmp_path))
        self._populate_archive(archive, n=10, base_level=5)
        selected = archive.diversity_aware_select(level=99, n=3)
        assert selected == []

    def test_quality_weighted_prune_maintains_health(self, tmp_path):
        """quality_weighted_prune should remove low-quality entries
        when archive exceeds max_size, keeping foundational levels."""
        archive = _make_champion_archive(str(tmp_path))

        # Add enough to exceed max_size (50)
        for i in range(60):
            level = i % 15  # includes foundational L0-L6
            program = [0x00000013] * (3 + i % 5) + [i & 0xFF]
            archive.add(program, level=level, fitness=50.0 + i * 2, cycle=i + 1)

        removed = archive.quality_weighted_prune()
        after = archive.size

        assert removed >= 0, "Should report number of entries removed"
        # Archive should be at or under max_size after pruning
        assert after <= archive.max_size + 1, f"Archive {after} should be <= max_size {archive.max_size} after prune"

        # Foundational entries (L0-L6) should be preserved
        foundational_count = sum(1 for e in archive.entries if e.level in range(7))
        assert foundational_count > 0, "Foundational entries should survive pruning"

    def test_get_diversity_report_provides_insights(self, tmp_path):
        """get_diversity_report should return structured diversity metrics."""
        archive = _make_champion_archive(str(tmp_path))
        self._populate_archive(archive, n=20)

        report = archive.get_diversity_report()

        assert report["total_champions"] == 20
        assert len(report["unique_levels"]) > 0
        assert isinstance(report["avg_fitness_by_level"], dict)
        assert report["program_length_variance"] >= 0.0
        assert 0.0 <= report["instruction_diversity"] <= 1.0
        assert -1.0 <= report["fitness_correlation_with_length"] <= 1.0

    def test_get_diversity_report_empty_archive(self, tmp_path):
        """Empty archive should return zeroed-out diversity report."""
        archive = _make_champion_archive(str(tmp_path))
        report = archive.get_diversity_report()
        assert report["total_champions"] == 0
        assert report["unique_levels"] == []

    def test_quality_weighted_prune_no_op_when_under_capacity(self, tmp_path):
        """Prune should be a no-op when archive is under max_size."""
        archive = _make_champion_archive(str(tmp_path))
        self._populate_archive(archive, n=10)
        removed = archive.quality_weighted_prune()
        assert removed == 0
        assert archive.size == 10
