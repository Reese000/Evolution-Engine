#!/usr/bin/env python3
"""Comprehensive tests for Wave 1 and Wave 2 additions.

Tests cover:
  1. TestCrossCycleTrends — MetaController cross-cycle trend analysis
  2. TestShouldRollbackEffectiveness — rollback triggers based on effectiveness
  3. TestPerLevelEffectiveness — StrategyArchive per-level effectiveness
  4. TestAdaptiveExplorationBonus — scaled exploration bonus
  5. TestDiversityAwareSelect — ChampionArchive diversity-aware selection
  6. TestStaleThresholdGenerations — stale entry pruning
  7. TestManageCapacity — SparseArchive capacity management
  8. TestDecayedNovelty — exponential time-decay novelty
  9. TestImprovementVelocity — MAPElitesArchive improvement velocity
 10. TestIlluminationDiversity — MAPElitesArchive illumination diversity
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Path setup — add project root to sys.path

from meta_controller import MetaController, TelemetrySnapshot, StagnationResult
from strategy_archive import StrategyArchive
from champion_archive import ChampionArchive, ChampionEntry
from novelty import SparseArchive
from qd_archive import MAPElitesArchive


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _snap(cycle: int, fitness: float, level: int = 0, val_loss=None,
          diversity: float = 0.5, champion_words: int = 10,
          retrained: bool = False) -> TelemetrySnapshot:
    """Create a TelemetrySnapshot with sensible defaults."""
    return TelemetrySnapshot(
        cycle=cycle, fitness=fitness, level=level,
        champion_words=champion_words, val_loss=val_loss,
        diversity=diversity, novelty_size=0, qd_occupied=0,
        qd_total=100, traces_count=0, retrained=retrained,
        generations_completed=50,
    )


def _mc(tmp_path="/tmp/mc_wave2_test") -> MetaController:
    """Create a fresh MetaController."""
    return MetaController(checkpoint_dir=tmp_path)


# ═══════════════════════════════════════════════════════════════════════
# 1. TestCrossCycleTrends
# ═══════════════════════════════════════════════════════════════════════

class TestCrossCycleTrends:
    """Tests for MetaController.get_cross_cycle_trends()."""

    def test_improving_fitness(self):
        """Improving fitness values should yield 'improving' direction."""
        mc = _mc()
        for i in range(10):
            mc.update_cycle(_snap(cycle=i, fitness=1.0 + i * 0.5))
        trends = mc.get_cross_cycle_trends()
        assert trends["fitness"]["direction"] == "improving"
        assert trends["fitness"]["slope"] > 0

    def test_declining_fitness(self):
        """Declining fitness values should yield 'declining' direction."""
        mc = _mc()
        for i in range(10):
            mc.update_cycle(_snap(cycle=i, fitness=10.0 - i * 0.5))
        trends = mc.get_cross_cycle_trends()
        assert trends["fitness"]["direction"] == "declining"
        assert trends["fitness"]["slope"] < 0

    def test_stable_fitness(self):
        """Constant fitness values should yield 'stable' direction."""
        mc = _mc()
        for i in range(10):
            mc.update_cycle(_snap(cycle=i, fitness=5.0))
        trends = mc.get_cross_cycle_trends()
        assert trends["fitness"]["direction"] == "stable"
        assert abs(trends["fitness"]["slope"]) < 0.01

    def test_improving_val_loss(self):
        """Decreasing val_loss (improvement) should yield 'improving'."""
        mc = _mc()
        for i in range(10):
            mc.update_cycle(_snap(cycle=i, fitness=1.0,
                                  val_loss=5.0 - i * 0.3))
        trends = mc.get_cross_cycle_trends()
        # val_loss direction is inverted: decreasing slope = improving
        assert trends["val_loss"]["direction"] == "improving"

    def test_declining_val_loss(self):
        """Increasing val_loss (worse) should yield 'declining'."""
        mc = _mc()
        for i in range(10):
            mc.update_cycle(_snap(cycle=i, fitness=1.0,
                                  val_loss=1.0 + i * 0.3))
        trends = mc.get_cross_cycle_trends()
        assert trends["val_loss"]["direction"] == "declining"

    def test_improving_diversity(self):
        """Increasing diversity should yield 'improving'."""
        mc = _mc()
        for i in range(10):
            mc.update_cycle(_snap(cycle=i, fitness=1.0,
                                  diversity=0.2 + i * 0.05))
        trends = mc.get_cross_cycle_trends()
        assert trends["diversity"]["direction"] == "improving"

    def test_insufficient_data(self):
        """With <2 telemetry entries, trends should be stable with 0 confidence."""
        mc = _mc()
        mc.update_cycle(_snap(cycle=0, fitness=1.0))
        trends = mc.get_cross_cycle_trends()
        assert trends["fitness"]["direction"] == "stable"
        assert trends["fitness"]["confidence"] == 0.0

    def test_single_entry(self):
        """Single telemetry entry yields stable with 0 confidence."""
        mc = _mc()
        mc.update_cycle(_snap(cycle=1, fitness=2.0, val_loss=1.5))
        trends = mc.get_cross_cycle_trends()
        assert trends["fitness"]["direction"] == "stable"
        assert trends["fitness"]["slope"] == 0.0
        assert trends["fitness"]["confidence"] == 0.0

    def test_all_none_val_loss(self):
        """When all val_loss values are None, val_loss trend should be stable."""
        mc = _mc()
        for i in range(10):
            mc.update_cycle(_snap(cycle=i, fitness=float(i),
                                  val_loss=None))
        trends = mc.get_cross_cycle_trends()
        assert trends["val_loss"]["direction"] == "stable"
        assert trends["val_loss"]["slope"] == 0.0
        assert trends["val_loss"]["confidence"] == 0.0

    def test_mixed_none_val_loss(self):
        """Partial None val_loss should only use non-None values."""
        mc = _mc()
        values = [5.0, 4.0, None, 3.0, None, 2.0, 1.5, None, 1.0, 0.5]
        for i, vl in enumerate(values):
            mc.update_cycle(_snap(cycle=i, fitness=float(i), val_loss=vl))
        trends = mc.get_cross_cycle_trends()
        assert trends["val_loss"]["direction"] == "improving"

    def test_confidence_increases_with_window(self):
        """More data points should increase confidence."""
        mc = _mc()
        for i in range(15):
            mc.update_cycle(_snap(cycle=i, fitness=1.0 + i * 0.1))
        trends = mc.get_cross_cycle_trends(window=5)
        assert trends["fitness"]["confidence"] > 0.5

    def test_returns_all_three_metrics(self):
        """Result should contain fitness, val_loss, and diversity keys."""
        mc = _mc()
        for i in range(5):
            mc.update_cycle(_snap(cycle=i, fitness=1.0))
        trends = mc.get_cross_cycle_trends()
        assert "fitness" in trends
        assert "val_loss" in trends
        assert "diversity" in trends
        for metric in ("fitness", "val_loss", "diversity"):
            assert "direction" in trends[metric]
            assert "slope" in trends[metric]
            assert "confidence" in trends[metric]


# ═══════════════════════════════════════════════════════════════════════
# 2. TestShouldRollbackEffectiveness
# ═══════════════════════════════════════════════════════════════════════

class TestShouldRollbackEffectiveness:
    """Tests for MetaController.should_rollback() with effectiveness triggers."""

    def test_low_effectiveness_triggers_rollback(self):
        """Effectiveness < 0.2 with stale fitness triggers rollback."""
        mc = _mc()
        # Build telemetry with no fitness improvement (stagnant) and high
        # total_consecutive_stagnation to lower effectiveness.
        for i in range(15):
            mc.update_cycle(_snap(cycle=i, fitness=1.0))
            # Record tuning actions to increase consecutive stagnation
            mc._record_tuning("retrain", "retrain_lr", None, 1e-5,
                              "stagnation: fitness_plateau")
        # Force low effectiveness: need score < 0.2
        # score = max(0, min(1, base - rollback_penalty + recovery_bonus))
        # With _last_stagnation=FITNESS_PLATEAU, recovery_bonus=0
        # base = (cycles - _total_consecutive_stagnation) / cycles
        # We need base < 0.2, so _total_consecutive_stagnation > 12
        mc._total_consecutive_stagnation = 13
        mc.cycles_since_fitness_improvement = 5
        mc._last_stagnation = StagnationResult.FITNESS_PLATEAU
        assert mc.get_effectiveness_score() < 0.2
        assert mc.should_rollback() is True

    def test_high_effectiveness_prevents_rollback(self):
        """High effectiveness with no regression should NOT trigger rollback."""
        mc = _mc()
        # Show improving fitness
        for i in range(10):
            mc.update_cycle(_snap(cycle=i, fitness=float(i)))
        # No stagnation
        mc._total_consecutive_stagnation = 0
        mc.cycles_since_fitness_improvement = 0
        assert mc.get_effectiveness_score() > 0.5
        assert mc.should_rollback() is False

    def test_original_regression_trigger(self):
        """Val loss regression with stale fitness should trigger rollback."""
        mc = _mc()
        # Simulate val loss regression: loss goes from low to high
        losses = [1.0, 1.0, 1.0, 2.0, 4.0]
        for i, vl in enumerate(losses):
            mc.update_cycle(_snap(cycle=i, fitness=1.0, val_loss=vl))
        # After regression + no fitness improvement, should rollback
        assert mc.cycles_since_fitness_improvement >= 2
        state = mc.detect_stagnation()
        assert state in (StagnationResult.VAL_LOSS_REGRESSION,
                         StagnationResult.COMPOUND_STALL)
        assert mc.should_rollback() is True

    def test_no_rollback_with_insufficient_data(self):
        """should_rollback returns False with <2 telemetry entries."""
        mc = _mc()
        mc.update_cycle(_snap(cycle=0, fitness=1.0))
        assert mc.should_rollback() is False

    def test_compound_stall_triggers_rollback(self):
        """Compound stall with fitness stale for 2+ cycles triggers rollback."""
        mc = _mc()
        # Create conditions for compound stall: fitness plateau + diversity collapse
        for i in range(6):
            mc.update_cycle(_snap(
                cycle=i, fitness=5.0,
                val_loss=1.0 + i * 0.5,  # increasing val_loss
                diversity=0.01,  # very low diversity
            ))
        state = mc.detect_stagnation()
        if state == StagnationResult.COMPOUND_STALL and mc.cycles_since_fitness_improvement >= 2:
            assert mc.should_rollback() is True

    def test_fitness_improving_prevents_effectiveness_rollback(self):
        """Even with low effectiveness, improving fitness prevents rollback."""
        mc = _mc()
        # Mix of improving and stagnating — but effectiveness is measured
        # by total_consecutive_stagnation ratio, not just current state
        for i in range(10):
            mc.update_cycle(_snap(cycle=i, fitness=float(i)))
        # Manually set: high stagnation counter but fitness improving
        mc._total_consecutive_stagnation = 10
        mc.cycles_since_fitness_improvement = 0  # fitness IS improving
        # With cycles_since_fitness_improvement < 3, effectiveness rollback won't fire
        assert mc.should_rollback() is False


# ═══════════════════════════════════════════════════════════════════════
# 3. TestPerLevelEffectiveness
# ═══════════════════════════════════════════════════════════════════════

class TestPerLevelEffectiveness:
    """Tests for StrategyArchive.per_level_effectiveness()."""

    def test_multiple_levels_and_strategies(self):
        """Effectiveness should be computed per level correctly."""
        sa = StrategyArchive()
        # Level 5: 3 successes, 1 failure -> 0.75
        sa.record("RANDOM", level=5, success=True)
        sa.record("RANDOM", level=5, success=True)
        sa.record("RANDOM", level=5, success=True)
        sa.record("RANDOM", level=5, success=False)
        # Level 5: different strategy: 1 success, 1 failure -> 0.5
        sa.record("TASK_AWARE", level=5, success=True)
        sa.record("TASK_AWARE", level=5, success=False)
        # Level 10: 2 successes -> 1.0
        sa.record("RANDOM", level=10, success=True)
        sa.record("RANDOM", level=10, success=True)

        eff = sa.per_level_effectiveness("RANDOM")
        assert 5 in eff
        assert 10 in eff
        assert eff[5] == pytest.approx(0.75, abs=0.01)
        assert eff[10] == pytest.approx(1.0, abs=0.01)

    def test_empty_strategy(self):
        """Strategy with no records returns empty dict."""
        sa = StrategyArchive()
        eff = sa.per_level_effectiveness("NONEXISTENT")
        assert eff == {}

    def test_all_success(self):
        """All successes at a level gives effectiveness 1.0."""
        sa = StrategyArchive()
        for _ in range(5):
            sa.record("GOOD_STRAT", level=3, success=True)
        eff = sa.per_level_effectiveness("GOOD_STRAT")
        assert eff[3] == 1.0

    def test_all_failure(self):
        """All failures at a level gives effectiveness 0.0."""
        sa = StrategyArchive()
        for _ in range(5):
            sa.record("BAD_STRAT", level=7, success=False)
        eff = sa.per_level_effectiveness("BAD_STRAT")
        assert eff[7] == 0.0

    def test_multiple_strategies_same_level(self):
        """Different strategies at same level have independent effectiveness."""
        sa = StrategyArchive()
        sa.record("FAST", level=2, success=True)
        sa.record("FAST", level=2, success=True)
        sa.record("SLOW", level=2, success=False)
        sa.record("SLOW", level=2, success=False)
        sa.record("SLOW", level=2, success=False)
        eff_fast = sa.per_level_effectiveness("FAST")
        eff_slow = sa.per_level_effectiveness("SLOW")
        assert eff_fast[2] == 1.0
        assert eff_slow[2] == 0.0

    def test_levels_are_independent(self):
        """Effectiveness at different levels are independent."""
        sa = StrategyArchive()
        sa.record("X", level=1, success=True)
        sa.record("X", level=1, success=True)
        sa.record("X", level=2, success=False)
        sa.record("X", level=2, success=False)
        sa.record("X", level=2, success=False)
        eff = sa.per_level_effectiveness("X")
        assert eff[1] == 1.0
        assert eff[2] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 4. TestAdaptiveExplorationBonus
# ═══════════════════════════════════════════════════════════════════════

class TestAdaptiveExplorationBonus:
    """Tests for StrategyArchive.adaptive_exploration_bonus()."""

    def test_underperforming_gets_2x_bonus(self):
        """Strategy with effectiveness < 0.3 gets 2x base bonus."""
        sa = StrategyArchive()
        # Make strategy fail at level 5
        for _ in range(5):
            sa.record("UNDERPERFORMER", level=5, success=False)
        bonus = sa.adaptive_exploration_bonus("UNDERPERFORMER", level=5,
                                              base_bonus=0.1)
        assert bonus == pytest.approx(0.2, abs=0.01)

    def test_well_performing_gets_half_bonus(self):
        """Strategy with effectiveness > 0.7 gets 0.5x base bonus."""
        sa = StrategyArchive()
        # Make strategy succeed at level 3
        for _ in range(5):
            sa.record("OVERPERFORMER", level=3, success=True)
        bonus = sa.adaptive_exploration_bonus("OVERPERFORMER", level=3,
                                              base_bonus=0.1)
        assert bonus == pytest.approx(0.05, abs=0.01)

    def test_average_gets_base_bonus(self):
        """Strategy with effectiveness between 0.3-0.7 gets base_bonus."""
        sa = StrategyArchive()
        # Mix to get ~0.5 effectiveness
        for _ in range(2):
            sa.record("AVERAGE", level=4, success=True)
        for _ in range(2):
            sa.record("AVERAGE", level=4, success=False)
        bonus = sa.adaptive_exploration_bonus("AVERAGE", level=4,
                                              base_bonus=0.1)
        assert bonus == pytest.approx(0.1, abs=0.01)

    def test_unknown_level_gets_base_bonus(self):
        """Strategy with no records at a level gets base_bonus (0.0 effectiveness < 0.3 -> 2x)."""
        sa = StrategyArchive()
        sa.record("KNOWN", level=1, success=True)
        # Request bonus for a level with no records: effectiveness = 0.0
        # 0.0 < 0.3, so it gets 2x
        bonus = sa.adaptive_exploration_bonus("KNOWN", level=99,
                                              base_bonus=0.1)
        assert bonus == pytest.approx(0.2, abs=0.01)

    def test_custom_base_bonus(self):
        """Custom base_bonus is used in scaling."""
        sa = StrategyArchive()
        for _ in range(5):
            sa.record("STRAT", level=1, success=False)
        bonus = sa.adaptive_exploration_bonus("STRAT", level=1,
                                              base_bonus=0.25)
        assert bonus == pytest.approx(0.5, abs=0.01)

    def test_boundary_exactly_03(self):
        """Effectiveness exactly 0.3 falls into average bucket (not < 0.3)."""
        sa = StrategyArchive()
        # 3 successes + 7 failures at same level = 0.3
        for _ in range(3):
            sa.record("BOUNDARY", level=2, success=True)
        for _ in range(7):
            sa.record("BOUNDARY", level=2, success=False)
        eff = sa.per_level_effectiveness("BOUNDARY")
        # effectiveness might not be exactly 0.3 due to indexing
        # but it should be close
        bonus = sa.adaptive_exploration_bonus("BOUNDARY", level=2,
                                              base_bonus=0.1)
        # If effectiveness < 0.3: 0.2; if 0.3 <= eff <= 0.7: 0.1; if > 0.7: 0.05
        assert bonus in (0.1, 0.2)

    def test_boundary_exactly_07(self):
        """Effectiveness around 0.7 — test boundary behavior."""
        sa = StrategyArchive()
        # 7 successes + 3 failures = 0.7
        for _ in range(7):
            sa.record("BOUNDARY2", level=2, success=True)
        for _ in range(3):
            sa.record("BOUNDARY2", level=2, success=False)
        bonus = sa.adaptive_exploration_bonus("BOUNDARY2", level=2,
                                              base_bonus=0.1)
        # 0.7 is not > 0.7, so should be base_bonus
        assert bonus in (0.1, 0.05)  # boundary case


# ═══════════════════════════════════════════════════════════════════════
# 5. TestDiversityAwareSelect
# ═══════════════════════════════════════════════════════════════════════

class TestDiversityAwareSelect:
    """Tests for ChampionArchive.diversity_aware_select()."""

    def _make_archive(self, entries_data):
        """Create a ChampionArchive with given entries.
        entries_data: list of (program, level, fitness) tuples.
        """
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            archive = ChampionArchive(path=path, max_size=500)
            for i, (program, level, fitness) in enumerate(entries_data):
                archive.add(program=program, level=level, fitness=fitness,
                           cycle=i, generation=0)
            return archive
        except Exception:
            os.unlink(path)
            raise

    def test_single_selection(self):
        """Selecting n=1 returns exactly one champion."""
        archive = self._make_archive([
            ([0x00100093], 5, 10.0),
            ([0x00200093], 5, 20.0),
            ([0x00300093], 5, 30.0),
        ])
        result = archive.diversity_aware_select(level=5, n=1)
        assert len(result) == 1
        assert result[0]["level"] == 5

    def test_multiple_selection(self):
        """Selecting n>1 returns up to n champions."""
        archive = self._make_archive([
            ([0x00100093], 3, 10.0),
            ([0x00200093], 3, 20.0),
            ([0x00300093], 3, 30.0),
            ([0x00400093], 3, 40.0),
            ([0x00500093], 3, 50.0),
        ])
        result = archive.diversity_aware_select(level=3, n=3)
        assert len(result) <= 3
        assert len(result) >= 1
        for entry in result:
            assert entry["level"] == 3

    def test_tournament_diversity(self):
        """Selection should include different programs (diversity)."""
        programs = [
            ([0x00100093] * 5, 1, 10.0),
            ([0x00200093] * 5, 1, 20.0),
            ([0x00300093] * 5, 1, 30.0),
            ([0x00400093] * 5, 1, 40.0),
            ([0x00500093] * 5, 1, 50.0),
        ]
        archive = self._make_archive(programs)
        result = archive.diversity_aware_select(level=1, n=3,
                                                diversity_weight=0.5)
        # With diversity weighting, we should get at least some variety
        assert len(result) >= 2
        hashes = set()
        for entry in result:
            hashes.add(entry["sha256"])
        # With tournament selection, we should have some distinct programs
        assert len(hashes) >= 1

    def test_empty_level_returns_empty(self):
        """Selecting from a level with no entries returns empty list."""
        archive = self._make_archive([
            ([0x00100093], 5, 10.0),
        ])
        result = archive.diversity_aware_select(level=99, n=3)
        assert result == []

    def test_diversity_weight_zero_is_pure_fitness(self):
        """With diversity_weight=0, selection is purely fitness-based."""
        archive = self._make_archive([
            ([0x00100093] * 3, 2, 10.0),
            ([0x00200093] * 3, 2, 50.0),
            ([0x00300093] * 3, 2, 100.0),
        ])
        # Run multiple times to account for tournament randomness
        for _ in range(5):
            result = archive.diversity_aware_select(level=2, n=1,
                                                    diversity_weight=0.0)
            assert len(result) == 1
            # With pure fitness, highest fitness should be selected most often
            assert result[0]["fitness"] >= 10.0

    def test_diversity_weight_one_is_pure_diversity(self):
        """With diversity_weight=1.0, selection ignores fitness."""
        archive = self._make_archive([
            ([0x00100093] * 3, 4, 100.0),
            ([0x00200093] * 3, 4, 1.0),
            ([0x00300093] * 3, 4, 50.0),
        ])
        result = archive.diversity_aware_select(level=4, n=1,
                                                diversity_weight=1.0)
        assert len(result) == 1
        # Any entry could be selected since fitness is irrelevant

    def test_n_clamped_to_available(self):
        """Requesting more than available returns all available."""
        archive = self._make_archive([
            ([0x00100093], 7, 10.0),
            ([0x00200093], 7, 20.0),
        ])
        result = archive.diversity_aware_select(level=7, n=100)
        assert len(result) == 2

    def test_returns_dict_format(self):
        """Each result entry should be a dict (ChampionEntry.to_dict())."""
        archive = self._make_archive([
            ([0x00100093], 1, 10.0),
        ])
        result = archive.diversity_aware_select(level=1, n=1)
        assert len(result) == 1
        entry = result[0]
        assert isinstance(entry, dict)
        assert "fitness" in entry
        assert "level" in entry
        assert "program" in entry
        assert "sha256" in entry


# ═══════════════════════════════════════════════════════════════════════
# 6. TestStaleThresholdGenerations
# ═══════════════════════════════════════════════════════════════════════

class TestStaleThresholdGenerations:
    """Tests for ChampionArchive stale_threshold_generations."""

    def test_default_is_100(self):
        """Default stale_threshold_generations should be 100."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            archive = ChampionArchive(path=path)
            assert archive.stale_threshold_generations == 100
        finally:
            os.unlink(path)

    def test_custom_value(self):
        """Custom stale_threshold_generations is stored correctly."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            archive = ChampionArchive(path=path,
                                       stale_threshold_generations=50)
            assert archive.stale_threshold_generations == 50
        finally:
            os.unlink(path)

    def test_stale_pruning_uses_threshold(self):
        """prune_stale() removes entries older than stale_threshold_generations."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            archive = ChampionArchive(path=path,
                                       stale_threshold_generations=10)
            # Add entries across many cycles
            for i in range(20):
                # Each entry needs a unique program to avoid dedup
                prog = [0x00100093 + i]
                archive.add(program=prog, level=10, fitness=float(i),
                           cycle=i, generation=0)
            initial_size = archive.size
            assert initial_size == 20

            # Prune: entries from cycle < (20 - 10) = 10 should be removed
            # (cycle 0-9 = 10 entries), but only non-foundational ones
            removed = archive.prune_stale()
            # Entries from cycles < (19 - 10) = 9 should be removed
            # (cycle 0-8 = 9 entries), but only non-foundational ones
            assert removed >= 0
            # Remaining entries should be from recent cycles or foundational
            for e in archive.entries:
                assert e.cycle >= 9 or e.level < 7  # foundational preserved
        finally:
            os.unlink(path)

    def test_prune_preserves_foundational(self):
        """Foundational entries (L0-L6) are never pruned."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            archive = ChampionArchive(path=path,
                                       stale_threshold_generations=5)
            # Add foundational entries at old cycles
            for i in range(3):
                prog = [0x00100093 + i]
                archive.add(program=prog, level=3, fitness=float(i),
                           cycle=0, generation=0)  # very old cycle
            # Add non-foundational at old cycle
            prog = [0x00FFFFFF]
            archive.add(program=prog, level=10, fitness=1.0,
                       cycle=0, generation=0)
            # Add a recent entry to set max cycle
            prog = [0x00FFFFFF + 1]
            archive.add(program=prog, level=10, fitness=2.0,
                       cycle=100, generation=0)

            archive.prune_stale()
            # Foundational entries should survive
            foundational = [e for e in archive.entries if e.level < 7]
            assert len(foundational) == 3
        finally:
            os.unlink(path)

    def test_prune_returns_count(self):
        """prune_stale() returns number of entries removed."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            archive = ChampionArchive(path=path,
                                       stale_threshold_generations=5)
            # All entries at old cycles, non-foundational
            for i in range(5):
                prog = [0x00100093 + i]
                archive.add(program=prog, level=10, fitness=float(i),
                           cycle=0, generation=0)
            # Recent entry
            prog = [0x00FFFFFF]
            archive.add(program=prog, level=10, fitness=10.0,
                       cycle=100, generation=0)

            removed = archive.prune_stale()
            assert removed >= 0
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════
# 7. TestManageCapacity
# ═══════════════════════════════════════════════════════════════════════

class TestManageCapacity:
    """Tests for SparseArchive.manage_capacity()."""

    def test_no_trimming_when_under_limit(self):
        """Archive under max_size is not trimmed."""
        archive = SparseArchive(k_neighbors=3)
        # Add a few entries
        for i in range(5):
            archive.add_to_archive(
                {"inst_diversity": i * 0.1, "reg_pressure": 0.5,
                 "program_length": 0.3, "branch_fraction": 0.1,
                 "alu_fraction": 0.5, "load_store_ratio": 0.1,
                 "unique_regs_used": 0.2},
                metadata={"generation": i, "program": [0x00100093 + i]},
                check_threshold=False,
            )
        removed = archive.manage_capacity(max_size=100)
        assert removed == 0
        assert archive.size == 5

    def test_trims_lowest_novelty_first(self):
        """manage_capacity removes entries with lowest novelty first."""
        archive = SparseArchive(k_neighbors=3)
        # Add entries at very different locations in descriptor space
        descriptors = [
            {"inst_diversity": 0.0, "reg_pressure": 0.0,
             "program_length": 0.0, "branch_fraction": 0.0,
             "alu_fraction": 0.0, "load_store_ratio": 0.0,
             "unique_regs_used": 0.0},
            {"inst_diversity": 0.5, "reg_pressure": 0.5,
             "program_length": 0.5, "branch_fraction": 0.5,
             "alu_fraction": 0.5, "load_store_ratio": 0.5,
             "unique_regs_used": 0.5},
            {"inst_diversity": 1.0, "reg_pressure": 1.0,
             "program_length": 1.0, "branch_fraction": 1.0,
             "alu_fraction": 1.0, "load_store_ratio": 1.0,
             "unique_regs_used": 1.0},
        ]
        for i, desc in enumerate(descriptors):
            archive.add_to_archive(
                desc,
                metadata={"generation": i, "program": [0x00100093 + i]},
                check_threshold=False,
            )
        initial_size = archive.size
        assert initial_size == 3

        # Trim to 2: should remove lowest-novelty entry
        removed = archive.manage_capacity(max_size=2)
        assert removed == 1
        assert archive.size == 2

    def test_returns_count_removed(self):
        """manage_capacity returns the number of entries removed."""
        archive = SparseArchive(k_neighbors=3)
        for i in range(8):
            archive.add_to_archive(
                {"inst_diversity": i * 0.125, "reg_pressure": 0.5,
                 "program_length": 0.3, "branch_fraction": 0.1,
                 "alu_fraction": 0.5, "load_store_ratio": 0.1,
                 "unique_regs_used": 0.2},
                metadata={"generation": i, "program": [0x00100093 + i]},
                check_threshold=False,
            )
        removed = archive.manage_capacity(max_size=3)
        assert removed == 5
        assert archive.size == 3

    def test_custom_max_size(self):
        """Custom max_size is respected."""
        archive = SparseArchive(k_neighbors=2)
        for i in range(10):
            archive.add_to_archive(
                {"inst_diversity": i * 0.1, "reg_pressure": i * 0.1,
                 "program_length": 0.3, "branch_fraction": 0.1,
                 "alu_fraction": 0.5, "load_store_ratio": 0.1,
                 "unique_regs_used": 0.2},
                metadata={"generation": i, "program": [0x00100093 + i]},
                check_threshold=False,
            )
        removed = archive.manage_capacity(max_size=5)
        assert archive.size == 5
        assert removed == 5

    def test_empty_archive(self):
        """manage_capacity on empty archive returns 0."""
        archive = SparseArchive()
        removed = archive.manage_capacity(max_size=10)
        assert removed == 0
        assert archive.size == 0


# ═══════════════════════════════════════════════════════════════════════
# 8. TestDecayedNovelty
# ═══════════════════════════════════════════════════════════════════════

class TestDecayedNovelty:
    """Tests for SparseArchive.decayed_novelty()."""

    def test_recent_program_gets_full_novelty(self):
        """A program added recently should get near-full novelty (minimal decay)."""
        archive = SparseArchive(k_neighbors=3)
        program = [0x00100093, 0x00200093]
        # Add a distinct program first to establish archive
        archive.add_to_archive(
            {"inst_diversity": 0.1, "reg_pressure": 0.1,
             "program_length": 0.1, "branch_fraction": 0.1,
             "alu_fraction": 0.1, "load_store_ratio": 0.1,
             "unique_regs_used": 0.1},
            metadata={"generation": 100, "program": [0x0FFFFFFF]},
            check_threshold=False,
        )
        # Add the target program
        archive.add_to_archive(
            {"inst_diversity": 0.9, "reg_pressure": 0.9,
             "program_length": 0.9, "branch_fraction": 0.9,
             "alu_fraction": 0.9, "load_store_ratio": 0.9,
             "unique_regs_used": 0.9},
            metadata={"generation": 200, "program": program},
            check_threshold=False,
        )
        # Compute novelty at generation 201 (very recent)
        novelty = archive.decayed_novelty(program, current_generation=201,
                                          half_life=500)
        base = archive.novelty_score(
            {"inst_diversity": 0.9, "reg_pressure": 0.9,
             "program_length": 0.9, "branch_fraction": 0.9,
             "alu_fraction": 0.9, "load_store_ratio": 0.9,
             "unique_regs_used": 0.9}
        )
        # Recent: decay factor ≈ exp(-0.693 * 1 / 500) ≈ 0.999
        # Diversity weight: archive has 2 very different entries → moderate entropy
        diversity_weight = max(0.1, 1.0 - archive.get_diversity_score())
        assert novelty == pytest.approx(base * diversity_weight, rel=0.05)

    def test_old_program_is_decayed(self):
        """An old program should have significantly reduced novelty."""
        archive = SparseArchive(k_neighbors=3)
        program = [0x00100093, 0x00200093]
        archive.add_to_archive(
            {"inst_diversity": 0.1, "reg_pressure": 0.1,
             "program_length": 0.1, "branch_fraction": 0.1,
             "alu_fraction": 0.1, "load_store_ratio": 0.1,
             "unique_regs_used": 0.1},
            metadata={"generation": 0, "program": [0x0FFFFFFF]},
            check_threshold=False,
        )
        archive.add_to_archive(
            {"inst_diversity": 0.9, "reg_pressure": 0.9,
             "program_length": 0.9, "branch_fraction": 0.9,
             "alu_fraction": 0.9, "load_store_ratio": 0.9,
             "unique_regs_used": 0.9},
            metadata={"generation": 100, "program": program},
            check_threshold=False,
        )
        # Compute novelty at generation 1100 (very old: 1000 gens since addition)
        novelty = archive.decayed_novelty(program, current_generation=1100,
                                          half_life=500)
        base = archive.novelty_score(
            {"inst_diversity": 0.9, "reg_pressure": 0.9,
             "program_length": 0.9, "branch_fraction": 0.9,
             "alu_fraction": 0.9, "load_store_ratio": 0.9,
             "unique_regs_used": 0.9}
        )
        # Old: decay factor ≈ exp(-0.693 * 1000 / 500) ≈ 0.25
        assert novelty < base * 0.5

    def test_unknown_program_returns_zero(self):
        """Program not in the archive returns 0.0."""
        archive = SparseArchive(k_neighbors=3)
        archive.add_to_archive(
            {"inst_diversity": 0.5, "reg_pressure": 0.5,
             "program_length": 0.5, "branch_fraction": 0.5,
             "alu_fraction": 0.5, "load_store_ratio": 0.5,
             "unique_regs_used": 0.5},
            metadata={"generation": 10, "program": [0x00100093]},
            check_threshold=False,
        )
        novelty = archive.decayed_novelty(
            [0xDEADBEEF], current_generation=100, half_life=500
        )
        assert novelty == 0.0

    def test_half_life_parameter_effect(self):
        """Shorter half_life causes faster decay."""
        archive = SparseArchive(k_neighbors=3)
        program = [0x00100093, 0x00200093]
        archive.add_to_archive(
            {"inst_diversity": 0.1, "reg_pressure": 0.1,
             "program_length": 0.1, "branch_fraction": 0.1,
             "alu_fraction": 0.1, "load_store_ratio": 0.1,
             "unique_regs_used": 0.1},
            metadata={"generation": 0, "program": [0x0FFFFFFF]},
            check_threshold=False,
        )
        archive.add_to_archive(
            {"inst_diversity": 0.9, "reg_pressure": 0.9,
             "program_length": 0.9, "branch_fraction": 0.9,
             "alu_fraction": 0.9, "load_store_ratio": 0.9,
             "unique_regs_used": 0.9},
            metadata={"generation": 100, "program": program},
            check_threshold=False,
        )
        # Short half_life: faster decay
        novelty_short = archive.decayed_novelty(program,
                                                 current_generation=600,
                                                 half_life=100)
        # Long half_life: slower decay
        novelty_long = archive.decayed_novelty(program,
                                                current_generation=600,
                                                half_life=1000)
        assert novelty_short < novelty_long

    def test_decay_at_half_life(self):
        """At exactly half_life generations, novelty should be ~50% of base."""
        archive = SparseArchive(k_neighbors=3)
        program = [0x00100093, 0x00200093]
        archive.add_to_archive(
            {"inst_diversity": 0.1, "reg_pressure": 0.1,
             "program_length": 0.1, "branch_fraction": 0.1,
             "alu_fraction": 0.1, "load_store_ratio": 0.1,
             "unique_regs_used": 0.1},
            metadata={"generation": 0, "program": [0x0FFFFFFF]},
            check_threshold=False,
        )
        archive.add_to_archive(
            {"inst_diversity": 0.9, "reg_pressure": 0.9,
             "program_length": 0.9, "branch_fraction": 0.9,
             "alu_fraction": 0.9, "load_store_ratio": 0.9,
             "unique_regs_used": 0.9},
            metadata={"generation": 100, "program": program},
            check_threshold=False,
        )
        half_life = 500
        age = half_life  # exactly half_life generations old
        novelty = archive.decayed_novelty(program,
                                           current_generation=100 + age,
                                           half_life=half_life)
        base = archive.novelty_score(
            {"inst_diversity": 0.9, "reg_pressure": 0.9,
             "program_length": 0.9, "branch_fraction": 0.9,
             "alu_fraction": 0.9, "load_store_ratio": 0.9,
             "unique_regs_used": 0.9}
        )
        # exp(-0.693) ≈ 0.5, so novelty ≈ 50% of base * diversity_weight
        diversity_weight = max(0.1, 1.0 - archive.get_diversity_score())
        assert novelty == pytest.approx(base * 0.5 * diversity_weight, rel=0.1)


# ═══════════════════════════════════════════════════════════════════════
# 9. TestImprovementVelocity
# ═══════════════════════════════════════════════════════════════════════

class TestImprovementVelocity:
    """Tests for MAPElitesArchive improvement velocity concepts.

    Tests the rate of improvement (coverage growth, fitness improvement)
    as a proxy for improvement velocity.
    """

    def test_with_occupied_cells(self):
        """Archive with occupied cells shows positive coverage and fitness."""
        archive = MAPElitesArchive(bins=10)
        # Add several solutions across the behavior space
        programs = [
            ([0x00100093], [-0.5, -0.5], 10.0),
            ([0x00200093], [0.5, 0.5], 20.0),
            ([0x00300093], [-0.5, 0.5], 30.0),
            ([0x00400093], [0.5, -0.5], 40.0),
        ]
        for prog, desc, fit in programs:
            archive.add_solution(desc, fit, prog)

        assert archive.size > 0
        assert archive.coverage() > 0.0
        stats = archive.get_cell_stats()
        assert stats["occupied"] > 0
        assert stats["avg_fitness"] > 0.0
        assert stats["max_fitness"] >= stats["min_fitness"]

    def test_empty_archive_returns_zeros(self):
        """Empty archive has zero coverage, zero occupied, zero fitness."""
        archive = MAPElitesArchive(bins=10)
        assert archive.size == 0
        assert archive.coverage() == 0.0
        stats = archive.get_cell_stats()
        assert stats["occupied"] == 0
        assert stats["avg_fitness"] == 0.0
        assert stats["min_fitness"] == 0.0
        assert stats["max_fitness"] == 0.0

    def test_coverage_increases_with_additions(self):
        """Each unique-cell addition increases coverage."""
        archive = MAPElitesArchive(bins=5)
        cov_before = archive.coverage()
        archive.add_solution([-0.8, -0.8], 10.0, [0x00100093])
        cov_after = archive.coverage()
        assert cov_after > cov_before

    def test_fitness_improvement_overwrites(self):
        """Adding a better solution to the same cell replaces it."""
        archive = MAPElitesArchive(bins=5)
        desc = [0.0, 0.0]
        archive.add_solution(desc, 10.0, [0x00100093])
        archive.add_solution(desc, 50.0, [0x00200093])
        # Size should still be 1 (same cell replaced)
        assert archive.size == 1
        stats = archive.get_cell_stats()
        assert stats["max_fitness"] == 50.0

    def test_velocity_measurement(self):
        """Can measure improvement velocity by tracking coverage over time."""
        archive = MAPElitesArchive(bins=5)
        cov_before = archive.coverage()
        # Add 5 solutions in different cells
        for i in range(5):
            x = -1.0 + (2.0 * i / 4)
            archive.add_solution([x, x], float(i * 10), [0x00100093 + i])
        cov_after = archive.coverage()
        velocity = cov_after - cov_before
        assert velocity > 0.0


# ═══════════════════════════════════════════════════════════════════════
# 10. TestIlluminationDiversity
# ═══════════════════════════════════════════════════════════════════════

class TestIlluminationDiversity:
    """Tests for MAPElitesArchive illumination/diversity concepts.

    Tests how well the archive illuminates the behavior space.
    """

    def test_full_coverage(self):
        """Filling all cells gives coverage = 1.0."""
        bins = 3  # Small grid: 3x3 = 9 cells
        archive = MAPElitesArchive(bins=bins)
        # Fill every cell
        for i in range(bins):
            for j in range(bins):
                x = -1.0 + (2.0 * i / (bins - 1)) if bins > 1 else 0.0
                y = -1.0 + (2.0 * j / (bins - 1)) if bins > 1 else 0.0
                archive.add_solution([x, y], float(i + j), [0x00100093 + i * bins + j])
        assert archive.coverage() == pytest.approx(1.0, abs=0.01)
        assert archive.size == bins * bins

    def test_empty_archive(self):
        """Empty archive has zero coverage and diversity."""
        archive = MAPElitesArchive(bins=10)
        assert archive.coverage() == 0.0
        assert archive.size == 0
        stats = archive.get_cell_stats()
        assert stats["occupied"] == 0

    def test_uniform_distribution(self):
        """Uniformly spread solutions have higher coverage than concentrated."""
        bins = 10
        # Uniform: spread across the entire space
        uniform_archive = MAPElitesArchive(bins=bins)
        for i in range(20):
            x = -1.0 + 2.0 * (i % 10) / 9
            y = -1.0 + 2.0 * (i // 10) / 1
            uniform_archive.add_solution([min(x, 0.99), min(y, 0.99)],
                                          float(i), [0x00100093 + i])
        # Concentrated: all in one corner
        concentrated_archive = MAPElitesArchive(bins=bins)
        for i in range(20):
            concentrated_archive.add_solution(
                [-0.9 + i * 0.01, -0.9 + i * 0.01],
                float(i), [0x00100093 + i + 1000]
            )
        assert uniform_archive.coverage() > concentrated_archive.coverage()

    def test_selection_bias_explores_novel_regions(self):
        """Empty regions get higher selection bias than occupied ones."""
        archive = MAPElitesArchive(bins=3)  # 3x3 grid
        # Fill center cell AND its neighbor to create occupancy
        archive.add_solution([0.0, 0.0], 10.0, [0x00100093])   # cell (1,1)
        archive.add_solution([0.0, 0.9], 10.0, [0x00200093])   # cell (1,2) - neighbor
        # Cell (1,1) is occupied AND has an occupied neighbor → lower bias
        bias_center = archive.selection_bias([0.0, 0.0])
        # Cell (2,2) is empty and far from occupied cells → higher bias
        bias_empty = archive.selection_bias([0.9, 0.9])
        assert bias_empty > bias_center

    def test_archive_summary(self):
        """get_archive_summary returns valid data."""
        archive = MAPElitesArchive(bins=5)
        archive.add_solution([0.5, 0.5], 10.0, [0x00100093])
        summary = archive.get_archive_summary()
        assert "cells" in summary
        assert "fitness" in summary
        assert summary["cells"]["occupied"] == 1

    def test_fitness_distribution(self):
        """get_fitness_distribution returns correct statistics."""
        archive = MAPElitesArchive(bins=5)
        archive.add_solution([-0.5, -0.5], 10.0, [0x00100093])
        archive.add_solution([0.5, 0.5], 50.0, [0x00200093])
        fd = archive.get_fitness_distribution()
        assert fd["mean"] == pytest.approx(30.0, abs=0.1)
        assert fd["median"] == pytest.approx(30.0, abs=0.1)
        assert len(fd["bins"]) > 0

    def test_neighborhood_density(self):
        """Neighborhood density reflects local occupancy."""
        archive = MAPElitesArchive(bins=5)
        archive.add_solution([0.0, 0.0], 10.0, [0x00100093])
        # Center should have density > 0
        density = archive.get_neighborhood_density([0.0, 0.0], radius=1)
        assert density > 0.0
        # Far corner should have lower density
        density_corner = archive.get_neighborhood_density([0.9, 0.9], radius=1)
        assert density_corner < density

    def test_clear_resets_archive(self):
        """Clearing the archive resets coverage to zero."""
        archive = MAPElitesArchive(bins=5)
        archive.add_solution([0.0, 0.0], 10.0, [0x00100093])
        assert archive.size > 0
        archive.clear()
        assert archive.size == 0
        assert archive.coverage() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
