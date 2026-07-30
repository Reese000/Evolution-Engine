#!/usr/bin/env python3
"""Comprehensive tests for Wave 4 additions.

Tests cover:
  1. TestTimeWeightedSuccessRate — exponential time-decay success rate
  2. TestDominanceReport — monoculture detection and balanced state
  3. TestQualityWeightedPrune — champion archive quality-based pruning
  4. TestDiversityReport — champion population diversity analysis
  5. TestRecoveryConfidence — MetaController recovery mode assessment
  6. TestTuningEscalationHistory — chronological escalation events
  7. TestPipelineAdaptiveExploration — evolution_engine integration (mock)
  8. TestPipelineNoveltyCapacity — evolution_engine integration (mock)
"""
from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy_archive import StrategyArchive
from champion_archive import ChampionArchive, ChampionEntry
from meta_controller import MetaController, TelemetrySnapshot, TuningAction, StagnationResult
from novelty import SparseArchive


# ══════════════════════════════════════════════════════════════════════
# 1. TestTimeWeightedSuccessRate
# ══════════════════════════════════════════════════════════════════════

class TestTimeWeightedSuccessRate:
    """Tests for StrategyArchive.time_weighted_success_rate()."""

    def test_recent_records_weighted_higher(self):
        """Records added later (more recent) should dominate the score."""
        sa = StrategyArchive()
        # Record 10 failures first (old)
        for i in range(10):
            sa.record("STRAT_A", level=0, success=False, generation=i)
        # Record 10 successes last (recent)
        for i in range(10, 20):
            sa.record("STRAT_A", level=0, success=True, generation=i)

        rate = sa.time_weighted_success_rate("STRAT_A", decay_halflife=100)
        # Recent successes should outweigh old failures → rate > 0.5
        assert rate > 0.5, f"Expected rate > 0.5, got {rate}"

    def test_empty_strategy_returns_zero(self):
        """Strategy with no records should return 0.0."""
        sa = StrategyArchive()
        sa.record("OTHER", level=0, success=True, generation=0)
        rate = sa.time_weighted_success_rate("NONEXISTENT", decay_halflife=100)
        assert rate == 0.0

    def test_decay_halflife_effect(self):
        """Smaller halflife should weight recent records more heavily."""
        sa = StrategyArchive()
        # 8 old failures, 2 recent successes
        for i in range(8):
            sa.record("STRAT_B", level=0, success=False, generation=i)
        for i in range(8, 10):
            sa.record("STRAT_B", level=0, success=True, generation=i)

        rate_short_hl = sa.time_weighted_success_rate("STRAT_B", decay_halflife=5)
        rate_long_hl = sa.time_weighted_success_rate("STRAT_B", decay_halflife=100)

        # Short halflife emphasizes the 2 recent successes more → higher rate
        assert rate_short_hl > rate_long_hl, (
            f"Short halflife ({rate_short_hl}) should exceed long ({rate_long_hl})"
        )

    def test_all_successes(self):
        """All successes should yield a rate near 1.0."""
        sa = StrategyArchive()
        for i in range(20):
            sa.record("STRAT_C", level=0, success=True, generation=i)
        rate = sa.time_weighted_success_rate("STRAT_C")
        assert rate == pytest.approx(1.0, abs=0.01)

    def test_all_failures(self):
        """All failures should yield a rate of 0.0."""
        sa = StrategyArchive()
        for i in range(20):
            sa.record("STRAT_D", level=0, success=False, generation=i)
        rate = sa.time_weighted_success_rate("STRAT_D")
        assert rate == pytest.approx(0.0, abs=0.01)


# ══════════════════════════════════════════════════════════════════════
# 2. TestDominanceReport
# ══════════════════════════════════════════════════════════════════════

class TestDominanceReport:
    """Tests for StrategyArchive.get_dominance_report()."""

    def test_monoculture_detection(self):
        """When one strategy dominates, is_monoculture should be True."""
        sa = StrategyArchive()
        # 90% RANDOM successes
        for i in range(45):
            sa.record("RANDOM", level=0, success=True, generation=i)
        for i in range(45):
            sa.record("RANDOM", level=0, success=True, generation=i + 45)
        # 10% HYBRID (some failures)
        for i in range(5):
            sa.record("HYBRID", level=0, success=False, generation=i)

        report = sa.get_dominance_report()
        assert report["dominant_strategy"] == "RANDOM"
        assert report["is_monoculture"] is True
        assert report["dominance_ratio"] > 0.6
        assert report["recommendation"] == "explore"

    def test_balanced_state(self):
        """When strategies are roughly balanced, is_monoculture should be False."""
        sa = StrategyArchive()
        for i in range(10):
            sa.record("STRAT_A", level=0, success=True, generation=i)
        for i in range(10):
            sa.record("STRAT_B", level=0, success=True, generation=i)
        for i in range(10):
            sa.record("STRAT_C", level=0, success=True, generation=i)

        report = sa.get_dominance_report()
        assert report["is_monoculture"] is False
        assert report["recommendation"] == "balanced"
        assert report["dominance_ratio"] <= 0.6

    def test_empty_archive(self):
        """Empty archive should return safe defaults."""
        sa = StrategyArchive()
        report = sa.get_dominance_report()
        assert report["dominant_strategy"] is None
        assert report["dominance_ratio"] == 0.0
        assert report["is_monoculture"] is False
        assert report["recommendation"] == "balanced"


# ══════════════════════════════════════════════════════════════════════
# 3. TestQualityWeightedPrune
# ══════════════════════════════════════════════════════════════════════

class TestQualityWeightedPrune:
    """Tests for ChampionArchive.quality_weighted_prune()."""

    def test_removes_low_quality_champions(self):
        """Pruning should remove the lowest-quality entries first."""
        ca = ChampionArchive(path="/tmp/test_cwprune.jsonl", max_size=100)
        # Add 15 entries — some low fitness, some high
        for i in range(10):
            ca.add([0x00000013] * (i + 1), level=10, fitness=10.0 + i,
                   cycle=1, generation=i)
        for i in range(5):
            ca.add([0x00000013] * (i + 50), level=10, fitness=500.0 + i,
                   cycle=1, generation=100 + i)
        assert ca.size == 15

        # Manually set max_size to force over-capacity pruning
        ca.max_size = 10
        removed = ca.quality_weighted_prune()
        assert removed > 0
        assert ca.size <= 10

    def test_returns_count(self):
        """quality_weighted_prune should return the number of entries removed."""
        ca = ChampionArchive(path="/tmp/test_cwprune2.jsonl", max_size=3)
        for i in range(10):
            ca.add([0x00000013] * (i + 1), level=10, fitness=float(i),
                   cycle=1, generation=i)
        count = ca.quality_weighted_prune()
        assert isinstance(count, int)
        assert count >= 0

    def test_noop_when_under_capacity(self):
        """When under max_size, nothing should be removed."""
        ca = ChampionArchive(path="/tmp/test_cwprune3.jsonl", max_size=50)
        for i in range(5):
            ca.add([0x00000013] * (i + 1), level=10, fitness=100.0,
                   cycle=1, generation=i)
        removed = ca.quality_weighted_prune()
        assert removed == 0
        assert ca.size == 5

    def test_preserves_foundational_entries(self):
        """Foundational level entries (L0-L6) should never be pruned."""
        ca = ChampionArchive(path="/tmp/test_cwprune4.jsonl", max_size=3)
        # Add foundational entries (level 0-6) with low fitness
        for i in range(3):
            ca.add([0x00000013] * (i + 1), level=i, fitness=10.0,
                   cycle=1, generation=i)
        # Add high-level entries with high fitness
        for i in range(5):
            ca.add([0x00000013] * (i + 10), level=10, fitness=100.0 + i,
                   cycle=1, generation=10 + i)

        ca.quality_weighted_prune()
        # Foundational entries should survive
        foundational = [e for e in ca.entries if e.level < 7]
        assert len(foundational) == 3


# ══════════════════════════════════════════════════════════════════════
# 4. TestDiversityReport
# ══════════════════════════════════════════════════════════════════════

class TestDiversityReport:
    """Tests for ChampionArchive.get_diversity_report()."""

    def test_with_multi_level_champions(self):
        """Report should correctly aggregate across multiple levels."""
        ca = ChampionArchive(path="/tmp/test_divrep.jsonl", max_size=100)
        for i in range(5):
            ca.add([0x00000013] * (i + 5), level=3, fitness=100.0,
                   cycle=1, generation=i)
        for i in range(5):
            ca.add([0x00000013] * (i + 10), level=10, fitness=200.0,
                   cycle=1, generation=i)

        report = ca.get_diversity_report()
        assert report["total_champions"] == 10
        assert 3 in report["unique_levels"]
        assert 10 in report["unique_levels"]
        assert report["instruction_diversity"] >= 0.0
        assert report["program_length_variance"] >= 0.0

    def test_empty_archive(self):
        """Empty archive should return zeroed-out report."""
        ca = ChampionArchive(path="/tmp/test_divrep_empty.jsonl", max_size=100)
        report = ca.get_diversity_report()
        assert report["total_champions"] == 0
        assert report["unique_levels"] == []
        assert report["program_length_variance"] == 0.0
        assert report["instruction_diversity"] == 0.0

    def test_program_length_variance(self):
        """Variance should be higher for programs with varied lengths."""
        ca = ChampionArchive(path="/tmp/test_divrep_var.jsonl", max_size=100)
        # Uniform length programs
        for i in range(10):
            ca.add([0x00000013] * 5, level=10, fitness=100.0,
                   cycle=1, generation=i)
        report = ca.get_diversity_report()
        assert report["program_length_variance"] == 0.0

        # Now add programs with varied lengths
        ca2 = ChampionArchive(path="/tmp/test_divrep_var2.jsonl", max_size=100)
        for i in range(10):
            ca2.add([0x00000013] * (i + 1), level=10, fitness=100.0,
                    cycle=1, generation=i)
        report2 = ca2.get_diversity_report()
        assert report2["program_length_variance"] > 0.0


# ══════════════════════════════════════════════════════════════════════
# 5. TestRecoveryConfidence
# ══════════════════════════════════════════════════════════════════════

class TestRecoveryConfidence:
    """Tests for MetaController.get_recovery_confidence()."""

    def _make_mc_with_recovery(self, recovery_cycles=3):
        """Helper: build a MetaController with simulated recovery state."""
        mc = MetaController(checkpoint_dir="/tmp/mc_recovery_test")
        mc.recovery_mode = True
        # Add consecutive recovery tuning log entries
        for i in range(recovery_cycles):
            mc._tuning_log.append(
                TuningAction("recovery", "param", None, 0.5,
                             f"recovery_cycle_{i}", cycle=i + 1)
            )
        return mc

    def test_recovering_state(self):
        """Should detect active recovery with correct cycle count."""
        mc = self._make_mc_with_recovery(recovery_cycles=3)
        result = mc.get_recovery_confidence()
        assert result["is_recovering"] is True
        assert result["cycles_in_recovery"] == 3

    def test_should_exit_recovery_after_5_cycles(self):
        """should_exit_recovery should be True when >= 5 cycles and stable/declining."""
        mc = self._make_mc_with_recovery(recovery_cycles=5)
        mc.recovery_mode = True
        # Add telemetry for recovery cycles with flat fitness (stable)
        for i in range(5):
            mc.telemetry.append(
                TelemetrySnapshot(
                    cycle=i + 1, fitness=100.0, level=0,
                    champion_words=10, val_loss=1.0, diversity=0.5,
                    novelty_size=0, qd_occupied=0, qd_total=100,
                    traces_count=0, retrained=False,
                    generations_completed=50
                )
            )
        result = mc.get_recovery_confidence()
        assert result["should_exit_recovery"] is True
        assert result["fitness_trend_during_recovery"] in ("stable", "declining")

    def test_non_recovering_state(self):
        """Non-recovering MetaController should report is_recovering=False."""
        mc = MetaController(checkpoint_dir="/tmp/mc_recovery_test2")
        mc.recovery_mode = False
        result = mc.get_recovery_confidence()
        assert result["is_recovering"] is False
        assert result["cycles_in_recovery"] == 0

    def test_improving_trend_high_confidence(self):
        """Recovery with improving fitness should have high confidence."""
        mc = self._make_mc_with_recovery(recovery_cycles=4)
        mc.recovery_mode = True
        # Add telemetry with steadily improving fitness
        for i in range(4):
            mc.telemetry.append(
                TelemetrySnapshot(
                    cycle=i + 1, fitness=100.0 + i * 10, level=0,
                    champion_words=10, val_loss=1.0, diversity=0.5,
                    novelty_size=0, qd_occupied=0, qd_total=100,
                    traces_count=0, retrained=False,
                    generations_completed=50
                )
            )
        result = mc.get_recovery_confidence()
        assert result["fitness_trend_during_recovery"] == "improving"
        assert result["confidence"] == 0.8


# ══════════════════════════════════════════════════════════════════════
# 6. TestTuningEscalationHistory
# ══════════════════════════════════════════════════════════════════════

class TestTuningEscalationHistory:
    """Tests for MetaController.get_tuning_escalation_history()."""

    def test_returns_chronological_events(self):
        """Events should be returned in chronological order."""
        mc = MetaController(checkpoint_dir="/tmp/mc_escalation_test")
        mc._tuning_log = [
            TuningAction("evolution", "mutation_rate", 0.5, 0.55,
                         "stagnation: fitness_plateau", cycle=1),
            TuningAction("retrain", "retrain_lr", 3e-4, 1.5e-4,
                         "regression: val_loss_spike", cycle=2),
            TuningAction("evolution", "mutation_rate", 0.55, 0.65,
                         "stagnation: diversity_collapse", cycle=3),
        ]

        history = mc.get_tuning_escalation_history()
        assert len(history) == 3
        assert history[0]["cycle"] == 1
        assert history[1]["cycle"] == 2
        assert history[2]["cycle"] == 3
        # stagnation_type is the part before ':' in the reason
        assert "stagnation" in history[0]["stagnation_type"]
        assert "mutation_rate" in history[0]["action_taken"]

    def test_empty_tuning_log(self):
        """Empty tuning log should return empty list."""
        mc = MetaController(checkpoint_dir="/tmp/mc_escalation_test2")
        history = mc.get_tuning_escalation_history()
        assert history == []

    def test_action_taken_format(self):
        """action_taken should be in domain.param = new_value format."""
        mc = MetaController(checkpoint_dir="/tmp/mc_escalation_test3")
        mc._tuning_log = [
            TuningAction("retrain", "retrain_lr", 3e-4, 1e-4,
                         "stagnation: fitness_plateau", cycle=1),
        ]
        history = mc.get_tuning_escalation_history()
        assert history[0]["action_taken"] == "retrain.retrain_lr = 0.0001"


# ══════════════════════════════════════════════════════════════════════
# 7. TestPipelineAdaptiveExploration
# ══════════════════════════════════════════════════════════════════════

class TestPipelineAdaptiveExploration:
    """Pipeline integration test: _choose_mutation_strategy calls adaptive_exploration_bonus."""

    def test_calls_adaptive_exploration_bonus(self):
        """Verify adaptive_exploration_bonus is called during strategy selection."""
        from tests.conftest import make_engine
        from mutation_proposer import MutationStrategy

        engine = make_engine(with_archive=True)
        engine.config.use_strategy_archive = True
        # Seed strategy archive with enough records
        for i in range(15):
            engine.strategy_archive.record("HYBRID", level=0, success=True, generation=i)

        mock_bonus = MagicMock(return_value=0.05)
        engine.strategy_archive.adaptive_exploration_bonus = mock_bonus

        # Call _choose_mutation_strategy
        engine._choose_mutation_strategy(idx=0)

        # adaptive_exploration_bonus should have been called at least once
        assert mock_bonus.called, "adaptive_exploration_bonus was not called"

    def test_bonus_triggers_override(self):
        """When bonus is high enough, it should trigger strategy override."""
        from tests.conftest import make_engine

        engine = make_engine(with_archive=True)
        engine.config.use_strategy_archive = True
        engine.rng.seed(42)

        for i in range(15):
            engine.strategy_archive.record("HYBRID", level=0, success=True, generation=i)

        # Force a high bonus that will trigger the override path
        with patch.object(
            engine.strategy_archive, "adaptive_exploration_bonus",
            return_value=0.9
        ):
            # Run multiple times — the override should fire when rng < 0.9
            strategies_used = set()
            for _ in range(20):
                engine._choose_mutation_strategy(idx=0)
                strategies_used.add(engine._last_strategy_used)

            # With a 0.9 bonus and many iterations, we should see at least 2
            # different strategies (the override forces a random non-GROW)
            assert len(strategies_used) >= 2, (
                f"Expected strategy diversity, got: {strategies_used}"
            )


# ══════════════════════════════════════════════════════════════════════
# 8. TestPipelineNoveltyCapacity
# ══════════════════════════════════════════════════════════════════════

class TestPipelineNoveltyCapacity:
    """Pipeline integration test: _update_novelty_archive calls manage_capacity."""

    def test_calls_manage_capacity(self):
        """Verify manage_capacity is called after adding to novelty archive."""
        from tests.conftest import make_engine

        engine = make_engine(with_archive=False)
        engine.champion_changed_since_last_novelty = True
        engine._champion_changed_since_last_novelty = True

        mock_archive = MagicMock()
        mock_archive.size = 5
        mock_archive.add_to_archive.return_value = (True, 0.5)
        engine.novelty_archive = mock_archive

        engine._extract_behavioral_descriptor = MagicMock(return_value={"k": 1.0})

        engine._update_novelty_archive()

        mock_archive.manage_capacity.assert_called_once()

    def test_skips_when_champion_unchanged(self):
        """_update_novelty_archive should skip when champion hasn't changed."""
        from tests.conftest import make_engine

        engine = make_engine()
        engine._champion_changed_since_last_novelty = False
        mock_archive = MagicMock()
        engine.novelty_archive = mock_archive

        engine._update_novelty_archive()

        mock_archive.add_to_archive.assert_not_called()
        mock_archive.manage_capacity.assert_not_called()

    def test_skips_when_no_archive(self):
        """Should silently skip when novelty_archive is None."""
        from tests.conftest import make_engine

        engine = make_engine()
        engine.novelty_archive = None
        # Should not raise
        engine._update_novelty_archive()
