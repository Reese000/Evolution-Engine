#!/usr/bin/env python3
"""Wave 12 — Final comprehensive test suite.

Tests 6 targeted areas:
  1. Config validation edge cases
  2. Mutation tracker full cycle (DirectedMutator)
  3. Dominance report full cycle (StrategyArchive)
  4. Recovery confidence full cycle (MetaController)
  5. Quality prune full cycle (ChampionArchive)
  6. End-to-end method availability

All tests import from source files directly, are focused and fast.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

# ── Path setup ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, EvolutionSection, load_config
from meta_controller import MetaController, TelemetrySnapshot, TuningAction
from mutation_proposer import DirectedMutator
from strategy_archive import StrategyArchive
from champion_archive import ChampionArchive
from novelty import SparseArchive
from qd_archive import MAPElitesArchive
from metrics import MetricsCollector
from progression_tracker import ProgressionTracker, CycleMetrics


# ═══════════════════════════════════════════════════════════════════════
# 1. TestConfigValidationEdgeCases
# ═══════════════════════════════════════════════════════════════════════


class TestConfigValidationEdgeCases:
    """Tests for EvolutionSection.validate_config() and get_config_summary()."""

    def test_validate_all_warnings_triggered(self):
        """All five warning conditions triggered simultaneously."""
        evo = EvolutionSection(
            novelty_weight=0.95,
            population_size=5,
            eval_budget=50,
            max_consecutive_same_strategy=2,
            use_novelty_archive=False,
            use_qd_archive=False,
            use_strategy_archive=False,
        )
        warnings = evo.validate_config()
        assert len(warnings) >= 5
        # Each warning condition should appear
        joined = "\n".join(warnings)
        assert "novelty_weight" in joined
        assert "population_size" in joined
        assert "eval_budget" in joined
        assert "max_consecutive_same_strategy" in joined
        assert "No diversity mechanisms" in joined

    def test_validate_no_warnings_default(self):
        """Default config produces no warnings."""
        evo = EvolutionSection()
        warnings = evo.validate_config()
        assert warnings == []

    def test_get_config_summary_all_defaults(self):
        """get_config_summary returns correct values for default config."""
        evo = EvolutionSection()
        summary = evo.get_config_summary()

        assert summary["seed"] == 42
        assert summary["use_gpu"] is False
        assert summary["auto_curriculum"] is True
        assert summary["archives_enabled"] is True  # novelty archive on by default
        assert summary["novelty_weight"] == 0.3
        assert summary["qd_bonus_weight"] == 0.1
        assert summary["mutation_rate"] == 0.3
        assert summary["population_size"] == 50
        assert summary["eval_budget"] == 5000
        assert summary["max_consecutive_same_strategy"] == 30
        assert "total_config_fields" in summary
        assert summary["total_config_fields"] > 0

    def test_config_with_custom_seed_and_gpu(self):
        """Config with custom seed and GPU settings."""
        evo = EvolutionSection(seed=12345, use_gpu=True)
        assert evo.seed == 12345
        assert evo.use_gpu is True

        summary = evo.get_config_summary()
        assert summary["seed"] == 12345
        assert summary["use_gpu"] is True

    def test_validate_partial_warnings(self):
        """Only some warning conditions triggered."""
        evo = EvolutionSection(
            novelty_weight=0.9,
            population_size=50,  # fine
            eval_budget=5000,  # fine
            max_consecutive_same_strategy=30,  # fine
            use_novelty_archive=True,
            use_qd_archive=True,
            use_strategy_archive=True,
        )
        warnings = evo.validate_config()
        assert len(warnings) == 1
        assert "novelty_weight" in warnings[0]

    def test_config_to_dict_roundtrip(self):
        """Config can be serialized to dict and back."""
        cfg = Config(config_path=None)
        d = cfg.to_dict()
        assert "evolution" in d
        assert "retrain" in d
        assert "meta_controller" in d
        assert "autonomous_runner" in d
        assert d["evolution"]["seed"] == 42


# ═══════════════════════════════════════════════════════════════════════
# 2. TestMutationTrackerFullCycle
# ═══════════════════════════════════════════════════════════════════════


class TestMutationTrackerFullCycle:
    """Tests for DirectedMutator mutation tracking (static class-level state)."""

    @pytest.fixture(autouse=True)
    def _reset_tracker(self):
        """Reset DirectedMutator class-level tracking state before each test."""
        DirectedMutator._mutation_success.clear()
        DirectedMutator._mutation_attempts.clear()
        DirectedMutator._mutation_history.clear()
        yield
        DirectedMutator._mutation_success.clear()
        DirectedMutator._mutation_attempts.clear()
        DirectedMutator._mutation_history.clear()

    def test_empty_tracker_state(self):
        """Empty tracker returns zeroed stats."""
        stats = DirectedMutator.get_mutation_stats()
        assert stats["total_attempts"] == 0
        assert stats["total_successes"] == 0
        assert stats["by_type"] == {}
        assert stats["recent_avg_delta"] == 0.0

    def test_record_single_outcome(self):
        """Record a single successful mutation."""
        DirectedMutator.record_mutation_outcome("ADDI_replace_success", 1.5)
        stats = DirectedMutator.get_mutation_stats()

        assert stats["total_attempts"] == 1
        assert stats["total_successes"] == 1
        assert stats["recent_avg_delta"] == 1.5

    def test_record_multiple_outcomes(self):
        """Record multiple outcomes with mixed success/failure."""
        DirectedMutator.record_mutation_outcome("ADDI_replace_success", 2.0)
        DirectedMutator.record_mutation_outcome("ADDI_replace_failure", -1.0)
        DirectedMutator.record_mutation_outcome("SUB_insert_success", 3.0)
        DirectedMutator.record_mutation_outcome("SUB_insert_failure", -0.5)

        stats = DirectedMutator.get_mutation_stats()
        assert stats["total_attempts"] == 4
        assert stats["total_successes"] == 2

        # Check per-prefix stats
        by_type = stats["by_type"]
        assert "ADDI" in by_type
        assert "SUB" in by_type
        assert by_type["ADDI"]["attempts"] == 2
        assert by_type["ADDI"]["successes"] == 1
        assert by_type["SUB"]["attempts"] == 2
        assert by_type["SUB"]["successes"] == 1

    def test_success_rate_calculation(self):
        """Success rate is computed correctly per prefix."""
        for _ in range(7):
            DirectedMutator.record_mutation_outcome("XOR_foo_success", 1.0)
        for _ in range(3):
            DirectedMutator.record_mutation_outcome("XOR_bar_fail", -1.0)

        stats = DirectedMutator.get_mutation_stats()
        xor_stats = stats["by_type"]["XOR"]
        assert xor_stats["attempts"] == 10
        assert xor_stats["successes"] == 7
        assert xor_stats["success_rate"] == pytest.approx(0.7)

    def test_history_truncation_at_200(self):
        """History is capped at 200 entries."""
        for i in range(250):
            DirectedMutator.record_mutation_outcome(f"TYPE_{i}_x", float(i))

        assert len(DirectedMutator._mutation_history) == 200
        stats = DirectedMutator.get_mutation_stats()
        # recent_avg_delta uses last 50 entries (indices 200..249 mapped to actual values 200..249)
        # But the stored history is the last 200, so last 50 are 200..249
        expected_avg = sum(range(200, 250)) / 50.0
        assert stats["recent_avg_delta"] == pytest.approx(expected_avg)

    def test_stats_calculation_accuracy(self):
        """Stats accurately reflect recorded data."""
        DirectedMutator.record_mutation_outcome("A_x", 5.0)
        DirectedMutator.record_mutation_outcome("A_y", 3.0)
        DirectedMutator.record_mutation_outcome("B_x", -2.0)

        stats = DirectedMutator.get_mutation_stats()
        assert stats["total_attempts"] == 3
        assert stats["total_successes"] == 2
        avg = (5.0 + 3.0 + (-2.0)) / 3.0
        assert stats["recent_avg_delta"] == pytest.approx(avg)


# ═══════════════════════════════════════════════════════════════════════
# 3. TestDominanceReportFullCycle
# ═══════════════════════════════════════════════════════════════════════


class TestDominanceReportFullCycle:
    """Tests for StrategyArchive.get_dominance_report()."""

    def test_empty_archive(self):
        """Empty archive returns zeroed dominance report."""
        archive = StrategyArchive()
        report = archive.get_dominance_report()

        assert report["dominant_strategy"] is None
        assert report["dominance_ratio"] == 0.0
        assert report["is_monoculture"] is False
        assert report["recommendation"] == "balanced"
        assert report["all_strategy_rates"] == {}

    def test_single_strategy_100_percent(self):
        """Single strategy achieves 100% dominance."""
        archive = StrategyArchive()
        for _ in range(20):
            archive.record("RANDOM", level=0, success=True, generation=1)

        report = archive.get_dominance_report()

        assert report["dominant_strategy"] == "RANDOM"
        assert report["dominance_ratio"] == pytest.approx(1.0, abs=0.01)
        assert report["is_monoculture"] is True
        assert report["recommendation"] == "explore"

    def test_balanced_strategies(self):
        """Two strategies with equal success rates are balanced."""
        archive = StrategyArchive()
        # Record alternating successes for two strategies at same rate
        for i in range(40):
            strat = "RANDOM" if i % 2 == 0 else "TASK_AWARE"
            archive.record(strat, level=0, success=True, generation=1)

        report = archive.get_dominance_report()

        assert report["dominant_strategy"] in ("RANDOM", "TASK_AWARE")
        # Both strategies have same rate → dominance_ratio should be ~0.5
        assert report["dominance_ratio"] == pytest.approx(0.5, abs=0.01)
        assert report["is_monoculture"] is False
        assert report["recommendation"] == "balanced"
        assert len(report["all_strategy_rates"]) == 2

    def test_many_strategies(self):
        """Report handles many strategies without error."""
        archive = StrategyArchive()
        strat_names = [f"STRAT_{i}" for i in range(10)]
        for i in range(100):
            strat = strat_names[i % len(strat_names)]
            archive.record(strat, level=i % 5, success=(i % 3 != 0), generation=i)

        report = archive.get_dominance_report()

        assert report["dominant_strategy"] is not None
        assert report["dominance_ratio"] >= 0.0
        assert report["dominance_ratio"] <= 1.0
        assert len(report["all_strategy_rates"]) == 10

    def test_dominance_ratio_bounded(self):
        """Dominance ratio is always in [0, 1]."""
        archive = StrategyArchive()
        for _ in range(50):
            archive.record("A", level=0, success=True, generation=1)
        for _ in range(10):
            archive.record("B", level=0, success=True, generation=1)

        report = archive.get_dominance_report()
        assert 0.0 <= report["dominance_ratio"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# 4. TestRecoveryConfidenceFullCycle
# ═══════════════════════════════════════════════════════════════════════


class TestRecoveryConfidenceFullCycle:
    """Tests for MetaController.get_recovery_confidence()."""

    @pytest.fixture
    def mc(self, tmp_path):
        """Create a fresh MetaController for each test."""
        return MetaController(checkpoint_dir=str(tmp_path))

    def test_no_recovery(self):
        """When not recovering, confidence report shows inactive state."""
        mc = MetaController(checkpoint_dir="/tmp/mc_test_no_recovery")
        mc.recovery_mode = False

        report = mc.get_recovery_confidence()

        assert report["is_recovering"] is False
        assert report["cycles_in_recovery"] == 0
        assert report["confidence"] == pytest.approx(0.3)
        assert report["should_exit_recovery"] is False

    def test_active_recovery(self):
        """Active recovery mode is detected."""
        mc = MetaController(checkpoint_dir="/tmp/mc_test_active_recovery")
        mc.recovery_mode = True

        report = mc.get_recovery_confidence()

        assert report["is_recovering"] is True

    def test_exit_condition_at_5_cycles(self):
        """Exit condition triggers at 5+ cycles with stable/declining trend."""
        mc = MetaController(checkpoint_dir="/tmp/mc_test_exit")
        mc.recovery_mode = True

        # Add 5 recovery-related tuning log entries
        for cycle in range(1, 6):
            mc._tuning_log.append(
                TuningAction(
                    domain="recovery",
                    param="retrain_lr",
                    old_value=1e-4,
                    new_value=5e-5,
                    reason=f"recovery stagnation cycle {cycle}",
                    cycle=cycle,
                )
            )
            # Add corresponding telemetry with stable fitness
            mc.telemetry.append(
                TelemetrySnapshot(
                    cycle=cycle,
                    fitness=10.0,  # stable
                    level=5,
                    champion_words=20,
                    val_loss=1.0,
                    diversity=0.3,
                    novelty_size=10,
                    qd_occupied=5,
                    qd_total=100,
                    traces_count=50,
                    retrained=True,
                    generations_completed=100,
                )
            )

        report = mc.get_recovery_confidence()

        assert report["is_recovering"] is True
        assert report["cycles_in_recovery"] >= 5
        assert report["should_exit_recovery"] is True

    def test_confidence_calculation_improving(self):
        """Improving fitness during recovery yields high confidence."""
        mc = MetaController(checkpoint_dir="/tmp/mc_test_conf")
        mc.recovery_mode = True

        # Add recovery tuning entries
        for cycle in range(1, 5):
            mc._tuning_log.append(
                TuningAction(
                    domain="recovery",
                    param="retrain_lr",
                    old_value=1e-4,
                    new_value=5e-5,
                    reason=f"recovery cycle {cycle}",
                    cycle=cycle,
                )
            )
            # Improving fitness
            mc.telemetry.append(
                TelemetrySnapshot(
                    cycle=cycle,
                    fitness=10.0 + cycle * 5,  # improving
                    level=5,
                    champion_words=20,
                    val_loss=1.0 - cycle * 0.1,
                    diversity=0.3,
                    novelty_size=10,
                    qd_occupied=5,
                    qd_total=100,
                    traces_count=50,
                    retrained=True,
                    generations_completed=100,
                )
            )

        report = mc.get_recovery_confidence()

        assert report["is_recovering"] is True
        assert report["fitness_trend_during_recovery"] == "improving"
        assert report["confidence"] == pytest.approx(0.8)
        assert report["should_exit_recovery"] is False

    def test_no_exit_when_improving(self):
        """Exit condition does NOT trigger when fitness is improving."""
        mc = MetaController(checkpoint_dir="/tmp/mc_test_no_exit")
        mc.recovery_mode = True

        for cycle in range(1, 7):
            mc._tuning_log.append(
                TuningAction(
                    domain="recovery",
                    param="lr",
                    old_value=1e-4,
                    new_value=5e-5,
                    reason=f"recovery cycle {cycle}",
                    cycle=cycle,
                )
            )
            mc.telemetry.append(
                TelemetrySnapshot(
                    cycle=cycle,
                    fitness=10.0 + cycle * 10,  # strongly improving
                    level=5 + cycle,
                    champion_words=20,
                    val_loss=1.0,
                    diversity=0.3,
                    novelty_size=10,
                    qd_occupied=5,
                    qd_total=100,
                    traces_count=50,
                    retrained=True,
                    generations_completed=100,
                )
            )

        report = mc.get_recovery_confidence()

        assert report["should_exit_recovery"] is False


# ═══════════════════════════════════════════════════════════════════════
# 5. TestQualityPruneFullCycle
# ═══════════════════════════════════════════════════════════════════════


class TestQualityPruneFullCycle:
    """Tests for ChampionArchive.quality_weighted_prune()."""

    # Module-level counter for unique programs
    _champion_counter = 0

    @pytest.fixture(autouse=True)
    def _reset_champion_counter(self):
        """Reset the champion counter before each test."""
        TestQualityPruneFullCycle._champion_counter = 0
        yield

    @pytest.fixture
    def archive(self, tmp_path):
        """Create a fresh ChampionArchive with small max_size for easy testing."""
        path = str(tmp_path / "test_archive.jsonl")
        return ChampionArchive(path=path, max_size=10, stale_threshold_generations=50)

    def _add_champion(self, archive, fitness, cycle, level=7):
        """Helper to add a unique champion to the archive."""
        # Generate unique program to avoid hash collisions.
        # Use a class-level counter so every call produces a different program.
        TestQualityPruneFullCycle._champion_counter += 1
        nonce = TestQualityPruneFullCycle._champion_counter
        program = [0x00100073]  # EBREAK as base
        # Add nonce-proportional padding + fitness-proportional data
        for i in range(max(1, (int(fitness) + nonce) % 20)):
            program.append(((int(fitness * 1000) + nonce * 100 + i) & 0xFFFFFFFF))
        program.append(0x00100073)  # EBREAK at end
        archive.add(program, level=level, fitness=fitness, cycle=cycle)

    def test_pruning_returns_zero_when_under_capacity(self):
        """No pruning needed when under max_size."""
        archive = ChampionArchive(
            path="/tmp/test_prune_under.jsonl",
            max_size=100,
        )
        for i in range(5):
            self._add_champion(archive, fitness=float(100 + i), cycle=1)

        removed = archive.quality_weighted_prune()
        assert removed == 0

    def test_pruning_removes_low_quality(self):
        """Pruning removes low-fitness entries when over capacity."""
        archive = ChampionArchive(
            path="/tmp/test_prune_low.jsonl",
            max_size=100,  # large to prevent auto-evict
            stale_threshold_generations=100,
        )
        # Add 10 entries at level 7+ (non-foundational) to force eviction
        for i in range(10):
            self._add_champion(archive, fitness=float(10 + i), cycle=1, level=7)

        initial_size = archive.size
        assert initial_size == 10

        # Now shrink max_size so prune triggers
        archive.max_size = 5
        removed = archive.quality_weighted_prune()

        assert removed > 0
        assert archive.size < initial_size

    def test_pruning_preserves_high_quality(self):
        """High-fitness entries survive pruning better than low-fitness ones."""
        archive = ChampionArchive(
            path="/tmp/test_prune_preserve.jsonl",
            max_size=5,
            stale_threshold_generations=100,
        )
        # Add 10 entries at level 7+
        for i in range(10):
            self._add_champion(archive, fitness=float(10 + i * 10), cycle=1, level=7)

        archive.quality_weighted_prune()

        # Check that remaining entries have decent fitness
        remaining_fitnesses = [e.fitness for e in archive.entries]
        # At least some high-fitness entries should survive
        assert max(remaining_fitnesses) >= 50.0

    def test_pruning_with_all_equal_fitness(self):
        """Pruning works correctly when all fitnesses are equal."""
        archive = ChampionArchive(
            path="/tmp/test_prune_equal.jsonl",
            max_size=100,
            stale_threshold_generations=100,
        )
        for i in range(10):
            self._add_champion(archive, fitness=100.0, cycle=1, level=7)

        archive.max_size = 5
        removed = archive.quality_weighted_prune()

        assert removed > 0
        assert archive.size <= 10  # should be reduced

    def test_pruning_returns_correct_count(self):
        """Pruning returns the exact number of entries removed."""
        archive = ChampionArchive(
            path="/tmp/test_prune_count.jsonl",
            max_size=100,
            stale_threshold_generations=100,
        )
        for i in range(8):
            self._add_champion(archive, fitness=float(i), cycle=1, level=7)

        initial_size = archive.size
        archive.max_size = 3
        removed = archive.quality_weighted_prune()

        assert removed > 0
        assert archive.size == initial_size - removed


# ═══════════════════════════════════════════════════════════════════════
# 6. TestEndToEndMethodAvailability
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndMethodAvailability:
    """Verify all new methods exist on their respective classes."""

    def test_meta_controller_has_get_recovery_confidence(self):
        """MetaController has get_recovery_confidence method."""
        assert hasattr(MetaController, "get_recovery_confidence")
        assert callable(MetaController.get_recovery_confidence)

    def test_strategy_archive_has_get_dominance_report(self):
        """StrategyArchive has get_dominance_report method."""
        assert hasattr(StrategyArchive, "get_dominance_report")
        assert callable(StrategyArchive.get_dominance_report)

    def test_champion_archive_has_quality_weighted_prune(self):
        """ChampionArchive has quality_weighted_prune method."""
        assert hasattr(ChampionArchive, "quality_weighted_prune")
        assert callable(ChampionArchive.quality_weighted_prune)

    def test_novelty_has_manage_capacity(self):
        """SparseArchive (novelty) has manage_capacity method."""
        assert hasattr(SparseArchive, "manage_capacity")
        assert callable(SparseArchive.manage_capacity)

    def test_qd_archive_has_get_improvement_velocity(self):
        """MAPElitesArchive (qd_archive) has get_improvement_velocity method."""
        assert hasattr(MAPElitesArchive, "get_improvement_velocity")
        assert callable(MAPElitesArchive.get_improvement_velocity)

    def test_mutation_proposer_has_get_mutation_stats(self):
        """DirectedMutator has get_mutation_stats method."""
        assert hasattr(DirectedMutator, "get_mutation_stats")
        assert callable(DirectedMutator.get_mutation_stats)

    def test_metrics_has_detect_anomalies(self):
        """MetricsCollector has detect_anomalies method."""
        assert hasattr(MetricsCollector, "detect_anomalies")
        assert callable(MetricsCollector.detect_anomalies)

    def test_data_rotation_has_get_disk_usage(self):
        """data_rotation module has get_disk_usage function."""
        from data_rotation import get_disk_usage
        assert callable(get_disk_usage)

    def test_progression_tracker_has_get_plateau_analysis(self):
        """ProgressionTracker has get_plateau_analysis method."""
        assert hasattr(ProgressionTracker, "get_plateau_analysis")
        assert callable(ProgressionTracker.get_plateau_analysis)

    def test_config_has_validate_config(self):
        """EvolutionSection has validate_config method."""
        assert hasattr(EvolutionSection, "validate_config")
        assert callable(EvolutionSection.validate_config)

    def test_config_has_get_config_summary(self):
        """EvolutionSection has get_config_summary method."""
        assert hasattr(EvolutionSection, "get_config_summary")
        assert callable(EvolutionSection.get_config_summary)

    def test_config_class_has_validate(self):
        """Config class has _validate method."""
        assert hasattr(Config, "_validate")
        assert callable(Config._validate)
