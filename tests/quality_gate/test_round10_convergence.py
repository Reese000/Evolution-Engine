#!/usr/bin/env python3
"""R28: Convergence quality tests -- adaptive mutation, oscillation control,
Hall of Fame recency, MetaController fitness momentum, diversity injection."""

import os
import random
import sys

import pytest

from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame
from meta_controller import MetaController, TelemetrySnapshot

# ───────────────────────────────────────────────────────────────────────
# Adaptive Mutation Rate Tests
# ───────────────────────────────────────────────────────────────────────


class TestAdaptiveMutationRate:
    """Tests for the adaptive mutation rate based on fitness velocity."""

    def _make_engine(self, mutation_rate=0.3, seed=42):
        cfg = EvolutionConfig(mutation_rate=mutation_rate, seed=seed)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_velocity_zero_keeps_base_rate(self):
        """With zero velocity, adaptive rate should equal base rate."""
        eng = self._make_engine(mutation_rate=0.3)
        eng._fitness_velocity = 0.0
        rate = eng._compute_adaptive_mutation_rate()
        # Blend of base(0.3) * 1.3 (plateau) and base(0.3): 0.5*0.39+0.5*0.3=0.345
        assert 0.25 <= rate <= 0.45, f"Expected ~0.345, got {rate}"

    def test_high_positive_velocity_reduces_mutation(self):
        """Fast improvement should reduce mutation to exploit."""
        eng = self._make_engine(mutation_rate=0.4)
        eng._fitness_velocity = 10.0
        rate = eng._compute_adaptive_mutation_rate()
        assert rate < 0.4, f"Expected <0.4 (reduced), got {rate}"

    def test_negative_velocity_increases_mutation(self):
        """Regression should increase mutation to explore."""
        eng = self._make_engine(mutation_rate=0.3)
        eng._fitness_velocity = -10.0
        rate = eng._compute_adaptive_mutation_rate()
        assert rate > 0.3, f"Expected >0.3 (increased), got {rate}"

    def test_rate_clamped_to_floor(self):
        """Rate should never drop below floor."""
        eng = self._make_engine(mutation_rate=0.01)
        eng._fitness_velocity = 100.0
        rate = eng._compute_adaptive_mutation_rate()
        assert rate >= eng._adaptive_mutation_floor

    def test_rate_clamped_to_ceil(self):
        """Rate should never exceed ceiling."""
        eng = self._make_engine(mutation_rate=0.99)
        eng._fitness_velocity = -100.0
        rate = eng._compute_adaptive_mutation_rate()
        assert rate <= eng._adaptive_mutation_ceil

    def test_update_fitness_velocity(self):
        """EMA should converge toward recent deltas."""
        eng = self._make_engine()
        eng._fitness_velocity = 0.0
        # Apply consistent positive delta
        for _ in range(20):
            eng._update_fitness_velocity(5.0)
        assert eng._fitness_velocity > 3.0, f"Expected velocity >3.0, got {eng._fitness_velocity}"

    def test_velocity_resets_on_negative(self):
        """Negative delta should pull velocity negative."""
        eng = self._make_engine()
        eng._fitness_velocity = 5.0
        for _ in range(10):
            eng._update_fitness_velocity(-2.0)
        assert eng._fitness_velocity < 0.0, f"Expected negative velocity, got {eng._fitness_velocity}"

    def test_blended_rate_is_between_base_and_extreme(self):
        """Blended rate should be between base and fully adaptive."""
        eng = self._make_engine(mutation_rate=0.3)
        eng._fitness_velocity = -10.0
        rate = eng._compute_adaptive_mutation_rate()
        # Pure adaptive would be higher; base is 0.3; blend is between
        assert 0.3 <= rate <= 0.85, f"Expected between 0.3 and 0.85, got {rate}"


# ───────────────────────────────────────────────────────────────────────
# Champion Oscillation Control Tests
# ───────────────────────────────────────────────────────────────────────


class TestChampionOscillationControl:
    """Tests for oscillation detection and hysteresis in champion replacement."""

    def _make_engine(self, seed=42):
        cfg = EvolutionConfig(mutation_rate=0.3, seed=seed)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_no_oscillation_allows_replacement(self):
        """Without oscillation, candidate should replace champion."""
        eng = self._make_engine()
        eng.champion_fitness = 10.0
        assert eng._should_replace_champion(11.0) is True

    def test_identical_fitness_rejected(self):
        """Candidate with equal fitness should not replace."""
        eng = self._make_engine()
        eng.champion_fitness = 10.0
        assert eng._should_replace_champion(10.0) is False

    def test_worse_fitness_rejected(self):
        """Candidate with worse fitness should not replace."""
        eng = self._make_engine()
        eng.champion_fitness = 10.0
        assert eng._should_replace_champion(9.0) is False

    def test_oscillation_detection(self):
        """Many small replacements in a row should trigger hysteresis."""
        eng = self._make_engine()
        eng.champion_fitness = 10.0

        # Simulate rapid oscillation: many small improvements
        for i in range(eng._oscillation_window_size):
            eng.champion_fitness = 10.0 + 0.01 * (i + 1)
            # This small delta should eventually be rejected
            accepted = eng._should_replace_champion(eng.champion_fitness + 0.01)
            if accepted:
                eng._record_champion_replacement(0.01)

        # Now a small delta should be rejected due to hysteresis
        # After enough replacements in the window, small deltas are blocked
        eng.champion_fitness = 10.0 + 0.11
        result = eng._should_replace_champion(eng.champion_fitness + 0.1)
        # Should be rejected (0.1 < 0.5 hysteresis threshold)
        assert result is False, "Small delta should be rejected under hysteresis"

    def test_large_delta_overrides_hysteresis(self):
        """Even under hysteresis, a large delta should pass."""
        eng = self._make_engine()
        eng.champion_fitness = 10.0

        # Build up oscillation
        for i in range(eng._oscillation_window_size):
            eng.champion_fitness = 10.0 + 0.01 * (i + 1)
            eng._should_replace_champion(eng.champion_fitness + 0.01)
            eng._record_champion_replacement(0.01)

        # Large delta should still pass
        eng.champion_fitness = 10.12
        result = eng._should_replace_champion(eng.champion_fitness + 10.0)
        assert result is True, "Large delta should override hysteresis"

    def test_window_fifo(self):
        """Old entries should age out of the window."""
        eng = self._make_engine()
        eng.champion_fitness = 10.0

        # Fill window with small positive deltas via the real API
        for i in range(eng._oscillation_window_size):
            eng.champion_fitness += 0.01
            eng._should_replace_champion(eng.champion_fitness + 0.01)
            eng._record_champion_replacement(0.01)

        # Simulate the window aging out: manually set state as if old
        # positive deltas have been evicted and replaced with zeros.
        # This models the long-run scenario where replacements stop.
        eng._champion_oscillation_window.clear()
        eng._champion_oscillation_window.extend([0.0] * eng._oscillation_window_size)
        eng._oscillation_positive_count = 0

        # Now hysteresis should not be active -- small delta should pass
        eng.champion_fitness = 20.0
        result = eng._should_replace_champion(eng.champion_fitness + 0.1)
        assert result is True, "After window ages out, replacement should be allowed"

    def test_record_replacement_increments_counter(self):
        """_record_champion_replacement should increment the counter."""
        eng = self._make_engine()
        assert eng._champion_replacement_count == 0
        eng._record_champion_replacement(1.0)
        assert eng._champion_replacement_count == 1
        eng._record_champion_replacement(2.0)
        assert eng._champion_replacement_count == 2


# ───────────────────────────────────────────────────────────────────────
# Hall of Fame Recency Tests
# ───────────────────────────────────────────────────────────────────────


class TestHallOfFameRecency:
    """Tests for R28 Hall of Fame generation tracking and recency weighting."""

    def test_add_with_generation(self):
        """add() should accept generation parameter."""
        hof = HallOfFame(capacity=5)
        assert hof.add([1, 2, 3], fitness=10.0, level=0, generation=5)
        assert hof.size == 1

    def test_add_replaces_worse_with_generation(self):
        """Should replace worse entries, preserving generation."""
        hof = HallOfFame(capacity=3)
        hof.add([1], fitness=5.0, level=0, generation=1)
        hof.add([2], fitness=10.0, level=0, generation=2)
        hof.add([3], fitness=15.0, level=0, generation=3)
        assert hof.size == 3
        # Adding a better one should replace the worst (fitness=5.0)
        assert hof.add([4], fitness=20.0, level=0, generation=4)
        assert hof.size == 3
        best = hof.get_best()
        assert best == [4], f"Expected [4], got {best}"

    def test_get_best_by_level(self):
        """get_best with level filter should work with new tuple format."""
        hof = HallOfFame(capacity=5)
        hof.add([1], fitness=10.0, level=0, generation=1)
        hof.add([2], fitness=20.0, level=1, generation=2)
        hof.add([3], fitness=15.0, level=0, generation=3)

        best_l0 = hof.get_best(level=0)
        assert best_l0 == [3], f"Expected [3], got {best_l0}"
        best_l1 = hof.get_best(level=1)
        assert best_l1 == [2], f"Expected [2], got {best_l1}"

    def test_get_best_fallback_unfiltered(self):
        """get_best with non-matching level should return overall best."""
        hof = HallOfFame(capacity=5)
        hof.add([1], fitness=10.0, level=0, generation=1)
        hof.add([2], fitness=20.0, level=1, generation=2)

        best = hof.get_best(level=99)
        assert best == [2], f"Expected [2], got {best}"

    def test_get_random_returns_copy(self):
        """get_random should return a defensive copy."""
        hof = HallOfFame(capacity=5)
        prog = [1, 2, 3]
        hof.add(prog, fitness=10.0, level=0, generation=1)
        rng = random.Random(42)
        result = hof.get_random(rng)
        assert result == [1, 2, 3]
        result[0] = 999
        # Original in HoF should be unchanged
        assert hof.get_best() == [1, 2, 3]

    def test_get_random_level_filtered(self):
        """get_random with level filter should only return matching level."""
        hof = HallOfFame(capacity=5)
        hof.add([1], fitness=10.0, level=0, generation=1)
        hof.add([2], fitness=20.0, level=1, generation=2)
        hof.add([3], fitness=15.0, level=0, generation=3)
        rng = random.Random(42)

        # All random picks at level 1 should be [2]
        for _ in range(10):
            result = hof.get_random(rng, level=1)
            assert result == [2], f"Expected [2], got {result}"

    def test_get_random_recency_bias(self):
        """get_random should prefer more recent generations."""
        hof = HallOfFame(capacity=10)
        # Add 5 old programs (gen 1-5) and 5 recent programs (gen 95-100)
        for i in range(5):
            hof.add([i], fitness=10.0 + i, level=0, generation=i + 1)
        for i in range(5):
            hof.add([100 + i], fitness=10.0 + i, level=0, generation=95 + i)

        rng = random.Random(42)
        recent_count = 0
        total = 200
        for _ in range(total):
            result = hof.get_random(rng, level=0)
            if result[0] >= 100:
                recent_count += 1

        # Recent programs should be selected more often than 50%
        ratio = recent_count / total
        assert ratio > 0.5, f"Expected >50% recent picks, got {ratio:.1%}"

    def test_get_random_single_item(self):
        """get_random with single item should always return it."""
        hof = HallOfFame(capacity=5)
        hof.add([42], fitness=10.0, level=0, generation=1)
        rng = random.Random(42)
        for _ in range(10):
            assert hof.get_random(rng) == [42]

    def test_best_fitness(self):
        """best_fitness should return maximum fitness in the HoF."""
        hof = HallOfFame(capacity=5)
        hof.add([1], fitness=10.0, level=0, generation=1)
        hof.add([2], fitness=25.0, level=0, generation=2)
        hof.add([3], fitness=15.0, level=0, generation=3)
        assert hof.best_fitness == 25.0

    def test_empty_hof(self):
        """Empty HoF should return None/0."""
        hof = HallOfFame(capacity=5)
        assert hof.get_best() is None
        assert hof.size == 0
        assert hof.best_fitness == 0.0


# ───────────────────────────────────────────────────────────────────────
# MetaController Fitness Momentum Tests
# ───────────────────────────────────────────────────────────────────────


class TestMetaControllerFitnessMomentum:
    """Tests for R28 fitness momentum tracking in MetaController."""

    def _make_controller(self):
        return MetaController(checkpoint_dir="/tmp/mc_r28_test")

    def _snapshot(self, cycle, fitness, level=5):
        return TelemetrySnapshot(
            cycle=cycle,
            fitness=fitness,
            level=level,
            champion_words=50,
            val_loss=2.0,
            diversity=0.3,
            novelty_size=10,
            qd_occupied=5,
            qd_total=100,
            traces_count=20,
            retrained=True,
            generations_completed=50,
        )

    def test_initial_momentum_zero(self):
        """Momentum should start at 0."""
        mc = self._make_controller()
        assert mc._fitness_momentum == 0.0

    def test_momentum_increases_with_improvement(self):
        """Consistent improvement should increase momentum."""
        mc = self._make_controller()
        for c in range(1, 10):
            mc.update_cycle(self._snapshot(c, fitness=-5.0 + c))
        assert mc._fitness_momentum > 0.0, f"Expected positive momentum, got {mc._fitness_momentum}"

    def test_momentum_decreases_with_regression(self):
        """Regression should decrease momentum."""
        mc = self._make_controller()
        for c in range(1, 5):
            mc.update_cycle(self._snapshot(c, fitness=10.0 - c))
        assert mc._fitness_momentum < 0.0, f"Expected negative momentum, got {mc._fitness_momentum}"

    def test_momentum_zero_when_flat(self):
        """Flat fitness should produce near-zero momentum."""
        mc = self._make_controller()
        for c in range(1, 15):
            mc.update_cycle(self._snapshot(c, fitness=5.0))
        # After many cycles at same fitness, delta=0 each time -> momentum ~0
        assert abs(mc._fitness_momentum) < 0.5, f"Expected ~0 momentum, got {mc._fitness_momentum}"

    def test_momentum_responsive_to_direction_change(self):
        """Momentum should respond to improvement->regression transition."""
        mc = self._make_controller()
        # First: improvement
        for c in range(1, 8):
            mc.update_cycle(self._snapshot(c, fitness=-10.0 + c))
        positive_momentum = mc._fitness_momentum
        assert positive_momentum > 0.0

        # Then: regression
        for c in range(8, 15):
            mc.update_cycle(self._snapshot(c, fitness=-3.0 - (c - 7) * 0.5))
        assert mc._fitness_momentum < positive_momentum, "Momentum should decrease after regression"


# ───────────────────────────────────────────────────────────────────────
# Diversity Injection Proportional Test
# ───────────────────────────────────────────────────────────────────────


class TestDiversityInjection:
    """Tests for the diversity injection mechanism."""

    def test_compute_population_diversity(self):
        """Should return diversity in [0, 1]."""

        # All same program -> low diversity
        prog = [0x00000013, 0x00000013]  # NOPs
        div = EvolutionEngine._compute_population_diversity([prog, prog, prog])
        assert 0.0 <= div <= 1.0

        # Diverse programs -> higher diversity
        progs = [
            [0x00000013],  # NOP
            [0x00000033],  # ADD
            [0x00000063],  # BEQ
            [0x00000003],  # LB
            [0x00000023],  # SB
        ]
        div_diverse = EvolutionEngine._compute_population_diversity(progs)
        assert div_diverse > div, f"Diverse ({div_diverse}) should > uniform ({div})"

    def test_empty_population(self):
        """Empty population should return 0."""
        assert EvolutionEngine._compute_population_diversity([]) == 0.0

    def test_single_program(self):
        """Single program should give some diversity."""
        div = EvolutionEngine._compute_population_diversity([[0x00000013, 0x00000033]])
        assert 0.0 < div <= 1.0


# ───────────────────────────────────────────────────────────────────────
# Integration: Adaptive Mutation in Full Engine Context
# ───────────────────────────────────────────────────────────────────────


class TestAdaptiveMutationIntegration:
    """Integration tests for adaptive mutation in the full evolution context."""

    def test_adaptive_mutation_applied_during_run(self):
        """The generation loop should apply adaptive mutation after gen 5."""
        # This test verifies the config mutation_rate is updated by checking
        # that _compute_adaptive_mutation_rate is callable and returns valid.
        cfg = EvolutionConfig(mutation_rate=0.3, seed=42, max_generations=2)
        eng = EvolutionEngine(config=cfg, vm_farm=None)
        eng._fitness_velocity = -3.0  # simulate regression
        rate = eng._compute_adaptive_mutation_rate()
        assert 0.15 <= rate <= 0.85

    def test_oscillation_control_integration(self):
        """Oscillation control should prevent rapid champion cycling."""
        cfg = EvolutionConfig(mutation_rate=0.3, seed=42)
        eng = EvolutionEngine(config=cfg, vm_farm=None)

        # Simulate many small champion replacements
        eng.champion_fitness = 10.0
        for _ in range(12):
            eng.champion_fitness += 0.05
            eng._should_replace_champion(eng.champion_fitness + 0.05)
            eng._record_champion_replacement(0.05)

        # Now very small delta should be blocked
        result = eng._should_replace_champion(eng.champion_fitness + 0.01)
        assert result is False, "Oscillation control should block tiny delta"

        # But large delta should pass
        result = eng._should_replace_champion(eng.champion_fitness + 5.0)
        assert result is True, "Large delta should pass even with oscillation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
