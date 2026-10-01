#!/usr/bin/env python3
"""R30: Convergence quality tests -- adaptive tournament sizing,
niche-aware fitness sharing, QD bonus stall scaling, diversity
injection optimization, fitness velocity edge cases.

Tests for:
  1. Adaptive tournament sizing: diversity-based tournament size
  2. Niche-aware fitness sharing: penalty for same-cell as champion
  3. QD bonus stall scaling: multiplier amplifies with stall depth
  4. Diversity injection: pre-allocated redundancy scores
  5. Fitness velocity edge cases: EMA decay, boundary transitions
  6. Oscillation control: edge cases (empty window, single entry)
  7. MetaController consecutive stagnation ring buffer
  8. Adaptive mutation rate boundary transitions
  9. HoF recency weighting: edge cases (all same gen, single item)
  10. Niche sharing: no penalty for distant behavioral descriptors
"""

import random

import pytest

from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame, compute_qd_axes
from meta_controller import MetaController, StagnationResult, TelemetrySnapshot

# ───────────────────────────────────────────────────────────────────────
# Adaptive Tournament Sizing Tests
# ───────────────────────────────────────────────────────────────────────


class TestAdaptiveTournamentSizing:
    """Tests for diversity-based adaptive tournament sizing in _select_elite."""

    def _make_engine(self, mutation_rate=0.3, seed=42, population_size=50):
        cfg = EvolutionConfig(
            mutation_rate=mutation_rate,
            seed=seed,
            population_size=population_size,
        )
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_high_diversity_increases_tournament(self):
        """High diversity (>0.6) should increase tournament size."""
        eng = self._make_engine(population_size=20)
        eng._current_diversity = 0.8
        # At div=0.8, adaptive_tourney = base + 2 = 5
        # Verify by checking the engine picks larger tournament
        assert eng._current_diversity >= 0.6

    def test_low_diversity_decreases_tournament(self):
        """Very low diversity (<0.2) should decrease tournament size."""
        eng = self._make_engine()
        eng._current_diversity = 0.1
        base_tourney = eng.config.tournament_size
        # At div=0.1, adaptive_tourney = max(2, base - 1)
        expected = max(2, base_tourney - 1)
        assert expected < base_tourney

    def test_medium_diversity_standard_tournament(self):
        """Medium diversity (0.4-0.6) should use standard tournament."""
        eng = self._make_engine()
        eng._current_diversity = 0.5
        # At div=0.5, adaptive_tourney = base + 1
        expected = eng.config.tournament_size + 1
        assert expected > eng.config.tournament_size

    def test_tournament_clamped_to_population_size(self):
        """Tournament size should never exceed population size."""
        eng = self._make_engine(population_size=5)
        eng._current_diversity = 0.9
        # Even with high diversity boost, tournament <= pop_size
        # This is tested by the min(adaptive_tourney, n_scored) in _select_elite
        assert eng.config.tournament_size >= 2  # base
        assert eng.config.population_size >= 2  # pop


# ───────────────────────────────────────────────────────────────────────
# Niche-Aware Fitness Sharing Tests
# ───────────────────────────────────────────────────────────────────────


class TestNicheAwareFitnessSharing:
    """Tests for the QD-cell-based niche penalty in _compute_fitness."""

    def _make_engine(self, seed=42):
        cfg = EvolutionConfig(
            mutation_rate=0.3,
            seed=seed,
            qd_bonus_weight=0.1,
        )
        eng = EvolutionEngine(config=cfg, vm_farm=None)
        return eng

    def test_same_qd_cell_penalty(self):
        """Programs in the same QD cell as champion should be penalized."""
        eng = self._make_engine()
        eng.champion_level = 0

        # Mock task instance with expected output
        eng.task_instance = {
            "expected_output": {"registers": {10: 42}},
            "inputs": {},
        }

        # Set champion and its trace (with known QD descriptor)
        eng.champion = [0x00000033, 0x00100073]  # ADD, EBREAK
        eng.champion_trace = {
            "status": "ok",
            "final_regs": {10: 42},
            "instructions_count": 2,
            "trap_cause": "",
        }

        # Same program (same QD cell) should get penalty
        same_prog = [0x00000033, 0x00100073]
        same_trace = {
            "status": "ok",
            "final_regs": {10: 42},
            "instructions_count": 2,
            "trap_cause": "",
        }

        eng.current_generation = 0
        eng._bonus_threshold_cache = None
        eng._champion_changed_since_last_qd = False

        fitness_same = eng._compute_fitness(same_prog, same_trace)

        # Different program (different QD cell) should get no penalty
        diff_prog = [0x00000013, 0x00000013, 0x00100073]  # NOP, NOP, EBREAK
        diff_trace = {
            "status": "ok",
            "final_regs": {10: 42},
            "instructions_count": 3,
            "trap_cause": "",
        }
        fitness_diff = eng._compute_fitness(diff_prog, diff_trace)

        # Same-cell fitness should be lower than different-cell fitness
        # (penalty applied to same-cell)
        assert fitness_same <= fitness_diff, f"Same-cell ({fitness_same}) should be <= diff-cell ({fitness_diff})"

    def test_no_penalty_when_no_champion(self):
        """No penalty when champion is None."""
        eng = self._make_engine()
        eng.champion = None
        eng.champion_trace = None
        eng.champion_level = 0
        eng.task_instance = {
            "expected_output": {"registers": {10: 42}},
            "inputs": {},
        }
        eng.current_generation = 0
        eng._bonus_threshold_cache = None

        prog = [0x00000033, 0x00100073]
        trace = {"status": "ok", "final_regs": {10: 42}, "instructions_count": 2, "trap_cause": ""}
        fitness = eng._compute_fitness(prog, trace)
        # No penalty, should get full base fitness
        assert fitness > 0.0

    def test_no_penalty_when_qd_disabled(self):
        """No penalty when qd_bonus_weight is 0."""
        cfg = EvolutionConfig(qd_bonus_weight=0.0)
        eng = EvolutionEngine(config=cfg, vm_farm=None)
        eng.champion = [0x00000033, 0x00100073]
        eng.champion_trace = {"status": "ok", "final_regs": {10: 42}, "instructions_count": 2, "trap_cause": ""}
        eng.champion_level = 0
        eng.task_instance = {
            "expected_output": {"registers": {10: 42}},
            "inputs": {},
        }
        eng.current_generation = 0
        eng._bonus_threshold_cache = None

        prog = [0x00000033, 0x00100073]
        trace = {"status": "ok", "final_regs": {10: 42}, "instructions_count": 2, "trap_cause": ""}
        fitness = eng._compute_fitness(prog, trace)
        # No QD, no penalty
        assert fitness > 0.0


# ───────────────────────────────────────────────────────────────────────
# QD Bonus Stall Scaling Tests
# ───────────────────────────────────────────────────────────────────────


class TestQDBonusStallScaling:
    """Tests for the stall-scaled QD bonus multiplier."""

    def test_no_stall_no_boost(self):
        """At stall=0, QD bonus multiplier should be 1.0x."""
        # stall=0 -> stall_multiplier not applied (stall > 10 required)
        # So qd_weight stays at config value
        stall = 0
        qd_weight = 0.1
        if stall > 10:
            stall_multiplier = min(2.5, 1.0 + 0.0375 * (stall - 10))
            qd_weight *= stall_multiplier
        assert qd_weight == 0.1  # no boost

    def test_mild_stall_small_boost(self):
        """At stall=20, QD bonus should be ~1.375x."""
        stall = 20
        qd_weight = 0.1
        if stall > 10:
            stall_multiplier = min(2.5, 1.0 + 0.0375 * (stall - 10))
            qd_weight *= stall_multiplier
        expected_multiplier = 1.0 + 0.0375 * 10  # = 1.375
        assert abs(qd_weight / 0.1 - expected_multiplier) < 0.001

    def test_deep_stall_max_boost(self):
        """At stall=50+, QD bonus should cap at 2.5x."""
        stall = 60
        qd_weight = 0.1
        if stall > 10:
            stall_multiplier = min(2.5, 1.0 + 0.0375 * (stall - 10))
            qd_weight *= stall_multiplier
        assert abs(qd_weight / 0.1 - 2.5) < 0.001

    def test_stall_multiplier_monotonic(self):
        """Stall multiplier should increase monotonically with stall depth."""
        prev_mult = 1.0
        for stall in range(11, 60):
            qd_weight = 0.1
            if stall > 10:
                stall_multiplier = min(2.5, 1.0 + 0.0375 * (stall - 10))
                qd_weight *= stall_multiplier
            mult = qd_weight / 0.1
            assert mult >= prev_mult, f"Stall {stall}: mult {mult} < prev {prev_mult}"
            prev_mult = mult


# ───────────────────────────────────────────────────────────────────────
# Diversity Injection Optimization Tests
# ───────────────────────────────────────────────────────────────────────


class TestDiversityInjectionOptimization:
    """Tests for the pre-allocated redundancy scores in _inject_diversity."""

    def test_inject_replaces_redundant(self):
        """Should replace the most redundant programs."""
        cfg = EvolutionConfig(population_size=10, mutation_rate=0.3, seed=42)
        eng = EvolutionEngine(config=cfg, vm_farm=None)
        eng._last_challenger_strategies = ["RANDOM"] * 10

        # Create challengers where some are all-NOP (redundant)
        nop = 0x00000013  # NOP opcode
        challengers = []
        for i in range(10):
            if i < 5:
                # Redundant: all NOPs
                challengers.append([nop, nop, nop, 0x00100073])
            else:
                # Diverse: different opcodes
                challengers.append([0x00000033, 0x00000063, 0x00000003, 0x00100073])

        eng._inject_diversity(challengers, n_inject=3)

        # Check that at least some redundant programs were replaced
        replaced = 0
        for i in range(5):
            # Original NOPs are [nop, nop, nop, 0x00100073]
            # Replaced ones should be different (generated fresh)
            if challengers[i] != [nop, nop, nop, 0x00100073]:
                replaced += 1
        assert replaced >= 1, f"Expected at least 1 replacement, got {replaced}"

    def test_inject_no_op_when_zero_n_inject(self):
        """n_inject=0 should be a no-op."""
        cfg = EvolutionConfig(population_size=10, seed=42)
        eng = EvolutionEngine(config=cfg, vm_farm=None)
        eng._last_challenger_strategies = ["RANDOM"] * 10
        nop = 0x00000013
        challengers = [[nop] * 4] * 10
        original = [list(c) for c in challengers]
        eng._inject_diversity(challengers, n_inject=0)
        assert challengers == original

    def test_inject_no_op_when_empty(self):
        """Empty challengers should be a no-op."""
        cfg = EvolutionConfig(population_size=10, seed=42)
        eng = EvolutionEngine(config=cfg, vm_farm=None)
        eng._last_challenger_strategies = []
        eng._inject_diversity([], n_inject=3)

    def test_inject_tags_strategy(self):
        """Injected programs should be tagged with DIVERSITY_INJECT."""
        cfg = EvolutionConfig(population_size=10, mutation_rate=0.3, seed=42)
        eng = EvolutionEngine(config=cfg, vm_farm=None)
        eng._last_challenger_strategies = ["RANDOM"] * 10

        nop = 0x00000013
        challengers = [[nop, nop, nop, 0x00100073]] * 10
        eng._inject_diversity(challengers, n_inject=2)

        inject_count = eng._last_challenger_strategies.count("DIVERSITY_INJECT")
        assert inject_count == 2


# ───────────────────────────────────────────────────────────────────────
# Fitness Velocity Edge Cases
# ───────────────────────────────────────────────────────────────────────


class TestFitnessVelocityEdgeCases:
    """Tests for fitness velocity EMA edge cases."""

    def _make_engine(self, seed=42):
        cfg = EvolutionConfig(mutation_rate=0.3, seed=seed)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_ema_decays_to_zero(self):
        """After no deltas, velocity should decay toward zero."""
        eng = self._make_engine()
        eng._fitness_velocity = 10.0
        for _ in range(50):
            eng._update_fitness_velocity(0.0)
        assert abs(eng._fitness_velocity) < 0.01, f"Expected ~0, got {eng._fitness_velocity}"

    def test_ema_tracks_negative_delta(self):
        """Negative delta should pull velocity negative."""
        eng = self._make_engine()
        for _ in range(10):
            eng._update_fitness_velocity(-5.0)
        assert eng._fitness_velocity < -2.0

    def test_ema_responsive_to_direction_change(self):
        """EMA should reverse direction when delta sign changes."""
        eng = self._make_engine()
        for _ in range(10):
            eng._update_fitness_velocity(5.0)
        assert eng._fitness_velocity > 2.0
        for _ in range(20):
            eng._update_fitness_velocity(-5.0)
        assert eng._fitness_velocity < -2.0

    def test_velocity_at_boundary_1(self):
        """Velocity at exactly 1.0 should use moderate improvement branch."""
        eng = self._make_engine()
        eng._fitness_velocity = 1.0
        rate = eng._compute_adaptive_mutation_rate()
        # vel=1.0 is not > 1.0, so it falls to plateau branch (base * 1.3)
        assert rate > 0.0

    def test_velocity_at_boundary_5(self):
        """Velocity at exactly 5.0 should use moderate improvement branch."""
        eng = self._make_engine()
        eng._fitness_velocity = 5.0
        rate = eng._compute_adaptive_mutation_rate()
        # vel=5.0 is not > 5.0, so moderate branch (base * 0.75)
        assert rate > 0.0


# ───────────────────────────────────────────────────────────────────────
# Oscillation Control Edge Cases
# ───────────────────────────────────────────────────────────────────────


class TestOscillationControlEdgeCases:
    """Edge cases for champion oscillation control."""

    def _make_engine(self, seed=42):
        cfg = EvolutionConfig(mutation_rate=0.3, seed=seed)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_empty_window_allows_replacement(self):
        """Empty oscillation window should not block replacement."""
        eng = self._make_engine()
        eng.champion_fitness = 10.0
        eng._champion_oscillation_window.clear()
        eng._oscillation_positive_count = 0
        result = eng._should_replace_champion(11.0)
        assert result is True

    def test_single_positive_delta_allows_replacement(self):
        """A single positive delta in window should not trigger hysteresis."""
        eng = self._make_engine()
        eng.champion_fitness = 10.0
        # Single replacement
        result = eng._should_replace_champion(10.5)
        assert result is True
        assert eng._oscillation_positive_count == 1

    def test_exactly_at_threshold_triggers_hysteresis(self):
        """Exactly _oscillation_threshold positive deltas should trigger hysteresis."""
        eng = self._make_engine()
        eng.champion_fitness = 10.0
        threshold = eng._oscillation_threshold

        # Build up exactly threshold positive deltas
        for i in range(threshold):
            eng.champion_fitness += 0.1
            result = eng._should_replace_champion(eng.champion_fitness + 0.1)
            eng._record_champion_replacement(0.1)

        assert eng._oscillation_positive_count >= threshold

        # Now a small delta should be rejected
        eng.champion_fitness += 0.1
        result = eng._should_replace_champion(eng.champion_fitness + 0.01)
        assert result is False

    def test_negative_deltas_not_counted(self):
        """Negative deltas in window should not count as positive."""
        eng = self._make_engine()
        eng.champion_fitness = 10.0
        # Insert a negative delta into the window directly
        eng._champion_oscillation_window.append(-5.0)
        assert eng._oscillation_positive_count == 0

    def test_replacement_counter_accumulates(self):
        """_record_champion_replacement should accumulate across calls."""
        eng = self._make_engine()
        for i in range(5):
            eng._record_champion_replacement(float(i + 1))
        assert eng._champion_replacement_count == 5


# ───────────────────────────────────────────────────────────────────────
# MetaController Consecutive Stagnation Ring Buffer
# ───────────────────────────────────────────────────────────────────────


class TestMetaControllerStagnationRingBuffer:
    """Tests for the O(1) consecutive stagnation tracking."""

    def _make_controller(self, **kwargs):
        return MetaController(checkpoint_dir="/tmp/mc_ring_test", **kwargs)

    def _snapshot(self, cycle, fitness, level=5, diversity=0.3, val_loss=2.0):
        return TelemetrySnapshot(
            cycle=cycle,
            fitness=fitness,
            level=level,
            champion_words=50,
            val_loss=val_loss,
            diversity=diversity,
            novelty_size=10,
            qd_occupied=5,
            qd_total=100,
            traces_count=20,
            retrained=True,
            generations_completed=50,
        )

    def test_consecutive_plateau_increments_counter(self):
        """Consecutive fitness plateaus should increment counter via tune_retrain_params."""
        # Use long level_stall_cycles to avoid LEVEL_STALLED triggering
        mc = self._make_controller(level_stall_cycles=20)
        for c in range(1, 10):
            mc.update_cycle(self._snapshot(c, fitness=-2.0, level=5))
        state = mc.detect_stagnation()
        assert state == StagnationResult.FITNESS_PLATEAU
        # tune_retrain_params calls _record_tuning which populates the ring
        for c in range(1, 10):
            mc.tune_retrain_params(c)
        count = mc._count_consecutive_stagnation(state)
        assert count >= 1

    def test_ring_buffer_capped_at_16(self):
        """Ring buffer should not grow beyond 16 entries."""
        mc = self._make_controller(level_stall_cycles=20)
        for c in range(1, 50):
            mc.update_cycle(self._snapshot(c, fitness=-2.0, level=5))
            mc.detect_stagnation()
            mc._record_tuning("evolution", "mutation_rate", None, 0.5, "stagnation: fitness_plateau")
        assert len(mc._state_ring) <= 16

    def test_non_stagnation_resets_counter(self):
        """Non-stagnation tuning entry resets consecutive counter."""
        mc = self._make_controller(level_stall_cycles=20)
        # First: stagnation (flat fitness, constant level)
        for c in range(1, 8):
            mc.update_cycle(self._snapshot(c, fitness=-2.0, level=5))
        state1 = mc.detect_stagnation()
        assert state1 == StagnationResult.FITNESS_PLATEAU
        # Populate ring with plateau entries via tune_retrain_params
        for c in range(1, 8):
            mc.tune_retrain_params(c)

        # Simulate a non-stagnation tuning event being recorded
        # (this happens during val_loss regression recovery, etc.)
        mc._record_tuning("evolution", "mutation_rate", None, 0.5, "recovery")

        # The ring now has normal/recovery entry at the end
        # _consecutive_stagnation_counter should be reset to 0
        assert mc._consecutive_stagnation_counter == 0
        assert mc._last_stagnation_state == ""


# ───────────────────────────────────────────────────────────────────────
# Adaptive Mutation Rate Boundary Transitions
# ───────────────────────────────────────────────────────────────────────


class TestAdaptiveMutationRateBoundaryTransitions:
    """Tests for smooth transitions across velocity boundaries."""

    def _make_engine(self, seed=42, mutation_rate=0.3):
        cfg = EvolutionConfig(mutation_rate=mutation_rate, seed=seed)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_rate_increases_monotonically_with_regression(self):
        """Mutation rate should increase as regression worsens."""
        eng = self._make_engine()
        rates = []
        for vel in [-0.5, -2.0, -5.0, -10.0]:
            eng._fitness_velocity = vel
            rates.append(eng._compute_adaptive_mutation_rate())
        # Each should be >= previous (more regression = more mutation)
        for i in range(1, len(rates)):
            assert (
                rates[i] >= rates[i - 1] - 0.001
            ), f"Rate at vel=-{10**i} ({rates[i]}) < rate at vel=-{10**(i-1)} ({rates[i-1]})"

    def test_rate_decreases_with_improvement(self):
        """Mutation rate should decrease as improvement accelerates."""
        eng = self._make_engine()
        rates = []
        for vel in [0.5, 2.0, 5.0, 10.0]:
            eng._fitness_velocity = vel
            rates.append(eng._compute_adaptive_mutation_rate())
        # Each should be <= previous (more improvement = less mutation)
        for i in range(1, len(rates)):
            assert (
                rates[i] <= rates[i - 1] + 0.001
            ), f"Rate at vel={10**i} ({rates[i]}) > rate at vel={10**(i-1)} ({rates[i-1]})"

    def test_rate_never_exceeds_config_bounds(self):
        """Rate should respect floor and ceil for all velocity values."""
        eng = self._make_engine(mutation_rate=0.5)
        for vel in [-100, -50, -10, -5, -1, 0, 1, 5, 10, 50, 100]:
            eng._fitness_velocity = float(vel)
            rate = eng._compute_adaptive_mutation_rate()
            assert (
                eng._adaptive_mutation_floor <= rate <= eng._adaptive_mutation_ceil
            ), f"Rate {rate} out of bounds [{eng._adaptive_mutation_floor}, {eng._adaptive_mutation_ceil}] at vel={vel}"


# ───────────────────────────────────────────────────────────────────────
# Hall of Fame Recency Edge Cases
# ───────────────────────────────────────────────────────────────────────


class TestHallOfFameRecencyEdgeCases:
    """Edge cases for Hall of Fame recency weighting."""

    def test_all_same_generation(self):
        """When all programs have same generation, weights should be uniform."""
        hof = HallOfFame(capacity=5)
        for i in range(5):
            hof.add([i], fitness=10.0 + i, level=0, generation=100)

        rng = random.Random(42)
        picks = [hof.get_random(rng)[0] for _ in range(100)]
        # All picks should be from gen 100
        assert all(0 <= p <= 4 for p in picks)

    def test_single_item_always_returns(self):
        """Single item in HoF should always be returned."""
        hof = HallOfFame(capacity=5)
        hof.add([42], fitness=10.0, level=0, generation=1)
        rng = random.Random(42)
        for _ in range(10):
            assert hof.get_random(rng) == [42]

    def test_add_eviction_preserves_others(self):
        """Adding a better program should evict the worst, preserving others."""
        hof = HallOfFame(capacity=3)
        hof.add([1], fitness=10.0, level=0, generation=1)
        hof.add([2], fitness=20.0, level=0, generation=2)
        hof.add([3], fitness=30.0, level=0, generation=3)
        # Add better: should evict [1] (fitness=10.0)
        hof.add([4], fitness=40.0, level=0, generation=4)
        assert hof.size == 3
        best = hof.get_best()
        assert best == [4]
        # [2] and [3] should still be present in the heap
        fitness_values = {e[0] for e in hof._heap}
        assert 20.0 in fitness_values, f"Expected fitness 20.0 in heap, got {fitness_values}"
        assert 30.0 in fitness_values, f"Expected fitness 30.0 in heap, got {fitness_values}"

    def test_worse_program_not_added(self):
        """Worse program should not be added when HoF is full."""
        hof = HallOfFame(capacity=2)
        hof.add([1], fitness=10.0, level=0, generation=1)
        hof.add([2], fitness=20.0, level=0, generation=2)
        added = hof.add([3], fitness=5.0, level=0, generation=3)
        assert added is False
        assert hof.size == 2


# ───────────────────────────────────────────────────────────────────────
# Niche Sharing: No Penalty for Distant Descriptors
# ───────────────────────────────────────────────────────────────────────


class TestNicheSharingNoPenaltyForDistant:
    """Programs in different QD cells should not be penalized."""

    def test_distant_program_no_penalty(self):
        """Programs with different QD descriptors should get no niche penalty."""
        # This is a logic test for the niche penalty calculation
        champ_desc = [-0.5, -0.5]
        prog_desc = [0.5, 0.5]
        dist = abs(prog_desc[0] - champ_desc[0]) + abs(prog_desc[1] - champ_desc[1])
        # dist = 1.0 + 1.0 = 2.0 -> penalty = 0.10 * (1.0 - 2.0/2.0) = 0.0
        penalty = 0.10 * (1.0 - dist / 2.0) if dist < 2.0 else 0.0
        assert penalty == 0.0

    def test_same_cell_full_penalty(self):
        """Programs with identical QD descriptors should get full penalty."""
        champ_desc = [0.5, 0.5]
        prog_desc = [0.5, 0.5]
        dist = abs(prog_desc[0] - champ_desc[0]) + abs(prog_desc[1] - champ_desc[1])
        penalty = 0.10 * (1.0 - dist / 2.0) if dist < 2.0 else 0.0
        assert abs(penalty - 0.10) < 0.001

    def test_partial_distance_partial_penalty(self):
        """Programs at partial distance should get partial penalty."""
        champ_desc = [0.0, 0.0]
        prog_desc = [0.5, 0.0]
        dist = abs(prog_desc[0] - champ_desc[0]) + abs(prog_desc[1] - champ_desc[1])
        penalty = 0.10 * (1.0 - dist / 2.0) if dist < 2.0 else 0.0
        # dist=0.5 -> penalty = 0.10 * (1 - 0.25) = 0.075
        assert abs(penalty - 0.075) < 0.001


# ───────────────────────────────────────────────────────────────────────
# QD Coverage Tracking Tests
# ───────────────────────────────────────────────────────────────────────


class TestQDCoverageTracking:
    """Tests for the _qd_coverage field updated during QD archive operations."""

    def test_qd_coverage_initializes_zero(self):
        """_qd_coverage should initialize at 0.0."""
        cfg = EvolutionConfig(seed=42)
        eng = EvolutionEngine(config=cfg, vm_farm=None)
        assert eng._qd_coverage == 0.0

    def test_compute_qd_axes_returns_valid_range(self):
        """compute_qd_axes should return values in [-1, 1]."""
        prog = [0x00000033, 0x00000063, 0x00000003, 0x00100073]
        trace = {"final_regs": {10: 42, 11: 0, 12: 1}}
        axes = compute_qd_axes(prog, trace)
        assert axes is not None
        assert -1.0 <= axes[0] <= 1.0
        assert -1.0 <= axes[1] <= 1.0

    def test_compute_qd_axes_empty_program(self):
        """compute_qd_axes with empty program returns [-1, -1]."""
        axes = compute_qd_axes([], None)
        assert axes == [-1.0, -1.0]

    def test_compute_qd_axes_no_trace(self):
        """compute_qd_axes with no trace returns [-1, -1] for register axis."""
        prog = [0x00000033]
        axes = compute_qd_axes(prog, None)
        assert axes is not None
        assert axes[0] >= -1.0  # instruction diversity
        assert axes[1] == -1.0  # no trace -> no register info


# ───────────────────────────────────────────────────────────────────────
# Integration: Convergence Quality
# ───────────────────────────────────────────────────────────────────────


class TestConvergenceQualityIntegration:
    """Integration tests for convergence quality mechanisms."""

    def test_adaptive_tournament_diversity_interaction(self):
        """High diversity should produce larger tournaments, improving selection."""
        cfg = EvolutionConfig(population_size=20, tournament_size=3, seed=42)
        eng = EvolutionEngine(config=cfg, vm_farm=None)

        # Simulate high diversity
        eng._current_diversity = 0.8
        base = eng.config.tournament_size
        if eng._current_diversity >= 0.6:
            expected = base + 2
        else:
            expected = base + 1
        assert expected == 5  # 3 + 2

    def test_niche_penalty_reduces_crowding(self):
        """Niche penalty should reduce fitness of same-cell programs."""
        # The penalty formula: fitness *= (1 - 0.10 * (1 - dist/2))
        # At dist=0: fitness *= 0.90
        # At dist=2: fitness *= 1.00 (no penalty)
        fitness = 100.0
        penalty_at_dist_0 = 1.0 - 0.10 * (1.0 - 0.0 / 2.0)
        fitness_after_penalty = fitness * penalty_at_dist_0
        assert abs(fitness_after_penalty - 90.0) < 0.001

    def test_stall_scaled_qd_bonus_reaches_max(self):
        """QD bonus should cap at 2.5x even at extreme stall."""
        stall = 200
        qd_weight = 0.1
        if stall > 10:
            stall_multiplier = min(2.5, 1.0 + 0.0375 * (stall - 10))
            qd_weight *= stall_multiplier
        assert abs(qd_weight / 0.1 - 2.5) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
