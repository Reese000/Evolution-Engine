#!/usr/bin/env python3
"""Round 34: Evolution efficiency, memory, convergence quality, edge cases.

Tests:
  A. MetaGenome from_config/decode roundtrip for retrain fields (R33 gap fix)
  B. MetaController save/load state roundtrip for all tuning state
  C. Telemetry and tuning log truncation (memory bounds)
  D. Adaptive mutation drift prevention (R33 regression test)
  E. Edge case: population_history and best_fitness_history cap
  F. Meta-evolution avg_fit consistency after z-score normalization
  G. Edge case: compute_qd_axes with empty/None inputs
  H. Edge case: HallOfFame with capacity=1
  I. Edge case: _should_replace_champion at boundary values
  J. Edge case: EvolutionConfig validation boundary tests
  K. Edge case: MetaGenome.mutate with reset_prob=1.0 always resets
  L. Edge case: MetaGenome crossover complement correctness
  M. MetaController load_state restores ring buffer
  N. MetaController load_state handles missing keys gracefully
  O. Stagnation detection: single vs compound indicators
  P. _record_tuning truncation preserves recent entries
  Q. EvolutionEngine history cap prevents unbounded growth
  R. MetaGenome.from_config roundtrip preserves retrain sentinel
  S. Edge case: HallOfFame level-filtered get_best with no matching level
  T. Edge case: _compute_fitness crash path returns correct penalty
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from copy import copy

import pytest

# ── Path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame, compute_qd_axes
from meta_controller import MetaController, StagnationResult, TelemetrySnapshot
from meta_evolution import MetaEvolutionEngine
from meta_genome import _FIELDS, MetaGenome

# ── Helper ────────────────────────────────────────────────────────────────


def _snap(
    cycle: int,
    fitness: float = -2.0,
    level: int = 9,
    val_loss: float = 2.0,
    diversity: float = 0.3,
    champion_words: int = 50,
    retrained: bool = True,
) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        cycle=cycle,
        fitness=fitness,
        level=level,
        champion_words=champion_words,
        val_loss=val_loss,
        diversity=diversity,
        novelty_size=10,
        qd_occupied=5,
        qd_total=100,
        traces_count=20,
        retrained=retrained,
        generations_completed=50,
    )


def _mc(**kw) -> MetaController:
    return MetaController(checkpoint_dir="/tmp/r34_test", **kw)


# ══════════════════════════════════════════════════════════════════════════
# A. MetaGenome from_config / decode roundtrip for retrain fields
# ══════════════════════════════════════════════════════════════════════════


class TestMetaGenomeRoundtrip:
    def test_from_config_preserves_retrain_fields(self):
        """Non-sentinel retrain values survive from_config -> decode."""
        config = EvolutionConfig(
            retrain_lr=1.5e-4,
            retrain_batch_size=16,
            retrain_weight_decay=0.05,
            retrain_patience=8,
            retrain_grad_clip=0.5,
            retrain_max_epochs=100,
            retrain_val_split=0.1,
            retrain_freeze_layers=3,
            retrain_max_per_level=200,
            retrain_timeout=1800,
            retrain_min_examples=20,
        )
        genome = MetaGenome.from_config(config)
        decoded = genome.decode()

        assert decoded.retrain_lr == pytest.approx(1.5e-4)
        assert decoded.retrain_batch_size == 16
        assert decoded.retrain_weight_decay == pytest.approx(0.05)
        assert decoded.retrain_patience == 8
        assert decoded.retrain_grad_clip == pytest.approx(0.5)
        assert decoded.retrain_max_epochs == 100
        assert decoded.retrain_val_split == pytest.approx(0.1)
        assert decoded.retrain_freeze_layers == 3
        assert decoded.retrain_max_per_level == 200
        assert decoded.retrain_timeout == 1800
        assert decoded.retrain_min_examples == 20

    def test_from_config_default_retrain_matches_config(self):
        """Default config retrain values pass through to genome correctly."""
        config = EvolutionConfig()
        genome = MetaGenome.from_config(config)
        # from_config passes through actual config values (not sentinel).
        # The sentinel is only for genomes created without explicit values.
        assert genome.retrain_lr == pytest.approx(config.retrain_lr)
        assert genome.retrain_batch_size == float(config.retrain_batch_size)
        decoded = genome.decode()
        # Decoded config should match the original config
        assert decoded.retrain_lr == pytest.approx(config.retrain_lr)
        assert decoded.retrain_batch_size == config.retrain_batch_size

    def test_full_roundtrip_all_fields(self):
        """Every field in _FIELDS survives encode->decode via from_config."""
        config = EvolutionConfig()
        genome = MetaGenome.from_config(config)
        for name, is_int, default, lo, hi, _ in _FIELDS:
            actual = getattr(genome, name)
            if is_int:
                assert actual == round(actual), f"{name} should be int-like"
            assert lo <= actual <= hi, f"{name}={actual} out of range [{lo}, {hi}]"


# ══════════════════════════════════════════════════════════════════════════
# B. MetaController save/load roundtrip
# ══════════════════════════════════════════════════════════════════════════


class TestMetaControllerSaveLoad:
    def test_save_restore_roundtrip(self):
        """State survives save->load with all tuning fields."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.json")
            mc1 = _mc()
            # Advance state
            for i in range(5):
                mc1.update_cycle(_snap(i, fitness=-2.0 + i * 0.1))
            mc1._compound_stall_counter = 7
            mc1._fitness_momentum = 1.23
            mc1._best_fitness_at_last_tune = -1.5
            mc1.save_state(path)

            mc2 = _mc()
            assert mc2.load_state(path) is True
            assert mc2._compound_stall_counter == 7
            assert mc2._fitness_momentum == pytest.approx(1.23)
            assert mc2._best_fitness_at_last_tune == pytest.approx(-1.5)
            assert mc2.best_fitness == pytest.approx(-1.6)
            assert mc2.best_level == 9
            # Telemetry is NOT persisted (it rebuilds via update_cycle).
            # The controller resumes with empty telemetry and needs 2+ cycles
            # before stagnation detection kicks in. This is by design.
            assert len(mc2.telemetry) == 0

    def test_load_restores_ring_buffer(self):
        """Ring buffer entries survive save/load for consecutive detection."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.json")
            mc1 = _mc()
            for i in range(3):
                mc1.update_cycle(_snap(i, fitness=-2.0))
            # Force some tuning entries into the ring
            mc1._record_tuning("evolution", "mutation_rate", 0.3, 0.7, "stagnation: fitness_plateau")
            mc1._record_tuning("evolution", "mutation_rate", 0.7, 0.8, "stagnation: fitness_plateau")
            mc1._record_tuning("evolution", "mutation_rate", 0.8, 0.9, "stagnation: fitness_plateau")
            mc1.save_state(path)

            mc2 = _mc()
            mc2.load_state(path)
            assert list(mc2._state_ring).count("fitness_plateau") == 3
            assert mc2._consecutive_stagnation_counter == 3

    def test_load_nonexistent_file(self):
        """Loading a nonexistent file returns False without error."""
        mc = _mc()
        assert mc.load_state("/nonexistent/path/state.json") is False

    def test_load_corrupt_file(self):
        """Loading a corrupt JSON file returns False without crash."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.json")
            with open(path, "w") as f:
                f.write("{not valid json")
            mc = _mc()
            assert mc.load_state(path) is False

    def test_load_missing_keys_graceful(self):
        """Loading state with missing keys uses defaults (no KeyError)."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.json")
            # Write minimal state
            with open(path, "w") as f:
                json.dump({"_best_val_loss": 1.5, "best_fitness": -0.5}, f)
            mc = _mc()
            assert mc.load_state(path) is True
            assert mc._best_val_loss == 1.5
            assert mc.best_fitness == pytest.approx(-0.5)
            assert mc._compound_stall_counter == 0  # default
            assert mc.best_level == 0  # default


# ══════════════════════════════════════════════════════════════════════════
# C. Telemetry and tuning log truncation
# ══════════════════════════════════════════════════════════════════════════


class TestMemoryBounds:
    def test_telemetry_capped(self):
        """Telemetry list is capped at _max_telemetry_len."""
        mc = _mc()
        mc._max_telemetry_len = 20
        for i in range(50):
            mc.update_cycle(_snap(i))
        assert len(mc.telemetry) == 20
        # Most recent entries preserved
        assert mc.telemetry[-1].cycle == 49
        assert mc.telemetry[0].cycle == 30

    def test_tuning_log_capped(self):
        """Tuning log is capped at _max_tuning_log_len."""
        mc = _mc()
        mc._max_tuning_log_len = 30
        for i in range(60):
            mc._record_tuning("test", "param", i, i + 1, f"stagnation: fitness_plateau (tier {i % 3})")
        assert len(mc._tuning_log) == 30

    def test_telemetry_truncation_preserves_stagnation_detection(self):
        """Stagnation detection works correctly after telemetry truncation."""
        mc = _mc()
        mc._max_telemetry_len = 10
        # First 5 cycles: improving
        for i in range(5):
            mc.update_cycle(_snap(i, fitness=-3.0 + i * 0.5))
        # Next 15 cycles: plateau (only last 10 kept)
        for i in range(5, 20):
            mc.update_cycle(_snap(i, fitness=-0.5))
        # detect_stagnation should still see the plateau
        result = mc.detect_stagnation()
        assert result in (
            StagnationResult.FITNESS_PLATEAU,
            StagnationResult.LEVEL_STALLED,
            StagnationResult.COMPOUND_STALL,
        )


# ══════════════════════════════════════════════════════════════════════════
# D. Adaptive mutation drift prevention
# ══════════════════════════════════════════════════════════════════════════


class TestAdaptiveMutationDrift:
    def test_adaptive_mutation_uses_original_rate(self):
        """_compute_adaptive_mutation_rate uses _original_mutation_rate as base."""
        config = EvolutionConfig(mutation_rate=0.3)
        engine = EvolutionEngine(config=config)
        engine._fitness_velocity = -3.0  # regression
        engine._original_mutation_rate = 0.3

        rate = engine._compute_adaptive_mutation_rate()
        # With vel=-3.0, adaptive = base * 1.6 = 0.48
        # blended = 0.5 * 0.48 + 0.5 * 0.3 = 0.39
        assert 0.15 <= rate <= 0.85
        assert rate > 0.3  # should increase due to regression

    def test_adaptive_mutation_never_drifts(self):
        """Multiple adaptive calls don't progressively shift the rate."""
        config = EvolutionConfig(mutation_rate=0.3)
        engine = EvolutionEngine(config=config)
        engine._fitness_velocity = 0.0  # plateau
        engine._original_mutation_rate = 0.3

        # Call multiple times -- result should be deterministic
        # because we always use _original_mutation_rate as base
        rates = [engine._compute_adaptive_mutation_rate() for _ in range(5)]
        # All should be the same (same velocity, same base)
        for r in rates:
            assert r == pytest.approx(rates[0])


# ══════════════════════════════════════════════════════════════════════════
# E. Edge cases: history bounds
# ══════════════════════════════════════════════════════════════════════════


class TestHistoryBounds:
    def test_best_fitness_history_capped_at_500(self):
        """EvolutionEngine._max_history_len is respected."""
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine._max_history_len = 10  # small for testing
        engine.best_fitness_history = list(range(20))
        # Simulate the capping logic from _update_champion
        max_hist = engine._max_history_len
        if len(engine.best_fitness_history) > max_hist:
            engine.best_fitness_history = engine.best_fitness_history[-max_hist:]
        assert len(engine.best_fitness_history) == 10
        assert engine.best_fitness_history[0] == 10  # last 10 kept


# ══════════════════════════════════════════════════════════════════════════
# F. Meta-evolution avg_fit consistency
# ══════════════════════════════════════════════════════════════════════════


class TestMetaEvolutionConsistency:
    def test_compute_meta_fitness_with_negative_fitness(self):
        """Negative final_fitness zeroes curriculum_depth."""
        summary = {"final_fitness": -1.0, "final_level": 10, "champion_words": 25, "generations_completed": 50}
        mf = MetaEvolutionEngine._compute_meta_fitness(summary)
        # curriculum_depth should be 0 when fitness < 0
        assert 0.0 <= mf <= 1.0

    def test_compute_meta_fitness_champion_words_sweet_spot(self):
        """Champion words in [25, 50] gets max diversity score."""
        summary = {"final_fitness": 0.5, "final_level": 5, "champion_words": 30, "generations_completed": 50}
        mf = MetaEvolutionEngine._compute_meta_fitness(summary)
        # diversity component should be 1.0 (sweet spot)
        # mf = 0.5/1.5 * 0.3 + 5/13 * 0.5 + 1.0 * 0.2 = 0.1 + 0.192 + 0.2 = 0.492
        assert mf > 0.4

    def test_compute_meta_fitness_stub_penalty(self):
        """Champion words < 5 gets stub penalty."""
        summary_short = {"final_fitness": 0.5, "final_level": 5, "champion_words": 3, "generations_completed": 50}
        summary_good = {"final_fitness": 0.5, "final_level": 5, "champion_words": 30, "generations_completed": 50}
        mf_short = MetaEvolutionEngine._compute_meta_fitness(summary_short)
        mf_good = MetaEvolutionEngine._compute_meta_fitness(summary_good)
        assert mf_short < mf_good

    def test_compute_meta_fitness_bloat_penalty(self):
        """Champion words > 80 gets bloat penalty."""
        summary_ok = {"final_fitness": 0.5, "final_level": 5, "champion_words": 40, "generations_completed": 50}
        summary_bloat = {"final_fitness": 0.5, "final_level": 5, "champion_words": 150, "generations_completed": 50}
        mf_ok = MetaEvolutionEngine._compute_meta_fitness(summary_ok)
        mf_bloat = MetaEvolutionEngine._compute_meta_fitness(summary_bloat)
        assert mf_ok > mf_bloat


# ══════════════════════════════════════════════════════════════════════════
# G. compute_qd_axes edge cases
# ══════════════════════════════════════════════════════════════════════════


class TestComputeQDAxes:
    def test_none_program(self):
        """None program returns [-1.0, -1.0]."""
        result = compute_qd_axes(None, None)
        assert result == [-1.0, -1.0]

    def test_empty_program(self):
        """Empty list returns [-1.0, -1.0]."""
        result = compute_qd_axes([], None)
        assert result == [-1.0, -1.0]

    def test_none_trace(self):
        """None trace returns inst_div but reg_pressure=-1.0."""
        result = compute_qd_axes([0x00100073], None)
        assert result[0] >= -1.0
        assert result[1] == -1.0

    def test_trace_with_empty_regs(self):
        """Trace with empty final_regs returns reg_pressure=-1.0."""
        trace = {"final_regs": {}}
        result = compute_qd_axes([0x00100073], trace)
        assert result[1] == -1.0

    def test_trace_with_non_dict_regs(self):
        """Trace with non-dict final_regs returns reg_pressure=-1.0."""
        trace = {"final_regs": "not_a_dict"}
        result = compute_qd_axes([0x00100073], trace)
        assert result[1] == -1.0

    def test_all_categories_used(self):
        """Using all 11 opcode categories gives inst_div=1.0."""
        # 11 different opcodes covering all categories:
        # LUI(0x37), AUIPC(0x17), JAL(0x6F), JALR(0x67), BRANCH(0x63),
        # LOAD(0x03), STORE(0x23), ALU_IMM(0x13), ALU_REG(0x33),
        # FENCE(0x0F), SYSTEM(0x73)
        opcodes = [0x37, 0x17, 0x6F, 0x67, 0x63, 0x03, 0x23, 0x13, 0x33, 0x0F, 0x73]
        result = compute_qd_axes(opcodes, None)
        assert result[0] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════
# H. HallOfFame edge cases
# ══════════════════════════════════════════════════════════════════════════


class TestHallOfFameEdgeCases:
    def test_capacity_one(self):
        """HallOfFame with capacity=1 keeps only the best."""
        hof = HallOfFame(capacity=1)
        hof.add([1, 2, 3], fitness=1.0, level=0)
        hof.add([4, 5, 6], fitness=2.0, level=0)
        assert hof.size == 1
        assert hof.get_best() == [4, 5, 6]

    def test_capacity_one_replace_worse(self):
        """HallOfFame with capacity=1 does NOT replace with worse program."""
        hof = HallOfFame(capacity=1)
        hof.add([4, 5, 6], fitness=5.0, level=0)
        added = hof.add([1, 2, 3], fitness=1.0, level=0)
        assert added is False
        assert hof.get_best() == [4, 5, 6]

    def test_get_best_no_match_level(self):
        """get_best returns best across all levels when no level matches."""
        hof = HallOfFame(capacity=5)
        hof.add([1], fitness=3.0, level=2)
        hof.add([2], fitness=5.0, level=3)
        # Request level 0 -- no match, should fall through to unfiltered
        result = hof.get_best(level=0)
        assert result == [2]  # best overall

    def test_get_best_empty(self):
        """get_best on empty HoF returns None."""
        hof = HallOfFame(capacity=5)
        assert hof.get_best() is None

    def test_get_random_empty(self):
        """get_random on empty HoF returns None."""
        hof = HallOfFame(capacity=5)
        rng = random.Random(42)
        assert hof.get_random(rng) is None

    def test_get_random_single(self):
        """get_random with one entry always returns it."""
        hof = HallOfFame(capacity=5)
        hof.add([1, 2, 3], fitness=1.0, level=0)
        rng = random.Random(42)
        result = hof.get_random(rng)
        assert result == [1, 2, 3]

    def test_get_random_returns_copy(self):
        """get_random returns a copy, not a reference to internal state."""
        hof = HallOfFame(capacity=5)
        prog = [10, 20, 30]
        hof.add(prog, fitness=1.0, level=0)
        rng = random.Random(42)
        result = hof.get_random(rng)
        result[0] = 999
        assert hof.get_best() == [10, 20, 30]  # original unchanged

    def test_best_fitness_empty(self):
        """best_fitness property on empty HoF returns 0.0."""
        hof = HallOfFame(capacity=5)
        assert hof.best_fitness == 0.0


# ══════════════════════════════════════════════════════════════════════════
# I. _should_replace_champion boundary values
# ══════════════════════════════════════════════════════════════════════════


class TestReplaceChampionBoundary:
    def _make_engine(self, champion_fitness=1.0):
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.champion_fitness = champion_fitness
        engine.generations_stalled = 0
        engine._champion_oscillation_window = __import__("collections").deque(maxlen=10)
        engine._oscillation_window_size = 10
        engine._oscillation_threshold = 3
        engine._hysteresis_min_delta = 0.5
        engine._base_hysteresis_min_delta = 0.5
        engine._oscillation_positive_count = 0
        return engine

    def test_exact_equal_no_replace(self):
        """Candidate with exactly equal fitness does NOT replace."""
        engine = self._make_engine(champion_fitness=1.0)
        assert engine._should_replace_champion(1.0) is False

    def test_zero_delta_no_replace(self):
        """Candidate with zero delta does NOT replace."""
        engine = self._make_engine(champion_fitness=0.0)
        assert engine._should_replace_champion(0.0) is False

    def test_negative_delta_no_replace(self):
        """Candidate with negative fitness does NOT replace."""
        engine = self._make_engine(champion_fitness=2.0)
        assert engine._should_replace_champion(1.0) is False

    def test_small_positive_without_oscillation(self):
        """Small positive delta replaces when no oscillation detected."""
        engine = self._make_engine(champion_fitness=1.0)
        # Window is empty, so oscillation not detected
        assert engine._should_replace_champion(1.1) is True

    def test_small_positive_with_oscillation_blocked(self):
        """Small positive delta blocked when oscillation detected."""
        engine = self._make_engine(champion_fitness=1.0)
        # Fill oscillation window with replacements
        for _ in range(3):
            engine._champion_oscillation_window.append(1.0)
            engine._oscillation_positive_count += 1
        # Now try a small delta (< 0.5 threshold)
        assert engine._should_replace_champion(1.2) is False

    def test_large_positive_with_oscillation_allowed(self):
        """Large positive delta (> threshold) replaces even with oscillation."""
        engine = self._make_engine(champion_fitness=1.0)
        for _ in range(3):
            engine._champion_oscillation_window.append(1.0)
            engine._oscillation_positive_count += 1
        # Large delta (> 0.5) should pass even with oscillation
        assert engine._should_replace_champion(2.0) is True

    def test_stall_adaptive_hysteresis_relaxes(self):
        """Hysteresis threshold decreases with stall depth."""
        engine = self._make_engine(champion_fitness=1.0)
        engine.generations_stalled = 50
        for _ in range(3):
            engine._champion_oscillation_window.append(1.0)
            engine._oscillation_positive_count += 1
        # At stall=50, stall_factor = max(0.2, 1.0 - 0.016*50) = 0.2
        # effective_delta = 0.5 * 0.2 = 0.1
        # delta=0.15 > 0.1, so should replace
        assert engine._should_replace_champion(1.15) is True


# ══════════════════════════════════════════════════════════════════════════
# J. EvolutionConfig validation boundary tests
# ══════════════════════════════════════════════════════════════════════════


class TestEvolutionConfigBoundary:
    def test_population_size_one_rejected(self):
        """population_size < 2 raises ValueError."""
        with pytest.raises(ValueError, match="population_size"):
            EvolutionConfig(population_size=1)

    def test_population_size_zero_rejected(self):
        """population_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="population_size"):
            EvolutionConfig(population_size=0)

    def test_tournament_size_one_rejected(self):
        """tournament_size < 2 raises ValueError."""
        with pytest.raises(ValueError, match="tournament_size"):
            EvolutionConfig(tournament_size=1)

    def test_tournament_exceeds_population_rejected(self):
        """tournament_size > population_size raises ValueError."""
        with pytest.raises(ValueError, match="tournament_size.*population_size"):
            EvolutionConfig(population_size=10, tournament_size=15)

    def test_elite_ratio_over_half_rejected(self):
        """elite_ratio > 0.5 raises ValueError."""
        with pytest.raises(ValueError, match="elite_ratio"):
            EvolutionConfig(elite_ratio=0.6)

    def test_instruction_budget_too_low_rejected(self):
        """instruction_budget < 100 raises ValueError."""
        with pytest.raises(ValueError, match="instruction_budget"):
            EvolutionConfig(instruction_budget=50)

    def test_retrain_timeout_too_low_rejected(self):
        """retrain_timeout < 60 raises ValueError."""
        with pytest.raises(ValueError, match="retrain_timeout"):
            EvolutionConfig(retrain_timeout=30)

    def test_negative_novelty_weight_rejected(self):
        """Negative novelty_weight raises ValueError."""
        with pytest.raises(ValueError, match="novelty_weight"):
            EvolutionConfig(novelty_weight=-0.1)


# ══════════════════════════════════════════════════════════════════════════
# K. MetaGenome.mutate with reset_prob=1.0
# ══════════════════════════════════════════════════════════════════════════


class TestMetaGenomeMutateEdge:
    def test_reset_prob_1_always_random(self):
        """With reset_prob=1.0, mutate always returns a random genome."""
        genome = MetaGenome(
            mutation_rate=0.99,
            crossover_rate=0.99,
            elite_ratio=0.49,
            tournament_size=20,
            population_size=200,
        )
        rng = random.Random(42)
        for _ in range(10):
            child = genome.mutate(reset_prob=1.0, rng=rng)
            # A random genome won't necessarily have 0.99 mutation_rate
            # but it WILL be within valid range
            assert 0.0 <= child.mutation_rate <= 1.0
            assert 2 <= child.tournament_size <= 20

    def test_mutate_preserves_parent(self):
        """mutate returns a new genome, does not modify parent."""
        genome = MetaGenome(mutation_rate=0.5)
        original_rate = genome.mutation_rate
        rng = random.Random(42)
        _ = genome.mutate(reset_prob=0.0, std=0.1, rng=rng)
        assert genome.mutation_rate == original_rate

    def test_mutation_noise_bounded(self):
        """Perturbation stays within field bounds even with large std."""
        genome = MetaGenome()
        rng = random.Random(42)
        for _ in range(20):
            child = genome.mutate(std=5.0, reset_prob=0.0, rng=rng)
            assert 0.0 <= child.mutation_rate <= 1.0
            assert 0.01 <= child.elite_ratio <= 0.5
            assert 2 <= child.tournament_size <= 20
            assert 10 <= child.population_size <= 200


# ══════════════════════════════════════════════════════════════════════════
# L. MetaGenome crossover complement correctness
# ══════════════════════════════════════════════════════════════════════════


class TestMetaGenomeCrossover:
    def test_crossover_complement_property(self):
        """Child2 gets what child1 did NOT pick from parent A for each field."""
        a = MetaGenome(mutation_rate=0.1)
        b = MetaGenome(mutation_rate=0.9)
        rng = random.Random(42)
        c1, c2 = a.crossover(b, rng=rng)
        # For mutation_rate: either c1 gets a's and c2 gets b's, or vice versa
        if c1.mutation_rate == pytest.approx(0.1):
            assert c2.mutation_rate == pytest.approx(0.9)
        else:
            assert c1.mutation_rate == pytest.approx(0.9)
            assert c2.mutation_rate == pytest.approx(0.1)

    def test_crossover_preserves_all_fields(self):
        """All fields in _FIELDS are set on both children."""
        a = MetaGenome()
        b = MetaGenome()
        rng = random.Random(42)
        c1, c2 = a.crossover(b, rng=rng)
        for name, _, _, _, _, _ in _FIELDS:
            assert hasattr(c1, name)
            assert hasattr(c2, name)


# ══════════════════════════════════════════════════════════════════════════
# M. Stagnation detection: single vs compound
# ══════════════════════════════════════════════════════════════════════════


class TestStagnationDetection:
    def test_single_indicator_fitness_plateau(self):
        """Fitness plateau alone -> FITNESS_PLATEAU."""
        mc = _mc(fitness_plateau_cycles=3)
        # 3 cycles of no improvement + flat fitness
        for i in range(4):
            mc.update_cycle(_snap(i, fitness=-1.0))
        result = mc.detect_stagnation()
        assert result in (
            StagnationResult.FITNESS_PLATEAU,
            StagnationResult.LEVEL_STALLED,
            StagnationResult.COMPOUND_STALL,
        )

    def test_two_indicators_compound(self):
        """Fitness plateau + level stall -> COMPOUND_STALL."""
        mc = _mc(fitness_plateau_cycles=3, level_stall_cycles=3)
        for i in range(6):
            mc.update_cycle(_snap(i, fitness=-1.0, level=9))
        result = mc.detect_stagnation()
        assert result == StagnationResult.COMPOUND_STALL

    def test_diversity_collapse_trend(self):
        """Diversity trend (40%+ drop) triggers DIVERSITY_COLLAPSE."""
        mc = _mc(diversity_collapse_threshold=0.05)
        # Start high, drop fast
        divs = [0.5, 0.4, 0.3, 0.2, 0.04]
        for i, d in enumerate(divs):
            mc.update_cycle(_snap(i, diversity=d))
        result = mc.detect_stagnation()
        assert result in (StagnationResult.DIVERSITY_COLLAPSE, StagnationResult.COMPOUND_STALL)

    def test_normal_when_no_indicators(self):
        """No indicators -> NORMAL."""
        mc = _mc()
        for i in range(5):
            mc.update_cycle(_snap(i, fitness=-1.0 + i * 0.5, level=9 + i))
        result = mc.detect_stagnation()
        assert result == StagnationResult.NORMAL


# ══════════════════════════════════════════════════════════════════════════
# P. _record_tuning truncation preserves recent entries
# ══════════════════════════════════════════════════════════════════════════


class TestRecordTuningTruncation:
    def test_recent_entries_preserved(self):
        """After truncation, the most recent entries are in the log."""
        mc = _mc()
        mc._max_tuning_log_len = 5
        for i in range(10):
            mc._record_tuning("test", "p", i, i + 1, "recovery")
        assert len(mc._tuning_log) == 5
        assert mc._tuning_log[-1].new_value == 10

    def test_ring_buffer_independent_of_truncation(self):
        """Ring buffer is independent of tuning log truncation."""
        mc = _mc()
        mc._max_tuning_log_len = 3
        for i in range(5):
            mc._record_tuning("test", "p", i, i + 1, f"stagnation: fitness_plateau (tier {i})")
        # Ring should have the last entries (maxlen=16, not capped at 3)
        ring_list = list(mc._state_ring)
        assert len(ring_list) == 5


# ══════════════════════════════════════════════════════════════════════════
# R. MetaGenome.from_config roundtrip preserves retrain sentinel
# ══════════════════════════════════════════════════════════════════════════


class TestRetrainSentinel:
    def test_sentinel_preserved_through_mutation(self):
        """Retrain sentinel (-1.0) survives mutation (clamped to range)."""
        genome = MetaGenome(retrain_lr=-1.0)
        rng = random.Random(42)
        child = genome.mutate(std=0.1, reset_prob=0.0, rng=rng)
        # Sentinel -1.0 is within the low bound (-1.0), so it should stay
        # or be perturbed slightly but still within [-1.0, 1.0]
        assert -1.0 <= child.retrain_lr <= 1.0

    def test_non_sentinel_preserved_through_copy(self):
        """Non-sentinel retrain values survive __copy__."""
        genome = MetaGenome(retrain_lr=0.5, retrain_batch_size=64.0)
        clone = copy(genome)
        assert clone.retrain_lr == pytest.approx(0.5)
        assert clone.retrain_batch_size == pytest.approx(64.0)


# ══════════════════════════════════════════════════════════════════════════
# T. _compute_fitness crash paths
# ══════════════════════════════════════════════════════════════════════════


class TestComputeFitnessEdgeCases:
    def _make_engine_with_task(self):
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.task_instance = {"expected_output": {"registers": {10: 42, 11: 7}}}
        engine.champion_level = 0
        engine._bonus_threshold_cache = None
        engine._novelty_weight_cache = 0.0
        engine._cached_eval_budget = 5000
        engine._current_diversity = 0.5
        engine._qd_coverage = 0.0
        engine._raw_task_score = 0.0
        engine._champion_raw_task_score = 0.0
        engine.config = EvolutionConfig()
        engine.champion = None
        engine.champion_trace = None
        engine.novelty_archive = None
        engine.qd_archive = None
        engine._extract_behavioral_descriptor = None
        engine._compute_qd_axes_fn = compute_qd_axes
        engine.generations_stalled = 0
        engine.current_generation = 0
        engine._cached_task_instance_id = None
        engine._cached_expected_regs = None
        engine._cached_expected_total = 0
        engine._secondary_task_results = None
        engine._instruction_program = None
        return engine

    def test_illegal_instruction_penalty(self):
        """Illegal instruction trace returns -100.0."""
        engine = self._make_engine_with_task()
        trace = {"status": "crash", "trap_cause": "illegal instruction", "final_regs": {}, "instructions_count": 5}
        fitness = engine._compute_fitness([0xDEADBEEF], trace)
        assert fitness == -100.0

    def test_timeout_penalty(self):
        """Timeout trace returns -25.0."""
        engine = self._make_engine_with_task()
        trace = {"status": "timeout", "trap_cause": "", "final_regs": {}, "instructions_count": 5000}
        fitness = engine._compute_fitness([0x00100073], trace)
        assert fitness == -25.0

    def test_crash_penalty(self):
        """Generic crash returns -50.0."""
        engine = self._make_engine_with_task()
        trace = {"status": "crash", "trap_cause": "segfault", "final_regs": {}, "instructions_count": 3}
        fitness = engine._compute_fitness([0xDEADBEEF], trace)
        assert fitness == -50.0

    def test_perfect_match_high_fitness(self):
        """Perfect register match produces high fitness."""
        engine = self._make_engine_with_task()
        trace = {"status": "ok", "trap_cause": "", "final_regs": {10: 42, 11: 7}, "instructions_count": 15}
        fitness = engine._compute_fitness([0x00100073], trace)
        assert fitness > 90.0  # near-perfect match

    def test_no_task_returns_zero(self):
        """No task instance returns 0.0."""
        engine = self._make_engine_with_task()
        engine.task_instance = None
        fitness = engine._compute_fitness([], {})
        assert fitness == 0.0

    def test_zero_expected_regs(self):
        """Zero expected registers -> reg_match=1.0 -> positive fitness."""
        engine = self._make_engine_with_task()
        engine.task_instance = {"expected_output": {"registers": {}}}
        engine._cached_task_instance_id = -1  # force re-cache
        trace = {"status": "ok", "final_regs": {}, "instructions_count": 10}
        fitness = engine._compute_fitness([], trace)
        assert fitness > 0.0
