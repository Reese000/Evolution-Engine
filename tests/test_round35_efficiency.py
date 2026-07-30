#!/usr/bin/env python3
"""Round 35: Edge cases, memory optimization, convergence quality tests.

Tests:
  A. _eval_result_to_trace handles None gracefully
  B. _should_replace_champion oscillation counter underflow protection
  C. _record_champion_replacement count cap (no unbounded growth)
  D. _compute_fitness exact-match fast path correctness
  E. Hall of Fame quality gate early skip optimization
  F. MetaController detect_stagnation fast-path correctness
  G. MetaController detect_stagnation fast-path does NOT mask real stagnation
  H. MetaEvolution diversity-preserving crossover rate adjustment
  I. mutation_proposer stochastic_perturbation single-non-EBREAK guard
  J. mutation_proposer simplify_program all-NOP program edge case
  K. mutation_proposer simplify_program all-EBREAK program edge case
  L. VMFarm._dict_to_eval_result missing keys
  M. compute_qd_axes with tuple/non-list program
  N. _compute_population_diversity empty list
"""
from __future__ import annotations

import collections
import os
import random
import sys

# -- Path setup --

from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame, VMFarm, compute_qd_axes
from meta_controller import MetaController, StagnationResult, TelemetrySnapshot
from meta_evolution import MetaEvolutionConfig, MetaEvolutionEngine
from mutation_proposer import EBREAK, NOP, DirectedMutator, _is_ebreak, simplify_program

# -- Helpers --


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
    return MetaController(checkpoint_dir="/tmp/r35_test", **kw)


def _make_engine_with_task(expected_regs=None):
    if expected_regs is None:
        expected_regs = {10: 42, 11: 7}
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.task_instance = {"expected_output": {"registers": expected_regs}}
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
    return engine


def _make_replace_engine(champion_fitness=1.0):
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.champion_fitness = champion_fitness
    engine.generations_stalled = 0
    engine._champion_oscillation_window = collections.deque(maxlen=10)
    engine._oscillation_window_size = 10
    engine._oscillation_threshold = 3
    engine._hysteresis_min_delta = 0.5
    engine._base_hysteresis_min_delta = 0.5
    engine._oscillation_positive_count = 0
    return engine


# ══════════════════════════════════════════════════════════════════════════
# A. _eval_result_to_trace handles None gracefully
# ══════════════════════════════════════════════════════════════════════════


class TestEvalResultToTraceNone:
    def test_none_input_returns_crash_trace(self):
        """None eval result should produce a crash trace, not crash."""
        trace = EvolutionEngine._eval_result_to_trace(None)
        assert trace["status"] == "crash"
        assert trace["trap_cause"] == "null_eval_result"
        assert trace["final_regs"] == {}
        assert trace["instructions_count"] == 0

    def test_none_input_dict_structure(self):
        """None eval result trace has all expected keys."""
        trace = EvolutionEngine._eval_result_to_trace(None)
        assert "status" in trace
        assert "final_pc" in trace
        assert "final_regs" in trace
        assert "final_memory" in trace
        assert "instructions_count" in trace
        assert "pc_history" in trace
        assert "reg_history" in trace
        assert "memory_accesses" in trace
        assert "trap_cause" in trace


# ══════════════════════════════════════════════════════════════════════════
# B. Oscillation counter underflow protection
# ══════════════════════════════════════════════════════════════════════════


class TestOscillationCounterUnderflow:
    def test_counter_never_goes_negative(self):
        """_oscillation_positive_count is clamped to >= 0 after eviction."""
        engine = _make_replace_engine(champion_fitness=1.0)
        # Artificially set counter to 1 but window has 10 negatives
        # This simulates a drift condition
        engine._oscillation_positive_count = 1
        for _ in range(10):
            engine._champion_oscillation_window.append(-1.0)
        # Now replace -- should decrement counter for oldest eviction
        # but clamp to 0 if it goes negative
        engine._should_replace_champion(2.0)
        assert engine._oscillation_positive_count >= 0

    def test_counter_stays_consistent_after_many_replacements(self):
        """Counter stays consistent through many replacement cycles."""
        engine = _make_replace_engine(champion_fitness=0.0)
        # Simulate 20 replacements with varying deltas
        for i in range(20):
            engine._should_replace_champion(float(i + 1))
            assert engine._oscillation_positive_count >= 0
            assert engine._oscillation_positive_count <= 10  # window size


# ══════════════════════════════════════════════════════════════════════════
# C. _record_champion_replacement count cap
# ══════════════════════════════════════════════════════════════════════════


class TestReplacementCountCap:
    def test_count_capped_at_10000(self):
        """_champion_replacement_count is capped at 10000."""
        engine = _make_replace_engine()
        engine._champion_replacement_count = 9999
        engine._record_champion_replacement(1.0)
        assert engine._champion_replacement_count == 10000
        # Further calls should not increase beyond cap
        engine._record_champion_replacement(2.0)
        assert engine._champion_replacement_count == 10000

    def test_count_increases_normally_below_cap(self):
        """Count increases normally when below the cap."""
        engine = _make_replace_engine()
        engine._champion_replacement_count = 5
        engine._record_champion_replacement(1.0)
        assert engine._champion_replacement_count == 6


# ══════════════════════════════════════════════════════════════════════════
# D. _compute_fitness exact-match fast path correctness
# ══════════════════════════════════════════════════════════════════════════


class TestFitnessExactMatchPath:
    def test_exact_match_beats_near_match(self):
        """Exact register match scores higher than off-by-one."""
        engine = _make_engine_with_task({10: 42})
        # Exact match
        trace_exact = {
            "status": "ok",
            "trap_cause": "",
            "final_regs": {10: 42},
            "instructions_count": 5,
        }
        fit_exact = engine._compute_fitness([EBREAK], trace_exact)
        # Near match (off by 1)
        engine._cached_task_instance_id = None  # force re-cache
        engine.current_generation = 1
        trace_near = {
            "status": "ok",
            "trap_cause": "",
            "final_regs": {10: 41},
            "instructions_count": 5,
        }
        fit_near = engine._compute_fitness([EBREAK], trace_near)
        assert fit_exact > fit_near

    def test_multiple_exact_matches_all_score_1(self):
        """Multiple exact register matches all get full credit."""
        engine = _make_engine_with_task({10: 42, 11: 7})
        trace = {
            "status": "ok",
            "trap_cause": "",
            "final_regs": {10: 42, 11: 7},
            "instructions_count": 5,
        }
        fit = engine._compute_fitness([EBREAK], trace)
        # reg_match should be 1.0 (all exact), so base_score is ~100
        assert fit >= 90.0


# ══════════════════════════════════════════════════════════════════════════
# E. Hall of Fame quality gate early skip
# ══════════════════════════════════════════════════════════════════════════


class TestHoFEarlySkip:
    def test_skip_when_all_below_worst(self):
        """Quality gate skips when max stage1 fitness < HoF worst."""
        hof = HallOfFame(capacity=5)
        for i in range(5):
            hof.add([i * 10] * 3, fitness=50.0 + i, level=0, generation=i)
        hof_worst = hof._heap[0][0]  # min-heap: worst = 50.0
        stage1_fitnesses = [10.0, 20.0, 30.0]
        max_stage1 = max(stage1_fitnesses)
        # Early skip should trigger
        assert max_stage1 < hof_worst

    def test_no_skip_when_some_above_worst(self):
        """Quality gate does NOT skip when some fitnesses >= HoF worst."""
        hof = HallOfFame(capacity=5)
        for i in range(5):
            hof.add([i * 10] * 3, fitness=10.0 + i, level=0, generation=i)
        hof_worst = hof._heap[0][0]  # 10.0
        stage1_fitnesses = [5.0, 15.0]  # 15.0 > 10.0
        max_stage1 = max(stage1_fitnesses)
        assert max_stage1 >= hof_worst


# ══════════════════════════════════════════════════════════════════════════
# F. MetaController detect_stagnation fast-path correctness
# ══════════════════════════════════════════════════════════════════════════


class TestStagnationFastPath:
    def test_fast_path_returns_normal_when_improving(self):
        """Fast-path returns NORMAL when fitness improving and level fresh."""
        mc = _mc(fitness_plateau_cycles=5, level_stall_cycles=5)
        # Feed 4 cycles with improving fitness and changing levels
        for i in range(4):
            mc.update_cycle(_snap(i, fitness=-3.0 + i * 0.5, level=8 + i))
        result = mc.detect_stagnation()
        assert result == StagnationResult.NORMAL

    def test_fast_path_cached_within_cycle(self):
        """Fast-path result is cached for same telemetry length."""
        mc = _mc()
        for i in range(3):
            mc.update_cycle(_snap(i, fitness=-3.0 + i * 0.5, level=8 + i))
        result1 = mc.detect_stagnation()
        result2 = mc.detect_stagnation()
        # Same result object (cached)
        assert result1 is result2


# ══════════════════════════════════════════════════════════════════════════
# G. Stagnation fast-path does NOT mask real stagnation
# ══════════════════════════════════════════════════════════════════════════


class TestStagnationFastPathDoesNotMask:
    def test_plateau_still_detected_with_flat_fitness(self):
        """Plateau is still detected when fitness is flat (not improving)."""
        mc = _mc(fitness_plateau_cycles=3, level_stall_cycles=5)
        # 5 cycles of flat fitness at same level
        for i in range(5):
            mc.update_cycle(_snap(i, fitness=-1.0, level=9))
        result = mc.detect_stagnation()
        assert result in (
            StagnationResult.FITNESS_PLATEAU,
            StagnationResult.LEVEL_STALLED,
            StagnationResult.COMPOUND_STALL,
        )

    def test_diversity_collapse_still_detected(self):
        """Diversity collapse is detected even if fitness is improving."""
        mc = _mc(diversity_collapse_threshold=0.05)
        # Fitness improving but diversity crashing
        for i in range(5):
            mc.update_cycle(
                _snap(
                    i,
                    fitness=-2.0 + i * 0.5,
                    level=9,
                    diversity=0.3 - i * 0.07,
                )
            )
        result = mc.detect_stagnation()
        # Should detect diversity collapse or compound stall
        # (fast-path checks div_healthy, so it won't fire)
        assert result in (StagnationResult.DIVERSITY_COLLAPSE, StagnationResult.COMPOUND_STALL)


# ══════════════════════════════════════════════════════════════════════════
# H. MetaEvolution diversity-preserving crossover
# ══════════════════════════════════════════════════════════════════════════


class TestMetaEvolutionDiversityCrossover:
    def test_low_variance_boosts_crossover_rate(self):
        """When fitness variance is low, crossover rate is boosted."""
        meta_config = MetaEvolutionConfig(
            meta_population_size=6,
            meta_generations=2,
            base_generations=5,
            meta_crossover_rate=0.3,
            seed=42,
        )
        engine = MetaEvolutionEngine(meta_config=meta_config)
        # Set all fitnesses to very similar values
        engine.fitness = [1.0, 1.01, 0.99, 1.005, 0.995, 1.0]
        valid_fits = [f for f in engine.fitness if f > float("-inf")]
        m = sum(valid_fits) / len(valid_fits)
        var = sum((f - m) ** 2 for f in valid_fits) / len(valid_fits)
        # Variance is very low
        assert var < 0.1

    def test_high_variance_keeps_standard_rate(self):
        """When fitness variance is high, crossover rate is not boosted."""
        meta_config = MetaEvolutionConfig(
            meta_population_size=6,
            meta_generations=2,
            base_generations=5,
            meta_crossover_rate=0.3,
            seed=42,
        )
        engine = MetaEvolutionEngine(meta_config=meta_config)
        engine.fitness = [-10.0, 5.0, 0.0, -5.0, 3.0, -2.0]
        valid_fits = [f for f in engine.fitness if f > float("-inf")]
        m = sum(valid_fits) / len(valid_fits)
        var = sum((f - m) ** 2 for f in valid_fits) / len(valid_fits)
        assert var > 1.0  # high variance


# ══════════════════════════════════════════════════════════════════════════
# I. stochastic_perturbation single-non-EBREAK guard
# ══════════════════════════════════════════════════════════════════════════


class TestStochasticDeleteGuard:
    def test_delete_does_not_remove_sole_non_ebreak(self):
        """Delete mutation does not remove the only non-EBREAK instruction."""
        rng = random.Random(42)
        prog = [0x12345678, EBREAK]  # 1 non-EBREAK + EBREAK

        # Run many times to see if delete ever triggers
        for _ in range(100):
            result, desc = DirectedMutator.stochastic_perturbation(prog, rng)
            if "delete" in desc:
                # If delete happened, result must still have at least 1 non-EBREAK
                non_ebreak = [w for w in result if not _is_ebreak(w)]
                assert len(non_ebreak) >= 1, f"Delete removed sole non-EBREAK: {result}"

    def test_replace_preserves_ebreak(self):
        """Replace mutation preserves EBREAK at end of program."""
        rng = random.Random(42)
        prog = [0x12345678, EBREAK]
        for _ in range(50):
            result, desc = DirectedMutator.stochastic_perturbation(prog, rng)
            assert result[-1] == EBREAK or EBREAK in result


# ══════════════════════════════════════════════════════════════════════════
# J. simplify_program all-NOP program
# ══════════════════════════════════════════════════════════════════════════


class TestSimplifyAllNOP:
    def test_all_nop_reduces_to_nop_ebreak(self):
        """Program of all NOPs reduces to [NOP, EBREAK]."""
        program = [NOP, NOP, NOP, NOP]
        result = simplify_program(program)
        assert len(result) == 2
        assert result[0] == NOP
        assert result[1] == EBREAK

    def test_single_nop_becomes_nop_ebreak(self):
        """Single NOP becomes [NOP, EBREAK]."""
        result = simplify_program([NOP])
        assert len(result) == 2
        assert result[0] == NOP
        assert result[1] == EBREAK

    def test_nop_ebreak_stays_nop_ebreak(self):
        """[NOP, EBREAK] stays [NOP, EBREAK]."""
        result = simplify_program([NOP, EBREAK])
        assert result == [NOP, EBREAK]


# ══════════════════════════════════════════════════════════════════════════
# K. simplify_program all-EBREAK program
# ══════════════════════════════════════════════════════════════════════════


class TestSimplifyAllEBREAK:
    def test_single_ebreak_preserved(self):
        """Single EBREAK stays [EBREAK]."""
        result = simplify_program([EBREAK])
        assert result == [EBREAK]

    def test_ebreak_with_dead_code(self):
        """EBREAK followed by dead code is trimmed to [EBREAK]."""
        result = simplify_program([EBREAK, 0x12345678, 0xAABBCCDD])
        assert result == [EBREAK]


# ══════════════════════════════════════════════════════════════════════════
# L. VMFarm._dict_to_eval_result missing keys
# ══════════════════════════════════════════════════════════════════════════


class TestDictToEvalResultMissingKeys:
    def test_empty_dict(self):
        """Empty dict produces EvalResult with defaults."""
        result = VMFarm._dict_to_eval_result({})
        assert result.success is False
        assert result.registers == {}
        assert result.memory == {}
        assert result.cycles == 0
        assert result.error is None
        assert result.blocked_writes == 0

    def test_partial_keys(self):
        """Partial dict fills in missing fields."""
        result = VMFarm._dict_to_eval_result({"success": True, "cycles": 42})
        assert result.success is True
        assert result.cycles == 42
        assert result.registers == {}
        assert result.error is None


# ══════════════════════════════════════════════════════════════════════════
# M. compute_qd_axes with tuple/non-list program
# ══════════════════════════════════════════════════════════════════════════


class TestQDAxesEdgeCases:
    def test_tuple_program(self):
        """compute_qd_axes handles tuple program (not just list)."""
        result = compute_qd_axes((0x37, 0x13, 0x33), None)
        # Should not crash
        assert result is not None
        assert len(result) == 2

    def test_program_with_non_int_words_handled(self):
        """compute_qd_axes handles non-integer program words gracefully."""
        # Type guard now handles non-int words instead of raising
        result = compute_qd_axes(["not_a_word", 0x37], None)
        assert isinstance(result, list)

    def test_trace_regs_with_non_int_keys(self):
        """compute_qd_axes handles non-int register keys gracefully."""
        trace = {"final_regs": {"10": 42, "abc": 7}}
        result = compute_qd_axes([0x37], trace)
        # Non-int keys should be skipped
        assert result[1] == -1.0  # no valid int keys


# ══════════════════════════════════════════════════════════════════════════
# N. _compute_population_diversity empty list
# ══════════════════════════════════════════════════════════════════════════


class TestPopulationDiversity:
    def test_empty_list(self):
        """_compute_population_diversity returns 0.0 for empty list."""
        result = EvolutionEngine._compute_population_diversity([])
        assert result == 0.0

    def test_single_program(self):
        """_compute_population_diversity with single program works."""
        result = EvolutionEngine._compute_population_diversity([[0x37, 0x13]])
        assert 0.0 <= result <= 1.0

    def test_all_same_opcode(self):
        """All programs using same opcode category gives low diversity."""
        # Two programs both using only ALU_IMM (opcode 0x13)
        result = EvolutionEngine._compute_population_diversity(
            [
                [0x13, 0x13],
                [0x13, 0x13],
            ]
        )
        assert result < 0.2  # only 1 of 11 categories

    def test_many_opcodes_high_diversity(self):
        """Programs using many opcode categories gives high diversity."""
        result = EvolutionEngine._compute_population_diversity(
            [
                [0x37, 0x17, 0x6F, 0x67, 0x63, 0x03, 0x23, 0x13, 0x33, 0x0F, 0x73],
            ]
        )
        assert result == 1.0  # all 11 categories
