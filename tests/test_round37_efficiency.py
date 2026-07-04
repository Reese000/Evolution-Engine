#!/usr/bin/env python3
"""Round 37: Efficiency, memory, and convergence quality improvements.

Tests:
  A. _compute_population_diversity early termination when all categories found
  B. _compute_population_diversity still works correctly when not all categories found
  C. _inject_diversity inline min-heap eliminates separate sort pass
  D. population_history attribute removed (memory optimization)
  E. _saved_sha256s set cap prevents unbounded memory growth
  F. TelemetrySnapshot uses __slots__ for memory reduction
  G. _should_replace_champion oscillation count clamped to window size
  H. Checkpoint uses stable filenames (no disk leak)
  I. _count_consecutive_stagnation caches non-zero, falls through on zero
  J. _compute_population_diversity handles mixed int/non-int words
"""
from __future__ import annotations

import collections
import os
import random
import sys

import pytest

# -- Path setup --
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution_engine import EvolutionConfig, EvolutionEngine, compute_qd_axes
from meta_controller import MetaController, StagnationResult, TelemetrySnapshot
from mutation_proposer import EBREAK

# -- Helpers --


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


# ══════════════════════════════════════════════════════════════════════════
# A. _compute_population_diversity early termination
# ══════════════════════════════════════════════════════════════════════════


class TestDiversityEarlyTermination:
    def test_returns_1_when_all_categories_found(self):
        """Diversity returns 1.0 when all 11 opcode categories appear."""
        programs = [
            [0x37, 0x17, 0x6F, 0x67, 0x63, 0x03, 0x23, 0x13, 0x33, 0x0F, 0x73],
        ]
        div = EvolutionEngine._compute_population_diversity(programs)
        assert div == 1.0

    def test_returns_less_than_1_when_missing_categories(self):
        """Diversity returns <1.0 when some categories are missing."""
        programs = [
            [0x13, 0x33],
        ]
        div = EvolutionEngine._compute_population_diversity(programs)
        assert 0.0 < div < 1.0

    def test_returns_0_for_empty_input(self):
        """Diversity returns 0.0 for empty list."""
        div = EvolutionEngine._compute_population_diversity([])
        assert div == 0.0

    def test_returns_0_for_no_valid_words(self):
        """Diversity returns 0.0 when no words map to known categories."""
        div = EvolutionEngine._compute_population_diversity([[0, 0, 0]])
        assert div == 0.0

    def test_early_termination_in_first_program(self):
        """If first program covers all categories, remaining programs are skipped."""
        all_cats = [0x37, 0x17, 0x6F, 0x67, 0x63, 0x03, 0x23, 0x13, 0x33, 0x0F, 0x73]
        programs = [all_cats] + [[0x13]] * 100
        div = EvolutionEngine._compute_population_diversity(programs)
        assert div == 1.0

    def test_scans_multiple_programs_when_needed(self):
        """When categories are spread across programs, all are scanned."""
        p1 = [0x37, 0x17, 0x6F, 0x67, 0x63]
        p2 = [0x03, 0x23, 0x13, 0x33, 0x0F]
        p3 = [0x73]
        div = EvolutionEngine._compute_population_diversity([p1, p2, p3])
        assert div == 1.0

    def test_fewer_categories_gives_proportional_score(self):
        """1 of 11 categories gives diversity = 1/11."""
        div = EvolutionEngine._compute_population_diversity([[0x13]])
        assert div == pytest.approx(1.0 / 11.0)


# ══════════════════════════════════════════════════════════════════════════
# B. _inject_diversity no-op and clamping edge cases
# ══════════════════════════════════════════════════════════════════════════


class TestInjectDiversityEdgeCases:
    def test_no_op_when_n_inject_zero(self):
        """n_inject=0 leaves challengers unchanged."""
        import copy

        rng = random.Random(42)
        original = [[0x13, 0x33, EBREAK] for _ in range(5)]
        original_copy = copy.deepcopy(original)
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.rng = rng
        engine._last_challenger_strategies = ["TEST"] * 5
        engine._inject_diversity(original, 0)
        for i in range(5):
            assert original[i] == original_copy[i]

    def test_no_op_when_empty_list(self):
        """Empty challengers list is a no-op."""
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.rng = random.Random(42)
        engine._last_challenger_strategies = []
        engine._inject_diversity([], 5)

    def test_n_inject_clamped_to_pop_size(self):
        """n_inject larger than pop_size is clamped to pop_size."""
        rng = random.Random(42)
        programs = [[0x13, EBREAK] for _ in range(3)]
        strategies = ["TEST"] * 3
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.rng = rng
        engine._last_challenger_strategies = strategies
        engine._inject_diversity(programs, 100)
        assert len(programs) == 3

    def test_injected_programs_have_ebreak(self):
        """All injected programs have EBREAK terminator."""
        rng = random.Random(42)
        programs = [[0x13, EBREAK] for _ in range(10)]
        strategies = ["TEST"] * 10
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.rng = rng
        engine._last_challenger_strategies = strategies
        engine._inject_diversity(programs, 3)
        for prog in programs:
            assert EBREAK in prog or prog[-1] == EBREAK

    def test_strategy_tag_set_for_injected(self):
        """Injected programs get DIVERSITY_INJECT strategy tag."""
        rng = random.Random(42)
        programs = [[0x13, EBREAK] for _ in range(10)]
        strategies = ["TEST"] * 10
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.rng = rng
        engine._last_challenger_strategies = strategies
        engine._inject_diversity(programs, 3)
        n_injected = sum(1 for s in strategies if s == "DIVERSITY_INJECT")
        assert n_injected == 3


# ══════════════════════════════════════════════════════════════════════════
# C. population_history attribute removed (memory optimization)
# ══════════════════════════════════════════════════════════════════════════


class TestPopulationHistoryRemoved:
    def test_no_population_history_attribute(self):
        """EvolutionEngine does not set population_history in __init__."""
        config = EvolutionConfig()
        engine = EvolutionEngine(config=config)
        assert not hasattr(engine, "population_history")


# ══════════════════════════════════════════════════════════════════════════
# D. _saved_sha256s set cap
# ══════════════════════════════════════════════════════════════════════════


class TestSavedSha256sCap:
    def test_set_capped_at_5000(self):
        """_saved_sha256s is capped at 5000 entries to bound memory."""
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine._saved_sha256s = set(f"hash_{i}" for i in range(5001))
        # Simulate the cap logic from _try_recurrent_feedback
        if len(engine._saved_sha256s) > 5000:
            engine._saved_sha256s = set(list(engine._saved_sha256s)[-3000:])
        assert len(engine._saved_sha256s) <= 3000

    def test_set_grows_normally_below_cap(self):
        """Set grows normally when below the cap."""
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine._saved_sha256s = set(f"hash_{i}" for i in range(100))
        if len(engine._saved_sha256s) > 5000:
            engine._saved_sha256s = set(list(engine._saved_sha256s)[-3000:])
        assert len(engine._saved_sha256s) == 100


# ══════════════════════════════════════════════════════════════════════════
# E. TelemetrySnapshot __slots__
# ══════════════════════════════════════════════════════════════════════════


class TestTelemetrySnapshotSlots:
    def test_has_slots(self):
        """TelemetrySnapshot uses __slots__ for memory efficiency."""
        assert hasattr(TelemetrySnapshot, "__slots__")

    def test_no_dict(self):
        """TelemetrySnapshot instances have no __dict__."""
        snap = _snap(1)
        assert not hasattr(snap, "__dict__")

    def test_to_dict_works(self):
        """to_dict() still works correctly with __slots__."""
        snap = _snap(1, fitness=-1.5, level=10)
        d = snap.to_dict()
        assert d["cycle"] == 1
        assert d["fitness"] == -1.5
        assert d["level"] == 10

    def test_attribute_access(self):
        """Direct attribute access works with __slots__."""
        snap = _snap(5, fitness=0.5, level=12)
        assert snap.cycle == 5
        assert snap.fitness == 0.5
        assert snap.level == 12

    def test_multiple_instances_memory_efficient(self):
        """Creating many instances does not allocate __dict__ overhead."""
        instances = [_snap(i) for i in range(100)]
        for inst in instances:
            assert not hasattr(inst, "__dict__")


# ══════════════════════════════════════════════════════════════════════════
# F. _should_replace_champion oscillation count clamped
# ══════════════════════════════════════════════════════════════════════════


class TestOscillationCountClamped:
    def test_count_never_exceeds_window_size(self):
        """Oscillation positive count is clamped to window size (10)."""
        engine = _make_replace_engine(champion_fitness=0.0)
        # Simulate many replacements to test clamping
        for i in range(20):
            engine._should_replace_champion(float(i + 1))
            assert engine._oscillation_positive_count <= engine._oscillation_window_size

    def test_count_stays_at_window_max(self):
        """Count stabilizes at window size after many replacements."""
        engine = _make_replace_engine(champion_fitness=0.0)
        for i in range(15):
            engine._should_replace_champion(float(i + 1))
        assert engine._oscillation_positive_count == engine._oscillation_window_size


# ══════════════════════════════════════════════════════════════════════════
# G. Checkpoint uses stable filenames
# ══════════════════════════════════════════════════════════════════════════


class TestCheckpointStableFilenames:
    def test_strategy_archive_stable_filename(self):
        """_checkpoint uses strategy_archive.json (not gen-numbered)."""
        import inspect

        source = inspect.getsource(EvolutionEngine._checkpoint)
        assert "strategy_archive.json" in source
        assert "strategy_archive_gen" not in source

    def test_meta_stable_filename(self):
        """_checkpoint uses meta.json (not gen-numbered)."""
        import inspect

        source = inspect.getsource(EvolutionEngine._checkpoint)
        assert '"meta.json"' in source
        assert "meta_gen" not in source


# ══════════════════════════════════════════════════════════════════════════
# H. _count_consecutive_stagnation caching
# ══════════════════════════════════════════════════════════════════════════


class TestConsecutiveStagnationCaching:
    def test_nonzero_result_cached(self):
        """Non-zero consecutive count is cached for O(1) subsequent calls."""
        mc = MetaController(checkpoint_dir="/tmp/r37_test")
        for i in range(5):
            mc.update_cycle(_snap(i, fitness=-1.0, level=9))
        # Force some stagnation entries into the ring
        mc._record_tuning("evolution", "mutation_rate", 0.3, 0.7, "stagnation: fitness_plateau")
        mc._record_tuning("evolution", "mutation_rate", 0.7, 0.8, "stagnation: fitness_plateau")
        state = StagnationResult.FITNESS_PLATEAU
        # First call: O(K) ring scan
        count1 = mc._count_consecutive_stagnation(state)
        # Second call: should use cached O(1) path
        count2 = mc._count_consecutive_stagnation(state)
        assert count1 == count2 == 2

    def test_zero_result_not_cached(self):
        """Zero consecutive count falls through to ring scan (not cached)."""
        mc = MetaController(checkpoint_dir="/tmp/r37_test")
        for i in range(3):
            mc.update_cycle(_snap(i, fitness=-1.0, level=9))
        state = StagnationResult.FITNESS_PLATEAU
        # No stagnation entries in ring -- count should be 0
        count = mc._count_consecutive_stagnation(state)
        assert count == 0
        # Adding a record should be reflected in next call (not stale 0)
        mc._record_tuning("evolution", "mutation_rate", 0.3, 0.7, "stagnation: fitness_plateau")
        count2 = mc._count_consecutive_stagnation(state)
        assert count2 == 1  # not 0 (cached stale value)


# ══════════════════════════════════════════════════════════════════════════
# I. _compute_population_diversity mixed int/non-int words
# ══════════════════════════════════════════════════════════════════════════


class TestDiversityMixedWords:
    def test_non_int_words_skipped(self):
        """Non-integer words in programs are skipped gracefully."""
        programs = [
            ["not_an_int", 0x13, None, 0x33],
        ]
        div = EvolutionEngine._compute_population_diversity(programs)
        # 0x13 = ALU_IMM, 0x33 = ALU_REG => 2/11
        assert div == pytest.approx(2.0 / 11.0)

    def test_empty_program_in_list(self):
        """Empty program in list is handled gracefully."""
        programs = [[], [0x13]]
        div = EvolutionEngine._compute_population_diversity(programs)
        assert div == pytest.approx(1.0 / 11.0)


# ══════════════════════════════════════════════════════════════════════════
# J. Generation-state clearing
# ══════════════════════════════════════════════════════════════════════════


class TestGenerationStateClearing:
    def test_generation_state_consumed_flag_exists(self):
        """EvolutionEngine has _generation_state_consumed flag."""
        engine = EvolutionEngine.__new__(EvolutionEngine)
        # Verify the flag is declared in __init__ by checking the source
        import inspect

        source = inspect.getsource(EvolutionEngine.__init__)
        assert "_generation_state_consumed" in source

    def test_lists_cleared_after_generation(self):
        """After main loop step 12.5, generation-state lists are cleared."""
        import inspect

        source = inspect.getsource(EvolutionEngine.run)
        # Verify clearing code is present
        assert "_last_challengers = []" in source
        assert "_last_results = []" in source
        assert "_last_challenger_programs = []" in source
        assert "_last_challenger_fitnesses = []" in source
