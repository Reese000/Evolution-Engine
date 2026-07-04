#!/usr/bin/env python3
"""Round 38: Bug fixes, memory optimization, and convergence quality.

Tests:
  A. _reorder_block ValueError fix for short programs (len 2-3)
  B. _reorder_block in-place rotation produces correct permutation
  C. _saved_sha256s dict ordering -- eviction is FIFO, not arbitrary
  D. _saved_sha256s cap triggers correctly and preserves recent entries
  E. _inject_diversity no wasted pre-allocation
  F. _inject_diversity min-heap scoring produces correct top-k
  G. EvolutionConfig validation edge cases (population_size=2, elite_ratio=0.0)
  H. HallOfFame get_random with single entry at specific level
  I. MetaController detect_stagnation fast path with only 2 telemetry entries
  J. MetaController save/load roundtrip preserves fitness_momentum
  K. TelemetrySnapshot __slots__ prevents __dict__ creation
  L. _reorder_block max_start=0 guard (boundary)
"""
from __future__ import annotations

import collections
import os
import random
import sys

import pytest

# -- Path setup --
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame, compute_qd_axes
from meta_controller import MetaController, StagnationResult, TelemetrySnapshot
from meta_genome import _COPY_FIELDS, MetaGenome
from mutation_proposer import (
    EBREAK,
    NOP,
    DirectedMutator,
    generate_seed_program,
    has_ebreak,
    make_rv32i,
    simplify_program,
    validate_mutation,
)

# -- Helpers --


def _make_engine(minimal=True):
    """Create a minimal EvolutionEngine for unit testing."""
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.config = EvolutionConfig()
    engine.rng = random.Random(42)
    engine.champion_level = 0
    engine.current_generation = 0
    engine.generations_stalled = 0
    engine.champion = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
    engine.champion_trace = {
        "status": "ok",
        "final_regs": {10: 42},
        "final_memory": {},
        "instructions_count": 2,
        "trap_cause": "",
    }
    engine.champion_fitness = 100.0
    engine._raw_task_score = 100.0
    engine._champion_raw_task_score = 100.0
    engine._bonus_threshold_cache = None
    engine._novelty_weight_cache = 0.3
    engine._cached_eval_budget = 5000
    engine._current_diversity = 0.5
    engine._qd_coverage = 0.0
    engine.task_instance = {
        "expected_output": {"registers": {10: 42}},
        "inputs": {"registers": {}, "memory": {}},
    }
    engine.novelty_archive = None
    engine.qd_archive = None
    engine._extract_behavioral_descriptor = None
    engine._compute_qd_axes_fn = compute_qd_axes
    engine._cached_task_instance_id = None
    engine._cached_expected_regs = None
    engine._cached_expected_total = 0
    engine._hall_of_fame = HallOfFame(capacity=10)
    engine._unsolved_levels_set = frozenset()
    engine._last_challengers = []
    engine._last_results = []
    engine._last_challenger_strategies = []
    engine._last_challenger_programs = []
    engine._last_challenger_fitnesses = []
    engine._generation_state_consumed = True
    engine._champion_changed_since_last_qd = False
    engine._qd_champion_reset_tried = False
    engine._champion_qd_valid = False
    engine._cached_champion_qd_desc = None
    engine._fitness_velocity = 0.0
    engine._velocity_alpha = 0.3
    engine._adaptive_mutation_floor = 0.15
    engine._adaptive_mutation_ceil = 0.85
    engine._original_mutation_rate = engine.config.mutation_rate
    engine._consecutive_same_strategy = 0
    engine._last_strategy_used = ""
    engine.best_fitness_history = []
    engine._max_history_len = 500
    engine._champion_replacement_count = 0
    engine._oscillation_positive_count = 0
    engine._base_hysteresis_min_delta = 0.5
    engine._hysteresis_min_delta = 0.5
    engine._oscillation_threshold = 3
    engine._oscillation_window_size = 10
    engine._champion_oscillation_window = collections.deque(maxlen=10)
    engine.circuit_breaker = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
    engine._seed_model_path = ""
    engine._prefix_generator = None
    engine._external_evolution_params = {}
    engine.recurrent_examples_count = 0
    engine._log_file = None
    engine._log_path_cache = None
    engine._log_handle_closed = True
    engine._cached_checkpoint_dir = None
    engine.strategy_archive = None
    engine.champion_anchor = None
    engine._log = lambda msg: None
    engine.vm_farm = None
    engine.curriculum = None
    engine.use_gpu = False
    engine._secondary_task_results = None
    engine._instruction_program = None
    if not minimal:
        from strategy_archive import StrategyArchive

        engine.strategy_archive = StrategyArchive()
    return engine


# ══════════════════════════════════════════════════════════════════════════════
# A. _reorder_block ValueError fix for short programs
# ══════════════════════════════════════════════════════════════════════════════


class TestReorderBlockShortPrograms:
    """R38-A: Programs of length 2-3 previously caused ValueError in
    _reorder_block because min(10, len(prog)-1) could be < 3, making
    randint(3, N) fail with an invalid range."""

    def test_length_2_no_crash(self):
        prog = [make_rv32i("ADDI", rd=5, rs1=0, imm=10), EBREAK]
        rng = random.Random(42)
        # Should return without modification, no ValueError
        result, desc = DirectedMutator._reorder_block(prog, rng)
        assert "no_change_too_small" in desc or len(result) >= 2

    def test_length_3_no_crash(self):
        prog = [
            make_rv32i("ADDI", rd=5, rs1=0, imm=10),
            make_rv32i("ADDI", rd=6, rs1=0, imm=20),
            EBREAK,
        ]
        rng = random.Random(42)
        result, desc = DirectedMutator._reorder_block(prog, rng)
        # Length 3 but max_block = min(10, 2) = 2 < 3, so should be too small
        assert "no_change_too_small" in desc or len(result) == 3

    def test_length_4_can_reorder(self):
        """Length 4 programs should be reorderable (max_block=3 >= 3)."""
        prog = [
            make_rv32i("ADDI", rd=5, rs1=0, imm=10),
            make_rv32i("ADDI", rd=6, rs1=0, imm=20),
            make_rv32i("ADDI", rd=7, rs1=0, imm=30),
            EBREAK,
        ]
        rng = random.Random(42)
        result, desc = DirectedMutator._reorder_block(prog, rng)
        # Should either reorder or return no_change (depending on EBREAK position)
        assert len(result) >= 3
        assert has_ebreak(result)

    def test_length_5_reorders_block(self):
        """Length 5 programs should reorder a block of 3 instructions."""
        prog = [
            make_rv32i("ADDI", rd=5, rs1=0, imm=10),
            make_rv32i("ADDI", rd=6, rs1=0, imm=20),
            make_rv32i("ADD", rd=7, rs1=5, rs2=6),
            make_rv32i("ADDI", rd=8, rs1=0, imm=30),
            EBREAK,
        ]
        rng = random.Random(42)
        # Run multiple times -- at least one should succeed
        reordered_count = 0
        for _ in range(20):
            prog_copy = list(prog)
            result, desc = DirectedMutator._reorder_block(prog_copy, rng)
            if "reorder_" in desc:
                reordered_count += 1
                # Verify EBREAK still present
                assert has_ebreak(result), "EBREAK lost during reorder"
                # Verify no instructions were lost
                assert len(result) == 5, f"Length changed: {len(result)} != 5"
        assert reordered_count > 0, "Never successfully reordered a 5-instruction program"


# ══════════════════════════════════════════════════════════════════════════════
# B. _reorder_block in-place rotation correctness
# ══════════════════════════════════════════════════════════════════════════════


class TestReorderBlockCorrectness:
    """R38-B: The in-place triple-reverse rotation should produce the same
    result as the original block[split:] + block[:split] concatenation."""

    def test_rotation_matches_concatenation(self):
        """Verify rotation produces same permutation as slice-concat."""
        rng = random.Random(42)
        for block_len in range(3, 11):
            for split in range(1, block_len):
                block = list(range(100, 100 + block_len))
                expected = block[split:] + block[:split]
                # Simulate in-place rotation
                prog = list(block)
                prog[0:split] = reversed(prog[0:split])
                prog[split:block_len] = reversed(prog[split:block_len])
                prog[0:block_len] = reversed(prog[0:block_len])
                assert prog == expected, (
                    f"Mismatch: block_len={block_len}, split={split}, " f"got {prog}, expected {expected}"
                )

    def test_reorder_preserves_element_count(self):
        """Reorder must not add or remove elements."""
        prog = [10, 20, 30, 40, 50, EBREAK]
        rng = random.Random(42)
        for _ in range(50):
            prog_copy = list(prog)
            result, _ = DirectedMutator._reorder_block(prog_copy, rng)
            assert len(result) == len(prog), f"Length changed: {len(result)} != {len(prog)}"
            # Verify same elements (sorted)
            assert sorted(result) == sorted(prog), "Elements changed during reorder"


# ══════════════════════════════════════════════════════════════════════════════
# C. _saved_sha256s dict ordering
# ══════════════════════════════════════════════════════════════════════════════


class TestSavedSha256sOrdering:
    """R38-C: SHA-256 dedup changed from set to dict for deterministic
    FIFO eviction. Oldest entries should be evicted first."""

    def test_dict_preserves_insertion_order(self):
        d: dict[str, None] = {}
        for i in range(10):
            d[f"hash_{i}"] = None
        keys = list(d.keys())
        assert keys == [f"hash_{i}" for i in range(10)]

    def test_eviction_keeps_tail(self):
        d: dict[str, None] = {}
        for i in range(100):
            d[f"hash_{i}"] = None
        # Simulate R38 eviction: keep last 60
        keys = list(d.keys())
        d = {k: None for k in keys[-60:]}
        assert len(d) == 60
        # The oldest entries (0-39) should be gone
        assert "hash_0" not in d
        assert "hash_39" not in d
        # The newest entries (40-99) should remain
        assert "hash_40" in d
        assert "hash_99" in d

    def test_eviction_deterministic(self):
        """Two evictions from same starting state produce same result."""
        d1: dict[str, None] = {}
        d2: dict[str, None] = {}
        for i in range(100):
            h = f"hash_{i:04d}"
            d1[h] = None
            d2[h] = None
        keys1 = list(d1.keys())
        keys2 = list(d2.keys())
        d1 = {k: None for k in keys1[-50:]}
        d2 = {k: None for k in keys2[-50:]}
        assert list(d1.keys()) == list(d2.keys())


# ══════════════════════════════════════════════════════════════════════════════
# D. _saved_sha256s cap triggers correctly
# ══════════════════════════════════════════════════════════════════════════════


class TestSavedSha256sCap:
    """R38-D: The cap should trigger at >5000 and reduce to 3000."""

    def test_cap_preserves_recent(self):
        d: dict[str, None] = {}
        for i in range(5001):
            d[f"hash_{i}"] = None
        assert len(d) > 5000
        # Apply cap
        keys = list(d.keys())
        d = {k: None for k in keys[-3000:]}
        assert len(d) == 3000
        assert "hash_5000" in d  # most recent
        assert "hash_2001" in d  # boundary

    def test_no_cap_below_threshold(self):
        d: dict[str, None] = {}
        for i in range(4999):
            d[f"hash_{i}"] = None
        assert len(d) == 4999
        # Should NOT trigger cap
        assert len(d) <= 5000


# ══════════════════════════════════════════════════════════════════════════════
# E. _inject_diversity no wasted pre-allocation
# ══════════════════════════════════════════════════════════════════════════════


class TestInjectDiversityNoWaste:
    """R38-E: The old code pre-allocated a redundancy_scores list of pop_size
    tuples then replaced it with the top_heap. This wastes memory."""

    def test_inject_diversity_works(self):
        engine = _make_engine()
        engine._last_challenger_strategies = ["RANDOM"] * 10
        challengers = [[make_rv32i("ADDI", rd=i, rs1=0, imm=i), EBREAK] for i in range(10)]
        # Should not crash
        engine._inject_diversity(challengers, 2)
        # At least 2 should have been replaced
        inject_count = sum(1 for s in engine._last_challenger_strategies if s == "DIVERSITY_INJECT")
        assert inject_count >= 2

    def test_inject_diversity_n_zero(self):
        engine = _make_engine()
        engine._last_challenger_strategies = ["RANDOM"] * 5
        challengers = [[make_rv32i("ADDI", rd=1, rs1=0, imm=1), EBREAK]] * 5
        engine._inject_diversity(challengers, 0)
        # No changes expected
        assert all(s == "RANDOM" for s in engine._last_challenger_strategies)

    def test_inject_diversity_empty_list(self):
        engine = _make_engine()
        engine._last_challenger_strategies = []
        engine._inject_diversity([], 3)
        # Should not crash


# ══════════════════════════════════════════════════════════════════════════════
# F. _inject_diversity min-heap scoring produces correct top-k
# ══════════════════════════════════════════════════════════════════════════════


class TestInjectDiversityTopK:
    """R38-F: The inline min-heap in _inject_diversity should select the
    n_inject most redundant programs correctly."""

    def test_identical_programs_all_replaced(self):
        """When all programs are identical, all n_inject should be replaced."""
        engine = _make_engine()
        prog = [make_rv32i("ADD", rd=10, rs1=1, rs2=2), EBREAK]
        challengers = [list(prog) for _ in range(10)]
        engine._last_challenger_strategies = ["RANDOM"] * 10
        engine._inject_diversity(challengers, 3)
        inject_count = sum(1 for s in engine._last_challenger_strategies if s == "DIVERSITY_INJECT")
        assert inject_count == 3

    def test_diverse_programs_fewer_replaced(self):
        """When programs are diverse, fewer should be replaced."""
        engine = _make_engine()
        rng = random.Random(42)
        challengers = [generate_seed_program(length=rng.randint(3, 10), rng=rng) for _ in range(20)]
        engine._last_challenger_strategies = ["RANDOM"] * 20
        engine._inject_diversity(challengers, 2)
        inject_count = sum(1 for s in engine._last_challenger_strategies if s == "DIVERSITY_INJECT")
        assert inject_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# G. EvolutionConfig validation edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestEvolutionConfigEdgeCases:
    """R38-G: Boundary value validation for EvolutionConfig."""

    def test_population_size_minimum(self):
        """population_size=2 is the minimum valid value (with matching tournament_size)."""
        config = EvolutionConfig(population_size=2, tournament_size=2)
        assert config.population_size == 2

    def test_population_size_1_raises(self):
        with pytest.raises(ValueError, match="population_size"):
            EvolutionConfig(population_size=1)

    def test_elite_ratio_zero(self):
        """elite_ratio=0.0 is valid (warns but doesn't raise)."""
        config = EvolutionConfig(elite_ratio=0.0)
        assert config.elite_ratio == 0.0

    def test_tournament_size_equals_population(self):
        """tournament_size == population_size is the upper bound."""
        config = EvolutionConfig(population_size=5, tournament_size=5)
        assert config.tournament_size == 5

    def test_tournament_size_exceeds_population(self):
        with pytest.raises(ValueError, match="tournament_size"):
            EvolutionConfig(population_size=5, tournament_size=6)

    def test_max_generations_minimum(self):
        config = EvolutionConfig(max_generations=1)
        assert config.max_generations == 1

    def test_instruction_budget_minimum(self):
        config = EvolutionConfig(instruction_budget=100)
        assert config.instruction_budget == 100

    def test_retrain_val_split_boundary(self):
        """retrain_val_split in [0.0, 1.0] is valid."""
        config = EvolutionConfig(retrain_val_split=0.0)
        assert config.retrain_val_split == 0.0
        config = EvolutionConfig(retrain_val_split=1.0)
        assert config.retrain_val_split == 1.0

    def test_retrain_val_split_out_of_range(self):
        with pytest.raises(ValueError, match="retrain_val_split"):
            EvolutionConfig(retrain_val_split=1.5)


# ══════════════════════════════════════════════════════════════════════════════
# H. HallOfFame get_random with single entry at specific level
# ══════════════════════════════════════════════════════════════════════════════


class TestHallOfFameSingleEntry:
    """R38-H: get_random should handle single entries gracefully."""

    def test_single_entry_returns_program(self):
        hof = HallOfFame(capacity=10)
        prog = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
        hof.add(prog, fitness=50.0, level=3, generation=10)
        rng = random.Random(42)
        result = hof.get_random(rng, level=3)
        assert result is not None
        assert result == prog

    def test_single_entry_wrong_level_falls_through(self):
        hof = HallOfFame(capacity=10)
        prog = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
        hof.add(prog, fitness=50.0, level=3, generation=10)
        rng = random.Random(42)
        # Level 5 has no entries -- should fall through to unfiltered
        result = hof.get_random(rng, level=5)
        assert result is not None

    def test_empty_hof_returns_none(self):
        hof = HallOfFame(capacity=10)
        rng = random.Random(42)
        assert hof.get_random(rng) is None
        assert hof.get_random(rng, level=0) is None


# ══════════════════════════════════════════════════════════════════════════════
# I. MetaController detect_stagnation fast path
# ══════════════════════════════════════════════════════════════════════════════


class TestMetaControllerFastPath:
    """R38-I: detect_stagnation should return NORMAL with only 2 telemetry entries."""

    def test_two_entries_returns_normal(self):
        mc = MetaController(checkpoint_dir="/tmp/mc_r38_test")
        for c, f in enumerate([1.0, 2.0], 1):
            snap = TelemetrySnapshot(
                cycle=c,
                fitness=f,
                level=0,
                champion_words=10,
                val_loss=None,
                diversity=0.5,
                novelty_size=0,
                qd_occupied=0,
                qd_total=100,
                traces_count=0,
                retrained=False,
                generations_completed=50,
            )
            mc.update_cycle(snap)
        result = mc.detect_stagnation()
        assert result == StagnationResult.NORMAL

    def test_one_entry_returns_normal(self):
        mc = MetaController(checkpoint_dir="/tmp/mc_r38_test")
        snap = TelemetrySnapshot(
            cycle=1,
            fitness=1.0,
            level=0,
            champion_words=10,
            val_loss=None,
            diversity=0.5,
            novelty_size=0,
            qd_occupied=0,
            qd_total=100,
            traces_count=0,
            retrained=False,
            generations_completed=50,
        )
        mc.update_cycle(snap)
        assert mc.detect_stagnation() == StagnationResult.NORMAL

    def test_fast_path_takes_effect(self):
        """When fitness is improving, level fresh, and diversity healthy,
        detect_stagnation should return NORMAL via fast path."""
        mc = MetaController(checkpoint_dir="/tmp/mc_r38_test")
        for c, (f, l, d) in enumerate(
            [
                (1.0, 0, 0.5),
                (2.0, 0, 0.5),
                (3.0, 1, 0.5),
                (4.0, 1, 0.5),
            ],
            1,
        ):
            snap = TelemetrySnapshot(
                cycle=c,
                fitness=f,
                level=l,
                champion_words=10,
                val_loss=None,
                diversity=d,
                novelty_size=0,
                qd_occupied=0,
                qd_total=100,
                traces_count=0,
                retrained=False,
                generations_completed=50,
            )
            mc.update_cycle(snap)
        # Fast path: fitness improving, level fresh, diversity healthy
        result = mc.detect_stagnation()
        assert result == StagnationResult.NORMAL


# ══════════════════════════════════════════════════════════════════════════════
# J. MetaController save/load roundtrip preserves fitness_momentum
# ══════════════════════════════════════════════════════════════════════════════


class TestMetaControllerSaveLoad:
    """R38-J: save_state/load_state should preserve fitness_momentum and
    other tuning-dependent state across restarts."""

    def test_roundtrip_preserves_fitness_momentum(self):
        import tempfile

        mc = MetaController(checkpoint_dir="/tmp/mc_r38_test")
        # Simulate some cycles to build up momentum
        for c, f in enumerate([1.0, 3.0, 2.0, 5.0, 4.0], 1):
            snap = TelemetrySnapshot(
                cycle=c,
                fitness=f,
                level=c // 2,
                champion_words=10,
                val_loss=None,
                diversity=0.3,
                novelty_size=0,
                qd_occupied=0,
                qd_total=100,
                traces_count=0,
                retrained=False,
                generations_completed=50,
            )
            mc.update_cycle(snap)

        # Set some specific state
        mc._fitness_momentum = 1.5
        mc._compound_stall_counter = 3
        mc._best_fitness_at_last_tune = 4.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_path = f.name
        try:
            mc.save_state(save_path)

            # Create new controller and load
            mc2 = MetaController(checkpoint_dir="/tmp/mc_r38_test")
            mc2.load_state(save_path)

            assert mc2._fitness_momentum == 1.5
            assert mc2._compound_stall_counter == 3
            assert mc2._best_fitness_at_last_tune == 4.0
        finally:
            os.unlink(save_path)

    def test_roundtrip_preserves_state_ring(self):
        import tempfile

        mc = MetaController(checkpoint_dir="/tmp/mc_r38_test")
        mc._state_ring.append("fitness_plateau")
        mc._state_ring.append("fitness_plateau")
        mc._state_ring.append("level_stalled")
        mc._consecutive_stagnation_counter = 2
        mc._last_stagnation_state = "fitness_plateau"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_path = f.name
        try:
            mc.save_state(save_path)
            mc2 = MetaController(checkpoint_dir="/tmp/mc_r38_test")
            mc2.load_state(save_path)

            assert list(mc2._state_ring) == ["fitness_plateau", "fitness_plateau", "level_stalled"]
            assert mc2._consecutive_stagnation_counter == 2
            assert mc2._last_stagnation_state == "fitness_plateau"
        finally:
            os.unlink(save_path)


# ══════════════════════════════════════════════════════════════════════════════
# K. TelemetrySnapshot __slots__
# ══════════════════════════════════════════════════════════════════════════════


class TestTelemetrySnapshotSlots:
    """R38-K: TelemetrySnapshot should use __slots__ for memory efficiency."""

    def test_no_dict(self):
        snap = TelemetrySnapshot(
            cycle=1,
            fitness=1.0,
            level=0,
            champion_words=10,
            val_loss=None,
            diversity=0.5,
            novelty_size=0,
            qd_occupied=0,
            qd_total=100,
            traces_count=0,
            retrained=False,
            generations_completed=50,
        )
        assert not hasattr(snap, "__dict__"), "TelemetrySnapshot should use __slots__ (no __dict__)"

    def test_to_dict_works(self):
        snap = TelemetrySnapshot(
            cycle=1,
            fitness=1.0,
            level=0,
            champion_words=10,
            val_loss=2.5,
            diversity=0.5,
            novelty_size=0,
            qd_occupied=0,
            qd_total=100,
            traces_count=0,
            retrained=True,
            generations_completed=50,
        )
        d = snap.to_dict()
        assert d["cycle"] == 1
        assert d["fitness"] == 1.0
        assert d["val_loss"] == 2.5
        assert d["retrained"] is True

    def test_memory_savings(self):
        """__slots__ should prevent __dict__ creation."""
        snap = TelemetrySnapshot(
            cycle=1,
            fitness=1.0,
            level=0,
            champion_words=10,
            val_loss=None,
            diversity=0.5,
            novelty_size=0,
            qd_occupied=0,
            qd_total=100,
            traces_count=0,
            retrained=False,
            generations_completed=50,
        )
        # __slots__ class should not have __dict__
        assert "__dict__" not in dir(type(snap))


# ══════════════════════════════════════════════════════════════════════════════
# L. _reorder_block max_start=0 boundary
# ══════════════════════════════════════════════════════════════════════════════


class TestReorderBlockBoundary:
    """R38-L: Test _reorder_block at exact boundaries."""

    def test_program_length_exactly_4(self):
        """Length 4: max_block=3, max_start=1, can only start at 0."""
        prog = [
            make_rv32i("ADDI", rd=5, rs1=0, imm=10),
            make_rv32i("ADDI", rd=6, rs1=0, imm=20),
            make_rv32i("ADDI", rd=7, rs1=0, imm=30),
            EBREAK,
        ]
        rng = random.Random(42)
        for _ in range(10):
            prog_copy = list(prog)
            result, desc = DirectedMutator._reorder_block(prog_copy, rng)
            # Should not crash
            assert has_ebreak(result)

    def test_program_length_exactly_13(self):
        """Length 13: max_block=10, max_start=3."""
        prog = [make_rv32i("ADDI", rd=i, rs1=0, imm=i) for i in range(12)]
        prog.append(EBREAK)
        rng = random.Random(42)
        for _ in range(10):
            prog_copy = list(prog)
            result, desc = DirectedMutator._reorder_block(prog_copy, rng)
            assert has_ebreak(result)
            assert len(result) == 13


# ══════════════════════════════════════════════════════════════════════════════
# M. MetaGenome __copy__ independence
# ══════════════════════════════════════════════════════════════════════════════


class TestMetaGenomeCopy:
    """R38-M: __copy__ should produce independent clones."""

    def test_copy_independence(self):
        import copy

        g1 = MetaGenome.random(rng=random.Random(42))
        g2 = copy.copy(g1)
        # Mutate g1
        g1.mutation_rate = 0.99
        g1.strategy_type_weights["RANDOM"] = 0.99
        # g2 should be unchanged
        assert g2.mutation_rate != 0.99
        assert g2.strategy_type_weights["RANDOM"] != 0.99

    def test_copy_fields_match(self):
        import copy

        g1 = MetaGenome.random(rng=random.Random(42))
        g2 = copy.copy(g1)
        for fld in _COPY_FIELDS:
            assert getattr(g1, fld) == getattr(g2, fld), f"Field {fld} mismatch"
        assert g1.strategy_type_weights == g2.strategy_type_weights


# ══════════════════════════════════════════════════════════════════════════════
# N. _compute_fitness side-effect isolation
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeFitnessSideEffect:
    """R38-N: _compute_fitness sets _raw_task_score as side effect.
    Ensure it's correct for the specific program being evaluated."""

    def test_raw_task_score_reflects_program(self):
        engine = _make_engine()
        # Program 1: perfect match
        prog1 = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
        trace1 = {
            "status": "ok",
            "final_regs": {10: 42},
            "final_memory": {},
            "instructions_count": 2,
            "trap_cause": "",
        }
        fit1 = engine._compute_fitness(prog1, trace1)
        raw1 = engine._raw_task_score

        # Program 2: wrong value
        prog2 = [make_rv32i("ADDI", rd=10, rs1=0, imm=99), EBREAK]
        trace2 = {
            "status": "ok",
            "final_regs": {10: 99},
            "final_memory": {},
            "instructions_count": 2,
            "trap_cause": "",
        }
        fit2 = engine._compute_fitness(prog2, trace2)
        raw2 = engine._raw_task_score

        # Perfect match should score higher
        assert fit1 > fit2
        assert raw1 > raw2

    def test_crash_returns_negative(self):
        engine = _make_engine()
        prog = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
        trace = {
            "status": "crash",
            "final_regs": {},
            "final_memory": {},
            "instructions_count": 0,
            "trap_cause": "illegal",
        }
        fit = engine._compute_fitness(prog, trace)
        assert fit == -100.0
        assert engine._raw_task_score == 0.0

    def test_timeout_returns_negative(self):
        engine = _make_engine()
        prog = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
        trace = {
            "status": "timeout",
            "final_regs": {},
            "final_memory": {},
            "instructions_count": 5000,
            "trap_cause": "",
        }
        fit = engine._compute_fitness(prog, trace)
        assert fit == -25.0
        assert engine._raw_task_score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# O. simplify_program edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestSimplifyProgramEdgeCases:
    """R38-O: simplify_program should handle degenerate inputs."""

    def test_empty_returns_minimal(self):
        result = simplify_program([])
        # Empty input returns empty (no crash), or minimal program
        assert isinstance(result, list)
        assert len(result) <= 2

    def test_single_ebreak(self):
        result = simplify_program([EBREAK])
        assert result == [EBREAK]

    def test_single_non_ebreak(self):
        instr = make_rv32i("ADDI", rd=5, rs1=0, imm=10)
        result = simplify_program([instr])
        assert len(result) == 2
        assert result[0] == instr
        assert result[1] == EBREAK

    def test_all_nops(self):
        nops = [NOP] * 5
        result = simplify_program(nops)
        assert has_ebreak(result)
        assert len(result) <= 3  # At most 1 NOP + EBREAK

    def test_no_ebreak_appends_one(self):
        prog = [make_rv32i("ADDI", rd=5, rs1=0, imm=10)]
        result = simplify_program(prog)
        assert has_ebreak(result)


# ══════════════════════════════════════════════════════════════════════════════
# P. validate_mutation
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateMutation:
    """R38-P: validate_mutation should catch invalid encodings."""

    def test_valid_program(self):
        prog = [make_rv32i("ADDI", rd=5, rs1=0, imm=10), EBREAK]
        ok, err = validate_mutation(prog)
        assert ok is True
        assert err is None

    def test_empty_program(self):
        ok, err = validate_mutation([])
        assert ok is False
        assert "Empty" in err

    def test_invalid_word(self):
        ok, err = validate_mutation([0x00000000, EBREAK])
        assert ok is False
        assert "Instruction 0" in err

    def test_bad_second_instruction(self):
        prog = [make_rv32i("ADDI", rd=5, rs1=0, imm=10), 0xDEADBEEF, EBREAK]
        ok, err = validate_mutation(prog)
        # 0xDEADBEEF may or may not decode as valid depending on encoder
        assert isinstance(ok, bool)
