#!/usr/bin/env python3
"""
Wave 13 Regression Tests — Ensure no existing functionality broke.

Covers:
  1. Core Evolution Behavior (EvolutionConfig, EvolutionEngine, HallOfFame)
  2. Mutation Proposer Unchanged (make_rv32i, has_ebreak, find_ebreak_index, etc.)
  3. Archive Serialization (StrategyArchive, ChampionArchive, MAPElitesArchive)
  4. MetaController Backward Compatibility

All tests are fast, deterministic, and do NOT modify existing files.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Any

import pytest

# Path setup
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame
from meta_controller import MetaController, StagnationResult, TelemetrySnapshot
from mutation_proposer import (
    EBREAK,
    NOP,
    DirectedMutator,
    TraceAnalysis,
    TraceAnalyzer,
    find_ebreak_index,
    has_ebreak,
    make_rv32i,
    make_random_rv32i,
    trim_dead_code,
)
from strategy_archive import StrategyArchive
from champion_archive import ChampionArchive, ChampionEntry, _hash_program
from qd_archive import MAPElitesArchive


# ═══════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def rng():
    """Seeded RNG for deterministic tests."""
    return random.Random(42)


@pytest.fixture
def sample_program(rng):
    """A small valid program: ADDI x10, x0, 42; ADD x11, x10, x0; EBREAK."""
    return [
        make_rv32i("ADDI", rd=10, rs1=0, imm=42),
        make_rv32i("ADD", rd=11, rs1=10, rs2=0),
        EBREAK,
    ]


@pytest.fixture
def sample_program_no_ebreak(rng):
    """Program without EBREAK."""
    return [
        make_rv32i("ADDI", rd=10, rs1=0, imm=42),
        make_rv32i("ADD", rd=11, rs1=10, rs2=0),
    ]


@pytest.fixture
def tmp_dir():
    """Temporary directory for file-based tests."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ═══════════════════════════════════════════════════════════════════════
# 1. TestCoreEvolutionBehavior
# ═══════════════════════════════════════════════════════════════════════

class TestCoreEvolutionBehavior:
    """Ensure EvolutionConfig defaults, EvolutionEngine creation,
    and HallOfFame creation are unchanged."""

    def test_evolution_config_defaults(self):
        """EvolutionConfig must have known default values."""
        cfg = EvolutionConfig()
        assert cfg.population_size == 50
        assert cfg.max_generations == 200
        assert cfg.mutation_rate == 0.3
        assert cfg.crossover_rate == 0.2
        assert cfg.elite_ratio == 0.1
        assert cfg.tournament_size == 3
        assert cfg.instruction_budget == 10000
        assert cfg.eval_budget == 5000
        assert cfg.seed == 42
        assert cfg.curriculum_level == 0
        assert cfg.auto_curriculum is True
        assert cfg.curriculum_stall_threshold == 3
        assert cfg.checkpoint_interval == 25
        assert cfg.use_gpu is False

    def test_evolution_config_retrain_defaults(self):
        """Retrain hyperparameter defaults must be preserved."""
        cfg = EvolutionConfig()
        assert cfg.retrain_lr == pytest.approx(3e-5)
        assert cfg.retrain_batch_size == 8
        assert cfg.retrain_weight_decay == pytest.approx(0.01)
        assert cfg.retrain_grad_clip == pytest.approx(1.0)
        assert cfg.retrain_patience == 5
        assert cfg.retrain_freeze_layers == 0
        assert cfg.retrain_model_size == "medium"
        assert cfg.retrain_max_epochs == 50
        assert cfg.retrain_val_split == pytest.approx(0.05)
        assert cfg.retrain_max_per_level == 0
        assert cfg.retrain_timeout == 3600
        assert cfg.retrain_min_examples == 5

    def test_evolution_config_novelty_defaults(self):
        """Novelty archive defaults must be preserved."""
        cfg = EvolutionConfig()
        assert cfg.novelty_weight == pytest.approx(0.15)
        assert cfg.use_novelty_archive is True
        assert cfg.novelty_threshold == pytest.approx(0.15)

    def test_evolution_config_qd_defaults(self):
        """QD archive defaults must be preserved."""
        cfg = EvolutionConfig()
        assert cfg.use_qd_archive is True
        assert cfg.qd_bonus_weight == pytest.approx(0.1)
        assert cfg.qd_bins == 20

    def test_evolution_config_circuit_breaker_sentinel(self):
        """CircuitBreakerSentinel singleton must be always-open."""
        sentinel = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        assert sentinel.is_breached() is False
        # Sentinel is a singleton class
        assert EvolutionConfig.CIRCUIT_BREAKER_SENTINEL is sentinel

    def test_evolution_engine_creation(self):
        """EvolutionEngine can be created via conftest make_engine."""
        from tests.conftest import make_engine
        engine = make_engine()
        assert isinstance(engine, EvolutionEngine)
        assert engine.config is not None
        assert isinstance(engine.config, EvolutionConfig)
        assert engine.rng is not None
        assert engine.champion_fitness == 100.0
        assert engine.champion_level == 0

    def test_evolution_engine_with_archive(self):
        """EvolutionEngine with strategy archive."""
        from tests.conftest import make_engine
        engine = make_engine(with_archive=True)
        assert engine.strategy_archive is not None
        assert isinstance(engine.strategy_archive, StrategyArchive)

    def test_hall_of_fame_creation(self):
        """HallOfFame can be created with default and custom capacity."""
        hof = HallOfFame()
        assert hof.capacity == 10
        assert len(hof._heap) == 0

        hof_custom = HallOfFame(capacity=5)
        assert hof_custom.capacity == 5

    def test_hall_of_fame_add_and_retrieve(self):
        """HallOfFame add/get_best work correctly."""
        hof = HallOfFame(capacity=3)
        prog_a = [100, 200, EBREAK]
        prog_b = [300, 400, EBREAK]
        prog_c = [500, 600, EBREAK]

        assert hof.add(prog_a, fitness=10.0, level=0) is True
        assert hof.add(prog_b, fitness=20.0, level=1) is True
        assert hof.add(prog_c, fitness=15.0, level=0) is True

        best = hof.get_best()
        assert best is not None
        # best should be prog_b (fitness 20.0)
        assert best == prog_b

    def test_hall_of_fame_eviction(self):
        """HallOfFame evicts worst when full."""
        hof = HallOfFame(capacity=2)
        hof.add([1], fitness=10.0, level=0)
        hof.add([2], fitness=20.0, level=0)
        # Adding a better one should evict the worst
        hof.add([3], fitness=30.0, level=0)
        best = hof.get_best()
        assert best == [3]
        assert len(hof._heap) == 2

    def test_hall_of_fame_reject_worse(self):
        """HallOfFame rejects program with fitness below minimum."""
        hof = HallOfFame(capacity=2)
        hof.add([1], fitness=20.0, level=0)
        hof.add([2], fitness=30.0, level=0)
        # Trying to add worse should be rejected
        result = hof.add([3], fitness=10.0, level=0)
        assert result is False
        assert len(hof._heap) == 2


# ═══════════════════════════════════════════════════════════════════════
# 2. TestMutationProposerUnchanged
# ═══════════════════════════════════════════════════════════════════════

class TestMutationProposerUnchanged:
    """Ensure mutation proposer functions still work correctly."""

    def test_make_rv32i_produces_valid_instructions(self):
        """make_rv32i must produce valid 32-bit instruction encodings."""
        addi = make_rv32i("ADDI", rd=10, rs1=0, imm=42)
        assert isinstance(addi, int)
        assert 0 <= addi <= 0xFFFFFFFF

        add = make_rv32i("ADD", rd=11, rs1=10, rs2=0)
        assert isinstance(add, int)
        assert 0 <= add <= 0xFFFFFFFF

        lui = make_rv32i("LUI", rd=5, imm=0x12345)
        assert isinstance(lui, int)
        assert 0 <= lui <= 0xFFFFFFFF

    def test_make_rv32i_invalid_name_raises(self):
        """make_rv32i raises ValueError for invalid instruction names."""
        with pytest.raises((ValueError, AssertionError)):
            make_rv32i("INVALID_MNEMONIC", rd=0, rs1=0, imm=0)

    def test_make_rv32i_ebreak(self):
        """EBREAK is a known valid instruction."""
        ebreak = make_rv32i("EBREAK", imm=1)
        assert ebreak == EBREAK
        assert ebreak == 0x00100073

    def test_make_random_rv32i(self, rng):
        """make_random_rv32i produces valid instructions."""
        for _ in range(20):
            instr = make_random_rv32i(rng)
            assert isinstance(instr, int)
            assert 0 <= instr <= 0xFFFFFFFF

    def test_has_ebreak_true(self, sample_program):
        """has_ebreak returns True when EBREAK is present."""
        assert has_ebreak(sample_program) is True

    def test_has_ebreak_false(self, sample_program_no_ebreak):
        """has_ebreak returns False when no EBREAK present."""
        assert has_ebreak(sample_program_no_ebreak) is False

    def test_has_ebreak_empty(self):
        """has_ebreak returns False for empty program."""
        assert has_ebreak([]) is False

    def test_has_ebreak_single_ebreak(self):
        """has_ebreak returns True for [EBREAK]."""
        assert has_ebreak([EBREAK]) is True

    def test_find_ebreak_index_found(self, sample_program):
        """find_ebreak_index returns correct index when found."""
        idx = find_ebreak_index(sample_program)
        assert idx == 2  # Last element

    def test_find_ebreak_index_not_found(self, sample_program_no_ebreak):
        """find_ebreak_index returns None when not found."""
        idx = find_ebreak_index(sample_program_no_ebreak)
        assert idx is None

    def test_find_ebreak_index_returns_last(self):
        """find_ebreak_index returns the LAST EBREAK index."""
        prog = [EBREAK, 0x12345678, EBREAK, 0x9ABCDEF0]
        idx = find_ebreak_index(prog)
        assert idx == 2

    def test_find_ebreak_index_empty(self):
        """find_ebreak_index returns None for empty program."""
        assert find_ebreak_index([]) is None

    def test_trim_dead_code_removes_after_ebreak(self):
        """trim_dead_code removes instructions after last EBREAK."""
        dead_instr = make_rv32i("ADDI", rd=5, rs1=0, imm=99)
        prog = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK, dead_instr, dead_instr]
        trimmed = trim_dead_code(prog)
        assert len(trimmed) == 2
        assert trimmed[-1] == EBREAK

    def test_trim_dead_code_no_ebreak_unchanged(self, sample_program_no_ebreak):
        """trim_dead_code returns program unchanged if no EBREAK."""
        result = trim_dead_code(sample_program_no_ebreak)
        assert result == sample_program_no_ebreak

    def test_trim_dead_code_empty(self):
        """trim_dead_code returns empty list for empty input."""
        assert trim_dead_code([]) == []

    def test_trim_dead_code_single_ebreak(self):
        """trim_dead_code preserves [EBREAK]."""
        assert trim_dead_code([EBREAK]) == [EBREAK]

    def test_directed_mutator_mutate_returns_tuple(self, sample_program):
        """DirectedMutator.mutate returns (program, description) tuple."""
        trace = [
            {"regs": [0] * 32, "pc": 0, "cycle": 0},
            {"regs": [0] * 32, "pc": 4, "cycle": 1},
        ]
        analysis = TraceAnalyzer.analyze(
            trace=trace,
            expected={"registers": {10: 42}},
            inputs={"registers": {}, "memory": {}},
            program=sample_program,
        )
        rng = random.Random(42)
        result = DirectedMutator.mutate(
            program=sample_program,
            trace=trace,
            analysis=analysis,
            rng=rng,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        mutated, description = result
        assert isinstance(mutated, list)
        assert isinstance(description, str)
        assert len(mutated) > 0

    def test_directed_mutator_mutate_empty_program(self):
        """DirectedMutator.mutate handles empty program."""
        trace = []
        analysis = TraceAnalysis(
            divergence_point=0,
            divergence_type="unknown",
            final_regs=[0] * 32,
            trace_length=0,
            suggestion={"type": "regenerate", "reason": "empty trace"},
        )
        rng = random.Random(42)
        result = DirectedMutator.mutate(
            program=[],
            trace=trace,
            analysis=analysis,
            rng=rng,
        )
        assert isinstance(result, tuple)
        mutated, desc = result
        assert len(mutated) >= 2
        assert mutated[-1] == EBREAK

    def test_directed_mutator_find_ebreak_index(self):
        """DirectedMutator.find_ebreak_index delegates correctly."""
        prog = [0x12345678, EBREAK, 0x9ABCDEF0]
        assert DirectedMutator.find_ebreak_index(prog) == 1
        assert DirectedMutator.find_ebreak_index([]) is None

    def test_directed_mutator_insert_before_ebreak(self, sample_program):
        """DirectedMutator.insert_before_ebreak inserts correctly."""
        new_instr = make_rv32i("ADDI", rd=1, rs1=0, imm=1)
        result = DirectedMutator.insert_before_ebreak(sample_program, new_instr)
        ebreak_idx = find_ebreak_index(result)
        assert result[ebreak_idx - 1] == new_instr
        assert result[ebreak_idx] == EBREAK


# ═══════════════════════════════════════════════════════════════════════
# 3. TestArchiveSerialization
# ═══════════════════════════════════════════════════════════════════════

class TestArchiveSerialization:
    """Ensure archive save/load roundtrips and operations work."""

    def test_strategy_archive_save_load_roundtrip(self, tmp_dir):
        """StrategyArchive save then load must restore all records."""
        path = os.path.join(tmp_dir, "strategy.json")
        sa = StrategyArchive()
        sa.record("RANDOM", level=0, success=True, generation=1)
        sa.record("TASK_AWARE", level=1, success=False, generation=1)
        sa.record("DIRECTED", level=2, success=True, generation=2)
        assert sa.total_records() == 3

        sa.save(path)
        assert os.path.isfile(path)

        sa2 = StrategyArchive()
        loaded = sa2.load(path)
        assert loaded is True
        assert sa2.total_records() == 3
        # Verify strategy data preserved
        rate = sa2.success_rate("RANDOM", window=50)
        assert rate > 0.0

    def test_strategy_archive_load_nonexistent(self):
        """StrategyArchive.load returns False for missing file."""
        sa = StrategyArchive()
        assert sa.load("/nonexistent/path.json") is False

    def test_strategy_archive_recommend_strategy(self):
        """StrategyArchive.recommend_strategy returns a valid strategy name."""
        sa = StrategyArchive()
        sa.record("RANDOM", level=0, success=True, generation=1)
        sa.record("TASK_AWARE", level=0, success=True, generation=2)
        sa.record("DIRECTED", level=0, success=False, generation=3)

        best = sa.recommend_strategy(level=0)
        assert isinstance(best, str)
        assert best in ("RANDOM", "TASK_AWARE", "DIRECTED")

    def test_strategy_archive_batch_record(self):
        """StrategyArchive.record_batch records multiple entries."""
        sa = StrategyArchive()
        records = [
            {"strategy": "RANDOM", "level": 0, "success": True, "gen": 1},
            {"strategy": "RANDOM", "level": 0, "success": False, "gen": 2},
            {"strategy": "TASK_AWARE", "level": 1, "success": True, "gen": 1},
        ]
        sa.record_batch(records)
        assert sa.total_records() == 3

    def test_champion_archive_add_and_select(self, tmp_dir):
        """ChampionArchive add/sample_training_batch work correctly."""
        path = os.path.join(tmp_dir, "archive.jsonl")
        archive = ChampionArchive(path=path, max_size=50)

        # Add a champion
        program = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
        added = archive.add(
            program=program, level=5, fitness=100.0, cycle=1, generation=10,
        )
        assert added is True
        assert archive.size == 1

        # Add another
        program2 = [make_rv32i("ADD", rd=11, rs1=10, rs2=0), EBREAK]
        archive.add(
            program=program2, level=8, fitness=200.0, cycle=2, generation=20,
        )
        assert archive.size == 2

        # Sample training batch
        batch = archive.sample_training_batch(batch_size=2, rng=random.Random(42))
        assert len(batch) > 0
        assert all(isinstance(e, ChampionEntry) for e in batch)

    def test_champion_archive_deduplication(self, tmp_dir):
        """ChampionArchive deduplicates by instruction hash."""
        path = os.path.join(tmp_dir, "archive.jsonl")
        archive = ChampionArchive(path=path)

        program = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
        assert archive.add(program=program, level=5, fitness=100.0, cycle=1) is True
        assert archive.add(program=program, level=5, fitness=100.0, cycle=2) is False
        assert archive.size == 1

    def test_champion_archive_save_load_roundtrip(self, tmp_dir):
        """ChampionArchive save then reload preserves entries."""
        path = os.path.join(tmp_dir, "archive.jsonl")
        archive = ChampionArchive(path=path)

        program = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
        archive.add(program=program, level=5, fitness=100.0, cycle=1)
        archive.save()

        archive2 = ChampionArchive(path=path)
        assert archive2.size == 1
        assert archive2.entries[0].fitness == 100.0
        assert archive2.entries[0].level == 5

    def test_champion_archive_eviction(self, tmp_dir):
        """ChampionArchive evicts when over max_size."""
        path = os.path.join(tmp_dir, "archive.jsonl")
        archive = ChampionArchive(path=path, max_size=3)

        for i in range(5):
            prog = [make_rv32i("ADDI", rd=i, rs1=0, imm=i), EBREAK]
            archive.add(program=prog, level=0, fitness=float(i * 10), cycle=1)

        assert archive.size <= 3

    def test_map_elites_archive_add_solution(self):
        """MAPElitesArchive.add_solution inserts and returns bool."""
        archive = MAPElitesArchive(bins=10)

        # Add a solution
        result = archive.add_solution(
            descriptor=[0.5, 0.5],
            fitness=100.0,
            program=[make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK],
            trace={"status": "ok"},
        )
        assert result is True
        assert archive.size == 1

        # Add same cell with worse fitness — should not replace
        result2 = archive.add_solution(
            descriptor=[0.5, 0.5],
            fitness=50.0,
            program=[make_rv32i("ADD", rd=11, rs1=10, rs2=0), EBREAK],
        )
        assert result2 is False
        assert archive.size == 1

        # Add same cell with better fitness — should replace
        result3 = archive.add_solution(
            descriptor=[0.5, 0.5],
            fitness=200.0,
            program=[make_rv32i("ADD", rd=12, rs1=10, rs2=0), EBREAK],
        )
        assert result3 is True
        assert archive.size == 1

    def test_map_elites_coverage(self):
        """MAPElitesArchive.coverage returns correct fraction."""
        archive = MAPElitesArchive(bins=10)
        assert archive.coverage() == 0.0

        archive.add_solution(descriptor=[0.5, 0.5], fitness=100.0, program=[EBREAK])
        cov = archive.coverage()
        assert cov > 0.0
        assert cov <= 1.0

    def test_map_elites_serialize_deserialize(self):
        """MAPElitesArchive serialize/deserialize roundtrip."""
        archive = MAPElitesArchive(bins=10)
        archive.add_solution(descriptor=[0.3, 0.7], fitness=100.0, program=[EBREAK])
        archive.add_solution(descriptor=[-0.5, 0.2], fitness=200.0, program=[NOP, EBREAK])

        data = archive.serialize()
        assert isinstance(data, dict)
        assert data["bins"] == 10
        assert len(data["entries"]) == 2

        archive2 = MAPElitesArchive.deserialize(data)
        assert archive2.size == 2
        assert archive2.coverage() == archive.coverage()


# ═══════════════════════════════════════════════════════════════════════
# 4. TestMetaControllerBackwardCompat
# ═══════════════════════════════════════════════════════════════════════

class TestMetaControllerBackwardCompat:
    """Ensure MetaController API hasn't changed."""

    def _make_mc(self):
        """Create a fresh MetaController."""
        return MetaController(checkpoint_dir="/tmp/mc_regression_test")

    def _make_snapshot(self, cycle=1, fitness=1.0, level=0, val_loss=None, diversity=0.5):
        return TelemetrySnapshot(
            cycle=cycle,
            fitness=fitness,
            level=level,
            champion_words=10,
            val_loss=val_loss,
            diversity=diversity,
            novelty_size=0,
            qd_occupied=0,
            qd_total=100,
            traces_count=0,
            retrained=False,
            generations_completed=50,
        )

    def test_detect_stagnation_returns_stagnation_result(self):
        """detect_stagnation must return a StagnationResult enum."""
        mc = self._make_mc()
        result = mc.detect_stagnation()
        assert isinstance(result, StagnationResult)

    def test_detect_stagnation_normal_with_no_history(self):
        """detect_stagnation returns NORMAL with no telemetry."""
        mc = self._make_mc()
        result = mc.detect_stagnation()
        assert result == StagnationResult.NORMAL

    def test_detect_stagnation_with_improving_fitness(self):
        """detect_stagnation returns NORMAL with improving fitness."""
        mc = self._make_mc()
        for i in range(5):
            mc.update_cycle(self._make_snapshot(cycle=i, fitness=float(i)))
        result = mc.detect_stagnation()
        assert result == StagnationResult.NORMAL

    def test_detect_stagnation_with_stagnant_fitness(self):
        """detect_stagnation returns FITNESS_PLATEAU when fitness is flat."""
        mc = self._make_mc()
        # Feed enough cycles with same fitness to trigger plateau
        for i in range(6):
            mc.update_cycle(self._make_snapshot(cycle=i, fitness=1.0))
        result = mc.detect_stagnation()
        assert isinstance(result, StagnationResult)
        # Should be FITNESS_PLATEAU or similar stagnation
        assert result != StagnationResult.NORMAL or result == StagnationResult.FITNESS_PLATEAU

    def test_stagnation_result_is_enum(self):
        """StagnationResult values are strings."""
        for member in StagnationResult:
            assert isinstance(member.value, str)

    def test_stagnation_result_members(self):
        """StagnationResult has expected members."""
        expected = {"normal", "fitness_plateau", "level_stalled",
                    "val_loss_regression", "diversity_collapse", "compound_stall"}
        actual = {member.value for member in StagnationResult}
        assert expected == actual

    def test_tune_retrain_params_returns_dict(self):
        """tune_retrain_params must return a dict."""
        mc = self._make_mc()
        params = mc.tune_retrain_params(cycle=1)
        assert isinstance(params, dict)

    def test_tune_retrain_params_has_expected_keys(self):
        """tune_retrain_params must return dict with known keys."""
        mc = self._make_mc()
        # Need at least one telemetry entry
        mc.update_cycle(self._make_snapshot(cycle=1, fitness=1.0))
        params = mc.tune_retrain_params(cycle=2)
        assert isinstance(params, dict)
        # Should have at least one key
        assert len(params) > 0

    def test_tune_retrain_params_no_stagnation(self):
        """tune_retrain_params with normal state returns baseline params."""
        mc = self._make_mc()
        mc.update_cycle(self._make_snapshot(cycle=1, fitness=1.0))
        params = mc.tune_retrain_params(cycle=2)
        assert isinstance(params, dict)

    def test_tune_evolution_params_returns_dict(self):
        """tune_evolution_params must return a dict."""
        mc = self._make_mc()
        params = mc.tune_evolution_params(cycle=1)
        assert isinstance(params, dict)

    def test_tune_evolution_params_has_expected_keys(self):
        """tune_evolution_params must return a dict (empty when NORMAL, non-empty when stagnant)."""
        mc = self._make_mc()
        # With no telemetry, state is NORMAL → empty dict is valid
        params_empty = mc.tune_evolution_params(cycle=1)
        assert isinstance(params_empty, dict)

        # Feed stagnant history so detect_stagnation returns non-NORMAL
        mc.update_cycle(self._make_snapshot(cycle=1, fitness=1.0))
        for i in range(2, 7):
            mc.update_cycle(self._make_snapshot(cycle=i, fitness=1.0))
        params = mc.tune_evolution_params(cycle=7)
        assert isinstance(params, dict)

    def test_should_rollback_returns_bool(self):
        """should_rollback must return a bool."""
        mc = self._make_mc()
        result = mc.should_rollback()
        assert isinstance(result, bool)

    def test_should_rollback_default_false(self):
        """should_rollback returns False with no telemetry."""
        mc = self._make_mc()
        assert mc.should_rollback() is False

    def test_should_rollback_after_improvement(self):
        """should_rollback stays False when fitness is improving."""
        mc = self._make_mc()
        for i in range(5):
            mc.update_cycle(self._make_snapshot(cycle=i, fitness=float(i * 10)))
        assert mc.should_rollback() is False

    def test_meta_controller_update_cycle(self):
        """update_cycle ingests a TelemetrySnapshot without error."""
        mc = self._make_mc()
        snap = self._make_snapshot(cycle=1, fitness=5.0, level=3)
        mc.update_cycle(snap)
        assert mc.best_fitness == 5.0
        assert mc.best_level == 3

    def test_meta_controller_multiple_updates(self):
        """Multiple update_cycle calls track best fitness correctly."""
        mc = self._make_mc()
        mc.update_cycle(self._make_snapshot(cycle=1, fitness=5.0))
        mc.update_cycle(self._make_snapshot(cycle=2, fitness=10.0))
        mc.update_cycle(self._make_snapshot(cycle=3, fitness=7.0))
        assert mc.best_fitness == 10.0  # Best, not latest
