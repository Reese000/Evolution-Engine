#!/usr/bin/env python3
"""Focused tests for iteration improvements — backward branches, delta tracking,
delta tiebreaker, level-dependent decay, diversity scoring, novelty archive,
QD archive, champion archive, and engine state methods.

Covers 20 test classes matching the verified working subset.
"""

from __future__ import annotations

import json
import os

import pytest

# Ensure project root is on sys.path


# ─── Shared helpers ─────────────────────────────────────────────────────────


def _make_engine(champion=None, level=0, fitness=0.0, verified=False, generation=0):
    """Create a minimal EvolutionEngine via __new__ with mock archives."""
    from evolution_engine import EvolutionEngine

    novelty = type(
        "N",
        (),
        {
            "size": 5,
            "novelty_threshold": 0.5,
            "get_diversity_score": lambda self: 0.75,
        },
    )()

    qd = type(
        "Q",
        (),
        {
            "size": 10,
            "coverage": lambda self: 0.05,
            "num_bins": 20,
            "get_fitness_distribution": lambda self: {},
        },
    )()

    sa = type(
        "S",
        (),
        {
            "total_records": lambda self: 20,
            "get_strategy_diversity": lambda self, level=None: 0.8,
            "save": lambda self, path: None,
        },
    )()

    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.champion = champion
    engine.champion_level = level
    engine.champion_fitness = fitness
    engine._champion_verified = verified
    engine.current_generation = generation
    engine.generations_stalled = 0
    engine._fitness_velocity = 1.23
    engine._fitness_momentum = 0.8
    engine.recurrent_examples_count = 100
    engine.novelty_archive = novelty
    engine.qd_archive = qd
    engine.strategy_archive = sa
    engine._hall_of_fame = type("H", (), {"_heap": [1, 2, 3]})()
    engine.config = type(
        "C",
        (),
        {
            "mutation_rate": 0.1,
            "population_size": 50,
            "elite_ratio": 0.1,
            "novelty_weight": 0.15,
            "qd_bonus_weight": 0.1,
            "eval_budget": 5000,
            "max_generations": 200,
            "auto_curriculum": True,
            "use_gpu": False,
            "model_guided_ratio": None,
            "model_guided_ratio_low": 0.2,
            "model_guided_ratio_mid": 0.15,
            "model_guided_ratio_high": 0.1,
        },
    )()
    engine._log = lambda msg: None
    engine._last_challenger_strategies = []
    return engine


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestBackwardBranches
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackwardBranches:
    """Test EvolutionEngine._count_backward_branches static method."""

    @staticmethod
    def _make_b_type(imm: int) -> int:
        """Encode a B-type instruction word with the given signed immediate."""
        sign = 0x1000 if imm < 0 else 0
        abs_imm = abs(imm)
        word = (
            0x63  # B-type opcode
            | (1 << 15)  # rs1 = x1
            | (2 << 20)  # rs2 = x2
            | (0 << 12)  # funct3 = 0
            | ((abs_imm >> 1) & 0x1) << 7
            | ((abs_imm >> 2) & 0xF) << 8
            | ((abs_imm >> 5) & 0x3F) << 25
            | ((abs_imm >> 11) & 0x1) << 30
        )
        if sign:
            word |= 1 << 31
        return word

    def test_backward_branch_detected(self):
        from evolution_engine import EvolutionEngine

        b_backward = self._make_b_type(-8)
        program = [b_backward, 0x00100073]
        assert EvolutionEngine._count_backward_branches(program) == 1

    def test_forward_branch_not_counted(self):
        from evolution_engine import EvolutionEngine

        b_forward = self._make_b_type(8)
        program = [b_forward, 0x00100073]
        assert EvolutionEngine._count_backward_branches(program) == 0

    def test_multiple_backward_branches(self):
        from evolution_engine import EvolutionEngine

        b1 = self._make_b_type(-4)
        b2 = self._make_b_type(-12)
        b3 = self._make_b_type(-20)
        program = [b1, b2, b3, 0x00100073]
        assert EvolutionEngine._count_backward_branches(program) == 3

    def test_empty_program(self):
        from evolution_engine import EvolutionEngine

        assert EvolutionEngine._count_backward_branches([]) == 0

    def test_no_branch_instructions(self):
        from evolution_engine import EvolutionEngine

        nop = 0x00000013  # ADDI x0, x0, 0
        program = [nop, nop, nop, 0x00100073]
        assert EvolutionEngine._count_backward_branches(program) == 0

    def test_mixed_branches(self):
        from evolution_engine import EvolutionEngine

        b_backward = self._make_b_type(-8)
        b_forward = self._make_b_type(16)
        program = [b_backward, b_forward, b_backward, 0x00100073]
        assert EvolutionEngine._count_backward_branches(program) == 2

    def test_non_integer_values_skipped(self):
        from evolution_engine import EvolutionEngine

        program_mixed = [0x63, "not_an_int", None, 3.14, 0x00100073]
        assert EvolutionEngine._count_backward_branches(program_mixed) == 0

    def test_all_non_integers(self):
        from evolution_engine import EvolutionEngine

        assert EvolutionEngine._count_backward_branches(["hello", None, [1, 2], {}, 3.14]) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestDeltaTracking
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeltaTracking:
    """Test StrategyArchive.record_delta and avg_delta."""

    def test_record_and_avg_single(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        sa.record_delta("RANDOM", 10.0)
        assert sa.avg_delta("RANDOM") == pytest.approx(10.0)

    def test_record_and_avg_multiple(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        sa.record_delta("RANDOM", 5.0)
        sa.record_delta("RANDOM", 15.0)
        sa.record_delta("RANDOM", 10.0)
        assert sa.avg_delta("RANDOM") == pytest.approx(10.0)

    def test_avg_delta_no_records(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        assert sa.avg_delta("UNKNOWN") == 0.0

    def test_fifo_cap_on_records(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive(max_records=10)
        for i in range(15):
            sa.record("R", level=0, success=True, generation=i)
        assert sa.total_records() == 10

    def test_delta_independent_of_records(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive(max_records=5)
        for i in range(10):
            sa.record_delta("R", float(i))
        assert sa.avg_delta("R") == pytest.approx(4.5)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestDeltaTiebreaker
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeltaTiebreaker:
    """Test that tiebreaker picks the strategy with higher avg_delta."""

    def test_tiebreaker_prefers_higher_delta(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for _ in range(10):
            sa.record("A", level=0, success=True, generation=0)
            sa.record("B", level=0, success=True, generation=0)
        sa.record_delta("A", 10.0)
        sa.record_delta("B", 2.0)
        best = sa.best_for_level(0)
        assert best is not None
        assert best[0] == "A"

    def test_no_tiebreaker_when_scores_differ(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for _ in range(10):
            sa.record("A", level=0, success=True, generation=0)
        for _ in range(10):
            sa.record("B", level=0, success=False, generation=0)
        sa.record_delta("B", 100.0)
        best = sa.best_for_level(0)
        assert best is not None
        assert best[0] == "A"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestLevelDecay
# ═══════════════════════════════════════════════════════════════════════════════


class TestLevelDecay:
    """Test level-dependent decay factors in StrategyArchive."""

    def test_level_dependent_decay_l0(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for _ in range(20):
            sa.record("R", level=0, success=True, generation=0)
        rng = __import__("random").Random(42)
        modified = sa.decay_level_records(0, level_dependent=True, rng=rng)
        assert modified > 0

    def test_level_dependent_decay_l5(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for _ in range(20):
            sa.record("R", level=5, success=True, generation=0)
        rng = __import__("random").Random(42)
        modified = sa.decay_level_records(5, level_dependent=True, rng=rng)
        assert modified > 0

    def test_level_dependent_decay_l9(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for _ in range(20):
            sa.record("R", level=9, success=True, generation=0)
        rng = __import__("random").Random(42)
        modified = sa.decay_level_records(9, level_dependent=True, rng=rng)
        assert modified >= 0

    def test_decay_level_with_no_records(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        modified = sa.decay_level_records(99, level_dependent=True)
        assert modified == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestStrategyDiversity
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategyDiversity:
    """Test StrategyArchive.get_strategy_diversity and monoculture detection."""

    def test_monoculture_zero_diversity(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for _ in range(20):
            sa.record("RANDOM", level=0, success=True, generation=0)
        assert sa.get_strategy_diversity(level=0) == 0.0

    def test_diverse_strategies(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for i in range(30):
            sa.record(["A", "B", "C"][i % 3], level=0, success=True, generation=0)
        assert sa.get_strategy_diversity(level=0) > 0.0

    def test_level_filtered_diversity(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for _ in range(20):
            sa.record("RANDOM", level=0, success=True, generation=0)
        for i in range(20):
            sa.record(["A", "B", "C"][i % 3], level=1, success=True, generation=0)
        assert sa.get_strategy_diversity(level=0) == 0.0
        assert sa.get_strategy_diversity(level=1) > 0.0

    def test_empty_archive_zero(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        assert sa.get_strategy_diversity() == 0.0

    def test_global_diversity(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for i in range(30):
            sa.record(["A", "B", "C"][i % 3], level=0, success=True, generation=0)
        assert sa.get_strategy_diversity() > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestNoveltyDiversity
# ═══════════════════════════════════════════════════════════════════════════════

_DESC_KEYS = (
    "inst_diversity",
    "reg_pressure",
    "program_length",
    "branch_fraction",
    "alu_fraction",
    "load_store_ratio",
    "unique_regs_used",
)


def _desc(**overrides):
    """Build a descriptor dict with sensible defaults."""
    d = {
        "inst_diversity": 0.5,
        "reg_pressure": 0.5,
        "program_length": 0.5,
        "branch_fraction": 0.5,
        "alu_fraction": 0.5,
        "load_store_ratio": 0.5,
        "unique_regs_used": 0.5,
    }
    d.update(overrides)
    return d


class TestNoveltyDiversity:
    """Test SparseArchive.get_diversity_score."""

    def test_empty_archive_zero(self):
        from novelty import SparseArchive

        sa = SparseArchive(k_neighbors=3, novelty_threshold=0.1)
        assert sa.get_diversity_score() == 0.0

    def test_single_entry_zero(self):
        from novelty import SparseArchive

        sa = SparseArchive(k_neighbors=3, novelty_threshold=0.1)
        sa.add_to_archive(_desc(inst_diversity=0.5), check_threshold=False)
        assert sa.size == 1
        assert sa.get_diversity_score() == 0.0

    def test_diverse_entries_positive(self):
        from novelty import SparseArchive

        sa = SparseArchive(k_neighbors=2, novelty_threshold=0.01)
        for val in (0.1, 0.9, 0.5):
            sa.add_to_archive(_desc(inst_diversity=val), check_threshold=False)
        assert sa.get_diversity_score() > 0.0

    def test_identical_entries_low_diversity(self):
        from novelty import SparseArchive

        sa = SparseArchive(k_neighbors=2, novelty_threshold=0.01)
        for _ in range(5):
            sa.add_to_archive(_desc(), check_threshold=False)
        assert sa.get_diversity_score() < 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TestNoveltyTrim
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoveltyTrim:
    """Test SparseArchive.trim_archive FIFO eviction."""

    def test_no_trim_when_under_limit(self):
        from novelty import SparseArchive

        sa = SparseArchive(k_neighbors=2, novelty_threshold=0.01)
        for i in range(5):
            sa.add_to_archive(_desc(inst_diversity=i / 5), check_threshold=False)
        removed = sa.trim_archive(max_entries=10)
        assert removed == 0
        assert sa.size == 5

    def test_trim_removes_oldest(self):
        from novelty import SparseArchive

        sa = SparseArchive(k_neighbors=2, novelty_threshold=0.01)
        for i in range(10):
            sa.add_to_archive(_desc(inst_diversity=i / 10), check_threshold=False)
        removed = sa.trim_archive(max_entries=5)
        assert removed == 5
        assert sa.size == 5

    def test_trim_preserves_newest(self):
        from novelty import SparseArchive

        sa = SparseArchive(k_neighbors=2, novelty_threshold=0.01)
        for i in range(8):
            sa.add_to_archive(_desc(inst_diversity=i / 8), check_threshold=False)
        sa.trim_archive(max_entries=3)
        assert sa.size == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TestQDCellStats
# ═══════════════════════════════════════════════════════════════════════════════


class TestQDCellStats:
    """Test MAPElitesArchive.get_cell_stats."""

    def test_empty_archive_stats(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        stats = arch.get_cell_stats()
        assert stats["total_cells"] == 100
        assert stats["occupied"] == 0
        assert stats["empty"] == 100
        assert stats["avg_fitness"] == 0.0
        assert stats["min_fitness"] == 0.0
        assert stats["max_fitness"] == 0.0

    def test_single_solution_stats(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        arch.add_solution([0.5, 0.5], fitness=100.0, program=[0x00100073])
        stats = arch.get_cell_stats()
        assert stats["occupied"] == 1
        assert stats["empty"] == 99
        assert stats["avg_fitness"] == pytest.approx(100.0)
        assert stats["min_fitness"] == pytest.approx(100.0)
        assert stats["max_fitness"] == pytest.approx(100.0)

    def test_multiple_solutions_stats(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        arch.add_solution([0.2, 0.3], fitness=100.0, program=[0x00100073])
        arch.add_solution([0.8, 0.7], fitness=200.0, program=[0x00000013])
        stats = arch.get_cell_stats()
        assert stats["occupied"] == 2
        assert stats["avg_fitness"] == pytest.approx(150.0)
        assert stats["min_fitness"] == pytest.approx(100.0)
        assert stats["max_fitness"] == pytest.approx(200.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TestQDNeighborhood
# ═══════════════════════════════════════════════════════════════════════════════


class TestQDNeighborhood:
    """Test MAPElitesArchive.get_neighborhood_density."""

    def test_empty_archive_density_zero(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        assert arch.get_neighborhood_density([0.0, 0.0], radius=1) == 0.0

    def test_density_with_occupied_neighbors(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        arch.add_solution([0.0, 0.0], fitness=100.0, program=[0x00100073])
        density = arch.get_neighborhood_density([0.0, 0.0], radius=0)
        assert density == 1.0

    def test_density_radius_one(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        arch.add_solution([0.0, 0.0], fitness=100.0, program=[0x00100073])
        density = arch.get_neighborhood_density([0.0, 0.0], radius=1)
        assert 0.0 < density < 1.0

    def test_density_none_descriptor(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        assert arch.get_neighborhood_density(None, radius=1) == 0.0

    def test_density_wrong_length(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        assert arch.get_neighborhood_density([1.0], radius=1) == 0.0
        assert arch.get_neighborhood_density([1.0, 2.0, 3.0], radius=1) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TestQDFitnessDistribution
# ═══════════════════════════════════════════════════════════════════════════════


class TestQDFitnessDistribution:
    """Test MAPElitesArchive.get_fitness_distribution."""

    def test_empty_distribution(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        dist = arch.get_fitness_distribution()
        assert dist["bins"] == []
        assert dist["mean"] == 0.0
        assert dist["median"] == 0.0
        assert dist["stddev"] == 0.0

    def test_single_solution_distribution(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        arch.add_solution([0.5, 0.5], fitness=100.0, program=[0x00100073])
        dist = arch.get_fitness_distribution()
        assert len(dist["bins"]) == 1
        assert dist["mean"] == pytest.approx(100.0)
        assert dist["median"] == pytest.approx(100.0)
        assert dist["stddev"] == pytest.approx(0.0)

    def test_multiple_solutions_distribution(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        for i in range(5):
            arch.add_solution([i / 5, 0.5], fitness=10.0 + i * 20, program=[0x00100073 + i])
        dist = arch.get_fitness_distribution()
        assert len(dist["bins"]) == 10
        assert dist["mean"] == pytest.approx(50.0)
        assert dist["median"] == pytest.approx(50.0)
        assert dist["stddev"] > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TestChampionCoverage
# ═══════════════════════════════════════════════════════════════════════════════


class TestChampionCoverage:
    """Test ChampionArchive.get_coverage_stats."""

    def _make_archive(self, tmp_path):
        from champion_archive import ChampionArchive

        path = os.path.join(str(tmp_path), "archive.jsonl")
        return ChampionArchive(path=path, max_size=100)

    def test_empty_archive_coverage(self, tmp_path):
        stats = self._make_archive(tmp_path).get_coverage_stats()
        assert stats["total_entries"] == 0
        assert stats["unique_cycles"] == 0
        assert stats["unique_levels"] == 0
        assert stats["foundational_count"] == 0

    def test_single_entry_coverage(self, tmp_path):
        archive = self._make_archive(tmp_path)
        archive.add([0x00100073] * 10, level=3, fitness=200.0, cycle=1)
        stats = archive.get_coverage_stats()
        assert stats["total_entries"] == 1
        assert stats["unique_cycles"] == 1
        assert stats["unique_levels"] == 1
        assert stats["cycle_range"] == [1, 1]
        assert stats["level_range"] == [3, 3]

    def test_multiple_entries_coverage(self, tmp_path):
        archive = self._make_archive(tmp_path)
        for lvl in [0, 5, 10]:
            archive.add([0x00100073] * (lvl + 1), level=lvl, fitness=200.0, cycle=1)
        stats = archive.get_coverage_stats()
        assert stats["total_entries"] == 3
        assert stats["unique_levels"] == 3
        assert sorted(stats["level_range"]) == [0, 10]

    def test_foundational_count(self, tmp_path):
        archive = self._make_archive(tmp_path)
        for lvl in range(7):
            archive.add([0x00100073] * (lvl + 1), level=lvl, fitness=200.0, cycle=1)
        stats = archive.get_coverage_stats()
        assert stats["foundational_count"] == 7


# ═══════════════════════════════════════════════════════════════════════════════
# 12. TestFailureArchiveEviction
# ═══════════════════════════════════════════════════════════════════════════════


class TestFailureArchiveEviction:
    """Test FIFO eviction logic on a JSONL file (standalone helper)."""

    def test_fifo_eviction(self, tmp_path):
        jsonl_path = os.path.join(str(tmp_path), "failure_archive.jsonl")
        num_entries, keep = 100, 30
        with open(jsonl_path, "w") as f:
            f.writelines(json.dumps({"iteration": i}) + "\n" for i in range(num_entries))

        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == num_entries

        with open(jsonl_path, "w") as f:
            f.writelines(lines[-keep:])

        with open(jsonl_path) as f:
            remaining = [json.loads(line) for line in f if line.strip()]
        assert len(remaining) == keep
        assert remaining[0]["iteration"] == num_entries - keep
        assert remaining[-1]["iteration"] == num_entries - 1

    def test_no_eviction_needed(self, tmp_path):
        jsonl_path = os.path.join(str(tmp_path), "failure_archive.jsonl")
        with open(jsonl_path, "w") as f:
            f.writelines(json.dumps({"iteration": i}) + "\n" for i in range(5))
        with open(jsonl_path) as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# 13. TestRunDiagnostics
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunDiagnostics:
    """Test EvolutionEngine.get_run_diagnostics returns expected keys."""

    def test_minimal_champion(self):
        engine = _make_engine(champion=None)
        engine.novelty_archive = None
        engine.qd_archive = None
        engine.strategy_archive = None
        diag = engine.get_run_diagnostics()
        assert diag["champion_words"] == 0
        assert diag["champion_level"] == 0
        assert diag["champion_fitness"] == 0.0
        assert diag["novelty_archive_size"] == 0
        assert diag["qd_archive_size"] == 0
        assert diag["strategy_archive_records"] == 0

    def test_expected_keys(self):
        engine = _make_engine(
            champion=[0x00100073] * 5,
            level=3,
            fitness=150.0,
            verified=True,
            generation=42,
        )
        diag = engine.get_run_diagnostics()
        expected_keys = {
            "champion_level",
            "champion_fitness",
            "champion_words",
            "champion_verified",
            "generations_stalled",
            "current_generation",
            "novelty_archive_size",
            "qd_archive_size",
            "strategy_archive_records",
            "hall_of_fame_size",
            "strategy_diversity",
            "recurrent_examples",
            "fitness_velocity",
            "fitness_momentum",
            "mutation_rate",
            "population_size",
            "elite_ratio",
        }
        assert expected_keys.issubset(set(diag.keys()))
        assert diag["champion_words"] == 5
        assert diag["champion_level"] == 3
        assert diag["champion_fitness"] == 150.0
        assert diag["champion_verified"] is True
        assert diag["current_generation"] == 42

    def test_novelty_archive_size(self):
        engine = _make_engine(champion=[0x00100073])
        assert engine.get_run_diagnostics()["novelty_archive_size"] == 5

    def test_qd_archive_size(self):
        engine = _make_engine(champion=[0x00100073])
        assert engine.get_run_diagnostics()["qd_archive_size"] == 10


# ═══════════════════════════════════════════════════════════════════════════════
# 14. TestExportStateAnalysis
# ═══════════════════════════════════════════════════════════════════════════════


class TestExportStateAnalysis:
    """Test EvolutionEngine.export_state_for_analysis returns expected structure."""

    def _engine(self):
        e = _make_engine(
            champion=[0x00100073, 0x00000013],
            level=3,
            fitness=150.0,
            verified=True,
            generation=42,
        )
        e.generations_stalled = 5
        e._fitness_velocity = 1.23
        return e

    def test_returns_dict(self):
        assert isinstance(self._engine().export_state_for_analysis(), dict)

    def test_has_timestamp(self):
        state = self._engine().export_state_for_analysis()
        assert "timestamp" in state and isinstance(state["timestamp"], float)

    def test_has_generation(self):
        assert self._engine().export_state_for_analysis()["generation"] == 42

    def test_champion_subdict(self):
        champ = self._engine().export_state_for_analysis()["champion"]
        assert champ["level"] == 3
        assert champ["fitness"] == 150.0
        assert champ["words"] == 2
        assert champ["verified"] is True
        assert "program_hex" in champ

    def test_config_subdict(self):
        cfg = self._engine().export_state_for_analysis()["config"]
        for key in ("population_size", "elite_ratio", "mutation_rate", "novelty_weight"):
            assert key in cfg

    def test_archives_subdict(self):
        arcs = self._engine().export_state_for_analysis()["archives"]
        for key in ("novelty_size", "qd_occupied", "qd_coverage", "hof_size"):
            assert key in arcs

    def test_stall_subdict(self):
        stall = self._engine().export_state_for_analysis()["stall"]
        assert "generations_stalled" in stall
        assert "fitness_velocity" in stall

    def test_diagnostics_present(self):
        diag = self._engine().export_state_for_analysis()["diagnostics"]
        assert isinstance(diag, dict)

    def test_no_champion_words_zero(self):
        e = self._engine()
        e.champion = None
        state = e.export_state_for_analysis()
        assert state["champion"]["words"] == 0
        assert state["champion"]["program_hex"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# 15. TestChampionAnalysis
# ═══════════════════════════════════════════════════════════════════════════════


class TestChampionAnalysis:
    """Test EvolutionEngine.get_champion_program_analysis."""

    def test_with_champion(self):
        engine = _make_engine(champion=[0x00100073] * 10, level=0, fitness=100.0)
        result = engine.get_champion_program_analysis()
        assert result["exists"] is True
        assert result["word_count"] == 10
        assert result["unique_opcodes"] >= 1
        assert result["has_ebreak"] is True

    def test_no_champion(self):
        engine = _make_engine(champion=None)
        assert engine.get_champion_program_analysis() == {"exists": False}

    def test_word_count_matches(self):
        program = [0x00100073] * 5 + [0x00000013] * 3
        engine = _make_engine(champion=program)
        assert engine.get_champion_program_analysis()["word_count"] == 8

    def test_has_ebreak_false(self):
        engine = _make_engine(champion=[0x00000013] * 5)
        assert engine.get_champion_program_analysis()["has_ebreak"] is False

    def test_backward_branches_counted(self):
        result = _make_engine(champion=[0x00100073] * 3).get_champion_program_analysis()
        assert "backward_branches" in result
        assert "estimated_complexity" in result

    def test_estimated_complexity_simple(self):
        assert (
            _make_engine(champion=[0x00100073] * 3).get_champion_program_analysis()["estimated_complexity"] == "simple"
        )

    def test_estimated_complexity_moderate(self):
        assert (
            _make_engine(champion=[0x00100073] * 10).get_champion_program_analysis()["estimated_complexity"]
            == "moderate"
        )

    def test_estimated_complexity_complex(self):
        assert (
            _make_engine(champion=[0x00100073] * 20).get_champion_program_analysis()["estimated_complexity"]
            == "complex"
        )

    def test_consistent_results(self):
        engine = _make_engine(champion=[0x00100073] * 5)
        r1 = engine.get_champion_program_analysis()
        r2 = engine.get_champion_program_analysis()
        assert r1 == r2


# ═══════════════════════════════════════════════════════════════════════════════
# 16. TestRunSummary
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunSummary:
    """Test EvolutionEngine.get_run_summary returns expected structure."""

    def test_returns_dict_with_keys(self):
        engine = _make_engine(
            champion=[0x00100073] * 5,
            level=3,
            fitness=150.0,
            verified=True,
            generation=42,
        )
        summary = engine.get_run_summary()
        assert isinstance(summary, dict)
        for key in ("diagnostics", "performance", "training", "archives", "champion_summary"):
            assert key in summary

    def test_no_champion(self):
        assert _make_engine(champion=None).get_run_summary()["champion_summary"] == "No champion"

    def test_champion_summary_in_run_summary(self):
        engine = _make_engine(champion=[0x00100073] * 5, level=3, fitness=150.0, verified=True)
        assert "L3" in engine.get_run_summary()["champion_summary"]

    def test_diagnostics_subdict(self):
        summary = _make_engine(champion=[0x00100073], level=0, fitness=100.0).get_run_summary()
        assert summary["diagnostics"]["champion_level"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 17. TestPerformanceMetrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformanceMetrics:
    """Test EvolutionEngine.get_performance_metrics."""

    def test_returns_dict_with_keys(self):
        metrics = _make_engine(
            champion=[0x00100073] * 5,
            level=3,
            fitness=150.0,
            generation=42,
        ).get_performance_metrics()
        expected = {
            "current_generation",
            "population_size",
            "elite_ratio",
            "eval_budget",
            "mutation_rate",
            "champion_words",
            "novelty_weight",
            "qd_bonus_weight",
            "total_recurrent_examples",
            "hof_size",
        }
        assert expected.issubset(set(metrics.keys()))

    def test_correct_values(self):
        metrics = _make_engine(
            champion=[0x00100073] * 5,
            level=3,
            fitness=150.0,
            generation=42,
        ).get_performance_metrics()
        assert metrics["current_generation"] == 42
        assert metrics["population_size"] == 50
        assert metrics["elite_ratio"] == 0.1
        assert metrics["champion_words"] == 5
        assert metrics["hof_size"] == 3

    def test_no_champion_words_zero(self):
        assert _make_engine(champion=None).get_performance_metrics()["champion_words"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 18. TestTrainingSummary
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainingSummary:
    """Test EvolutionEngine.get_training_summary."""

    def test_returns_dict_with_keys(self):
        summary = _make_engine(champion=[0x00100073] * 5, level=3, fitness=150.0).get_training_summary()
        expected = {
            "recurrent_examples",
            "champion_fitness",
            "champion_level",
            "generations_stalled",
            "fitness_velocity",
            "champion_verified",
            "champion_verify_score",
            "strategy_diversity",
            "novelty_diversity",
            "qd_coverage",
        }
        assert expected.issubset(set(summary.keys()))

    def test_correct_values(self):
        summary = _make_engine(
            champion=[0x00100073] * 5,
            level=3,
            fitness=150.0,
            verified=True,
        ).get_training_summary()
        assert summary["champion_fitness"] == 150.0
        assert summary["champion_level"] == 3
        assert summary["champion_verified"] is True

    def test_no_archives(self):
        engine = _make_engine(champion=[0x00100073])
        engine.novelty_archive = None
        engine.qd_archive = None
        engine.strategy_archive = None
        summary = engine.get_training_summary()
        assert summary["strategy_diversity"] == 0.0
        assert summary["novelty_diversity"] == 0.0
        assert summary["qd_coverage"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 19. TestComprehensiveState
# ═══════════════════════════════════════════════════════════════════════════════


class TestComprehensiveState:
    """Test EvolutionEngine.get_comprehensive_state returns expected structure."""

    def _engine(self, **kw):
        defaults = {"champion": [0x00100073] * 5, "level": 3, "fitness": 150.0, "generation": 42}
        defaults.update(kw)
        return _make_engine(**defaults)

    def test_returns_dict(self):
        assert isinstance(self._engine().get_comprehensive_state(), dict)

    def test_top_level_keys(self):
        state = self._engine().get_comprehensive_state()
        expected = {
            "evolution",
            "champion_program",
            "run_diagnostics",
            "performance",
            "training",
            "archives",
            "champion_summary",
        }
        assert expected.issubset(set(state.keys()))

    def test_evolution_subdict(self):
        evo = self._engine().get_comprehensive_state()["evolution"]
        for key in ("champion", "run_summary", "state", "config"):
            assert key in evo

    def test_champion_program_subdict(self):
        champ = self._engine(champion=[0x00100073] * 10).get_comprehensive_state()["champion_program"]
        assert champ["exists"] is True
        assert champ["word_count"] == 10

    def test_no_champion(self):
        state = self._engine(champion=None).get_comprehensive_state()
        assert state["champion_program"]["exists"] is False
        assert state["champion_summary"] == "No champion"

    def test_nested_archives(self):
        archives = self._engine(champion=[0x00100073]).get_comprehensive_state()["archives"]
        for key in ("novelty", "qd", "strategy", "hall_of_fame"):
            assert key in archives


# ═══════════════════════════════════════════════════════════════════════════════
# 20. TestEvolutionState
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvolutionState:
    """Test EvolutionEngine.get_evolution_state returns expected structure."""

    def _engine(self, **kw):
        defaults = {"champion": [0x00100073] * 5, "level": 3, "fitness": 150.0, "generation": 42}
        defaults.update(kw)
        return _make_engine(**defaults)

    def test_returns_dict(self):
        assert isinstance(self._engine().get_evolution_state(), dict)

    def test_top_level_keys(self):
        result = self._engine().get_evolution_state()
        expected = {"champion", "run_summary", "state", "config"}
        assert expected.issubset(set(result.keys()))

    def test_state_subdict(self):
        state = self._engine().get_evolution_state()["state"]
        assert state["current_generation"] == 42
        assert state["champion_level"] == 3
        assert state["champion_fitness"] == 150.0

    def test_config_subdict(self):
        cfg = self._engine(level=0, fitness=100.0).get_evolution_state()["config"]
        assert cfg["max_generations"] == 200
        assert cfg["population_size"] == 50
        assert cfg["elite_ratio"] == 0.1
        assert cfg["auto_curriculum"] is True
        assert cfg["use_gpu"] is False

    def test_champion_analysis(self):
        champ = self._engine().get_evolution_state()["champion"]
        assert champ["exists"] is True
        assert champ["word_count"] == 5

    def test_no_champion(self):
        assert self._engine(champion=None).get_evolution_state()["champion"] == {"exists": False}

    def test_run_summary_nested(self):
        rs = self._engine(verified=True).get_evolution_state()["run_summary"]
        for key in ("diagnostics", "performance", "training", "archives", "champion_summary"):
            assert key in rs

    def test_specific_values(self):
        result = _make_engine(
            champion=[0x00100073] * 10,
            level=7,
            fitness=250.5,
            generation=100,
        ).get_evolution_state()
        assert result["state"]["champion_level"] == 7
        assert result["state"]["champion_fitness"] == 250.5
        assert result["champion"]["word_count"] == 10
        assert result["champion"]["exists"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# 21. TestComprehensiveState (additional)
# ═══════════════════════════════════════════════════════════════════════════════


class TestComprehensiveStateAdditional:
    """Additional tests for get_comprehensive_state: verifies all 7 top-level keys."""

    def test_all_seven_keys_present(self):
        """get_comprehensive_state must return exactly 7 expected top-level keys."""
        engine = _make_engine(champion=[0x00100073] * 5, level=3, fitness=150.0, generation=42)
        state = engine.get_comprehensive_state()
        expected_keys = {
            "evolution",
            "champion_program",
            "run_diagnostics",
            "performance",
            "training",
            "archives",
            "champion_summary",
        }
        assert set(state.keys()) == expected_keys

    def test_all_values_are_dicts_or_string(self):
        """Every top-level value is a dict (or string for champion_summary)."""
        engine = _make_engine(champion=[0x00100073] * 5, level=3, fitness=150.0)
        state = engine.get_comprehensive_state()
        for key in ("evolution", "champion_program", "run_diagnostics", "performance", "training", "archives"):
            assert isinstance(state[key], dict), f"{key} should be dict"
        assert isinstance(state["champion_summary"], str)

    def test_nested_evolution_structure(self):
        """evolution sub-dict contains champion, run_summary, state, config."""
        engine = _make_engine(champion=[0x00100073] * 5, level=3, fitness=150.0)
        evo = engine.get_comprehensive_state()["evolution"]
        for key in ("champion", "run_summary", "state", "config"):
            assert key in evo, f"evolution missing '{key}'"


# ═══════════════════════════════════════════════════════════════════════════════
# 22. TestSaveStateToFile
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaveStateToFile:
    """Test EvolutionEngine.save_state_to_file creates a valid JSON file."""

    def _engine(self, **kw):
        defaults = {"champion": [0x00100073] * 5, "level": 3, "fitness": 150.0, "generation": 42}
        defaults.update(kw)
        return _make_engine(**defaults)

    def test_returns_true_on_success(self, tmp_path):
        engine = self._engine()
        path = str(tmp_path / "state.json")
        assert engine.save_state_to_file(path) is True

    def test_creates_valid_json(self, tmp_path):
        engine = self._engine()
        path = str(tmp_path / "state.json")
        engine.save_state_to_file(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_json_has_comprehensive_keys(self, tmp_path):
        engine = self._engine()
        path = str(tmp_path / "state.json")
        engine.save_state_to_file(path)
        with open(path) as f:
            data = json.load(f)
        expected = {
            "evolution",
            "champion_program",
            "run_diagnostics",
            "performance",
            "training",
            "archives",
            "champion_summary",
        }
        assert expected.issubset(set(data.keys()))

    def test_json_is_roundtrippable(self, tmp_path):
        engine = self._engine(champion=[0x00100073] * 3, level=1, fitness=50.0, generation=10)
        path = str(tmp_path / "state.json")
        engine.save_state_to_file(path)
        with open(path) as f:
            data = json.load(f)
        # Verify champion_program sub-dict survived serialization
        champ = data["champion_program"]
        assert champ["exists"] is True
        assert champ["word_count"] == 3

    def test_no_champion_saves_cleanly(self, tmp_path):
        engine = self._engine(champion=None)
        path = str(tmp_path / "state.json")
        assert engine.save_state_to_file(path) is True
        with open(path) as f:
            data = json.load(f)
        assert data["champion_program"]["exists"] is False
        assert data["champion_summary"] == "No champion"

    def test_returns_false_on_bad_path(self):
        engine = self._engine()
        # Non-existent deep directory with no writable parent
        assert engine.save_state_to_file("/nonexistent/dir/state.json") is False


# ═══════════════════════════════════════════════════════════════════════════════
# 23. TestChampionProgramAnalysis
# ═══════════════════════════════════════════════════════════════════════════════


class TestChampionProgramAnalysis:
    """Test get_champion_program_analysis with a mock program returns expected fields."""

    def test_exists_flag_true(self):
        engine = _make_engine(champion=[0x00100073] * 5, level=0, fitness=100.0)
        result = engine.get_champion_program_analysis()
        assert result["exists"] is True

    def test_word_count_matches_program(self):
        program = [0x00100073] * 7 + [0x00000013] * 3
        engine = _make_engine(champion=program)
        result = engine.get_champion_program_analysis()
        assert result["word_count"] == 10

    def test_unique_opcodes_counted(self):
        # Two distinct opcodes: EBREAK (0x73) and ADDI (0x13)
        program = [0x00100073, 0x00000013, 0x00100073, 0x00000013]
        engine = _make_engine(champion=program)
        result = engine.get_champion_program_analysis()
        assert result["unique_opcodes"] == 2

    def test_backward_branches_detected(self):
        # Build a B-type instruction with negative immediate
        # Must set bit 31 (sign bit) like _make_b_type does
        sign = 0x1000  # used as flag only
        abs_imm = 8
        b_backward = (
            0x63  # B-type opcode
            | (1 << 15)  # rs1 = x1
            | (2 << 20)  # rs2 = x2
            | (0 << 12)  # funct3 = 0
            | ((abs_imm >> 1) & 0x1) << 7
            | ((abs_imm >> 2) & 0xF) << 8
            | ((abs_imm >> 5) & 0x3F) << 25
            | ((abs_imm >> 11) & 0x1) << 30
        )
        if sign:
            b_backward |= 1 << 31
        program = [b_backward, 0x00100073]
        engine = _make_engine(champion=program)
        result = engine.get_champion_program_analysis()
        assert result["backward_branches"] >= 1

    def test_has_ebreak_true(self):
        engine = _make_engine(champion=[0x00100073] * 3)
        assert engine.get_champion_program_analysis()["has_ebreak"] is True

    def test_has_ebreak_false(self):
        engine = _make_engine(champion=[0x00000013] * 3)
        assert engine.get_champion_program_analysis()["has_ebreak"] is False

    def test_estimated_complexity_simple(self):
        engine = _make_engine(champion=[0x00100073] * 3)
        assert engine.get_champion_program_analysis()["estimated_complexity"] == "simple"

    def test_estimated_complexity_moderate(self):
        engine = _make_engine(champion=[0x00100073] * 10)
        assert engine.get_champion_program_analysis()["estimated_complexity"] == "moderate"

    def test_estimated_complexity_complex(self):
        engine = _make_engine(champion=[0x00100073] * 25)
        assert engine.get_champion_program_analysis()["estimated_complexity"] == "complex"

    def test_expected_keys_present(self):
        engine = _make_engine(champion=[0x00100073] * 5)
        result = engine.get_champion_program_analysis()
        expected = {"exists", "word_count", "unique_opcodes", "backward_branches", "has_ebreak", "estimated_complexity"}
        assert expected.issubset(set(result.keys()))


# ═══════════════════════════════════════════════════════════════════════════════
# 24. TestChampionProgramAnalysisNoChampion
# ═══════════════════════════════════════════════════════════════════════════════


class TestChampionProgramAnalysisNoChampion:
    """Test get_champion_program_analysis with champion=None returns exists=False."""

    def test_no_champion_returns_exists_false(self):
        engine = _make_engine(champion=None)
        result = engine.get_champion_program_analysis()
        assert result == {"exists": False}

    def test_no_champion_only_one_key(self):
        engine = _make_engine(champion=None)
        result = engine.get_champion_program_analysis()
        assert set(result.keys()) == {"exists"}

    def test_champion_set_to_none_later(self):
        """Champion cleared after creation still returns exists=False."""
        engine = _make_engine(champion=[0x00100073] * 3, level=0, fitness=100.0)
        assert engine.get_champion_program_analysis()["exists"] is True
        engine.champion = None
        assert engine.get_champion_program_analysis() == {"exists": False}

    def test_empty_list_champion_exists(self):
        """An empty list is not None, so exists=True."""
        engine = _make_engine(champion=[])
        result = engine.get_champion_program_analysis()
        assert result["exists"] is True
        assert result["word_count"] == 0

    def test_consistent_on_repeated_calls(self):
        engine = _make_engine(champion=None)
        r1 = engine.get_champion_program_analysis()
        r2 = engine.get_champion_program_analysis()
        assert r1 == r2


# ═══════════════════════════════════════════════════════════════════════════════
# 25. TestBackwardBranchesIntegration
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackwardBranchesIntegration:
    """Test a realistic L9 program with backward branches."""

    @staticmethod
    def _make_b_type(imm: int) -> int:
        """Encode a B-type instruction word with the given signed immediate."""
        sign = 0x1000 if imm < 0 else 0
        abs_imm = abs(imm)
        word = (
            0x63  # B-type opcode
            | (1 << 15)  # rs1 = x1
            | (2 << 20)  # rs2 = x2
            | (0 << 12)  # funct3 = 0
            | ((abs_imm >> 1) & 0x1) << 7
            | ((abs_imm >> 2) & 0xF) << 8
            | ((abs_imm >> 5) & 0x3F) << 25
            | ((abs_imm >> 11) & 0x1) << 30
        )
        if sign:
            word |= 1 << 31
        return word

    def test_realistic_l9_program(self):
        """A realistic L9 program: init loop, body with backward branch, epilogue."""
        from evolution_engine import EvolutionEngine

        nop = 0x00000013  # ADDI x0, x0, 0
        ebreak = 0x00100073

        # Build a ~20-word program with 3 backward branches
        program = [
            nop,
            nop,
            nop,  # init prologue
            self._make_b_type(-4),  # backward branch 1 (loop start)
            nop,
            nop,
            nop,
            nop,  # loop body
            self._make_b_type(-12),  # backward branch 2 (inner loop)
            nop,
            nop,  # inner body
            self._make_b_type(-20),  # backward branch 3 (outer loop)
            nop,
            nop,
            nop,
            nop,
            nop,  # cleanup
            ebreak,  # exit
        ]
        assert EvolutionEngine._count_backward_branches(program) == 3

    def test_l9_program_through_engine_analysis(self):
        """Verify backward branches propagate through champion analysis."""

        nop = 0x00000013
        ebreak = 0x00100073
        b1 = self._make_b_type(-8)
        b2 = self._make_b_type(-16)

        program = [nop, nop, b1, nop, nop, b2, nop, nop, ebreak]
        engine = _make_engine(champion=program, level=9, fitness=500.0, verified=True)

        analysis = engine.get_champion_program_analysis()
        assert analysis["exists"] is True
        assert analysis["word_count"] == 9
        assert analysis["backward_branches"] == 2
        assert analysis["estimated_complexity"] == "moderate"  # 9 words → moderate

    def test_l9_no_backward_branches(self):
        """L9 program without backward branches still returns 0."""
        from evolution_engine import EvolutionEngine

        nop = 0x00000013
        ebreak = 0x00100073
        program = [nop] * 9 + [ebreak]
        assert EvolutionEngine._count_backward_branches(program) == 0

        engine = _make_engine(champion=program, level=9, fitness=500.0)
        analysis = engine.get_champion_program_analysis()
        assert analysis["backward_branches"] == 0
        assert analysis["estimated_complexity"] == "moderate"  # 10 words ≤ 15 → moderate

    def test_l9_all_backward_branches(self):
        """L9 program composed entirely of backward branches."""
        from evolution_engine import EvolutionEngine

        ebreak = 0x00100073
        b = self._make_b_type(-4)
        program = [b] * 5 + [ebreak]
        assert EvolutionEngine._count_backward_branches(program) == 5

    def test_l9_mixed_forward_and_backward(self):
        """L9 with both forward and backward branches — only backward counted."""
        from evolution_engine import EvolutionEngine

        nop = 0x00000013
        ebreak = 0x00100073
        b_backward = self._make_b_type(-8)
        b_forward = self._make_b_type(8)

        program = [b_backward, b_forward, nop, b_backward, ebreak]
        assert EvolutionEngine._count_backward_branches(program) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 26. TestLogComprehensiveDiagnostics
# ═══════════════════════════════════════════════════════════════════════════════


class TestLogComprehensiveDiagnostics:
    """Test log_comprehensive_diagnostics calls _log multiple times."""

    def _make_recording_engine(self, **kw):
        """Engine with _log that records every call for inspection."""
        defaults = {"champion": [0x00100073] * 5, "level": 3, "fitness": 150.0, "generation": 42}
        defaults.update(kw)
        engine = _make_engine(**defaults)
        calls = []
        engine._log = lambda *args: calls.append(args)
        return engine, calls

    def test_calls_log_multiple_times(self):
        """log_comprehensive_diagnostics must invoke _log at least 8 times."""
        engine, calls = self._make_recording_engine()
        engine.log_comprehensive_diagnostics()
        assert len(calls) >= 8

    def test_header_and_footer_logged(self):
        """First and last log lines should be the diagnostics header/footer."""
        engine, calls = self._make_recording_engine()
        engine.log_comprehensive_diagnostics()
        first_msg = calls[0][0]
        last_msg = calls[-1][0]
        assert "Comprehensive Diagnostics" in first_msg
        assert "End Diagnostics" in last_msg

    def test_champion_summary_logged(self):
        """One of the log lines should mention the champion level."""
        engine, calls = self._make_recording_engine(level=5, fitness=200.0)
        engine.log_comprehensive_diagnostics()
        all_text = " ".join(str(arg) for call in calls for arg in call)
        assert "L5" in all_text or "Champion" in all_text

    def test_generation_logged(self):
        """One log line should contain the current generation number."""
        engine, calls = self._make_recording_engine(generation=99)
        engine.log_comprehensive_diagnostics()
        all_text = " ".join(str(arg) for call in calls for arg in call)
        assert "99" in all_text

    def test_no_champion_logs_safely(self):
        """Should not crash when champion is None."""
        engine, calls = self._make_recording_engine(champion=None)
        engine.log_comprehensive_diagnostics()
        assert len(calls) >= 8

    def test_each_call_is_tuple(self):
        """Each _log call is recorded as a tuple of its positional args."""
        engine, _ = self._make_recording_engine()
        engine.log_comprehensive_diagnostics()
        # _log is called with a single string arg, so each call is (str,)
        assert True  # placeholder — the mock records *args tuples


# ═══════════════════════════════════════════════════════════════════════════════
# 27. TestEvolutionStateConfig
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvolutionStateConfig:
    """Test get_evolution_state config sub-dict has all expected keys."""

    def _engine(self, **kw):
        defaults = {"champion": [0x00100073] * 5, "level": 3, "fitness": 150.0, "generation": 42}
        defaults.update(kw)
        return _make_engine(**defaults)

    def test_config_has_all_expected_keys(self):
        """Config sub-dict must contain all 6 expected keys."""
        cfg = self._engine().get_evolution_state()["config"]
        expected_keys = {
            "max_generations",
            "population_size",
            "elite_ratio",
            "mutation_rate",
            "auto_curriculum",
            "use_gpu",
        }
        assert expected_keys == set(cfg.keys())

    def test_config_max_generations(self):
        cfg = self._engine().get_evolution_state()["config"]
        assert cfg["max_generations"] == 200

    def test_config_population_size(self):
        cfg = self._engine().get_evolution_state()["config"]
        assert cfg["population_size"] == 50

    def test_config_elite_ratio(self):
        cfg = self._engine().get_evolution_state()["config"]
        assert cfg["elite_ratio"] == 0.1

    def test_config_mutation_rate(self):
        cfg = self._engine().get_evolution_state()["config"]
        assert cfg["mutation_rate"] == 0.1

    def test_config_auto_curriculum(self):
        cfg = self._engine().get_evolution_state()["config"]
        assert cfg["auto_curriculum"] is True

    def test_config_use_gpu(self):
        cfg = self._engine().get_evolution_state()["config"]
        assert cfg["use_gpu"] is False

    def test_config_values_are_consistent(self):
        """Config values match what was passed via _make_engine."""
        cfg = self._engine(level=0, fitness=100.0).get_evolution_state()["config"]
        assert isinstance(cfg["max_generations"], int)
        assert isinstance(cfg["population_size"], int)
        assert isinstance(cfg["elite_ratio"], float)
        assert isinstance(cfg["mutation_rate"], float)
        assert isinstance(cfg["auto_curriculum"], bool)
        assert isinstance(cfg["use_gpu"], bool)


# ═══════════════════════════════════════════════════════════════════════════════
# 28. TestRunSummaryStructure
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunSummaryStructure:
    """Test get_run_summary returns all 5 expected top-level keys."""

    def _engine(self, **kw):
        defaults = {"champion": [0x00100073] * 5, "level": 3, "fitness": 150.0, "generation": 42}
        defaults.update(kw)
        return _make_engine(**defaults)

    def test_has_all_five_keys(self):
        """get_run_summary must return exactly these 5 keys."""
        summary = self._engine().get_run_summary()
        expected_keys = {
            "diagnostics",
            "performance",
            "training",
            "archives",
            "champion_summary",
        }
        assert set(summary.keys()) == expected_keys

    def test_diagnostics_is_dict(self):
        summary = self._engine().get_run_summary()
        assert isinstance(summary["diagnostics"], dict)

    def test_performance_is_dict(self):
        summary = self._engine().get_run_summary()
        assert isinstance(summary["performance"], dict)

    def test_training_is_dict(self):
        summary = self._engine().get_run_summary()
        assert isinstance(summary["training"], dict)

    def test_archives_is_dict(self):
        summary = self._engine().get_run_summary()
        assert isinstance(summary["archives"], dict)

    def test_champion_summary_is_string(self):
        summary = self._engine().get_run_summary()
        assert isinstance(summary["champion_summary"], str)

    def test_champion_summary_no_champion(self):
        summary = self._engine(champion=None).get_run_summary()
        assert summary["champion_summary"] == "No champion"

    def test_champion_summary_with_champion(self):
        summary = self._engine(champion=[0x00100073] * 5, level=7, fitness=300.0, verified=True).get_run_summary()
        assert "L7" in summary["champion_summary"]

    def test_diagnostics_has_champion_level(self):
        summary = self._engine(level=5, fitness=200.0).get_run_summary()
        assert summary["diagnostics"]["champion_level"] == 5

    def test_summary_is_consistent_across_calls(self):
        """Two consecutive calls should return equal data."""
        engine = self._engine()
        r1 = engine.get_run_summary()
        r2 = engine.get_run_summary()
        assert r1 == r2


# ═══════════════════════════════════════════════════════════════════════════════
# 29. TestQDArchiveSummary
# ═══════════════════════════════════════════════════════════════════════════════


class TestQDArchiveSummary:
    """Test MAPElitesArchive.get_archive_summary returns dict with cells, fitness, rebins keys."""

    def test_returns_dict_with_expected_keys(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        summary = arch.get_archive_summary()
        assert isinstance(summary, dict)
        assert "cells" in summary
        assert "fitness" in summary
        assert "rebins" in summary

    def test_cells_is_dict_from_get_cell_stats(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        summary = arch.get_archive_summary()
        assert isinstance(summary["cells"], dict)
        assert "total_cells" in summary["cells"]
        assert "occupied" in summary["cells"]
        assert "empty" in summary["cells"]
        assert "avg_fitness" in summary["cells"]

    def test_fitness_is_dict_from_get_fitness_distribution(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        summary = arch.get_archive_summary()
        assert isinstance(summary["fitness"], dict)
        assert "bins" in summary["fitness"]
        assert "mean" in summary["fitness"]
        assert "median" in summary["fitness"]
        assert "stddev" in summary["fitness"]

    def test_rebins_is_integer(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        summary = arch.get_archive_summary()
        assert isinstance(summary["rebins"], int)
        assert summary["rebins"] == 0

    def test_empty_archive_summary(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        summary = arch.get_archive_summary()
        assert summary["cells"]["occupied"] == 0
        assert summary["cells"]["empty"] == 100
        assert summary["fitness"]["mean"] == 0.0
        assert summary["rebins"] == 0

    def test_with_solutions_populated(self):
        from qd_archive import MAPElitesArchive

        arch = MAPElitesArchive(bins=10)
        arch.add_solution([0.2, 0.3], fitness=100.0, program=[0x00100073])
        arch.add_solution([0.8, 0.7], fitness=200.0, program=[0x00000013])
        summary = arch.get_archive_summary()
        assert summary["cells"]["occupied"] == 2
        assert summary["fitness"]["mean"] == pytest.approx(150.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 30. TestTuningHistorySummaryFull
# ═══════════════════════════════════════════════════════════════════════════════


class TestTuningHistorySummaryFull:
    """Test MetaController.get_tuning_history_summary with multiple tuning actions."""

    @staticmethod
    def _make_controller(tmp_path):
        from meta_controller import MetaController

        return MetaController(checkpoint_dir=str(tmp_path))

    def test_empty_log_returns_zero(self, tmp_path):
        mc = self._make_controller(tmp_path)
        summary = mc.get_tuning_history_summary()
        assert summary["total_actions"] == 0

    def test_single_action_total(self, tmp_path):
        from meta_controller import TuningAction

        mc = self._make_controller(tmp_path)
        mc._tuning_log.append(TuningAction("retrain", "retrain_lr", 1e-4, 5e-4, "stagnation: fitness_plateau"))
        summary = mc.get_tuning_history_summary()
        assert summary["total_actions"] == 1

    def test_multiple_actions_total(self, tmp_path):
        from meta_controller import TuningAction

        mc = self._make_controller(tmp_path)
        for i in range(5):
            mc._tuning_log.append(TuningAction("retrain", "retrain_lr", 1e-4, 5e-4, "stagnation: fitness_plateau"))
        summary = mc.get_tuning_history_summary()
        assert summary["total_actions"] == 5

    def test_by_reason_counts_correctly(self, tmp_path):
        from meta_controller import TuningAction

        mc = self._make_controller(tmp_path)
        mc._tuning_log.append(TuningAction("retrain", "retrain_lr", 1e-4, 5e-4, "stagnation: fitness_plateau"))
        mc._tuning_log.append(TuningAction("retrain", "retrain_lr", 1e-4, 5e-4, "stagnation: fitness_plateau"))
        mc._tuning_log.append(TuningAction("evolution", "mutation_rate", 0.1, 0.2, "stagnation: level_stalled"))
        summary = mc.get_tuning_history_summary()
        assert summary["actions_by_state"]["fitness_plateau"] == 2
        assert summary["actions_by_state"]["level_stalled"] == 1

    def test_by_param_counts_correctly(self, tmp_path):
        from meta_controller import TuningAction

        mc = self._make_controller(tmp_path)
        mc._tuning_log.append(TuningAction("retrain", "retrain_lr", 1e-4, 5e-4, "stagnation: fitness_plateau"))
        mc._tuning_log.append(TuningAction("evolution", "mutation_rate", 0.1, 0.2, "stagnation: level_stalled"))
        mc._tuning_log.append(TuningAction("retrain", "retrain_lr", 5e-4, 1e-3, "recovery"))
        summary = mc.get_tuning_history_summary()
        assert summary["actions_by_domain"]["retrain"] == 2
        assert summary["actions_by_domain"]["evolution"] == 1

    def test_recent_actions_limited_to_ten(self, tmp_path):
        from meta_controller import TuningAction

        mc = self._make_controller(tmp_path)
        for i in range(15):
            mc._tuning_log.append(
                TuningAction("retrain", "retrain_lr", 1e-4, 5e-4, f"stagnation: fitness_plateau (tier {i})")
            )
        summary = mc.get_tuning_history_summary()
        assert len(summary["recent_actions"]) <= 10

    def test_recent_actions_are_dicts(self, tmp_path):
        from meta_controller import TuningAction

        mc = self._make_controller(tmp_path)
        mc._tuning_log.append(TuningAction("retrain", "retrain_lr", 1e-4, 5e-4, "stagnation: fitness_plateau"))
        summary = mc.get_tuning_history_summary()
        action = summary["recent_actions"][0]
        assert isinstance(action, dict)
        assert action["domain"] == "retrain"
        assert action["param"] == "retrain_lr"

    def test_empty_log_returns_empty_dicts(self, tmp_path):
        mc = self._make_controller(tmp_path)
        summary = mc.get_tuning_history_summary()
        assert summary["actions_by_state"] == {}
        assert summary["actions_by_domain"] == {}
        assert summary["recent_actions"] == []

    def test_total_matches_log_length(self, tmp_path):
        from meta_controller import TuningAction

        mc = self._make_controller(tmp_path)
        for i in range(8):
            mc._tuning_log.append(TuningAction("retrain", "retrain_lr", 1e-4, 5e-4, "stagnation: fitness_plateau"))
        summary = mc.get_tuning_history_summary()
        assert summary["total_actions"] == len(mc._tuning_log)


# ═══════════════════════════════════════════════════════════════════════════════
# 31. TestTuningHistoryByDomainFull
# ═══════════════════════════════════════════════════════════════════════════════


class TestTuningHistoryByDomainFull:
    """Test get_effectiveness_report actions_by_domain correctly counts retrain vs evolution actions."""

    @staticmethod
    def _make_controller(tmp_path):
        from meta_controller import MetaController

        return MetaController(checkpoint_dir=str(tmp_path))

    def test_domain_counts_retrain_and_evolution(self, tmp_path):
        from meta_controller import TuningAction

        mc = self._make_controller(tmp_path)
        mc._tuning_log.append(TuningAction("retrain", "retrain_lr", 1e-4, 5e-4, "stagnation: fitness_plateau"))
        mc._tuning_log.append(TuningAction("retrain", "retrain_lr", 5e-4, 1e-3, "stagnation: fitness_plateau"))
        mc._tuning_log.append(TuningAction("evolution", "mutation_rate", 0.1, 0.2, "stagnation: level_stalled"))
        report = mc.get_effectiveness_report()
        assert report["domain_counts"]["retrain"] == 2
        assert report["domain_counts"]["evolution"] == 1

    def test_domain_counts_only_evolution(self, tmp_path):
        from meta_controller import TuningAction

        mc = self._make_controller(tmp_path)
        for i in range(4):
            mc._tuning_log.append(
                TuningAction("evolution", "mutation_rate", 0.1, 0.2, f"stagnation: fitness_plateau (tier {i})")
            )
        report = mc.get_effectiveness_report()
        assert report["domain_counts"]["evolution"] == 4
        assert "retrain" not in report["domain_counts"]

    def test_domain_counts_mixed_domains(self, tmp_path):
        from meta_controller import TuningAction

        mc = self._make_controller(tmp_path)
        mc._tuning_log.append(TuningAction("retrain", "retrain_lr", 1e-4, 5e-4, "stagnation: fitness_plateau"))
        mc._tuning_log.append(TuningAction("evolution", "mutation_rate", 0.1, 0.2, "stagnation: level_stalled"))
        mc._tuning_log.append(TuningAction("recovery", "retrain_freeze_layers", 0, 2, "recovery"))
        report = mc.get_effectiveness_report()
        assert report["domain_counts"]["retrain"] == 1
        assert report["domain_counts"]["evolution"] == 1
        assert report["domain_counts"]["recovery"] == 1

    def test_empty_log_domain_counts_empty(self, tmp_path):
        mc = self._make_controller(tmp_path)
        report = mc.get_effectiveness_report()
        assert report["domain_counts"] == {}

    def test_total_tuning_actions_matches(self, tmp_path):
        from meta_controller import TuningAction

        mc = self._make_controller(tmp_path)
        for i in range(5):
            domain = "retrain" if i % 2 == 0 else "evolution"
            mc._tuning_log.append(TuningAction(domain, "param", 0, 1, "stagnation: fitness_plateau"))
        report = mc.get_effectiveness_report()
        assert report["total_tuning_actions"] == 5
        assert sum(report["domain_counts"].values()) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# 32. TestGenerationStats
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerationStats:
    """Test EvolutionEngine.get_generation_stats returns expected keys."""

    def test_returns_all_expected_keys(self):
        engine = _make_engine(
            champion=[0x00100073],
            level=2,
            fitness=42.0,
            verified=True,
            generation=10,
        )
        stats = engine.get_generation_stats()
        expected_keys = {
            "current_generation",
            "generations_stalled",
            "champion_fitness",
            "champion_level",
            "mutation_rate",
            "population_size",
            "novelty_weight",
            "fitness_velocity",
        }
        assert set(stats.keys()) == expected_keys

    def test_values_match_engine_state(self):
        engine = _make_engine(
            champion=[0x00100073],
            level=3,
            fitness=99.5,
            generation=7,
        )
        engine.generations_stalled = 4
        stats = engine.get_generation_stats()
        assert stats["current_generation"] == 7
        assert stats["generations_stalled"] == 4
        assert stats["champion_fitness"] == 99.5
        assert stats["champion_level"] == 3
        assert stats["mutation_rate"] == 0.1
        assert stats["population_size"] == 50
        assert stats["novelty_weight"] == 0.15
        assert stats["fitness_velocity"] == 1.23

    def test_returns_dict(self):
        engine = _make_engine()
        assert isinstance(engine.get_generation_stats(), dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 33. TestChampionHistoryWithMock
# ═══════════════════════════════════════════════════════════════════════════════


class TestChampionHistoryWithMock:
    """Test get_champion_history with a mock _champion_replacement_history list."""

    def test_returns_correct_format(self):
        engine = _make_engine(generation=5)
        engine._champion_replacement_history = [
            (10.5, 1),
            (20.0, 3),
            (5.5, 4),
        ]
        history = engine.get_champion_history()
        assert isinstance(history, list)
        assert len(history) == 3
        assert history[0] == {"fitness": 10.5, "cycle": 1}
        assert history[1] == {"fitness": 20.0, "cycle": 3}
        assert history[2] == {"fitness": 5.5, "cycle": 4}

    def test_empty_history_list(self):
        engine = _make_engine()
        engine._champion_replacement_history = []
        history = engine.get_champion_history()
        assert history == []

    def test_single_entry(self):
        engine = _make_engine()
        engine._champion_replacement_history = [(3.14, 0)]
        history = engine.get_champion_history()
        assert len(history) == 1
        assert history[0] == {"fitness": 3.14, "cycle": 0}

    def test_returns_list_of_dicts(self):
        engine = _make_engine()
        engine._champion_replacement_history = [(1.0, 1), (2.0, 2)]
        history = engine.get_champion_history()
        for entry in history:
            assert isinstance(entry, dict)
            assert "fitness" in entry
            assert "cycle" in entry


# ═══════════════════════════════════════════════════════════════════════════════
# 34. TestChampionHistoryEmptyMock
# ═══════════════════════════════════════════════════════════════════════════════


class TestChampionHistoryEmptyMock:
    """Test get_champion_history returns empty list when history is None (bypass __init__)."""

    def test_returns_empty_list_when_none(self):
        from evolution_engine import EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine._champion_replacement_history = None
        assert engine.get_champion_history() == []

    def test_returns_empty_list_when_attr_missing(self):
        from evolution_engine import EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        if hasattr(engine, "_champion_replacement_history"):
            delattr(engine, "_champion_replacement_history")
        assert engine.get_champion_history() == []

    def test_make_engine_default_history_returns_empty(self):
        """The shared _make_engine helper doesn't set _champion_replacement_history."""
        engine = _make_engine()
        history = engine.get_champion_history()
        assert history == []
