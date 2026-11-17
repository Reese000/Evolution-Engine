#!/usr/bin/env python3
"""Tests for novelty.py — Novelty Search archive with k-nearest-neighbor scoring.

Covers SparseArchive, extract_behavioral_descriptor, and related utilities.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from novelty import (
    SparseArchive,
    _count_instruction_categories,
    _flatten_descriptor,
    _opcode_category_bits,
    extract_behavioral_descriptor,
)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_descriptor(
    inst_diversity: float = 0.5,
    reg_pressure: float = 0.3,
    program_length: float = 0.2,
    branch_fraction: float = 0.1,
    alu_fraction: float = 0.4,
    load_store_ratio: float = 0.05,
    unique_regs_used: float = 0.2,
) -> dict[str, float]:
    """Create a valid behavioural descriptor."""
    return {
        "inst_diversity": inst_diversity,
        "reg_pressure": reg_pressure,
        "program_length": program_length,
        "branch_fraction": branch_fraction,
        "alu_fraction": alu_fraction,
        "load_store_ratio": load_store_ratio,
        "unique_regs_used": unique_regs_used,
    }


def _make_n_descriptors(n: int, seed: int = 42) -> list[dict[str, float]]:
    """Generate n random descriptors."""
    rng = random.Random(seed)
    return [
        _make_descriptor(
            inst_diversity=rng.random(),
            reg_pressure=rng.random(),
            program_length=rng.random(),
            branch_fraction=rng.random(),
            alu_fraction=rng.random(),
            load_store_ratio=rng.random(),
            unique_regs_used=rng.random(),
        )
        for _ in range(n)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Tests for _flatten_descriptor
# ══════════════════════════════════════════════════════════════════════════════


class TestFlattenDescriptor:
    def test_basic_flatten(self):
        d = _make_descriptor()
        flat = _flatten_descriptor(d)
        assert isinstance(flat, list)
        assert len(flat) == 7  # 7 descriptor keys

    def test_ordering(self):
        d = _make_descriptor()
        flat = _flatten_descriptor(d)
        # Values should be in some consistent order (not necessarily the key order)
        assert len(flat) == 7
        assert all(isinstance(v, (int, float)) for v in flat)


# ══════════════════════════════════════════════════════════════════════════════
# Tests for _opcode_category_bits
# ══════════════════════════════════════════════════════════════════════════════


class TestOpcodeCategoryBits:
    def test_empty_program(self):
        assert _opcode_category_bits([]) == 0

    def test_non_int_ignored(self):
        assert _opcode_category_bits([None, "not_int"]) == 0

    def test_alu_opcode(self):
        # 0x33 = ALU (R-type)
        assert _opcode_category_bits([0x00310033]) & 1  # bit 0 = ALU

    def test_branch_opcode(self):
        # 0x63 = BRANCH
        assert _opcode_category_bits([0x00510063]) & (1 << 1)  # bit 1 = BRANCH

    def test_load_opcode(self):
        # 0x03 = LOAD
        assert _opcode_category_bits([0x00510003]) & (1 << 3)  # bit 3 = LOAD

    def test_store_opcode(self):
        # 0x23 = STORE
        assert _opcode_category_bits([0x00510023]) & (1 << 4)  # bit 4 = STORE


# ══════════════════════════════════════════════════════════════════════════════
# Tests for _count_instruction_categories
# ══════════════════════════════════════════════════════════════════════════════


class TestCountInstructionCategories:
    def test_empty_program(self):
        result = _count_instruction_categories([])
        assert result == (0, 0, 0, 0, set())

    def test_branch_count(self):
        # 0x63 = BRANCH opcode
        result = _count_instruction_categories([0x00510063])
        assert result[0] == 1  # branch_count

    def test_load_count(self):
        # 0x03 = LOAD opcode
        result = _count_instruction_categories([0x00510003])
        assert result[1] == 1  # load_count

    def test_store_count(self):
        # 0x23 = STORE opcode
        result = _count_instruction_categories([0x00510023])
        assert result[2] == 1  # store_count

    def test_non_int_ignored(self):
        result = _count_instruction_categories([None, "not_int"])
        assert result == (0, 0, 0, 0, set())


# ══════════════════════════════════════════════════════════════════════════════
# Tests for SparseArchive
# ══════════════════════════════════════════════════════════════════════════════


class TestSparseArchiveInit:
    def test_default_init(self):
        archive = SparseArchive()
        assert archive.size == 0
        assert archive.k_neighbors == 5
        assert archive.novelty_threshold == 0.15

    def test_custom_init(self):
        archive = SparseArchive(k_neighbors=3, novelty_threshold=0.2)
        assert archive.k_neighbors == 3
        assert archive.novelty_threshold == 0.2


class TestSparseArchiveAdd:
    def test_add_first_entry(self):
        archive = SparseArchive()
        desc = _make_descriptor()
        added, _score = archive.add_to_archive(desc, metadata={"program": [1, 2, 3]})
        assert added is True
        assert archive.size == 1

    def test_add_with_check_threshold(self):
        archive = SparseArchive(novelty_threshold=0.5)
        desc = _make_descriptor()
        added, _score = archive.add_to_archive(desc, check_threshold=True)
        assert added is True  # First entry always added

    def test_add_without_check_threshold(self):
        archive = SparseArchive(novelty_threshold=0.5)
        desc = _make_descriptor()
        added, _score = archive.add_to_archive(desc, check_threshold=False)
        assert added is True

    def test_add_duplicate_descriptor(self):
        archive = SparseArchive(novelty_threshold=0.0)
        desc = _make_descriptor()
        archive.add_to_archive(desc, metadata={"program": [1]})
        _added, _score = archive.add_to_archive(desc, metadata={"program": [2]})
        # Duplicate may be added depending on threshold logic
        assert archive.size >= 1

    def test_add_multiple_descriptors(self):
        archive = SparseArchive(novelty_threshold=0.0)
        descs = _make_n_descriptors(10)
        for d in descs:
            archive.add_to_archive(d)
        assert archive.size == 10

    def test_add_with_metadata_none(self):
        archive = SparseArchive()
        desc = _make_descriptor()
        added, _score = archive.add_to_archive(desc, metadata=None)
        assert added is True

    def test_add_returns_score_tuple(self):
        archive = SparseArchive()
        desc = _make_descriptor()
        result = archive.add_to_archive(desc)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestSparseArchiveNoveltyScore:
    def test_empty_archive_returns_zero(self):
        archive = SparseArchive()
        desc = _make_descriptor()
        score = archive.novelty_score(desc)
        assert score == 0.0

    def test_score_after_add(self):
        archive = SparseArchive()
        archive.add_to_archive(_make_descriptor(inst_diversity=0.1))
        score = archive.novelty_score(_make_descriptor(inst_diversity=0.9))
        assert score > 0.0

    def test_score_similar_descriptor(self):
        archive = SparseArchive()
        archive.add_to_archive(_make_descriptor(inst_diversity=0.5))
        score = archive.novelty_score(_make_descriptor(inst_diversity=0.501))
        assert score < 0.1


class TestSparseArchiveDiversityScore:
    def test_empty_archive(self):
        archive = SparseArchive()
        diversity = archive.get_diversity_score()
        assert diversity == 0.0

    def test_nonempty_archive(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(5):
            archive.add_to_archive(d)
        diversity = archive.get_diversity_score()
        assert 0.0 <= diversity <= 1.0


class TestSparseArchiveRecentScores:
    def test_empty_archive(self):
        archive = SparseArchive()
        scores = archive.recent_novelty_scores(5)
        assert scores == []

    def test_with_entries(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(10):
            archive.add_to_archive(d)
        scores = archive.recent_novelty_scores(5)
        assert len(scores) <= 5


class TestSparseArchiveTrim:
    def test_trim_empty_archive(self):
        archive = SparseArchive()
        trimmed = archive.trim_archive(max_entries=10)
        assert trimmed == 0

    def test_trim_keeps_entries(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(10):
            archive.add_to_archive(d)
        trimmed = archive.trim_archive(max_entries=5)
        assert trimmed == 5
        assert archive.size == 5

    def test_trim_no_trim_needed(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(3):
            archive.add_to_archive(d)
        trimmed = archive.trim_archive(max_entries=10)
        assert trimmed == 0
        assert archive.size == 3


class TestSparseArchiveSummary:
    def test_summary_empty(self):
        archive = SparseArchive()
        summary = archive.get_archive_summary()
        assert summary["size"] == 0

    def test_summary_with_entries(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(5):
            archive.add_to_archive(d)
        summary = archive.get_archive_summary()
        assert summary["size"] == 5


class TestSparseArchiveBatchLookup:
    def test_batch_lookup_empty(self):
        archive = SparseArchive()
        result = archive.batch_lookup([], current_generation=10, half_life=100)
        assert result == {}

    def test_batch_lookup_with_programs(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(5):
            archive.add_to_archive(d, metadata={"program": [1, 2]})
        result = archive.batch_lookup([[1, 2], [3, 4]], current_generation=10, half_life=100)
        assert isinstance(result, dict)


class TestSparseArchiveNoveltyStats:
    def test_novelty_stats_empty(self):
        archive = SparseArchive()
        stats = archive.get_novelty_stats()
        assert isinstance(stats, dict)

    def test_novelty_stats_with_entries(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(5):
            archive.add_to_archive(d)
        stats = archive.get_novelty_stats()
        assert isinstance(stats, dict)


class TestSparseArchiveTrajectory:
    def test_trajectory_empty(self):
        archive = SparseArchive()
        traj = archive.get_novelty_trajectory("nonexistent")
        assert traj == []

    def test_trajectory_with_entries(self):
        archive = SparseArchive()
        d = _make_descriptor()
        archive.add_to_archive(d, metadata={"program": [1, 2]})
        traj = archive.get_novelty_trajectory("nonexistent")
        assert isinstance(traj, list)


class TestSparseArchiveDistribution:
    def test_distribution_stats_empty(self):
        archive = SparseArchive()
        stats = archive.get_novelty_distribution_stats()
        assert isinstance(stats, dict)

    def test_distribution_stats_with_entries(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(10):
            archive.add_to_archive(d)
        stats = archive.get_novelty_distribution_stats()
        assert isinstance(stats, dict)

    def test_record_distribution_snapshot(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(5):
            archive.add_to_archive(d)
        snapshot = archive.record_distribution_snapshot(generation=1)
        assert isinstance(snapshot, dict)

    def test_distribution_history(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(5):
            archive.add_to_archive(d)
        archive.record_distribution_snapshot(generation=1)
        archive.record_distribution_snapshot(generation=2)
        history = archive.get_novelty_distribution_history(max_snapshots=10)
        assert isinstance(history, list)
        assert len(history) == 2

    def test_track_distribution_evolution(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(5):
            archive.add_to_archive(d)
        archive.record_distribution_snapshot(generation=1)
        evolution = archive.track_distribution_evolution(window=10)
        assert isinstance(evolution, dict)


class TestSparseArchiveImpactCorrelation:
    def test_impact_correlation_empty(self):
        archive = SparseArchive()
        corr = archive.get_novelty_impact_correlation()
        assert isinstance(corr, dict)

    def test_impact_correlation_with_entries(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(10):
            archive.add_to_archive(d, metadata={"fitness": random.random()})
        corr = archive.get_novelty_impact_correlation()
        assert isinstance(corr, dict)


# ══════════════════════════════════════════════════════════════════════════════
# Tests for extract_behavioral_descriptor
# ══════════════════════════════════════════════════════════════════════════════


class TestExtractBehavioralDescriptor:
    def test_empty_trace(self):
        desc = extract_behavioral_descriptor([], [], budget=100)
        assert isinstance(desc, dict)
        assert "inst_diversity" in desc

    def test_with_trace(self):
        trace = [{"op": "ADD", "rd": 1, "rs1": 2, "rs2": 3}]
        program = [0x00310033]  # ADD instruction
        desc = extract_behavioral_descriptor(trace, program, budget=100)
        assert isinstance(desc, dict)
        assert len(desc) >= 1

    def test_with_budget(self):
        desc = extract_behavioral_descriptor([], [], budget=50)
        assert isinstance(desc, dict)

    def test_program_length_normalised(self):
        # Long program
        program = [0x00310033] * 100
        desc = extract_behavioral_descriptor([], program, budget=100)
        assert desc["program_length"] >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Tests for adaptive threshold
# ══════════════════════════════════════════════════════════════════════════════


class TestAdaptiveThreshold:
    def test_empty_archive(self):
        archive = SparseArchive()
        threshold = archive.adaptive_threshold()
        assert threshold > 0.0

    def test_threshold_decreases_with_size(self):
        archive = SparseArchive()
        t1 = archive.adaptive_threshold()
        for d in _make_n_descriptors(10):
            archive.add_to_archive(d)
        t2 = archive.adaptive_threshold()
        assert t2 <= t1


# ══════════════════════════════════════════════════════════════════════════════
# Tests for _k_nn_distance
# ══════════════════════════════════════════════════════════════════════════════


class TestKNnDistance:
    def test_empty_archive(self):
        archive = SparseArchive()
        dist = archive._k_nn_distance([0.0, 0.0, 0.0])
        assert dist == 0.0

    def test_single_entry(self):
        archive = SparseArchive()
        archive.add_to_archive(_make_descriptor(inst_diversity=0.5))
        dist = archive._k_nn_distance(_flatten_descriptor(_make_descriptor(inst_diversity=0.5)))
        assert dist < 0.01  # Same point, distance ~0

    def test_multiple_entries(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(10):
            archive.add_to_archive(d)
        dist = archive._k_nn_distance(_flatten_descriptor(_make_descriptor(inst_diversity=0.5)))
        assert dist > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Tests for _rebuild_tree and _rebuild_prog_hash_index
# ══════════════════════════════════════════════════════════════════════════════


class TestInternalHelpers:
    def test_rebuild_tree_empty(self):
        archive = SparseArchive()
        archive._rebuild_tree()
        assert archive._tree is None

    def test_rebuild_tree_with_entries(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(5):
            archive.add_to_archive(d)
        archive._rebuild_tree()
        assert archive._tree is not None

    def test_rebuild_prog_hash_index_empty(self):
        archive = SparseArchive()
        archive._rebuild_prog_hash_index()
        assert len(archive._prog_hash_index) == 0

    def test_rebuild_prog_hash_index_with_program(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(5):
            archive.add_to_archive(d, metadata={"program": [1, 2, 3]})
        archive._rebuild_prog_hash_index()
        assert len(archive._prog_hash_index) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Tests for _relative_novelty_multiplier
# ══════════════════════════════════════════════════════════════════════════════


class TestRelativeNoveltyMultiplier:
    def test_empty_archive(self):
        archive = SparseArchive()
        mult = archive._relative_novelty_multiplier(0.5)
        assert mult == 1.0

    def test_novel_program_higher_multiplier(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(10):
            archive.add_to_archive(d)
        mult = archive._relative_novelty_multiplier(10.0)
        assert mult > 1.0

    def test_boring_program_lower_multiplier(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(10):
            archive.add_to_archive(d)
        mult = archive._relative_novelty_multiplier(0.001)
        assert mult < 1.0

    def test_multiplier_bounded(self):
        archive = SparseArchive()
        for d in _make_n_descriptors(10):
            archive.add_to_archive(d)
        mult = archive._relative_novelty_multiplier(100.0)
        assert 0.3 <= mult <= 2.5
