#!/usr/bin/env python3
"""Tests for verify_curriculum.py — Verify curriculum reference solutions."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from verify_curriculum import LEVEL_CLASSES, VERIFY_REGS, encode_program, run_level

# ══════════════════════════════════════════════════════════════════════════════
# Tests for encode_program
# ══════════════════════════════════════════════════════════════════════════════


class TestEncodeProgram:
    def test_empty_program(self):
        words = encode_program([], add_ebreak=True)
        assert words == [0x00100073]  # EBREAK

    def test_single_instruction(self):
        words = encode_program([{"name": "ADDI", "rd": 1, "rs1": 0, "imm": 42}], add_ebreak=True)
        assert isinstance(words, list)
        assert len(words) >= 1  # At least the instruction + EBREAK

    def test_no_ebreak(self):
        words = encode_program([{"name": "ADDI", "rd": 1, "rs1": 0, "imm": 42}], add_ebreak=False)
        assert isinstance(words, list)

    def test_ebreak_present(self):
        words = encode_program([{"name": "EBREAK"}], add_ebreak=True)
        assert 0x00100073 in words

    def test_ebreak_not_duplicated(self):
        words = encode_program([{"name": "EBREAK"}], add_ebreak=True)
        assert words.count(0x00100073) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Tests for VERIFY_REGS
# ══════════════════════════════════════════════════════════════════════════════


class TestVerifyRegs:
    def test_all_levels_defined(self):
        for level_num in range(17):
            assert level_num in VERIFY_REGS

    def test_verify_regs_are_lists(self):
        for regs in VERIFY_REGS.values():
            assert isinstance(regs, list)

    def test_verify_regs_increase(self):
        # Higher levels should have more registers
        assert len(VERIFY_REGS[0]) < len(VERIFY_REGS[16])


# ══════════════════════════════════════════════════════════════════════════════
# Tests for run_level
# ══════════════════════════════════════════════════════════════════════════════


class TestRunLevel:
    def test_run_level_returns_dict(self):
        result = run_level(LEVEL_CLASSES[0], seed=42)
        assert isinstance(result, dict)
        assert "level" in result
        assert "pass" in result
        assert "vm_success" in result
        assert "error" in result
        assert "cycles" in result
        assert "expected" in result
        assert "actual" in result
        assert "mismatches" in result

    def test_run_level_0(self):
        result = run_level(LEVEL_CLASSES[0], seed=42)
        assert result["level"] == 0

    def test_run_level_1(self):
        result = run_level(LEVEL_CLASSES[1], seed=42)
        assert result["level"] == 1

    def test_run_level_2(self):
        result = run_level(LEVEL_CLASSES[2], seed=42)
        assert result["level"] == 2

    def test_run_level_all_levels_run(self):
        """Run all levels and check they return valid results."""
        for cls in LEVEL_CLASSES:
            level_num = int(cls.__name__.replace("Level", ""))
            result = run_level(cls, seed=42 + level_num)
            assert result["level"] == level_num
            assert isinstance(result["mismatches"], dict)


# ══════════════════════════════════════════════════════════════════════════════
# Tests for LEVEL_CLASSES
# ══════════════════════════════════════════════════════════════════════════════


class TestLevelClasses:
    def test_all_17_levels(self):
        assert len(LEVEL_CLASSES) == 17

    def test_level_names(self):
        for i, cls in enumerate(LEVEL_CLASSES):
            assert cls.__name__ == f"Level{i}"

    def test_generate_instance(self):
        for cls in LEVEL_CLASSES:
            inst = cls(seed=42).generate_instance()
            assert isinstance(inst, dict)
            assert "inputs" in inst
            assert "expected_output" in inst

    def test_reference_solution(self):
        for cls in LEVEL_CLASSES:
            ref = cls.reference_solution()
            assert isinstance(ref, list)
            assert len(ref) > 0
