#!/usr/bin/env python3
"""Tests for EvolutionEngine.validate_program (R45 bad-champion pre-check)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "milestone1"))


class TestValidateProgram(unittest.TestCase):
    """Test the static validate_program method."""

    def _validate(self, program):
        from evolution_engine import EvolutionEngine

        return EvolutionEngine.validate_program(program)

    def test_empty_program(self):
        """Empty program is invalid."""
        valid, reason = self._validate([])
        self.assertFalse(valid)
        self.assertEqual(reason, "empty_program")

    def test_single_ebreak(self):
        """Single EBREAK is invalid (all_ebreak)."""
        valid, reason = self._validate([0x00100073])
        self.assertFalse(valid)
        self.assertEqual(reason, "all_ebreak")

    def test_two_instructions(self):
        """Two live instructions is invalid (too_short)."""
        valid, reason = self._validate([0x00100013, 0x00100073])
        self.assertFalse(valid)
        self.assertIn("too_short", reason)

    def test_three_instructions(self):
        """Three live instructions (4 with EBREAK) is valid."""
        valid, reason = self._validate([0x00100013, 0x00200013, 0x00300013, 0x00100073])
        self.assertTrue(valid)
        self.assertEqual(reason, "ok")

    def test_valid_program(self):
        """Typical valid program with EBREAK at end."""
        prog = [0x00100013, 0x00200013, 0x00300013, 0x00100073]
        valid, reason = self._validate(prog)
        self.assertTrue(valid)

    def test_overflow(self):
        """Program with value > 0xFFFFFFFF is invalid."""
        valid, reason = self._validate([0x100000000, 0x00100013, 0x00100073])
        self.assertFalse(valid)
        self.assertEqual(reason, "overflow")

    def test_negative_value(self):
        """Program with negative value is invalid."""
        valid, reason = self._validate([-1, 0x00100013, 0x00100073])
        self.assertFalse(valid)
        self.assertEqual(reason, "overflow")

    def test_many_ebreaks_at_end(self):
        """Program with many trailing EBREAKs but valid body is OK."""
        prog = [0x00100013, 0x00200013, 0x00300013] + [0x00100073] * 10
        valid, reason = self._validate(prog)
        self.assertTrue(valid)


if __name__ == "__main__":
    unittest.main()
