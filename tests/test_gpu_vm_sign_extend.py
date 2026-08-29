#!/usr/bin/env python3
"""Tests for RV32I sign-extension helpers in gpu_vm_farm.

_sign_extend / _sign_extend32 underpin immediate decoding and signed
comparisons in the RV32I VM. A bug here silently corrupts every evaluation,
so these pure functions are worth locking down (they had no direct coverage).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_vm_farm import _sign_extend, _sign_extend32


class TestSignExtend:
    """_sign_extend(val, bits) — precomputed-mask widths and the general path."""

    def test_12bit_common_width(self):
        assert _sign_extend(0x7FF, 12) == 2047
        assert _sign_extend(0x800, 12) == -2048
        assert _sign_extend(0xFFF, 12) == -1
        assert _sign_extend(0x000, 12) == 0

    def test_8bit_common_width(self):
        assert _sign_extend(0x7F, 8) == 127
        assert _sign_extend(0x80, 8) == -128
        assert _sign_extend(0xFF, 8) == -1

    def test_high_bits_are_masked_off(self):
        # Values wider than `bits` must be masked before sign detection.
        assert _sign_extend(0x1FFF, 12) == -1  # low 12 bits = 0xFFF
        assert _sign_extend(0xF000 | 0x123, 12) == 0x123

    def test_uncommon_width_general_path(self):
        # 5 is not a precomputed width -> exercises the fallback branch.
        assert _sign_extend(0x0F, 5) == 15
        assert _sign_extend(0x10, 5) == -16
        assert _sign_extend(0x1F, 5) == -1

    def test_other_precomputed_widths(self):
        for bits in (13, 16, 20, 21):
            assert _sign_extend(0, bits) == 0
            assert _sign_extend((1 << bits) - 1, bits) == -1
            assert _sign_extend(1 << (bits - 1), bits) == -(1 << (bits - 1))
            assert _sign_extend((1 << (bits - 1)) - 1, bits) == (1 << (bits - 1)) - 1


class TestSignExtend32:
    def test_boundaries(self):
        assert _sign_extend32(0) == 0
        assert _sign_extend32(0x7FFFFFFF) == 2147483647
        assert _sign_extend32(0x80000000) == -2147483648
        assert _sign_extend32(0xFFFFFFFF) == -1

    def test_values_above_32_bits_are_masked(self):
        assert _sign_extend32(0x1_FFFF_FFFF) == -1  # low 32 bits = 0xFFFFFFFF
        assert _sign_extend32(0x1_0000_0000) == 0
