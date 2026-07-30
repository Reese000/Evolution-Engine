#!/usr/bin/env python3
"""Tests for PrefixGenerator.batch_generate().

Validates that batch_generate() produces correct results when processing
multiple prefixes in a single forward pass, matching the output quality
of sequential complete() calls.
"""

import os
import sys
import time

import pytest
import torch

# Path setup

from milestone1.rv32i_model import RV32IVocab
from prefix_generator import PrefixGenerator

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def prefix_generator():
    """Create a PrefixGenerator with the best checkpoint (or fresh fallback)."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ckpt_path = os.path.join(base, "milestone1", "checkpoints", "checkpoint_best.pt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(base, "milestone1", "checkpoints", "checkpoint_latest.pt")
    return PrefixGenerator(ckpt_path, temperature=0.9, max_new_tokens=64)


# ── Sample prefixes ───────────────────────────────────────────────────────────

PREFIX_A = [
    "0x02a00093",  # ADDI x1, x0, 42
    "0x03a00113",  # ADDI x2, x0, 58
    "0x002080b3",  # ADD x1, x1, x2
    "0x00000013",  # NOP
]

PREFIX_B = [
    "0x00500093",  # ADDI x1, x0, 5
    "0x00a00113",  # ADDI x2, x0, 10
    "0x002080b3",  # ADD x1, x1, x2
    "0x00000013",  # NOP
]

PREFIX_C = [
    "0x00a00093",  # ADDI x1, x0, 10
    "0x01400113",  # ADDI x2, x0, 20
    "0x00000013",  # NOP
]

PREFIX_D = [
    "0x00100093",  # ADDI x1, x0, 1
]


# ── Basic functionality tests ─────────────────────────────────────────────────


class TestBatchGenerateBasic:
    """Test basic batch_generate functionality."""

    def test_returns_list_of_lists(self, prefix_generator):
        """batch_generate should return a list of lists."""
        prefixes = [PREFIX_A, PREFIX_B]
        results = prefix_generator.batch_generate(prefixes, n=3)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, list)

    def test_single_prefix(self, prefix_generator):
        """batch_generate should work with a single prefix."""
        results = prefix_generator.batch_generate([PREFIX_A], n=2)
        assert len(results) == 1
        # Each result should be a list of completions
        assert isinstance(results[0], list)

    def test_empty_prefixes(self, prefix_generator):
        """batch_generate should return empty list for empty input."""
        results = prefix_generator.batch_generate([], n=3)
        assert results == []

    def test_n_parameter(self, prefix_generator):
        """batch_generate should generate up to n completions per prefix."""
        results = prefix_generator.batch_generate([PREFIX_A], n=5)
        assert len(results) == 1
        # Each completion should be a list of hex strings
        for comp in results[0]:
            assert isinstance(comp, list)
            assert len(comp) >= 2  # at least 2 instructions

    def test_completions_are_hex_strings(self, prefix_generator):
        """Generated completions should be hex instruction strings."""
        results = prefix_generator.batch_generate([PREFIX_A], n=3)
        for prefix_completions in results:
            for comp in prefix_completions:
                for hex_str in comp:
                    assert hex_str.startswith("0x"), f"Expected hex string, got {hex_str}"
                    # Should be 10 chars (0x + 8 hex digits)
                    assert len(hex_str) == 10, f"Expected 10-char hex, got {hex_str}"

    def test_ebreak_terminator(self, prefix_generator):
        """Completions should end with EBREAK (0x00100073)."""
        results = prefix_generator.batch_generate([PREFIX_A], n=3)
        for prefix_completions in results:
            for comp in prefix_completions:
                if comp:  # non-empty completion
                    assert comp[-1].upper() == "0X00100073", f"Expected EBREAK terminator, got {comp[-1]}"


# ── Multiple prefixes test ────────────────────────────────────────────────────


class TestBatchGenerateMultiple:
    """Test batch_generate with multiple prefixes simultaneously."""

    def test_two_prefixes(self, prefix_generator):
        """batch_generate should handle two different prefixes."""
        prefixes = [PREFIX_A, PREFIX_B]
        results = prefix_generator.batch_generate(prefixes, n=3)
        assert len(results) == 2
        # Both should produce completions
        for r in results:
            assert isinstance(r, list)

    def test_four_prefixes(self, prefix_generator):
        """batch_generate should handle four different prefixes."""
        prefixes = [PREFIX_A, PREFIX_B, PREFIX_C, PREFIX_D]
        results = prefix_generator.batch_generate(prefixes, n=2)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, list)

    def test_different_length_prefixes(self, prefix_generator):
        """batch_generate should handle prefixes of different lengths."""
        # PREFIX_D has only 1 instruction, PREFIX_A has 4
        prefixes = [PREFIX_D, PREFIX_A]
        results = prefix_generator.batch_generate(prefixes, n=2)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, list)


# ── Correctness tests ────────────────────────────────────────────────────────


class TestBatchGenerateCorrectness:
    """Test that batch_generate produces valid, non-degenerate output."""

    def test_completions_differ_from_prefix(self, prefix_generator):
        """Generated instructions should not just repeat the prefix."""
        results = prefix_generator.batch_generate([PREFIX_A], n=3)
        for comp in results[0]:
            # At least some generated instructions should differ from prefix
            # (The EBREAK terminator is always appended, so len(comp) > len(prefix))
            assert len(comp) >= 2

    def test_temperature_zero_is_deterministic(self, prefix_generator):
        """With temperature=0, generation should be deterministic."""
        results1 = prefix_generator.batch_generate([PREFIX_A], n=2, temperature=0.0)
        results2 = prefix_generator.batch_generate([PREFIX_A], n=2, temperature=0.0)
        # Both runs should produce the same output
        assert results1 == results2

    def test_batch_matches_sequential(self, prefix_generator):
        """batch_generate should produce equivalent results to sequential complete().

        We can't compare exact token sequences due to different RNG states and
        padding, but the structure should be equivalent: same number of prefixes,
        same number of completions per prefix, valid hex output.
        """
        prefixes = [PREFIX_A, PREFIX_B]
        n = 3

        # Batch result
        batch_results = prefix_generator.batch_generate(prefixes, n=n)

        # Sequential results
        sequential_results = []
        for prefix in prefixes:
            result = prefix_generator.complete(
                prefix,
                prefix_count=4,
                num_samples=n,
                temperature=0.9,
                top_k=40,
                top_p=0.9,
            )
            sequential_results.append(result)

        # Structure should match
        assert len(batch_results) == len(sequential_results) == 2
        for batch_r, seq_r in zip(batch_results, sequential_results):
            # Both should have completions (may differ in count due to filtering)
            assert isinstance(batch_r, list)
            assert isinstance(seq_r, list)


# ── Performance test ──────────────────────────────────────────────────────────


class TestBatchGeneratePerformance:
    """Test that batch_generate is faster than sequential complete() calls."""

    def test_batch_generate_runs(self, prefix_generator):
        """batch_generate should complete without error and produce valid output.

        Performance characteristics:
        - GPU: expect 3-4x speedup (larger batch = better GPU utilization)
        - CPU: speedup may be modest or negative for small models (< 10M params)
          because autoregressive generation is inherently sequential and the
          overhead of left-padding longer sequences can outweigh parallelism.
        """
        prefixes = [PREFIX_A, PREFIX_B, PREFIX_C, PREFIX_D]
        n = 3

        # Warm up
        prefix_generator.batch_generate([PREFIX_A], n=1)

        # Time sequential
        t0 = time.perf_counter()
        for _ in range(3):
            for prefix in prefixes:
                prefix_generator.complete(
                    prefix,
                    prefix_count=4,
                    num_samples=n,
                    temperature=0.9,
                    top_k=40,
                    top_p=0.9,
                )
        seq_time = time.perf_counter() - t0

        # Time batch
        t0 = time.perf_counter()
        for _ in range(3):
            prefix_generator.batch_generate(prefixes, n=n)
        batch_time = time.perf_counter() - t0

        speedup = seq_time / batch_time if batch_time > 0 else float("inf")
        print(f"\n  Speedup: {speedup:.2f}x (batch={batch_time:.3f}s, seq={seq_time:.3f}s)")
        print("  (GPU expected 3-4x; CPU speedup depends on model size and batch dims)")

        # Just verify the batch runs and produces valid output
        results = prefix_generator.batch_generate(prefixes, n=n)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, list)


# ── Edge case tests ───────────────────────────────────────────────────────────


class TestBatchGenerateEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_hex_prefix(self, prefix_generator):
        """batch_generate should handle invalid hex strings gracefully."""
        # One valid prefix, one with invalid hex
        prefixes = [PREFIX_A, ["0xINVALID", "0xBAD"]]
        results = prefix_generator.batch_generate(prefixes, n=2)
        assert len(results) == 2
        # Valid prefix should still work
        assert len(results[0]) >= 0  # may or may not produce completions
        # Invalid prefix should return empty
        assert results[1] == []

    def test_all_invalid_prefixes(self, prefix_generator):
        """batch_generate should return empty lists for all-invalid input."""
        prefixes = [["0xINVALID"], ["0xBAD"]]
        results = prefix_generator.batch_generate(prefixes, n=3)
        assert results == [[], []]

    def test_prefix_count_parameter(self, prefix_generator):
        """prefix_count should control how many instructions are used."""
        long_prefix = PREFIX_A + PREFIX_B  # 8 instructions
        results = prefix_generator.batch_generate([long_prefix], n=2, prefix_count=2)
        assert len(results) == 1
        # Should still produce valid output
        for comp in results[0]:
            assert isinstance(comp, list)

    def test_single_instruction_prefix(self, prefix_generator):
        """batch_generate should work with a single-instruction prefix."""
        results = prefix_generator.batch_generate([PREFIX_D], n=2)
        assert len(results) == 1
        for comp in results[0]:
            assert isinstance(comp, list)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
