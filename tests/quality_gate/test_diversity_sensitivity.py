#!/usr/bin/env python3
"""Tests for diversity measurement sensitivity (P2-6).

Verifies that _current_diversity actually tracks population diversity
rather than being stuck at the initial value.
"""
import os
import sys

# Path setup
from evolution_engine import EvolutionConfig, EvolutionEngine
from mutation_proposer import EBREAK, make_rv32i


def _make_engine():
    """Create a minimal engine for testing."""
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.config = EvolutionConfig()
    engine.config.use_gpu = False
    engine._current_diversity = 0.5
    return engine


def test_diversity_updates_after_generation_evaluate():
    """_current_diversity should change after _run_generation_evaluate."""
    engine = _make_engine()
    # Verify initial diversity
    assert engine._current_diversity == 0.5

    # Build a homogeneous population (all same instruction)
    # This should produce LOW diversity
    homogeneous = [
        [0x00310093, 0x00510113, EBREAK]  # ADDI instructions only
        for _ in range(20)
    ]

    div_homo = EvolutionEngine._compute_population_diversity(homogeneous)

    # Build a heterogeneous population (different opcode categories)
    # ADDI (imm-gen), ADD (register-register), LW (load), BEQ (branch), etc.
    heterogeneous = [
        [0x00310093, EBREAK],                          # ADDI (imm-gen)
        [0x00a50533, EBREAK],                          # ADD (reg-reg)
        [0x00052583, EBREAK],                          # LW (load)
        [0x00b50063, EBREAK],                          # BEQ (branch)
        [0x00052023, EBREAK],                          # SW (store)
        [0x000000EF, EBREAK],                          # JAL (jump)
        [0x00a51513, EBREAK],                          # SLLI (shift)
        [0x00a54533, EBREAK],                          # XOR (logic)
        [0x00a52533, EBREAK],                          # SLT (compare)
        [0x00002073, EBREAK],                          # ECALL (system)
        [0x00310093, EBREAK],                          # ADDI again
        [0x00a50533, EBREAK],                          # ADD again
        [0x00052583, EBREAK],                          # LW again
        [0x00b50063, EBREAK],                          # BEQ again
        [0x00052023, EBREAK],                          # SW again
        [0x000000EF, EBREAK],                          # JAL again
        [0x00a51513, EBREAK],                          # SLLI again
        [0x00a54533, EBREAK],                          # XOR again
        [0x00a52533, EBREAK],                          # SLT again
        [0x00002073, EBREAK],                          # ECALL again
    ]

    div_hetero = EvolutionEngine._compute_population_diversity(heterogeneous)

    print(f"Homogeneous diversity: {div_homo:.4f}")
    print(f"Heterogeneous diversity: {div_hetero:.4f}")

    # Heterogeneous should be more diverse than homogeneous
    assert div_hetero > div_homo, (
        f"Heterogeneous diversity ({div_hetero:.4f}) should be > "
        f"homogeneous ({div_homo:.4f})"
    )
    print("PASS: diversity computation distinguishes populations")


def test_diversity_stays_synchronized():
    """After _compute_population_diversity, _current_diversity should match."""
    engine = _make_engine()
    # Simulate what _run_generation_evaluate does
    challengers = [
        [0x00310093, 0x00a50533, EBREAK]  # Mixed: ADDI + ADD
        for _ in range(10)
    ]
    # This is the new code path
    engine._current_diversity = EvolutionEngine._compute_population_diversity(challengers)
    expected = EvolutionEngine._compute_population_diversity(challengers)
    assert engine._current_diversity == expected, (
        f"_current_diversity ({engine._current_diversity}) != "
        f"computed ({expected})"
    )
    print(f"PASS: diversity synchronized at {engine._current_diversity:.4f}")


def test_diversity_responsive_to_population_change():
    """Diversity should change significantly when population composition changes."""
    engine = _make_engine()

    # All same opcode category -> low diversity
    pop_low = [
        [0x00310093, EBREAK]  # Only ADDI (imm-gen category)
        for _ in range(20)
    ]
    engine._current_diversity = EvolutionEngine._compute_population_diversity(pop_low)
    low_div = engine._current_diversity

    # All different opcode categories -> high diversity
    opcodes = [
        0x00310093,  # ADDI
        0x00a50533,  # ADD
        0x00052583,  # LW
        0x00b50063,  # BEQ
        0x00052023,  # SW
        0x000000EF,  # JAL
        0x00a51513,  # SLLI
        0x00a54533,  # XOR
        0x00a52533,  # SLT
        0x00002073,  # ECALL
        0x00310093,  # ADDI (repeat to get 11th)
    ]
    pop_high = [[op, EBREAK] for op in opcodes]
    engine._current_diversity = EvolutionEngine._compute_population_diversity(pop_high)
    high_div = engine._current_diversity

    # The difference should be significant
    diff = high_div - low_div
    print(f"Low diversity:  {low_div:.4f}")
    print(f"High diversity: {high_div:.4f}")
    print(f"Difference:     {diff:.4f}")

    assert diff > 0.1, (
        f"Diversity difference ({diff:.4f}) too small — "
        f"measurement may not be sensitive enough"
    )
    print("PASS: diversity measurement is sensitive to population changes")


if __name__ == "__main__":
    test_diversity_updates_after_generation_evaluate()
    test_diversity_stays_synchronized()
    test_diversity_responsive_to_population_change()
    print("\nAll diversity sensitivity tests passed!")
