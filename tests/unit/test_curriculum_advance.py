"""Anti-plateau curriculum advance gates."""

from __future__ import annotations

from evolution.curriculum_advance import (
    curriculum_fitness_gate,
    curriculum_verify_threshold,
    depth_carry_verify_floor,
    forced_advance_verify_floor,
    level_verify_reg_count,
    missed_regs_allowance,
    nudge_fitness_passes,
    nudge_verify_passes,
)


def test_l13_threshold_allows_875():
    assert curriculum_verify_threshold(13, 3, 3) == 0.875


def test_l17_threshold_allows_875_not_full_verify():
    """L17 was stuck: legacy VERIFY_REGS ended at L16 -> required 1.0."""
    assert level_verify_reg_count(17) == 8
    assert curriculum_verify_threshold(17, 3, 3) == 0.875


def test_l18_threshold_scales_with_regs():
    assert level_verify_reg_count(18) == 9
    assert curriculum_verify_threshold(18, 3, 3) == 8 / 9


def test_deep_stall_relaxes_threshold():
    assert curriculum_verify_threshold(17, 6, 3) == 0.75
    assert curriculum_verify_threshold(17, 12, 3) == 0.5


def test_missed_regs_progressive():
    assert missed_regs_allowance(3, 3) == 1
    assert missed_regs_allowance(6, 3) == 2
    assert missed_regs_allowance(9, 3) == 3
    assert missed_regs_allowance(15, 3) == 4


def test_fitness_gate_relaxes_with_stall():
    assert curriculum_fitness_gate(17, 3, 128.0, 3) is True
    assert curriculum_fitness_gate(17, 3, 90.0, 3) is False
    assert curriculum_fitness_gate(17, 20, 90.0, 3) is True


def test_forced_escape_floor_only_at_deep_stall():
    assert forced_advance_verify_floor(17, 10, 3) == 1.0
    assert forced_advance_verify_floor(17, 24, 3) < 0.875


def test_depth_carry_verify_floor_scales_at_depth():
    assert depth_carry_verify_floor(22) == 0.63
    assert depth_carry_verify_floor(24) == 0.6
    assert round(depth_carry_verify_floor(25), 2) == 0.55
    assert depth_carry_verify_floor(26) == 0.5
    assert depth_carry_verify_floor(28) == 0.47
    assert depth_carry_verify_floor(29) == 0.47


def test_nudge_verify_uses_highest_level_floor():
    """L23 must use 0.63 floor, not fall through to 0.65."""
    assert nudge_verify_passes(23, 0.643, 10) is True
    assert nudge_verify_passes(23, 0.62, 10) is False
    assert nudge_verify_passes(21, 0.75, 8) is True
    assert nudge_verify_passes(24, 0.60, 8) is True
    assert nudge_verify_passes(24, 0.59, 8) is False
    assert nudge_verify_passes(25, 0.57, 8) is True
    assert nudge_verify_passes(25, 0.5625, 8) is True
    assert nudge_verify_passes(25, 0.54, 8) is False
    assert nudge_verify_passes(28, 0.5263, 8) is True
    assert nudge_verify_passes(28, 0.46, 8) is False
    assert nudge_verify_passes(29, 0.4737, 8) is True
    assert nudge_verify_passes(29, 0.46, 8) is False


def test_nudge_fitness_passes_at_depth_with_lower_fitness():
    assert nudge_fitness_passes(24, 56.4, 0.6) is True
    assert nudge_fitness_passes(24, 49.0, 0.6) is False
