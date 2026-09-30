"""Curriculum advance gates with progressive anti-plateau relaxation."""

from __future__ import annotations

import os


def level_verify_reg_count(level: int) -> int:
    """Number of registers scored by continuous verify at this level."""
    from tasks_rv32i import LEVEL_VERIFY_REGS

    regs = LEVEL_VERIFY_REGS.get(level)
    if regs:
        return len(regs)
    # Fallback for levels not yet in map
    try:
        from verify_curriculum import VERIFY_REGS

        legacy = VERIFY_REGS.get(level, [])
        if legacy:
            return len(legacy)
    except ImportError:
        pass
    return 1


def missed_regs_allowance(stall: int, stall_threshold: int) -> int:
    """How many verify registers may be wrong; increases with stall depth."""
    if stall < stall_threshold:
        return 0
    periods = 1 + (stall - stall_threshold) // stall_threshold
    return min(4, periods)


def curriculum_verify_threshold(level: int, stall: int, stall_threshold: int) -> float:
    """Minimum partial-verify score to advance curriculum at *level*."""
    override = os.environ.get("EE_CURRICULUM_VERIFY_MIN")
    if override is not None:
        return float(override)

    if level < 8:
        return 1.0

    reg_count = level_verify_reg_count(level)
    if reg_count <= 1:
        return 1.0

    missed = missed_regs_allowance(stall, stall_threshold)
    return max(0.5, (reg_count - missed) / reg_count)


def curriculum_fitness_gate(level: int, stall: int, raw_score: float, stall_threshold: int) -> bool:
    """Whether raw task score allows curriculum advance (relaxes when stalled)."""
    override = os.environ.get("EE_CURRICULUM_RAW_MIN")
    if override is not None:
        return raw_score >= float(override)

    # Level-aware base: high levels already carry harder tasks
    if level >= 10:
        minimum = 95.0 + (level - 10) * 0.5
    else:
        minimum = 110.0

    if stall >= stall_threshold:
        relax = min(50.0, (stall - stall_threshold + 1) * 2.5)
        minimum -= relax

    return raw_score >= max(55.0, minimum)


def depth_carry_verify_floor(level: int) -> float:
    """Minimum verify to hold curriculum depth across cycles (below certify bar)."""
    lvl = int(level)
    if lvl >= 24:
        # Min steps down with depth — L29 escape lands ~0.474; fixed 0.52 min caused L28/L29 oscillation.
        return max(0.47, 0.60 - (lvl - 24) * 0.05)
    if lvl >= 22:
        return 0.63
    if lvl == 21:
        return 0.72
    floor_by_level = ((20, 0.72), (19, 0.70), (18, 0.68), (17, 0.65))
    floor = 0.80
    for min_lvl, min_ver in floor_by_level:
        if lvl >= min_lvl:
            floor = min_ver
            break
    return floor


def nudge_verify_passes(level: int, verify: float, cycles_at_level: int) -> bool:
    """Whether verify qualifies for cross-cycle start-level nudge.

    Uses a level floor below the strict certify bar so stable L21@0.75 can nudge
    to L22 after a few cycles without waiting for stall relaxation to 5+ cycles.
    """
    min_cycles = int(os.environ.get("EE_NUDGE_CYCLES", "3"))
    if cycles_at_level < min_cycles:
        return False
    lvl = int(level)
    ver = float(verify)
    return ver >= depth_carry_verify_floor(lvl)


def nudge_fitness_floor(level: int) -> float:
    """Minimum champion_fitness for cross-cycle start-level nudge.

    High curriculum levels use a lower champion_fitness scale than L17; a flat
    95 gate blocks nudges indefinitely and stalls EPI growth.
    """
    override = os.environ.get("EE_NUDGE_MIN_FITNESS")
    base = float(override) if override is not None else 95.0
    lvl = int(level)
    if lvl >= 21:
        return max(62.0, base - (lvl - 16) * 6.0)
    if lvl >= 17:
        return max(72.0, base - (lvl - 16) * 4.0)
    return base


def nudge_fitness_passes(level: int, fitness: float, verify: float) -> bool:
    """Fitness gate for nudge — relaxes when verify is certify-stable at depth."""
    if float(fitness) >= nudge_fitness_floor(level):
        return True
    if int(level) >= 20 and float(verify) >= 0.72 and float(fitness) >= 62.0:
        return True
    lvl = int(level)
    if lvl >= 22 and float(fitness) >= 50.0 and float(verify) >= depth_carry_verify_floor(lvl):
        return True
    return False


def forced_advance_verify_floor(level: int, stall: int, stall_threshold: int) -> float:
    """Last-resort verify floor for deep stall escape (never permanent plateau)."""
    if stall < 4 * stall_threshold:
        return 1.0
    base = curriculum_verify_threshold(level, stall, stall_threshold)
    if stall >= 8 * stall_threshold:
        return max(0.5, base - 0.15)
    if stall >= 6 * stall_threshold:
        return max(0.55, base - 0.1)
    return base
