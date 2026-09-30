"""Block champion replacement when graft bloat would regress verify at high curriculum."""

from __future__ import annotations


def level_word_cap(level: int) -> int | None:
    """Hard word cap for champion at curriculum level (from reference length)."""
    if level < 20:
        return None
    try:
        from evolution.register_repair import reference_champion_program

        words, _ = reference_champion_program(int(level))
        if words:
            return len(words)
    except (ImportError, TypeError, ValueError, OSError):
        pass
    if level >= 22:
        return 48
    if level >= 21:
        return 36
    return 40


def champion_bloat_blocks(
    level: int,
    champion_words: int,
    challenger_words: int,
    challenger_verify: float,
    champion_verify: float,
) -> str | None:
    """Return a short reason if replacement must be blocked, else None."""
    if level < 20:
        return None
    hard_cap = level_word_cap(level) or (36 if level >= 21 else 40)
    if challenger_words > hard_cap:
        return f"words={challenger_words}>{hard_cap}"
    growth = challenger_words - champion_words
    if level >= 21:
        if challenger_words < 12:
            return f"stub={challenger_words}w at L{level}"
        if champion_words >= 20 and challenger_words < max(12, champion_words - 8):
            if challenger_verify < champion_verify - 0.05:
                return f"shrink={challenger_words}w from {champion_words}w"
        if growth >= 1 and challenger_verify < champion_verify - 1e-6:
            return f"growth={growth}w with verify drop"
    elif growth > 2 and challenger_verify <= champion_verify + 0.02:
        return f"growth={growth}w without verify gain"
    if (
        int(level) >= 29
        and float(champion_verify) >= 0.99 - 1e-6
        and int(champion_words) >= 60
        and growth >= 1
        and float(challenger_verify) <= float(champion_verify) + 1e-6
    ):
        return f"growth={growth}w at saturated L{level} {champion_words}w floor"
    return None


def certified_champion_pinned(level: int, champion_words: int, champion_verify: float) -> bool:
    """True when champion is stable enough to reject downgrades at depth."""
    import os

    from evolution.curriculum_advance import depth_carry_verify_floor

    max_level = int(os.environ.get("EE_MAX_LEVEL", "30"))
    if int(level) >= max_level:
        return float(champion_verify) >= 1.0 - 1e-6

    cap = level_word_cap(level)
    floor = depth_carry_verify_floor(level)
    if cap is not None and level == 21 and champion_verify >= floor and champion_words <= cap:
        return True
    if level >= 22 and champion_verify >= floor and champion_words >= 36:
        return True
    if level == 20 and champion_verify >= 0.70 and champion_words <= 38:
        return True
    return False
