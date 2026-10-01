"""Fitness-only champion ticks must not reset generations_stalled."""


def stall_after_generation(*, fitter: bool, shorter: bool, level_up: bool, stalled: int) -> int:
    """Mirror _run_generation_maintenance stall policy."""
    if fitter or shorter or level_up:
        return 0
    return stalled + 1


def test_fitness_tick_same_program_increments_stall():
    assert stall_after_generation(fitter=False, shorter=False, level_up=False, stalled=4) == 5


def test_shorter_or_fitter_resets_stall():
    assert stall_after_generation(fitter=True, shorter=False, level_up=False, stalled=12) == 0
    assert stall_after_generation(fitter=False, shorter=True, level_up=False, stalled=12) == 0
    assert stall_after_generation(fitter=False, shorter=False, level_up=True, stalled=12) == 0


def test_same_length_hash_swap_increments_stall():
    assert stall_after_generation(fitter=False, shorter=False, level_up=False, stalled=3) == 4


def test_no_improvement_increments_stall():
    assert stall_after_generation(fitter=False, shorter=False, level_up=False, stalled=0) == 1
