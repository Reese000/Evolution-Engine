"""Tests for cross-cycle level stall nudge."""

from evolution.level_stall_tracker import load_state, save_state, update_and_maybe_nudge_start_level


def test_l19_nudge_at_observed_fitness(tmp_path):
    root = str(tmp_path)
    for _ in range(2):
        assert (
            update_and_maybe_nudge_start_level(root, 19, 0.9, final_fitness=83.3, stall_cycles_before_nudge=3) is None
        )
    nudge = update_and_maybe_nudge_start_level(root, 19, 0.9, final_fitness=83.3, stall_cycles_before_nudge=3)
    assert nudge == 20


def test_nudge_after_repeated_stall(tmp_path):
    root = str(tmp_path)
    for _ in range(2):
        assert (
            update_and_maybe_nudge_start_level(root, 17, 0.875, final_fitness=130.0, stall_cycles_before_nudge=3)
            is None
        )
    nudge = update_and_maybe_nudge_start_level(root, 17, 0.875, final_fitness=130.0, stall_cycles_before_nudge=3)
    assert nudge == 18
    state = load_state(root)
    assert state["cycles"] == 3


def test_resets_on_level_change(tmp_path):
    root = str(tmp_path)
    save_state(root, {"level": 17, "cycles": 10, "verify": 0.875})
    assert update_and_maybe_nudge_start_level(root, 18, 0.9, stall_cycles_before_nudge=5) is None
    state = load_state(root)
    assert state["level"] == 18
    assert state["cycles"] == 1
