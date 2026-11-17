"""Tests for max-pop verify promotion and fresh verify guards."""

import os
from unittest.mock import MagicMock

os.environ.setdefault("EE_USE_GPU", "0")


def test_maybe_promote_max_pop_verify_champion():
    from evolution.selection_mixin import SelectionMixin

    class Engine(SelectionMixin):
        pass

    eng = Engine()
    eng.champion_level = 30
    eng.config = MagicMock(max_level=30)
    eng.champion = [1, 2, 3]
    eng.champion_trace = {"status": "ok", "final_regs": {}, "final_memory": {}}
    eng._champion_verify_score = 1.0
    eng._last_max_pop_verify = 1.0
    eng._last_max_pop_program = [9, 8, 7]
    eng._last_max_pop_trace = {"status": "ok", "final_regs": {25: 1}, "final_memory": {}}
    eng._raw_task_score = 0.0
    eng._log = MagicMock()
    eng._verify_score_from_trace = MagicMock(return_value=0.842)
    eng._verify_and_replace_champion = MagicMock(return_value=True)

    from evolution.fitness_helpers import selection_fitness

    eng._compute_fitness = MagicMock(return_value=200.0)
    assert eng._maybe_promote_max_pop_verify_champion() is True
    eng._verify_and_replace_champion.assert_called_once()


def test_stale_verify_does_not_block_promotion_pin():
    from evolution.champion_bloat_guard import certified_champion_pinned

    assert certified_champion_pinned(30, 73, 1.0) is True
    assert certified_champion_pinned(30, 73, 0.842) is False
