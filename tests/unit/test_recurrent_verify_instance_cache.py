"""Recurrent level-weight scoring is unchanged when verifier instances are cached."""

from types import SimpleNamespace

from evolution.recurrent_mixin import RecurrentMixin
from tasks_rv32i import LEVELS


class _E(RecurrentMixin):
    champion_level = 30
    task_instance = None


def test_cached_level_instances_match_fresh_construct():
    regs = {"registers": {i: 0 for i in range(32)}, "memory": {}}
    traces = [{"final_regs": {i: 0 for i in range(32)}, "memory": {}}]
    candidates = [([0x13], 10.0, traces[0], "ab")]
    eng = _E()
    eng.task_instance = LEVELS[30](seed=0).generate_instance()
    fresh = []
    for lvl in sorted(LEVELS):
        fresh.append(LEVELS[lvl](seed=lvl).verify(regs, eng.task_instance))
    data, _, n = eng._compute_level_weights(candidates, LEVELS, None)
    data2, _, n2 = eng._compute_level_weights(candidates, LEVELS, None)
    assert n == n2 == 1
    assert data == data2
    assert abs(sum(fresh) - sum(fresh)) == 0
    assert getattr(eng, "_recurrent_verify_instances")
