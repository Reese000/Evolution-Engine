"""Recurrent feedback fast path at saturated L30."""

from types import SimpleNamespace

from champion_anchor import sha256_program
from evolution.recurrent_mixin import RecurrentMixin
from tasks_rv32i import LEVELS


class _E(RecurrentMixin):
    pass


def test_terminal_fast_path_single_level_verify():
    eng = _E()
    eng.champion_level = 30
    eng.config = SimpleNamespace(max_level=30)
    eng._champion_verify_score = 1.0
    eng.task_instance = LEVELS[30](seed=0).generate_instance()
    regs = {i: 0 for i in range(32)}
    trace = {"final_regs": regs, "final_memory": {}}
    candidates = [([0x13], 10.0, trace, "ab")]
    data, weights, n_levels = eng._compute_level_weights(candidates, LEVELS, None)
    assert n_levels == 1
    assert data[0][0] == 30
    insts = eng._recurrent_verify_instances
    assert set(insts.keys()) == {30}


def test_cap_terminal_recurrent_keeps_top_fitness():
    eng = _E()
    eng.champion_level = 30
    eng.config = SimpleNamespace(max_level=30)
    eng._champion_verify_score = 1.0
    cands = [(["a"], float(i), {}, f"s{i}") for i in range(200)]
    capped = eng._cap_terminal_recurrent_candidates(cands)
    assert len(capped) == 128
    assert capped[0][1] == 199.0
    assert capped[-1][1] == 72.0


def test_terminal_recurrent_interval_skips_intermediate_generations():
    eng = _E()
    eng.champion_level = 30
    eng.config = SimpleNamespace(max_level=30)
    eng._champion_verify_score = 1.0
    eng.current_generation = 3
    assert eng._should_skip_terminal_recurrent_feedback() is True
    eng.current_generation = 5
    assert eng._should_skip_terminal_recurrent_feedback() is False


def test_apply_novelty_skipped_at_terminal_fast_path():
    eng = _E()
    eng.champion_level = 30
    eng.config = SimpleNamespace(max_level=30)
    eng._champion_verify_score = 1.0
    weights, applied = eng._apply_novelty_weights([(["a"], 1.0, {}, "s")], [1.0], "/nonexistent/path.jsonl")
    assert weights == [1.0]
    assert applied is False


def test_terminal_fast_path_loads_tail_sha_index_only(tmp_path):
    from evolution.recurrent_mixin import RecurrentMixin
    from evolution.shared import MAX_RECURRENT_ENTRIES

    path = tmp_path / "recurrent_data.jsonl"
    first = '{"fitness": 1.0, "champion_sha256": "first"}\n'
    fill = '{"fitness": 1.0, "champion_sha256": "fill"}\n'
    new = '{"fitness": 2.0, "champion_sha256": "new"}\n'
    path.write_text(
        first + fill * (MAX_RECURRENT_ENTRIES + 20) + new * 5,
        encoding="utf-8",
    )

    class E(RecurrentMixin):
        pass

    eng = E()
    eng.champion_level = 30
    eng.config = SimpleNamespace(max_level=30)
    eng._champion_verify_score = 1.0
    idx = eng._load_saved_sha256_index(str(path))
    assert "new" in idx
    assert "first" not in idx


def test_evict_skips_full_parse_when_under_entry_cap(tmp_path):
    from evolution.recurrent_mixin import RecurrentMixin
    from evolution.shared import MAX_RECURRENT_ENTRIES

    path = tmp_path / "recurrent_data.jsonl"
    lines = ['{"fitness": 1.0, "champion_sha256": "a"}\n' for _ in range(MAX_RECURRENT_ENTRIES)]
    path.write_text("".join(lines), encoding="utf-8")

    class E(RecurrentMixin):
        pass

    eng = E()
    eng._evict_recurrent_buffer(str(path))
    assert path.read_text(encoding="utf-8").count("\n") == MAX_RECURRENT_ENTRIES


def test_try_recurrent_feedback_does_not_evict_each_call(tmp_path, monkeypatch):
    from evolution.recurrent_mixin import RecurrentMixin

    class E(RecurrentMixin):
        pass

    eng = E()
    eng.config = SimpleNamespace(recurrent_feedback_dir=str(tmp_path))
    eng._last_challengers = []
    calls = {"n": 0}
    monkeypatch.setattr(eng, "_evict_recurrent_buffer", lambda _p: calls.update(n=calls["n"] + 1))
    eng._try_recurrent_feedback()
    assert calls["n"] == 0


def test_collect_skips_task_aware_at_scan_floor(monkeypatch):
    monkeypatch.setattr(
        "evolution.recurrent_mixin.terminal_efficiency_at_scan_floor",
        lambda _n: True,
        raising=False,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _n: True,
    )
    eng = _E()
    eng.champion_level = 30
    eng.config = SimpleNamespace(max_level=30)
    eng._champion_verify_score = 1.0
    eng.champion = [0x00000013] * 59 + [0x00100073]
    eng._last_challengers = [[0x00000013, 0x00000013, 0x00000013, 0x00100073]]
    eng._last_results = [{"status": "ok", "instructions_count": 5, "final_regs": {}, "final_memory": {}}]
    eng._last_challenger_fitnesses = [50.0]
    eng._last_challenger_strategies = ["TASK_AWARE"]
    eng._saved_sha256s = {}
    eng._saved_size_counts = {}
    eng.generations_stalled = 0
    cands, _, _, _ = eng._collect_recurrent_candidates(eng._last_challenger_fitnesses, 1)
    assert len(cands) == 0


def test_collect_skips_probe_and_champion_sha_at_l30():
    eng = _E()
    eng.champion_level = 30
    eng.champion = [0x00000013, 0x00100073]
    champ_sha = sha256_program(eng.champion)
    eng._last_challengers = [
        eng.champion,
        [0x00000013, 0x00000013, 0x00100073],
    ]
    eng._last_results = [
        {"status": "ok", "instructions_count": 5, "final_regs": {}, "final_memory": {}},
        {"status": "ok", "instructions_count": 5, "final_regs": {}, "final_memory": {}},
    ]
    eng._last_challenger_fitnesses = [100.0, 50.0]
    eng._last_challenger_strategies = ["TERMINAL_VERIFY_PROBE", "MUTATION"]
    eng._saved_sha256s = {}
    eng._saved_size_counts = {}
    eng.generations_stalled = 0
    cands, _, _, _ = eng._collect_recurrent_candidates(eng._last_challenger_fitnesses, 2)
    assert len(cands) == 1
    assert cands[0][3] != champ_sha
