"""Archive-seeded scan-floor terminal probes."""

import json
import os

import pytest


def _live_60w_champion():
    if not os.path.isfile("multi_cycle_results.jsonl"):
        pytest.skip("needs live results jsonl")
    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("champion_words") == 60 and candidate.get("final_level") == 30:
            row = candidate
    assert row is not None
    return [int(x, 16) for x in row["champion_program"]]


def test_archive_program_probes_gate_qualified(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        get_scan_floor_gate,
        record_terminal_efficiency_scan_floor,
        terminal_archive_program_probe_variants,
    )

    prog = _live_60w_champion()
    gate = get_scan_floor_gate(prog)
    alt = list(prog)
    alt[0] ^= 1
    parents = [alt]
    if not gate.qualifies(alt):
        parents = [list(prog[:-1] + [prog[-1] ^ 1])]

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)

    qualified = [p for p in parents if gate.qualifies(p) and p != prog]
    if not qualified:
        pytest.skip("no gate-qualified archive parent for live champion")
    out = terminal_archive_program_probe_variants(
        prog, qualified, max_evals=4, start_offset=0, gate=gate
    )
    assert out
    for variant in out:
        assert len(variant) == len(prog)
        assert variant != prog
        assert gate.qualifies(variant)


def test_deterministic_archive_crossover_rotates(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        get_scan_floor_gate,
        record_terminal_efficiency_scan_floor,
        terminal_deterministic_archive_crossover_probe_variants,
    )

    prog = _live_60w_champion()
    gate = get_scan_floor_gate(prog)
    bb519_row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("cycle") == 5963 and candidate.get("champion_program"):
            bb519_row = candidate
            break
    if bb519_row is None:
        pytest.skip("needs C5963 parent in jsonl")
    parent = [int(x, 16) for x in bb519_row["champion_program"]]
    if len(parent) != len(prog) or not gate.qualifies(parent):
        pytest.skip("C5963 parent not gate-qualified same-length")

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)

    first = terminal_deterministic_archive_crossover_probe_variants(
        prog, [parent], max_evals=6, start_offset=0, gate=gate
    )
    second = terminal_deterministic_archive_crossover_probe_variants(
        prog, [parent], max_evals=6, start_offset=12, gate=gate
    )
    if not first and not second:
        pytest.skip("archive crossover produced no gate-qualified variants for live parent")
    assert first or second
    for variant in first + second:
        assert len(variant) == len(prog)
        assert variant != prog
        assert gate.qualifies(variant)


def test_archive_intercross_rotates_offsets(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        get_scan_floor_gate,
        record_terminal_efficiency_scan_floor,
        terminal_archive_intercross_probe_variants,
    )

    prog = _live_60w_champion()
    gate = get_scan_floor_gate(prog)
    parents: list[list[int]] = []
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("final_level") != 30:
            continue
        if int(candidate.get("champion_words", 0)) != len(prog):
            continue
        parent = [int(x, 16) for x in candidate["champion_program"]]
        if parent == prog or not gate.qualifies(parent):
            continue
        parents.append(parent)
        if len(parents) >= 4:
            break
    if len(parents) < 2:
        pytest.skip("need two gate-qualified same-length archive parents")

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)

    first = terminal_archive_intercross_probe_variants(
        prog, parents, max_evals=4, start_offset=0, gate=gate
    )
    second = terminal_archive_intercross_probe_variants(
        prog, parents, max_evals=4, start_offset=17, gate=gate
    )
    for variant in first + second:
        assert len(variant) == len(prog)
        assert variant != prog
        assert gate.qualifies(variant)


def test_jsonl_archive_parents_include_historical_lineage():
    from evolution.champion_rollback import collect_jsonl_archive_parents

    if not os.path.isfile("multi_cycle_results.jsonl"):
        pytest.skip("needs live results jsonl")
    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("champion_words") == 60 and candidate.get("final_level") == 30:
            row = candidate
    assert row is not None
    prog = [int(x, 16) for x in row["champion_program"]]
    parents = collect_jsonl_archive_parents(
        "multi_cycle_results.jsonl", prog, 30, max_parents=12
    )
    assert parents
    assert all(len(p) == len(prog) for p in parents)
    assert all(tuple(p) != tuple(prog) for p in parents)


def test_archive_mutate_probes_from_jsonl_parent(tmp_path, monkeypatch):
    from evolution.champion_rollback import collect_jsonl_archive_parents
    from evolution.terminal_stagnation import (
        get_scan_floor_gate,
        record_terminal_efficiency_scan_floor,
        terminal_archive_mutate_probe_variants,
    )

    if not os.path.isfile("multi_cycle_results.jsonl"):
        pytest.skip("needs live results jsonl")
    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("cycle") == 5964:
            row = candidate
            break
    assert row is not None
    prog = [int(x, 16) for x in row["champion_program"]]
    parents = collect_jsonl_archive_parents(
        "multi_cycle_results.jsonl", prog, 30, max_parents=12
    )
    if not parents:
        pytest.skip("no jsonl archive parents")

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    gate = get_scan_floor_gate(prog)
    out = terminal_archive_mutate_probe_variants(
        prog, parents, max_evals=8, start_offset=0, gate=gate
    )
    for variant in out:
        assert len(variant) == len(prog)
        assert variant != prog
        assert gate.qualifies(variant)


def test_scan_floor_probe_pool_returns_expanded_cycle_pool(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor, select_terminal_probes_for_batch

    prog = _live_60w_champion()
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    monkeypatch.setenv("EE_SCAN_FLOOR_PROBE_POOL", "24")
    monkeypatch.setenv("EE_SCAN_FLOOR_CAPABILITY_RANK_BUDGET", "0")
    pool = select_terminal_probes_for_batch(
        prog, 30, __import__("random").Random(0), n_probe=8, pop_size=12
    )
    assert len(pool) >= 8
    assert len(pool) <= 24


def test_scan_floor_capability_rank_budget_when_stalled(monkeypatch):
    from evolution.terminal_stagnation import _scan_floor_capability_rank_budget

    monkeypatch.setenv("EE_SCAN_FLOOR_CAPABILITY_RANK_BUDGET", "12")
    monkeypatch.setenv("EE_SCAN_FLOOR_CAPABILITY_RANK_BUDGET_STALLED", "16")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    assert _scan_floor_capability_rank_budget() == 16


def test_stalled_scan_floor_prioritizes_archive_before_wide_imm(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        select_terminal_probes_for_batch,
    )

    prog = _live_60w_champion()
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    marker = [0x00DEAD00, 0x00100073]
    archive_variant = list(prog[:-1]) + marker[:1]
    archive_variant[-1] = prog[-1]
    archive_variant[0] = 0x00DEAD00

    def _fake_segment(*_a, **_k):
        return [archive_variant]

    def _fake_wide(*_a, **_k):
        return [list(prog) for _ in range(40)]

    monkeypatch.setattr(
        "evolution.champion_rollback.collect_jsonl_long_archive_parents",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_long_archive_compress_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_long_parent_window_graft_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_archive_consensus_graft_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_dual_word_diff_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_archive_intercross_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_segment_archive_crossover_probe_variants",
        _fake_segment,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_deterministic_archive_crossover_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_archive_word_diff_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_archive_program_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_archive_mutate_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_wide_imm_sweep_probe_variants",
        _fake_wide,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_wide_imm_perturb_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_equal_length_refine_variants",
        lambda *_a, **_k: [],
    )

    class _Gate:
        def qualifies(self, variant):
            return True

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _p: _Gate(),
    )
    monkeypatch.setenv("EE_SCAN_FLOOR_PROBE_POOL", "16")
    monkeypatch.setenv("EE_SCAN_FLOOR_CAPABILITY_RANK_BUDGET", "0")
    monkeypatch.setenv("EE_SCAN_FLOOR_CAPABILITY_PROBE_RESERVE", "0")
    parents = [archive_variant]
    pool = select_terminal_probes_for_batch(
        prog,
        30,
        __import__("random").Random(0),
        n_probe=8,
        pop_size=12,
        archive_parents=parents,
    )
    assert pool
    assert archive_variant in pool[:4]


def test_stalled_scan_floor_prioritizes_capability_before_wide_imm(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        select_terminal_probes_for_batch,
    )

    prog = _live_60w_champion()
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    high = list(prog)
    high[1] = int(high[1]) ^ 0x1000
    low = list(prog)
    low[2] = int(low[2]) ^ 0x1000

    def _fake_refine(*_a, **_k):
        return [low, high]

    def _fake_wide(*_a, **_k):
        return [list(prog) for _ in range(40)]

    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_equal_length_refine_variants",
        _fake_refine,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_wide_imm_sweep_probe_variants",
        _fake_wide,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_wide_imm_perturb_probe_variants",
        lambda *_a, **_k: [],
    )

    def _fake_score(_champ, variant):
        return 400.0 if variant[1] == high[1] else 200.0

    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_capability_score",
        _fake_score,
    )

    class _Gate:
        def qualifies(self, variant):
            return True

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _p: _Gate(),
    )
    monkeypatch.setenv("EE_SCAN_FLOOR_PROBE_POOL", "16")
    monkeypatch.setenv("EE_SCAN_FLOOR_CAPABILITY_RANK_BUDGET", "0")
    monkeypatch.setenv("EE_SCAN_FLOOR_CAPABILITY_PROBE_RESERVE", "8")
    pool = select_terminal_probes_for_batch(
        prog, 30, __import__("random").Random(0), n_probe=8, pop_size=12
    )
    assert pool
    assert pool[0] == high


def test_rank_scan_floor_picked_probes_prefers_capability(monkeypatch):
    from evolution.terminal_stagnation import _rank_scan_floor_picked_probes

    champ = [1, 2, 3, 0x00100073]
    low = [1, 2, 4, 0x00100073]
    high = [1, 2, 5, 0x00100073]
    picked = [low, high]

    def _fake_score(_champ, variant):
        return 300.0 if variant[2] == 5 else 200.0

    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_capability_score",
        _fake_score,
    )
    ranked = _rank_scan_floor_picked_probes(champ, picked, budget=2)
    assert ranked[0] == high


def test_archive_word_diff_probes_gate_qualified(tmp_path, monkeypatch):
    from evolution.champion_rollback import collect_jsonl_archive_parents
    from evolution.terminal_stagnation import (
        get_scan_floor_gate,
        record_terminal_efficiency_scan_floor,
        terminal_archive_word_diff_probe_variants,
    )

    prog = _live_60w_champion()
    parents = collect_jsonl_archive_parents(
        "multi_cycle_results.jsonl", prog, 30, max_parents=12
    )
    if not parents:
        pytest.skip("no jsonl archive parents")

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    gate = get_scan_floor_gate(prog)
    out = terminal_archive_word_diff_probe_variants(
        prog, parents, max_evals=16, start_offset=0, gate=gate
    )
    assert out
    for variant in out:
        assert len(variant) == len(prog)
        assert variant != prog
        assert gate.qualifies(variant)


def test_segment_archive_crossover_probes_gate_qualified(tmp_path, monkeypatch):
    from evolution.champion_rollback import collect_jsonl_archive_parents
    from evolution.terminal_stagnation import (
        get_scan_floor_gate,
        record_terminal_efficiency_scan_floor,
        terminal_segment_archive_crossover_probe_variants,
    )

    prog = _live_60w_champion()
    parents = collect_jsonl_archive_parents(
        "multi_cycle_results.jsonl", prog, 30, max_parents=12
    )
    if not parents:
        pytest.skip("no jsonl archive parents")

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    gate = get_scan_floor_gate(prog)
    out = terminal_segment_archive_crossover_probe_variants(
        prog, parents, max_evals=12, start_offset=0, gate=gate
    )
    for variant in out:
        assert len(variant) == len(prog)
        assert variant != prog
        assert gate.qualifies(variant)


def test_scan_floor_model_guided_dual_slot_when_capability_stalled(tmp_path, monkeypatch):
    from unittest.mock import patch

    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor
    from evolution_engine import EvolutionConfig, EvolutionEngine

    prog = _live_60w_champion()
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, model_guided_ratio=None))
    eng.champion = prog
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    challengers = [list(prog) for _ in range(12)]
    eng._last_challenger_strategies = ["DIRECTED"] * 12

    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text(
        '{"cycles_at_level": 12}', encoding="utf-8"
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    monkeypatch.setenv("EE_SCAN_FLOOR_MODEL_GUIDED", "1")
    eng.config.max_generations = 30
    eng._scan_floor_inject_generation = 29
    eng.current_generation = 29
    eng._cycle_number = 6280

    calls: list[tuple[bool | None, int | None]] = []

    def _fake_replace(_challengers, **_kwargs):
        calls.append(
            (
                _kwargs.get("scan_floor_growth_mode"),
                _kwargs.get("growth_anchor_rank_offset"),
            )
        )
        return [len(calls) + 2]

    with patch.object(eng, "_replace_with_model_guided", side_effect=_fake_replace):
        with patch(
            "evolution.terminal_stagnation._scan_floor_capability_stalled",
            return_value=True,
        ):
            eng._apply_model_guided_slots(challengers)
    # Dual stalled slots (growth False then True) plus the dual/triple
    # anchor injects that fire when generation == inject generation
    # (EE_GROWTH_MG_DUAL_ANCHOR_INJECT / _TRIPLE_ default on; offsets 1, 2).
    assert calls == [(False, None), (True, None), (True, 1), (True, 2)]


def test_scan_floor_model_guided_fires_when_ratio_none(tmp_path, monkeypatch):
    from unittest.mock import patch

    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor
    from evolution_engine import EvolutionConfig, EvolutionEngine

    prog = _live_60w_champion()
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, model_guided_ratio=None))
    eng.champion = prog
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    challengers = [list(prog) for _ in range(12)]
    eng._last_challenger_strategies = ["DIRECTED"] * 12

    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text(
        '{"cycles_at_level": 12}', encoding="utf-8"
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    monkeypatch.setenv("EE_SCAN_FLOOR_MODEL_GUIDED", "1")
    eng.config.max_generations = 30
    eng._scan_floor_inject_generation = 29
    eng.current_generation = 29
    eng._cycle_number = 6280

    with patch.object(eng, "_replace_with_model_guided", return_value=[4]) as mock_replace:
        with patch(
            "evolution.terminal_stagnation._scan_floor_capability_stalled",
            return_value=False,
        ):
            with patch(
                "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
                return_value=False,
            ):
                eng._apply_model_guided_slots(challengers)
        mock_replace.assert_called_once()
    assert eng._last_challenger_strategies[4] == "MODEL_GUIDED"


def test_replace_with_model_guided_mutates_live_challengers():
    from unittest.mock import patch

    from evolution_engine import EvolutionConfig, EvolutionEngine

    prog = _live_60w_champion()
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, model_guided_ratio=0.2))
    eng.champion = prog
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    challengers = [list(prog) for _ in range(8)]
    eng._last_challenger_strategies = ["DIRECTED"] * 8
    fake_prog = list(prog)
    fake_prog[0] ^= 1

    def _fake_inference(_prefix, _temp, _num, _counts, **_kwargs):
        challengers[2] = fake_prog
        return [2]

    with patch.object(eng, "_run_model_inference", side_effect=_fake_inference):
        replaced = eng._replace_with_model_guided(challengers, model_ratio=0.25)
    assert replaced == [2]
    assert challengers[2] == fake_prog


def test_scan_floor_compact_tail_min_position():
    from evolution.terminal_stagnation import scan_floor_compact_tail_min_position

    assert scan_floor_compact_tail_min_position(48) == 36


def test_terminal_tail_wide_imm_respects_min_position():
    from evolution.terminal_stagnation import (
        get_scan_floor_gate,
        scan_floor_compact_tail_min_position,
        terminal_tail_wide_imm_probe_variants,
    )

    prog = _live_60w_champion()[:48]
    if prog[-1] != 0x00100073:
        prog = prog[:47] + [0x00100073]
    gate = get_scan_floor_gate(prog)
    min_pos = scan_floor_compact_tail_min_position(len(prog))
    out = terminal_tail_wide_imm_probe_variants(
        prog,
        min_position=min_pos,
        max_evals=32,
        start_offset=0,
        gate=gate,
    )
    for variant in out:
        assert len(variant) == len(prog)
        diffs = [i for i, (a, b) in enumerate(zip(prog, variant)) if a != b]
        assert diffs and min(diffs) >= min_pos
        assert gate.qualifies(variant)


def test_compress_model_guided_growth_candidate_greedy_drop(monkeypatch):
    from evolution.mutation_helpers import compress_model_guided_growth_candidate

    champ = [10, 20, 30, 40, 0x00100073]
    grown = [10, 20, 99, 30, 88, 40, 0x00100073]

    def _compress_ok(_c, variant):
        return len(variant) >= len(champ)

    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_compress_drop_qualifies",
        _compress_ok,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.semantic_compress",
        lambda variant: list(variant),
    )

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(champ)

    out = compress_model_guided_growth_candidate(
        champ,
        grown,
        gate=_Gate(),
        max_drop_trials=8,
    )
    assert out == champ


def test_scan_floor_growth_compress_drop_qualifies_intermediate():
    from evolution.mutation_helpers import (
        scan_floor_growth_compress_drop_qualifies,
        scan_floor_growth_heldout_qualifies,
    )

    champ = [1, 2, 3, 0x00100073]
    longer = [1, 2, 9, 3, 0x00100073]
    assert scan_floor_growth_heldout_qualifies(champ, longer) is True
    assert scan_floor_growth_heldout_qualifies(champ, champ) is False
    assert scan_floor_growth_compress_drop_qualifies(champ, longer) is True


def test_scan_floor_growth_compress_drop_qualifies_equal_length_requires_gate(monkeypatch):
    from evolution.mutation_helpers import scan_floor_growth_compress_drop_qualifies

    champ = [1, 2, 3, 0x00100073]
    bad = [1, 9, 3, 0x00100073]

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(champ)

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: _Gate(),
    )
    assert scan_floor_growth_compress_drop_qualifies(champ, champ) is True
    assert scan_floor_growth_compress_drop_qualifies(champ, bad) is False


def test_terminal_dual_imm_probe_variants_smoke():
    from evolution.terminal_stagnation import (
        get_scan_floor_gate,
        terminal_dual_imm_probe_variants,
    )

    champ = _live_60w_champion()[:48]
    if champ[-1] != 0x00100073:
        champ = champ[:47] + [0x00100073]
    gate = get_scan_floor_gate(champ)
    out = terminal_dual_imm_probe_variants(
        champ,
        min_position=0,
        max_evals=16,
        start_offset=0,
        gate=gate,
    )
    assert isinstance(out, list)
    assert all(len(v) == len(champ) for v in out)


def test_compress_pair_drop_reaches_champion_length(monkeypatch):
    from evolution.mutation_helpers import compress_model_guided_growth_candidate

    champ = [10, 20, 30, 40, 0x00100073]
    grown = [10, 99, 20, 88, 30, 40, 0x00100073]

    def _drop_ok(_c, variant):
        return len(variant) >= len(champ)

    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_compress_drop_qualifies",
        _drop_ok,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.semantic_compress",
        lambda variant: list(variant),
    )

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(champ)

    out = compress_model_guided_growth_candidate(
        champ,
        grown,
        gate=_Gate(),
        max_drop_trials=2,
    )
    assert out == champ


def test_compress_growth_probe_pool_repairs_gate_near_miss(monkeypatch):
    from evolution.mutation_helpers import compress_growth_probe_pool

    champ = [10, 20, 30, 40, 0x00100073]
    grown = [10, 20, 99, 30, 88, 40, 0x00100073]
    repaired = [10, 20, 77, 40, 0x00100073]

    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda _c, _v, gate=None, **kwargs: [10, 20, 99, 40, 0x00100073],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._repair_compressed_growth_probe",
        lambda _c, _v, gate=None, **kwargs: repaired,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, variant, gate=None, **kwargs: tuple(variant) == tuple(repaired),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 400.0 if tuple(prog) == tuple(repaired) else 300.0,
    )

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(repaired)

    pool = compress_growth_probe_pool(champ, [grown], gate=_Gate())
    assert pool[0] == repaired


def test_build_scan_floor_growth_crossover_pool_repairs_gate_near_miss(monkeypatch):
    from evolution.mutation_helpers import build_scan_floor_growth_crossover_pool

    champ = [10, 20, 30, 40, 0x00100073]
    fills = [[9, 8, 7, 6, 5, 0x00100073]]
    repaired = [10, 20, 77, 40, 0x00100073]

    monkeypatch.setattr(
        "mutation_proposer._crossover_mutation",
        lambda program, task_level, rng, champion, champion_crossover_prob=0.5: [
            10,
            20,
            99,
            30,
            88,
            40,
            0x00100073,
        ],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_heldout_qualifies",
        lambda _c, _v: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda _c, _v, gate=None, **kwargs: [10, 20, 99, 40, 0x00100073],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._repair_compressed_growth_probe",
        lambda _c, _v, gate=None, **kwargs: repaired,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, variant, gate=None: tuple(variant) == tuple(repaired),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 324.24 if tuple(prog) == tuple(repaired) else 324.23,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_min_raw_gain",
        lambda _n: 0.01,
    )

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(repaired)

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: _Gate(),
    )

    pool = build_scan_floor_growth_crossover_pool(
        champ,
        fills,
        __import__("random").Random(0),
        max_attempts=2,
    )
    assert pool
    assert pool[0] == repaired


def test_maybe_anchor_champion_at_scan_floor_restores_certified_bytes(monkeypatch):
    from evolution_engine import EvolutionConfig, EvolutionEngine

    live = [1, 2, 3, 0x00100073]
    anchor = [9, 8, 7, 6, 5, 0x00100073]
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = list(live)
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 500.0
    eng.champion_trace = {"status": "ok", "instructions_count": len(live), "final_regs": {}}

    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_plateau_armed",
        lambda: (True, len(anchor)),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda _c: list(anchor),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 324.24 if len(prog) == len(anchor) else 300.0,
    )
    monkeypatch.setattr(eng, "_evaluate", lambda _p: eng.champion_trace)
    monkeypatch.setattr(eng, "_verify_champion", lambda: (True, 1.0))
    monkeypatch.setattr(eng, "_compute_fitness", lambda _p, _t: 500.0)

    assert eng._maybe_anchor_champion_at_scan_floor() is True
    assert eng.champion == anchor


def test_build_scan_floor_growth_crossover_pool_accepts_near_length_heldout(monkeypatch):
    from evolution.mutation_helpers import (
        build_scan_floor_growth_crossover_pool,
        scan_floor_growth_crossover_max_words,
    )

    champ = [1, 2, 3, 0x00100073]
    fills = [[9, 8, 7, 6, 5, 0x00100073]]
    hybrid = [1, 2, 99, 3, 0x00100073]

    monkeypatch.setattr(
        "mutation_proposer._crossover_mutation",
        lambda program, task_level, rng, champion, champion_crossover_prob=0.5: hybrid,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_heldout_qualifies",
        lambda _c, variant: tuple(variant) == tuple(hybrid),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda _c, variant, gate=None, **kwargs: list(variant),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: False})(),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 400.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_min_raw_gain",
        lambda _n: 0.01,
    )

    assert scan_floor_growth_crossover_max_words(len(champ)) == len(champ) + 6
    pool = build_scan_floor_growth_crossover_pool(
        champ,
        fills,
        __import__("random").Random(0),
        max_attempts=2,
    )
    assert pool == [hybrid]


def test_build_scan_floor_growth_crossover_pool_dedupes(monkeypatch):
    from evolution.mutation_helpers import build_scan_floor_growth_crossover_pool

    champ = [1, 2, 3, 0x00100073]
    fills = [[9, 8, 7, 6, 0x00100073], [4, 5, 6, 7, 0x00100073]]

    monkeypatch.setattr(
        "mutation_proposer._crossover_mutation",
        lambda program, task_level, rng, champion, champion_crossover_prob=0.5: [
            1,
            99,
            3,
            0x00100073,
        ],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_heldout_qualifies",
        lambda _c, _v: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 400.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_min_raw_gain",
        lambda _n: 0.01,
    )

    pool = build_scan_floor_growth_crossover_pool(
        champ,
        fills,
        __import__("random").Random(0),
        max_attempts=4,
    )
    assert len(pool) == 1
    assert pool[0] == [1, 99, 3, 0x00100073]


def test_collect_scan_floor_growth_crossover_fills_merges_material(monkeypatch):
    from evolution.mutation_helpers import collect_scan_floor_growth_crossover_fills

    champ = [1, 2, 3, 0x00100073]
    ranked = [[1, 2, 3, 4, 0x00100073]]
    material = [[1, 2, 3, 5, 6, 0x00100073]]
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.ranked_scan_floor_growth_fills",
        lambda _a: ranked,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.recent_scan_floor_growth_challenger_fills",
        lambda _a, **kwargs: [],
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.collect_growth_material_parents",
        lambda _a, **kwargs: material,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: False,
    )
    merged = collect_scan_floor_growth_crossover_fills(champ)
    assert merged == ranked + material


def test_collect_scan_floor_growth_crossover_fills_prefers_shorter_parents(monkeypatch):
    from evolution.mutation_helpers import collect_scan_floor_growth_crossover_fills

    champ = [1, 2, 3, 0x00100073]
    long = [1, 2, 3, 4, 5, 6, 7, 8, 0x00100073]
    short = [1, 2, 3, 4, 0x00100073]
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: False,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.ranked_scan_floor_growth_fills",
        lambda _a: [long],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.recent_scan_floor_growth_challenger_fills",
        lambda _a, **kwargs: [],
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.collect_growth_material_parents",
        lambda _a, **kwargs: [short],
    )
    merged = collect_scan_floor_growth_crossover_fills(champ)
    assert merged[0] == short


def test_recent_challenger_fills_pulls_near_length_from_jsonl(tmp_path, monkeypatch):
    from evolution.mutation_helpers import (
        collect_scan_floor_growth_crossover_fills,
        recent_scan_floor_growth_challenger_fills,
    )

    champ = [0x10, 0x20, 0x30, 0x00100073]
    near = [0x10, 0x20, 0x30, 0x40, 0x00100073]
    jsonl = tmp_path / "multi_cycle_results.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "cycle": 99,
                "final_level": 30,
                "champion_words": len(near),
                "champion_program": [f"{w:08X}" for w in near],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._evolution_project_root",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_plateau_armed",
        lambda: (True, len(champ)),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _a, _p: True,
    )
    challengers = recent_scan_floor_growth_challenger_fills(champ)
    assert challengers == [near]
    monkeypatch.setattr(
        "evolution.terminal_stagnation.ranked_scan_floor_growth_fills",
        lambda _a: [],
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.collect_growth_material_parents",
        lambda _a, **kwargs: [],
    )
    merged = collect_scan_floor_growth_crossover_fills(champ)
    assert merged == [near]


def test_scan_floor_breeding_active_when_live_champion_shorter_than_floor(monkeypatch):
    from evolution.mutation_helpers import scan_floor_breeding_active

    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_plateau_armed",
        lambda: (True, 48),
    )
    assert scan_floor_breeding_active(27, level=30, verify=1.0) is True
    assert scan_floor_breeding_active(48, level=30, verify=1.0) is True
    assert scan_floor_breeding_active(27, level=29, verify=1.0) is False


def test_growth_crossover_uses_certified_anchor_when_live_champion_shorter(monkeypatch):
    from evolution_engine import EvolutionConfig, EvolutionEngine

    live = [1, 2, 3, 0x00100073]
    anchor = [9, 8, 7, 6, 0x00100073]
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = live
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    eng._cycle_number = 100
    eng.current_generation = 0
    eng.generations_stalled = 5
    eng.rng = __import__("random").Random(0)

    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_breeding_active",
        lambda _w, level=30, verify=1.0: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda _c: list(anchor),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_scan_floor_growth_crossover_fills",
        lambda _a, **kwargs: [[10, 11, 12, 13, 0x00100073]],
    )
    built: list[list[int]] = []

    def _build(champion, fills, rng, **kwargs):
        built.append(list(champion))
        return [list(champion)]

    monkeypatch.setattr(
        "evolution.mutation_helpers.build_scan_floor_growth_crossover_pool",
        _build,
    )

    out = eng._try_scan_floor_growth_crossover()
    assert out == anchor
    assert built == [anchor]


def test_scan_floor_growth_crossover_cache_reused(monkeypatch):
    from evolution_engine import EvolutionConfig, EvolutionEngine

    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = [1, 2, 3, 0x00100073]
    eng.champion_level = 30
    eng._cycle_number = 99
    eng.current_generation = 0
    eng.rng = __import__("random").Random(0)
    calls = {"n": 0}

    def _pool(*_a, **_k):
        calls["n"] += 1
        return [[1, 9, 3, 0x00100073]]

    monkeypatch.setattr(
        "evolution.mutation_helpers.build_scan_floor_growth_crossover_pool",
        _pool,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_breeding_active",
        lambda _w, level=30, verify=1.0: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda p: list(p),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_scan_floor_growth_crossover_fills",
        lambda _a, **kwargs: [[9, 8, 7, 0x00100073]],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda p: p,
    )

    first = eng._try_scan_floor_growth_crossover()
    second = eng._try_scan_floor_growth_crossover()
    assert first == second
    assert calls["n"] == 1


def test_scan_floor_growth_crossover_variants_rotates_ranked_pool(monkeypatch):
    from evolution.mutation_helpers import scan_floor_growth_crossover_variants

    champ = [1, 2, 3, 0x00100073]
    pool = [[1, 2, 88, 0x00100073], [1, 99, 3, 0x00100073]]
    monkeypatch.setattr(
        "evolution.mutation_helpers.build_scan_floor_growth_crossover_pool",
        lambda *_a, **_k: list(pool),
    )

    out = scan_floor_growth_crossover_variants(
        champ,
        [],
        __import__("random").Random(0),
        cycle_no=0,
        gen_idx=1,
    )
    assert out == pool[1]


def test_scan_floor_growth_inject_rank_prefers_prom_beating_shorter_growth(monkeypatch):
    from evolution.mutation_helpers import scan_floor_growth_inject_rank

    champ = [1, 2, 3, 0x00100073]
    beat55 = [1, 2, 3, 4, 0x00100073]
    bloat70 = [1] * 69 + [0x00100073]

    class _Gate:
        def qualifies(self, variant):
            return False

    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 325.0 if len(prog) == 5 else (324.5 if len(prog) == 70 else 324.0),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, gate=None: len(_v) > len(_c),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_min_raw_gain",
        lambda _n: 0.01,
    )

    rank_beat = scan_floor_growth_inject_rank(champ, beat55, gate=_Gate())
    rank_bloat = scan_floor_growth_inject_rank(champ, bloat70, gate=_Gate())
    assert rank_beat > rank_bloat


def test_scan_floor_growth_inject_rank_stalled_prefers_growth_phase_longer(monkeypatch):
    from evolution.mutation_helpers import (
        clear_scan_floor_inject_caches,
        scan_floor_growth_inject_rank,
    )

    # Module-level prom/l10 cycle caches retain values from earlier tests in
    # the same session; clear them so this test's fitness mock is authoritative.
    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 0x00100073]
    eq_tie = [1, 2, 9, 0x00100073]
    growth54 = [1, 2, 3, 4, 5, 0x00100073]

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) in {tuple(champ), tuple(eq_tie)}

    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 325.5 if len(prog) == 4 else 325.03,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, gate=None: True,
    )

    rank_growth = scan_floor_growth_inject_rank(
        champ, growth54, gate=_Gate(), stalled=True
    )
    rank_eq_stalled = scan_floor_growth_inject_rank(
        champ, eq_tie, gate=_Gate(), stalled=True
    )
    assert rank_growth > rank_eq_stalled


def test_growth_inject_merges_crossover_hybrids(tmp_path, monkeypatch):
    from evolution_engine import EvolutionConfig, EvolutionEngine
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    champ = [1, 2, 3, 0x00100073]
    fill = [9, 8, 7, 6, 5, 0x00100073]
    hybrid = [1, 2, 99, 3, 0x00100073]
    eng = EvolutionEngine(
        config=EvolutionConfig(max_level=30, model_guided_ratio=None, population_size=12)
    )
    eng.champion = list(champ)
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    eng.config.max_generations = 30
    eng.current_generation = 28
    eng._scan_floor_growth_inject_generation = 28
    eng._cycle_number = 42
    eng.rng.seed(0)
    challengers = [list(champ) for _ in range(12)]
    eng._last_challenger_strategies = ["DIRECTED"] * 12

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, len(champ), 1.0)
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _n: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_capability_growth_allowed",
        lambda _n: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: False,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_injection_generation",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_scan_floor_growth_crossover_fills",
        lambda _c, max_fills=12: [list(fill)],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.build_scan_floor_growth_crossover_pool",
        lambda _c, _f, _r, **kwargs: [list(hybrid)],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_growth_tail_perturb_variants",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "evolution.epi_momentum_governor.promotion_raw_capability_stalled",
        lambda _r: True,
    )
    monkeypatch.setattr(
        "evolution.epi_momentum_governor.scan_floor_length_prom_stalled",
        lambda _r, _n: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_compress_sources",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_scan_floor_fresh_compression_variants",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.ranked_scan_floor_compression_variants",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(eng, "_terminal_probe_pool", lambda _pop: [])
    monkeypatch.setattr(eng, "_inject_diversity", lambda _c, _n: None)
    monkeypatch.setenv("EE_SCAN_FLOOR_GROWTH_SLOTS", "12")

    eng._inject_terminal_verify_probes(challengers, 12)
    assert hybrid in challengers
    assert "GROWTH_CROSSOVER" in eng._last_challenger_strategies


def test_compress_growth_probe_pool_frontloads_equal_length(monkeypatch):
    from evolution.mutation_helpers import compress_growth_probe_pool

    champ = [10, 20, 30, 40, 0x00100073]
    grown = [10, 20, 99, 30, 88, 40, 0x00100073]
    longer = [10, 20, 9, 30, 40, 0x00100073]

    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_compress_drop_qualifies",
        lambda _c, _v: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.semantic_compress",
        lambda variant: list(variant),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda _p: 400.0,
    )

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(champ)

    pool = compress_growth_probe_pool(champ, [longer, grown], gate=_Gate())
    assert pool[0] == champ


def test_scan_floor_growth_completion_qualifies_equal_length():
    from evolution.mutation_helpers import (
        scan_floor_growth_completion_qualifies,
        scan_floor_growth_heldout_qualifies,
    )

    champ = _live_60w_champion()[:48]
    if champ[-1] != 0x00100073:
        champ = champ[:47] + [0x00100073]

    class _Gate:
        def qualifies(self, variant):
            return len(variant) == len(champ)

    assert scan_floor_growth_heldout_qualifies(champ, champ) is False
    assert scan_floor_growth_completion_qualifies(champ, champ, gate=_Gate()) is True


def test_compress_model_guided_growth_candidate_prefers_shorter(monkeypatch):
    from evolution.mutation_helpers import compress_model_guided_growth_candidate

    champ = _live_60w_champion()[:48]
    if champ[-1] != 0x00100073:
        champ = champ[:47] + [0x00100073]
    grown = list(champ) + [champ[1]]
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_heldout_qualifies",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_compression_probe_qualifies",
        lambda *_a, **_k: True,
    )

    class _Gate:
        def qualifies(self, variant):
            return len(variant) == len(champ)

    out = compress_model_guided_growth_candidate(
        champ,
        grown,
        gate=_Gate(),
        max_drop_trials=12,
    )
    assert len(out) <= len(grown)


def test_lock_model_guided_champion_prefix_keeps_head():
    from evolution.mutation_helpers import lock_model_guided_champion_prefix

    champ = [10, 20, 30, 40, 0x00100073]
    variant = [99, 88, 30, 44, 0x00100073]
    locked = lock_model_guided_champion_prefix(variant, champ, 2)
    assert locked == [10, 20, 30, 44, 0x00100073]


def test_normalize_model_guided_program_matches_champion_length():
    from evolution.mutation_helpers import normalize_model_guided_program

    champ = [1, 2, 3, 0x00100073]
    assert normalize_model_guided_program([9, 8, 0x00100073], champ) == [9, 8, 3, 0x00100073]
    long_prog = list(range(20)) + [0x00100073]
    out = normalize_model_guided_program(long_prog, champ)
    assert out is not None and len(out) == len(champ) and out[-1] == 0x00100073


def test_scan_floor_model_guided_inference_params_use_champion_tail():
    from evolution.mutation_helpers import scan_floor_model_guided_inference_params

    champ = [f"0x{w:08x}" for w in range(60)]
    archive = [[f"0x{w:08x}" for w in range(60, 120)]]
    prefix, counts, name = scan_floor_model_guided_inference_params(
        champ, archive, cycle_offset=0, generation=0
    )
    assert prefix == champ
    assert counts[0] >= 56
    assert counts[0] < len(champ)
    assert "champion-tail-complete" in name


def test_scan_floor_model_guided_inference_params_compact_champion():
    from evolution.mutation_helpers import scan_floor_model_guided_inference_params

    champ = [f"0x{w:08x}" for w in range(48)]
    prefix, counts, name = scan_floor_model_guided_inference_params(
        champ, [], cycle_offset=0, generation=0
    )
    assert prefix == champ
    assert 1 <= counts[0] < len(champ)
    assert counts[0] <= 48 - 2
    assert "champion-tail-complete" in name


def test_scan_floor_model_guided_slot_allowed(tmp_path, monkeypatch):
    from evolution.mutation_helpers import scan_floor_model_guided_slot_allowed
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    monkeypatch.setenv("EE_SCAN_FLOOR_MODEL_GUIDED", "1")
    assert (
        scan_floor_model_guided_slot_allowed(
            30, 1.0, 60, 8, generation=29, max_generations=30
        )
        is True
    )
    assert (
        scan_floor_model_guided_slot_allowed(
            30, 1.0, 60, 8, generation=0, max_generations=30
        )
        is False
    )
    assert scan_floor_model_guided_slot_allowed(30, 1.0, 60, 8, generation=5) is False
    assert scan_floor_model_guided_slot_allowed(30, 1.0, 60, 3, generation=0) is False
    assert scan_floor_model_guided_slot_allowed(28, 1.0, 60, 8, generation=0) is False


def test_scan_floor_model_guided_auto_when_local_length_stalled(tmp_path, monkeypatch):
    import json

    from evolution.mutation_helpers import scan_floor_model_guided_slot_allowed
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 54, 1.0)
    monkeypatch.delenv("EE_SCAN_FLOOR_MODEL_GUIDED", raising=False)
    results = tmp_path / "multi_cycle_results.jsonl"
    with open(results, "w", encoding="utf-8") as f:
        for i in range(8):
            f.write(
                json.dumps(
                    {
                        "cycle": 7000 + i,
                        "final_level": 30,
                        "promotion_raw_fitness": 324.63,
                        "champion_words": 54,
                        "champion_verify_score": 1.0,
                    }
                )
                + "\n"
            )
        for i in range(12):
            f.write(
                json.dumps(
                    {
                        "cycle": 7100 + i,
                        "final_level": 30,
                        "promotion_raw_fitness": 325.0 + (0.02 if i == 11 else 0.0),
                        "champion_words": 60,
                    }
                )
                + "\n"
            )
    assert (
        scan_floor_model_guided_slot_allowed(
            30,
            1.0,
            54,
            8,
            generation=29,
            max_generations=30,
            project_root=str(tmp_path),
        )
        is True
    )


def test_scan_floor_model_guided_auto_when_promotion_raw_stalled(tmp_path, monkeypatch):
    import json

    from evolution.mutation_helpers import scan_floor_model_guided_slot_allowed
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 61, 1.0)
    monkeypatch.delenv("EE_SCAN_FLOOR_MODEL_GUIDED", raising=False)
    results = tmp_path / "multi_cycle_results.jsonl"
    with open(results, "w", encoding="utf-8") as f:
        for i in range(25):
            f.write(
                json.dumps(
                    {
                        "cycle": 6000 + i,
                        "final_level": 30,
                        "promotion_raw_fitness": 321.85,
                        "champion_words": 61,
                    }
                )
                + "\n"
            )
    assert (
        scan_floor_model_guided_slot_allowed(
            30,
            1.0,
            61,
            8,
            generation=29,
            max_generations=30,
            project_root=str(tmp_path),
        )
        is True
    )
    assert (
        scan_floor_model_guided_slot_allowed(
            30,
            1.0,
            61,
            8,
            generation=5,
            max_generations=30,
            project_root=str(tmp_path),
        )
        is False
    )
    monkeypatch.setenv("EE_SCAN_FLOOR_MODEL_GUIDED", "0")
    assert (
        scan_floor_model_guided_slot_allowed(
            30,
            1.0,
            61,
            8,
            generation=29,
            max_generations=30,
            project_root=str(tmp_path),
        )
        is False
    )


def test_normalize_model_guided_tail_growth_locks_prefix():
    from evolution.mutation_helpers import normalize_model_guided_tail_growth

    champ = [1, 2, 3, 4, 0x00100073]
    raw = [1, 2, 3, 99, 88, 77, 0x00100073]
    out = normalize_model_guided_tail_growth(
        raw, champ, prefix_count=3, max_words=8
    )
    assert out == [1, 2, 3, 99, 88, 77, 0x00100073]


def test_scan_floor_model_guided_inference_params_growth_mode():
    from evolution.mutation_helpers import scan_floor_model_guided_inference_params

    champ = [f"0x{w:08x}" for w in range(61)]
    prefix, counts, name = scan_floor_model_guided_inference_params(
        champ, [], cycle_offset=0, generation=0, growth_mode=True
    )
    assert prefix == champ
    assert counts == [60]
    assert "tail-growth" in name


def test_scan_floor_compression_probe_qualifies_shorter_only():
    from evolution.terminal_stagnation import scan_floor_compression_probe_qualifies

    champ = [1, 2, 3, 4, 0x00100073]
    assert not scan_floor_compression_probe_qualifies(champ, champ)
    assert not scan_floor_compression_probe_qualifies(champ, champ + [5, 0x00100073])
    assert scan_floor_compression_probe_qualifies(champ, [1, 2, 3, 0x00100073])


def test_scan_floor_compression_promote_qualifies_requires_shorter():
    from evolution.terminal_stagnation import scan_floor_compression_promote_qualifies

    champ = [1, 2, 3, 4, 0x00100073]
    assert not scan_floor_compression_promote_qualifies(champ, champ)
    assert not scan_floor_compression_promote_qualifies(champ, champ + [5, 0x00100073])


def test_scan_floor_growth_injection_penultimate_gen():
    from evolution.mutation_helpers import (
        scan_floor_growth_injection_generation,
        scan_floor_injection_generation,
    )

    assert scan_floor_growth_injection_generation(28, max_generations=30) is True
    assert scan_floor_growth_injection_generation(29, max_generations=30) is False
    assert scan_floor_injection_generation(29, max_generations=30) is True
    assert scan_floor_injection_generation(28, max_generations=30) is False


def test_scan_floor_compression_injection_first_gen():
    from evolution.mutation_helpers import scan_floor_compression_injection_generation

    assert scan_floor_compression_injection_generation(0, max_generations=30) is True
    assert scan_floor_compression_injection_generation(1, max_generations=30) is False


def test_inject_scan_floor_eq_refine_seed_challengers(monkeypatch):
    from evolution.mutation_helpers import inject_scan_floor_eq_refine_seed_challengers

    champ = [1, 2, 3, 0x00100073]
    alt = [1, 2, 9, 0x00100073]
    challengers = [list(champ), list(champ), list(champ)]
    strategies = ["", "", ""]

    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _n: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=4, probe_material=False: [alt],
    )

    n = inject_scan_floor_eq_refine_seed_challengers(
        champ,
        challengers,
        __import__("random").Random(0),
        cycle_no=1,
        strategies=strategies,
    )
    assert n >= 1
    assert alt in challengers
    assert "EQ_REFINE_SEED" in strategies


def test_scan_floor_eq_refine_seeds_record_and_load(tmp_path, monkeypatch):
    from evolution.mutation_helpers import (
        load_scan_floor_eq_refine_seeds,
        record_scan_floor_eq_refine_seeds,
        scan_floor_eq_refine_seeds_path,
    )

    champ = [1, 2, 3, 0x00100073]
    alt = [1, 2, 9, 0x00100073]

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) in {tuple(champ), tuple(alt)}

    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: _Gate(),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_eq_refine_seeds_path",
        lambda: str(tmp_path / "eq_seeds.json"),
    )

    seed_path = str(tmp_path / "eq_seeds.json")
    # 4-word test anchor sits below the production persist floor (20 words);
    # disable the size guard so this roundtrip test exercises record/load.
    monkeypatch.setenv("EE_EQ_REFINE_PERSIST_MIN_WORDS", "0")
    record_scan_floor_eq_refine_seeds(champ, [alt, champ], cycle=42)
    loaded = load_scan_floor_eq_refine_seeds(champ, max_seeds=4)
    assert loaded == [alt]
    assert seed_path.endswith("eq_seeds.json")
    assert (tmp_path / "eq_seeds.json").is_file()


def test_scan_floor_stalled_compression_injection_dual_gen():
    from evolution.mutation_helpers import scan_floor_stalled_compression_injection_generation

    assert scan_floor_stalled_compression_injection_generation(0, max_generations=30) is True
    assert scan_floor_stalled_compression_injection_generation(28, max_generations=30) is True
    assert scan_floor_stalled_compression_injection_generation(14, max_generations=30) is False


def test_scan_floor_stalled_growth_injection_dual_gen():
    from evolution.mutation_helpers import scan_floor_stalled_growth_injection_generation

    assert scan_floor_stalled_growth_injection_generation(0, max_generations=50) is True
    assert scan_floor_stalled_growth_injection_generation(48, max_generations=50) is True
    assert scan_floor_stalled_growth_injection_generation(25, max_generations=50) is False


def test_collect_scan_floor_fresh_compression_variants_filters_shorter(monkeypatch):
    champion = [1, 2, 3, 4, 5]
    shorter = [1, 2, 3, 4]
    long_source = [1, 2, 3, 4, 5, 6]

    def fake_variants(source, rng, max_evals=8, max_drop_span=1):
        if source == champion:
            return [shorter, champion, long_source[:5]]
        return [shorter]

    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_variants",
        fake_variants,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_drop_span",
        lambda n, cap: 1,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_compression_promote_qualifies",
        lambda c, v: len(v) < len(c),
    )
    monkeypatch.setattr("evolution.champion_bloat_guard.level_word_cap", lambda lvl: 64)
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: False,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda *_a, **_k: [],
    )

    from evolution.mutation_helpers import collect_scan_floor_fresh_compression_variants

    out = collect_scan_floor_fresh_compression_variants(
        champion,
        long_sources=[long_source],
        cycle_no=1,
        gen_idx=0,
    )
    assert shorter in out
    assert all(len(v) < len(champion) for v in out)


def test_growth_material_diff_graft_variants_single_site(monkeypatch):
    from evolution.mutation_helpers import growth_material_diff_graft_variants

    champ = [1, 2, 3, 4, 0x00100073]
    material = [[1, 2, 99, 4, 0x00100073, 9, 10]]

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 400.0 if 99 in prog else 300.0,
    )

    out = growth_material_diff_graft_variants(champ, material)
    assert out == [[1, 2, 99, 4, 0x00100073]]


def test_growth_material_diff_graft_variants_pair_sites(monkeypatch):
    from evolution.mutation_helpers import growth_material_diff_graft_variants

    champ = [1, 2, 3, 4, 5, 0x00100073]
    material = [[1, 88, 3, 77, 5, 0x00100073]]

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )

    def _prom(prog):
        if 88 in prog and 77 in prog:
            return 500.0
        if 88 in prog or 77 in prog:
            return 400.0
        return 300.0

    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        _prom,
    )

    out = growth_material_diff_graft_variants(champ, material, max_pair_combos=4)
    assert [1, 88, 3, 77, 5, 0x00100073] in out
    assert out[0] == [1, 88, 3, 77, 5, 0x00100073]


def test_expand_growth_material_for_grafts_projects_tail_only(monkeypatch):
    from evolution.mutation_helpers import (
        _expand_growth_material_for_grafts,
        clear_scan_floor_inject_caches,
    )

    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 4, 0x00100073]
    tail_only = [1, 2, 3, 4, 0x00100073, 9, 10]
    projected = [1, 2, 99, 4, 0x00100073]

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.equal_length_from_longer_growth",
        lambda _c, _g, **_: list(projected),
    )

    gate = type("_G", (), {"qualifies": lambda self, v: True})()
    out = _expand_growth_material_for_grafts(
        champ, [tail_only], gate=gate, cycle_no=1
    )
    assert tail_only in out
    assert projected in out


def test_growth_material_diff_graft_variants_cross_parent_pair(monkeypatch):
    from evolution.mutation_helpers import (
        clear_scan_floor_inject_caches,
        growth_material_diff_graft_variants,
    )

    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 4, 5, 0x00100073]
    material_a = [[1, 2, 88, 4, 5, 0x00100073]]
    material_b = [[1, 2, 3, 4, 77, 0x00100073]]

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    monkeypatch.setenv("EE_GROWTH_MATERIAL_GRAFT_PAIR_BUDGET", "0")
    monkeypatch.setenv("EE_GROWTH_MATERIAL_GRAFT_CROSS_PARENT_BUDGET", "4")

    def _prom(prog):
        if 88 in prog and 77 in prog:
            return 600.0
        if 88 in prog or 77 in prog:
            return 400.0
        return 300.0

    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        _prom,
    )

    out = growth_material_diff_graft_variants(
        champ, material_a + material_b, max_pair_combos=0
    )
    assert [1, 2, 88, 4, 77, 0x00100073] in out
    assert out[0] == [1, 2, 88, 4, 77, 0x00100073]


def test_growth_material_diff_graft_variants_cycle_eval_budget(monkeypatch):
    from evolution.mutation_helpers import (
        clear_scan_floor_inject_caches,
        growth_material_diff_graft_variants,
    )

    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 4, 0x00100073]
    material_a = [[1, 2, 99, 4, 0x00100073]]
    material_b = [[1, 2, 3, 88, 0x00100073]]
    prom_calls: list[int] = []

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    monkeypatch.setenv("EE_GROWTH_MATERIAL_GRAFT_EVAL_BUDGET", "1")
    monkeypatch.setenv("EE_GROWTH_MATERIAL_GRAFT_PAIR_BUDGET", "0")

    def _prom(prog):
        prom_calls.append(1)
        return 325.627

    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        _prom,
    )

    first = growth_material_diff_graft_variants(
        champ, material_a, max_variants=4, cycle_no=42
    )
    second = growth_material_diff_graft_variants(
        champ, material_b, max_variants=4, cycle_no=42
    )
    assert len(first) == 1
    assert second == first
    assert len(prom_calls) == 5


def test_collect_scan_floor_fresh_compression_prefers_target_length_sources(
    monkeypatch,
):
    from evolution.mutation_helpers import collect_scan_floor_fresh_compression_variants

    champion = [1, 2, 3, 0x00100073]
    near = [1, 2, 3, 4, 5, 6, 7, 0x00100073]
    short = [1, 2, 3, 4, 0x00100073]
    source_lens: list[int] = []

    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 325.627 if prog == champion else (325.627 if prog == near else 325.428),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_variants",
        lambda source, *_a, **_k: source_lens.append(len(source)) or [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_drop_span",
        lambda n, cap: 1,
    )
    monkeypatch.setattr("evolution.champion_bloat_guard.level_word_cap", lambda lvl: 64)
    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda *_a, **_k: [],
    )

    collect_scan_floor_fresh_compression_variants(
        champion,
        long_sources=[short, near],
        cycle_no=1,
        gen_idx=0,
    )
    assert source_lens[0] == len(champion)
    assert source_lens[1] == len(near)


def test_terminal_efficiency_promote_skip_quiescent_when_capability_stalled(
    tmp_path, monkeypatch
):
    from evolution.terminal_stagnation import (
        record_terminal_efficiency_stable,
        terminal_efficiency_promote_quiescent,
        terminal_efficiency_promote_skip_quiescent,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_stable(6350, 61)
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    assert terminal_efficiency_promote_quiescent(6367, 61, 1.0) is True
    assert terminal_efficiency_promote_skip_quiescent(6367, 61, 1.0) is False

    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: False,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_capability_growth_allowed",
        lambda _w: False,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: False,
    )
    assert terminal_efficiency_promote_skip_quiescent(6367, 61, 1.0) is True


def test_should_rescan_terminal_efficiency_compression_at_floor(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        should_rescan_terminal_efficiency_compression,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: False,
    )
    record_terminal_efficiency_scan_floor(6543, 48, 1.0)
    assert should_rescan_terminal_efficiency_compression(6543, 48) is True
    assert should_rescan_terminal_efficiency_compression(6544, 48) is False
    assert should_rescan_terminal_efficiency_compression(6562, 48) is False
    assert should_rescan_terminal_efficiency_compression(6563, 48) is True


def test_skip_empty_compression_rescan_for_same_champion(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        _record_scan_floor_compression_rescan,
        _skip_empty_compression_rescan,
        ranked_scan_floor_compression_variants,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.should_rescan_terminal_efficiency_compression",
        lambda _c, _w, *_a: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    calls = {"n": 0}

    def _expensive(_key):
        calls["n"] += 1
        return ()

    monkeypatch.setattr(
        "evolution.terminal_stagnation._ranked_scan_floor_compression_variants_cached",
        _expensive,
    )
    prog = [0x11111111] * 40
    ranked_scan_floor_compression_variants(prog, cycle_no=7212)
    assert calls["n"] == 1
    _record_scan_floor_compression_rescan(tuple(prog), cycle=7212, hits=0)
    assert _skip_empty_compression_rescan(tuple(prog), len(prog)) is True
    ranked_scan_floor_compression_variants(prog, cycle_no=7215)
    assert calls["n"] == 1


def test_skip_empty_compression_rescan_prom_raw_tie_bytes(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        _record_scan_floor_compression_rescan,
        _skip_empty_compression_rescan,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _w: True,
    )
    base = tuple(0x11111111 + i for i in range(40))
    tie = base[:27] + (0xD8F80D93,) + base[28:30] + (0xFE18C8E3,) + base[31:]
    _record_scan_floor_compression_rescan(base, cycle=7209, hits=0)
    assert _skip_empty_compression_rescan(tie, 40) is True


def test_should_rescan_terminal_efficiency_compression_stall_interval(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        should_rescan_terminal_efficiency_compression,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    record_terminal_efficiency_scan_floor(6543, 48, 1.0)
    assert should_rescan_terminal_efficiency_compression(6544, 48) is False
    assert should_rescan_terminal_efficiency_compression(6545, 48) is False
    assert should_rescan_terminal_efficiency_compression(6546, 48) is True


def test_ranked_compression_skips_rebuild_between_rescans(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        ranked_scan_floor_compression_variants,
        record_terminal_efficiency_scan_floor,
    )

    prog = [0x11111111] * 48
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: False,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._compression_plateau_exhausted_at_words",
        lambda _w: False,
    )
    record_terminal_efficiency_scan_floor(6543, 48, 1.0)
    calls = {"n": 0}

    def _fake_ranked(key):
        calls["n"] += 1
        return ()

    monkeypatch.setattr(
        "evolution.terminal_stagnation._ranked_scan_floor_compression_variants_cached",
        _fake_ranked,
    )
    ranked_scan_floor_compression_variants(prog, cycle_no=6543)
    assert calls["n"] == 1
    ranked_scan_floor_compression_variants(prog, cycle_no=6544)
    assert calls["n"] == 1
    ranked_scan_floor_compression_variants(prog, cycle_no=6563)
    assert calls["n"] == 2


def test_at_floor_exhausted_compression_boosts_capability_budget(monkeypatch):
    import random

    from evolution.terminal_stagnation import select_terminal_probes_for_batch

    prog = [0x11111111] * 47 + [0x00100073]
    refine_budgets: list[int] = []
    sweep_budgets: list[int] = []

    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_variants",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_equal_length_refine_variants",
        lambda *a, **kw: refine_budgets.append(int(kw.get("max_evals", 0))) or [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_wide_imm_sweep_probe_variants",
        lambda *a, **kw: sweep_budgets.append(int(kw.get("max_evals", 0))) or [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _p: type("G", (), {"qualifies": lambda self, _v: True})(),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._rank_scan_floor_picked_probes",
        lambda _p, picked, **kwargs: picked,
    )
    monkeypatch.setattr("evolution.terminal_stagnation._vm_score_disk_flush", lambda: None)

    select_terminal_probes_for_batch(prog, 30, random.Random(0), n_probe=8)
    assert refine_budgets
    assert sweep_budgets
    assert refine_budgets[0] >= 96
    assert sweep_budgets[0] >= 96


def test_terminal_efficiency_promote_skip_quiescent_when_compression_hot(monkeypatch):
    from evolution.terminal_stagnation import terminal_efficiency_promote_skip_quiescent

    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_promote_quiescent",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: False,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_capability_growth_allowed",
        lambda _w: True,
    )
    assert terminal_efficiency_promote_skip_quiescent(6485, 61, 1.0) is False


def test_at_floor_compression_rank_finds_shorter_candidates(monkeypatch):
    import json
    import os

    from evolution.terminal_stagnation import (
        _ranked_scan_floor_compression_variants_cached,
        record_terminal_efficiency_scan_floor,
        scan_floor_compression_promote_qualifies,
    )

    if not os.path.isfile("multi_cycle_results.jsonl"):
        pytest.skip("needs live results jsonl")
    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if (
            candidate.get("champion_words") == 59
            and candidate.get("final_level") == 30
            and candidate.get("promotion_raw_fitness")
        ):
            row = candidate
    if row is None:
        pytest.skip("needs live 59w champion")
    prog = [int(x, 16) for x in row["champion_program"]]

    agents = os.path.join(os.path.dirname(__file__), "..", "..", ".agents")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: os.path.abspath(agents),
    )
    record_terminal_efficiency_scan_floor(6489, 59, 1.0)
    _ranked_scan_floor_compression_variants_cached.cache_clear()
    ranked = _ranked_scan_floor_compression_variants_cached(tuple(prog))
    assert ranked
    assert any(len(item) < 59 for item in ranked)
    assert any(
        scan_floor_compression_promote_qualifies(prog, list(item)) for item in ranked
    )


def test_terminal_growth_tail_perturb_variants():
    import random

    from evolution.mutation_helpers import scan_floor_growth_heldout_qualifies
    from evolution.terminal_stagnation import terminal_growth_tail_perturb_variants

    champ = [1, 2, 3, 4, 0x00100073]
    anchor = champ[:-1] + [0x01010113, 0x02020213, 0x00100073]
    rng = random.Random(0)
    out = terminal_growth_tail_perturb_variants(
        champ, anchor, rng, max_evals=4, start_offset=0
    )
    assert out
    for variant in out:
        assert len(variant) > len(champ)
        assert variant != anchor
        assert scan_floor_growth_heldout_qualifies(champ, variant)


def test_scan_floor_model_guided_inference_params_growth_anchor():
    from evolution.mutation_helpers import scan_floor_model_guided_inference_params

    champ = [f"0x{w:08x}" for w in range(61)]
    anchor = champ + [f"0x{i:08x}" for i in range(100, 111)] + ["0x00100073"]
    prefix, counts, name = scan_floor_model_guided_inference_params(
        champ,
        [],
        cycle_offset=0,
        generation=0,
        growth_mode=True,
        growth_anchor_hex=anchor,
    )
    assert prefix == anchor
    assert counts == [61]
    assert "growth-anchor-tail" in name


def test_dual_word_diff_probe_variants():
    from evolution.terminal_stagnation import terminal_dual_word_diff_probe_variants

    prog = [0x100 + i for i in range(46)] + [0x00100073, 0x00100073]
    parent = list(prog)
    parent[5] ^= 1
    parent[10] ^= 2
    gate = type("G", (), {"qualifies": lambda _self, v: len(v) == len(prog)})()
    out = terminal_dual_word_diff_probe_variants(
        prog, [parent], max_evals=4, gate=gate
    )
    assert out
    for variant in out:
        assert len(variant) == len(prog)
        assert sum(1 for a, b in zip(variant, prog) if a != b) == 2


def test_long_parent_window_graft_probe_variants():
    from evolution.terminal_stagnation import terminal_long_parent_window_graft_probe_variants

    prog = [0x100 + i for i in range(46)] + [0x00100073, 0x00100073]
    parent = list(prog) + [0x200, 0x201, 0x202, 0x203, 0x204, 0x205, 0x206, 0x207, 0x208, 0x209, 0x20A, 0x20B]
    parent[20] = 0xABCDEF00
    parent[21] = 0xABCDEF01
    parent[22] = 0xABCDEF02
    gate = type("G", (), {"qualifies": lambda _self, v: len(v) == len(prog)})()
    out = terminal_long_parent_window_graft_probe_variants(
        prog, [parent], max_evals=6, gate=gate
    )
    assert out
    for variant in out:
        assert len(variant) == len(prog)


def test_archive_consensus_graft_probe_variants(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        get_scan_floor_gate,
        record_terminal_efficiency_scan_floor,
        terminal_archive_consensus_graft_probe_variants,
    )

    prog = _live_60w_champion()
    gate = get_scan_floor_gate(prog)
    parent_a = list(prog)
    parent_b = list(prog)
    parent_a[10] ^= 1
    parent_b[10] ^= 1
    if not gate.qualifies(parent_a) or not gate.qualifies(parent_b):
        pytest.skip("consensus graft parents fail gate")

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, len(prog), 1.0)

    out = terminal_archive_consensus_graft_probe_variants(
        prog,
        [parent_a, parent_b],
        max_evals=4,
        gate=gate,
    )
    assert out
    for variant in out:
        assert len(variant) == len(prog)
        assert variant != prog
        assert gate.qualifies(variant)
        assert sum(1 for a, b in zip(variant, prog) if a != b) == 1


def test_terminal_probe_pool_includes_archive_parents(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor
    from evolution_engine import EvolutionConfig, EvolutionEngine

    prog = _live_60w_champion()
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = prog
    eng.champion_level = 30
    eng.rng.seed(0)
    eng._hall_of_fame.add(prog, 100.0, 30)
    alt = list(prog)
    alt[5] ^= 1
    eng._hall_of_fame.add(alt, 90.0, 30)

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)

    parents = eng._terminal_archive_parents()
    assert any(tuple(p) == tuple(alt) for p in parents)
    pool = eng._terminal_probe_pool(24)
    assert pool


def test_repair_model_guided_for_gate_ranks_by_prom_raw(monkeypatch):
    from evolution.mutation_helpers import repair_model_guided_for_gate

    champ = [1, 2, 3, 4, 0x00100073]
    seed = [1, 2, 99, 4, 0x00100073]

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) in {
                tuple(champ),
                (1, 2, 3, 4, 0x00100073),
                (1, 2, 88, 4, 0x00100073),
            }

    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_wide_imm_sweep_probe_variants",
        lambda *_a, **_k: [[1, 2, 3, 4, 0x00100073]],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_imm_perturb_probe_variants",
        lambda *_a, **_k: [[1, 2, 88, 4, 0x00100073]],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_tail_wide_imm_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_dual_imm_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 300.0 if prog[2] == 88 else 200.0,
    )

    out = repair_model_guided_for_gate(
        seed,
        _Gate(),
        max_evals=8,
        champion=champ,
        prefix_count=2,
    )
    assert out == [1, 2, 88, 4, 0x00100073]


def test_run_model_inference_gate_collects_best_prom_raw(monkeypatch):
    from unittest.mock import MagicMock

    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = [10, 20, 30, 40, 0x00100073]
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = champ
    eng.rng.seed(0)
    eng._last_challengers = [list(champ) for _ in range(6)]
    # Bulk stall-collect (prom-scored best-of, the path this test is named for)
    # only runs when generation == inject generation AND stall mode is active;
    # the gen-0 default skips it and the raw first completion replaces the
    # slot instead. Pin both so the best-prom_raw collect path is exercised.
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    eng.config.max_generations = 30
    eng._scan_floor_inject_generation = 29
    eng.current_generation = 29

    class _Gate:
        def qualifies(self, variant):
            return len(variant) == len(champ)

    low = [10, 20, 30, 41, 0x00100073]
    high = [10, 20, 30, 42, 0x00100073]

    def _complete(_src, prefix_count=0, **_kwargs):
        if int(prefix_count) == 2:
            return [[f"0x{w:08x}" for w in low]]
        return [[f"0x{w:08x}" for w in high]]

    eng._prefix_generator = MagicMock()
    eng._prefix_generator.complete.side_effect = _complete
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 300.0 if prog[3] == 42 else 200.0,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.repair_model_guided_for_gate",
        lambda program, gate, **_kw: list(program),
    )

    replaced = eng._run_model_inference(
        [f"0x{w:08x}" for w in champ],
        0.8,
        1,
        [2, 3],
        gate=_Gate(),
        normalize_to_champion=True,
    )
    assert replaced
    assert eng._last_challengers[replaced[0]][3] == 42


def test_repair_model_guided_for_gate_rejects_regression(monkeypatch):
    from evolution.mutation_helpers import repair_model_guided_for_gate

    champ = [1, 2, 3, 4, 0x00100073]
    seed = [1, 2, 99, 4, 0x00100073]

    class _Gate:
        def qualifies(self, variant):
            return len(variant) == len(champ)

    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 400.0 if tuple(prog) == tuple(champ) else 100.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_wide_imm_sweep_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_imm_perturb_probe_variants",
        lambda *_a, **_k: [],
    )

    assert (
        repair_model_guided_for_gate(
            seed,
            _Gate(),
            max_evals=4,
            champion=champ,
            prefix_count=2,
        )
        is None
    )


def test_compact_l30_isolates_champion_state():
    from evolution import fitness_helpers as fh
    from evolution.fitness_helpers import _isolate_capability_score_engine, decode_program_words

    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("final_level") == 30 and candidate.get("champion_program"):
            row = candidate
    assert row is not None
    champ = decode_program_words(row["champion_program"])
    assert champ is not None
    mutant = list(champ)
    # Live L30 champions vary in length (recent jsonl rows are compact, e.g.
    # 27w); flip a valid interior word instead of the fixed index 37.
    mutant[len(champ) // 2] ^= 1

    eng = fh._CAPABILITY_SCORE_ENGINE
    if eng is None:
        fh.compact_l30_promotion_raw_fitness(champ)
        eng = fh._CAPABILITY_SCORE_ENGINE
    eng.champion = list(champ)
    eng.champion_trace = {"status": "ok", "final_regs": {}, "instructions_count": 1}
    eng.champion_fitness = 999.0

    _isolate_capability_score_engine(eng)
    assert eng.champion is None
    assert eng.champion_trace is None

    clean_mutant = fh.compact_l30_promotion_raw_fitness(mutant)
    clean_champ = fh.compact_l30_promotion_raw_fitness(champ)
    assert clean_mutant + 1e-6 < clean_champ


def test_repair_model_guided_for_gate_skips_champion_echo(monkeypatch):
    from evolution.mutation_helpers import repair_model_guided_for_gate

    champ = [1, 2, 3, 4, 0x00100073]

    class _Gate:
        def qualifies(self, variant):
            return True

    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: False,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_wide_imm_sweep_probe_variants",
        lambda *_a, **_k: [list(champ)],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_imm_perturb_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda _p: 100.0,
    )

    assert (
        repair_model_guided_for_gate(
            list(champ),
            _Gate(),
            max_evals=4,
            champion=champ,
            prefix_count=2,
        )
        is None
    )


def test_scan_floor_gate_repair_variant_uses_repair_on_near_miss(monkeypatch):
    from evolution.terminal_stagnation import _scan_floor_gate_repair_variant

    champ = [10, 20, 30, 40, 0x00100073]
    near = [10, 99, 30, 40, 0x00100073]
    repaired = [10, 77, 30, 40, 0x00100073]

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(repaired)

    gate = _Gate()
    assert _scan_floor_gate_repair_variant(champ, champ, gate) is None
    assert _scan_floor_gate_repair_variant(champ, repaired, gate) == repaired
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._PROBE_GATE_REPAIR_BUDGET",
        2,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.repair_model_guided_for_gate",
        lambda _v, _g, **kwargs: repaired,
    )
    assert _scan_floor_gate_repair_variant(champ, near, gate) == repaired


def test_segment_probe_uses_gate_repair(monkeypatch):
    from evolution.terminal_stagnation import terminal_segment_archive_crossover_probe_variants

    champ = [10, 20, 30, 40, 0x00100073]
    parent = [10, 99, 30, 40, 0x00100073]
    repaired = [10, 77, 30, 40, 0x00100073]

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(repaired)

    gate = _Gate()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_gate_repair_variant",
        lambda program, variant, _gate, **kwargs: (
            repaired if tuple(variant) != tuple(champ) else None
        ),
    )
    out = terminal_segment_archive_crossover_probe_variants(
        champ,
        [parent],
        max_evals=4,
        gate=gate,
        min_span=1,
        max_span=1,
    )
    assert repaired in out
