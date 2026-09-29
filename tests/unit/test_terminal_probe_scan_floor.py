"""Terminal probe injection respects held-out scan floor."""

import json
import os

import pytest

from evolution_engine import EvolutionConfig, EvolutionEngine


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


def test_scan_floor_skips_semantic_compress_and_easy_fill(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    prog = _live_60w_champion()
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = prog
    eng.champion_level = 30
    eng.champion_trace = {"registers": {}, "memory": {}}
    eng.task_instance = {}
    eng._champion_verify_score = 1.0
    eng.rng.seed(0)
    eng._last_challenger_strategies = ["MUTATION"] * 20
    challengers = [list(prog) for _ in range(20)]

    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    monkeypatch.setattr(
        "evolution.mutation_helpers._evolution_project_root",
        lambda: str(tmp_path),
    )

    eng._inject_terminal_verify_probes(challengers, 20)

    assert eng._last_challenger_strategies[0] != "SEMANTIC_COMPRESS"
    assert eng._last_challenger_strategies[1] != "L30_EASY_FILL"


def test_scan_floor_growth_allowed_when_champion_shorter_than_floor(tmp_path, monkeypatch):
    import json

    from evolution.mutation_helpers import scan_floor_capability_growth_allowed
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 61, 1.0)
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
                        "champion_verify_score": 1.0,
                        "champion_program": [f"0x{j:08X}" for j in range(61)],
                    }
                )
                + "\n"
            )
    monkeypatch.setattr(
        "evolution.mutation_helpers._evolution_project_root",
        lambda: str(tmp_path),
    )
    assert scan_floor_capability_growth_allowed(49, str(tmp_path)) is True


def test_terminal_efficiency_rejects_bloated_tracking(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        compact_l30_certified_champion_words,
        record_terminal_efficiency_scan_floor,
        record_terminal_efficiency_stable,
        reconcile_terminal_efficiency_compact_state,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    reconcile_terminal_efficiency_compact_state(
        61, 200, heldout=1.0, verify_score=1.0
    )
    record_terminal_efficiency_stable(201, 72)
    record_terminal_efficiency_scan_floor(201, 72, 1.0)
    record_terminal_efficiency_stable(202, 23)
    record_terminal_efficiency_scan_floor(202, 23, 1.0)
    state = json.loads((agents / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert state["terminal_efficiency_stable_words"] == 61
    assert state["terminal_efficiency_scan_floor_words"] == 61
    assert reconcile_terminal_efficiency_compact_state(
        23, 203, heldout=0.73, verify_score=0.73
    ) is False
    assert compact_l30_certified_champion_words(
        {
            "final_level": 30,
            "champion_verify_score": 0.73,
            "champion_words": 23,
            "promotion_raw_fitness": 0.0,
        },
        str(tmp_path),
    ) == 61


def test_maybe_rearm_scan_floor_above_champion(tmp_path, monkeypatch):
    from evolution.mutation_helpers import (
        scan_floor_capability_growth_allowed,
        scan_floor_plateau_armed,
    )
    from evolution.terminal_stagnation import (
        maybe_rearm_scan_floor_above_champion,
        record_terminal_efficiency_scan_floor,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.heldout_l30_verify_mean",
        lambda _prog, _seeds: 1.0,
    )
    record_terminal_efficiency_scan_floor(100, 58, 1.0)
    prog = [0x13] * 60 + [0x00100073]
    assert maybe_rearm_scan_floor_above_champion(101, prog, verify_score=1.0) is True
    assert scan_floor_plateau_armed() == (True, 61)
    assert scan_floor_capability_growth_allowed(61) is True
    assert maybe_rearm_scan_floor_above_champion(102, prog, verify_score=1.0) is False


def test_ranked_scan_floor_growth_fills_skip_timeout_sentinel():
    import os

    if not os.path.isfile("multi_cycle_results.jsonl"):
        pytest.skip("needs live results jsonl")
    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("champion_words") == 61 and candidate.get("final_level") == 30:
            row = candidate
    if row is None:
        pytest.skip("needs live 61w champion")
    from evolution.terminal_stagnation import ranked_scan_floor_growth_fills

    prog = [int(x, 16) for x in row["champion_program"]]
    ranked = ranked_scan_floor_growth_fills(prog)
    assert ranked
    assert all(len(item) > len(prog) for item in ranked)
    assert max(len(item) for item in ranked) >= 72


def test_ranked_scan_floor_growth_fills_returns_multiple_at_48w():
    import os

    if not os.path.isfile("multi_cycle_results.jsonl"):
        pytest.skip("needs live results jsonl")
    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("champion_words") == 48 and candidate.get("final_level") == 30:
            row = candidate
    if row is None:
        pytest.skip("needs live 48w champion")
    from evolution.terminal_stagnation import (
        _ranked_scan_floor_growth_fills_cached,
        ranked_scan_floor_growth_fills,
    )

    prog = [int(x, 16) for x in row["champion_program"]]
    _ranked_scan_floor_growth_fills_cached.cache_clear()
    ranked = ranked_scan_floor_growth_fills(prog)
    assert ranked
    assert len(ranked) >= 2
    assert all(len(item) > len(prog) for item in ranked)


def test_growth_inject_does_not_shadow_os_module(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    if not os.path.isfile("multi_cycle_results.jsonl"):
        pytest.skip("needs live results jsonl")
    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("champion_words") == 48 and candidate.get("final_level") == 30:
            row = candidate
    if row is None:
        pytest.skip("needs live 48w champion")
    prog = [int(x, 16) for x in row["champion_program"]]
    eng = EvolutionEngine(
        config=EvolutionConfig(max_level=30, model_guided_ratio=None, population_size=12)
    )
    eng.champion = prog
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    eng.config.max_generations = 30
    eng.current_generation = 28
    eng._scan_floor_growth_inject_generation = 28
    eng.rng.seed(0)
    challengers = [list(prog) for _ in range(12)]
    eng._last_challenger_strategies = ["DIRECTED"] * 12

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 48, 1.0)
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    eng._inject_terminal_verify_probes(challengers, 12)
    assert any(s == "L30_EASY_FILL" for s in eng._last_challenger_strategies)


def test_scan_floor_injects_growth_easy_fill_when_stalled(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    if not os.path.isfile("multi_cycle_results.jsonl"):
        pytest.skip("needs live results jsonl")
    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("champion_words") == 61 and candidate.get("final_level") == 30:
            row = candidate
    if row is None:
        pytest.skip("needs live 61w champion")
    prog = [int(x, 16) for x in row["champion_program"]]

    eng = EvolutionEngine(
        config=EvolutionConfig(max_level=30, diversity_inject_ratio=0.15)
    )
    eng.champion = prog
    eng.champion_level = 30
    eng.champion_trace = {"registers": {}, "memory": {}}
    eng.task_instance = {}
    eng._champion_verify_score = 1.0
    eng.rng.seed(0)
    challengers = [list(prog) for _ in range(20)]
    eng._last_challenger_strategies = ["MUTATION"] * 20

    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 61, 1.0)
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
    monkeypatch.setattr(
        "evolution.mutation_helpers._evolution_project_root",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.select_terminal_probes_for_batch",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    from evolution.terminal_stagnation import _ranked_scan_floor_growth_fills_cached

    _ranked_scan_floor_growth_fills_cached.cache_clear()
    grown = list(prog) + [int(prog[-2])]
    monkeypatch.setattr(
        "evolution.terminal_stagnation.ranked_scan_floor_growth_fills",
        lambda _p: [grown],
    )

    eng.config.max_generations = 30
    eng.current_generation = 28
    eng._scan_floor_growth_inject_generation = 28
    eng._scan_floor_inject_generation = 29
    eng._inject_terminal_verify_probes(challengers, 20)

    assert "L30_EASY_FILL" in eng._last_challenger_strategies
    assert any(len(c) > len(prog) for c in challengers)


def test_scan_floor_empty_probes_injects_diversity(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    prog = _live_60w_champion()
    eng = EvolutionEngine(
        config=EvolutionConfig(max_level=30, diversity_inject_ratio=0.1)
    )
    eng.champion = prog
    eng.champion_level = 30
    eng.champion_trace = {"registers": {}, "memory": {}}
    eng.task_instance = {}
    eng._champion_verify_score = 1.0
    eng.rng.seed(0)
    challengers = [list(prog) for _ in range(20)]
    eng._last_challenger_strategies = ["DIRECTED"] * 20
    calls = {"n": 0}

    def fake_inject(challengers, n_inject):
        calls["n"] = int(n_inject)

    monkeypatch.setattr(eng, "_inject_diversity", fake_inject)
    monkeypatch.setattr(
        "evolution.terminal_stagnation.select_terminal_probes_for_batch",
        lambda *a, **k: [],
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)

    eng._inject_terminal_verify_probes(challengers, 20)
    assert calls["n"] >= 2


def test_stalled_capability_injects_top_probes_without_rotation(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    prog = _live_60w_champion()[:48]
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = prog
    eng.champion_level = 30
    eng.champion_trace = {"registers": {}, "memory": {}}
    eng.task_instance = {}
    eng._champion_verify_score = 1.0
    eng.current_generation = 15
    eng.rng.seed(0)
    challengers = [list(prog) for _ in range(40)]
    eng._last_challenger_strategies = ["MUTATION"] * 40

    pool: list[list[int]] = []
    for marker in range(8):
        variant = list(prog)
        variant[1] = marker * 1000
        pool.append(variant)

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 48, 1.0)
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_probe_escalation_ratio",
        lambda *a, **k: 0.1,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.select_terminal_probes_for_batch",
        lambda *a, **k: pool,
    )

    eng._inject_terminal_verify_probes(challengers, 40)

    injected_markers = {
        int(c[1])
        for c in challengers
        if len(c) == len(prog) and int(c[1]) % 1000 == 0 and int(c[1]) < 8000
    }
    assert 7000 not in injected_markers
    assert injected_markers.issuperset({0, 1000, 2000})


def test_warm_steady_caps_terminal_probe_escalation(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    prog = _live_60w_champion()[:27]
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = prog
    eng.champion_level = 30
    eng.champion_trace = {"registers": {}, "memory": {}}
    eng.task_instance = {}
    eng._champion_verify_score = 1.0
    eng.rng.seed(0)
    pop_size = 100
    challengers = [list(prog) for _ in range(pop_size)]
    eng._last_challenger_strategies = ["MUTATION"] * pop_size

    pool: list[list[int]] = []
    for marker in range(30):
        variant = list(prog)
        variant[2] = marker * 1000
        pool.append(variant)

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 27, 1.0)
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_probe_escalation_ratio",
        lambda *a, **k: 0.72,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.select_terminal_probes_for_batch",
        lambda *a, **k: pool,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.should_skip_mg_stall_collect_for_warm_eq_refine",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(eng, "_terminal_probe_pool", lambda _pop: pool)
    monkeypatch.setattr(
        "evolution.terminal_stagnation._l11_fib_gate_imm",
        lambda _p: 13,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.semantic_compress",
        lambda _p: None,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_capability_growth_allowed",
        lambda _n: False,
    )

    eng._inject_terminal_verify_probes(challengers, pop_size)

    n_probe_slots = sum(
        1 for s in eng._last_challenger_strategies if s == "TERMINAL_VERIFY_PROBE"
    )
    assert n_probe_slots == 12
