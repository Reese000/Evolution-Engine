"""Population scan must consider saturated-verify fitness tie-breaks."""

import os

import pytest

os.environ.setdefault("EE_USE_GPU", "0")


def test_verify_led_candidate_eligible_at_saturated_verify():
    from evolution.terminal_stagnation import verify_led_candidate_eligible

    assert verify_led_candidate_eligible(1.0, 1.0, 0.005)
    assert not verify_led_candidate_eligible(1.0, 0.99, 0.005)
    assert verify_led_candidate_eligible(0.95, 0.97, 0.005)


def test_scan_population_finds_shorter_saturated_candidate(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    from evolution.fitness_helpers import selection_fitness
    from evolution_engine import EvolutionConfig, EvolutionEngine

    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    long_prog = [0x00100073] * 73
    short_prog = [0x00100073] * 71
    eng.champion = long_prog
    eng.champion_trace = {"status": "ok", "instructions_count": 10, "final_regs": {}}
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 264.04

    def _fake_eval(prog):
        return {"status": "ok", "instructions_count": len(prog), "final_regs": {}}

    eng._evaluate = _fake_eval  # type: ignore[method-assign]
    eng._verify_score_from_trace = lambda trace: 1.0  # type: ignore[method-assign]
    monkeypatch.setattr(
        "evolution.fitness_helpers.compute_fitness",
        lambda *a, **k: 237.88,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.partial_verify_multiplier",
        lambda *a, **k: 1.0,
    )

    eng._last_challengers = [long_prog, short_prog]
    eng._last_results = [_fake_eval(long_prog), _fake_eval(short_prog)]
    eng._last_challenger_fitnesses = [
        selection_fitness(eng, long_prog, eng._last_results[0]),
        selection_fitness(eng, short_prog, eng._last_results[1]),
    ]

    result = eng._scan_population_verify_candidates(1.0, None)
    assert result is not None
    assert len(result[2]) == 71
    assert result[1] > eng.champion_fitness


def test_find_best_challenger_applies_saturated_verify_best(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    from evolution.fitness_helpers import selection_fitness
    from evolution_engine import EvolutionConfig, EvolutionEngine

    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    long_prog = [0x00100073] * 73
    short_prog = [0x00100073] * 71
    eng.champion = long_prog
    trace = {"status": "ok", "instructions_count": 10, "final_regs": {}}
    eng.champion_trace = trace
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 264.04
    eng.task_instance = {"expected_output": {}}

    eng._evaluate = lambda prog: {**trace, "instructions_count": len(prog)}  # type: ignore[method-assign]
    eng._verify_score_from_trace = lambda tr: 1.0  # type: ignore[method-assign]
    monkeypatch.setattr(
        "evolution.fitness_helpers.compute_fitness",
        lambda *a, **k: 237.88,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.partial_verify_multiplier",
        lambda *a, **k: 1.0,
    )

    short_fit = selection_fitness(eng, short_prog, trace)
    eng._last_challengers = [long_prog, short_prog]
    eng._last_results = [trace, trace]
    eng._last_challenger_fitnesses = [
        selection_fitness(eng, long_prog, trace),
        short_fit,
    ]

    best_fit, best_prog, _, _, improved = eng._find_best_challenger(
        [long_prog],
        [trace],
        stage1_fitnesses=[selection_fitness(eng, long_prog, trace)],
    )
    assert improved is True
    assert best_prog is not None
    assert len(best_prog) == 71
    assert best_fit > eng.champion_fitness


def test_find_best_challenger_keeps_scan_floor_verify_best(monkeypatch, tmp_path):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    from evolution.fitness_helpers import promotion_raw_fitness, selection_fitness
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor
    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = [0x00000013] * 59 + [0x00100073]
    cand = list(champ)
    cand[10] = 0x002084B3
    trace = {"status": "ok", "instructions_count": 60, "final_regs": {}}
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = champ
    eng.champion_trace = trace
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 344.0
    eng._certified_fitness_floor = 344.0
    eng.task_instance = {"expected_output": {}}
    eng._verify_score_from_trace = lambda tr: 0.68  # type: ignore[method-assign]
    record_terminal_efficiency_scan_floor(100, 60, 1.0)

    def _raw(engine, program, tr, *, verify_score=None):
        return 140.0 if program[10] == 0x002084B3 else 120.0

    monkeypatch.setattr("evolution.fitness_helpers.promotion_raw_fitness", _raw)
    monkeypatch.setattr(
        "evolution.fitness_helpers.compute_fitness",
        lambda *a, **k: 120.0,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.partial_verify_multiplier",
        lambda *a, **k: 1.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.cached_best_compact_l30_fill",
        lambda _prog: tuple([0x00000013] * 65 + [0x00100073]),
    )

    eng._last_challengers = [champ, cand]
    eng._last_results = [trace, trace]
    eng._last_challenger_fitnesses = [
        selection_fitness(eng, champ, trace),
        selection_fitness(eng, cand, trace),
    ]
    _fit, best_prog, _, _, improved = eng._find_best_challenger(
        [champ],
        [trace],
        stage1_fitnesses=[selection_fitness(eng, champ, trace)],
    )
    assert improved is True
    assert best_prog is not None
    assert len(best_prog) == 60
    assert best_prog[10] == 0x002084B3


def test_selection_fitness_at_scan_floor_uses_promotion_raw(monkeypatch, tmp_path):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    from evolution.fitness_helpers import selection_fitness
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor
    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = [0x00000013] * 47 + [0x00100073]
    better = list(champ)
    better[5] = 0x002084B3
    trace = {"status": "ok", "instructions_count": 48, "final_regs": {}}
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = champ
    eng.champion_fitness = 648.0
    eng._certified_fitness_floor = 648.0
    eng._champion_verify_score = 1.0

    record_terminal_efficiency_scan_floor(100, 48, 1.0)
    monkeypatch.setattr(
        "evolution.fitness_helpers.promotion_raw_fitness",
        lambda _e, prog, _t, **kw: 324.5 if prog[5] == 0x002084B3 else 324.2,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.reporting_champion_fitness",
        lambda *a, **k: 648.0,
    )

    champ_fit = selection_fitness(eng, champ, trace)
    better_fit = selection_fitness(eng, better, trace)
    assert champ_fit == pytest.approx(324.2)
    assert better_fit == pytest.approx(324.5)
    assert better_fit > champ_fit


def test_find_best_challenger_at_scan_floor_ranks_promotion_raw(monkeypatch, tmp_path):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    from evolution.fitness_helpers import selection_fitness
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor
    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = [0x00000013] * 47 + [0x00100073]
    better = list(champ)
    better[5] = 0x002084B3
    worse_tr = list(champ)
    worse_tr[6] = 0x003084B3
    trace = {"status": "ok", "instructions_count": 48, "final_regs": {5: 1, 6: 99}}
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = champ
    eng.champion_trace = trace
    eng.champion_fitness = 648.0
    eng._champion_verify_score = 1.0
    eng.task_instance = {"expected_output": {}}
    eng._verify_score_from_trace = lambda tr: 1.0  # type: ignore[method-assign]
    record_terminal_efficiency_scan_floor(100, 48, 1.0)

    def _raw(_e, prog, _t, **kw):
        if prog[5] == 0x002084B3:
            return 324.5
        if prog[6] == 0x003084B3:
            return 324.1
        return 324.2

    monkeypatch.setattr("evolution.fitness_helpers.promotion_raw_fitness", _raw)
    monkeypatch.setattr(
        "evolution.fitness_helpers.compute_fitness",
        lambda *a, **k: 120.0,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.partial_verify_multiplier",
        lambda *a, **k: 1.0,
    )
    monkeypatch.setattr(
        "evolution.register_repair.score_target_register",
        lambda *a, **k: 1.0 if a[2].get(6) == 99 else 0.0,
    )

    _fit, best_prog, _, _, improved = eng._find_best_challenger(
        [worse_tr, better],
        [trace, trace],
        stage1_fitnesses=None,
    )
    assert improved is True
    assert best_prog is not None
    assert best_prog[5] == 0x002084B3
    assert selection_fitness(eng, best_prog, trace) == pytest.approx(324.5)


def test_find_best_challenger_verify_best_not_blocked_by_reporting_fitness(monkeypatch, tmp_path):
    """Population verify_best must apply when champion_fitness is reporting-inflated."""
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    from evolution.fitness_helpers import selection_fitness
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor
    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = [0x00000013] * 47 + [0x00100073]
    better = list(champ)
    better[8] = 0x002084B3
    trace = {"status": "ok", "instructions_count": 48, "final_regs": {}}
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = champ
    eng.champion_trace = trace
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 648.0
    eng.task_instance = {"expected_output": {}}
    record_terminal_efficiency_scan_floor(100, 48, 1.0)
    eng._verify_score_from_trace = lambda tr: 1.0  # type: ignore[method-assign]
    monkeypatch.setattr(
        "evolution.fitness_helpers.promotion_raw_fitness",
        lambda _e, prog, _t, **kw: 324.25 if prog[8] == 0x002084B3 else 324.2,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.reporting_champion_fitness",
        lambda *a, **k: 648.0,
    )
    eng._last_challengers = [champ, better]
    eng._last_results = [trace, trace]
    eng._last_challenger_fitnesses = [
        selection_fitness(eng, champ, trace),
        selection_fitness(eng, better, trace),
    ]
    best_fit, best_prog, _, _, improved = eng._find_best_challenger(
        [champ],
        [trace],
        stage1_fitnesses=[selection_fitness(eng, champ, trace)],
    )
    assert improved is True
    assert best_prog is not None
    assert best_prog[8] == 0x002084B3
    assert best_fit == pytest.approx(324.25)


def test_ensure_semantic_compress_elite_uses_selection_fitness_for_champion(monkeypatch, tmp_path):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor
    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = [0x00000013] * 47 + [0x00100073]
    trace = {"status": "ok", "instructions_count": 48, "final_regs": {}}
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = champ
    eng.champion_trace = trace
    eng.champion_fitness = 648.0
    eng._champion_verify_score = 1.0
    record_terminal_efficiency_scan_floor(100, 48, 1.0)
    monkeypatch.setattr(
        "evolution.fitness_helpers.promotion_raw_fitness",
        lambda *_a, **_k: 324.2,
    )
    elites, elite_fits, _indices = eng._ensure_semantic_compress_elite([], [], [])
    assert elites == [champ]
    assert elite_fits[0] == pytest.approx(324.2)


def test_verify_promotion_beats_champion_small_gain_at_scan_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        scan_floor_min_raw_gain,
        verify_promotion_beats_champion,
    )
    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = [0x00000013] * 47 + [0x00100073]
    better = list(champ)
    better[3] = 0x002084B3
    trace = {"status": "ok", "instructions_count": 48, "final_regs": {}}
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng._champion_verify_score = 1.0
    record_terminal_efficiency_scan_floor(100, 48, 1.0)
    assert scan_floor_min_raw_gain(48) == pytest.approx(0.01)

    monkeypatch.setattr(
        "evolution.fitness_helpers.promotion_raw_fitness",
        lambda _e, prog, _t, **kw: 324.25 if prog[3] == 0x002084B3 else 324.2,
    )
    assert verify_promotion_beats_champion(
        eng,
        champ,
        trace,
        better,
        trace,
        324.2,
        324.25,
        champion_verify=1.0,
        candidate_verify=1.0,
    )


def test_maybe_promote_terminal_efficiency_champion(monkeypatch, tmp_path):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    import json

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )

    from evolution_engine import EvolutionConfig, EvolutionEngine

    row = None
    for line in open("multi_cycle_results.jsonl"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("champion_words") == 73 and float(candidate.get("final_fitness", 0.0)) > 260.0:
            row = candidate
            break
    assert row is not None, "need a certified 73w row in multi_cycle_results.jsonl"
    prog = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = prog
    eng.champion_trace = eng._evaluate(prog)
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 264.0423516129032
    eng._certified_fitness_floor = 264.0423516129032
    assert len(prog) == 73
    promoted = eng._maybe_promote_terminal_efficiency_champion()
    assert promoted is True
    assert len(eng.champion) == 71
    assert eng.champion_fitness > 264.04
