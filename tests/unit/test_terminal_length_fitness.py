"""Length-efficiency fitness bonus at certified L30."""

import pytest

from evolution.fitness_helpers import _terminal_length_efficiency_bonus, reporting_champion_fitness, selection_fitness
from evolution_engine import EvolutionConfig, EvolutionEngine


def test_shorter_program_gets_length_bonus(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    monkeypatch.setenv("EE_TERMINAL_REF_WORDS", "73")
    monkeypatch.setenv("EE_TERMINAL_LENGTH_BONUS", "0.05")
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion_fitness = 264.04
    eng._champion_verify_score = 1.0
    eng.champion = [0] * 73
    short = [0] * 71
    base = 264.04
    bonus = _terminal_length_efficiency_bonus(eng, short, base)
    assert bonus == pytest.approx(264.14)
    same = _terminal_length_efficiency_bonus(eng, eng.champion, base)
    assert same == pytest.approx(264.04)


def test_align_scan_floor_persisted_metrics_overwrites_reporting(monkeypatch, tmp_path):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    from evolution.fitness_helpers import align_scan_floor_persisted_metrics
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    record_terminal_efficiency_scan_floor(100, 48, 1.0)
    metrics = {
        "champion_verified": True,
        "champion_words": 48,
        "final_fitness": 346.735,
        "promotion_raw_fitness": 324.236,
    }
    align_scan_floor_persisted_metrics(None, metrics)
    assert metrics["final_fitness"] == pytest.approx(324.236)
    assert metrics["verified_fitness"] == pytest.approx(324.236)


def test_end_of_cycle_champion_fitness_uses_prom_raw_at_scan_floor(monkeypatch, tmp_path):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    from evolution.fitness_helpers import end_of_cycle_champion_fitness
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor
    from evolution_engine import EvolutionConfig, EvolutionEngine

    prog = [0x00100073] * 48
    trace = {"status": "ok", "instructions_count": 48, "final_regs": {}}
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = prog
    eng.champion_fitness = 881.0
    eng._champion_verify_score = 1.0
    record_terminal_efficiency_scan_floor(100, 48, 1.0)
    eng._evaluate = lambda p: trace  # type: ignore[method-assign]
    monkeypatch.setattr(
        "evolution.fitness_helpers.promotion_raw_fitness",
        lambda *a, **k: 324.5,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.reporting_champion_fitness",
        lambda *a, **k: 881.0,
    )
    fit = end_of_cycle_champion_fitness(eng, prog, trace, 1.0)
    assert fit == pytest.approx(324.5)


def test_reporting_same_champion_is_idempotent(monkeypatch):
    """Repeated end-of-cycle reporting must not ratchet fitness on unchanged champion."""
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    monkeypatch.setenv("EE_TERMINAL_REF_WORDS", "73")
    monkeypatch.setenv("EE_TERMINAL_LENGTH_BONUS", "0.05")
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    prog = [0x00100073] * 48
    eng.champion = prog
    eng.champion_fitness = 806.0
    eng._certified_fitness_floor = 806.0
    eng._champion_verify_score = 1.0
    trace = {"status": "ok", "instructions_count": 48, "final_regs": {}}
    eng.task_instance = object()
    eng._evaluate = lambda p: trace  # type: ignore[method-assign]
    monkeypatch.setattr(
        "evolution.fitness_helpers.compute_fitness",
        lambda *a, **k: 648.0,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.partial_verify_multiplier",
        lambda *a, **k: 1.0,
    )
    first = reporting_champion_fitness(eng, prog, trace, 1.0)
    eng.champion_fitness = first
    second = reporting_champion_fitness(eng, prog, trace, 1.0)
    assert first == pytest.approx(second)


def test_reporting_applies_length_bonus_above_200(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    monkeypatch.setenv("EE_TERMINAL_REF_WORDS", "73")
    monkeypatch.setenv("EE_TERMINAL_LENGTH_BONUS", "0.05")
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion_fitness = 264.04
    eng._champion_verify_score = 1.0
    eng.champion = [0] * 73
    short = [0] * 71
    trace = {"status": "ok", "instructions_count": 10, "final_regs": {}}
    eng.task_instance = object()
    eng._evaluate = lambda prog: trace  # type: ignore[method-assign]
    monkeypatch.setattr(
        "evolution.fitness_helpers.compute_fitness",
        lambda *a, **k: 237.88,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.partial_verify_multiplier",
        lambda *a, **k: 1.0,
    )
    reported = reporting_champion_fitness(eng, short, trace, 1.0)
    assert reported == pytest.approx(264.14)
