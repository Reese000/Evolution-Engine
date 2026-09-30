"""Compact L30 champions may shrink when CPU held-out and side curriculum hold."""

import json
import os

import pytest

from evolution.terminal_stagnation import heldout_l30_verify_mean
from evolution_engine import EvolutionConfig, EvolutionEngine


@pytest.fixture
def isolated_progress_state(tmp_path, monkeypatch):
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    return agents


@pytest.mark.skipif(
    not os.path.isfile("multi_cycle_results.jsonl"),
    reason="needs live results jsonl",
)
def test_compact_terminal_efficiency_promotion(monkeypatch, isolated_progress_state):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if (
            candidate.get("final_level") != 30
            or int(candidate.get("champion_words", 0)) < 62
            or float(candidate.get("champion_verify_score", 0.0)) < 0.99
        ):
            continue
        prog = [int(x, 16) if isinstance(x, str) else int(x) for x in candidate["champion_program"]]
        if heldout_l30_verify_mean(prog, 32) < 0.99:
            continue
        row = candidate
    assert row is not None
    prog = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]

    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = prog
    eng._cycle_number = 9000
    eng.champion_trace = eng._evaluate(prog)
    eng._champion_verify_score = 1.0
    eng.champion_fitness = float(row["final_fitness"])
    eng._certified_fitness_floor = eng.champion_fitness

    before_h = heldout_l30_verify_mean(prog, 32)
    promoted = eng._maybe_promote_terminal_efficiency_champion()
    assert promoted is True
    assert len(eng.champion) < len(prog)
    after_h = heldout_l30_verify_mean(list(eng.champion), 32)
    assert after_h + 1e-6 >= before_h


def test_terminal_efficiency_promote_cooldown_skips_repeat(monkeypatch, isolated_progress_state):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    monkeypatch.setenv("EE_TERMINAL_EFFICIENCY_PROMOTE_COOLDOWN", "5")
    prog = [0x13] * 60 + [0x00100073]
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = prog
    eng.champion_trace = {"verify_score": 1.0}
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 300.0
    eng._certified_fitness_floor = 300.0
    eng._cycle_number = 10
    eng._terminal_efficiency_promote_cycle = 8

    assert eng._maybe_promote_terminal_efficiency_champion() is False

    eng._terminal_efficiency_promote_cycle = 4
    eng._cached_compact_champ_key = tuple(prog)
    eng._cached_compact_champ_heldout = 1.0
    eng._cached_compact_champ_side = 1.0
    # Cooldown elapsed; may still return False if no qualifying variant — must not
    # hit cooldown skip (no assertion on return value beyond not crashing).
    eng._maybe_promote_terminal_efficiency_champion()


def test_terminal_efficiency_promote_once_per_cycle(monkeypatch, isolated_progress_state):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    monkeypatch.setenv("EE_TERMINAL_EFFICIENCY_PROMOTE_COOLDOWN", "0")
    prog = [0x13] * 60 + [0x00100073]
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = prog
    eng.champion_trace = {"verify_score": 1.0}
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 300.0
    eng._certified_fitness_floor = 300.0
    eng._cycle_number = 20
    eng._cached_compact_champ_key = tuple(prog)
    eng._cached_compact_champ_heldout = 1.0
    eng._cached_compact_champ_side = 1.0

    eng._maybe_promote_terminal_efficiency_champion()
    assert eng._terminal_efficiency_promote_attempted_cycle == 20
    assert eng._maybe_promote_terminal_efficiency_champion() is False
