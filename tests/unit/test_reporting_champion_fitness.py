"""Tests for end-of-cycle reporting fitness."""

import json
import os

import pytest

os.environ.setdefault("EE_USE_GPU", "0")


@pytest.fixture
def l30_verified_champion():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "multi_cycle_results.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    row = next(
        (r for r in reversed(rows) if float(r.get("champion_verify_score", 0)) >= 1.0 - 1e-6),
        None,
    )
    if row is None:
        pytest.skip("no verified champion in results jsonl")
    return row


def test_reporting_fitness_for_verified_champion(l30_verified_champion):
    from evolution.fitness_helpers import reporting_champion_fitness
    from evolution_engine import EvolutionConfig, EvolutionEngine

    prog = [int(x, 16) if isinstance(x, str) else int(x) for x in l30_verified_champion["champion_program"]]
    level = int(l30_verified_champion["final_level"])
    eng = EvolutionEngine(EvolutionConfig(curriculum_level=level))
    eng.champion_level = level
    eng._refresh_task_instance()
    trace = eng._evaluate(prog)
    reported = reporting_champion_fitness(eng, prog, trace, 1.0)
    assert reported >= 200.0
