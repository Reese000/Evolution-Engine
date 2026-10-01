"""Restore certified champion when curriculum verify regresses at L30."""

import json
import os

os.environ.setdefault("EE_USE_GPU", "0")


def test_restore_certified_on_curriculum_verify_regression(tmp_path, monkeypatch):
    results = tmp_path / "multi_cycle_results.jsonl"
    good_prog = ["0x00008193"] * 26 + ["0x00100073"]
    row = {
        "cycle": 7574,
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 27,
        "final_fitness": 325.82,
        "promotion_raw_fitness": 325.8257,
        "champion_program": good_prog,
    }
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")

    from evolution_engine import EvolutionConfig, EvolutionEngine

    eng = EvolutionEngine(EvolutionConfig(curriculum_level=30, max_generations=1))
    eng.champion_level = 30
    eng.champion = [0x13] * 26
    eng._champion_verify_score = 0.894
    synced: list = []

    def _sync(words, desc, label, **kw):
        synced.append((list(words), label, kw))

    monkeypatch.setattr(eng, "_sync_champion_after_reseed", _sync)
    monkeypatch.setattr(
        "evolution.champion_rollback.find_best_certified_champion",
        lambda path, level, **kw: row if str(results) in str(path) or path == str(results) else row,
    )

    assert eng._restore_certified_on_curriculum_verify_regression(0.894, 30) is True
    assert synced
    assert len(synced[0][0]) == 27


def test_reconcile_initialized_champion_fitness_uses_reporting_scale(monkeypatch):
    from evolution_engine import EvolutionConfig, EvolutionEngine

    eng = EvolutionEngine(EvolutionConfig(curriculum_level=30, max_generations=1))
    eng.champion_level = 30
    eng.champion = [0x8193] * 27
    eng.champion_trace = {"status": "ok"}
    eng.champion_fitness = 77.0
    eng._champion_verify_score = 1.0

    monkeypatch.setattr(
        "evolution.fitness_helpers.end_of_cycle_champion_fitness",
        lambda *a, **k: 325.82,
    )

    eng._reconcile_initialized_champion_fitness()
    assert eng.champion_fitness == 325.82
