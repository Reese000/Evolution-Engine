"""Certified champion seeding at terminal depth."""

import json
import os

os.environ.setdefault("EE_USE_GPU", "0")


def test_apply_best_certified_champion_seed_loads_c3511(tmp_path, monkeypatch):
    results = tmp_path / "multi_cycle_results.jsonl"
    good_prog = ["0x00000013"] * 73
    rows = [
        {
            "cycle": 3511,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 73,
            "final_fitness": 264.0,
            "champion_program": good_prog,
        },
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    from evolution_engine import EvolutionConfig, EvolutionEngine

    eng = EvolutionEngine(EvolutionConfig(curriculum_level=30, max_generations=1))
    eng.champion_level = 30
    monkeypatch.setattr(
        eng,
        "_sync_champion_after_reseed",
        lambda words, desc, label, **kw: setattr(eng, "_seeded_words", words),
    )

    import evolution.champion_rollback as cr

    monkeypatch.setattr(cr, "find_best_certified_champion", lambda path, level, **kw: rows[0])
    assert eng._apply_best_certified_champion_seed() is True
    assert len(eng._seeded_words) == 73
