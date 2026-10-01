"""Terminal efficiency promotion for longer scan-floor growth candidates."""

import json
import os

import pytest

from evolution_engine import EvolutionConfig, EvolutionEngine


@pytest.mark.skipif(
    not os.path.isfile("multi_cycle_results.jsonl"),
    reason="needs live results jsonl",
)
def test_terminal_efficiency_promotes_live_growth_material(tmp_path, monkeypatch):
    """55w growth material at prom 324.83+ should promote over 54w @ 324.633."""
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    monkeypatch.setenv("EE_TERMINAL_EFFICIENCY_PROMOTE_COOLDOWN", "0")
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    row = json.loads([l for l in open("multi_cycle_results.jsonl", encoding="utf-8") if l.strip()][-1])
    champ = [int(x, 16) for x in row["champion_program"]]
    assert len(champ) == 54
    record_terminal_efficiency_scan_floor(int(row["cycle"]), len(champ), 1.0)

    material = json.load(open(".agents/scan_floor_growth_material.json", encoding="utf-8"))
    growth = max(
        material[next(iter(material))],
        key=lambda e: float(e.get("prom") or 0.0),
    )["program"]
    assert len(growth) > len(champ)

    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    assert compact_l30_promotion_raw_fitness(growth) > compact_l30_promotion_raw_fitness(champ) + 0.01

    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = list(champ)
    eng._cycle_number = int(row["cycle"]) + 1
    eng._champion_verify_score = 1.0
    eng.champion_fitness = float(row["final_fitness"])
    eng._certified_fitness_floor = eng.champion_fitness
    eng.champion_trace = {"verify_score": 1.0, "status": "ok"}
    eng._cached_compact_champ_key = tuple(champ)
    eng._cached_compact_champ_heldout = 1.0
    eng._cached_compact_champ_side = 1.0

    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_variants",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_promote_skip_quiescent",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.ranked_scan_floor_compression_variants",
        lambda *args, **kwargs: [],
    )
    replaced = {"prog": None}

    def _replace(prog, trace, raw, fit):
        replaced["prog"] = list(prog)
        eng.champion = list(prog)
        eng.champion_fitness = float(fit)
        return True

    monkeypatch.setattr(eng, "_verify_and_replace_champion", _replace)

    assert eng._maybe_promote_terminal_efficiency_champion() is True
    assert replaced["prog"] is not None
    assert len(replaced["prog"]) > len(champ)
