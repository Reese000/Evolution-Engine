"""Growth injection at scan floor (compress stays on mutation path)."""

import json
import random
from unittest.mock import patch

from evolution_engine import EvolutionConfig, EvolutionEngine
from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor


def _stall_jsonl(tmp_path) -> None:
    results = tmp_path / "multi_cycle_results.jsonl"
    with open(results, "w", encoding="utf-8") as f:
        for i in range(25):
            f.write(
                json.dumps(
                    {
                        "cycle": 6000 + i,
                        "final_level": 30,
                        "promotion_raw_fitness": 321.85,
                        "champion_words": 60,
                    }
                )
                + "\n"
            )


def test_apply_mutation_compresses_incidental_growth_at_stalled_scan_floor(
    tmp_path, monkeypatch
):
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    _stall_jsonl(tmp_path)
    monkeypatch.setattr(
        "evolution.mutation_helpers._evolution_project_root",
        lambda: str(tmp_path),
    )

    champ = [0x00100093 + i for i in range(59)] + [0x00100073]
    parent = list(champ)
    grown = parent[:-1] + [0x00a28293, 0x00100073]

    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = list(champ)
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    eng.rng = random.Random(0)

    with (
        patch(
            "evolution_engine.propose_mutation",
            return_value=(list(grown), "test_growth", None),
        ),
        patch(
            "evolution.terminal_stagnation.semantic_compress",
            side_effect=lambda p: p[:-2] + [p[-1]] if len(p) > 13 else p,
        ),
        patch(
            "evolution.terminal_stagnation.maybe_drop_interior_instruction",
            side_effect=lambda program, *_a, **_k: program,
        ),
    ):
        out = eng._apply_mutation("HYBRID", parent=parent)
    assert len(out) <= len(parent)


def test_apply_mutation_still_compresses_non_growth_at_stalled_scan_floor(tmp_path, monkeypatch):
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    _stall_jsonl(tmp_path)
    monkeypatch.setattr(
        "evolution.mutation_helpers._evolution_project_root",
        lambda: str(tmp_path),
    )

    champ = [0x00100093 + i for i in range(59)] + [0x00100073]
    parent = list(champ)
    same_len = list(parent)

    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = list(champ)
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    eng.rng = random.Random(0)

    with (
        patch(
            "evolution_engine.propose_mutation",
            return_value=(list(same_len), "test_same", None),
        ),
        patch(
            "evolution.terminal_stagnation.semantic_compress",
            side_effect=lambda p: p[:-2] + [p[-1]] if len(p) > 13 else p,
        ),
        patch(
            "evolution.terminal_stagnation.maybe_drop_interior_instruction",
            side_effect=lambda program, *_a, **_k: program,
        ),
    ):
        out = eng._apply_mutation("HYBRID", parent=parent)
    assert len(out) < len(parent)
