"""TASK_AWARE at compact L30 must not bloat; overlay keeps length."""

import random

from mutation_proposer import EBREAK, DirectedMutator


def test_overlay_keeps_length():
    prog = [0x00100093, 0x00200113, 0x00300193, 0x00400213, EBREAK]
    seq = [0x00A28293, 0x00B30313]
    out = DirectedMutator.overlay_seq_keep_length(prog, seq, random.Random(0))
    assert len(out) == len(prog)
    assert out[-1] == EBREAK
    assert out != prog


def test_task_aware_l30_compact_does_not_grow():
    body = [0x00100093 + i for i in range(18)]
    prog = body + [EBREAK]
    for seed in range(20):
        out, desc = DirectedMutator._task_aware_mutation(list(prog), random.Random(seed), 30)
        assert out[-1] == EBREAK
        if desc == "task_aware_l30_easy_fill":
            assert 12 <= len(out) <= 72
        else:
            assert len(out) <= len(prog), (seed, desc, len(out), len(prog))


def test_task_aware_skips_growth_easy_fill_when_promotion_raw_stalled(tmp_path, monkeypatch):
    import json

    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    body = [0x00100093 + i for i in range(59)]
    prog = body + [EBREAK]
    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
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
    monkeypatch.setattr(
        "evolution.mutation_helpers._evolution_project_root",
        lambda: str(tmp_path),
    )
    for seed in range(40):
        out, desc = DirectedMutator._task_aware_mutation(list(prog), random.Random(seed), 30)
        assert desc != "task_aware_l30_easy_fill"
        assert not str(desc).startswith("task_aware_growth_")
        assert len(out) == len(prog)


def test_task_aware_skips_easy_fill_at_scan_floor(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor

    body = [0x00100093 + i for i in range(59)]
    prog = body + [EBREAK]
    assert len(prog) == 60
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
    for seed in range(30):
        out, desc = DirectedMutator._task_aware_mutation(list(prog), random.Random(seed), 30)
        assert desc != "task_aware_l30_easy_fill"
        assert not str(desc).startswith("task_aware_growth_")
        assert len(out) == len(prog)
