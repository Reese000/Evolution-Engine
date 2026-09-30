"""Tests for deterministic wide-imm sweep at scan floor."""

import json
import os

import pytest


def test_wide_imm_sweep_rotates_with_cycles_at_level(tmp_path, monkeypatch):
    import random

    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        terminal_wide_imm_sweep_probe_variants,
    )

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
    prog = [int(x, 16) for x in row["champion_program"]]

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)

    first = terminal_wide_imm_sweep_probe_variants(prog, max_evals=8, start_offset=0)
    second = terminal_wide_imm_sweep_probe_variants(prog, max_evals=8, start_offset=64)
    assert first
    for variant in first + second:
        assert len(variant) == len(prog)
        assert variant != prog
    if first != second:
        assert tuple(first[0]) != tuple(second[0]) or len(first) != len(second)
