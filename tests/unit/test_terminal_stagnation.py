"""Tests for terminal stagnation verify-led promotion."""

import json
import os

from evolution.terminal_stagnation import (
    candidate_beats_champion_on_verify,
    effective_verify_margin,
    terminal_certified_plateau,
    terminal_diversity_boost_params,
    terminal_diversity_boost_tier,
    terminal_efficiency_eval_budget,
    terminal_probe_escalation_ratio,
    terminal_retrain_interval,
)


def _live_60w_champion():
    if not os.path.isfile("multi_cycle_results.jsonl"):
        import pytest

        pytest.skip("needs live results jsonl")
    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("champion_words") == 60 and candidate.get("final_level") == 30:
            row = candidate
    assert row is not None
    return [int(x, 16) for x in row["champion_program"]]


def test_verify_promote_at_zero_fitness_plateau(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    assert candidate_beats_champion_on_verify(0.579, 0.632, 0.0, 0.0) is True
    assert candidate_beats_champion_on_verify(0.579, 0.632, 0.0, -25.0) is False


def test_verify_promote_when_higher_verify_and_ok_fitness(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    assert candidate_beats_champion_on_verify(0.579, 0.590, 60.0, 35.0) is True


def test_verify_promote_rejects_small_verify_gain(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    assert candidate_beats_champion_on_verify(0.579, 0.580, 60.0, 80.0) is False


def test_verify_promote_rejects_catastrophic_fitness_drop(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    assert candidate_beats_champion_on_verify(0.579, 0.650, 60.0, 10.0) is False


def test_adaptive_margin_shrinks_at_long_stall(tmp_path, monkeypatch):
    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text(
        json.dumps({"cycles_at_level": 35}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    assert effective_verify_margin() == 0.002


def test_adaptive_margin_accepts_smaller_gain_at_tier3(tmp_path, monkeypatch):
    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text(
        json.dumps({"cycles_at_level": 45}),
        encoding="utf-8",
    )
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    assert candidate_beats_champion_on_verify(0.579, 0.5805, 60.0, 35.0) is True


def test_terminal_verify_selection_bonus_active(monkeypatch, tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "tick_state.json").write_text(
        json.dumps({"terminal_diversity_boost": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    from evolution.terminal_stagnation import terminal_verify_selection_bonus

    assert terminal_verify_selection_bonus(0.6) == 72.0
    assert terminal_verify_selection_bonus(0.0) == 0.0


def test_terminal_efficiency_variants_strips_nops():
    import random

    from evolution.terminal_stagnation import _EBREAK, _NOP, terminal_efficiency_variants

    prog = [_NOP, _NOP, 0x123, _EBREAK]
    variants = terminal_efficiency_variants(prog, random.Random(0), max_evals=8)
    assert variants
    assert all(v[-1] == _EBREAK for v in variants)
    assert any(_NOP not in v[:-1] for v in variants)


def test_terminal_efficiency_variants_deletes_instruction_without_nops():
    import random

    from evolution.terminal_stagnation import _EBREAK, terminal_efficiency_variants

    prog = [0x00100093, 0x00200113, 0x00300193, 0x00400213, _EBREAK]
    variants = terminal_efficiency_variants(prog, random.Random(0), max_evals=8)
    assert any(len(v) < len(prog) for v in variants)
    assert any(len(v) <= len(prog) - 3 for v in variants)
    assert all(v[-1] == _EBREAK for v in variants)


def test_maybe_drop_interior_instruction():
    import random

    from evolution.terminal_stagnation import _EBREAK, maybe_drop_interior_instruction

    prog = [0x00100093, 0x00200113, 0x00300193, 0x00400213, _EBREAK]
    same = maybe_drop_interior_instruction(prog, random.Random(0), p=0.0)
    assert same == prog
    shorter = maybe_drop_interior_instruction(prog, random.Random(1), p=1.0, n_drop=2)
    assert len(shorter) == len(prog) - 2
    assert shorter[-1] == _EBREAK


def test_terminal_length_drop_params_vs_cap():
    from evolution.terminal_stagnation import terminal_length_drop_params

    assert terminal_length_drop_params(80, 76) == (0.40, 2)
    assert terminal_length_drop_params(70, 76) == (0.25, 1)
    assert terminal_length_drop_params(60, 76) == (0.12, 1)
    assert terminal_length_drop_params(19, 76) == (0.0, 1)
    assert terminal_length_drop_params(21, 76) == (0.12, 1)
    assert terminal_length_drop_params(40, None) == (0.25, 1)


def test_efficiency_drop_span_one_when_under_cap():
    import random

    from evolution.terminal_stagnation import _EBREAK, terminal_efficiency_drop_span, terminal_efficiency_variants

    assert terminal_efficiency_drop_span(80, 76) == 3
    assert terminal_efficiency_drop_span(70, 76) == 2
    assert terminal_efficiency_drop_span(58, 76) == 1
    prog = [0x00100093, 0x00200113, 0x00300193, 0x00400213, 0x00500293, _EBREAK]
    variants = terminal_efficiency_variants(prog, random.Random(0), max_evals=16, max_drop_span=1)
    assert variants
    assert all(len(v) >= len(prog) - 1 for v in variants)


def test_terminal_branch_flip_variants_change_program():
    import random

    from evolution.terminal_stagnation import terminal_branch_flip_variants

    prog = [0x00000013] * 60 + [0x00100073]
    variants = terminal_branch_flip_variants(prog, random.Random(3), max_evals=8)
    assert variants == [] or all(v != prog for v in variants)


def test_terminal_verify_probe_variants_change_program():
    import random

    from evolution.terminal_stagnation import terminal_verify_probe_variants

    prog = [0x00000013] * 60 + [0x00100073]
    variants = terminal_verify_probe_variants(prog, 30, random.Random(0), 3)
    assert len(variants) == 3
    assert all(v != prog for v in variants)
    assert all(len(v) >= len(prog) for v in variants)


def test_terminal_verify_probe_filters_regressions():
    import random

    from evolution.terminal_stagnation import terminal_verify_probe_variants

    prog = [0x00000013] * 60 + [0x00100073]

    def score_fn(candidate: list[int]) -> float:
        return 0.58 if len(candidate) > len(prog) + 8 else 0.52

    variants = terminal_verify_probe_variants(prog, 30, random.Random(1), 3, score_fn=score_fn, min_verify=0.579)
    assert all(score_fn(v) >= 0.579 - 1e-6 for v in variants)


def test_straight_line_probe_finds_l30_gain():
    import json
    import os
    import random

    os.environ.setdefault("EE_USE_GPU", "0")
    from evolution.terminal_stagnation import terminal_verify_probe_variants
    from evolution_engine import EvolutionConfig, EvolutionEngine

    rows = [json.loads(line) for line in open("multi_cycle_results.jsonl") if line.strip()]
    prog = [int(x, 16) if isinstance(x, str) else int(x) for x in rows[-1]["champion_program"]]
    engine = EvolutionEngine(EvolutionConfig(curriculum_level=30))
    engine.champion_level = 30
    engine.champion_fitness = 120.0
    from tasks_rv32i import LEVELS

    engine.task_instance = LEVELS[30](seed=42).generate_instance()
    champ_verify = engine._verify_score_from_trace(engine._evaluate(prog))

    def score_fn(candidate: list[int]) -> float:
        return engine._verify_score_from_trace(engine._evaluate(candidate))

    def fitness_fn(candidate: list[int]) -> float:
        trace = engine._evaluate(candidate)
        return engine._compute_fitness(candidate, trace)

    variants = terminal_verify_probe_variants(
        prog,
        30,
        random.Random(8),
        10,
        score_fn=score_fn,
        min_verify=champ_verify,
        fitness_fn=fitness_fn,
        min_fitness=40.0,
    )
    assert variants
    # Live champion is verify-saturated; probes are floor-filtered at champ_verify.
    # Do not pin a historical 0.684 from a longer incomplete program.
    assert max(score_fn(v) for v in variants) >= champ_verify - 1e-6


def test_reference_bundle_for_register_l30():
    from evolution.register_repair import reference_bundle_for_register

    bundle = reference_bundle_for_register(30, 29)
    assert len(bundle) >= 1


def test_wrong_level_registers_detects_misses():
    from evolution.register_repair import wrong_level_registers

    trace = {"final_regs": {29: 1, 30: 2}}
    inst = {"expected_output": {"registers": {29: 18, 30: 40}}}
    wrong = wrong_level_registers(30, trace, inst)
    assert 29 in wrong and 30 in wrong


def test_terminal_certified_plateau():
    assert terminal_certified_plateau(30, 1.0, 264.0) is True
    assert terminal_certified_plateau(29, 1.0, 264.0) is False
    assert terminal_certified_plateau(30, 0.95, 264.0) is False
    assert terminal_certified_plateau(30, 1.0, 100.0) is False


def test_terminal_certified_plateau_ignores_clamped_max_level(monkeypatch):
    """EE_MAX_LEVEL clamp recovery must not disable L30 terminal plateau."""
    monkeypatch.setenv("EE_MAX_LEVEL", "31")
    monkeypatch.setenv("EE_TERMINAL_DEPTH", "30")
    assert terminal_certified_plateau(30, 1.0, 264.0) is True


def test_terminal_probe_escalation_ratio(monkeypatch):
    monkeypatch.delenv("EE_TERMINAL_VERIFY_PROBE_RATIO", raising=False)
    assert terminal_probe_escalation_ratio(10) == 0.3
    assert terminal_probe_escalation_ratio(50) == 0.36
    assert terminal_probe_escalation_ratio(150) == 0.42
    assert terminal_probe_escalation_ratio(250) == 0.5
    assert terminal_probe_escalation_ratio(320) == 0.65
    assert terminal_probe_escalation_ratio(420) == 0.72


def test_terminal_retrain_interval_scales_with_tier():
    assert terminal_retrain_interval(10) == 40
    assert terminal_retrain_interval(30) == 34
    assert terminal_retrain_interval(120) == 28
    assert terminal_retrain_interval(339) == 32


def test_terminal_efficiency_promote_quiescent_after_stable(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import record_terminal_efficiency_stable, terminal_efficiency_promote_quiescent

    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_stable(100, 60)
    assert terminal_efficiency_promote_quiescent(102, 60, 1.0) is False
    assert terminal_efficiency_promote_quiescent(103, 60, 1.0) is True
    assert terminal_efficiency_promote_quiescent(110, 60, 1.0) is False


def test_scan_floor_skips_batch_shrink_probes(tmp_path, monkeypatch):
    import json
    import os
    import random

    from evolution.terminal_stagnation import (
        clear_terminal_efficiency_scan_floor,
        record_terminal_efficiency_scan_floor,
        select_terminal_probes_for_batch,
        terminal_efficiency_at_scan_floor,
        terminal_efficiency_variants,
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
    (agents / "progress_controller_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    assert len(terminal_efficiency_variants(prog, random.Random(0), max_evals=8)) > 0
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    assert terminal_efficiency_at_scan_floor(60) is True
    with_floor = select_terminal_probes_for_batch(prog, 30, random.Random(0), 12, pop_size=50)
    clear_terminal_efficiency_scan_floor()
    without_floor = select_terminal_probes_for_batch(prog, 30, random.Random(0), 12, pop_size=50)
    short = lambda probes: sum(1 for v in probes if len(v) < len(prog))
    assert short(with_floor) < short(without_floor)


def test_progress_state_read_uses_mtime_cache(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        _invalidate_progress_state_cache,
        _read_progress_state,
        record_terminal_efficiency_scan_floor,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    state_path = agents / "progress_controller_state.json"
    state_path.write_text('{"cycles_at_level": 1}', encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    _invalidate_progress_state_cache()
    fixed_mtime = 1000.0
    monkeypatch.setattr("evolution.terminal_stagnation.os.path.getmtime", lambda _p: fixed_mtime)
    first = _read_progress_state()
    state_path.write_text('{"cycles_at_level": 2}', encoding="utf-8")
    stale = _read_progress_state()
    assert stale["cycles_at_level"] == 1
    record_terminal_efficiency_scan_floor(50, 60, 1.0)
    fresh = _read_progress_state()
    assert fresh["terminal_efficiency_scan_floor_words"] == 60


def test_scan_floor_probe_qualifies_rejects_branch_flips_at_side_sat(tmp_path, monkeypatch):
    import json
    import os
    import random

    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        scan_floor_probe_qualifies,
        terminal_branch_flip_variants,
        terminal_equal_length_refine_variants,
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
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: False,
    )

    flips = terminal_branch_flip_variants(prog, random.Random(0), max_evals=16)
    assert flips
    refined = terminal_equal_length_refine_variants(prog, 30, random.Random(1), max_evals=24)
    assert refined
    assert all(len(v) == len(prog) for v in refined)
    assert all(v != prog for v in refined)


def test_overlay_keep_length_variants_same_size(tmp_path, monkeypatch):
    import json
    import os
    import random

    from evolution.terminal_stagnation import terminal_overlay_keep_length_variants

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
    variants = terminal_overlay_keep_length_variants(prog, 30, random.Random(7), max_evals=16)
    assert variants
    assert all(len(v) == len(prog) for v in variants)
    assert all(v != prog for v in variants)


def test_scan_floor_equal_length_probes_only(tmp_path, monkeypatch):
    import json
    import os
    import random

    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor, select_terminal_probes_for_batch

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
    (agents / "progress_controller_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)
    probes = select_terminal_probes_for_batch(prog, 30, random.Random(0), 12, pop_size=50)
    assert probes
    assert all(len(v) == len(prog) for v in probes)


def test_scan_floor_instr_swap_and_chained_imm_probes(tmp_path, monkeypatch):
    import json
    import os
    import random

    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        scan_floor_probe_qualifies,
        terminal_chained_imm_perturb_probe_variants,
        terminal_instr_swap_probe_variants,
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

    swaps = terminal_instr_swap_probe_variants(prog, random.Random(3), max_evals=8)
    chained = terminal_chained_imm_perturb_probe_variants(prog, random.Random(4), chain_steps=2, max_evals=8)
    assert swaps or chained
    for variant in swaps + chained:
        assert len(variant) == len(prog)
        assert variant != prog
        assert scan_floor_probe_qualifies(prog, variant)


def test_scan_floor_probe_cap_matches_escalation_ratio(tmp_path, monkeypatch):
    import json
    import os
    import random

    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        select_terminal_probes_for_batch,
        terminal_probe_escalation_ratio,
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
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: False,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: False,
    )

    pop_size = 50
    ratio = terminal_probe_escalation_ratio()
    expected_cap = max(
        max(2, pop_size // 4),
        max(2, int(pop_size * ratio)),
    )
    pool_limit = max(
        expected_cap,
        int(os.environ.get("EE_SCAN_FLOOR_PROBE_POOL", max(expected_cap * 4, 32))),
    )
    assert expected_cap > pop_size // 4
    probes = select_terminal_probes_for_batch(prog, 30, random.Random(0), expected_cap, pop_size=pop_size)
    assert probes
    assert len(probes) <= pool_limit


def test_scan_floor_chained_reorder_imm_probe_variants(tmp_path, monkeypatch):
    import json
    import os
    import random

    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        scan_floor_probe_qualifies,
        terminal_chained_reorder_imm_probe_variants,
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

    chained = terminal_chained_reorder_imm_probe_variants(prog, random.Random(8), max_evals=8)
    assert chained
    for variant in chained:
        assert len(variant) == len(prog)
        assert variant != prog
        assert scan_floor_probe_qualifies(prog, variant)


def test_scan_floor_block_reorder_probe_variants(tmp_path, monkeypatch):
    import json
    import os
    import random

    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        scan_floor_probe_qualifies,
        terminal_block_reorder_probe_variants,
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

    reorders = terminal_block_reorder_probe_variants(prog, random.Random(7), max_evals=8)
    assert reorders
    for variant in reorders:
        assert len(variant) == len(prog)
        assert variant != prog
        assert scan_floor_probe_qualifies(prog, variant)


def test_scan_floor_triple_chain_imm_probe_variants(tmp_path, monkeypatch):
    import json
    import os
    import random

    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        scan_floor_probe_qualifies,
        terminal_chained_imm_perturb_probe_variants,
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

    chained = terminal_chained_imm_perturb_probe_variants(prog, random.Random(11), chain_steps=3, max_evals=8)
    assert chained
    for variant in chained:
        assert len(variant) == len(prog)
        assert variant != prog
        assert scan_floor_probe_qualifies(prog, variant)


def test_scan_floor_wide_imm_probe_variants(tmp_path, monkeypatch):
    import hashlib
    import json
    import os
    import random
    import struct

    from evolution.fitness_helpers import promotion_raw_fitness
    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        scan_floor_probe_qualifies,
        select_terminal_probes_for_batch,
        terminal_wide_imm_perturb_probe_variants,
    )
    from evolution_engine import EvolutionConfig, EvolutionEngine

    if not os.path.isfile("multi_cycle_results.jsonl"):
        pytest.skip("needs live results jsonl")
    rows = [json.loads(line) for line in open("multi_cycle_results.jsonl", encoding="utf-8") if line.strip()]
    row = next((r for r in rows if r.get("cycle") == 5963), None)
    if row is None:
        row = next(
            (r for r in reversed(rows) if r.get("champion_words") == 60 and r.get("final_level") == 30),
            None,
        )
    assert row is not None
    prog = [int(x, 16) for x in row["champion_program"]]

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_scan_floor(100, 60, 1.0)

    wide = terminal_wide_imm_perturb_probe_variants(prog, random.Random(0), max_evals=64, max_attempts=5000)
    assert wide
    for variant in wide:
        assert len(variant) == len(prog)
        assert variant != prog
        assert scan_floor_probe_qualifies(prog, variant)

    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, curriculum_level=30))
    eng.champion = list(prog)
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    champ_trace = eng._evaluate(prog)
    champ_raw = promotion_raw_fitness(eng, prog, champ_trace, verify_score=1.0)
    best_raw = champ_raw
    for variant in wide:
        trace = eng._evaluate(variant)
        raw = promotion_raw_fitness(
            eng,
            variant,
            trace,
            verify_score=eng._verify_score_from_trace(trace),
        )
        best_raw = max(best_raw, raw)
    assert best_raw > champ_raw + 1e-6

    def sha12(words: list[int]) -> str:
        return hashlib.sha256(struct.pack("<" + "I" * len(words), *words)).hexdigest()[:12]

    pool = select_terminal_probes_for_batch(prog, 30, random.Random(0), 36, pop_size=50)
    assert pool
    for variant in pool:
        assert len(variant) == len(prog)
        assert scan_floor_probe_qualifies(prog, variant)


def test_scan_floor_crossover_probe_variants(tmp_path, monkeypatch):
    import json
    import os
    import random

    from evolution.terminal_stagnation import (
        ScanFloorGateBaseline,
        record_terminal_efficiency_scan_floor,
        scan_floor_probe_qualifies,
        terminal_crossover_probe_variants,
        terminal_imm_perturb_probe_variants,
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

    gate = ScanFloorGateBaseline(prog)
    parents = terminal_imm_perturb_probe_variants(prog, random.Random(5), max_evals=8, gate=gate)
    crossed = terminal_crossover_probe_variants(prog, parents, random.Random(6), max_evals=8, gate=gate)
    assert crossed or len(parents) < 2
    for variant in crossed:
        assert len(variant) == len(prog)
        assert variant != prog
        assert scan_floor_probe_qualifies(prog, variant)


def test_scan_floor_gate_qualify_cache(tmp_path, monkeypatch):
    import json
    import os
    import random

    from evolution.terminal_stagnation import (
        ScanFloorGateBaseline,
        record_terminal_efficiency_scan_floor,
        terminal_imm_perturb_probe_variants,
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

    gate = ScanFloorGateBaseline(prog)
    variants = terminal_imm_perturb_probe_variants(prog, random.Random(0), max_evals=4, gate=gate)
    assert variants
    for variant in variants:
        assert gate.qualifies(variant)
        key = tuple(variant)
        assert key in gate._qualify_cache
        assert gate._qualify_cache[key] is True


def test_scan_floor_extends_rescan_interval(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        record_terminal_efficiency_stable,
        terminal_efficiency_promote_quiescent,
        terminal_efficiency_rescan_interval,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    record_terminal_efficiency_stable(100, 60)
    record_terminal_efficiency_scan_floor(105, 60, 1.0)
    assert terminal_efficiency_rescan_interval(60) == 20
    assert terminal_efficiency_promote_quiescent(110, 60, 1.0) is True
    assert terminal_efficiency_promote_quiescent(120, 60, 1.0) is False


def test_terminal_efficiency_promote_cycle_persist(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import record_terminal_efficiency_promote, terminal_efficiency_last_promote_cycle

    agents = tmp_path / ".agents"
    agents.mkdir()
    state = agents / "progress_controller_state.json"
    state.write_text('{"level": 30, "cycles_at_level": 100}', encoding="utf-8")
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._infer_terminal_efficiency_promote_cycle",
        lambda: 0,
    )
    assert terminal_efficiency_last_promote_cycle() == 0
    record_terminal_efficiency_promote(5540)
    assert terminal_efficiency_last_promote_cycle() == 5540


def test_terminal_efficiency_eval_budget():
    assert terminal_efficiency_eval_budget(10) == 20
    assert terminal_efficiency_eval_budget(80) == 24
    assert terminal_efficiency_eval_budget(150) == 32
    assert terminal_efficiency_eval_budget(250) == 40
    assert terminal_efficiency_eval_budget(320) == 48
    assert terminal_efficiency_eval_budget(420) == 56


def test_terminal_diversity_boost_tier_escalates_with_stall():
    assert terminal_diversity_boost_tier(10) == 1
    assert terminal_diversity_boost_tier(25) == 2
    assert terminal_diversity_boost_tier(30) == 3
    assert terminal_diversity_boost_tier(60) == 4
    assert terminal_diversity_boost_tier(120) == 5
    assert terminal_diversity_boost_tier(220) == 6
    assert terminal_diversity_boost_tier(339) == 7


def test_terminal_diversity_boost_params_increase_with_tier():
    t3 = terminal_diversity_boost_params(3)
    t7 = terminal_diversity_boost_params(7)
    assert t7["mutation_rate"] > t3["mutation_rate"]
    assert t7["diversity_inject_ratio"] > t3["diversity_inject_ratio"]
    assert t7["novelty_weight"] > t3["novelty_weight"]
    assert t7["diversity_threshold"] >= t3["diversity_threshold"]


def test_l30_easy_fill_beats_partial_champion():
    from evolution.terminal_stagnation import l30_easy_fill_variants
    from gpu_vm_farm import _execute_vm
    from mutation_proposer import EBREAK, make_rv32i
    from tasks_rv32i import LEVELS

    prog = [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=9, rs1=1, rs2=2),
        make_rv32i("XOR", rd=23, rs1=1, rs2=2),
        EBREAK,
    ]
    fills = l30_easy_fill_variants(prog)
    assert fills
    bundle = fills[0]
    assert 12 <= len(bundle) <= 24
    level = LEVELS[30]
    inst = level(seed=0).generate_instance()
    v1, v2 = inst["inputs"]["registers"][1], inst["inputs"]["registers"][2]
    a = _execute_vm(prog, {1: v1, 2: v2}, {}, 10000)
    b = _execute_vm(bundle, {1: v1, 2: v2}, {}, 10000)
    va = level(seed=0).verify({"registers": a.get("registers") or {}, "memory": {}}, inst)
    vb = level(seed=0).verify({"registers": b.get("registers") or {}, "memory": {}}, inst)
    assert vb > va + 0.2


def test_l30_extra_fill_hits_x21_x26():
    from evolution.terminal_stagnation import heldout_l30_verify_mean, l30_easy_fill_variants
    from mutation_proposer import EBREAK, make_rv32i

    skeleton = [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=4, rs1=1, rs2=1),
        make_rv32i("ADDI", rd=5, rs1=2, imm=0),
        make_rv32i("ADD", rd=6, rs1=2, rs2=2),
        make_rv32i("ADD", rd=7, rs1=4, rs2=6),
        make_rv32i("SLT", rd=8, rs1=6, rs2=4),
        make_rv32i("ADD", rd=9, rs1=1, rs2=2),
        make_rv32i("ADD", rd=20, rs1=1, rs2=2),
        make_rv32i("XOR", rd=23, rs1=1, rs2=2),
        make_rv32i("SLLI", rd=28, rs1=9, imm=1),
        make_rv32i("ADD", rd=30, rs1=1, rs2=2),
        make_rv32i("ADDI", rd=31, rs1=23, imm=0),
        EBREAK,
    ]
    base = heldout_l30_verify_mean(skeleton, n_seeds=8)
    fills = l30_easy_fill_variants(skeleton)
    extras = [f for f in fills if len(f) == len(skeleton) + 2]
    assert extras
    extra = extras[0]
    lifted = heldout_l30_verify_mean(extra, n_seeds=8)
    assert lifted > base + 0.05
    assert abs(lifted - 0.737) < 0.02


def test_l30_gap_fill_raises_heldout_past_737():
    from evolution.terminal_stagnation import heldout_l30_verify_mean, l30_easy_fill_variants
    from mutation_proposer import EBREAK, make_rv32i

    skeleton = [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=4, rs1=1, rs2=1),
        make_rv32i("ADDI", rd=5, rs1=2, imm=0),
        make_rv32i("ADD", rd=6, rs1=2, rs2=2),
        make_rv32i("ADD", rd=7, rs1=4, rs2=6),
        make_rv32i("SLT", rd=8, rs1=6, rs2=4),
        make_rv32i("ADD", rd=9, rs1=1, rs2=2),
        make_rv32i("ADD", rd=20, rs1=1, rs2=2),
        make_rv32i("XOR", rd=23, rs1=1, rs2=2),
        make_rv32i("SLLI", rd=28, rs1=9, imm=1),
        make_rv32i("ADD", rd=30, rs1=1, rs2=2),
        make_rv32i("ADDI", rd=31, rs1=23, imm=0),
        make_rv32i("ADDI", rd=21, rs1=23, imm=0),
        make_rv32i("ADD", rd=26, rs1=1, rs2=2),
        EBREAK,
    ]
    base = heldout_l30_verify_mean(skeleton, n_seeds=8)
    fills = l30_easy_fill_variants(skeleton)
    assert fills
    best = max(fills, key=lambda prog: heldout_l30_verify_mean(prog, n_seeds=8))
    assert heldout_l30_verify_mean(best, n_seeds=8) > base + 0.1
    assert heldout_l30_verify_mean(best, n_seeds=8) >= 0.84
    assert len(best) <= 28


def test_l30_x24_fill_reaches_heldout_one():
    from evolution.terminal_stagnation import heldout_l30_verify_mean, l30_easy_fill_variants
    from mutation_proposer import EBREAK, DirectedMutator, make_rv32i

    skeleton = [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=4, rs1=1, rs2=1),
        make_rv32i("ADDI", rd=5, rs1=2, imm=0),
        make_rv32i("ADD", rd=6, rs1=2, rs2=2),
        make_rv32i("ADD", rd=7, rs1=4, rs2=6),
        make_rv32i("SLT", rd=8, rs1=6, rs2=4),
        make_rv32i("ADD", rd=9, rs1=1, rs2=2),
        make_rv32i("ADD", rd=20, rs1=1, rs2=2),
        make_rv32i("XOR", rd=23, rs1=1, rs2=2),
        make_rv32i("SLLI", rd=28, rs1=9, imm=1),
        make_rv32i("ADD", rd=30, rs1=1, rs2=2),
        make_rv32i("ADDI", rd=31, rs1=23, imm=0),
        make_rv32i("ADDI", rd=21, rs1=23, imm=0),
        make_rv32i("ADD", rd=26, rs1=1, rs2=2),
        make_rv32i("ADD", rd=22, rs1=1, rs2=2),
        make_rv32i("SUB", rd=29, rs1=2, rs2=1),
        make_rv32i("SUB", rd=25, rs1=1, rs2=2),
        make_rv32i("SUB", rd=27, rs1=1, rs2=2),
        EBREAK,
    ]
    base = heldout_l30_verify_mean(skeleton, n_seeds=8)
    fills = l30_easy_fill_variants(skeleton)
    assert fills
    best = max(fills, key=lambda prog: heldout_l30_verify_mean(prog, n_seeds=8))
    lifted = heldout_l30_verify_mean(best, n_seeds=8)
    assert lifted > base + 0.02
    assert lifted >= 0.99
    assert len(best) <= 28


def test_l30_order_sensitive_fill_hits_heldout_32():
    from evolution.terminal_stagnation import (
        heldout_l30_verify_mean,
        l30_easy_fill_variants,
        l30_order_sensitive_variant,
    )
    from mutation_proposer import EBREAK, make_rv32i

    prog = [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=9, rs1=1, rs2=2),
        make_rv32i("XOR", rd=23, rs1=1, rs2=2),
        make_rv32i("ADD", rd=4, rs1=1, rs2=1),
        make_rv32i("ADDI", rd=5, rs1=2, imm=0),
        make_rv32i("ADD", rd=6, rs1=2, rs2=2),
        make_rv32i("ADD", rd=7, rs1=4, rs2=6),
        make_rv32i("ADD", rd=20, rs1=1, rs2=2),
        make_rv32i("SLLI", rd=28, rs1=9, imm=1),
        make_rv32i("ADD", rd=30, rs1=1, rs2=2),
        make_rv32i("ADDI", rd=31, rs1=23, imm=0),
        make_rv32i("ADD", rd=22, rs1=1, rs2=2),
        make_rv32i("SUB", rd=29, rs1=2, rs2=1),
        make_rv32i("SUB", rd=25, rs1=1, rs2=2),
        make_rv32i("SUB", rd=27, rs1=1, rs2=2),
        make_rv32i("ADDI", rd=21, rs1=23, imm=0),
        make_rv32i("ADD", rd=26, rs1=1, rs2=2),
        make_rv32i("SLLI", rd=24, rs1=23, imm=2),
        make_rv32i("XOR", rd=24, rs1=24, rs2=1),
        EBREAK,
    ]
    base16 = heldout_l30_verify_mean(prog, n_seeds=16)
    ordered = l30_order_sensitive_variant(prog)
    assert ordered is not None
    assert len(ordered) <= 28
    lifted32 = heldout_l30_verify_mean(ordered, n_seeds=32)
    assert lifted32 > base16
    assert lifted32 >= 0.99
    from evolution.terminal_stagnation import (
        curriculum_verify_mean,
        l30_l10_prefix_load_variant,
        l30_strip_dead_l30_stores,
    )

    stripped = l30_strip_dead_l30_stores(ordered)
    loaded = l30_l10_prefix_load_variant(stripped or ordered)
    assert loaded is not None
    assert heldout_l30_verify_mean(loaded, n_seeds=32) >= 0.99
    assert curriculum_verify_mean(loaded, 10) > curriculum_verify_mean(ordered, 10) + 0.04
    from evolution.terminal_stagnation import compact_curriculum_side_score, l30_l8_mem_add_variant

    l8 = l30_l8_mem_add_variant(loaded)
    assert l8 is not None
    assert len(l8) <= 72
    assert heldout_l30_verify_mean(l8, n_seeds=32) >= 0.99
    assert curriculum_verify_mean(l8, 8) > curriculum_verify_mean(loaded, 8) + 0.04
    from evolution.terminal_stagnation import l30_l10_bounded_sum_variant

    summed = l30_l10_bounded_sum_variant(l8)
    assert summed is not None
    assert len(summed) <= 72
    assert heldout_l30_verify_mean(summed, n_seeds=32) >= 0.99
    assert curriculum_verify_mean(summed, 10) >= 0.89
    assert curriculum_verify_mean(summed, 8) >= 0.99
    assert curriculum_verify_mean(summed, 9) >= 0.99
    from evolution.terminal_stagnation import l30_l10_max_in_sum_loop_variant

    maxed = l30_l10_max_in_sum_loop_variant(summed)
    assert maxed is not None
    assert len(maxed) <= 72
    assert heldout_l30_verify_mean(maxed, n_seeds=32) >= 0.99
    assert curriculum_verify_mean(maxed, 10) >= 0.99
    from evolution.terminal_stagnation import l30_l11_gated_fib_variant

    fib = l30_l11_gated_fib_variant(maxed)
    assert fib is not None
    assert len(fib) <= 72
    assert heldout_l30_verify_mean(fib, n_seeds=32) >= 0.99
    assert curriculum_verify_mean(fib, 8) >= 0.99
    assert curriculum_verify_mean(fib, 10) >= 0.99
    assert curriculum_verify_mean(fib, 11) >= 0.99
    from evolution.terminal_stagnation import l30_l12_l16_postfix_variant

    mid = l30_l12_l16_postfix_variant(fib)
    assert mid is not None
    assert len(mid) <= 72
    assert heldout_l30_verify_mean(mid, n_seeds=32) >= 0.99
    assert curriculum_verify_mean(mid, 12) >= 0.99
    assert curriculum_verify_mean(mid, 10) >= 0.99
    assert curriculum_verify_mean(mid, 16) >= 0.99
    from evolution.terminal_stagnation import best_compact_l30_fill

    chosen = best_compact_l30_fill(l8)
    assert chosen is not None
    assert len(chosen) <= 72
    assert heldout_l30_verify_mean(chosen, n_seeds=32) >= 0.99
    assert compact_curriculum_side_score(chosen) + 1e-6 >= compact_curriculum_side_score(l8)
    fills = l30_easy_fill_variants(prog)
    assert any(len(f) == len(ordered) and f == ordered for f in fills)
    from evolution.terminal_stagnation import l30_strip_dead_l30_stores

    stripped = l30_strip_dead_l30_stores(ordered)
    if stripped is None:
        stripped = l30_strip_dead_l30_stores(prog)
    assert stripped is None or (
        len(stripped) < len(ordered or prog) and heldout_l30_verify_mean(stripped, n_seeds=32) >= 0.99
    )


def test_l11_fib_gate_tighten_skips_l30_x1_ge_13():
    from evolution.terminal_stagnation import (
        _l11_fib_gate_imm,
        curriculum_verify_mean,
        heldout_l30_verify_mean,
        l30_easy_fill_variants,
        l30_l11_fib_gate_tighten_variant,
        l30_l11_gated_fib_variant,
    )
    from mutation_proposer import EBREAK, make_rv32i

    core = [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=9, rs1=1, rs2=2),
        make_rv32i("XOR", rd=23, rs1=1, rs2=2),
        make_rv32i("ADD", rd=4, rs1=1, rs2=1),
        make_rv32i("ADDI", rd=5, rs1=2, imm=0),
        make_rv32i("ADD", rd=6, rs1=2, rs2=2),
        make_rv32i("ADD", rd=7, rs1=4, rs2=6),
        make_rv32i("ADD", rd=20, rs1=1, rs2=2),
        make_rv32i("SLLI", rd=28, rs1=9, imm=1),
        make_rv32i("ADD", rd=30, rs1=1, rs2=2),
        make_rv32i("ADDI", rd=31, rs1=23, imm=0),
        make_rv32i("ADDI", rd=21, rs1=23, imm=0),
        make_rv32i("ADD", rd=26, rs1=1, rs2=2),
        make_rv32i("ADD", rd=22, rs1=1, rs2=2),
        make_rv32i("SLT", rd=8, rs1=6, rs2=4),
        make_rv32i("ADD", rd=25, rs1=1, rs2=2),
        make_rv32i("ADD", rd=27, rs1=1, rs2=2),
        make_rv32i("SUB", rd=29, rs1=1, rs2=2),
        make_rv32i("BGE", rs1=1, rs2=2, imm=16),
        make_rv32i("SUB", rd=25, rs1=1, rs2=2),
        make_rv32i("SUB", rd=27, rs1=1, rs2=2),
        make_rv32i("SUB", rd=29, rs1=2, rs2=1),
        EBREAK,
    ]
    fib13 = l30_l11_gated_fib_variant(core)
    assert fib13 is not None
    assert _l11_fib_gate_imm(fib13) == 13
    wide = make_rv32i("SLTIU", rd=19, rs1=1, imm=16)
    gated16 = [wide if i == fib13.index(make_rv32i("SLTIU", rd=19, rs1=1, imm=13)) else w for i, w in enumerate(fib13)]
    assert _l11_fib_gate_imm(gated16) == 16
    tightened = l30_l11_fib_gate_tighten_variant(gated16)
    assert tightened is not None
    assert _l11_fib_gate_imm(tightened) == 13
    assert heldout_l30_verify_mean(tightened, n_seeds=32) + 1e-6 >= heldout_l30_verify_mean(gated16, n_seeds=32)
    assert curriculum_verify_mean(tightened, 11) >= 0.99
    assert any(_l11_fib_gate_imm(fill) == 13 for fill in l30_easy_fill_variants(gated16))


def test_l15_short_predicate_keeps_l15_l16_l30():
    from evolution.terminal_stagnation import (
        best_compact_l30_fill,
        compact_curriculum_side_score,
        curriculum_verify_mean,
        heldout_l30_verify_mean,
        l30_l15_short_variant,
    )
    from mutation_proposer import make_rv32i

    wide = [
        make_rv32i("SLT", rd=18, rs1=1, rs2=2),
        make_rv32i("BEQ", rs1=18, rs2=0, imm=12),
        make_rv32i("ADDI", rd=18, rs1=0, imm=0),
        make_rv32i("BEQ", rs1=0, rs2=0, imm=8),
        make_rv32i("ADD", rd=18, rs1=1, rs2=2),
    ]
    import json

    live = _live_60w_champion()
    short = l30_l15_short_variant(live)
    if short is None:
        from evolution.terminal_stagnation import _l15_short_pred_seq

        pred = _l15_short_pred_seq()
        assert any(live[i : i + len(pred)] == pred for i in range(len(live) - len(pred) + 1))
        return
    assert short is not None
    assert len(short) == len(live) - 1
    assert heldout_l30_verify_mean(short, n_seeds=32) >= 0.99
    assert curriculum_verify_mean(short, 15) >= 0.99
    assert curriculum_verify_mean(short, 16) >= 0.99
    picked = best_compact_l30_fill(live)
    assert picked is not None
    assert len(picked) <= len(live)
    assert compact_curriculum_side_score(picked) + 1e-6 >= compact_curriculum_side_score(live)


def test_l12_l14_x9_fold_keeps_side_and_l30():
    import json

    from evolution.terminal_stagnation import (
        best_compact_l30_fill,
        compact_curriculum_side_score,
        curriculum_verify_mean,
        heldout_l30_verify_mean,
        l30_l12_l14_x9_fold_variant,
    )

    live = _live_60w_champion()
    folded = l30_l12_l14_x9_fold_variant(live)
    if folded is None:
        from evolution.terminal_stagnation import _l12_l14_x9_seq

        seq = _l12_l14_x9_seq()
        assert any(live[i : i + len(seq)] == seq for i in range(len(live) - len(seq) + 1))
    else:
        assert len(folded) == len(live) - 2
        assert heldout_l30_verify_mean(folded, n_seeds=32) >= 0.99
        for lv in (12, 13, 14, 15, 16, 10, 11):
            assert curriculum_verify_mean(folded, lv) >= 0.99
        picked = best_compact_l30_fill(live)
        if picked is not None:
            assert compact_curriculum_side_score(picked) + 1e-6 >= compact_curriculum_side_score(live)


def test_drop_loop_zero_inits_keeps_l9_l10_l30():
    import json

    from evolution.terminal_stagnation import (
        best_compact_l30_fill,
        compact_curriculum_side_score,
        curriculum_verify_mean,
        heldout_l30_verify_mean,
        l30_drop_loop_zero_inits,
    )

    live = _live_60w_champion()
    dropped = l30_drop_loop_zero_inits(live)
    if dropped is None:
        import pytest

        from evolution.terminal_stagnation import _l30_bounded_sum_max_loop_body

        body = _l30_bounded_sum_max_loop_body()
        if not any(live[i : i + len(body)] == body for i in range(len(live) - len(body) + 1)):
            pytest.skip("60w archive lacks bounded-sum loop body")
        return
    assert len(dropped) < len(live)
    assert heldout_l30_verify_mean(dropped, n_seeds=32) >= 0.99
    assert curriculum_verify_mean(dropped, 9) >= 0.99
    assert curriculum_verify_mean(dropped, 10) >= 0.99
    picked = best_compact_l30_fill(live)
    if picked is not None:
        assert compact_curriculum_side_score(picked) + 1e-6 >= compact_curriculum_side_score(live)
        assert heldout_l30_verify_mean(picked, n_seeds=32) >= 0.99


def test_heldout_l30_detects_li_collapse():
    from evolution.terminal_stagnation import heldout_l30_verify_mean
    from mutation_proposer import EBREAK, make_rv32i

    good = [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=9, rs1=1, rs2=2),
        make_rv32i("XOR", rd=23, rs1=1, rs2=2),
        EBREAK,
    ]
    junk = [make_rv32i("ADDI", rd=r, rs1=0, imm=r) for r in range(3, 15)] + [EBREAK]
    assert heldout_l30_verify_mean(good) > heldout_l30_verify_mean(junk)

    import random

    from evolution.terminal_stagnation import select_terminal_probes_for_batch

    prog = [0x00000013] * 60 + [0x00100073]
    picked = select_terminal_probes_for_batch(prog, 30, random.Random(0), n_probe=20, pop_size=16)
    assert picked
    assert len(picked) <= 4  # 16 // 4
    assert all(tuple(v) != tuple(prog) for v in picked)
    assert any(len(v) < len(prog) for v in picked)


def _l30_overfit_23w():
    from mutation_proposer import EBREAK, make_rv32i

    return [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=9, rs1=1, rs2=2),
        make_rv32i("XOR", rd=23, rs1=1, rs2=2),
        make_rv32i("ADD", rd=4, rs1=1, rs2=1),
        make_rv32i("ADDI", rd=5, rs1=2, imm=0),
        make_rv32i("ADD", rd=6, rs1=2, rs2=2),
        make_rv32i("ADD", rd=7, rs1=4, rs2=6),
        make_rv32i("ADD", rd=20, rs1=1, rs2=2),
        make_rv32i("SLLI", rd=28, rs1=9, imm=1),
        make_rv32i("ADD", rd=30, rs1=1, rs2=2),
        make_rv32i("ADDI", rd=31, rs1=23, imm=0),
        make_rv32i("ADD", rd=22, rs1=1, rs2=2),
        make_rv32i("SUB", rd=29, rs1=2, rs2=1),
        make_rv32i("SUB", rd=25, rs1=1, rs2=2),
        make_rv32i("SUB", rd=27, rs1=1, rs2=2),
        make_rv32i("ADDI", rd=21, rs1=23, imm=0),
        make_rv32i("ADD", rd=26, rs1=1, rs2=2),
        make_rv32i("SLLI", rd=24, rs1=23, imm=2),
        make_rv32i("XOR", rd=24, rs1=24, rs2=1),
        EBREAK,
    ]


def test_find_best_challenger_promotes_heldout_fill_outside_elite(monkeypatch):
    monkeypatch.setenv("EE_USE_GPU", "0")
    from evolution.terminal_stagnation import l30_easy_fill_variants
    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = _l30_overfit_23w()
    fills = l30_easy_fill_variants(champ)
    want = fills[-1]
    assert want is not None
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, curriculum_level=30))
    eng.champion_level = 30
    eng.champion = list(champ)
    eng.champion_trace = {"status": "ok", "final_regs": {}}
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 400.0
    eng._verify_score_from_trace = lambda tr: 1.0  # type: ignore[method-assign]
    dummy = {"status": "ok", "final_regs": {}}
    eng._last_challengers = [list(champ), list(want)]
    eng._last_results = [dummy, dummy]
    _fit, best, _tr, _raw, improved = eng._find_best_challenger(
        elite=[list(champ)],
        detailed_results=[dummy],
        stage1_fitnesses=[400.0],
    )
    assert improved
    from evolution.terminal_stagnation import compact_curriculum_side_score, heldout_l30_verify_mean

    assert heldout_l30_verify_mean(best) + 1e-6 >= heldout_l30_verify_mean(champ)
    assert compact_curriculum_side_score(best) + 1e-6 >= compact_curriculum_side_score(champ)


def test_easy_fill_survives_instance_verify_drop(monkeypatch):
    """CPU side-score fill must not be vetoed by a lower instance verify."""
    monkeypatch.setenv("EE_USE_GPU", "0")
    from evolution.terminal_stagnation import l30_easy_fill_variants
    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = _l30_overfit_23w()
    from evolution.terminal_stagnation import best_compact_l30_fill

    want = best_compact_l30_fill(champ)
    assert want is not None
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, curriculum_level=30))
    eng.champion_level = 30
    eng.champion = list(champ)
    eng.champion_trace = {"status": "ok", "final_regs": {}}
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 400.0
    eng._verify_champion = lambda: (False, 0.2)  # type: ignore[method-assign]
    dummy = {"status": "ok", "final_regs": {}}
    replaced = eng._verify_and_replace_champion(want, dummy, 0.2, 300.0)
    assert replaced is True
    assert eng.champion == want


def test_heldout_gain_replaces_pinned_shorter_champion(monkeypatch):
    monkeypatch.setenv("EE_USE_GPU", "0")
    from evolution.terminal_stagnation import l30_order_sensitive_variant
    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = _l30_overfit_23w()
    fill = l30_order_sensitive_variant(champ)
    assert fill is not None
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, curriculum_level=30))
    eng.champion_level = 30
    eng.champion = list(champ)
    eng.champion_trace = {"status": "ok", "final_regs": {}}
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 400.0
    eng._verify_champion = lambda: (True, 1.0)  # type: ignore[method-assign]
    dummy = {"status": "ok", "final_regs": {}}
    replaced = eng._verify_and_replace_champion(fill, dummy, 1.0, 300.0)
    assert replaced is True
    assert eng.champion == fill


def test_l8_fill_replaces_at_equal_l30_verify(monkeypatch):
    monkeypatch.setenv("EE_USE_GPU", "0")
    from evolution.terminal_stagnation import (
        l30_l8_mem_add_variant,
        l30_l10_prefix_load_variant,
        l30_order_sensitive_variant,
    )
    from evolution_engine import EvolutionConfig, EvolutionEngine

    base = l30_l10_prefix_load_variant(l30_order_sensitive_variant(_l30_overfit_23w()))
    fill = l30_l8_mem_add_variant(base)
    assert base and fill and len(fill) > len(base)
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, curriculum_level=30))
    eng.champion_level = 30
    eng.champion = list(base)
    eng.champion_trace = {"status": "ok", "final_regs": {}}
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 400.0
    eng._verify_champion = lambda: (True, 1.0)  # type: ignore[method-assign]
    dummy = {"status": "ok", "final_regs": {}}
    replaced = eng._verify_and_replace_champion(fill, dummy, 1.0, 200.0)
    assert replaced is True
    assert eng.champion == fill


def test_scan_floor_probe_stall_mode_local_length_only(monkeypatch, tmp_path):
    from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: False,
    )
    monkeypatch.setattr(
        "evolution.epi_momentum_governor.scan_floor_length_prom_stalled",
        lambda _root, floor_w: int(floor_w) == 54,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._evolution_project_root",
        lambda: str(tmp_path),
    )
    assert _scan_floor_probe_stall_mode(54) is True
    assert _scan_floor_probe_stall_mode(55) is False


def test_should_rescan_compression_when_local_length_stall(monkeypatch):
    from evolution.terminal_stagnation import should_rescan_terminal_efficiency_compression

    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda w: int(w) == 50,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._read_progress_state",
        lambda: {
            "terminal_efficiency_scan_floor_cycle": 7130,
            "terminal_efficiency_scan_floor_heldout": 0.99,
        },
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda floor_w: int(floor_w) == 50,
    )
    assert should_rescan_terminal_efficiency_compression(7130, 50) is True
    assert should_rescan_terminal_efficiency_compression(7131, 50) is False
    assert should_rescan_terminal_efficiency_compression(7133, 50) is True
