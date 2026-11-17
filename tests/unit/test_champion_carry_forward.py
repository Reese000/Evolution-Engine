"""Tests for evolution.champion_carry_forward."""

import json

from evolution.champion_carry_forward import (
    _read_store,
    load_carry_forward_champion,
    load_carry_forward_start_level,
    persist_carry_forward_champion,
)


def _prog(n: int = 36) -> list[str]:
    return [f"0x{i:08X}" for i in range(n)] + ["0x00100073"]


def test_persist_and_load_same_level(tmp_path):
    evo = {
        "cycle": 42,
        "final_level": 21,
        "champion_verify_score": 0.75,
        "champion_words": 36,
        "final_fitness": 80.0,
        "champion_program": _prog(),
    }
    persist_carry_forward_champion(str(tmp_path), evo)
    loaded = load_carry_forward_champion(str(tmp_path), 21)
    assert loaded is not None
    words, desc = loaded
    assert len(words) == 37
    assert "carry_forward_cycle_42" in desc


def test_persist_keeps_both_levels(tmp_path):
    l22 = {
        "cycle": 10,
        "final_level": 22,
        "champion_verify_score": 0.769,
        "champion_words": 45,
        "final_fitness": 69.0,
        "champion_program": _prog(45),
    }
    l21 = {
        "cycle": 11,
        "final_level": 21,
        "champion_verify_score": 0.75,
        "champion_words": 36,
        "final_fitness": 103.0,
        "champion_program": _prog(),
    }
    persist_carry_forward_champion(str(tmp_path), l22)
    persist_carry_forward_champion(str(tmp_path), l21)
    levels = _read_store(str(tmp_path / ".agents" / "champion_carry_forward.json"))
    assert "21" in levels and "22" in levels
    assert load_carry_forward_champion(str(tmp_path), 21) is not None
    assert load_carry_forward_champion(str(tmp_path), 22) is not None


def test_load_l22_with_lower_verify_floor(tmp_path):
    evo = {
        "cycle": 10,
        "final_level": 22,
        "champion_verify_score": 0.692,
        "champion_words": 36,
        "final_fitness": 69.0,
        "champion_program": _prog(),
    }
    persist_carry_forward_champion(str(tmp_path), evo)
    assert load_carry_forward_champion(str(tmp_path), 22) is not None


def test_persist_and_load_l23_nudge_quality(tmp_path):
    evo = {
        "cycle": 2994,
        "final_level": 23,
        "champion_verify_score": 0.643,
        "champion_words": 44,
        "final_fitness": 54.88,
        "champion_program": _prog(44),
    }
    persist_carry_forward_champion(str(tmp_path), evo)
    assert load_carry_forward_champion(str(tmp_path), 23) is not None


def test_load_depth_nudge_plus_one(tmp_path):
    evo = {
        "cycle": 10,
        "final_level": 21,
        "champion_verify_score": 0.75,
        "champion_words": 36,
        "final_fitness": 80.0,
        "champion_program": _prog(),
    }
    persist_carry_forward_champion(str(tmp_path), evo)
    assert load_carry_forward_champion(str(tmp_path), 22) is not None


def test_load_carry_forward_start_level(tmp_path):
    l22 = {
        "cycle": 10,
        "final_level": 22,
        "champion_verify_score": 0.769,
        "champion_words": 45,
        "final_fitness": 84.87,
        "champion_program": _prog(45),
    }
    l21 = {
        "cycle": 11,
        "final_level": 21,
        "champion_verify_score": 0.75,
        "champion_words": 36,
        "final_fitness": 96.0,
        "champion_program": _prog(),
    }
    persist_carry_forward_champion(str(tmp_path), l22)
    persist_carry_forward_champion(str(tmp_path), l21)
    assert load_carry_forward_start_level(str(tmp_path), sustained_cap=21) == 22


def test_persist_skips_compact_l30_trusted_one(tmp_path):
    honest = {
        "cycle": 4876,
        "final_level": 30,
        "champion_verify_score": 0.737,
        "champion_words": 24,
        "final_fitness": 167.8,
        "champion_program": _prog(23),
    }
    lie = {
        "cycle": 4878,
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 22,
        "final_fitness": 3288.0,
        "champion_program": _prog(21),
    }
    persist_carry_forward_champion(str(tmp_path), honest)
    persist_carry_forward_champion(str(tmp_path), lie)
    loaded = load_carry_forward_champion(str(tmp_path), 30)
    assert loaded is not None
    words, desc = loaded
    assert len(words) == 24
    assert "4876" in desc


def test_persist_compact_l30_heldout_one_as_pin(tmp_path):
    from evolution.terminal_stagnation import l30_order_sensitive_variant
    from mutation_proposer import EBREAK, make_rv32i

    overfit = [
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
    ordered = l30_order_sensitive_variant(overfit)
    assert ordered is not None
    persist_carry_forward_champion(
        str(tmp_path),
        {
            "cycle": 1,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": len(overfit),
            "final_fitness": 400.0,
            "champion_program": [hex(w) for w in overfit],
        },
    )
    persist_carry_forward_champion(
        str(tmp_path),
        {
            "cycle": 2,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": len(ordered),
            "final_fitness": 200.0,
            "champion_program": [hex(w) for w in ordered],
        },
    )
    assert load_carry_forward_champion(str(tmp_path), 30) is None
    store = _read_store(str(tmp_path / ".agents" / "champion_carry_forward.json"))
    assert store["30"]["verify_score"] == 0.989
    assert store["30"]["heldout_l30"] >= 0.99
    assert store["30"]["cycle"] == 2


def test_carry_prefers_higher_promotion_raw_over_reporting_fitness(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evolution.terminal_stagnation.heldout_l30_verify_mean",
        lambda _p, _n=32: 1.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.compact_curriculum_side_score",
        lambda _p: 1.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda _p, _l: 1.0,
    )
    stale = _prog(59)
    better = list(stale)
    better[39] = "0xFE008CE3"
    base = {
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 60,
    }
    persist_carry_forward_champion(
        str(tmp_path),
        {
            **base,
            "cycle": 5710,
            "final_fitness": 384.9,
            "task_fitness": 302.06,
            "promotion_raw_fitness": 312.81,
            "champion_program": stale,
        },
    )
    persist_carry_forward_champion(
        str(tmp_path),
        {
            **base,
            "cycle": 5964,
            "final_fitness": 349.4,
            "task_fitness": 296.0,
            "promotion_raw_fitness": 313.31,
            "champion_program": better,
        },
    )
    store = _read_store(str(tmp_path / ".agents" / "champion_carry_forward.json"))
    assert store["30"]["cycle"] == 5964
    assert store["30"]["promotion_raw_fitness"] == 313.31


def test_carry_equal_length_higher_fitness_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evolution.terminal_stagnation.heldout_l30_verify_mean",
        lambda _p, _n=32: 1.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.compact_curriculum_side_score",
        lambda _p: 1.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda _p, _l: 1.0,
    )
    prog = _prog(59)
    base = {
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 60,
        "champion_program": prog,
    }
    persist_carry_forward_champion(str(tmp_path), {**base, "cycle": 10, "final_fitness": 340.0})
    persist_carry_forward_champion(str(tmp_path), {**base, "cycle": 11, "final_fitness": 360.0})
    store = _read_store(str(tmp_path / ".agents" / "champion_carry_forward.json"))
    assert store["30"]["cycle"] == 11
    assert store["30"]["fitness"] == 360.0


def test_persist_prefers_l10_when_l30_tied(tmp_path):
    from evolution.terminal_stagnation import l30_l10_prefix_load_variant, l30_order_sensitive_variant
    from mutation_proposer import EBREAK, make_rv32i

    overfit = [
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
    ordered = l30_order_sensitive_variant(overfit)
    loaded = l30_l10_prefix_load_variant(ordered)
    assert ordered and loaded
    persist_carry_forward_champion(
        str(tmp_path),
        {
            "cycle": 1,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": len(ordered),
            "final_fitness": 400.0,
            "champion_program": [hex(w) for w in ordered],
        },
    )
    persist_carry_forward_champion(
        str(tmp_path),
        {
            "cycle": 2,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": len(loaded),
            "final_fitness": 200.0,
            "champion_program": [hex(w) for w in loaded],
        },
    )
    assert load_carry_forward_champion(str(tmp_path), 30) is None
    store = _read_store(str(tmp_path / ".agents" / "champion_carry_forward.json"))
    assert store["30"]["cycle"] == 2


def test_enforce_restores_carry_on_equal_l30_side_gain(monkeypatch, tmp_path):
    monkeypatch.setenv("EE_USE_GPU", "0")
    from evolution.terminal_stagnation import best_compact_l30_fill
    from evolution_engine import EvolutionConfig, EvolutionEngine
    from mutation_proposer import EBREAK, make_rv32i

    champ = [
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
    fill = best_compact_l30_fill(champ)
    assert fill is not None and len(fill) > len(champ)
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, curriculum_level=30))
    eng.champion_level = 30
    eng.champion = list(champ)
    eng.champion_trace = {"status": "ok"}
    eng._champion_verify_score = 0.989
    monkeypatch.setattr(
        "evolution.champion_carry_forward.load_carry_forward_champion",
        lambda *_a, **_k: (list(fill), "carry_test"),
    )
    assert eng._enforce_l30_certified_champion() is True
    assert eng.champion == fill
