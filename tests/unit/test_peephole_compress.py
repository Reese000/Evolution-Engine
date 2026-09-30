"""Peephole compress must shorten ADDI chains without extra VM."""

import random

from evolution.terminal_stagnation import (
    li_mem_fold_compress,
    overwrite_dead_compress,
    peephole_compress,
    semantic_compress,
    terminal_efficiency_variants,
    zero_identity_compress,
)
from mutation_proposer import EBREAK, DirectedMutator, make_rv32i


def test_fold_addi_chain():
    prog = [
        make_rv32i("ADDI", rd=5, rs1=1, imm=2),
        make_rv32i("ADDI", rd=5, rs1=5, imm=3),
        EBREAK,
    ]
    out = peephole_compress(prog)
    assert len(out) == 2
    assert out[-1] == EBREAK
    assert out[0] == make_rv32i("ADDI", rd=5, rs1=1, imm=5)


def test_drop_addi_zero():
    prog = [
        make_rv32i("ADDI", rd=5, rs1=1, imm=1),
        make_rv32i("ADDI", rd=5, rs1=5, imm=0),
        EBREAK,
    ]
    out = peephole_compress(prog)
    assert len(out) == 2
    assert out[0] == make_rv32i("ADDI", rd=5, rs1=1, imm=1)


def test_efficiency_variants_include_peephole():
    prog = [
        make_rv32i("ADDI", rd=5, rs1=1, imm=2),
        make_rv32i("ADDI", rd=5, rs1=5, imm=3),
        make_rv32i("ADDI", rd=6, rs1=2, imm=1),
        EBREAK,
    ]
    variants = terminal_efficiency_variants(prog, random.Random(0), max_evals=16, max_drop_span=1)
    assert any(len(v) < len(prog) for v in variants)


def test_overwrite_dead_drops_killed_writes():
    prog = [
        make_rv32i("ADDI", rd=22, rs1=0, imm=1),
        make_rv32i("SLLI", rd=22, rs1=2, imm=20),
        make_rv32i("SUB", rd=22, rs1=1, rs2=2),
        EBREAK,
    ]
    out = overwrite_dead_compress(prog)
    assert out == [make_rv32i("SUB", rd=22, rs1=1, rs2=2), EBREAK]


def test_overwrite_dead_keeps_used_write():
    prog = [
        make_rv32i("ADDI", rd=22, rs1=0, imm=1),
        make_rv32i("OR", rd=21, rs1=21, rs2=22),
        make_rv32i("SUB", rd=22, rs1=1, rs2=2),
        EBREAK,
    ]
    out = overwrite_dead_compress(prog)
    assert len(out) == 4


def test_efficiency_variants_include_overwrite_dead():
    prog = [
        make_rv32i("ADD", rd=26, rs1=2, rs2=1),
        make_rv32i("SLLI", rd=26, rs1=26, imm=1),
        make_rv32i("ADDI", rd=26, rs1=0, imm=35),
        EBREAK,
    ]
    variants = terminal_efficiency_variants(prog, random.Random(0), max_evals=16, max_drop_span=1)
    compressed = overwrite_dead_compress(prog)
    assert len(compressed) == 2
    assert any(v == compressed for v in variants)


def test_li_mem_fold_store_and_load():
    addi = make_rv32i("ADDI", rd=23, rs1=0, imm=0)
    sw = make_rv32i("SW", rs1=23, rs2=1, imm=0)
    xor = make_rv32i("XOR", rd=23, rs1=1, rs2=2)
    prog = [addi, sw, xor, EBREAK]
    out = li_mem_fold_compress(prog)
    assert make_rv32i("SW", rs1=0, rs2=1, imm=0) in out
    assert addi not in out
    assert xor in out


def test_semantic_compress_live_18w_pattern():
    li = make_rv32i("ADDI", rd=25, rs1=0, imm=-11)
    lb = make_rv32i("LB", rd=31, rs1=25, imm=0)
    xor = make_rv32i("XOR", rd=31, rs1=31, rs2=2)
    li2 = make_rv32i("ADDI", rd=25, rs1=0, imm=-39)
    prog = [li, lb, xor, li2, EBREAK]
    out = semantic_compress(prog)
    assert len(out) < len(prog)
    assert out[-1] == EBREAK
    assert li2 in out


def test_zero_identity_drops_or_with_uninit_reg():
    ori = make_rv32i("ADDI", rd=21, rs1=0, imm=1)
    dead_or = make_rv32i("OR", rd=21, rs1=21, rs2=23)
    sub = make_rv32i("SUB", rd=22, rs1=1, rs2=2)
    prog = [ori, dead_or, sub, EBREAK]
    out = zero_identity_compress(prog)
    assert dead_or not in out
    assert ori in out
    assert sub in out


def test_semantic_compress_folds_li_or_and_identity():
    li = make_rv32i("ADDI", rd=22, rs1=0, imm=1)
    orr = make_rv32i("OR", rd=21, rs1=21, rs2=22)
    orz = make_rv32i("OR", rd=21, rs1=21, rs2=23)
    sub = make_rv32i("SUB", rd=22, rs1=1, rs2=2)
    prog = [li, orr, orz, sub, EBREAK]
    out = semantic_compress(prog)
    assert len(out) <= 3
    assert out[-1] == EBREAK
    assert sub in out


def test_saturated_shorter_swap_ok():
    from evolution.terminal_stagnation import saturated_shorter_swap_ok, terminal_elite_heap_key

    assert saturated_shorter_swap_ok(22, 13, 1.0, 1.0)
    assert not saturated_shorter_swap_ok(13, 22, 1.0, 1.0)
    assert not saturated_shorter_swap_ok(22, 13, 1.0, 0.9)
    scored = [(100.0, 0, [0] * 22, None), (100.0, 1, [0] * 13, None)]
    ordered = sorted(scored, key=lambda x: terminal_elite_heap_key(x[0], x[2]))
    assert len(ordered[0][2]) == 13


def test_traces_observably_equal():
    from evolution.terminal_stagnation import traces_observably_equal

    a = {"status": "ok", "final_regs": {3: 1, "9": 2}, "final_memory": {}}
    b = {"status": "ok", "final_regs": {3: 1, 9: 2, 4: 0}, "final_memory": {}}
    c = {"status": "ok", "final_regs": {3: 1, 9: 3}, "final_memory": {}}
    assert traces_observably_equal(a, b)
    assert not traces_observably_equal(a, c)


def test_inherit_verify_if_semantic_clone():
    from evolution.terminal_stagnation import inherit_verify_if_semantic_clone, semantic_compress
    from mutation_proposer import EBREAK, make_rv32i

    t = {"status": "ok", "final_regs": {3: 1}, "final_memory": {}}
    assert inherit_verify_if_semantic_clone(22, 13, t, t, 0.21, 1.0) == 1.0
    other = {"status": "ok", "final_regs": {3: 2}, "final_memory": {}}
    assert inherit_verify_if_semantic_clone(22, 13, t, other, 0.21, 1.0) == 0.21
    assert inherit_verify_if_semantic_clone(13, 22, t, t, 0.21, 1.0) == 0.21
    li = make_rv32i("ADDI", rd=22, rs1=0, imm=1)
    dead = make_rv32i("SLLI", rd=22, rs1=2, imm=20)
    sub = make_rv32i("SUB", rd=22, rs1=1, rs2=2)
    prog = [li, dead, sub, EBREAK]
    assert len(semantic_compress(prog)) < len(prog)


def test_ensure_semantic_compress_elite_prepends(monkeypatch):
    from types import SimpleNamespace

    from evolution.selection_mixin import SelectionMixin
    from mutation_proposer import EBREAK, make_rv32i

    class _E(SelectionMixin):
        pass

    eng = _E()
    eng.champion_level = 30
    eng.config = SimpleNamespace(max_level=30)
    eng._champion_verify_score = 1.0
    li = make_rv32i("ADDI", rd=22, rs1=0, imm=1)
    orr = make_rv32i("OR", rd=21, rs1=21, rs2=22)
    orz = make_rv32i("OR", rd=21, rs1=21, rs2=23)
    sub = make_rv32i("SUB", rd=22, rs1=1, rs2=2)
    # Longer than L30_COMPACT_MAX_WORDS so prepend remains active.
    eng.champion = [li, orr, orz, sub] * 20 + [EBREAK]
    elites, fits, idxs = eng._ensure_semantic_compress_elite([[1] * 40], [1.0], [3])
    assert -1 in idxs
    assert any(len(p) < len(eng.champion) for p in elites)


def test_ensure_semantic_compress_elite_skips_compact_l30():
    from types import SimpleNamespace

    from evolution.selection_mixin import SelectionMixin
    from mutation_proposer import EBREAK, make_rv32i

    class _E(SelectionMixin):
        pass

    eng = _E()
    eng.champion_level = 30
    eng.config = SimpleNamespace(max_level=30)
    eng._champion_verify_score = 1.0
    li = make_rv32i("ADDI", rd=22, rs1=0, imm=1)
    orr = make_rv32i("OR", rd=21, rs1=21, rs2=22)
    orz = make_rv32i("OR", rd=21, rs1=21, rs2=23)
    sub = make_rv32i("SUB", rd=22, rs1=1, rs2=2)
    eng.champion = [li, orr, orz, sub] * 8 + [EBREAK]
    elites, fits, idxs = eng._ensure_semantic_compress_elite([[1] * 40], [1.0], [3])
    assert -1 not in idxs
    assert all(len(p) >= len(eng.champion) for p in elites)


def test_compact_overlay_is_local():
    body = [0x00100093 + i for i in range(18)]
    prog = body + [EBREAK]
    out, desc = DirectedMutator._task_aware_mutation(list(prog), random.Random(1), 30)
    if desc == "task_aware_l30_easy_fill":
        assert 12 <= len(out) <= 72
    else:
        assert len(out) <= len(prog)
        if desc.startswith("task_aware_overlay"):
            assert len(out) == len(prog)
            matched = sum(1 for a, b in zip(out, prog) if a == b)
            assert matched >= len(prog) - 4
