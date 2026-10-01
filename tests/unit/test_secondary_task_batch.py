"""Secondary task eval must match serial results when batched."""

import random

from evolution_engine import EvolutionConfig, EvolutionEngine
from mutation_proposer import EBREAK, make_rv32i


def test_secondary_many_matches_one_at_a_time():
    cfg = EvolutionConfig(
        max_generations=1,
        population_size=4,
        use_gpu=False,
        use_secondary_tasks=True,
        auto_curriculum=False,
    )
    engine = EvolutionEngine(config=cfg)
    engine.rng = random.Random(0)
    engine.champion_level = 30
    engine._cached_eval_budget = 500
    prog = [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=4, rs1=1, rs2=2),
        EBREAK,
    ]
    serial = []
    engine.rng = random.Random(0)
    serial.append(engine._evaluate_secondary_tasks(prog))
    serial.append(engine._evaluate_secondary_tasks(prog))
    engine.rng = random.Random(0)
    batched = engine._evaluate_secondary_tasks_many([prog, prog])
    assert len(batched) == 2
    for one, many in zip(serial, batched, strict=True):
        assert len(one) == len(many)
        for (t_a, tr_a), (t_b, tr_b) in zip(one, many, strict=True):
            assert t_a["inputs"] == t_b["inputs"]
            assert tr_a.get("final_regs") == tr_b.get("final_regs")
            assert tr_a.get("instructions_count") == tr_b.get("instructions_count")
