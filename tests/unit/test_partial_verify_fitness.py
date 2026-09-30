"""Per-program partial verify scaling at terminal depth."""

import os
import random
import unittest


class TestPartialVerifyFitness(unittest.TestCase):
    def test_full_verify_program_not_scaled_by_champion_partial(self):
        os.environ.setdefault("EE_USE_GPU", "0")
        from evolution.fitness_helpers import partial_verify_multiplier
        from evolution_engine import EvolutionConfig, EvolutionEngine

        eng = EvolutionEngine(EvolutionConfig(curriculum_level=30))
        eng.champion_level = 30
        eng._champion_partial_verify = 0.842
        eng._refresh_task_instance()
        rows = __import__("json").load(open(".agents/champion_carry_forward.json"))  # use multi_cycle if needed
        # load L30 champion from last line of multi_cycle_results.jsonl
        import json

        lines = [json.loads(l) for l in open("multi_cycle_results.jsonl") if l.strip()]
        prog = [int(x, 16) if isinstance(x, str) else int(x) for x in lines[-1]["champion_program"]]
        trace = eng._evaluate(prog)
        eng._champion_verify_score = 0.842
        mult = partial_verify_multiplier(eng, trace)
        self.assertGreater(mult, 0.99, f"1.0-verify program should not use champion 0.842 partial, got {mult}")
        fit = eng._compute_fitness(prog, trace)
        self.assertGreater(fit, 150.0, f"expected full fitness ~264, got {fit}")


if __name__ == "__main__":
    unittest.main()
