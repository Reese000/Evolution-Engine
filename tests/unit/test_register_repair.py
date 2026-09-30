"""Tests for evolution.register_repair — high-curriculum self-sustainment."""

from __future__ import annotations

import os
import random
import unittest

from evolution.register_repair import (
    critical_wrong_register,
    reference_words,
    register_repair_boost_active,
    repair_program,
    scaled_eval_budget,
    score_target_register,
    should_reseed_champion,
    target_register,
    trace_guided_immediate_variants,
    weighted_register_match,
    wrong_level_registers,
)
from mutation_proposer import TraceAnalysis


class TestRegisterRepair(unittest.TestCase):
    def test_l17_target_register(self) -> None:
        self.assertEqual(target_register(17), 20)
        self.assertEqual(target_register(19), 22)

    def test_l17_reference_words_nonempty(self) -> None:
        words, desc = reference_words(17, segment="full")
        self.assertGreater(len(words), 10)
        self.assertIn("l17", desc)

    def test_scaled_eval_budget_increases_at_l17(self) -> None:
        base = 5000
        self.assertGreater(scaled_eval_budget(base, 19), scaled_eval_budget(base, 10))

    def test_critical_wrong_register_prefers_target(self) -> None:
        wrong = {3: {"actual": 0, "expected": 1}, 20: {"actual": 0, "expected": 42}}
        self.assertEqual(critical_wrong_register(17, wrong), 20)

    def test_score_target_register_exact(self) -> None:
        score = score_target_register(17, {20: 100}, {20: 100})
        self.assertAlmostEqual(score, 1.0)

    def test_weighted_register_match_boosts_target(self) -> None:
        expected = {3: 1, 4: 2, 20: 100}
        finals_good = {3: 1, 4: 2, 20: 100}
        finals_bad_target = {3: 1, 4: 2, 20: 50}
        good = weighted_register_match(17, expected, 3, finals_good)
        bad = weighted_register_match(17, expected, 3, finals_bad_target)
        self.assertGreater(good, bad)

    def test_repair_program_injects_instructions(self) -> None:
        prog = [0x00000013, 0x00100073]  # NOP, EBREAK
        analysis = TraceAnalysis(
            divergence_point=1,
            divergence_type="wrong_reg_value",
            final_regs=[],
            trace_length=2,
            suggestion={"type": "fix_register", "reg": 20},
            expected={},
            inputs={},
        )
        rng = random.Random(42)
        mutated, desc = repair_program(prog, 17, analysis, rng)
        self.assertGreater(len(mutated), len(prog))
        self.assertIn("register_repair", desc)

    def test_should_reseed_only_when_deeply_stuck(self) -> None:
        self.assertFalse(should_reseed_champion(18, 0.55, 85.0))
        self.assertFalse(should_reseed_champion(18, 0.60, -50.0))
        self.assertTrue(should_reseed_champion(18, 0.20, 85.0))
        self.assertTrue(should_reseed_champion(18, 0.50, -10.0))

    def test_l18_reference_includes_lui_mask(self) -> None:
        words, desc = reference_words(18, segment="full")
        self.assertGreater(len(words), 20, desc)
        from milestone1.rv32i_mask import decode

        tail = words[-5:]
        names = [decode(w, validate=False).get("name") for w in tail]
        self.assertEqual(names[-4:], ["XOR", "LUI", "ADDI", "AND"])

    def test_register_repair_boost_reads_tick_state(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            tick_path = os.path.join(agents, "tick_state.json")
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump({"epi_plateau_boost": True, "force_register_repair": True}, f)
            old_env = os.environ.pop("EE_REGISTER_REPAIR_BOOST", None)
            try:
                import evolution.register_repair as rr

                orig = rr._project_root
                rr._project_root = lambda: tmp  # type: ignore[method-assign]
                self.assertTrue(rr.register_repair_boost_active())
            finally:
                rr._project_root = orig  # type: ignore[method-assign]
                if old_env is not None:
                    os.environ["EE_REGISTER_REPAIR_BOOST"] = old_env

    def test_repair_program_boost_replaces_instead_of_stacking(self) -> None:
        prog = [0x00000013] * 40 + [0x00100073]
        analysis = TraceAnalysis(
            divergence_point=1,
            divergence_type="wrong_reg_value",
            final_regs=[],
            trace_length=2,
            suggestion={"type": "fix_register", "reg": 20},
            expected={},
            inputs={},
        )
        rng = random.Random(0)  # roll 0.0 < full_thresh under boost
        old_env = os.environ.get("EE_REGISTER_REPAIR_BOOST")
        os.environ["EE_REGISTER_REPAIR_BOOST"] = "1"
        try:
            mutated, desc = repair_program(prog, 19, analysis, rng)
            self.assertIn("register_repair_replace", desc)
            self.assertLess(len(mutated), len(prog))
        finally:
            if old_env is None:
                os.environ.pop("EE_REGISTER_REPAIR_BOOST", None)
            else:
                os.environ["EE_REGISTER_REPAIR_BOOST"] = old_env

    def test_l21_tail_repair_inserts_not_replaces(self) -> None:
        from evolution.register_repair import _wholesale_replace_allowed, reference_words

        tail_words, _ = reference_words(21, segment="level_only")
        self.assertGreater(len(tail_words), 0)
        self.assertLess(len(tail_words), 20)
        self.assertFalse(_wholesale_replace_allowed(21, "level_only", tail_words))

        prog = [0x00000013] * 36 + [0x00100073]
        analysis = TraceAnalysis(
            divergence_point=1,
            divergence_type="wrong_reg_value",
            final_regs=[],
            trace_length=2,
            suggestion={"type": "fix_register", "reg": 24},
            expected={},
            inputs={},
        )
        rng = random.Random(99)
        old_env = os.environ.get("EE_REGISTER_REPAIR_BOOST")
        os.environ["EE_REGISTER_REPAIR_BOOST"] = "1"
        try:
            rolls = [rng.random() for _ in range(3)]
            mutated, desc = repair_program(prog, 21, analysis, random.Random(99))
            if rolls[0] >= 0.92 and rolls[1] >= 0.98:
                self.assertNotIn("register_repair_replace", desc)
                self.assertGreater(len(mutated), len(tail_words))
            else:
                full_words, _ = reference_words(21, segment="full")
                self.assertGreaterEqual(len(mutated), int(0.85 * len(full_words)))
        finally:
            if old_env is None:
                os.environ.pop("EE_REGISTER_REPAIR_BOOST", None)
            else:
                os.environ["EE_REGISTER_REPAIR_BOOST"] = old_env

    def test_terminal_mode_uses_incremental_not_replace(self) -> None:
        import json
        import random
        import tempfile

        from evolution.register_repair import repair_program
        from mutation_proposer import TraceAnalysis

        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            with open(os.path.join(agents, "tick_state.json"), "w", encoding="utf-8") as f:
                json.dump({"terminal_diversity_boost": True}, f)
            import evolution.register_repair as rr

            old_root = rr._project_root
            rr._project_root = lambda: tmp  # type: ignore[method-assign]
            try:
                prog = [0x00000013] * 60 + [0x00100073]
                analysis = TraceAnalysis(
                    divergence_point=1,
                    divergence_type="wrong_reg_value",
                    final_regs=[],
                    trace_length=2,
                    suggestion={"type": "fix_register", "reg": 30},
                    expected={},
                    inputs={},
                )
                mutated, desc = repair_program(prog, 30, analysis, random.Random(1))
                self.assertNotIn("register_repair_replace", desc)
                self.assertIn("terminal_incr", desc)
            finally:
                rr._project_root = old_root  # type: ignore[method-assign]

    def test_trace_guided_immediate_variants_l30_champion(self) -> None:
        import json
        import os

        os.environ.setdefault("EE_USE_GPU", "0")
        from evolution_engine import EvolutionConfig, EvolutionEngine

        results_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "multi_cycle_results.jsonl",
        )
        prog = None
        if os.path.isfile(results_path):
            rows = [json.loads(line) for line in open(results_path, encoding="utf-8") if line.strip()]
            for row in reversed(rows):
                if int(row.get("final_level") or 0) != 30:
                    continue
                if float(row.get("champion_verify_score") or 0) >= 0.84:
                    if float(row.get("champion_verify_score") or 0) < 0.85:
                        prog = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
                        break
        if prog is None:
            cf_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                ".agents",
                "champion_carry_forward.json",
            )
            if not os.path.isfile(cf_path):
                self.skipTest("no L30 plateau champion snapshot")
            cf = json.load(open(cf_path, encoding="utf-8"))
            level_data = cf.get("levels", {}).get("30")
            if not level_data:
                self.skipTest("L30 champion missing")
            prog = [int(x, 16) if isinstance(x, str) else int(x) for x in level_data["champion_program"]]
        eng = EvolutionEngine(EvolutionConfig(curriculum_level=30))
        eng._refresh_task_instance()
        trace = eng._evaluate(prog)
        base = eng._verify_score_from_trace(trace)
        wrong = wrong_level_registers(30, trace, eng.task_instance)
        if not wrong:
            self.skipTest("plateau champion no longer reproduces wrong registers")
        self.assertIn(25, wrong)
        variants = trace_guided_immediate_variants(
            prog,
            30,
            trace,
            eng.task_instance,
            random.Random(0),
            max_evals=64,
        )
        self.assertGreater(len(variants), 0)
        best = base
        for variant in variants:
            scored = eng._verify_score_from_trace(eng._evaluate(variant))
            best = max(best, scored)
        self.assertGreater(best, base)
        self.assertGreaterEqual(best, 1.0)


if __name__ == "__main__":
    unittest.main()
