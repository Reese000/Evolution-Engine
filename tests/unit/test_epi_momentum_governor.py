"""Tests for evolution.epi_momentum_governor."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from evolution.epi_momentum_governor import (
    apply_growth_interventions,
    apply_plateau_boost,
    assess_epi_momentum,
    assess_exponential_growth,
    assess_plateau_boost,
    clear_intervention_flags,
    promotion_raw_capability_stalled,
    run_cycle_governor,
    should_suppress_champion_reseed,
    should_suppress_register_repair,
    should_suppress_terminal_diversity_boost,
)


class TestEpiMomentumGovernor(unittest.TestCase):
    def test_goal_met_when_avg_above_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            path = os.path.join(agents, "progress_trajectory.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(100):
                    f.write(
                        json.dumps(
                            {
                                "cycle": i,
                                "epi_momentum": 0.6 + (i % 3) * 0.1,
                                "fitness": 200.0 + i * 0.2,
                                "epi": 80.0 + i * 0.1,
                            }
                        )
                        + "\n"
                    )
            report = assess_epi_momentum(tmp, window=100, target_avg=0.5)
            self.assertTrue(report.momentum_gate_met)
            self.assertTrue(report.growth_gate_met)
            self.assertTrue(report.goal_met)
            self.assertGreater(report.avg_momentum, 0.5)

    def test_goal_not_met_without_fitness_growth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            path = os.path.join(agents, "progress_trajectory.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(100):
                    f.write(
                        json.dumps(
                            {
                                "cycle": i,
                                "epi_momentum": 0.8,
                                "fitness": 264.04,
                                "epi": 87.5,
                            }
                        )
                        + "\n"
                    )
            report = assess_epi_momentum(tmp, window=100, target_avg=0.5)
            growth = assess_exponential_growth(tmp, window=100)
            self.assertTrue(report.momentum_gate_met)
            self.assertFalse(growth.growth_met)
            self.assertFalse(report.growth_gate_met)
            self.assertFalse(report.goal_met)

    def test_intervention_when_avg_below_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            path = os.path.join(agents, "progress_trajectory.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for _ in range(50):
                    f.write(json.dumps({"epi_momentum": 0.0}) + "\n")
            report = assess_epi_momentum(tmp, window=100, target_avg=0.5)
            self.assertFalse(report.goal_met)
            self.assertTrue(report.needs_intervention)
            apply_growth_interventions(tmp, report)
            tick = json.loads(open(os.path.join(agents, "tick_state.json"), encoding="utf-8").read())
            self.assertTrue(tick.get("force_champion_reseed"))

    def test_plateau_boost_on_negative_streak_without_reseed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            path = os.path.join(agents, "progress_trajectory.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(1000):
                    mom = 0.0 if i < 900 else (0.8 if i < 995 else -0.5)
                    f.write(json.dumps({"cycle": i, "epi_momentum": mom}) + "\n")
            report = assess_epi_momentum(tmp, window=1000, target_avg=0.5)
            self.assertFalse(report.needs_intervention)
            self.assertGreater(report.recent_avg, 0.5)
            boost = assess_plateau_boost(tmp, report)
            self.assertTrue(boost.needs_boost)
            self.assertGreaterEqual(boost.negative_streak, 5)
            apply_plateau_boost(tmp)
            tick = json.loads(open(os.path.join(agents, "tick_state.json"), encoding="utf-8").read())
            self.assertTrue(tick.get("force_register_repair"))
            self.assertNotIn("force_champion_reseed", tick)

    def test_plateau_escalation_reseed_on_long_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            path = os.path.join(agents, "progress_trajectory.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(1000):
                    mom = 0.0 if i < 900 else (0.8 if i < 990 else -0.5)
                    f.write(json.dumps({"cycle": i, "epi_momentum": mom, "level": 19}) + "\n")
            report = assess_epi_momentum(tmp, window=1000, target_avg=0.5)
            boost = assess_plateau_boost(tmp, report)
            self.assertTrue(boost.needs_boost)
            self.assertGreaterEqual(boost.negative_streak, 10)
            apply_plateau_boost(tmp, escalate_reseed=True, champion_level=19)
            tick = json.loads(open(os.path.join(agents, "tick_state.json"), encoding="utf-8").read())
            self.assertTrue(tick.get("force_champion_reseed"))
            self.assertEqual(tick.get("reseed_level"), 19)

        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            path = os.path.join(agents, "progress_trajectory.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(1000):
                    mom = 0.0 if i < 900 else 0.8
                    f.write(json.dumps({"cycle": i, "epi_momentum": mom}) + "\n")
            report = assess_epi_momentum(tmp, window=1000, target_avg=0.5)
            self.assertFalse(report.goal_met)
            self.assertGreater(report.recent_avg, 0.5)
            self.assertFalse(report.needs_intervention)

    def test_should_suppress_register_repair_on_positive_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            path = os.path.join(agents, "progress_trajectory.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(1000):
                    if i < 997:
                        mom = 0.0
                    else:
                        mom = 0.3 + (i - 997) * 0.1
                    f.write(json.dumps({"cycle": i, "epi_momentum": mom}) + "\n")
            report = assess_epi_momentum(tmp, window=1000, target_avg=0.5)
            self.assertTrue(report.needs_intervention)
            self.assertGreater(report.latest_momentum, 0.0)
            self.assertTrue(should_suppress_register_repair(tmp, report))
            apply_growth_interventions(tmp, report)
            tick = json.loads(open(os.path.join(agents, "tick_state.json"), encoding="utf-8").read())
            self.assertNotIn("force_register_repair", tick)
            self.assertTrue(tick.get("epi_momentum_intervention"))

    def test_should_suppress_champion_reseed_on_stable_l21(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            state_path = os.path.join(agents, "progress_controller_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "level": 21,
                        "verify_score": 0.75,
                        "champion_fitness": 80.0,
                        "cycles_at_level": 3,
                    },
                    f,
                )
            traj = os.path.join(agents, "progress_trajectory.jsonl")
            with open(traj, "w", encoding="utf-8") as f:
                for i in range(1000):
                    mom = 0.0 if i < 997 else 4.8
                    f.write(json.dumps({"cycle": i, "epi_momentum": mom}) + "\n")
            report = assess_epi_momentum(tmp, window=1000, target_avg=0.5)
            self.assertTrue(report.needs_intervention)
            self.assertTrue(should_suppress_champion_reseed(tmp, report))
            apply_growth_interventions(tmp, report)
            tick = json.loads(open(os.path.join(agents, "tick_state.json"), encoding="utf-8").read())
            self.assertNotIn("force_champion_reseed", tick)
            self.assertNotIn("force_reference_champion", tick)

    def test_should_suppress_plateau_boost_on_stable_l23(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            state_path = os.path.join(agents, "progress_controller_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "level": 23,
                        "verify_score": 0.643,
                        "raw_fitness": 74.0,
                        "cycles_at_level": 10,
                    },
                    f,
                )
            traj = os.path.join(agents, "progress_trajectory.jsonl")
            with open(traj, "w", encoding="utf-8") as f:
                for i in range(1000):
                    mom = 0.0 if i < 900 else (0.8 if i < 995 else -0.5)
                    f.write(json.dumps({"cycle": i, "epi_momentum": mom}) + "\n")
            report = assess_epi_momentum(tmp, window=1000, target_avg=0.5)
            self.assertTrue(should_suppress_register_repair(tmp, report))

    def test_deep_stable_l24_suppresses_intervention_reseed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            state_path = os.path.join(agents, "progress_controller_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "level": 24,
                        "verify_score": 0.6,
                        "raw_fitness": 71.6,
                        "cycles_at_level": 5,
                    },
                    f,
                )
            traj = os.path.join(agents, "progress_trajectory.jsonl")
            with open(traj, "w", encoding="utf-8") as f:
                for i in range(1000):
                    mom = 0.0 if i < 900 else -0.5
                    f.write(json.dumps({"cycle": i, "epi_momentum": mom}) + "\n")
            report = assess_epi_momentum(tmp, window=1000, target_avg=0.5)
            self.assertTrue(report.needs_intervention)
            self.assertTrue(should_suppress_champion_reseed(tmp, report))
            self.assertTrue(should_suppress_register_repair(tmp, report))
            tick_path = os.path.join(agents, "tick_state.json")
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump({"depth_nudge_level": 25}, f)
            apply_growth_interventions(tmp, report)
            tick = json.loads(open(tick_path, encoding="utf-8").read())
            self.assertEqual(tick.get("depth_nudge_level"), 25)
            self.assertNotIn("force_champion_reseed", tick)
            self.assertNotIn("force_register_repair", tick)

    def test_recovery_at_stable_l25_skips_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            state_path = os.path.join(agents, "progress_controller_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "level": 25,
                        "verify_score": 0.625,
                        "raw_fitness": 76.2,
                        "cycles_at_level": 5,
                    },
                    f,
                )
            traj = os.path.join(agents, "progress_trajectory.jsonl")
            with open(traj, "w", encoding="utf-8") as f:
                for i in range(1000):
                    if i < 990:
                        mom = -1.5
                    elif i < 993:
                        mom = -1.23
                    else:
                        mom = 0.07 + (i - 993) * 0.04
                    f.write(json.dumps({"cycle": i, "epi_momentum": mom, "level": 25}) + "\n")
            report = assess_epi_momentum(tmp, window=1000, target_avg=0.5)
            self.assertFalse(report.needs_intervention)
            self.assertGreater(report.latest_momentum, 0.0)

    def test_fresh_depth_hold_suppresses_reseed_at_l26(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            with open(os.path.join(agents, "progress_controller_state.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "level": 26,
                        "verify_score": 0.5882,
                        "raw_fitness": 66.2,
                        "cycles_at_level": 1,
                    },
                    f,
                )
            with open(os.path.join(agents, "progress_trajectory.jsonl"), "w", encoding="utf-8") as f:
                for i in range(100):
                    f.write(json.dumps({"cycle": i, "epi_momentum": -0.5}) + "\n")
            report = assess_epi_momentum(tmp, window=100, target_avg=0.5)
            self.assertTrue(report.needs_intervention)
            self.assertTrue(should_suppress_champion_reseed(tmp, report))
            apply_growth_interventions(tmp, report)
            tick = json.loads(open(os.path.join(agents, "tick_state.json"), encoding="utf-8").read())
            self.assertNotIn("force_champion_reseed", tick)

    def test_clear_intervention_preserves_timeout_reseed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            tick_path = os.path.join(agents, "tick_state.json")
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epi_momentum_intervention": True,
                        "epi_momentum_reseed": True,
                        "force_champion_reseed": True,
                        "force_register_repair": True,
                    },
                    f,
                )
            clear_intervention_flags(tmp)
            tick = json.loads(open(tick_path, encoding="utf-8").read())
            self.assertNotIn("epi_momentum_intervention", tick)
            self.assertNotIn("force_champion_reseed", tick)

    def test_clear_intervention_preserves_collapse_reseed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            tick_path = os.path.join(agents, "tick_state.json")
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epi_momentum_intervention": True,
                        "epi_momentum_reseed": True,
                        "force_champion_reseed": True,
                        "force_reference_champion": True,
                        "collapse_reseed": True,
                        "reseed_level": 21,
                    },
                    f,
                )
            clear_intervention_flags(tmp)
            tick = json.loads(open(tick_path, encoding="utf-8").read())
            self.assertTrue(tick.get("force_champion_reseed"))
            self.assertTrue(tick.get("collapse_reseed"))

    def test_clear_intervention_preserves_timeout_reseed_continued(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            tick_path = os.path.join(agents, "tick_state.json")
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "force_champion_reseed": True,
                        "reseed_level": 18,
                    },
                    f,
                )
            clear_intervention_flags(tmp)
            tick = json.loads(open(tick_path, encoding="utf-8").read())
            self.assertTrue(tick.get("force_champion_reseed"))
            self.assertEqual(tick.get("reseed_level"), 18)

    def test_terminal_depth_stagnation_uses_diversity_not_register_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            state_path = os.path.join(agents, "progress_controller_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "level": 30,
                        "verify_score": 0.579,
                        "raw_fitness": 64.7,
                        "cycles_at_level": 20,
                    },
                    f,
                )
            traj = os.path.join(agents, "progress_trajectory.jsonl")
            with open(traj, "w", encoding="utf-8") as f:
                for i in range(1000):
                    mom = 0.0 if i < 900 else -1.0
                    f.write(json.dumps({"cycle": i, "epi_momentum": mom, "level": 30}) + "\n")
            report = assess_epi_momentum(tmp, window=1000, target_avg=0.5)
            self.assertFalse(should_suppress_register_repair(tmp, report))
            boost = assess_plateau_boost(tmp, report)
            self.assertTrue(boost.needs_boost)
            apply_plateau_boost(tmp, report=report)
            tick = json.loads(open(os.path.join(agents, "tick_state.json"), encoding="utf-8").read())
            self.assertTrue(tick.get("terminal_diversity_boost"))
            self.assertTrue(tick.get("epi_plateau_boost"))
            self.assertNotIn("force_register_repair", tick)
            import evolution.register_repair as rr

            orig = rr._project_root
            rr._project_root = lambda: tmp  # type: ignore[method-assign]
            try:
                self.assertFalse(rr.register_repair_boost_active())
            finally:
                rr._project_root = orig  # type: ignore[method-assign]

    def test_terminal_intervention_blocks_reference_reseed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            state_path = os.path.join(agents, "progress_controller_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "level": 30,
                        "verify_score": 0.579,
                        "raw_fitness": 25.95,
                        "cycles_at_level": 36,
                    },
                    f,
                )
            traj = os.path.join(agents, "progress_trajectory.jsonl")
            with open(traj, "w", encoding="utf-8") as f:
                for i in range(1000):
                    mom = -1.0 if i >= 900 else 0.6
                    f.write(json.dumps({"cycle": i, "epi_momentum": mom, "level": 30}) + "\n")
            report = assess_epi_momentum(tmp, window=1000, target_avg=0.5)
            self.assertTrue(report.needs_intervention)
            self.assertTrue(should_suppress_champion_reseed(tmp, report))
            apply_growth_interventions(tmp, report)
            tick = json.loads(open(os.path.join(agents, "tick_state.json"), encoding="utf-8").read())
            self.assertTrue(tick.get("terminal_diversity_boost"))
            self.assertNotIn("force_champion_reseed", tick)
            self.assertNotIn("force_reference_champion", tick)
            self.assertNotIn("force_register_repair", tick)

    def test_terminal_stall_suppressed_during_sustained_goal_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            state_path = os.path.join(agents, "progress_controller_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "level": 30,
                        "verify_score": 0.579,
                        "raw_fitness": 400.0,
                        "cycles_at_level": 20,
                    },
                    f,
                )
            traj = os.path.join(agents, "progress_trajectory.jsonl")
            tick_path = os.path.join(agents, "tick_state.json")
            with open(traj, "w", encoding="utf-8") as f:
                for i in range(200):
                    mom = -0.5 if i < 120 else 1.5
                    f.write(
                        json.dumps(
                            {
                                "cycle": i,
                                "epi_momentum": mom,
                                "fitness": 400.0 + i * 0.5,
                                "epi": 100.0 + i * 0.05,
                                "level": 30,
                            }
                        )
                        + "\n"
                    )
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump({"terminal_diversity_boost": True}, f)
            report = assess_epi_momentum(tmp, window=200, target_avg=0.5)
            self.assertFalse(report.needs_intervention)
            self.assertTrue(report.growth_gate_met)
            self.assertGreater(report.recent_avg, 0.5)
            boost = assess_plateau_boost(tmp, report)
            self.assertFalse(boost.needs_boost)
            run_cycle_governor(tmp)
            tick = json.loads(open(tick_path, encoding="utf-8").read())
            self.assertNotIn("terminal_diversity_boost", tick)

    def test_promotion_raw_stall_overrides_epi_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            traj = os.path.join(agents, "progress_trajectory.jsonl")
            with open(traj, "w", encoding="utf-8") as f:
                for i in range(200):
                    mom = -0.5 if i < 120 else 1.5
                    f.write(
                        json.dumps(
                            {
                                "cycle": i,
                                "epi_momentum": mom,
                                "fitness": 400.0 + i * 0.5,
                                "epi": 100.0 + i * 0.05,
                                "level": 30,
                            }
                        )
                        + "\n"
                    )
            results = os.path.join(tmp, "multi_cycle_results.jsonl")
            with open(results, "w", encoding="utf-8") as f:
                for i in range(25):
                    f.write(
                        json.dumps(
                            {
                                "cycle": 6000 + i,
                                "final_level": 30,
                                "promotion_raw_fitness": 321.85,
                                "champion_words": 61,
                            }
                        )
                        + "\n"
                    )
            self.assertTrue(promotion_raw_capability_stalled(tmp, min_cycles=20))
            self.assertFalse(should_suppress_terminal_diversity_boost(tmp))

    def test_promotion_raw_improving_keeps_epi_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            traj = os.path.join(agents, "progress_trajectory.jsonl")
            with open(traj, "w", encoding="utf-8") as f:
                for i in range(200):
                    mom = -0.5 if i < 120 else 1.5
                    f.write(
                        json.dumps(
                            {
                                "cycle": i,
                                "epi_momentum": mom,
                                "fitness": 400.0 + i * 0.5,
                                "epi": 100.0 + i * 0.05,
                                "level": 30,
                            }
                        )
                        + "\n"
                    )
            results = os.path.join(tmp, "multi_cycle_results.jsonl")
            with open(results, "w", encoding="utf-8") as f:
                for i in range(25):
                    f.write(
                        json.dumps(
                            {
                                "cycle": 6000 + i,
                                "final_level": 30,
                                "promotion_raw_fitness": 300.0 + i * 0.5,
                                "champion_words": 55,
                            }
                        )
                        + "\n"
                    )
            self.assertFalse(promotion_raw_capability_stalled(tmp, min_cycles=20))
            self.assertTrue(should_suppress_terminal_diversity_boost(tmp))

    def test_terminal_stagnation_not_triggered_below_max_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            state_path = os.path.join(agents, "progress_controller_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "level": 29,
                        "verify_score": 0.684,
                        "raw_fitness": 77.4,
                        "cycles_at_level": 70,
                    },
                    f,
                )
            traj = os.path.join(agents, "progress_trajectory.jsonl")
            with open(traj, "w", encoding="utf-8") as f:
                for i in range(1000):
                    f.write(json.dumps({"cycle": i, "epi_momentum": 0.0, "level": 29}) + "\n")
            report = assess_epi_momentum(tmp, window=1000, target_avg=0.5)
            self.assertTrue(should_suppress_register_repair(tmp, report))


if __name__ == "__main__":
    unittest.main()
