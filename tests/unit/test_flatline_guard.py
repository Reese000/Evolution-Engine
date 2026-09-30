"""Unit tests for evolution.flatline_guard."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from evolution.flatline_guard import apply_flatline_halt, assess_flatline, is_flatline_halted


def _write_flat_trajectory(agents_dir: str, n: int = 35) -> None:
    path = os.path.join(agents_dir, "progress_trajectory.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            row = {
                "cycle": 800 + i,
                "epi": 48.0 + (0.1 if i % 2 else -0.1),
                "fitness": -25.0,
                "verify_score": 0.8,
                "level": 19,
            }
            f.write(json.dumps(row) + "\n")


class TestFlatlineGuard(unittest.TestCase):
    def test_detects_sustained_flatline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            _write_flat_trajectory(agents, 35)
            report = assess_flatline(tmp, window=40, min_window=30)
            self.assertTrue(report.halted)
            self.assertGreater(len(report.reasons), 0)
            self.assertEqual(report.c3_passes, 0)

    def test_insufficient_history_does_not_halt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            _write_flat_trajectory(agents, 10)
            report = assess_flatline(tmp, window=40, min_window=30)
            self.assertFalse(report.halted)

    def test_tail_plateau_triggers_halt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            path = os.path.join(agents, "progress_trajectory.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(35):
                    # oscillating fitness but stuck verify/level at end
                    fit = 90.0 if i % 2 else 60.0
                    row = {
                        "cycle": 800 + i,
                        "epi": 70.0 + (5 if i % 2 else -5),
                        "fitness": fit,
                        "verify_score": 0.8,
                        "level": 19,
                    }
                    f.write(json.dumps(row) + "\n")
            report = assess_flatline(tmp, window=40, min_window=30)
            self.assertTrue(report.halted)
            self.assertTrue(any("tail plateau" in r for r in report.reasons))

    def test_apply_halt_sets_tick_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            _write_flat_trajectory(agents, 35)
            report = assess_flatline(tmp)
            apply_flatline_halt(tmp, report)
            self.assertTrue(is_flatline_halted(tmp))
            tick_path = os.path.join(agents, "tick_state.json")
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
            self.assertFalse(tick.get("evolution_paused"))
            self.assertTrue(tick.get("flatline_advisory"))
            self.assertTrue(tick["flatline_halted"])

    def test_certified_l30_plateau_does_not_halt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            path = os.path.join(agents, "progress_trajectory.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(35):
                    row = {
                        "cycle": 3700 + i,
                        "epi": 87.5,
                        "fitness": 264.04,
                        "verify_score": 1.0,
                        "level": 30,
                    }
                    f.write(json.dumps(row) + "\n")
            report = assess_flatline(tmp, window=40, min_window=30)
            self.assertFalse(report.halted)

    def test_certified_l30_ignores_persisted_flatline_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agents = os.path.join(tmp, ".agents")
            os.makedirs(agents)
            with open(os.path.join(agents, "flatline_state.json"), "w", encoding="utf-8") as f:
                json.dump({"halted": True, "report": {"reasons": ["user-directed force halt"]}}, f)
            with open(os.path.join(agents, "progress_controller_state.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {"level": 30, "verify_score": 1.0, "raw_fitness": 264.04, "cycles_at_level": 300},
                    f,
                )
            self.assertFalse(is_flatline_halted(tmp))


if __name__ == "__main__":
    unittest.main()
