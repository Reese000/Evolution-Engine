"""Tests for cycle result ranking and upsert (timeout vs healthy rows)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from scripts.dedupe_cycle_results import _row_rank

from run_multi_cycle_loop import _result_rank, _upsert_cycle_result


class TestCycleResultRank(unittest.TestCase):
    def test_healthy_row_beats_timeout_with_higher_verify(self) -> None:
        healthy = {
            "final_level": 19,
            "champion_verify_score": 0.6,
            "final_fitness": 60.0,
        }
        timeout = {
            "final_level": 19,
            "champion_verify_score": 0.8,
            "final_fitness": -80.0,
        }
        self.assertGreater(_result_rank(healthy), _result_rank(timeout))
        self.assertGreater(_row_rank(healthy), _row_rank(timeout))

    def test_upsert_keeps_healthy_when_timeout_has_higher_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.jsonl")
            _upsert_cycle_result(
                path,
                {
                    "cycle": 42,
                    "final_level": 19,
                    "champion_verify_score": 0.6,
                    "final_fitness": 60.0,
                },
            )
            _upsert_cycle_result(
                path,
                {
                    "cycle": 42,
                    "final_level": 19,
                    "champion_verify_score": 0.8,
                    "final_fitness": -80.0,
                },
            )
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertGreater(rows[0]["final_fitness"], 0)


if __name__ == "__main__":
    unittest.main()
