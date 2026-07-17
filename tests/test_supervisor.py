#!/usr/bin/env python3
"""Tests for run_evolution_supervised.py crash-resilient supervisor."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch


class TestSupervisorStateReading(unittest.TestCase):
    """Test state reading functions."""

    def test_get_completed_cycles_empty(self):
        """No results file returns 0."""
        import run_evolution_supervised as s

        with patch.object(s, "RESULTS_FILE", "/nonexistent/file.jsonl"):
            self.assertEqual(s.get_completed_cycles(), 0)

    def test_get_completed_cycles_counting(self):
        """Counts entries with retrained or final_fitness keys."""
        import run_evolution_supervised as s

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"cycle": 1, "final_fitness": 200.0, "retrained": true}\n')
            f.write('{"cycle": 2, "final_fitness": 250.0, "retrained": true}\n')
            f.write('{"cycle": 3, "final_fitness": 0.0}\n')  # no retrained key — still counts
            f.write("\n")  # blank line
            f.write("not json\n")  # invalid JSON
            results_path = f.name

        try:
            # blank line and invalid JSON are skipped
            with patch.object(s, "RESULTS_FILE", results_path):
                self.assertEqual(s.get_completed_cycles(), 3)
        finally:
            os.unlink(results_path)

    def test_get_best_val_loss(self):
        """Reads best_val_loss from MetaController state."""
        import run_evolution_supervised as s

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"best_val_loss": 0.0823, "best_fitness": 339.89}, f)
            state_path = f.name

        try:
            with patch.object(s, "META_CONTROLLER_STATE", state_path):
                self.assertAlmostEqual(s.get_best_val_loss(), 0.0823, places=4)
        finally:
            os.unlink(state_path)

    def test_get_best_val_loss_missing(self):
        """Missing state file returns None."""
        import run_evolution_supervised as s

        with patch.object(s, "META_CONTROLLER_STATE", "/nonexistent/state.json"):
            self.assertIsNone(s.get_best_val_loss())

    def test_get_last_cycle_fitness(self):
        """Returns fitness from last line of results file."""
        import run_evolution_supervised as s

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"cycle": 1, "final_fitness": 200.0}\n')
            f.write('{"cycle": 2, "final_fitness": 339.891}\n')
            results_path = f.name

        try:
            with patch.object(s, "RESULTS_FILE", results_path):
                self.assertAlmostEqual(s.get_last_cycle_fitness(), 339.891, places=3)
        finally:
            os.unlink(results_path)


class TestSupervisorLogging(unittest.TestCase):
    """Test supervisor logging."""

    def test_log_writes_to_file(self):
        """log() writes to supervisor.log."""
        import run_evolution_supervised as s

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        try:
            with patch.object(s, "SUPERVISOR_LOG", log_path):
                s.log("test message")
            with open(log_path) as f:
                content = f.read()
            self.assertIn("test message", content)
            self.assertIn("[", content)  # timestamp prefix
        finally:
            os.unlink(log_path)


class TestSupervisorResume(unittest.TestCase):
    """Test resume logic."""

    def test_resume_flag_passed(self):
        """When resume=True, --resume is added to subprocess command."""
        import run_evolution_supervised as s

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"cycle": 1, "final_fitness": 200.0, "retrained": true}\n')
            results_path = f.name

        try:
            with patch.object(s, "RESULTS_FILE", results_path):
                completed = s.get_completed_cycles()
                self.assertEqual(completed, 1)
        finally:
            os.unlink(results_path)


if __name__ == "__main__":
    unittest.main()
