#!/usr/bin/env python3
"""Tests for ProgressionTracker.emit_snapshot() and is_saturated().

Validates:
  - emit_snapshot produces correct TrendSnapshot from history
  - emit_snapshot handles empty history
  - emit_snapshot respects window parameter
  - emit_snapshot filters None val_loss values
  - is_saturated returns True when all saturation criteria met
  - is_saturated returns False when not enough data
  - is_saturated returns False when only some criteria met
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from progression_tracker import CycleMetrics, ProgressionTracker


def _import_types():
    """Import TrendSnapshot directly to avoid triggering milestone1.training.__init__."""
    types_path = PROJECT_ROOT / "milestone1" / "training" / "scaling" / "types.py"
    spec = importlib.util.spec_from_file_location("milestone1.training.scaling.types", str(types_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["milestone1.training.scaling.types"] = mod
    spec.loader.exec_module(mod)
    return mod


_types_mod = _import_types()
TrendSnapshot = _types_mod.TrendSnapshot


def _make_cycle(
    cycle: int, fitness: float = 50.0, level: int = 10, archive_size: int = 100, val_loss: float = 0.5, words: int = 20
) -> CycleMetrics:
    """Helper to create a CycleMetrics with sensible defaults."""
    return CycleMetrics(
        cycle=cycle,
        level=level,
        fitness=fitness,
        words=words,
        archive_size=archive_size,
        val_loss=val_loss,
        time_s=1.0,
        mg_count=5,
        total_challengers=20,
        champion_improvements=1,
    )


class TestEmitSnapshot(unittest.TestCase):
    """Tests for ProgressionTracker.emit_snapshot()."""

    def test_empty_history(self):
        """emit_snapshot with no history returns empty TrendSnapshot."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        snapshot = tracker.emit_snapshot()
        self.assertIsInstance(snapshot, TrendSnapshot)
        self.assertEqual(snapshot.fitness_history, [])
        self.assertEqual(snapshot.val_loss_history, [])
        self.assertEqual(snapshot.max_level_history, [])
        self.assertEqual(snapshot.archive_size, 0)
        self.assertEqual(snapshot.cycles_observed, 0)

    def test_single_cycle(self):
        """emit_snapshot with one cycle produces correct snapshot."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        tracker.record(_make_cycle(1, fitness=42.0, level=5, archive_size=80, val_loss=0.3))
        snapshot = tracker.emit_snapshot()
        self.assertEqual(snapshot.fitness_history, [42.0])
        self.assertEqual(snapshot.val_loss_history, [0.3])
        self.assertEqual(snapshot.max_level_history, [5])
        self.assertEqual(snapshot.archive_size, 80)
        self.assertEqual(snapshot.cycles_observed, 1)

    def test_window_parameter(self):
        """emit_snapshot respects window parameter."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        for i in range(30):
            tracker.record(_make_cycle(i + 1, fitness=float(i), level=i))
        # Default window=20 should give last 20
        snapshot_default = tracker.emit_snapshot()
        self.assertEqual(len(snapshot_default.fitness_history), 20)
        self.assertEqual(snapshot_default.fitness_history[0], 10.0)
        # Explicit window=5
        snapshot_5 = tracker.emit_snapshot(window=5)
        self.assertEqual(len(snapshot_5.fitness_history), 5)
        self.assertEqual(snapshot_5.fitness_history[0], 25.0)

    def test_val_loss_none_filtered(self):
        """emit_snapshot filters out None val_loss values."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        tracker.record(_make_cycle(1, val_loss=0.5))
        tracker.record(_make_cycle(2, val_loss=None))
        tracker.record(_make_cycle(3, val_loss=0.3))
        snapshot = tracker.emit_snapshot()
        self.assertEqual(snapshot.val_loss_history, [0.5, 0.3])
        self.assertEqual(len(snapshot.max_level_history), 3)

    def test_cycles_observed_is_total(self):
        """cycles_observed reflects total history length, not window."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        for i in range(50):
            tracker.record(_make_cycle(i + 1))
        snapshot = tracker.emit_snapshot(window=10)
        self.assertEqual(snapshot.cycles_observed, 50)
        self.assertEqual(len(snapshot.fitness_history), 10)

    def test_archive_size_from_last_entry(self):
        """archive_size comes from the last history entry."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        tracker.record(_make_cycle(1, archive_size=50))
        tracker.record(_make_cycle(2, archive_size=300))
        snapshot = tracker.emit_snapshot()
        self.assertEqual(snapshot.archive_size, 300)


class TestIsSaturated(unittest.TestCase):
    """Tests for ProgressionTracker.is_saturated()."""

    def test_not_saturated_empty_history(self):
        """is_saturated returns False with no data."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        self.assertFalse(tracker.is_saturated())

    def test_not_saturated_insufficient_data(self):
        """is_saturated returns False when not enough cycles."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        for i in range(5):
            tracker.record(_make_cycle(i + 1))
        self.assertFalse(tracker.is_saturated())

    def test_not_saturated_improving_fitness(self):
        """is_saturated returns False when fitness is still improving."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        for i in range(25):
            # Fitness increases each cycle, so not plateaued
            tracker.record(
                _make_cycle(
                    i + 1,
                    fitness=10.0 + i * 10.0,
                    level=17,
                    archive_size=300,
                    val_loss=0.001 - i * 0.0001,
                )
            )
        # Should NOT be saturated because fitness is improving
        # (only 1 of 4 criteria: curriculum saturated + archive sufficient + val_loss plateau + fitness plateau)
        # Fitness is NOT plateaued (slope is high)
        self.assertFalse(tracker.is_saturated())

    def test_not_saturated_low_archive(self):
        """is_saturated returns False when archive is too small."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        for i in range(25):
            # Flat fitness, level 17, small archive
            tracker.record(
                _make_cycle(
                    i + 1,
                    fitness=50.0,
                    level=17,
                    archive_size=10,  # way below 200
                    val_loss=0.5,
                )
            )
        self.assertFalse(tracker.is_saturated())

    def test_saturated_all_criteria_met(self):
        """is_saturated returns True when all four criteria are met."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        for i in range(25):
            tracker.record(
                _make_cycle(
                    i + 1,
                    fitness=50.0,  # flat fitness → plateau
                    level=17,  # all at level 17 → curriculum saturated
                    archive_size=250,  # > 200 → archive sufficient
                    val_loss=0.5,  # flat val_loss → plateau
                )
            )
        self.assertTrue(tracker.is_saturated())

    def test_is_saturated_returns_bool(self):
        """is_saturated always returns a bool, not a SaturationReport."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = ProgressionTracker(path=f.name)
        result = tracker.is_saturated()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
