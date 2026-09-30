"""Regression: daemon must not kill healthy long batches (BATCH_MAX_SEC vs cycle rate)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.continuous_evolution_daemon as daemon  # noqa: E402


def test_compute_batch_max_scales_with_cycle_count():
    assert daemon.compute_batch_max_sec(100, 600, 1800) == 61800
    assert daemon.compute_batch_max_sec(25, 600, 1800) == 16800
    assert daemon.compute_batch_max_sec(1, 600, 1800) == 3600  # floor


def test_should_not_kill_when_jsonl_still_advancing():
    # Old bug: killed at 10800s even though cycles were landing every ~450s.
    assert daemon.should_terminate_batch(12000.0, 200.0, batch_max_sec=10800) is False


def test_should_kill_when_over_budget_and_jsonl_stale():
    assert daemon.should_terminate_batch(12000.0, 1200.0, batch_max_sec=10800) is True


def test_should_not_kill_under_budget_even_if_jsonl_stale():
    assert daemon.should_terminate_batch(5000.0, 2000.0, batch_max_sec=10800) is False


def test_should_kill_when_jsonl_missing_and_over_budget():
    assert daemon.should_terminate_batch(12000.0, None, batch_max_sec=10800) is True
