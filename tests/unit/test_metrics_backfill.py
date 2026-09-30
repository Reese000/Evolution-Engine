"""Tests for evolution.metrics_backfill."""

from __future__ import annotations

import json
import os

from evolution.metrics_backfill import (
    backfill_misreported_certified_metrics,
    count_misreported_rows,
    row_needs_certified_backfill,
)


def _certified_row(cycle: int, fitness: float = 264.5) -> dict:
    prog = ["0x00000013", "0x00100073"]
    return {
        "cycle": cycle,
        "final_level": 30,
        "final_fitness": fitness,
        "champion_verify_score": 1.0,
        "champion_verified": True,
        "champion_words": 68,
        "champion_program": prog,
        "verified_fitness": fitness,
    }


def test_row_needs_certified_backfill_detects_low_fitness(tmp_path):
    certified = _certified_row(1, 264.5)
    prog_key = tuple(certified["champion_program"])
    degraded = dict(certified, cycle=2, final_fitness=50.0)
    assert row_needs_certified_backfill(
        degraded,
        certified_prog_key=prog_key,
        certified_fit=264.5,
        certified_verify=1.0,
        min_cycle=0,
    )
    assert not row_needs_certified_backfill(
        certified,
        certified_prog_key=prog_key,
        certified_fit=264.5,
        certified_verify=1.0,
        min_cycle=0,
    )


def test_backfill_corrects_degraded_rows(tmp_path):
    root = tmp_path
    results = root / "multi_cycle_results.jsonl"
    good = _certified_row(1, 264.5)
    bad = dict(good, cycle=2, final_fitness=80.0, verified_fitness=80.0)
    results.write_text(
        json.dumps(good) + "\n" + json.dumps(bad) + "\n",
        encoding="utf-8",
    )
    assert count_misreported_rows(str(root)) == 1
    summary = backfill_misreported_certified_metrics(str(root))
    assert summary["updated"] == 1
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert rows[1]["final_fitness"] == 264.5


def test_backfill_dry_run_leaves_file_unchanged(tmp_path):
    root = tmp_path
    results = root / "multi_cycle_results.jsonl"
    good = _certified_row(1, 264.5)
    bad = dict(good, cycle=2, final_fitness=80.0)
    original = json.dumps(good) + "\n" + json.dumps(bad) + "\n"
    results.write_text(original, encoding="utf-8")
    summary = backfill_misreported_certified_metrics(str(root), dry_run=True)
    assert summary["updated"] == 1
    assert results.read_text(encoding="utf-8") == original


def test_backfill_does_not_overwrite_other_programs(tmp_path):
    root = tmp_path
    results = root / "multi_cycle_results.jsonl"
    good = _certified_row(1, 264.5)
    other = {
        "cycle": 2,
        "final_level": 30,
        "final_fitness": 167.8,
        "champion_verify_score": 0.737,
        "champion_verified": False,
        "champion_words": 24,
        "champion_program": ["0x00100093"] * 23 + ["0x00100073"],
        "verified_fitness": 0.0,
    }
    results.write_text(
        json.dumps(good) + "\n" + json.dumps(other) + "\n",
        encoding="utf-8",
    )
    assert count_misreported_rows(str(root)) == 0
    summary = backfill_misreported_certified_metrics(str(root))
    assert summary.get("updated", 0) == 0
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert rows[1]["champion_words"] == 24
    assert rows[1]["champion_verify_score"] == 0.737
