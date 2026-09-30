"""Tests for evolution.loop_integrity."""

from __future__ import annotations

import json
import os

from evolution.loop_integrity import count_result_rows, ensure_metrics_integrity


def test_count_result_rows_unique(tmp_path):
    path = tmp_path / "multi_cycle_results.jsonl"
    rows = [
        {"cycle": 1, "final_fitness": 10.0},
        {"cycle": 2, "final_fitness": 11.0},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    total, unique = count_result_rows(str(path))
    assert total == 2
    assert unique == 2


def test_count_result_rows_duplicates(tmp_path):
    path = tmp_path / "multi_cycle_results.jsonl"
    rows = [
        {"cycle": 1, "final_fitness": 10.0},
        {"cycle": 1, "final_fitness": 5.0},
        {"cycle": 2, "final_fitness": 11.0},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    total, unique = count_result_rows(str(path))
    assert total == 3
    assert unique == 2


def test_ensure_metrics_integrity_dedupes(tmp_path, monkeypatch):
    results = tmp_path / "multi_cycle_results.jsonl"
    rows = [
        {"cycle": 1, "final_fitness": -80.0, "final_level": 19, "champion_verify_score": 0.9},
        {"cycle": 1, "final_fitness": 10.0, "final_level": 19, "champion_verify_score": 0.6},
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_trajectory.jsonl").write_text("", encoding="utf-8")

    def fake_sync():
        return 0

    monkeypatch.setattr("scripts.progress_trajectory.sync_trajectory", fake_sync)

    summary = ensure_metrics_integrity(str(tmp_path), log=lambda _m: None)
    assert summary["deduped"] is True
    assert summary["cycles_kept"] == 1
    kept = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(kept) == 1
    assert kept[0]["final_fitness"] == 10.0
