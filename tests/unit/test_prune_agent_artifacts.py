"""Tests for scripts.prune_agent_artifacts."""

from __future__ import annotations

import json

from scripts.prune_agent_artifacts import prune_archive_sidecars, prune_batch_logs


def test_prune_batch_logs_keeps_newest(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    for i in range(5):
        path = agents / f"batch_20260909_{i:06d}.log"
        path.write_text(f"log {i}\n", encoding="utf-8")
    removed = prune_batch_logs(str(agents), keep=2)
    assert removed == 3
    remaining = sorted(agents.glob("batch_*.log"))
    assert len(remaining) == 2
    assert remaining[-1].name == "batch_20260909_000004.log"


def test_prune_archive_sidecars_keeps_newest_and_latest(tmp_path):
    agents = tmp_path / ".agents"
    archive = agents / "archive_state"
    archive.mkdir(parents=True)
    for cycle in range(1, 6):
        path = archive / f"cycle_{cycle:05d}.json"
        path.write_text(json.dumps({"cycle": cycle}), encoding="utf-8")
    latest_path = archive / "cycle_00001.json"
    (archive / "latest.json").write_text(
        json.dumps({"cycle": 1, "path": str(latest_path)}),
        encoding="utf-8",
    )
    removed = prune_archive_sidecars(str(agents), keep=2)
    assert removed == 2
    remaining = sorted(archive.glob("cycle_*.json"))
    assert len(remaining) == 3
    assert (archive / "latest.json").is_file()
    assert (archive / "cycle_00001.json").is_file()


def test_prune_archive_sidecars_ignores_fixture_high_cycle_numbers(tmp_path):
    agents = tmp_path / ".agents"
    archive = agents / "archive_state"
    archive.mkdir(parents=True)
    for cycle in (10, 11, 12, 9999):
        path = archive / f"cycle_{cycle:05d}.json"
        path.write_text(json.dumps({"cycle": cycle}), encoding="utf-8")
    latest_path = archive / "cycle_00012.json"
    (archive / "latest.json").write_text(
        json.dumps({"cycle": 12, "path": str(latest_path)}),
        encoding="utf-8",
    )
    removed = prune_archive_sidecars(str(agents), keep=2)
    assert removed == 1
    remaining = {p.name for p in archive.glob("cycle_*.json")}
    assert "cycle_00012.json" in remaining
    assert "cycle_09999.json" not in remaining  # fixture purged above live ref cycle
    assert "cycle_00010.json" not in remaining
