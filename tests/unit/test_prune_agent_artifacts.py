"""Tests for scripts.prune_agent_artifacts."""

from __future__ import annotations

import json
import os
import re
import sys

from scripts.prune_agent_artifacts import (
    prune_all,
    prune_archive_sidecars,
    prune_batch_logs,
    purge_fixture_archive_sidecars,
)


def _write_sidecars(archive, cycles):
    for cycle in cycles:
        path = archive / f"cycle_{cycle:05d}.json"
        path.write_text(json.dumps({"cycle": cycle}), encoding="utf-8")


def _write_latest(archive, cycle):
    path = archive / f"cycle_{cycle:05d}.json"
    (archive / "latest.json").write_text(
        json.dumps({"cycle": cycle, "path": str(path)}),
        encoding="utf-8",
    )


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
    before = {p.name for p in archive.glob("cycle_*.json")}
    removed = prune_archive_sidecars(str(agents), keep=2)
    after = {p.name for p in archive.glob("cycle_*.json")}
    # The reported count must equal the deletions that actually happened: the
    # fixture purge used to be discarded, so ~13 files vanished at "0 removed".
    assert removed == len(before) - len(after) == 2
    assert "cycle_00012.json" in after
    assert "cycle_09999.json" not in after  # fixture purged above live ref cycle
    assert "cycle_00010.json" not in after


def test_prune_keeps_live_sidecars_above_static_max_cycle_cap(tmp_path):
    """Regression: the live run crossed EE_ARCHIVE_SIDECAR_MAX_CYCLE (7000).

    Observed 2026-09-30: at daemon restart the prune deleted cycle_07551..07563
    (written minutes earlier) while reporting "0 archive sidecars removed",
    leaving only the Sep-14 survivors cycle_06991..07000 and a dangling
    latest.json.
    """
    agents = tmp_path / ".agents"
    archive = agents / "archive_state"
    archive.mkdir(parents=True)
    stale = list(range(6991, 7001))  # at or below the stale static cap
    live = list(range(7595, 7603))  # above the stale static cap
    _write_sidecars(archive, stale + live)
    _write_latest(archive, 7602)
    before = {p.name for p in archive.glob("cycle_*.json")}
    removed = prune_archive_sidecars(str(agents), keep=10)
    after = {p.name for p in archive.glob("cycle_*.json")}
    assert removed == len(before) - len(after) == 8
    # Selection is newest-N by cycle number: every live sidecar survives.
    assert all(f"cycle_{c:05d}.json" in after for c in live)
    assert f"cycle_{7602:05d}.json" in after
    # ... and the eight oldest stale ones are pruned.
    assert all(f"cycle_{c:05d}.json" not in after for c in stale[:8])
    assert "cycle_06999.json" in after
    assert "cycle_07000.json" in after
    # latest.json must not be left dangling by the prune.
    meta = json.loads((archive / "latest.json").read_text(encoding="utf-8"))
    assert os.path.isfile(meta["path"])


def test_dry_run_count_matches_real_prune_deletions(tmp_path, monkeypatch, capsys):
    """A dry run must predict exactly what the real prune deletes."""
    agents = tmp_path / ".agents"
    archive = agents / "archive_state"
    archive.mkdir(parents=True)
    _write_sidecars(archive, list(range(6991, 7001)) + list(range(7595, 7603)))
    _write_latest(archive, 7602)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prune_agent_artifacts.py",
            "--agents",
            str(agents),
            "--archive-keep",
            "10",
            "--dry-run",
        ],
    )
    from scripts.prune_agent_artifacts import main

    assert main() == 0
    out = capsys.readouterr().out
    match = re.search(r"and (\d+) archive sidecars", out)
    assert match is not None, out
    predicted = int(match.group(1))
    # A dry run has no side effects.
    assert len(list(archive.glob("cycle_*.json"))) == 18

    removed = prune_archive_sidecars(str(agents), keep=10)
    assert removed == predicted
    assert removed == 8


def test_prune_keeps_newest_by_cycle_number_not_name_order(tmp_path):
    """Unpadded names sort differently as text than they do as numbers."""
    agents = tmp_path / ".agents"
    archive = agents / "archive_state"
    archive.mkdir(parents=True)
    for cycle in (9999, 10000, 10001):
        path = archive / f"cycle_{cycle}.json"  # deliberately unpadded
        path.write_text(json.dumps({"cycle": cycle}), encoding="utf-8")
    (archive / "latest.json").write_text(
        json.dumps({"cycle": 10001, "path": str(archive / "cycle_10001.json")}),
        encoding="utf-8",
    )
    removed = prune_archive_sidecars(str(agents), keep=2)
    remaining = {p.name for p in archive.glob("cycle_*.json")}
    assert removed == 1
    assert remaining == {"cycle_10000.json", "cycle_10001.json"}


def test_prune_parses_six_digit_cycle_names(tmp_path):
    """cycle_<n>.json is parsed as a whole number, not a fixed 5-char slice."""
    agents = tmp_path / ".agents"
    archive = agents / "archive_state"
    archive.mkdir(parents=True)
    _write_sidecars(archive, (99999, 100000, 100001))
    _write_latest(archive, 100001)
    removed = prune_archive_sidecars(str(agents), keep=2)
    remaining = {p.name for p in archive.glob("cycle_*.json")}
    assert removed == 1
    assert remaining == {"cycle_100000.json", "cycle_100001.json"}


def test_purge_fixture_archive_sidecars_counts_what_it_deletes(tmp_path):
    """Standalone fixture purge: only the far-ahead file, counted exactly."""
    agents = tmp_path / ".agents"
    archive = agents / "archive_state"
    archive.mkdir(parents=True)
    _write_sidecars(archive, (10, 11, 12, 9999))
    _write_latest(archive, 12)
    removed = purge_fixture_archive_sidecars(str(agents))
    remaining = {p.name for p in archive.glob("cycle_*.json")}
    assert removed == 1
    assert remaining == {"cycle_00010.json", "cycle_00011.json", "cycle_00012.json"}


def test_prune_all_reports_both_counts(tmp_path):
    """`prune_all` is what continuous_evolution_daemon._prune_artifacts calls."""
    agents = tmp_path / ".agents"
    archive = agents / "archive_state"
    archive.mkdir(parents=True)
    for i in range(5):
        (agents / f"batch_20260909_{i:06d}.log").write_text("log\n", encoding="utf-8")
    _write_sidecars(archive, (1, 2, 3, 4, 5))
    _write_latest(archive, 5)

    counts = prune_all(str(agents), batch_keep=2, archive_keep=2)

    assert counts == {"batch_logs": 3, "archive_sidecars": 3}
    assert len(list(agents.glob("batch_*.log"))) == 2
    assert {p.name for p in archive.glob("cycle_*.json")} == {
        "cycle_00004.json",
        "cycle_00005.json",
    }
    assert counts["batch_logs"] == 5 - len(list(agents.glob("batch_*.log")))
    assert counts["archive_sidecars"] == 5 - len(list(archive.glob("cycle_*.json")))


def test_prune_never_deletes_the_sidecar_latest_points_at(tmp_path):
    """Even when latest.json's `path` disagrees with its `cycle` number.

    latest.json always points at exactly one sidecar, so deleting that file is
    what leaves it dangling — the fixture purge honours protection too.
    """
    agents = tmp_path / ".agents"
    archive = agents / "archive_state"
    archive.mkdir(parents=True)
    _write_sidecars(archive, (10, 11, 9999))
    flagged = archive / "cycle_09999.json"  # cycle > ref+50, but it is "current"
    (archive / "latest.json").write_text(
        json.dumps({"cycle": 10, "path": str(flagged)}),
        encoding="utf-8",
    )

    removed = prune_archive_sidecars(str(agents), keep=2)

    assert removed == 0
    assert {p.name for p in archive.glob("cycle_*.json")} == {
        "cycle_00010.json",
        "cycle_00011.json",
        "cycle_09999.json",
    }


def test_dry_run_matches_real_prune_when_keep_below_one(tmp_path, monkeypatch, capsys):
    """`--keep 0` must predict 0: both prunes refuse keep < 1."""
    agents = tmp_path / ".agents"
    agents.mkdir()
    for i in range(3):
        (agents / f"batch_20260909_{i:06d}.log").write_text("log\n", encoding="utf-8")
    archive = agents / "archive_state"
    archive.mkdir()
    _write_sidecars(archive, (1, 2, 3))
    _write_latest(archive, 3)

    monkeypatch.setattr(
        sys,
        "argv",
        ["prune_agent_artifacts.py", "--agents", str(agents), "--keep", "0", "--archive-keep", "0", "--dry-run"],
    )
    from scripts.prune_agent_artifacts import main

    assert main() == 0
    out = capsys.readouterr().out
    assert "Would remove 0 batch logs" in out, out
    assert "and 0 archive sidecars" in out, out

    counts = prune_all(str(agents), batch_keep=0, archive_keep=0)
    assert counts == {"batch_logs": 0, "archive_sidecars": 0}
    assert len(list(agents.glob("batch_*.log"))) == 3
    assert len(list(archive.glob("cycle_*.json"))) == 3
