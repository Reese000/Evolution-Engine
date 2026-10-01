"""Cycle results and trajectory integrity helpers for the evolution loop."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

LogFn = Callable[[str], None]


def count_result_rows(path: str) -> tuple[int, int]:
    """Return (total_rows, unique_cycles) for a multi_cycle_results.jsonl file."""
    if not os.path.isfile(path):
        return 0, 0
    cycles: set[int] = set()
    total = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in row:
                continue
            cycle = int(row.get("cycle", -1))
            if cycle < 0:
                continue
            total += 1
            cycles.add(cycle)
    return total, len(cycles)


def ensure_metrics_integrity(
    project_root: str,
    *,
    force_dedupe: bool = False,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """Dedupe cycle results when needed and sync EPI trajectory from results."""
    _log = log or (lambda _msg: None)
    results = os.path.join(project_root, "multi_cycle_results.jsonl")
    summary: dict[str, Any] = {"deduped": False, "cycles_kept": 0, "trajectory_synced": False}

    total, unique = count_result_rows(results)
    summary["total_rows"] = total
    summary["unique_cycles"] = unique
    duplicates = max(0, total - unique)

    if force_dedupe or duplicates > 0:
        from scripts.dedupe_cycle_results import dedupe

        kept = dedupe(results)
        summary["deduped"] = True
        summary["cycles_kept"] = kept
        summary["duplicates_removed"] = duplicates
        _log(f"Metrics dedupe: {duplicates} duplicate rows removed, {kept} cycles kept")
    else:
        _log(f"Metrics dedupe skipped ({unique} unique cycles, no duplicates)")

    try:
        from evolution.metrics_backfill import backfill_misreported_certified_metrics, count_misreported_rows

        misreported = count_misreported_rows(project_root)
        summary["misreported_rows"] = misreported
        if misreported > 0:
            backfill_summary = backfill_misreported_certified_metrics(project_root)
            summary["certified_backfill"] = backfill_summary
            if backfill_summary.get("updated", 0) > 0:
                _log(f"Certified metrics backfill: {backfill_summary['updated']} rows corrected")
    except Exception as exc:  # noqa: BLE001
        summary["certified_backfill_error"] = repr(exc)
        _log(f"Certified metrics backfill failed: {exc!r}")

    try:
        from scripts.progress_trajectory import sync_trajectory

        sync_trajectory()
        summary["trajectory_synced"] = True
        _log("Trajectory sync complete")
    except Exception as exc:  # noqa: BLE001
        summary["trajectory_error"] = repr(exc)
        _log(f"Trajectory sync failed: {exc!r}")

    return summary
