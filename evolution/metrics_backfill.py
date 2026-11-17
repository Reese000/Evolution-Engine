"""Auto-backfill misreported certified champion metrics in cycle results."""

from __future__ import annotations

import json
import os
from typing import Any


def _results_path(project_root: str) -> str:
    return os.path.join(project_root, "multi_cycle_results.jsonl")


def row_needs_certified_backfill(
    row: dict[str, Any],
    *,
    certified_prog_key: tuple[Any, ...],
    certified_fit: float,
    certified_verify: float,
    min_cycle: int = 0,
    min_level: int = 30,
) -> bool:
    """True when a cycle row under-reports a known certified champion."""
    cycle = int(row.get("cycle", 0))
    level = int(row.get("final_level", 0))
    if cycle < min_cycle or level < min_level:
        return False
    verify = float(row.get("champion_verify_score", 0.0))
    fit = float(row.get("final_fitness", 0.0))
    row_prog = row.get("champion_program")
    same_certified_prog = isinstance(row_prog, list) and certified_prog_key and tuple(row_prog) == certified_prog_key
    misreported_fit = same_certified_prog and fit < certified_fit - 1.0
    misreported_verify = same_certified_prog and verify < certified_verify - 1e-6
    return misreported_verify or misreported_fit


def count_misreported_rows(
    project_root: str,
    *,
    min_cycle: int = 0,
    min_level: int = 30,
) -> int:
    """Count jsonl rows that would be corrected by certified backfill."""
    results = _results_path(project_root)
    if not os.path.isfile(results):
        return 0
    from evolution.champion_rollback import find_best_certified_champion

    clean = find_best_certified_champion(results, min_level, min_verify=0.99, min_fitness=100.0)
    if not clean:
        return 0
    certified_prog_key = tuple(clean.get("champion_program") or [])
    certified_fit = float(clean.get("final_fitness", 0.0))
    certified_verify = float(clean.get("champion_verify_score", 0.0))
    count = 0
    with open(results, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row_needs_certified_backfill(
                row,
                certified_prog_key=certified_prog_key,
                certified_fit=certified_fit,
                certified_verify=certified_verify,
                min_cycle=min_cycle,
                min_level=min_level,
            ):
                count += 1
    return count


def backfill_misreported_certified_metrics(
    project_root: str,
    *,
    min_cycle: int = 0,
    min_level: int = 30,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Restore degraded L30+ rows from the best certified champion snapshot."""
    summary: dict[str, Any] = {
        "updated": 0,
        "dry_run": dry_run,
        "min_cycle": min_cycle,
        "min_level": min_level,
    }
    results = _results_path(project_root)
    if not os.path.isfile(results):
        summary["reason"] = "missing_results"
        return summary

    from evolution.champion_rollback import find_best_certified_champion

    clean = find_best_certified_champion(results, min_level, min_verify=0.99, min_fitness=100.0)
    if not clean:
        summary["reason"] = "no_certified_champion"
        return summary

    logged_verify = float(clean.get("champion_verify_score", 0.0))
    logged_fit = float(clean.get("final_fitness", 0.0))
    prog = clean.get("champion_program")
    words = int(clean.get("champion_words", len(prog or [])))
    prog_key = tuple(prog or [])

    lines = open(results, encoding="utf-8").readlines()
    updated = 0
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        row = json.loads(line)
        if row_needs_certified_backfill(
            row,
            certified_prog_key=prog_key,
            certified_fit=logged_fit,
            certified_verify=logged_verify,
            min_cycle=min_cycle,
            min_level=min_level,
        ):
            row["champion_program"] = list(prog)
            row["champion_words"] = words
            row["champion_verify_score"] = logged_verify
            row["champion_verified"] = logged_verify >= 0.99 - 1e-6
            row["final_fitness"] = logged_fit
            row["verified_fitness"] = logged_fit if row["champion_verified"] else 0.0
            updated += 1
        out.append(json.dumps(row) + "\n")

    summary["updated"] = updated
    if updated == 0:
        summary["reason"] = "nothing_to_fix"
        return summary

    if dry_run:
        summary["reason"] = "dry_run"
        return summary

    backup = results + ".certified_bak"
    if not os.path.exists(backup):
        with open(backup, "w", encoding="utf-8") as f:
            f.writelines(lines)
    with open(results, "w", encoding="utf-8") as f:
        f.writelines(out)
    summary["reason"] = "backfilled"
    return summary
