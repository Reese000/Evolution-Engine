#!/usr/bin/env python3
"""Deduplicate multi_cycle_results.jsonl — one row per cycle (best quality wins)."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "multi_cycle_results.jsonl")
BACKUP = RESULTS + ".bak"


def _row_rank(row: dict[str, Any]) -> tuple:
    fit = float(row.get("final_fitness", -1e9))
    healthy = fit > -25.0
    return (
        1 if healthy else 0,
        int(row.get("final_level", 0)),
        float(row.get("champion_verify_score", 0)),
        fit,
    )


def dedupe(path: str = RESULTS) -> int:
    if not os.path.isfile(path):
        print("No results file")
        return 0
    best: dict[int, dict[str, Any]] = {}
    order: list[int] = []
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
            prev = best.get(cycle)
            if prev is None or _row_rank(row) > _row_rank(prev):
                if prev is None:
                    order.append(cycle)
                best[cycle] = row
    if os.path.isfile(path):
        os.replace(path, BACKUP)
    with open(path, "w", encoding="utf-8") as f:
        for cycle in sorted(order):
            f.write(json.dumps(best[cycle]) + "\n")
    with open(BACKUP, encoding="utf-8", errors="replace") as backup_fh:
        removed = sum(1 for _ in backup_fh) - len(best)
    print(f"Deduped: {len(best)} cycles kept, backup at {BACKUP}")
    return len(best)


if __name__ == "__main__":
    raise SystemExit(0 if dedupe() else 1)
