#!/usr/bin/env python3
"""Backfill degraded L30 jsonl rows from best certified champion (e.g. C3511)."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_RESULTS = os.path.join(PROJECT_ROOT, "multi_cycle_results.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=DEFAULT_RESULTS)
    parser.add_argument("--min-cycle", type=int, default=3512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from evolution.metrics_backfill import backfill_misreported_certified_metrics

    results_path = os.path.abspath(args.jsonl)
    summary = backfill_misreported_certified_metrics(
        PROJECT_ROOT,
        min_cycle=args.min_cycle,
        min_level=30,
        dry_run=args.dry_run,
    )
    updated = int(summary.get("updated", 0))
    if summary.get("reason") == "no_certified_champion":
        raise SystemExit("no certified L30 champion found")
    if args.dry_run:
        print(f"dry-run: would update {updated} rows")
        return 0
    if updated:
        try:
            import subprocess

            subprocess.run(
                [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "progress_trajectory.py"), "--rebuild"],
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
                check=False,
            )
            print("trajectory rebuilt after backfill")
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"trajectory rebuild skipped: {exc}")
    print(f"updated {updated} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
