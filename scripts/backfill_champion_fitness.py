#!/usr/bin/env python3
"""Re-evaluate champion fitness for cycles with stale logged final_fitness."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_RESULTS = os.path.join(PROJECT_ROOT, "multi_cycle_results.jsonl")


def recompute_fitness(champion_program: list, level: int) -> float:
    os.environ.setdefault("EE_USE_GPU", "0")
    from evolution_engine import EvolutionConfig, EvolutionEngine

    prog = [int(x, 16) if isinstance(x, str) else int(x) for x in champion_program]
    eng = EvolutionEngine(EvolutionConfig(curriculum_level=level))
    eng.champion_level = level
    eng._refresh_task_instance()
    trace = eng._evaluate(prog)
    return float(eng._compute_fitness(prog, trace))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jsonl", default=DEFAULT_RESULTS)
    parser.add_argument("--min-cycle", type=int, default=1)
    parser.add_argument("--verify-min", type=float, default=1.0 - 1e-6)
    args = parser.parse_args()

    results_path = os.path.abspath(args.jsonl)
    if not os.path.isfile(results_path):
        raise SystemExit(f"metrics jsonl not found: {results_path}")

    with open(results_path, encoding="utf-8") as fh:
        lines = fh.readlines()
    updated = 0
    out_lines: list[str] = []

    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
        row = json.loads(line)
        cycle = int(row.get("cycle", 0))
        verify = float(row.get("champion_verify_score", 0) or 0)
        logged = float(row.get("final_fitness", 0) or 0)
        level = int(row.get("final_level", 0))
        prog = row.get("champion_program")

        needs_fix = cycle >= args.min_cycle and verify >= args.verify_min and prog and logged < 200.0
        if needs_fix:
            live = recompute_fitness(prog, level)
            if abs(live - logged) > 1.0:
                print(f"C{cycle}: {logged:.2f} -> {live:.2f} (verify={verify})")
                row["final_fitness"] = live
                row["verified_fitness"] = live
                updated += 1
        out_lines.append(json.dumps(row) + "\n")

    if args.dry_run:
        print(f"dry-run: would update {updated} rows")
        return 0

    if updated:
        backup = results_path + ".bak"
        if not os.path.exists(backup):
            with open(backup, "w", encoding="utf-8") as f:
                f.writelines(lines)
        with open(results_path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
        print(f"Updated {updated} rows (backup at {backup})")
    else:
        print("No rows needed updating")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
