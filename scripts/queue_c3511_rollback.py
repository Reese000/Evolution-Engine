#!/usr/bin/env python3
"""Queue tick-state rollback to C3511 certified champion."""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "multi_cycle_results.jsonl")
TICK = os.path.join(ROOT, ".agents", "tick_state.json")


def main() -> int:
    prog = None
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if int(row.get("cycle", -1)) == 3511:
                prog = row["champion_program"]
                break
    if not prog:
        raise SystemExit("C3511 not found in multi_cycle_results.jsonl")

    tick: dict = {}
    if os.path.isfile(TICK):
        with open(TICK, encoding="utf-8") as f:
            tick = json.load(f)
    tick["champion_rollback_program"] = list(prog)
    tick["champion_rollback_cycle"] = 3511
    tick["apply_champion_rollback"] = True
    tick["notes"] = (
        "Rollback to C3511 (264 fit, 73w, verify 1.0); "
        "restore bypasses certified pin; low-fit L30 tick flags preserved."
    )
    os.makedirs(os.path.dirname(TICK), exist_ok=True)
    with open(TICK, "w", encoding="utf-8") as f:
        json.dump(tick, f, indent=2)
        f.write("\n")
    print(f"queued C3511 rollback ({len(prog)} words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
