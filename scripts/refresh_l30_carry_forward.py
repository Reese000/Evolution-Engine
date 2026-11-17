#!/usr/bin/env python3
"""Overwrite L30 carry-forward store with C3511 certified champion."""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "multi_cycle_results.jsonl")
STORE = os.path.join(ROOT, ".agents", "champion_carry_forward.json")


def main() -> int:
    row = None
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            parsed = json.loads(line)
            if int(parsed.get("cycle", -1)) == 3511:
                row = parsed
                break
    if not row:
        raise SystemExit("C3511 not found")

    store: dict = {"levels": {}}
    if os.path.isfile(STORE):
        with open(STORE, encoding="utf-8") as f:
            store = json.load(f)
    levels = store.get("levels", {})
    levels["30"] = {
        "level": 30,
        "verify_score": float(row["champion_verify_score"]),
        "fitness": float(row["final_fitness"]),
        "words": int(row["champion_words"]),
        "cycle": 3511,
        "champion_program": list(row["champion_program"]),
    }
    store["levels"] = levels
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    with open(STORE, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
        f.write("\n")
    print("updated L30 carry_forward from C3511")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
