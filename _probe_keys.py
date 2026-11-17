"""Inspect keys + timestamps of recent multi_cycle_results records."""
from __future__ import annotations

import json
import sys

PATH = "multi_cycle_results.jsonl"
lo = int(sys.argv[1]) if len(sys.argv) > 1 else 7500

rows = []
with open(PATH, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = rec.get("cycle")
        if isinstance(c, int) and c >= lo:
            rows.append(rec)

by = {}
for r in rows:
    by[r["cycle"]] = r

if by:
    first = next(iter(by.values()))
    print("KEYS:", sorted(first.keys()))
    print()

for c in sorted(by):
    r = by[c]
    ts = {k: v for k, v in r.items() if "time" in k.lower() or "ts" == k or k.endswith("_at")}
    print(
        c,
        "elapsed=", r.get("elapsed_seconds"),
        "gens=", r.get("generations_completed"),
        "| ts:",
        json.dumps(ts, default=str)[:400],
    )

print()
print("--- gen phase file first/last cycles ---")
seen = []
with open("generation_phase_timing.jsonl", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen.append((rec.get("cycle"), rec.get("ts"), rec.get("generation")))
print("first:", seen[:3])
print("last:", seen[-3:])
print("distinct cycles:", sorted({s[0] for s in seen}, key=lambda x: (x is None, x)))
