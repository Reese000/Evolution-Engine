"""Dump selected cycle records from multi_cycle_results.jsonl (read-only)."""

from __future__ import annotations

import json
import sys

want = set(int(x) for x in sys.argv[1:]) or {7555, 7556, 7557}
by_cycle: dict[int, dict] = {}
with open("multi_cycle_results.jsonl", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = rec.get("cycle")
        if isinstance(c, int) and c in want:
            by_cycle[c] = rec

for c in sorted(by_cycle):
    r = by_cycle[c]
    print(f"=== cycle {c} ({len(r)} keys) ===")
    for k in sorted(r):
        v = r[k]
        s = json.dumps(v, default=str)
        if len(s) > 300:
            s = s[:300] + "..."
        print(f"  {k}: {s}")
