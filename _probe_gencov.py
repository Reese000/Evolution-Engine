"""Summarize cycles present in generation_phase_timing.jsonl (read-only)."""
from __future__ import annotations

import collections
import json

cycles: collections.Counter = collections.Counter()
with open("generation_phase_timing.jsonl", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cycles[r.get("cycle")] += 1

items = sorted((k, v) for k, v in cycles.items() if isinstance(k, int))
print("n cycles:", len(items))
print("first:", items[:6])
print("last:", items[-12:])
