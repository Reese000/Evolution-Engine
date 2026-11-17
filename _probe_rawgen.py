"""Dump raw generation_phase_timing records for a cycle (read-only)."""

from __future__ import annotations

import json
import sys

want = set(int(x) for x in sys.argv[1:]) or {7594}
sel_gens = set(int(x) for x in sys.argv[2:]) if len(sys.argv) > 2 else None
with open("generation_phase_timing.jsonl", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("cycle") not in want:
            continue
        if sel_gens is not None and r.get("generation") not in sel_gens:
            continue
        print(json.dumps(r, indent=1, default=str))
        print("-" * 60)
