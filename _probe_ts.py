"""Print recent cycle timestamps from trajectory / gen-phase logs (read-only)."""
from __future__ import annotations

import json
import os

os.chdir(r"C:/Users/reese/Projects/Evolution Engine")


def tail_records(path: str, n: int = 8):
    recs = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"--- {path}: {len(recs)} records")
    for r in recs[-n:]:
        keep = {k: v for k, v in r.items()
                if k in ("ts", "timestamp", "cycle", "cycle_number", "elapsed_seconds",
                         "event", "kind", "status", "fitness", "prom_raw")}
        print(json.dumps(keep, default=str)[:400])


tail_records(".agents/progress_trajectory.jsonl", 10)
print()
tail_records("generation_phase_timing.jsonl", 4)
