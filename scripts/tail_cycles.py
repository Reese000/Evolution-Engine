#!/usr/bin/env python3
"""Print last N valid cycles with C3 verify gate status."""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evolution.progress_controller import ProgressController

RESULTS = PROJECT_ROOT / "multi_cycle_results.jsonl"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

with RESULTS.open(encoding="utf-8") as f:
    tail = deque(f, maxlen=max(n * 3, 30))

rows = []
seen: dict[int, dict] = {}
for line in tail:
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "error" in r:
        continue
    c = int(r.get("cycle", -1))
    prev = seen.get(c)
    if prev is None or (
        int(r.get("final_level", 0)),
        float(r.get("champion_verify_score", 0)),
        float(r.get("final_fitness", -1e9)),
    ) >= (
        int(prev.get("final_level", 0)),
        float(prev.get("champion_verify_score", 0)),
        float(prev.get("final_fitness", -1e9)),
    ):
        seen[c] = r

rows = [seen[c] for c in sorted(seen)]

for r in rows[-n:]:
    lvl = int(r.get("final_level", 0))
    ver = float(r.get("champion_verify_score", 0))
    ok = ProgressController.passes_certify_verify(lvl, ver)
    print(
        f"cycle={r.get('cycle')} L={lvl} verify={ver:.4f} C3={'pass' if ok else 'FAIL'} "
        f"clamp={r.get('start_level_clamped')} start={r.get('start_level')}"
    )

verified = sum(
    1
    for r in rows[-10:]
    if ProgressController.passes_certify_verify(int(r.get("final_level", 0)), float(r.get("champion_verify_score", 0)))
)
print(f"C3 window: {verified}/10")
