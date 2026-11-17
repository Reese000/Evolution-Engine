"""Per-generation T= timing from cycle_<n>/logs/evolution.log for recent cycles."""

from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r"T=(\d+(?:\.\d+)?)s")

dirs = sorted(glob.glob(os.path.join(ROOT, "cycle_*")))
want = set()
lo = int(sys.argv[1]) if len(sys.argv) > 1 else 7500
for d in dirs:
    name = os.path.basename(d)
    if not name.startswith("cycle_"):
        continue
    try:
        n = int(name.split("_")[1])
    except ValueError:
        continue
    if n >= lo:
        want.add(n)

for n in sorted(want):
    path = os.path.join(ROOT, f"cycle_{n}", "logs", "evolution.log")
    if not os.path.exists(path):
        print(f"cycle {n}: no evolution.log")
        continue
    ts = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PAT.search(line)
            if m:
                ts.append(float(m.group(1)))
    total = sum(ts)
    if not ts:
        print(f"cycle {n}: no T= markers (size={os.path.getsize(path)})")
        continue
    top = sorted(enumerate(ts), key=lambda kv: -kv[1])[:4]
    tops = ", ".join(f"gen{g}={v:.1f}" for g, v in top)
    print(f"cycle {n}: gens={len(ts)} sum_T={total:8.1f}s  top: {tops}")
