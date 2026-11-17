"""Print per-generation phase breakdown from generation_phase_timing.jsonl.

Usage: python _probe_gen_phases.py [min_cycle]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

PATH = "generation_phase_timing.jsonl"
min_cycle = int(sys.argv[1]) if len(sys.argv) > 1 else 0

by_cycle: dict[int, list[dict]] = defaultdict(list)
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
        if isinstance(c, int) and c >= min_cycle:
            by_cycle[c].append(rec)

for c in sorted(by_cycle):
    rows = sorted(by_cycle[c], key=lambda r: r.get("generation", 0))
    if not rows:
        continue
    print(f"=== cycle {c}: {len(rows)} generations ===")
    tot = defaultdict(float)
    for r in rows:
        ps = r.get("phase_seconds", {})
        gen = r.get("generation", -1)
        secs = r.get("generation_seconds", 0.0)
        mut = ps.get("mutation_propagation", 0.0)
        vm = ps.get("vm_eval", 0.0)
        sel = ps.get("selection_archives", 0.0)
        diag = ps.get("diagnostics_logging", 0.0)
        tot["secs"] += secs
        tot["mut"] += mut
        tot["vm"] += vm
        tot["sel"] += sel
        tot["diag"] += diag
        subs = r.get("subphase_seconds") or {}
        subtxt = " ".join(f"{k}={v:.1f}" for k, v in sorted(subs.items(), key=lambda kv: -kv[1]) if v >= 0.5)
        print(
            f"  gen {gen:3d} total={secs:7.1f} mut={mut:7.1f} vm={vm:6.1f} "
            f"sel={sel:5.1f} diag={diag:5.1f} stalled={r.get('generations_stalled')} {subtxt}"
        )
    print(
        f"  TOTAL secs={tot['secs']:.1f} mut={tot['mut']:.1f} vm={tot['vm']:.1f} "
        f"sel={tot['sel']:.1f} diag={tot['diag']:.1f}"
    )
