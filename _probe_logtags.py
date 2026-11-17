"""Group evolution.log lines by generation and summarise message tags.

Usage: python _probe_logtags.py <cycle> [gen0 gen1 ...]
"""
from __future__ import annotations

import collections
import os
import re
import sys

cycle = sys.argv[1] if len(sys.argv) > 1 else "7591"
gens = set(int(x) for x in sys.argv[2:]) or {0, 1, 28, 29}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"cycle_{cycle}", "logs", "evolution.log")

GEN_RE = re.compile(r"^Gen\s+(\d+)\s*\|\s*(.*)$")

by_gen: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
examples: dict[tuple[int, str], list[str]] = collections.defaultdict(list)

with open(path, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.rstrip("\n")
        m = GEN_RE.match(line)
        if not m:
            continue
        g = int(m.group(1))
        rest = m.group(2).strip()
        tag = rest.split("]")[0].lstrip("[") if rest.startswith("[") else rest.split(":")[0][:40]
        if rest.startswith("["):
            tag = rest.split("]")[0][1:]
        else:
            tag = rest[:40]
        by_gen[g][tag] += 1
        if g in gens and len(examples[(g, tag)]) < 2:
            examples[(g, tag)].append(rest[:200])

for g in sorted(by_gen):
    if gens and g not in gens:
        continue
    total = sum(by_gen[g].values())
    print(f"=== Gen {g}: {total} lines ===")
    for tag, n in by_gen[g].most_common(25):
        print(f"   {n:5d}  {tag}")
        for ex in examples.get((g, tag), [])[:1]:
            print(f"          e.g. {ex}")
