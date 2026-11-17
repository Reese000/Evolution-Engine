"""Time the JSONL-scanning helpers against the real results file (read-only)."""

from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "multi_cycle_results.jsonl")

print("results size:", os.path.getsize(RESULTS))

# 1) raw full parse
t0 = time.perf_counter()
n = 0
with open(RESULTS, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        json.loads(line)
        n += 1
print(f"full json.loads of {n} lines: {time.perf_counter() - t0:.2f}s")

from evolution.champion_rollback import collect_jsonl_archive_parents, find_best_certified_champion  # noqa: E402

t0 = time.perf_counter()
clean = find_best_certified_champion(RESULTS, 30, min_verify=0.99, min_fitness=100.0)
print(f"find_best_certified_champion: {time.perf_counter() - t0:.2f}s -> {bool(clean)}")

if clean:
    champ = [int(x, 16) if isinstance(x, str) else int(x) for x in clean["champion_program"]]
else:
    champ = [0x00008193, 0x002084B3, 0x00100073]

for kwargs in (
    {},
    {"max_parents": 24, "word_tolerance": 12, "rank_by_promotion_raw": True},
):
    t0 = time.perf_counter()
    parents = collect_jsonl_archive_parents(RESULTS, champ, 30, **kwargs)
    print(f"collect_jsonl_archive_parents {kwargs}: {time.perf_counter() - t0:.2f}s -> {len(parents)} parents")
