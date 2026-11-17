"""Probe 40w replace windows with L10 bounded-sum loop."""

import json
import os

os.environ.setdefault("EE_USE_GPU", "0")

from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
from evolution.terminal_stagnation import (
    _l30_bounded_sum_loop_seq,
    _l30_bounded_sum_max_loop_seq,
    curriculum_verify_mean,
    get_scan_floor_gate,
    l30_strip_dead_l30_stores,
    scan_floor_min_raw_gain,
    semantic_compress,
)

row = json.loads(open("multi_cycle_results.jsonl").readlines()[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
gate = get_scan_floor_gate(champ)
champ_pr = compact_l30_promotion_raw_fitness(champ)
bar = champ_pr + float(scan_floor_min_raw_gain(len(champ)))
print(f"C{row['cycle']} words={len(champ)} bar={bar:.6f} L10={curriculum_verify_mean(champ,10):.3f}")

bases = [list(champ)]
stripped = l30_strip_dead_l30_stores(champ)
if stripped:
    bases.append(stripped)
    print(f"strip dead -> {len(stripped)}w")
comp = semantic_compress(list(champ))
if comp and len(comp) != len(champ):
    bases.append(list(comp))
    print(f"semantic_compress -> {len(comp)}w")

loops = {
    "sum": _l30_bounded_sum_loop_seq(),
    "max": _l30_bounded_sum_max_loop_seq(),
}
beats = []
for base in bases:
    n = len(base)
    for name, seq in loops.items():
        if len(seq) > n:
            continue
        for start in range(0, n - len(seq) + 1):
            if start + len(seq) >= n and base[-1] == seq[-1]:
                continue
            trial = list(base[:start]) + list(seq) + list(base[start + len(seq) :])
            if len(trial) != n or tuple(trial) == tuple(champ):
                continue
            if not gate.qualifies(trial):
                continue
            pr = compact_l30_promotion_raw_fitness(trial)
            l10 = curriculum_verify_mean(trial, 10)
            if pr > bar + 1e-6:
                beats.append((pr, l10, name, start, trial))
            elif l10 > curriculum_verify_mean(champ, 10) + 0.05:
                print(f"  L10 lift no beat: {name}@{start} prom={pr:.4f} L10={l10:.3f}")

beats.sort(reverse=True)
print(f"beats above bar: {len(beats)}")
for pr, l10, name, start, _ in beats[:5]:
    print(f"  BEAT {name}@{start} prom={pr:.6f} L10={l10:.3f}")
