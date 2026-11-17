"""Eq-length 1-bit hamming probe at 28w champion."""
import json
import os

os.environ.setdefault("EE_USE_GPU", "0")

from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
from evolution.terminal_stagnation import (
    curriculum_verify_mean,
    get_scan_floor_gate,
    heldout_l30_verify_mean,
    scan_floor_min_raw_gain,
)

row = json.loads(open("multi_cycle_results.jsonl").readlines()[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
gate = get_scan_floor_gate(champ)
champ_pr = compact_l30_promotion_raw_fitness(champ)
bar = champ_pr + float(scan_floor_min_raw_gain(len(champ)))
print(f"C{row['cycle']} {len(champ)}w prom={champ_pr:.6f} bar={bar:.6f} L10={curriculum_verify_mean(champ,10):.3f}")

beats = []
for pos in range(len(champ) - 1):
    for delta in (0x00001000, 0x00002000, 0x00008000, 0x00100000):
        trial = list(champ)
        trial[pos] = (trial[pos] + delta) & 0xFFFFFFFF
        if tuple(trial) == tuple(champ):
            continue
        if not gate.qualifies(trial):
            continue
        pr = compact_l30_promotion_raw_fitness(trial)
        l10 = curriculum_verify_mean(trial, 10)
        if pr > bar + 1e-6:
            beats.append((pr, l10, pos, delta))
        elif l10 > curriculum_verify_mean(champ, 10) + 0.05:
            print(f"  L10 lift no beat: pos={pos} prom={pr:.4f} L10={l10:.3f}")

beats.sort(reverse=True)
print(f"beats above bar: {len(beats)}")
for pr, l10, pos, delta in beats[:5]:
    print(f"  BEAT pos={pos} delta={hex(delta)} prom={pr:.6f} L10={l10:.3f}")
