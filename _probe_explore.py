"""Explore: champion source, seed diff loci vs champion (prefix-locked), scorer timing."""

import json
import os
import time

os.environ.setdefault("EE_USE_GPU", "0")

from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
from evolution.terminal_stagnation import (
    curriculum_verify_mean,
    get_scan_floor_gate,
    scan_floor_min_raw_gain,
    terminal_efficiency_at_scan_floor,
)

# --- champion from carry/jsonl (last row), 27w prom 325.8257 ---
rows = open("multi_cycle_results.jsonl", encoding="utf-8").readlines()
row = json.loads(rows[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
print(f"jsonl last row: cycle={row.get('cycle')} words={len(champ)}")

payload = json.load(open(".agents/scan_floor_eq_refine_seeds.json", encoding="utf-8"))
anchor = [int(x) for x in payload["anchor"]]
seeds = [[int(x) for x in s] for s in payload["seeds"]]
print(f"plateau_key={payload['plateau_key']} anchor_words={len(anchor)} n_seeds={len(seeds)}")
print(f"anchor == jsonl champ: {anchor == champ}")
print(f"seed lengths: {sorted({len(s) for s in seeds})}")

# Prefer the jsonl champion when it matches the plateau; else use anchor.
base = champ if len(champ) == 27 else anchor
print(f"using base words={len(base)}")

t0 = time.time()
base_pr = compact_l30_promotion_raw_fitness(base)
t_base = time.time() - t0
gain = scan_floor_min_raw_gain(len(base))
bar = base_pr + gain
at_floor = terminal_efficiency_at_scan_floor(len(base))
print(f"base prom={base_pr:.6f}  min_gain={gain:.6f}  BAR={bar:.6f}  at_scan_floor={at_floor}")

t0 = time.time()
gate = get_scan_floor_gate(base)
t_gate = time.time() - t0
print(f"gate heldout={gate._heldout:.4f} side={tuple(round(x,4) for x in gate._side)} build={t_gate:.2f}s")

# --- diff loci per seed (prefix-locked: first divergence onwards) ---
print("\nseed diff loci vs base:")
all_locs: set[int] = set()
for i, s in enumerate(seeds):
    if len(s) != len(base):
        print(f"  seed{i}: LENGTH MISMATCH {len(s)}")
        continue
    diffs = [j for j in range(len(base)) if s[j] != base[j]]
    prefix = next((j for j in range(len(base)) if s[j] != base[j]), len(base))
    all_locs.update(diffs)
    t = time.time()
    pr = compact_l30_promotion_raw_fitness(s)
    dt = time.time() - t
    l10 = curriculum_verify_mean(s, 10)
    print(f"  seed{i}: diffs={diffs} first_div={prefix} " f"prom={pr:.4f} L10={l10:.3f} score={dt:.2f}s")

print(f"\nunion of diff loci ({len(all_locs)}): {sorted(all_locs)}")
print(f"timings: base_scoring={t_base:.2f}s gate_build={t_gate:.2f}s")
