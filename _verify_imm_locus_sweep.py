"""Verify the imm-locus sweep result: decode coverage + fresh (cache-bypassed) rescoring.

Answers:
  1. Which eq-seed diff loci (18..25) are imm-encodable on the champion?
  2. Full imm landscape at those loci (all 14 wide deltas), scored FRESH.
  3. Is the disk/in-memory score cache consistent with a fresh evaluation?
  4. Does anything reach/promote past the 325.836 task bar?
"""
import json
import os
import time

os.environ.setdefault("EE_USE_GPU", "0")

from evolution import fitness_helpers as fh
from evolution import terminal_stagnation as ts
from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
from evolution.terminal_stagnation import _SCAN_FLOOR_WIDE_IMM_DELTAS
from milestone1.rv32i_mask import decode as rv32i_decode, encode as rv32i_encode

TASK_BAR = 325.836
FINE = (-8, -4, -2, -1, 1, 2, 4, 8)
WIDE = tuple(_SCAN_FLOOR_WIDE_IMM_DELTAS)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


def fresh_score(prog):
    """Bypass in-memory + disk score caches, force a real evaluation."""
    fh._PROMOTION_RAW_CACHE.clear()
    ts._vm_score_disk_reset()
    return float(compact_l30_promotion_raw_fitness(list(prog)))


def apply_delta(prog, idx, delta):
    try:
        d = rv32i_decode(prog[idx], validate=False)
        if "imm" not in d:
            return None
        w = rv32i_encode(
            name=str(d["name"]), rd=int(d.get("rd", 0)), rs1=int(d.get("rs1", 0)),
            rs2=int(d.get("rs2", 0)), imm=int(d.get("imm", 0)) + int(delta),
            validate=True,
        )
    except (ValueError, TypeError, KeyError):
        return None
    v = list(prog)
    v[idx] = int(w)
    return v


rows = open("multi_cycle_results.jsonl", encoding="utf-8").readlines()
row = json.loads(rows[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
payload = json.load(open(".agents/scan_floor_eq_refine_seeds.json", encoding="utf-8"))
seeds = [[int(x) for x in s] for s in payload["seeds"]]
anchor = [int(x) for x in payload["anchor"]]
log(f"cycle={row.get('cycle')} words={len(champ)} anchor_match={anchor == champ}")

# daemon's own recorded value for cross-check
log(f"jsonl promotion_raw_fitness={row.get('promotion_raw_fitness')}")

champ_fresh = fresh_score(champ)
log(f"champion FRESH prom={champ_fresh:.10f}  (task bar {TASK_BAR})")

# ---- 1. decode coverage at the eq-seed diff loci ---------------------------
loci = sorted({j for s in seeds for j in range(len(champ)) if s[j] != champ[j]})
log(f"eq-seed diff loci: {loci}")
imm_loci = []
log("locus | champion word     | insn            | imm   | has_imm")
for i in loci:
    try:
        d = rv32i_decode(champ[i], validate=False)
        name = str(d.get("name", "?"))
        has = "imm" in d
        imm = d.get("imm", "-")
    except (ValueError, TypeError, KeyError):
        name, has, imm = "<decode-error>", False, "-"
    if has:
        imm_loci.append(i)
    log(f"  {i:3d} | 0x{champ[i]:08X} | {name:<15} | {str(imm):>5} | {has}")
log(f"imm-encodable loci: {imm_loci}  ({len(imm_loci)}/{len(loci)})")

# ---- 2. full fresh imm landscape on the champion ---------------------------
log(f"sweeping champion x {len(imm_loci)} imm loci x {len(WIDE)} deltas (FRESH)")
best = champ_fresh
best_desc = "champion"
ties = 0
above_champ = []
tested = 0
per_locus = {}
for i in imm_loci:
    vals = {}
    for axis, deltas in (("fine", FINE), ("wide", WIDE)):
        for dl in deltas:
            v = apply_delta(champ, i, dl)
            if v is None:
                continue
            tested += 1
            pr = fresh_score(v)
            vals[(axis, dl)] = pr
            if pr > best + 1e-9:
                best, best_desc = pr, f"locus={i} delta={dl} axis={axis}"
                above_champ.append((pr, i, dl, axis))
            elif abs(pr - champ_fresh) <= 1e-9:
                ties += 1
    per_locus[i] = vals

log("=" * 72)
log(f"distinct variants tested (champion locus sweep, fresh): {tested}")
log(f"variants tying champion exactly: {ties}")
log(f"variants strictly ABOVE champion: {len(above_champ)}")
log(f"best found = {best:.10f} via {best_desc}")
log(f"champion   = {champ_fresh:.10f}")
log(f"variants >= {TASK_BAR}: {sum(1 for v in [best] if v >= TASK_BAR)}")
log(f"gap to task bar = {TASK_BAR - best:.6f}")

log("per-locus distinct prom values (fresh):")
for i in imm_loci:
    uniq = sorted(set(round(v, 6) for v in per_locus[i].values()))
    log(f"  locus {i}: n={len(per_locus[i])} distinct_prom={uniq}")

# ---- 3. cache consistency ---------------------------------------------------
log("cache consistency (cached vs fresh):")
samples = [champ]
for i in imm_loci[:2]:
    v = apply_delta(champ, i, 4)
    if v:
        samples.append(v)
for s in (seeds[0], seeds[5]):
    samples.append(s)
for k, p in enumerate(samples):
    fh._PROMOTION_RAW_CACHE.clear()
    ts._vm_score_disk_reset()
    cached = float(compact_l30_promotion_raw_fitness(list(p)))  # warms caches
    again = float(compact_l30_promotion_raw_fitness(list(p)))  # pure cache hit
    fresh = fresh_score(p)
    ok = abs(cached - fresh) <= 1e-9 and abs(again - fresh) <= 1e-9
    log(f"  sample{k}: cached={cached:.10f} repeat={again:.10f} fresh={fresh:.10f} "
        f"{'OK' if ok else 'MISMATCH'}")

# ---- 4. seeds baseline (fresh) ---------------------------------------------
log("seed baselines (fresh):")
for i, s in enumerate(seeds):
    pr = fresh_score(s)
    log(f"  seed{i}: prom={pr:.6f}  delta_vs_champ={pr - champ_fresh:+.6f}")

out = {
    "champion_cycle": row.get("cycle"),
    "champion_fresh_prom": champ_fresh,
    "daemon_jsonl_prom": row.get("promotion_raw_fitness"),
    "task_bar": TASK_BAR,
    "diff_loci": loci,
    "imm_encodable_loci": imm_loci,
    "variants_tested_fresh": tested,
    "ties_champion": ties,
    "above_champion": len(above_champ),
    "best_prom": best,
    "best_desc": best_desc,
    "gap_to_task_bar": TASK_BAR - best,
    "per_locus_distinct_prom": {
        str(i): sorted(set(round(v, 6) for v in per_locus[i].values()))
        for i in imm_loci
    },
}
path = ".agents/evidence/t_6f4be05a_imm_locus_verify.json"
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
log(f"evidence -> {path}")
log("done")
