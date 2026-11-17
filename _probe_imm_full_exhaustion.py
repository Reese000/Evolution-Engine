"""Comprehensive imm exhaustion: champion + all 8 seeds, ALL subset sizes.

Fixes two weaknesses in the earlier handoff:
  1. Earlier phases each used their OWN dedup set, so summing them
     ("47,945 distinct") double-counted cross-phase overlap. This run keeps ONE
     shared `seen` set across every base and every subset size, so the reported
     total is the TRUE number of unique programs evaluated.
  2. Seeds were only swept at subset sizes 1-2. This runs seeds to FULL
     exhaustion (every subset size of their own imm-effectual diff loci).

Scoring: compact_l30_promotion_raw_fitness only, with the persistent disk
score cache disabled so every unique program gets one real evaluation.
No fitness/gate modification; read-only on the daemon.
"""
import itertools
import json
import os
import random
import time

os.environ.setdefault("EE_USE_GPU", "0")

from evolution import fitness_helpers as fh
from evolution import terminal_stagnation as ts
from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
from evolution.terminal_stagnation import (
    _SCAN_FLOOR_WIDE_IMM_DELTAS,
    get_scan_floor_gate,
    scan_floor_min_raw_gain,
    terminal_imm_perturb_probe_variants,
    terminal_wide_imm_sweep_probe_variants,
)
from milestone1.rv32i_mask import decode as rv32i_decode, encode as rv32i_encode
from mutation_proposer import _is_ebreak

TASK_BAR = 325.836
WIDE = tuple(_SCAN_FLOOR_WIDE_IMM_DELTAS)
RNG_SEED = 20260924
EVIDENCE_PATH = ".agents/evidence/t_6f4be05a_imm_full_exhaustion.json"
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def apply_delta_one(prog, idx, delta):
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


def effectual_loci(base, loci):
    out = []
    for i in loci:
        if _is_ebreak(base[i]):
            continue
        if any((nv := apply_delta_one(base, i, dl)) is not None and nv != base
               for dl in WIDE):
            out.append(i)
    return out


# ---- force one real evaluation per unique program ---------------------------
fh._PROMOTION_RAW_CACHE.clear()
ts._VM_SCORE_DISK_MEM = {}       # in-process only; _vm_score_disk_flush is
ts._VM_SCORE_DISK_LOADED = True  # unreachable here (lives in select_terminal_probes_for_batch)

rows = open("multi_cycle_results.jsonl", encoding="utf-8").readlines()
row = json.loads(rows[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
payload = json.load(open(".agents/scan_floor_eq_refine_seeds.json", encoding="utf-8"))
anchor = [int(x) for x in payload["anchor"]]
seeds = [[int(x) for x in s] for s in payload["seeds"]]
assert anchor == champ, "champion drifted from seeds anchor; aborting"

gate = get_scan_floor_gate(champ)
champ_t = tuple(champ)
champ_pr = float(compact_l30_promotion_raw_fitness(champ))
bar = champ_pr + scan_floor_min_raw_gain(len(champ))
union_loci = sorted({j for s in seeds for j in range(len(champ)) if s[j] != champ[j]})

log(f"cycle={row.get('cycle')} words={len(champ)} anchor_match={anchor == champ}")
log(f"champion FRESH prom={champ_pr:.10f} BAR={bar:.6f} task_bar={TASK_BAR}")
log(f"union diff loci={union_loci}")

bases = [("champion", champ, union_loci)]
for i, s in enumerate(seeds):
    d = [j for j in range(len(champ)) if s[j] != champ[j]]
    bases.append((f"seed{i}", s, d))

seen: set[tuple[int, ...]] = {champ_t}
best = champ_pr
best_desc = "champion(unmodified)"
hits: list[dict] = []
rows_out = []
grand_new = 0
grand_planned = 0
sum_base_new = 0
native_new = 0

for name, base, diff_loci in bases:
    eff = effectual_loci(base, diff_loci)
    t_base = time.time()
    planned = 0
    new_here = 0
    dups = 0
    base_baseline = float(compact_l30_promotion_raw_fitness(list(base)))
    base_best = base_baseline
    for sz in range(1, len(eff) + 1):
        for combo in itertools.combinations(eff, sz):
            for deltas in itertools.product(WIDE, repeat=sz):
                planned += 1
                v = list(base)
                ok = True
                for idx, dl in zip(combo, deltas):
                    nv = apply_delta_one(v, idx, dl)
                    if nv is None:
                        ok = False
                        break
                    v = nv
                if not ok:
                    continue
                key = tuple(int(x) for x in v)
                if key in seen:
                    dups += 1
                    continue
                seen.add(key)
                pr = float(compact_l30_promotion_raw_fitness(list(key)))
                new_here += 1
                if pr > best + 1e-9:
                    best = pr
                    best_desc = f"{name} loci={list(combo)} deltas={list(deltas)}"
                    log(f"  ** NEW BEST {pr:.6f} {best_desc}")
                base_best = max(base_best, pr)
                if pr >= TASK_BAR:
                    hits.append({"prom": pr, "base": name, "loci": list(combo),
                                 "deltas": list(deltas),
                                 "program": [int(x) for x in key]})
                    log(f"  !! >= task_bar {pr:.6f} {name} {list(combo)} {list(deltas)}")

    # project-native bounded probes, folded into the same shared dedup set
    rng = random.Random(RNG_SEED)
    nat_new = 0
    try:
        native = terminal_imm_perturb_probe_variants(list(base), rng, max_evals=24, gate=gate)
    except (ValueError, TypeError, KeyError, ImportError) as exc:
        log(f"  native imm_perturb error {name}: {exc}")
        native = []
    try:
        native += terminal_wide_imm_sweep_probe_variants(
            list(base), max_evals=64, start_offset=0, gate=gate)
    except (ValueError, TypeError, KeyError, ImportError) as exc:
        log(f"  native wide_sweep error {name}: {exc}")
    for v in native:
        key = tuple(int(x) for x in v)
        if key in seen or len(key) != len(champ):
            continue
        seen.add(key)
        pr = float(compact_l30_promotion_raw_fitness(list(key)))
        nat_new += 1
        if pr > best + 1e-9:
            best = pr
            best_desc = f"{name} native"
            log(f"  ** NEW BEST {pr:.6f} {best_desc}")
        base_best = max(base_best, pr)
        if pr >= TASK_BAR:
            hits.append({"prom": pr, "base": name, "loci": None, "deltas": None,
                         "program": [int(x) for x in key]})
            log(f"  !! >= task_bar {pr:.6f} {name} native")

    native_new += nat_new
    grand_planned += planned
    grand_new += new_here + nat_new
    sum_base_new += new_here + nat_new
    rows_out.append({
        "base": name, "effectual_loci": eff, "max_subset_size": len(eff),
        "planned_combos": planned, "globally_unique_new": new_here + nat_new,
        "native_new": nat_new, "cross_base_dups": dups,
        "baseline_prom": base_baseline,
        "best_prom": base_best, "runtime_s": time.time() - t_base,
    })
    log(f"{name}: eff={eff} planned={planned} unique_new={new_here + nat_new} "
        f"(native {nat_new}) dups={dups} best={base_best:.6f} "
        f"in {time.time() - t_base:.1f}s")

overlap = sum_base_new - grand_new
log("=" * 72)
log(f"grand total UNIQUE programs evaluated (shared dedup) = {grand_new}")
log(f"sum of per-base unique counts                       = {sum_base_new}")
log(f"cross-base overlap removed by shared dedup          = {overlap}")
log(f"planned combos (before dedup)                       = {grand_planned}")
log(f"native-probe contributions                          = {native_new}")
log(f"BEST = {best:.10f} ({best_desc})")
log(f"champion = {champ_pr:.10f}  gap to task bar = {TASK_BAR - best:.6f}")
log(f"hits >= {TASK_BAR}: {len(hits)}")

gate_hits = []
for h in hits:
    ok = gate.qualifies(h["program"])
    h["gate"] = ok
    if ok:
        gate_hits.append(h)
    log(f"  hit prom={h['prom']:.6f} gate={ok} base={h['base']}")
log(f"gate-qualified hits: {len(gate_hits)}")

os.makedirs(os.path.dirname(EVIDENCE_PATH), exist_ok=True)
evidence = {
    "task": "t_6f4be05a",
    "phase": "comprehensive imm exhaustion, single shared dedup set",
    "champion_cycle": row.get("cycle"),
    "champion_prom": champ_pr,
    "bar_real": bar,
    "task_bar": TASK_BAR,
    "union_diff_loci": union_loci,
    "delta_set": list(WIDE),
    "per_base": rows_out,
    "grand_unique_programs": grand_new,
    "sum_per_base_unique": sum_base_new,
    "cross_base_overlap": overlap,
    "planned_combos": grand_planned,
    "best_prom": best,
    "best_desc": best_desc,
    "gap_to_task_bar": TASK_BAR - best,
    "hits_ge_task_bar": hits,
    "gate_qualified_hits": len(gate_hits),
    "disk_cache_disabled": True,
    "runtime_s": time.time() - t0,
}
with open(EVIDENCE_PATH, "w", encoding="utf-8") as f:
    json.dump(evidence, f, indent=2)
log(f"evidence -> {EVIDENCE_PATH}")

if gate_hits:
    gate_hits.sort(key=lambda h: h["prom"], reverse=True)
    from evolution.mutation_helpers import record_scan_floor_eq_refine_seeds

    log("RECORDING breakthrough ...")
    record_scan_floor_eq_refine_seeds(champ, [list(gate_hits[0]["program"])],
                                      cycle=int(row.get("cycle") or 0))
    log("recorded.")
else:
    log("EXHAUSTED: no gate-qualified variant >= task bar anywhere in the imm space.")
log(f"done in {time.time() - t0:.1f}s")
