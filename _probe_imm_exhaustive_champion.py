"""Phase 4: EXHAUSTIVE imm space on the champion at eq-seed diff loci.

Covers every subset (size 1..4) of the imm-effectual diff loci x every
combination of the 14 wide-immediate deltas, with the persistent disk score
cache DISABLED so every evaluation is a real (non-cached) scoring pass.

Complements:
  - Phase 1/2 (_probe_imm_locus_sweep.py): single-site sweep, champion+seeds
  - Phase 3 (_probe_imm_combo_sweep.py): dual/triple combos, all bases
This run is the airtight version for the champion, where the gap is only 0.0103.
"""

import itertools
import json
import os
import time

os.environ.setdefault("EE_USE_GPU", "0")

from evolution import fitness_helpers as fh
from evolution import terminal_stagnation as ts
from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
from evolution.terminal_stagnation import _SCAN_FLOOR_WIDE_IMM_DELTAS, get_scan_floor_gate, scan_floor_min_raw_gain
from milestone1.rv32i_mask import decode as rv32i_decode
from milestone1.rv32i_mask import encode as rv32i_encode
from mutation_proposer import _is_ebreak

TASK_BAR = 325.836
WIDE = tuple(_SCAN_FLOOR_WIDE_IMM_DELTAS)
EVIDENCE_PATH = ".agents/evidence/t_6f4be05a_imm_exhaustive_champion.json"
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def apply_delta_one(prog, idx, delta):
    try:
        d = rv32i_decode(prog[idx], validate=False)
        if "imm" not in d:
            return None
        w = rv32i_encode(
            name=str(d["name"]),
            rd=int(d.get("rd", 0)),
            rs1=int(d.get("rs1", 0)),
            rs2=int(d.get("rs2", 0)),
            imm=int(d.get("imm", 0)) + int(delta),
            validate=True,
        )
    except (ValueError, TypeError, KeyError):
        return None
    v = list(prog)
    v[idx] = int(w)
    return v


# ---- force REAL evals: keep in-process memo, drop the persistent disk cache --
fh._PROMOTION_RAW_CACHE.clear()
ts._VM_SCORE_DISK_MEM = {}  # no disk lookups => no stale daemon-cached values
ts._VM_SCORE_DISK_LOADED = True

rows = open("multi_cycle_results.jsonl", encoding="utf-8").readlines()
row = json.loads(rows[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
payload = json.load(open(".agents/scan_floor_eq_refine_seeds.json", encoding="utf-8"))
anchor = [int(x) for x in payload["anchor"]]
seeds = [[int(x) for x in s] for s in payload["seeds"]]
assert anchor == champ, "champion drifted from seeds anchor; aborting"

gate = get_scan_floor_gate(champ)
champ_t = tuple(champ)
champ_pr = float(compact_l30_promotion_raw_fitness(champ))  # real eval
bar = champ_pr + scan_floor_min_raw_gain(len(champ))
loci = sorted({j for s in seeds for j in range(len(champ)) if s[j] != champ[j]})

effectual = []
for i in loci:
    if _is_ebreak(champ[i]):
        log(f"  locus {i}: EBREAK - skipped")
        continue
    mutable = any((nv := apply_delta_one(champ, i, dl)) is not None and tuple(nv) != champ_t for dl in WIDE)
    if mutable:
        effectual.append(i)
    else:
        d = rv32i_decode(champ[i], validate=False)
        log(f"  locus {i}: imm not encodable into word ({d.get('name')}) - skipped")

log(f"cycle={row.get('cycle')} words={len(champ)} anchor_match={anchor == champ}")
log(f"diff loci={loci}  imm-effectual={effectual}")
log(f"champion FRESH prom={champ_pr:.10f} BAR={bar:.6f} task_bar={TASK_BAR}")

seen: set[tuple[int, ...]] = {champ_t}
best = champ_pr
best_desc = "champion(unmodified)"
hits = []
hist: dict[str, int] = {}
total = 0
sizes_run = []

for sz in range(1, len(effectual) + 1):
    combos = list(itertools.combinations(effectual, sz))
    n_planned = len(combos) * (len(WIDE) ** sz)
    log(f"size {sz}: {len(combos)} locus-subsets x {len(WIDE)}^{sz} = {n_planned} planned")
    t_size = time.time()
    tested = 0
    for combo in combos:
        for deltas in itertools.product(WIDE, repeat=sz):
            v = list(champ)
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
                continue
            seen.add(key)
            pr = float(compact_l30_promotion_raw_fitness(list(key)))
            total += 1
            tested += 1
            hist[f"{pr:.6f}"] = hist.get(f"{pr:.6f}", 0) + 1
            if pr > best + 1e-9:
                best = pr
                best_desc = f"size{sz} loci={list(combo)} deltas={list(deltas)}"
                log(f"  ** NEW BEST {pr:.6f} {best_desc}")
            if pr >= TASK_BAR:
                hits.append({"prom": pr, "loci": list(combo), "deltas": list(deltas), "program": [int(x) for x in key]})
                log(f"  !! >= task_bar {pr:.6f} {list(combo)} {list(deltas)}")
    sizes_run.append({"size": sz, "planned": n_planned, "distinct_tested": tested})
    log(f"  size {sz} done: distinct={tested} in {time.time() - t_size:.1f}s " f"best={best:.6f}")

log("=" * 72)
log(f"TOTAL distinct real evals = {total}")
log(f"BEST = {best:.10f}  ({best_desc})")
log(f"champion = {champ_pr:.10f}")
log(f"gap to task bar = {TASK_BAR - best:.6f}")
log(f"hits >= task bar: {len(hits)}")
log("prom value histogram (top):")
for val, n in sorted(hist.items(), key=lambda kv: -kv[1])[:12]:
    log(f"  {val}: {n}")

gate_hits = []
for h in hits:
    ok = gate.qualifies(h["program"])
    h["gate"] = ok
    if ok:
        gate_hits.append(h)
    log(f"  hit prom={h['prom']:.6f} gate={ok} loci={h['loci']} deltas={h['deltas']}")
log(f"gate-qualified hits: {len(gate_hits)}")

os.makedirs(os.path.dirname(EVIDENCE_PATH), exist_ok=True)
evidence = {
    "task": "t_6f4be05a",
    "phase": "exhaustive champion imm space at eq-seed diff loci (fresh evals)",
    "champion_cycle": row.get("cycle"),
    "champion_prom": champ_pr,
    "bar_real": bar,
    "task_bar": TASK_BAR,
    "diff_loci": loci,
    "imm_effectual_loci": effectual,
    "delta_set": list(WIDE),
    "sizes_run": sizes_run,
    "total_distinct_evals": total,
    "best_prom": best,
    "best_desc": best_desc,
    "gap_to_task_bar": TASK_BAR - best,
    "prom_histogram": hist,
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
    record_scan_floor_eq_refine_seeds(champ, [list(gate_hits[0]["program"])], cycle=int(row.get("cycle") or 0))
    log("recorded.")
else:
    log("EXHAUSTED: no gate-qualified variant >= task bar in the full imm space.")

log(f"done in {time.time() - t0:.1f}s")
