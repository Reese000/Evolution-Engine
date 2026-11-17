"""Phase 6: program-wide imm sweep + gate-qualification of top variants.

Phase 5 exhausted the imm axis ONLY at the eq-seed diff loci (18..25),
excluding positions 0..17 as 'prefix-locked'. That supports the report's
recommendation ("move to a different axis") only if no imm mutation elsewhere
in the program can beat the champion either. This run closes that gap:

  A. exhaustive imm sweep over EVERY position 0..26, subset sizes 1 and 2,
     all 14 wide deltas, one shared dedup set, disk score cache disabled.
  B. gate-qualification of the top-N variants by promotion_raw -- a metric the
     earlier phases never measured (they only gate-checked >= task bar, of
     which there were none).

Read-only: no fitness/gate modification, no seeds-file write unless a
gate-qualified variant reaches the task bar.
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
MAX_SUBSET = int(os.environ.get("EE_PROBE_P6_MAX_SUBSET", "2"))
GATE_TOP_N = int(os.environ.get("EE_PROBE_P6_GATE_TOP", "200"))
EVIDENCE_PATH = ".agents/evidence/t_6f4be05a_imm_programwide.json"
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


# ---- one real evaluation per unique program --------------------------------
fh._PROMOTION_RAW_CACHE.clear()
ts._VM_SCORE_DISK_MEM = {}  # in-process only; flush unreachable here
ts._VM_SCORE_DISK_LOADED = True

rows = open("multi_cycle_results.jsonl", encoding="utf-8").readlines()
row = json.loads(rows[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
payload = json.load(open(".agents/scan_floor_eq_refine_seeds.json", encoding="utf-8"))
anchor = [int(x) for x in payload["anchor"]]
assert anchor == champ, "champion drifted from seeds anchor; aborting"

gate = get_scan_floor_gate(champ)
champ_t = tuple(champ)
champ_pr = float(compact_l30_promotion_raw_fitness(champ))
bar = champ_pr + scan_floor_min_raw_gain(len(champ))
diff_loci = sorted({j for s in payload["seeds"] for j in range(len(champ)) if int(s[j]) != champ[j]})

log(f"cycle={row.get('cycle')} words={len(champ)} anchor_match=True")
log(f"champion FRESH prom={champ_pr:.10f} BAR={bar:.6f} task_bar={TASK_BAR}")
log(f"diff loci={diff_loci}")

# effectual positions across the WHOLE program
effectual = []
non_effectual = []
for i in range(len(champ)):
    if _is_ebreak(champ[i]):
        non_effectual.append((i, "EBREAK"))
        continue
    name = ""
    try:
        name = str(rv32i_decode(champ[i], validate=False).get("name", "?"))
    except (ValueError, TypeError, KeyError, AttributeError):
        name = "?"
    if any((nv := apply_delta_one(champ, i, dl)) is not None and nv != champ for dl in WIDE):
        effectual.append(i)
    else:
        non_effectual.append((i, name))
log(f"effectual positions program-wide ({len(effectual)}): {effectual}")
log(f"non-effectual ({len(non_effectual)}): {non_effectual}")

seen: set[tuple[int, ...]] = {champ_t}
all_rows: list[dict] = []
best = champ_pr
best_desc = "champion(unmodified)"
hits = []
planned_total = 0

for sz in range(1, MAX_SUBSET + 1):
    combos = list(itertools.combinations(effectual, sz))
    planned = len(combos) * (len(WIDE) ** sz)
    planned_total += planned
    t_sz = time.time()
    new = 0
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
            new += 1
            rowr = {
                "prom": pr,
                "subset_size": sz,
                "loci": list(combo),
                "deltas": list(deltas),
                "program": [int(x) for x in key],
            }
            all_rows.append(rowr)
            if pr > best + 1e-9:
                best = pr
                best_desc = f"size{sz} loci={list(combo)} deltas={list(deltas)}"
                log(f"  ** NEW BEST {pr:.6f} {best_desc}")
            if pr >= TASK_BAR:
                hits.append(rowr)
                log(f"  !! >= task_bar {pr:.6f} {list(combo)} {list(deltas)}")
    log(f"size {sz}: planned={planned} unique={new} in {time.time() - t_sz:.1f}s")

log("=" * 72)
log(f"unique programs evaluated = {len(all_rows)} (planned {planned_total})")
log(f"BEST = {best:.10f} ({best_desc})")
log(f"gap to task bar = {TASK_BAR - best:.6f}")
log(f"hits >= {TASK_BAR}: {len(hits)}")

# ---- B. gate-qualification of the top variants ----------------------------
all_rows.sort(key=lambda r: r["prom"], reverse=True)
top = all_rows[:GATE_TOP_N]
log(f"gate-checking top {len(top)} by prom ...")
t_g = time.time()
gate_true = 0
max_gate_prom = None
max_gate_row = None
for r in top:
    ok = gate.qualifies(r["program"])
    r["gate"] = ok
    if ok:
        gate_true += 1
        if max_gate_prom is None or r["prom"] > max_gate_prom:
            max_gate_prom = r["prom"]
            max_gate_row = r
log(f"gate-checked {len(top)} in {time.time() - t_g:.1f}s; " f"gate-qualified={gate_true}")
if max_gate_row is not None:
    log(
        f"MAX gate-qualified prom = {max_gate_prom:.10f} "
        f"(loci={max_gate_row['loci']} deltas={max_gate_row['deltas']} "
        f"size={max_gate_row['subset_size']})"
    )
    log(f"max gate-qualified vs champion = {max_gate_prom - champ_pr:+.6f}")
    log(f"max gate-qualified vs task bar = {TASK_BAR - max_gate_prom:.6f} short")
else:
    log("no gate-qualified variant among the top scored")

# champion itself for reference
champ_gate = gate.qualifies(list(champ_t))
log(f"champion gate-qualifies: {champ_gate}")

gate_hits = [h for h in hits if h.get("gate")]
for h in hits:
    if "gate" not in h:
        h["gate"] = gate.qualifies(h["program"])
    if h["gate"] and h not in gate_hits:
        gate_hits.append(h)
log(f"hits >= task bar that also gate-qualify: {len(gate_hits)}")

os.makedirs(os.path.dirname(EVIDENCE_PATH), exist_ok=True)
evidence = {
    "task": "t_6f4be05a",
    "phase": "program-wide imm sweep (all 27 positions) + gate analysis",
    "champion_cycle": row.get("cycle"),
    "champion_prom": champ_pr,
    "champion_gate_qualifies": champ_gate,
    "bar_real": bar,
    "task_bar": TASK_BAR,
    "diff_loci": diff_loci,
    "effectual_positions": effectual,
    "non_effectual": non_effectual,
    "max_subset_size": MAX_SUBSET,
    "planned_combos": planned_total,
    "unique_programs": len(all_rows),
    "best_prom": best,
    "best_desc": best_desc,
    "gap_to_task_bar": TASK_BAR - best,
    "hits_ge_task_bar": hits,
    "gate_top_n": len(top),
    "gate_qualified_in_top": gate_true,
    "max_gate_qualified_prom": max_gate_prom,
    "max_gate_qualified_gap": (None if max_gate_prom is None else TASK_BAR - max_gate_prom),
    "gate_qualified_hits": len(gate_hits),
    "top10": [{k: r[k] for k in ("prom", "subset_size", "loci", "deltas", "gate")} for r in top[:10]],
    "disk_cache_disabled": True,
    "runtime_s": time.time() - t0,
}
with open(EVIDENCE_PATH, "w", encoding="utf-8") as f:
    json.dump(evidence, f, indent=2)
log(f"evidence -> {EVIDENCE_PATH}")

if gate_hits:
    from evolution.mutation_helpers import record_scan_floor_eq_refine_seeds

    gate_hits.sort(key=lambda h: h["prom"], reverse=True)
    log("RECORDING breakthrough ...")
    record_scan_floor_eq_refine_seeds(champ, [list(gate_hits[0]["program"])], cycle=int(row.get("cycle") or 0))
    log("recorded.")
else:
    log("EXHAUSTED program-wide too: no gate-qualified variant >= task bar.")
log(f"done in {time.time() - t0:.1f}s")
