"""Phase 3: dual/triple imm combinations at eq-seed diff loci.

Single-site imm sweep at loci 18..25 is exhausted (best == champion
325.825742, gap 0.010258 to the 325.836 task bar). This extends the SAME
axis (immediate values) over the SAME prefix-locked loci to 2-way and
3-way combinations, still scoring compact_l30_promotion_raw_fitness only.

No fitness/gate modification. Records via record_scan_floor_eq_refine_seeds
only if a variant reaches the task bar AND gate-qualifies.
"""
import itertools
import json
import os
import time

os.environ.setdefault("EE_USE_GPU", "0")

from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
from evolution.terminal_stagnation import (
    _SCAN_FLOOR_WIDE_IMM_DELTAS,
    get_scan_floor_gate,
    scan_floor_min_raw_gain,
)
from milestone1.rv32i_mask import decode as rv32i_decode, encode as rv32i_encode
from mutation_proposer import _is_ebreak

TASK_BAR = 325.836
WIDE = tuple(_SCAN_FLOOR_WIDE_IMM_DELTAS)
MAX_EVALS = int(os.environ.get("EE_PROBE_COMBO_MAX_EVALS", "40000"))
EVIDENCE_PATH = ".agents/evidence/t_6f4be05a_imm_combo_sweep.json"
t0 = time.time()
evals = 0
budget_hit = False


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
    """Loci where at least one wide delta actually changes the program."""
    out = []
    for i in loci:
        if _is_ebreak(base[i]):
            continue
        for dl in WIDE:
            nv = apply_delta_one(base, i, dl)
            if nv is not None and tuple(nv) != tuple(base):
                out.append(i)
                break
    return out


# ------------------------------------------------------------------- inputs
rows = open("multi_cycle_results.jsonl", encoding="utf-8").readlines()
row = json.loads(rows[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
payload = json.load(open(".agents/scan_floor_eq_refine_seeds.json", encoding="utf-8"))
anchor = [int(x) for x in payload["anchor"]]
seeds = [[int(x) for x in s] for s in payload["seeds"]]
assert anchor == champ, "champion drifted from seeds anchor; aborting"

gate = get_scan_floor_gate(champ)
champ_t = tuple(champ)
champ_pr = compact_l30_promotion_raw_fitness(champ)
bar = champ_pr + scan_floor_min_raw_gain(len(champ))
loci = sorted({j for s in seeds for j in range(len(champ)) if s[j] != champ[j]})
log(f"cycle={row.get('cycle')} prom={champ_pr:.6f} BAR={bar:.6f} task_bar={TASK_BAR}")
log(f"diff loci={loci}")

bases = [("champion", champ)] + [(f"seed{i}", s) for i, s in enumerate(seeds)]
eff = {name: effectual_loci(b, loci) for name, b in bases}
for name, _ in bases:
    log(f"  {name}: imm-effectual diff loci = {eff[name]}")

# ------------------------------------------------------------------- sweep
seen: set[tuple[int, ...]] = {champ_t}
best = champ_pr
best_desc = "champion(unmodified)"
hits: list[dict] = []
per_base_best: dict[str, float] = {"champion": champ_pr}
tested_by_base: dict[str, int] = {}
plan: list[tuple[str, list[int], int]] = []
# champion: 3-way then 2-way (closest to the bar first); seeds: 2-way only
for name, b in bases:
    e = eff[name]
    sizes = (3, 2) if name == "champion" else (2,)
    for sz in sizes:
        if len(e) >= sz:
            plan.append((name, list(itertools.combinations(e, sz)), sz))

log(f"plan: " + ", ".join(f"{n}/size{sz}x{len(c)}" for n, c, sz in plan))

for name, combos, sz in plan:
    base = dict(bases)[name]
    for combo in combos:
        for deltas in itertools.product(WIDE, repeat=sz):
            if evals >= MAX_EVALS:
                budget_hit = True
                break
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
                continue
            seen.add(key)
            evals += 1
            pr = float(compact_l30_promotion_raw_fitness(list(key)))
            tested_by_base[name] = tested_by_base.get(name, 0) + 1
            if pr > best + 1e-9:
                best = pr
                best_desc = f"{name} loci={list(combo)} deltas={list(deltas)}"
                log(f"  NEW BEST prom={pr:.6f} {best_desc}")
            per_base_best[name] = max(per_base_best.get(name, 0.0), pr)
            if pr >= TASK_BAR:
                hits.append({"prom": pr, "base": name, "loci": list(combo),
                             "deltas": list(deltas), "program": [int(x) for x in key]})
                log(f"  !! >= task_bar {pr:.6f} {name} {list(combo)} {list(deltas)}")
            if evals % 2000 == 0:
                log(f"  ...evals={evals}/{MAX_EVALS} best={best:.6f}")
        if budget_hit:
            break
    log(f"done {name} size{sz}: evals={evals} best={best:.6f}")
    if budget_hit:
        log("BUDGET HIT - stopping")
        break

# ------------------------------------------------------------ gate + report
log("=" * 72)
log(f"total distinct evals = {evals} (budget {MAX_EVALS}, budget_hit={budget_hit})")
log(f"per-base tested: {tested_by_base}")
log(f"per-base best prom: { {k: round(v, 6) for k, v in per_base_best.items()} }")
log(f"BEST overall = {best:.6f} ({best_desc})")
log(f"champion     = {champ_pr:.6f}   gap to task bar = {TASK_BAR - best:.6f}")

gate_ok_hits = []
for h in hits:
    ok = gate.qualifies(h["program"])
    h["gate"] = ok
    log(f"  hit prom={h['prom']:.6f} gate={ok} base={h['base']} "
        f"loci={h['loci']} deltas={h['deltas']}")
    if ok:
        gate_ok_hits.append(h)

log(f"hits >= task_bar: {len(hits)} | gate-qualified: {len(gate_ok_hits)}")

os.makedirs(os.path.dirname(EVIDENCE_PATH), exist_ok=True)
evidence = {
    "task": "t_6f4be05a",
    "phase": "dual/triple imm combination sweep at eq-seed diff loci",
    "champion_cycle": row.get("cycle"),
    "champion_prom": champ_pr,
    "bar_real": bar,
    "task_bar": TASK_BAR,
    "diff_loci": loci,
    "imm_effectual_loci": eff,
    "distinct_evals": evals,
    "eval_budget": MAX_EVALS,
    "budget_hit": budget_hit,
    "per_base_tested": tested_by_base,
    "per_base_best_prom": per_base_best,
    "best_prom": best,
    "best_desc": best_desc,
    "gap_to_task_bar": TASK_BAR - best,
    "hits_ge_task_bar": hits,
    "gate_qualified_hits": len(gate_ok_hits),
    "runtime_s": time.time() - t0,
}
with open(EVIDENCE_PATH, "w", encoding="utf-8") as f:
    json.dump(evidence, f, indent=2)
log(f"evidence -> {EVIDENCE_PATH}")

if gate_ok_hits:
    gate_ok_hits.sort(key=lambda h: h["prom"], reverse=True)
    from evolution.mutation_helpers import record_scan_floor_eq_refine_seeds

    top = gate_ok_hits[0]
    log("RECORDING breakthrough via record_scan_floor_eq_refine_seeds ...")
    record_scan_floor_eq_refine_seeds(champ, [list(top["program"])],
                                      cycle=int(row.get("cycle") or 0))
    log("recorded.")
else:
    log("NO gate-qualified variant >= task bar - combo axis exhausted (nothing recorded).")

log(f"done in {time.time() - t0:.1f}s")
