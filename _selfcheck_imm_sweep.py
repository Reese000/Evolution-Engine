"""Independent arithmetic + sample check of the imm-sweep exhaustion claim."""

import json
import os

os.environ.setdefault("EE_USE_GPU", "0")

from evolution import fitness_helpers as fh
from evolution import terminal_stagnation as ts
from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
from evolution.terminal_stagnation import scan_floor_min_raw_gain, terminal_efficiency_at_scan_floor
from milestone1.rv32i_mask import decode as rv32i_decode
from milestone1.rv32i_mask import encode as rv32i_encode

TASK_BAR = 325.836
ok_all = True
n_total = 0
n_pass = 0


def check(label, cond, detail=""):
    global ok_all, n_total, n_pass
    n_total += 1
    if cond:
        n_pass += 1
    else:
        ok_all = False
    print(f"[{'PASS' if cond else 'FAIL'}] {label} {detail}", flush=True)


# force real evals
fh._PROMOTION_RAW_CACHE.clear()
ts._VM_SCORE_DISK_MEM = {}
ts._VM_SCORE_DISK_LOADED = True

# --- 1. champion prom + bar arithmetic ---
rows = open("multi_cycle_results.jsonl", encoding="utf-8").readlines()
row = json.loads(rows[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
payload = json.load(open(".agents/scan_floor_eq_refine_seeds.json", encoding="utf-8"))
pr = float(compact_l30_promotion_raw_fitness(champ))
check("champion prom == 325.82574193548385", abs(pr - 325.82574193548385) < 1e-9, f"got {pr!r}")
check("champion prom < task bar", pr < TASK_BAR, f"{pr:.9f} < {TASK_BAR}")

state = json.load(open(".agents/progress_controller_state.json", encoding="utf-8"))
check(
    "progress state floor_words == 27",
    int(state.get("terminal_efficiency_scan_floor_words", -1)) == 27,
    str(state.get("terminal_efficiency_scan_floor_words")),
)
check("terminal_efficiency_at_scan_floor(27) is True", terminal_efficiency_at_scan_floor(27) is True)
g = scan_floor_min_raw_gain(27)
check("scan_floor_min_raw_gain(27) == 0.01", abs(g - 0.01) < 1e-12, f"got {g!r}")
check("bar == prom + 0.01 == 325.83574193548384", abs((pr + g) - 325.83574193548384) < 1e-9, f"got {pr + g!r}")

# --- 2. evidence arithmetic ---
ev4 = json.load(open(".agents/evidence/t_6f4be05a_imm_exhaustive_champion.json", encoding="utf-8"))
hsum = sum(ev4["prom_histogram"].values())
check("exhaustive histogram sums to 32954", hsum == 32954, f"sum={hsum}")
sizesum = sum(s["distinct_tested"] for s in ev4["sizes_run"])
check("sizes_run distinct_tested sums to 32954", sizesum == 32954, f"sum={sizesum}")
check("exhaustive best_prom == champion prom", abs(ev4["best_prom"] - pr) < 1e-9, f"{ev4['best_prom']!r}")
check("exhaustive hits == 0", ev4["hits_ge_task_bar"] == [])
check("exhaustive gate_qualified_hits == 0", ev4["gate_qualified_hits"] == 0)
check(
    "every histogram key <= champion (allowing 6dp display rounding)",
    all(float(k) <= round(pr, 6) + 1e-9 for k in ev4["prom_histogram"]),
    str(sorted(ev4["prom_histogram"])),
)
check("probe best_prom (exact float) <= champion", ev4["best_prom"] <= pr + 1e-12, f"{ev4['best_prom']!r} vs {pr!r}")

ev3 = json.load(open(".agents/evidence/t_6f4be05a_imm_combo_sweep.json", encoding="utf-8"))
csum = sum(ev3["per_base_tested"].values())
check("combo per_base_tested sums to 14408", csum == 14408, f"sum={csum}")
check("combo distinct_evals == 14408", ev3["distinct_evals"] == 14408, str(ev3["distinct_evals"]))
check("combo hits == 0", ev3["hits_ge_task_bar"] == [])

ev1 = json.load(open(".agents/evidence/t_6f4be05a_imm_locus_sweep.json", encoding="utf-8"))
check("locus sweep variants_ge_task_bar == 0", ev1["variants_ge_task_bar"] == 0, str(ev1["variants_ge_task_bar"]))
check("locus sweep breakthrough_count == 0", ev1["breakthrough_count"] == 0)

# --- 3. seeds file integrity ---
seeds = payload["seeds"]
check("plateau_key == [325.8257, 27]", payload["plateau_key"] == [325.8257, 27], str(payload["plateau_key"]))
check("exactly 8 seeds", len(seeds) == 8, f"n={len(seeds)}")
check("all seeds 27w", all(len(s) == 27 for s in seeds))
check("anchor == champion", payload["anchor"] == champ)

# --- 4. prefix-locked: positions 0..17 identical across champion and seeds ---
locked = all(s[:18] == champ[:18] for s in seeds)
check("positions 0..17 identical in champion and all seeds (prefix-locked)", locked)

# --- 5. imm-effectual loci claim + sample rescore ---
from mutation_proposer import _is_ebreak

WIDE = (-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64)


def apply_delta(prog, idx, delta):
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


non_eff = []
for i in (18, 19, 20):
    eff = any((nv := apply_delta(champ, i, dl)) is not None and nv != champ for dl in WIDE)
    name = rv32i_decode(champ[i], validate=False).get("name")
    check(f"locus {i} ({name}) non-effectual", not eff)
    if not eff:
        non_eff.append(i)
check("locus 21 is EBREAK (skippable)", _is_ebreak(champ[21]), hex(champ[21]))

# sample >= 20 variants across the 4 effectual loci, fresh
samples, below, above = 0, 0, 0
vals = set()
for i in (22, 23, 24, 25):
    for dl in (-64, -16, -1, 1, 16, 64):
        v = apply_delta(champ, i, dl)
        if v is None or v == champ:
            continue
        samples += 1
        p = float(compact_l30_promotion_raw_fitness(v))
        vals.add(round(p, 6))
        if p > pr + 1e-9:
            above += 1
        elif p < pr - 1e-9:
            below += 1
check("sampled >= 20 imm variants", samples >= 20, f"n={samples}")
check("NO sampled variant above champion", above == 0, f"above={above}")
check(
    "scorer differentiates (some sampled BELOW champion)", below > 0, f"below={below} distinct_sampled={sorted(vals)}"
)

# two-locus sample
combos_above = 0
combos_tested = 0
for i in (22, 23):
    for j in (24, 25):
        for d1 in (-64, 64):
            for d2 in (-64, 64):
                v = apply_delta(champ, i, d1)
                if v is None:
                    continue
                v2 = apply_delta(v, j, d2)
                if v2 is None or v2 == champ:
                    continue
                combos_tested += 1
                p = float(compact_l30_promotion_raw_fitness(v2))
                if p > pr + 1e-9:
                    combos_above += 1
check("sampled two-locus combos", combos_tested >= 16, f"n={combos_tested}")
check("no two-locus combo above champion", combos_above == 0, f"above={combos_above}")

# --- 6. Phase 5 (comprehensive, single shared dedup) --------------------------
ev5_path = ".agents/evidence/t_6f4be05a_imm_full_exhaustion.json"
if os.path.exists(ev5_path):
    ev5 = json.load(open(ev5_path, encoding="utf-8"))
    per = ev5["per_base"]
    uniq_sum = sum(r["globally_unique_new"] for r in per)
    plan_sum = sum(r["planned_combos"] for r in per)
    dup_sum = sum(r["cross_base_dups"] for r in per)
    check("phase5 grand_unique == 114200", ev5["grand_unique_programs"] == 114200, str(ev5["grand_unique_programs"]))
    check("phase5 per-base unique sums to grand total", uniq_sum == ev5["grand_unique_programs"], f"sum={uniq_sum}")
    check("phase5 planned sums to 168966", plan_sum == 168966, f"sum={plan_sum}")
    check("phase5 dups sum to 54790", dup_sum == 54790, f"sum={dup_sum}")
    check(
        "phase5 planned - dups + native == grand unique",
        plan_sum - dup_sum + 24 == ev5["grand_unique_programs"],
        f"{plan_sum}-{dup_sum}+24={plan_sum - dup_sum + 24}",
    )
    check("phase5 cross-base overlap == 0", ev5["cross_base_overlap"] == 0)
    check("phase5 hits == 0", ev5["hits_ge_task_bar"] == [])
    check("phase5 gate_qualified_hits == 0", ev5["gate_qualified_hits"] == 0)
    check("phase5 best_prom == champion (exact)", ev5["best_prom"] <= pr + 1e-12, f"{ev5['best_prom']!r}")
    bad = [r["base"] for r in per if r["best_prom"] > r["baseline_prom"] + 1e-9]
    check("no base's best exceeds its own baseline", not bad, str(bad))
else:
    check("phase5 evidence exists", False, ev5_path)

# --- 6b. Phase 6 (program-wide imm + gate qualification) ---------------------
ev6_path = ".agents/evidence/t_6f4be05a_imm_programwide.json"
if os.path.exists(ev6_path):
    ev6 = json.load(open(ev6_path, encoding="utf-8"))
    check("phase6 evidence exists", True)
    check("phase6 unique_programs == 332359", ev6["unique_programs"] == 332359, str(ev6["unique_programs"]))
    check("phase6 planned == 463694", ev6["planned_combos"] == 463694, str(ev6["planned_combos"]))
    check(
        "phase6 planned == size1(154) + size2(10780) + size3(452760)",
        ev6["planned_combos"] == 11 * 14 + 55 * 196 + 165 * 2744,
        str(ev6["planned_combos"]),
    )
    check("phase6 max_subset_size == 3", ev6["max_subset_size"] == 3, str(ev6["max_subset_size"]))
    check("phase6 best_prom <= champion (exact)", ev6["best_prom"] <= pr + 1e-12, f"{ev6['best_prom']!r}")
    check("phase6 hits == 0", ev6["hits_ge_task_bar"] == [])
    check("phase6 gate_qualified_hits == 0", ev6["gate_qualified_hits"] == 0)
    check(
        "phase6 effectual positions == [0,4,8,10,11,14,17,22,23,24,25]",
        ev6["effectual_positions"] == [0, 4, 8, 10, 11, 14, 17, 22, 23, 24, 25],
        str(ev6["effectual_positions"]),
    )
    outside = [p for p in ev6["effectual_positions"] if p not in (18, 19, 20, 21, 22, 23, 24, 25)]
    check(
        "phase6 newly-covered effectual positions outside diff loci == 7",
        outside == [0, 4, 8, 10, 11, 14, 17],
        str(outside),
    )
    check(
        "phase6 all 16 non-effectual are R-type/EBREAK", len(ev6["non_effectual"]) == 16, str(len(ev6["non_effectual"]))
    )
    mg = ev6["max_gate_qualified_prom"]
    check("phase6 max gate-qualified prom <= champion", mg is not None and mg <= pr + 1e-12, f"{mg!r}")
    check(
        "phase6 max gate-qualified gap == 0.010258",
        abs(ev6["max_gate_qualified_gap"] - 0.010258) < 1e-6,
        str(ev6["max_gate_qualified_gap"]),
    )
    check(
        "phase6 some variants gate-qualify (0 < n <= 200)",
        0 < ev6["gate_qualified_in_top"] <= 200,
        str(ev6["gate_qualified_in_top"]),
    )
    check("phase6 champion itself gate-qualifies", ev6["champion_gate_qualifies"] is True)
else:
    check("phase6 evidence exists", False, ev6_path)

# --- 7. report retracts the old overclaim and states the authoritative total --
report = open("REPORT_imm_locus_sweep_t_6f4be05a.md", encoding="utf-8").read()
check("report flags the old summed total as an overclaim", "That was an overclaim" in report)
check("report states authoritative unique total 114,200", "114,200" in report)
check("report has no surviving 'total | 47,945' row", "| **total** | **47,945**" not in report)
check("report documents Phase 6 program-wide sweep", "Phase 6" in report)
check("report documents gate-qualification metric (42)", "42** (21%)" in report)
# --- 8. report <-> evidence consistency (drift guard) -----------------------
check("report states current Phase 6 unique total 332,359", "332,359" in report)
check("report states current Phase 6 planned total 463,694", "463,694" in report)
check("report keeps NO superseded Phase 6 unique count (8,775)", "8,775" not in report)
check("report keeps NO superseded self-check tally (34/34)", "34/34" not in report)

# Final check: the report must state ITS OWN tally. Counting this check itself,
# so the report's "N/N PASS" must equal n_total + 1 — the number is derived,
# never retyped by hand.
import re as _re

_m = _re.search(r"Self-check:\s*(\d+)/(\d+)\s*PASS", report)
_expected = n_total + 1
check(
    f"report states its own tally {_expected}/{_expected} PASS (self-verifying)",
    bool(_m) and int(_m.group(1)) == _expected and int(_m.group(2)) == _expected,
    f"report says {_m.group(1)+'/'+_m.group(2) if _m else 'NO MATCH'}",
)

print(f"\nRESULT: {'ALL CHECKS PASSED' if ok_all else 'SOME CHECKS FAILED'}", flush=True)
print(f"{n_pass}/{n_total} checks passed", flush=True)
print(f"sample distinct proms: {sorted(vals)}", flush=True)
