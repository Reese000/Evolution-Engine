"""Prom-gate imm sweep at eq-seed diff loci.

Offline probe (no fitness/gate modification, no daemon writes except an
optional record_scan_floor_eq_refine_seeds call on breakthrough).

Phase 1: exhaustive locus-locked imm sweep at each seed's prefix-locked
         word-level diff positions vs the champion (champion + each seed).
Phase 2: bounded project-native terminal_imm_perturb + wide imm sweep.

Scores every candidate with compact_l30_promotion_raw_fitness only.
Records variants with prom_raw >= 325.836 AND gate-qualified.
"""

import json
import os
import random
import time

os.environ.setdefault("EE_USE_GPU", "0")

from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
from evolution.terminal_stagnation import (
    _SCAN_FLOOR_WIDE_IMM_DELTAS,
    curriculum_verify_mean,
    get_scan_floor_gate,
    scan_floor_min_raw_gain,
    terminal_imm_perturb_probe_variants,
    terminal_wide_imm_sweep_probe_variants,
)
from milestone1.rv32i_mask import decode as rv32i_decode
from milestone1.rv32i_mask import encode as rv32i_encode

FINE_DELTAS = (-8, -4, -2, -1, 1, 2, 4, 8)  # terminal_imm_perturb axis
WIDE_DELTAS = tuple(_SCAN_FLOOR_WIDE_IMM_DELTAS)  # wide imm sweep axis
TASK_BAR = 325.836  # gate stated in task
RNG_SEED = 20260924
EVIDENCE_PATH = ".agents/evidence/t_6f4be05a_imm_locus_sweep.json"

t_start = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


def apply_delta(prog: list[int], idx: int, delta: int) -> list[int] | None:
    """Re-encode prog[idx] with imm+delta; None when not imm-encodable."""
    try:
        decoded = rv32i_decode(prog[idx], validate=False)
        if "imm" not in decoded:
            return None
        new_word = rv32i_encode(
            name=str(decoded["name"]),
            rd=int(decoded.get("rd", 0)),
            rs1=int(decoded.get("rs1", 0)),
            rs2=int(decoded.get("rs2", 0)),
            imm=int(decoded.get("imm", 0)) + int(delta),
            validate=True,
        )
    except (ValueError, TypeError, KeyError):
        return None
    variant = list(prog)
    variant[idx] = int(new_word)
    return variant


# ---------------------------------------------------------------- load inputs
rows = open("multi_cycle_results.jsonl", encoding="utf-8").readlines()
row = json.loads(rows[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
payload = json.load(open(".agents/scan_floor_eq_refine_seeds.json", encoding="utf-8"))
anchor = [int(x) for x in payload["anchor"]]
seeds = [[int(x) for x in s] for s in payload["seeds"]]

log(f"champion: cycle={row.get('cycle')} words={len(champ)} anchor_match={anchor == champ}")
log(f"seeds: {len(seeds)} (lengths {sorted({len(s) for s in seeds})})")

gate = get_scan_floor_gate(champ)
champ_t = tuple(champ)
champ_pr = compact_l30_promotion_raw_fitness(champ)
gain = scan_floor_min_raw_gain(len(champ))
bar = champ_pr + gain
champ_l10 = curriculum_verify_mean(champ, 10)
log(f"champion prom={champ_pr:.6f} min_gain={gain:.6f} BAR={bar:.6f} task_bar={TASK_BAR}")
log(f"champion L10={champ_l10:.3f} gate_heldout={gate._heldout:.4f}")

# ------------------------------------------- prefix-locked diff loci per seed
seed_diffs: list[list[int]] = []
union_loci: set[int] = set()
for i, s in enumerate(seeds):
    if len(s) != len(champ):
        log(f"  seed{i}: LENGTH MISMATCH ({len(s)} != {len(champ)}) - skipped")
        seed_diffs.append([])
        continue
    diffs = [j for j in range(len(champ)) if s[j] != champ[j]]
    seed_diffs.append(diffs)
    union_loci.update(diffs)
    log(f"  seed{i}: first_div={diffs[0] if diffs else '-'} diffs={diffs}")

union_loci_sorted = sorted(union_loci)
log(f"union diff loci ({len(union_loci_sorted)}): {union_loci_sorted}")

# ------------------------------------------------------------- evaluate loop
seen: set[tuple[int, ...]] = {champ_t}
results: list[dict] = []
scored = 0


def evaluate(variant: list[int] | None, phase: str, base: str, locus, delta) -> None:
    global scored
    if variant is None or len(variant) != len(champ):
        return
    key = tuple(int(x) for x in variant)
    if key in seen:
        return
    seen.add(key)
    pr = compact_l30_promotion_raw_fitness(list(key))
    scored += 1
    results.append(
        {
            "prom": pr,
            "phase": phase,
            "base": base,
            "locus": locus,
            "delta": delta,
            "program": [int(x) for x in key],
        }
    )
    if pr >= TASK_BAR:
        log(f"  >= task_bar HIT prom={pr:.6f} phase={phase} base={base} " f"locus={locus} delta={delta}")
    if scored % 50 == 0:
        log(f"  ...scored {scored} variants, best so far " f"{max(r['prom'] for r in results):.6f}")


# ---- Phase 1a: champion at the union of eq-seed diff loci (prefix-locked) ---
log(
    f"Phase 1a: champion x {len(union_loci_sorted)} loci x " f"{len(FINE_DELTAS)} fine + {len(WIDE_DELTAS)} wide deltas"
)
for locus in union_loci_sorted:
    for axis, deltas in (("imm_perturb_fine", FINE_DELTAS), ("wide_imm_sweep", WIDE_DELTAS)):
        for d in deltas:
            evaluate(apply_delta(champ, locus, d), axis, "champion", locus, d)
log(f"Phase 1a done: scored={scored}")

# ---- Phase 1b: each seed at its own prefix-locked diff loci -----------------
for i, s in enumerate(seeds):
    if not seed_diffs[i]:
        continue
    for locus in seed_diffs[i]:
        for axis, deltas in (("imm_perturb_fine", FINE_DELTAS), ("wide_imm_sweep", WIDE_DELTAS)):
            for d in deltas:
                evaluate(apply_delta(s, locus, d), axis, f"seed{i}", locus, d)
    log(f"Phase 1b seed{i} done: scored={scored}")

# ---- Phase 2: bounded project-native named probes ---------------------------
log("Phase 2: terminal_imm_perturb_probe_variants + terminal_wide_imm_sweep_probe_variants")
rng = random.Random(RNG_SEED)
for name, prog in [("champion", champ)] + [(f"seed{i}", s) for i, s in enumerate(seeds)]:
    try:
        fine_vars = terminal_imm_perturb_probe_variants(list(prog), rng, max_evals=24, gate=gate)
    except (ValueError, TypeError, KeyError, ImportError) as exc:
        log(f"  imm_perturb error on {name}: {exc}")
        fine_vars = []
    for v in fine_vars:
        evaluate(v, "terminal_imm_perturb", name, None, None)
    try:
        wide_vars = terminal_wide_imm_sweep_probe_variants(list(prog), max_evals=64, start_offset=0, gate=gate)
    except (ValueError, TypeError, KeyError, ImportError) as exc:
        log(f"  wide_sweep error on {name}: {exc}")
        wide_vars = []
    for v in wide_vars:
        evaluate(v, "wide_imm_sweep_native", name, None, None)
    log(f"  Phase 2 {name}: imm_perturb={len(fine_vars)} wide_sweep={len(wide_vars)} " f"scored={scored}")

# ------------------------------------------------------------------ analysis
results.sort(key=lambda r: r["prom"], reverse=True)
log(f"TOTAL scored={scored} distinct variants")

# Gate-qualify: every variant at/above the task bar, plus top-15 overall.
gate_checked: dict[str, bool] = {}
to_check = [r for r in results if r["prom"] >= TASK_BAR]
for r in results[:15]:
    if r not in to_check:
        to_check.append(r)
log(f"gate-checking {len(to_check)} candidates")
for r in to_check:
    gk = tuple(r["program"])
    ok = gate.qualifies(list(gk))
    gate_checked[repr(gk)] = ok
    r["gate"] = ok

breakthroughs = [r for r in results if r["prom"] >= TASK_BAR and gate_checked.get(repr(tuple(r["program"])), False)]
bar_qualified = [r for r in results if r["prom"] > bar + 1e-6 and gate_checked.get(repr(tuple(r["program"])), False)]

log("=" * 70)
log(f"BAR (real)   = {bar:.6f}")
log(f"TASK BAR     = {TASK_BAR}")
log(f"best prom found = {results[0]['prom'] if results else float('nan'):.6f}")
log(f"champion prom   = {champ_pr:.6f}")
n_at_or_above = sum(1 for r in results if r["prom"] >= TASK_BAR)
n_above_real = sum(1 for r in results if r["prom"] > bar + 1e-6)
log(f"variants >= TASK_BAR: {n_at_or_above} | > real bar: {n_above_real}")
log(f"BREAKTHROUGHS (prom >= {TASK_BAR} AND gate qualifies): {len(breakthroughs)}")
for r in breakthroughs:
    l10 = curriculum_verify_mean(r["program"], 10)
    log(
        f"  * prom={r['prom']:.6f} L10={l10:.3f} phase={r['phase']} "
        f"base={r['base']} locus={r['locus']} delta={r['delta']}"
    )
log("top 10 by prom:")
for r in results[:10]:
    g = gate_checked.get(repr(tuple(r["program"])), None)
    log(
        f"  prom={r['prom']:.6f} gate={g} phase={r['phase']} base={r['base']} " f"locus={r['locus']} delta={r['delta']}"
    )

# ------------------------------------------------------------------- persist
os.makedirs(os.path.dirname(EVIDENCE_PATH), exist_ok=True)
evidence = {
    "task": "t_6f4be05a",
    "champion_cycle": row.get("cycle"),
    "champion_words": len(champ),
    "champion_prom": champ_pr,
    "champion_l10": champ_l10,
    "min_raw_gain": gain,
    "bar_real": bar,
    "task_bar": TASK_BAR,
    "union_diff_loci": union_loci_sorted,
    "seed_diffs": seed_diffs,
    "rng_seed": RNG_SEED,
    "variants_scored": scored,
    "variants_ge_task_bar": n_at_or_above,
    "variants_gt_real_bar": n_above_real,
    "breakthrough_count": len(breakthroughs),
    "breakthroughs": [
        {
            "prom": r["prom"],
            "phase": r["phase"],
            "base": r["base"],
            "locus": r["locus"],
            "delta": r["delta"],
            "l10": curriculum_verify_mean(r["program"], 10),
            "program": r["program"],
        }
        for r in breakthroughs
    ],
    "bar_qualified_count": len(bar_qualified),
    "top_results": [
        {
            "prom": r["prom"],
            "gate": gate_checked.get(repr(tuple(r["program"])), None),
            "phase": r["phase"],
            "base": r["base"],
            "locus": r["locus"],
            "delta": r["delta"],
        }
        for r in results[:20]
    ],
    "runtime_s": time.time() - t_start,
}
with open(EVIDENCE_PATH, "w", encoding="utf-8") as f:
    json.dump(evidence, f, indent=2)
log(f"evidence written -> {EVIDENCE_PATH}")

# ------------------------------------------- record breakthrough if any found
if breakthroughs:
    from evolution.mutation_helpers import record_scan_floor_eq_refine_seeds

    best = breakthroughs[0]
    log("RECORDING breakthrough via record_scan_floor_eq_refine_seeds ...")
    record_scan_floor_eq_refine_seeds(champ, [list(best["program"])], cycle=int(row.get("cycle") or 0))
    log("recorded.")
else:
    log("NO breakthrough >= task bar with gate qualification - nothing recorded (exhausted).")

log(f"done in {time.time() - t_start:.1f}s")
