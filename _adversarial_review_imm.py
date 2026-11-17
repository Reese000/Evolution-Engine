"""Adversarial independent review of t_6f4be05a imm-locus-sweep claim.

Recomputes champion prom, gate arithmetic, samples the exhaustion claim with
cache bypass, probes OUT-OF-SET deltas (never tested by the probe), checks
prefix-locking, R-type/EBREAK effectualness, dedup/no-op arithmetic, evidence
self-consistency, and seeds-file invariants.
"""

import hashlib
import itertools
import json
import os
import time

os.environ.setdefault("EE_USE_GPU", "0")

from evolution import fitness_helpers as fh
from evolution import terminal_stagnation as ts
from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
from evolution.terminal_stagnation import (
    _SCAN_FLOOR_WIDE_IMM_DELTAS,
    curriculum_verify_mean,
    get_scan_floor_gate,
    scan_floor_min_raw_gain,
    terminal_efficiency_at_scan_floor,
)
from milestone1.rv32i_mask import decode as rv32i_decode
from milestone1.rv32i_mask import encode as rv32i_encode
from mutation_proposer import _is_ebreak

TASK_BAR = 325.836
WIDE = tuple(_SCAN_FLOOR_WIDE_IMM_DELTAS)
CLAIM_CHAMP = 325.82574193548385
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def fresh_score(prog):
    """Force a real evaluation: clear in-memo cache, block disk cache load."""
    fh._PROMOTION_RAW_CACHE.clear()
    ts._VM_SCORE_DISK_MEM = {}
    ts._VM_SCORE_DISK_LOADED = True
    return float(compact_l30_promotion_raw_fitness(list(prog)))


def apply_delta(prog, idx, delta):
    try:
        d = rv32i_decode(prog[idx], validate=False)
        if "imm" not in d:
            return None, "no_imm_field"
        w = rv32i_encode(
            name=str(d["name"]),
            rd=int(d.get("rd", 0)),
            rs1=int(d.get("rs1", 0)),
            rs2=int(d.get("rs2", 0)),
            imm=int(d.get("imm", 0)) + int(delta),
            validate=True,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"encode_{type(exc).__name__}"
    v = list(prog)
    v[idx] = int(w)
    if v[idx] == prog[idx]:
        return v, "noop_same_word"
    return v, "ok"


# ---------------------------------------------------------------- 1. champion
rows = open("multi_cycle_results.jsonl", encoding="utf-8").readlines()
row = json.loads(rows[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
log(f"jsonl last row: cycle={row.get('cycle')} words={len(champ)} " f"stored_prom={row.get('promotion_raw_fitness')}")

payload = json.load(open(".agents/scan_floor_eq_refine_seeds.json", encoding="utf-8"))
anchor = [int(x) for x in payload["anchor"]]
seeds = [[int(x) for x in s] for s in payload["seeds"]]

pr1 = fresh_score(champ)
pr2 = fresh_score(champ)  # determinism re-check
log(f"[1] champion FRESH prom pass1={pr1!r}")
log(f"[1] champion FRESH prom pass2={pr2!r}  deterministic={pr1 == pr2}")
log(f"[1] equals claim value {CLAIM_CHAMP}: {pr1 == CLAIM_CHAMP}")
log(f"[1] equals jsonl stored: {pr1 == row.get('promotion_raw_fitness')}")
log(f"[1] below 325.836: {pr1 < TASK_BAR}  below real bar: {pr1 < pr1 + 0.01}")

# ------------------------------------------------------------- 2. gate / bar
gain = scan_floor_min_raw_gain(len(champ))
at_floor = terminal_efficiency_at_scan_floor(len(champ))
bar = pr1 + gain
log(f"[2] terminal_efficiency_at_scan_floor(27) = {at_floor}")
log(f"[2] scan_floor_min_raw_gain(27) = {gain!r}  bar = {bar!r}")
log(f"[2] EE_SCAN_FLOOR_MIN_RAW_GAIN env = {os.environ.get('EE_SCAN_FLOOR_MIN_RAW_GAIN')!r}")

# --------------------------------------------------- 3. prefix-lock / inputs
log(
    f"[3] anchor==champ: {anchor == champ}  n_seeds={len(seeds)} "
    f"plateau_key={payload.get('plateau_key')} cycle={payload.get('cycle')}"
)
bad_prefix = []
for i, s in enumerate(seeds):
    diffs = [j for j in range(len(champ)) if s[j] != champ[j]]
    if any(d < 18 for d in diffs):
        bad_prefix.append((i, diffs))
    log(f"[3] seed{i}: first_diff={diffs[0] if diffs else '-'} diffs={diffs}")
log(f"[3] prefix 0-17 identical in champion+all seeds: {not bad_prefix}")

# ------------------------------------- 4. effectualness of 18,19,20 (R-type)
for i in (18, 19, 20):
    d = rv32i_decode(champ[i], validate=False)
    res = [apply_delta(champ, i, dl) for dl in (-64, -1, 1, 64)]
    kinds = sorted({k for _, k in res})
    log(
        f"[4] locus {i} {d.get('name')} word=0x{champ[i]:08X} imm={d.get('imm')} "
        f"delta outcomes={kinds} changed_words={any(v and v[idx] != champ[i] for v, _ in [(r[0], 0) for r in res] for idx in [i]) if False else any((v is not None and v[i] != champ[i]) for v, _ in res)}"
    )

# ------------------------------------------------------- 5. EBREAK locus 21
ebk = [apply_delta(champ, 21, dl) for dl in WIDE]
log(
    f"[5] locus21 EBREAK word=0x{champ[21]:08X} outcomes={sorted({k for _, k in ebk})} "
    f"changed={any(v is not None and v[21] != champ[21] for v, _ in ebk)}"
)

# ------------------------------- 6. size-1 coverage: planned 56 breakdown
eff = []
breakdown = {}
for i in (22, 23, 24, 25):
    for dl in WIDE:
        v, k = apply_delta(champ, i, dl)
        breakdown.setdefault((i, dl), k)
        if k == "ok":
            eff.append(i)
counts = {}
for k in breakdown.values():
    counts[k] = counts.get(k, 0) + 1
log(f"[6] size1 (4x14=56) outcomes: {counts}")
log(f"[6] effectual loci from empirical ok-deltas: {sorted(set(eff))}")
noop_ones = [
    f"locus{i}{'' if breakdown[(i, dl)] == 'ok' else ':' + breakdown[(i, dl)]}"
    for i in (22, 23, 24, 25)
    for dl in (1, -1)
    if breakdown[(i, dl)] != "ok"
]
log(
    "[6] delta +/-1 outcome per locus: "
    + ", ".join(f"L{i}: +1={breakdown[(i, 1)]}, -1={breakdown[(i, -1)]}" for i in (22, 23, 24, 25))
)

# --------------------- 7. sample of the exhaustion claim (set deltas, fresh)
SAMPLE = []
for locus in (22, 23, 24, 25):
    for dl in (64, -64, 32, -32, 16, -16):
        SAMPLE.append((locus, dl))
SAMPLE += [(23, 1), (23, -1), (23, 8), (23, -8), (24, 64), (25, -64), (22, 1), (22, -1)]
results7 = []
for locus, dl in SAMPLE:
    v, k = apply_delta(champ, locus, dl)
    if v is None:
        results7.append((locus, dl, k, None))
        continue
    if k == "noop_same_word":
        results7.append((locus, dl, k, pr1))
        continue
    p = fresh_score(v)
    results7.append((locus, dl, "ok", p))
above = [r for r in results7 if r[3] is not None and r[3] > pr1 + 1e-9]
below = [r for r in results7 if r[3] is not None and r[3] < pr1 - 1e-9]
distinct = sorted({round(r[3], 6) for r in results7 if r[3] is not None})
log(f"[7] sample n={len(results7)} (set-delta variants, fresh) " f"distinct_prom={distinct}")
log(f"[7] above champion: {len(above)}  below champion: {len(below)}")
for r in below[:6]:
    log(f"[7]   below example: locus={r[0]} delta={r[1]} prom={r[3]!r}")
log(f"[7] any >= 325.836: {any(r[3] is not None and r[3] >= TASK_BAR for r in results7)}")

# ------------- 8. ADVERSARIAL: deltas the probe NEVER tested (out-of-set)
OUT_DELTAS = (
    3,
    5,
    6,
    7,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    17,
    20,
    24,
    28,
    36,
    40,
    48,
    56,
    72,
    96,
    100,
    120,
    127,
    128,
    129,
    150,
    192,
    200,
    256,
    384,
    512,
    768,
    1000,
    1024,
    1536,
    2047,
    -3,
    -5,
    -6,
    -7,
    -9,
    -10,
    -12,
    -15,
    -17,
    -20,
    -24,
    -36,
    -48,
    -96,
    -100,
    -127,
    -128,
    -129,
    -192,
    -256,
    -512,
    -1000,
    -1024,
    -1536,
    -2047,
)
best8 = pr1
best8_desc = "champion"
seen8 = {tuple(champ)}
n8 = 0
above8 = []
hist8 = {}
for locus in (22, 23, 24, 25):
    for dl in OUT_DELTAS:
        v, k = apply_delta(champ, locus, dl)
        if v is None or k == "noop_same_word":
            continue
        key = tuple(v)
        if key in seen8:
            continue
        seen8.add(key)
        p = fresh_score(v)
        n8 += 1
        hist8[f"{p:.6f}"] = hist8.get(f"{p:.6f}", 0) + 1
        if p > pr1 + 1e-9:
            above8.append((p, locus, dl))
        if p > best8 + 1e-9:
            best8, best8_desc = p, f"locus={locus} delta={dl}"
log(f"[8] out-of-set deltas: {n8} fresh evals, distinct prom values={sorted(hist8)}")
log(f"[8] above champion: {len(above8)}  best={best8!r} via {best8_desc}")
if above8:
    for p, lc, dl in sorted(above8, reverse=True)[:10]:
        log(f"[8]   !! ABOVE: locus={lc} delta={dl} prom={p!r}")

# coarse full-range scan on one BLT locus (imm swings far outside +/-64)
best8b = pr1
above8b = []
n8b = 0
for dl in range(-4000, 4001, 61):  # 132 values, off-grid vs probe's deltas
    v, k = apply_delta(champ, 23, dl)
    if v is None or k == "noop_same_word":
        continue
    key = tuple(v)
    if key in seen8:
        continue
    seen8.add(key)
    p = fresh_score(v)
    n8b += 1
    if p > pr1 + 1e-9:
        above8b.append((p, dl))
    if p > best8b + 1e-9:
        best8b = p
log(
    f"[8b] locus23 coarse range scan (-4000..4000 step61): {n8b} fresh evals, "
    f"above champion={len(above8b)}, best={best8b!r}"
)

# ------------------------------------------- 9. evidence arithmetic self-check
ev_ex = json.load(open(".agents/evidence/t_6f4be05a_imm_exhaustive_champion.json"))
hist_sum = sum(ev_ex["prom_histogram"].values())
planned_sum = sum(r["planned"] for r in ev_ex["sizes_run"])
tested_sum = sum(r["distinct_tested"] for r in ev_ex["sizes_run"])
log(
    f"[9] exhaustive hist sum={hist_sum} total_distinct={ev_ex['total_distinct_evals']} "
    f"match={hist_sum == ev_ex['total_distinct_evals']}"
)
log(
    f"[9] size rows: planned_sum={planned_sum} tested_sum={tested_sum} "
    f"report claims 50624/32954 -> {planned_sum == 50624 and tested_sum == 32954}"
)
ev_cb = json.load(open(".agents/evidence/t_6f4be05a_imm_combo_sweep.json"))
pb_sum = sum(ev_cb["per_base_tested"].values())
log(
    f"[9] combo per-base sum={pb_sum} distinct_evals={ev_cb['distinct_evals']} "
    f"match={pb_sum == ev_cb['distinct_evals'] == 14408}"
)
totals = [407, 176, 14408, 32954]
log(f"[9] report totals table sum={sum(totals)} (claims 47945), " f"forced fresh={176 + 32954} (claims 33130)")

# ------------- 10. phase-3 self-consistency: is size2 subset of size3 program set?
eff_c = [22, 23, 24, 25]


def gen(size):
    out = set()
    for combo in itertools.combinations(eff_c, size):
        for deltas in itertools.product(WIDE, repeat=size):
            v = list(champ)
            ok = True
            for idx, dl in zip(combo, deltas):
                nv, k = apply_delta(v, idx, dl)
                if nv is None:
                    ok = False
                    break
                v = nv
            if ok:
                out.add(tuple(v))
    return out


s1, s2, s3 = gen(1), gen(2), gen(3)
log(
    f"[10] encoding-only: |size1|={len(s1)} |size2|={len(s2)} |size3|={len(s3)} "
    f"union(s1,s2,s3)={len(s1 | s2 | s3)} (phase4 says 50/936/7776/8762 cum)"
)
log(
    f"[10] size2 subset of size3: {s2 <= s3} | size1 subset of size3: {s1 <= s3} "
    f"| champ in any: {tuple(champ) in (s1 | s2 | s3)}"
)

# -------------------------------- 11. gate baseline + side levels (report claim)
g = get_scan_floor_gate(list(champ))
log(f"[11] gate heldout={g._heldout!r} side={tuple(round(x, 4) for x in g._side)}")
l10 = curriculum_verify_mean(champ, 10)
seed_l10 = [curriculum_verify_mean(s, 10) for s in seeds]
log(f"[11] champion L10={l10} seed L10s={sorted(set(seed_l10))}")

# --------------------------------------------- 12. seeds-file integrity (now)
raw = open(".agents/scan_floor_eq_refine_seeds.json", "rb").read()
log(
    f"[12] seeds file: n={len(seeds)} plateau_key={payload.get('plateau_key')} "
    f"cycle={payload.get('cycle')} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}"
)
log("[12] report claimed sha256 d82002d225c92feeed4a87c7dc3d34af66e6773ef6d6a15bc8981fef3c49e0ef " "size 2293")

# ------------------------------------ 13. fresh seed baselines (report claims)
for i, s in enumerate(seeds):
    p = fresh_score(s)
    log(f"[13] seed{i} fresh prom={p!r}")

log("DONE")
