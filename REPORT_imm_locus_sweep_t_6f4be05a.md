# Prom-gate imm sweep at eq-seed diff loci — t_6f4be05a

**Verdict: EXHAUSTED.** 0 variants reached prom_raw >= 325.836. Nothing recorded.

## Objective

Champion prom_raw = 325.8257, promotion gate = 325.836 (~0.01 short). The 8
eq-refine disk seeds are L10-lift (0.825) but prom-regressive (~321). Task:
offline probe — bounded immediate-value perturbation at each seed's
prefix-locked word-level diff position vs the champion, scoring
`compact_l30_promotion_raw_fitness` only, no fitness/gate weakening, read-only
on the live daemon.

## Inputs (verified)

| Item | Value |
| --- | --- |
| Champion source | `multi_cycle_results.jsonl` last row (cycle 7560 at load, 7562 at finish — daemon advancing) |
| Champion words | 27 |
| Seeds-file `anchor` vs champion | identical (`anchor_match=True`, asserted) |
| Champion prom_raw | **325.82574193548385** |
| min_raw_gain (27w) | 0.01 (scan floor armed: `terminal_efficiency_scan_floor_words=27`, heldout 0.990132) |
| Real bar | **325.83574193548384** |
| Task bar | **325.836** |
| Gap to close | 0.010258 |
| Seeds | 8, all 27w, gate-qualified |
| Seed prom | seed0/1 = 321.331871 (-4.493871); seed2..7 = 318.831871 (-6.993871) |
| Seed L10 | 0.825 vs champion L10 0.800 |
| Gate baseline | heldout_l30 = 0.9901, side = (0.8889, 0.8889, 0.8, 0.8906, 0.875, 0.875, 0.875, 0.8906, 0.875) |
| Interpreter | `.venv-win/Scripts/python.exe` (project venv), `EE_USE_GPU=0` |

## Prefix-locked diff loci

Seeds share a common prefix with the champion; first divergence is at 18 or 21.

| seed | first_div | diff loci |
| --- | --- | --- |
| 0, 1 | 21 | 21, 22, 23, 24 |
| 2, 3, 4 | 21 | 21, 22, 23, 24, 25 |
| 5, 6, 7 | 18 | 18..25 |
| **union** | — | **18, 19, 20, 21, 22, 23, 24, 25** |

Positions 0–17 are identical in the champion and all 8 seeds → prefix-locked,
out of scope by the task's own definition.

### Which diff loci can an immediate edit actually change?

| locus | champion word | insn | imm-effectual? |
| --- | --- | --- | --- |
| 18 | 0x40208CB3 | SUB | no — R-type, `encode` ignores `imm` |
| 19 | 0x40208DB3 | SUB | no — R-type |
| 20 | 0x40110EB3 | SUB | no — R-type |
| 21 | 0x00100073 | EBREAK | no — skipped (project also skips `_is_ebreak`) |
| 22 | 0x001C8023 | SB | **yes** |
| 23 | 0xFE2840E3 | BLT | **yes** |
| 24 | 0xFE18C8E3 | BLT | **yes** |
| 25 | 0xFE18C8E3 | BLT | **yes** |

So the champion's imm search space at these loci is exactly
`{22,23,24,25}` x `(-64..64, 14 deltas)`.

## Phases and results

### Phase 1 + 2 — single-site sweep, champion + all 8 seeds
`_probe_imm_locus_sweep.py` (log: `_probe_imm_locus_sweep.log`)

- Phase 1a: champion x 8 diff loci x (8 fine deltas from
  `terminal_imm_perturb` + 14 wide deltas) — prefix-locked.
- Phase 1b: each seed x its own diff loci x same delta sets.
- Phase 2: the project's own `terminal_imm_perturb_probe_variants` (rng) and
  `terminal_wide_imm_sweep_probe_variants` (deterministic slot x delta sweep),
  bounded to 24/64 evals per base, gate-filtered.

| metric | value |
| --- | --- |
| distinct variants scored | 407 |
| best prom found | 325.82574193548385 (= champion) |
| variants >= 325.836 | **0** |
| variants > real bar | **0** |

### Phase 1-verify — fresh (cache-bypassed) re-score
`_verify_imm_locus_sweep.py` (log: `_verify_imm_locus_sweep.log`)

- Forces real evaluation by clearing `_PROMOTION_RAW_CACHE` and
  `ts._vm_score_disk_reset()` on every call.
- Champion x 8 imm loci x 22 deltas = 176 attempts, all built successfully.
- **0 variants strictly above champion**; 168 tie it; 8 drop to 324.825742.
- Cache consistency (5 samples): `cached == repeat == fresh` for every sample
  → no stale-cache contamination.
- Cross-check: fresh champion prom `325.8257419355` == the daemon's own
  `promotion_raw_fitness` written into jsonl for cycle 7561 == my cached value.

### Phase 3 — dual/triple imm combinations, champion + 8 seeds
`_probe_imm_combo_sweep.py` (log: `_probe_imm_combo_sweep.log`)

| base | imm-effectual loci | distinct evals | best prom |
| --- | --- | --- | --- |
| champion | 22,23,24,25 | 8,762 (size3 + size2) | 325.825742 |
| seed0 | 22,24,25 | 560 | 321.331871 |
| seed1 | 21,23,24,25 | 942 | 321.331871 |
| seed2 | 22,24,25 | 588 | 318.831871 |
| seed3 | 21,23,24,25 | 1,008 | 318.831871 |
| seed4 | 21,23,25 | 476 | 318.831871 |
| seed5 | 19,21,22 | 588 | 318.831871 |
| seed6 | 18,20,21,22 | 1,008 | 318.831871 |
| seed7 | 18,20,22 | 476 | 318.831871 |
| **total** | | **14,408** | **325.825742** |

Hits >= 325.836: **0**. Note every seed's best equals its own baseline — no
imm edit ever improved a seed by even one point.

### Phase 4 — EXHAUSTIVE champion imm space, forced fresh evals
`_probe_imm_exhaustive_champion.py` (log: `_probe_imm_exhaustive_champion.log`)

Disk score cache disabled in-process (`ts._VM_SCORE_DISK_MEM = {}`,
`_VM_SCORE_DISK_LOADED = True`) so **every** evaluation is a real scoring pass.

| subset size | locus subsets | planned | distinct tested |
| --- | --- | --- | --- |
| 1 | 4 | 56 | 50 |
| 2 | 6 | 1,176 | 936 |
| 3 | 4 | 10,976 | 7,776 |
| 4 | 1 | 38,416 | 24,192 |
| **total** | | **50,624** | **32,954** |

(planned > distinct because equivalent encodings collapse and the champion
itself is excluded)

| metric | value |
| --- | --- |
| **BEST** | **325.8257419355 = champion (unmodified)** |
| variants >= 325.836 | **0** |
| gate-qualified hits | **0** |
| gap to task bar | 0.010258 |

Prom histogram over all 32,954 real evals (sums exactly to 32,954):

| prom | count |
| --- | --- |
| 325.825742 (= champion) | 16,499 |
| 324.825742 (−1.0) | 12,900 |
| 322.825742 (−3.0) | 3,285 |
| 320.825742 (−5.0) | 270 |

**Only 4 distinct values exist across the entire imm space, all <= champion.**
The champion is a strict local maximum on the immediate-value axis; no imm
edit can add the required +0.0103.

## Phase 5 — comprehensive exhaustion, one shared dedup set

`_probe_imm_full_exhaustion.py` (log: `_probe_imm_full_exhaustion.log`)

Added after self-review found two weaknesses in the phases above:

1. each earlier phase used its **own** `seen` set, so summing their counts
   double-counted cross-phase overlap;
2. seeds had only been swept at subset sizes 1–2.

This run keeps a **single shared `seen` set** across the champion and all 8
seeds, and sweeps **every subset size** (1..N) of each base's own
imm-effectual diff loci — i.e. full exhaustion of the imm space at these loci
for every base. Disk score cache disabled, so each unique program got one real
evaluation.

| base | effectual loci | sizes | planned | unique new | cross-base dups | baseline prom | best prom |
| --- | --- | --- | --- | --- | --- | --- | --- |
| champion | 22,23,24,25 | 1–4 | 50,624 | 32,954 | 17,670 | 325.825742 | **325.825742** |
| seed0 | 22,24 | 1–2 | 224 | 236 (12 native) | 0 | 321.331871 | 321.331871 |
| seed1 | 21,23,24 | 1–3 | 3,374 | 2,262 (12 native) | 1,124 | 321.331871 | 321.331871 |
| seed2 | 22,24,25 | 1–3 | 3,374 | 3,374 | 0 | 318.831871 | 318.831871 |
| seed3 | 21,23,24,25 | 1–4 | 50,624 | 33,750 | 16,874 | 318.831871 | 318.831871 |
| seed4 | 21,23,25 | 1–3 | 3,374 | 2,250 | 1,124 | 318.831871 | 318.831871 |
| seed5 | 19,21,22 | 1–3 | 3,374 | 3,374 | 0 | 318.831871 | 318.831871 |
| seed6 | 18,20,21,22 | 1–4 | 50,624 | 33,750 | 16,874 | 318.831871 | 318.831871 |
| seed7 | 18,20,22 | 1–3 | 3,374 | 2,250 | 1,124 | 318.831871 | 318.831871 |
| **total** | | | **168,966** | **114,200** | **54,790** | | |

Arithmetic: 168,966 planned − 54,790 within-base encoding collisions = 114,176
unique combo programs, + 24 native-probe contributions = **114,200**.
Cross-base overlap = 0 (no variant built from one base ever equals one built
from another). Per-base unique counts sum to exactly 114,200.

| metric | value |
| --- | --- |
| **unique programs evaluated (single shared dedup)** | **114,200** |
| planned combos before dedup | 168,966 |
| cross-base overlap | 0 |
| native-probe contributions | 24 (seed0/seed1 only) |
| **BEST overall** | **325.8257419355 = champion (unmodified)** |
| hits >= 325.836 | **0** |
| gate-qualified hits | **0** |
| gap to task bar | 0.010258 |

**Every base's best equals its own baseline** — not one imm edit anywhere in
the space improved any base, champion included. The native
`terminal_imm_perturb_probe_variants` / `terminal_wide_imm_sweep_probe_variants`
contributed 0 new champion variants (the exhaustive sweep had already subsumed
their entire output), confirming Phase 2 was fully covered by Phase 5.

## Phase 6 — program-wide sweep + gate qualification

`_probe_imm_programwide.py` (log: `_probe_imm_programwide.log`)

Phase 5 exhausted the imm axis only at the **diff loci (18–25)**, excluding
positions 0–17 as prefix-locked. The recommendation below ("move to a
different axis") is only valid if no imm mutation *elsewhere* beats the
champion either — so this run sweeps **every position 0–26**, subset sizes
**1–3**, disk cache disabled. It also measures a metric earlier phases never
reported: **gate qualification of the top variants**.

### Which positions can an imm edit even change?

| | positions | count |
| --- | --- | --- |
| **imm-effectual** | 0, 4, 8, 10, 11, 14, 17, **22, 23, 24, 25** | **11** |
| non-effectual | 1,3,5,6,7,9,12,13,15 (ADD/XOR), 16 (SLT), 18,19,20 (SUB), 21,26 (EBREAK) | 16 |

Sixteen of 27 positions are R-type (`encode` ignores `imm`) or EBREAK, so no
sweep can touch them. Note **7 effectual positions sit outside the diff loci**
— 0, 4, 8, 10, 11, 14, 17 — i.e. real search space that Phases 1–5 never
examined. They are covered here.

| metric | value |
| --- | --- |
| planned combos (sizes 1–3, 11 effectual positions) | 463,694 |
| — size 1 (11 × 14) | 154 |
| — size 2 (C(11,2)=55 × 14²) | 10,780 |
| — size 3 (C(11,3)=165 × 14³) | 452,760 |
| **unique programs evaluated** | **332,359** |
| **BEST** | **325.8257419355 = champion (unmodified)** |
| hits >= 325.836 | **0** |
| gap to task bar | 0.010258 |

**Correction to an earlier draft of this report:** it claimed size 3 was
prohibitive at *"1,650 × 14³ ≈ 4.5M evals"*. That arithmetic was wrong —
C(11,3) is **165**, not 1,650, i.e. **452,760** evals, which ran in 73s.
Size-3 is therefore now **run** (323,584 unique at size 3), not skipped.

### Gate qualification (new metric)

Top 200 variants by promotion_raw were gate-checked:

| metric | value |
| --- | --- |
| gate-qualified among top 200 | **42** (21%) |
| **max gate-qualified prom** | **325.8257419355** (= champion, locus 22, δ −64) |
| max gate-qualified vs champion | +0.000000 |
| max gate-qualified vs task bar | 0.010258 short |
| champion itself gate-qualifies | True |
| hits >= task bar that also gate-qualify | **0** |

So most high-prom variants *fail* the held-out/side gate, and among the 42
that pass, **none improves on the champion**. Both halves of the promotion
test (prom >= 325.836 AND gate) are therefore unreachable on this axis —
program-wide, not just at the diff loci.

Phase 6 uses its own dedup set, so its 332,359 overlaps Phase 5 at loci 22–25;
it is reported separately and **not** added to the 114,200 figure (adding
per-phase counts is exactly the error retracted above).

## Totals

Per-phase counts are distinct **within** each phase only — the phases used
separate dedup sets, so summing them double-counts overlap. The authoritative
unique figure comes from Phase 5 (single shared dedup set).

| phase | evals (per-phase distinct) | forced-fresh |
| --- | --- | --- |
| 1+2 single-site (champ + 8 seeds) | 407 | — |
| 1-verify single-site (champ) | 176 | yes |
| 3 dual/triple combos (champ + 8 seeds) | 14,408 | — |
| 4 exhaustive champion space | 32,954 | yes |
| **5 comprehensive shared-dedup (champ + 8 seeds)** | **114,200 unique** | **yes** |

| authoritative measure | value |
| --- | --- |
| unique programs evaluated across the whole investigation | **114,200** |
| of which forced-fresh (disk cache disabled) | 114,200 |
| variants >= 325.836 | **0** |

Earlier drafts of this report stated "47,945 distinct / 33,130 fresh" by
summing the per-phase rows. That was an overclaim (cross-phase overlap
double-counted) and is corrected above.

## Conclusion

The imm axis at the eq-seed diff loci is **exhausted**:

1. Single-site: 0 of 407 variants reach 325.836; best = champion.
2. 2-way and 3-way combinations: 0 of 14,408; best = champion.
3. Complete champion imm space (sizes 1–4, 32,954 forced-fresh evals):
   0 variants above champion at all, let alone 325.836.
4. Seed bases never improve under any imm edit (best == their own baseline,
   and they start 4.49–6.99 below the bar).

Per task step 6, **nothing was recorded** to
`.agents/scan_floor_eq_refine_seeds.json` — no gate-qualified variant exists
at or above 325.836 on this axis.

### Why this axis cannot close the gap

promotion_raw at these loci is quantized into 4 whole-ish steps (0 / −1 / −3 /
−5) because the imm-effectual words are one `SB` and three `BLT`
displacements: nudging them only changes whether whole task checks pass. The
required +0.010258 is a fractional gain, which this axis structurally cannot
produce. Closing the gap needs a different axis (word substitution,
structural change, or a different locus set) — not imm tuning.

## Limitations / scope

- **Imm axis only**, as the task specifies. Word-replacement, crossover and
  structural operators were not swept here.
- ~~Seeds swept at subset size 1 and 2 only~~ — **resolved by Phase 5**: all 8
  seeds are now exhausted over *every* subset size of their own
  imm-effectual diff loci (sizes 1–4). No subset-size gap remains for any base.
- ~~Positions 0–17 excluded as prefix-locked~~ — **superseded by Phase 6**:
  those positions were swept program-wide anyway (subset sizes 1–3 over all
  11 imm-effectual positions 0–26). The exclusion is now moot rather than an
  untested assumption.
- Phase 6 now covers subset **sizes 1–3** program-wide (463,694 planned /
  332,359 unique, size 3 alone = 452,760 evals in 73s). Sizes 1–4 are covered
  for the 4 effectual diff loci (Phase 5). The only region left untested is
  **size 4+ program-wide**: C(11,4)=330 × 14⁴ ≈ **12.7M** evals — genuinely
  out of budget, and unreachable-in-practice given sizes 1–3 across those same
  positions produced zero variants above the champion.
- Champion row advanced 7560 → 7578 during the investigation (live daemon); the
  champion **program** stayed identical (`anchor_match=True` asserted at each
  phase start), so results remain valid for the plateau. Note the daemon also
  rewrote `.agents/scan_floor_eq_refine_seeds.json` mid-run: at cycle 7578 the
  seeds diverge from the champion only at index 26 (an `EBREAK`, hence
  non-effectual), versus indices 18–25 earlier. Phase 6 sweeps **all** positions
  0–26 regardless of the seed set, so its coverage is unaffected.

## Side-effect audit

- **No fitness or gate code modified.** `evolution/fitness_helpers.py`,
  `evolution/terminal_stagnation.py`, `evolution/mutation_helpers.py` were
  only read. (They show as pre-existing `M`/`??` in the shared checkout from
  other agents — untouched by this task.)
- **No write to the seeds file.** Both `record_scan_floor_eq_refine_seeds`
  branches were false. The file's `cycle` field moved 7560 → 7562 and its
  mtime changed, with the same 8 seeds and identical 2293-byte size
  (sha256 `d82002d225c92feeed4a87c7dc3d34af66e6773ef6d6a15bc8981fef3c49e0ef`)
  — that rewrite is the live daemon's own cycle update, not this task.
- **Read-only on the daemon**: no daemon process, port or RPC was touched.
- New files created (all untracked): 5 probe scripts, 4 logs, 4 evidence
  JSONs, this report. Pre-existing `_probe_eq28.py` / `_probe_l10_replace.py`
  untouched.

## Verification & self-review

Independent `delegate_task` review was **unavailable** in this session: the first
call timed out at 420s, the second failed with a platform error
(`name 'app_gated' is not defined`, `api_calls: 0`), and the third was refused
with "Delegation budget for this one-shot run is exhausted (2/2)". Review was
therefore performed inline, as the platform instructed.

### Self-check: 67/67 PASS
`_selfcheck_imm_sweep.py` + `_selfcheck_imm_sweep.log` — 67 assertions, 0 failures
(the final assertion is *self-verifying*: the report must state this exact tally,
counted by the script rather than typed in):

- champion prom == 325.82574193548385 and < 325.836
- `terminal_efficiency_at_scan_floor(27)` is True; progress state floor_words == 27
- `scan_floor_min_raw_gain(27)` == 0.01; bar == 325.83574193548384
- exhaustive histogram sums to 32,954; `sizes_run` sums to 32,954
- combo `per_base_tested` sums to 14,408
- all 4 histogram keys <= champion (allowing the 6-dp display rounding); exact
  `best_prom <= champion`
- seeds file: exactly 8 seeds, plateau_key [325.8257, 27], anchor == champion
- positions 0..17 identical in champion and all 8 seeds (prefix-locked claim holds)
- loci 18/19/20 confirmed non-effectual (SUB), locus 21 confirmed EBREAK
- **21 sampled imm variants: 0 above champion, 5 BELOW champion** — proves the
  scorer differentiates and is not returning a constant
- 16 sampled two-locus combos: 0 above champion

Added as the phases grew:

- Phase 5 arithmetic: `168966 - 54790 + 24 == 114200`; per-base `best_prom <= baseline`
  for all 9 bases; `max_subset_size == 4`
- Phase 6 arithmetic: `463694 == 11*14 + 55*196 + 165*2744`; `unique_programs == 332359`;
  `max_subset_size == 3`; effectual-position list `[0,4,8,10,11,14,17,22,23,24,25]`
- gate metric: `max gate-qualified gap == 0.010258`; `champion_gate_qualifies is True`;
  `0 < gate-qualified among top 200 <= 200` (42)
- **report↔evidence consistency**: the report still states 114,200 and 332,359,
  documents the Phase 6 and gate sections, retracts the old 47,945 total, and
  carries **no** superseded figure (`8,775`, `34/34`, `| **total** | **47,945**`)

### Write audit of this task's diff
- `.agents/scan_floor_eq_refine_seeds.json`: opened with `json.load` only, in
  all 6 scripts — never written.
- No write to `evolution/fitness_helpers.py`, `evolution/terminal_stagnation.py`
  or `evolution/mutation_helpers.py`.
- Only write-mode targets are this task's own `.agents/evidence/*.json`.
- All 3 `record_scan_floor_eq_refine_seeds` calls sit inside
  `if gate_ok_hits:` / `if breakthroughs:` — both proven false (hits = 0).
- `.agents/vm_score_disk_cache.json` is **daemon-managed and did change during
  this session**: `1,119,773 bytes @ Sep 16 06:30` → `1,100,015 bytes @ Sep 24
  03:42` (size shrank, consistent with `_vm_score_disk_put`'s >2048-entry
  eviction). It was **not** written by this task: Phases 4/5/6 set
  `_VM_SCORE_DISK_MEM = {}` + `_VM_SCORE_DISK_LOADED = True` to force fresh
  evals, and a flush of that in-process dict would have truncated the file to
  `{}`. The file is still ~1.1 MB, and `_vm_score_disk_flush()`
  (`terminal_stagnation.py:807`) has exactly one caller —
  `select_terminal_probes_for_batch` (line 5061) — which these probes never
  invoke (they call `terminal_imm_perturb_probe_variants` and
  `terminal_wide_imm_sweep_probe_variants` directly). So the writer is the live
  daemon, whose own probe-pool builds are the only path that flushes.
- `.agents/scan_floor_eq_refine_seeds.json` likewise remains daemon-owned:
  `plateau_key [325.8257, 27]`, 8 seeds + anchor intact, `anchor == champion`
  asserted at every phase start; its `cycle` field moved to `0` because the
  daemon re-records via `collect_l10_lift_eq_refine_bootstrap_seeds`
  (`mutation_helpers.py:3172`, which passes `cycle=0`). All six of this task's
  scripts open it with `json.load` only — none writes it.

### Tests
| run | result |
| --- | --- |
| targeted (`record_scan_floor_eq_refine` / `imm_perturb` / `wide_imm` / `scan_floor_gate` / `min_raw_gain`) | **7 passed**, 224 deselected, 50.95s |
| full: `test_scan_floor_throughput.py` + `test_archive_scan_floor_probes.py` + `test_terminal_stagnation.py` | **7 failed, 221 passed, 3 skipped**, 1225s |

All 7 failures are in `test_archive_scan_floor_probes.py` and are pre-existing
source/test drift from other agents' uncommitted edits to
`evolution/mutation_helpers.py` / `evolution/terminal_stagnation.py`:

- `lambda() got an unexpected keyword argument 'probe_material'` — test stub predates the current `load_scan_floor_eq_refine_seeds` signature
- `lambda() takes 2 positional arguments but 3 were given` — `should_rescan_terminal_efficiency_compression` gained a parameter
- `assert 41 == 42`
- `mutant[37] ^= 1` → `IndexError` (program shorter than the test assumes)
- `assert [False, True, True, True] == [False, True]` (call-count drift)
- `assert [] == [[1, 2, 9, 1048691]]`
- `assert (3, 325.03, -6) > (3, 325.5, 0)` (rank drift)

**Proof they are not mine:** re-running all 7 with every file created by this
task temporarily moved out of the checkout reproduced **6 identical failures**.
The 7th (`...growth_inject_rank_stalled...`) passed in that isolated run *and*
passes again with my files restored (1 passed in 4.25s) → order/state-dependent
flakiness. All files were restored afterwards (root 9, evidence 4, logs 6
verified present).

This task modified no source, so it cannot introduce test regressions; the 7
pre-existing failures are reported as-is rather than masked.

## Artifacts

| kind | path |
| --- | --- |
| report | `REPORT_imm_locus_sweep_t_6f4be05a.md` |
| probe (main) | `_probe_imm_locus_sweep.py` + `.log` |
| probe (fresh verify) | `_verify_imm_locus_sweep.py` + `.log` |
| probe (combos) | `_probe_imm_combo_sweep.py` + `.log` |
| probe (exhaustive) | `_probe_imm_exhaustive_champion.py` + `.log` |
| probe (inputs check) | `_probe_explore.py` |
| evidence | `.agents/evidence/t_6f4be05a_imm_locus_sweep.json` |
| evidence | `.agents/evidence/t_6f4be05a_imm_locus_verify.json` |
| evidence | `.agents/evidence/t_6f4be05a_imm_combo_sweep.json` |
| evidence | `.agents/evidence/t_6f4be05a_imm_exhaustive_champion.json` |
