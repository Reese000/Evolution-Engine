# Adversarial review of t_6f4be05a imm-locus-sweep claim

**Verdict: APPROVE** (with two required wording corrections — see REJECTED).

Reviewed claim: "no imm mutation of the 27-word champion at word positions 18–25
can raise `compact_l30_promotion_raw_fitness` above 325.836; 32,954 forced-fresh
evals over every subset (size 1..4) of the 4 imm-effectual loci x all 14 wide
deltas found only {325.825742, 324.825742, 322.825742, 320.825742}; handoff =
'exhausted', nothing recorded to `.agents/scan_floor_eq_refine_seeds.json`."

Review scripts (new, this review only): `_adversarial_review_imm.py`,
`_adversarial_full_enum.py`, `_adversarial_collision_check.py`.
No project source, evidence, or seeds file was modified by this review.

## VERIFIED (independently reproduced)

1. **Champion prom = 325.82574193548385** — cache-bypassed fresh recompute,
   run twice, deterministic; equals jsonl last-row `promotion_raw_fitness`
   (daemon cycle 7563 at review time); `< 325.836`.
2. **Bar arithmetic** — `terminal_efficiency_at_scan_floor(27)=True`
   (progress_controller_state.json: `terminal_efficiency_scan_floor_words=27`,
   `terminal_efficiency_scan_floor_heldout=0.9901315789473684 >= 0.99-1e-6`);
   `scan_floor_min_raw_gain(27)=0.01` (no `EE_SCAN_FLOOR_MIN_RAW_GAIN` override);
   `bar = 325.83574193548384`; gap to 325.836 = `0.010258064516165177`.
3. **Exhaustion sample + adversarial extension** — 32 fresh set-delta variants
   at loci 22–25 (incl. ±64 on locus 23): 0 above champion, 9 below
   (e.g. locus 23 delta 64 → `324.82574193548385`), none ≥ 325.836 → scorer
   differentiates, not constant. Additionally 477 fresh evals on deltas the
   probe NEVER tested (224 out-of-set: ±3,5,6,7,9…±2047 across all 4 loci;
   129-value full-range scan of locus 23, −4000..4000 step 61): **0 above
   champion**; only values seen {324.825742, 325.825742}.
4. **Report arithmetic self-consistency** — histogram 16,499+12,900+3,285+270 =
   32,954 ✓; size table planned 50,624 ✓ / tested 32,954 ✓; combo per-base sum
   = 14,408 ✓; totals 47,945 ✓ and forced-fresh 33,130 ✓.
5. **No side effects** — source mtimes all predate the probe window
   (01:18–01:42): `fitness_helpers.py` Sep 14 07:51, `mutation_proposer.py`
   Sep 14 01:43, `terminal_stagnation.py` Sep 24 01:00:55,
   `mutation_helpers.py` Sep 24 01:01:01 — untouched after. Seeds file: exactly
   8 seeds, `plateau_key [325.8257, 27]`, 2293 bytes; `cycle` field advanced by
   the live daemon (expected, not the probe's doing).
6. **Logical soundness** —
   - Positions 0–17 identical in champion and all 8 seeds (prefix-lock holds).
   - SUB loci 18/19/20: re-encode with any imm delta returns the identical
     word → genuinely non-effectual.
   - EBREAK locus 21: all 14 wide deltas → identical word; skipping is safe
     regardless of the project's `_is_ebreak` convention.
   - **Dedup/None do not shrink coverage**: independent encoding-only
     enumeration of the full size-1..4 x 14-delta space gives union =
     32,955 incl. champion = **32,954 non-champion distinct programs, 0 encode
     failures** — byte-exact match to the probe's tested count. Every reachable
     program was scored.
   - "planned > distinct" mechanism confirmed: delta **+1 at BLT loci
     23/24/25 is a no-op** (regenerates champion; 3x) and **−1 ≡ −2** there
     (B-type bit-0 truncation; 3 collisions): 56 − 3 − 3 = 50 ✓. Phase-3's
     "champion size2 added 0" reconciles: raw distinct {51, 987, 8,763} →
     cumulative-new {50, 936, 7,776}, s1∪s2 ⊆ s3.
   - Report side-data re-derived exactly: gate heldout
     `0.9901315789473684`, side `(0.8889, 0.8889, 0.8, 0.8906, 0.875, 0.875,
     0.875, 0.8906, 0.875)`, champion L10 `0.8`, all seed L10 `0.825`, seed
     proms `321.3318709677419` / `318.8318709677419`.

## REJECTED / OVERCLAIMS

- **"strict local maximum" is FALSE** (report lines 145–147, and claim text):
  16,499 of 32,954 neighbors TIE the champion exactly (23/32 in my sample).
  Correct wording: *non-strict* (weak) local maximum / maximal element with
  plateaus. The "nothing above" conclusion is unaffected.
- **Scope overclaim**: "NO immediate-value mutation … can raise" / "the
  champion's imm search space … is *exactly* … 14 deltas" (report lines 59–60).
  Exhaustiveness covers only the 14 power-of-2 deltas (|δ| ≤ 64) — the union of
  the project's two imm operators' delta sets (`immediate_perturbation` uses
  ±{1,2,4,8} ⊂ that set). Arbitrary representable imm values were not
  exhaustively swept; the operator-scoping caveat behind "exactly" is missing.
  Mitigating: 477 of my own out-of-set evals found zero counterexamples.
- **UNVERIFIED (minor)**: report line 18 "cycle 7560 at load" — earliest
  surviving artifact shows 7561. Report line 205 sha256 `d82002d2…` — current
  hash is `ddd67294a215b5e6e124e60befcaf123272e49c3f18e210ae3302ab9a72941c2`
  (daemon cycle 7562→7563, size still exactly 2293) — historical hash
  uncheckable, mechanism corroborated.
- **Cosmetic**: probe log calls SUB "imm not encodable" when encode succeeds and
  returns the same word (no-op); phase-3 champion row "8,762 (size3 + size2)"
  — size-3 alone covers all 8,762 (size-2 adds 0).

## Verdict

**APPROVE.** Every load-bearing number reproduces; the sweep is provably
complete over its defined space; "exhausted / record nothing" is the correct
handoff. Apply the two wording corrections (drop "strict"; scope "NO imm
mutation" to the project's imm-operator delta sets) before reusing the claim
verbatim.
