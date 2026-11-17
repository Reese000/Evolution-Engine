# Review: 5-edit diff in tests/unit/test_scan_floor_throughput.py

**Task:** t_5610b3e5 (deploy gate — step 2 test run)
**Verdict: NOT APPROVED — suite still red.**

## Observed test state (run from repo root, .venv-win/Scripts/python.exe)

```
tests\unit\test_scan_floor_throughput.py ...............................
..........F................................................
FAILED tests/unit/test_scan_floor_throughput.py::test_eq_refine_late_inject_reuses_merged_pool_within_cycle
tests\unit\test_scan_floor_throughput.py:1332: assert calls["locus"] == 1  →  0 == 1
1 failed, 89 passed in 8.75s
```

Edits 1, 3, 4, 5 are green. Edit 2 (`test_eq_refine_late_inject_reuses_merged_pool_within_cycle`,
file line 1252) is incomplete and fails in **both** possible modes:

### Root cause A — file-order cache poisoning (full-file run, fails at :1332)
- `test_eq_refine_late_inject_skips_perturb_when_warm_disk` (line 605) runs earlier with the same
  champion `[10, 20, 30, 40, 0x00100073]`, `warm_eq_refine_disk_seeds_sufficient` mocked `True`,
  and reaches the cache write at `evolution/mutation_helpers.py:3846`, storing
  `_eq_refine_late_merged_cache[((pr₅, 5), True, 8, False)]`.
- Test 2 computes the identical key (`mutation_helpers.py:3705-3710`; `_eq_refine_plateau_key`
  at :2788 uses the unmocked real fitness → same value in both tests), gets a **hit** at :3711,
  never probes locus → `calls["locus"] == 0`.
- `clear_scan_floor_inject_caches()` (`mutation_helpers.py:35`) deliberately retains this dict
  (`mutation_helpers.py:48`).
- **Proof:** `pytest -q -k "skips_perturb_when_warm_disk or reuses_merged_pool_within_cycle"`
  → test 2 fails at :1332 with `0 == 1`.

### Root cause B — `EE_EQ_REFINE_WARM_SKIP_PERTURB` still active (isolated run, fails at :1333)
- With a cold cache the edited `EE_EQ_REFINE_WARM_SKIP_LOCUS=0` works (locus probes once), but
  `skip_perturb` at `mutation_helpers.py:3770-3779` is `True` (warm mocked True + env default
  `"1"` + sufficient mocked True) → `perturb_budget = 0` → perturb never called →
  `assert calls["perturb"] == 1` fails.
- **Proof:** isolated run fails at :1333; isolated run with `EE_EQ_REFINE_WARM_SKIP_PERTURB=0`
  **passes** (1 passed in 4.40s).

No configuration passes as-is (file order deterministic; no randomization plugin).

## (a) Weakened oracles — none, but edit 2 does not hold
No assert statement was changed by any of the 5 edits:
- **Edit 1** (`:74`, comment `:72-73`): still asserts `payload["seeds"][0] == [1,2,88,4,...]` (`:83`).
  The env override only disables the 20-word size gate at `mutation_helpers.py:3193`, orthogonal
  to seed ranking. ✔ (Note: no other test covers `EE_EQ_REFINE_PERSIST_MIN_WORDS` — the default-20
  guard has no direct test.)
- **Edit 2** (`:1273`, comment `:1271-1272`): asserts unchanged (`:1332-1333`) but both fail. ✘
- **Edit 3** (`:2270-2273`, comment `:2268-2269`): still asserts double-call cache reuse
  `calls["n"] == 1` (`:2290-2292`). Gap-3→gap-2 is correct: production skips `gap>=3` sources at
  `mutation_helpers.py:3129` when shrink intermediates are mocked `[]` — the old gap-3 parent was
  never pair-dropped (stale setup, not weaker oracle). ✔
- **Edit 4** (`:2339-2342`, comment `:2337-2338`): still asserts per-anchor isolation
  (`:2361-2364`); plateau mocked *identical* for both champs (`:2333-2336`) so only anchor
  identity separates cache entries. Gap change same rationale as edit 3. ✔
- **Edit 5** (`:2182-2187`, comment `:2179-2181`): still asserts membership **and** ordering
  (`:2192-2194`). Patching `_curriculum_l10_cached` bypasses the leaking module-level
  `_curriculum_l10_cycle_cache` (`mutation_helpers.py:70-82`) — strictly stronger than patching
  `curriculum_verify_mean` — and matches the file's dominant convention (8+ sibling tests patch
  `_curriculum_l10_cached`). ✔

## (b) Env scoping — clean
Both overrides use the `monkeypatch` fixture (`:74`, `:1273`) → auto-restored at teardown, no
leakage. `tests/conftest.py` sets no `EE_EQ_*` vars (grep-verified). The cross-test leakage
channel in this file is the module-level cache dict, not the environment.

## (c) Risk to siblings `:1532` / `:1589` from `EE_EQ_REFINE_WARM_SKIP_LOCUS=0` — none
- **Env:** `monkeypatch.setenv` is torn down before `:1532` runs; siblings see default `"1"` and
  exercise the real skip branch.
- **Cache key:** test 2 and both siblings compute the *same* key `((pr₅, 5), True, 8, False)`
  (same champ, warm mocked True, seed_cap 8, force_late False, `_eq_refine_plateau_key` unmocked
  in all three; the plateau mocks at `:1497/:1517` are restored before `:1532`). Collision would
  occur — **but** in file-order runs test 2 gets a *hit* (written by `:605`) and never reaches the
  cache-write lines (`mutation_helpers.py:3845-3846` sit in the miss `else` branch) → edit 2 adds
  **no new** cross-test state.
- **Direction:** siblings assert *zero* calls (`:1586`, `:1656`); a cache hit makes them pass
  more easily, never fail. So they cannot break — the pre-existing problem is the opposite: in
  full-file runs they already pass **vacuously** (cache hit short-circuits before the warm-skip
  branch they claim to test). `clear_scan_floor_inject_caches()` at `:1538` doesn't clear it.
- **If test 2 is fixed to take the miss path**, it will write its merged pool under that shared
  key; siblings still pass (0-call asserts) but remain vacuous — recommend they also force a miss
  (patch `load_eq_refine_late_merged_cache`→`None`, precedent `:622-625`).

## (d) Style / hermeticity
- ✔ All 5 edits carry `#` comments at the edit site (`:72-73`, `:1271-1272`, `:2179-2181`,
  `:2268-2269`, `:2337-2338`); `setenv` matches the file's 14 existing `monkeypatch.setenv`
  calls; mock targets match sibling conventions.
- ✘ **Hermeticity gap (test 2, pre-existing in `:605/:1125/:1203`):** on a cache miss, production
  calls `save_eq_refine_late_merged_cache` (`mutation_helpers.py:3847`) writing the **real repo
  file** `.agents/scan_floor_eq_refine_late_merged_cache.json`; test 2 patches neither
  `scan_floor_eq_refine_late_merged_cache_path` (only `:588` does) nor sets
  `EE_EQ_REFINE_LATE_MERGED_DISK=0`. Demonstrated: isolated run of test 2 overwrote that file
  with `"cycle": 7` + test 2's exact seeds. (Load side never feeds back for this champ: save
  stores anchor-derived plateau `[[325.8257,27],…]` at `:3461-3466` vs caller champ-derived key
  `((pr₅),5)` at `:3705` — always mismatch — but the write mutates workspace state.)

## Required fix for edit 2 (both items proven necessary)
In `test_eq_refine_late_inject_reuses_merged_pool_within_cycle` (`:1252`), alongside `:1273`:
1. `monkeypatch.setenv("EE_EQ_REFINE_WARM_SKIP_PERTURB", "0")` — fixes isolated mode (proven).
2. Force a cache miss at test start: `_eq_refine_late_merged_cache.clear()` (precedent:
   `_eq_refine_bootstrapped.clear()` `:119`, `_l10_lift_eq_bootstrap_cache.clear()` `:2250`)
   **plus** patch `evolution.mutation_helpers.load_eq_refine_late_merged_cache` → `None`
   (precedent `:622-625`) — fixes file-order mode (proven: `:605` occupancy causes the :1332 fail).
3. Recommended: also `monkeypatch.setenv("EE_EQ_REFINE_LATE_MERGED_DISK", "0")` (or patch the
   cache path to `tmp_path`) so the test stops writing `.agents/…late_merged_cache.json`.

## Production code
`evolution/mutation_helpers.py` read-only during this review (5034-insertion diff is pre-existing
uncommitted work). No files edited by this review; only side effect =
`.agents/scan_floor_eq_refine_late_merged_cache.json` writes caused by running the suite itself
(tests `:605/:1125/:1203` write on every run — pre-existing, out of scope; current on-disk
content `cycle: 1` = test `:1203`'s write from the validation run; the probe deleted its own
scratch copy and left the reviewed test file byte-identical, `grep FIXPROBE` = 0 matches).

## Fix validation (empirical, run via `_fixprobe_scratch.py`, exit 0: ALL EXPECTATIONS MET)
The prescribed 3-part fix was applied to a **scratch copy** of the test file
(`test_scan_floor_fixprobe.py`, auto-deleted; original test file untouched) and the full file run:

| Run | Result |
|---|---|
| Original full file, run #1 | `1 failed, 89 passed in 8.75s` (failure `:1332`) |
| Original full file, run #2 (determinism re-run) | `1 failed, 89 passed in 8.15s` — deterministic |
| Fix copy, full file (90 tests, file order intact) | **`90 passed in 8.02s`**, exit 0 |
| `.agents/…late_merged_cache.json` after fix run (clean slate) | last writer `cycle: 1` (test `:1203`), **not** `cycle: 7` → test `:1252` writes no repo state with `EE_EQ_REFINE_LATE_MERGED_DISK=0` |

The fix components validated together: (1) `EE_EQ_REFINE_WARM_SKIP_PERTURB=0`,
(2) `_eq_refine_late_merged_cache.clear()` + `load_eq_refine_late_merged_cache`→`None`,
(3) `EE_EQ_REFINE_LATE_MERGED_DISK=0`. Siblings `:1532/:1589` remained green with the cache
forcing in place, confirming the (c) analysis.

## Resolution (observed after sibling subagent edit at 02:48:39)
- Sibling `20260924_010836_3c895a` applied the prescribed fix to test `:1252`; re-read confirms
  content at `:1271-1287`: explanatory `#` comment, `_eq_refine_late_merged_cache.clear()`,
  `EE_EQ_REFINE_WARM_SKIP_LOCUS=0`, `EE_EQ_REFINE_WARM_SKIP_PERTURB=0`,
  `load_eq_refine_late_merged_cache`→`None`, `save_eq_refine_late_merged_cache`→`None`
  (function-equivalent to the recommended `EE_EQ_REFINE_LATE_MERGED_DISK=0` for this test).
  Asserts `:1346-1347` (`locus==1`, `perturb==1`) **unchanged** — no weakened oracle.
  File delta 2805→2819 lines (+14) is exactly this block; no other test modified.
- **Fresh full run after the fix: `90 passed in 8.32s`, exit 0** → deploy step 2 gate **GREEN**.
  Suite verdict upgraded from NOT APPROVED to **APPROVED** (all 5 original edits plus the
  completion fix hold under full file order; production files untouched).
- Not attempted from this review session (parent's deploy steps 3-5): daemon restart
  (`scripts/restart_evolution_daemon.ps1`), 3-cycle timing verification, status update —
  impactful side effects requiring in-flight-cycle coordination; board transition likewise
  requires the unfenced dispatcher worker.


