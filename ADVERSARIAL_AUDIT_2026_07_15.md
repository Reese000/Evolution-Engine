# Evolution Engine — Adversarial Audit Report (v2)
**Date:** 2026-07-15 | **Scope:** Full codebase, test suite, prior audit follow-up
**Auditor:** Fresh adversarial pass — no prior session bias

---

## Executive Summary

The Evolution Engine is a ~22K-line Python system that evolves RISC-V RV32I programs via mutation/selection and retrains a transformer on champion traces. The system has been through multiple audit cycles. Some critical issues from the prior audit (2026-07-14) have been fixed. However, **the core goal — compounding self-improvement via the evolve→retrain loop — is still not working.**

**Bottom line:** The system evolves programs well (L17 reached) but the retrain feedback loop remains unreliable. The test suite claims 1373+ tests but **cannot actually run** in this environment due to missing `torch`. The prior audit's HERMES_REFINEMENT_PROMPT.md is stale and references bugs that have already been fixed. A new, accurate prompt is needed.

---

## PART 0: Test Suite Reality Check 🔴

### 0.1 Tests Cannot Run

```
ERROR tests/test_batch_prefix.py — ModuleNotFoundError: No module named 'torch'
ERROR tests/test_model_migrator.py — ModuleNotFoundError: No module named 'torch'
ERROR tests/test_warm_start.py — ModuleNotFoundError: No module named 'torch'
ERROR tests/test_scale_integration.py — ModuleNotFoundError: No module named 'torch'
FAILED tests/test_checkpoint_manager_integration.py (7 tests) — ModuleNotFoundError: No module named 'torch'
```

**Result:** 1314 passed, 7 failed, 4 files couldn't even collect. The test suite is running on **Python 3.13.9 (system Python)** which does NOT have torch installed. The project's `.venv` has torch but pytest is not using it.

**Impact:** Every test result reported in prior audits and CLAUDE.md ("1373 tests pass") is **unverifiable in this environment**. The tests that pass are the ones that don't need torch. The tests that need torch (training, checkpoint, warm start, model migration) all fail.

### 0.2 Trivial Tests Still Present

Prior audit identified ~80-100 trivially passing tests:
- Import/callable checks (`test_eval_result_importable`)
- File existence/string matching on Dockerfile/scripts
- Dataclass default value checks
- Hardcoded boolean expression tests

**Status: STILL PRESENT.** These inflate the test count and provide zero regression protection.

---

## PART 1: What Was Fixed Since Prior Audit ✅

### 1.1 CRITICAL-2: Binary Fitness → Distance-Weighted Partial Credit ✅ FIXED
**File:** `evolution_engine.py:1765+`
```python
error_ratio = abs(actual_val - expected_val) / max(1, abs(expected_val))
program_scores[reg_idx] = 1.0 / (1.0 + error_ratio)
```
**Evidence:** `grep` confirms partial credit scoring at lines 1765, 1789, 1884, 2138, 2226.

### 1.2 CRITICAL-3: Training Signal Contamination → Quality-Gated Storage ✅ FIXED
**File:** `evolution_engine.py:_try_recurrent_feedback()`
- `MIN_FITNESS_FOR_TRACE` threshold gates trace storage
- Deep stall exception (>= 30 gens) allows low-quality traces
- Duplicate detection via SHA256

### 1.3 CRITICAL-4: Data Pipeline Format Mismatch → FIXED (partially)
**File:** `milestone1/build_recurrent_dataset.py`
- Now outputs `champion_program` format matching `data/loading.py` expectations
- Writes to `milestone1/recurrent_data.jsonl` (correct path)
- **However:** The `_try_recurrent_feedback()` method in `evolution_engine.py` writes traces to `milestone1/recurrent_data/recurrent_data.jsonl` (subdirectory), NOT `milestone1/recurrent_data.jsonl` (root). This path mismatch STILL EXISTS in the autonomous evolution path.

### 1.4 CRITICAL-5: Stateless MetaController → FIXED ✅
**File:** `autonomous_runner.py:776+`
- Now uses `self._meta_controller` (persistent instance) instead of creating new ones
- Persist/restore state after each retrain

### 1.5 CRITICAL-6: Checkpoint Separation → FIXED ✅
**File:** `run_multi_cycle.py:863-866`
- Retrain writes to `checkpoint_cycle_{N}.pt` (separate from `checkpoint_best.pt`)
- Gate acceptance copies to best

### 1.6 barrier_tuner.py → CREATED ✅
- Full implementation with `BarrierType`, `CycleMetrics`, `BarrierTuner` classes
- 15,488 bytes, functional detection logic

---

## PART 2: What's Still Broken 🔴

### STILL-BROKEN-1: Retrain Skip Logic Still Aggressive
**File:** `run_multi_cycle.py:1750-1780`

The skip logic was improved but still has problems:

```python
min_cycles_between_retrains = 2  # never skip retrain in consecutive cycles
if cycle >= 3 and last_retrain_epochs > 0 and cycles_since_last_retrain >= min_cycles_between_retrains:
    fitness_threshold = max(abs(last_champion_fitness) * 0.05, 0.01)
    fitness_same = abs(current_fitness - last_champion_fitness) < fitness_threshold
    level_same = current_level == last_champion_level
    words_same = current_words == last_champion_words
    no_new_traces = evo_result.get("champion_traces", 0) == 0
    if fitness_same and level_same and words_same and no_new_traces:
        # ... skip retrain
```

**Problem 1:** `no_new_traces = evo_result.get("champion_traces", 0) == 0` — if the evolution engine generates any traces at all (even crash traces), this is False and retrain won't be skipped. But if `champion_traces` is reported as 0 (which happens when the key is missing or set to 0), retrain IS skipped even though the engine ran.

**Problem 2:** The `force_retrain_interval` check is INSIDE the `if fitness_same and level_same and words_same and no_new_traces:` block (line 1777). It's unreachable when any of those conditions is False. If fitness changed but level/words didn't, force retrain never fires.

**Problem 3:** Inside the force retrain block, `cycles_since_last_retrain += 1` is executed BEFORE the `>= force_retrain_interval` check. But this increment only runs when ALL skip conditions are met. So the counter only advances when the system wants to skip — it doesn't count cycles where retrain actually happened.

**Problem 4:** `last_retrain_epochs` starts at 0 and is only set after a successful retrain. If the first retrain fails (no data, permission error), `last_retrain_epochs` stays 0, and `cycle >= 3 and last_retrain_epochs > 0` is never True — meaning the skip logic is never even evaluated. The default `retrain_result = {"retrained": False, "reason": "skipped"}` persists.

**Evidence:** The `retrained: false, reason: "skipped"` appears in cycles 7-9 of the most recent run despite having 193-207 champion traces available.

### STILL-BROKEN-2: No Retrain Decision Logging
**File:** `run_multi_cycle.py`

The HERMES_REFINEMENT_PROMPT.md requested `retrain_decisions.jsonl` — **this was never implemented**. Grep finds zero matches for `retrain_decisions`. There's no systematic log of why retrain was skipped vs attempted. The "unknown" reason problem from the prior audit remains unsolved.

### STILL-BROKEN-3: meta_controller_state.json Is Empty
```
{"cycles": []}
```
Despite 1534+ autonomous cycles and 2 multi-cycle runs, the state file is empty. The MetaController state persistence is not working correctly — either it's writing to a different path or the save is failing silently.

### STILL-BROKEN-4: Dead Code in run_multi_cycle.py
**~110 lines of dead code identified:**
- `_cleanup_stale_checkpoints()` (line 2324-2330): Body is `pass` — no-op
- `_run_training_visibility_diagnostics()` (lines 2743-2828): 85-line function never called
- BarrierTuner diagnostic wiring (lines 2231-2256): Dead because `diagnostics` is hardcoded to `None`

### STILL-BROKEN-5: No End-to-End Retrain Test
No test exercises the actual retrain subprocess path. All retrain-related tests are unit tests with mocks. A 3-cycle smoke test has never been run (or at least not documented).

### STILL-BROKEN-6: health.json Is Stale
```json
{
  "status": "idle",
  "last_cycle": "2026-07-14T19:04:29.488404",
  "cycles_completed": 2,
  "current_fitness": 25.0,
  "current_level": 3
}
```
Only 2 cycles completed, fitness 25.0, level 3. This is from a short test run, not a real evolution run. The health system works but there's no active run to monitor.

---

## PART 3: Prior Audit Claims That Were WRONG or MISLEADING

### 3.1 "1373 Tests Pass" — MISLEADING
The test count is inflated by ~80-100 trivial tests. The torch-dependent tests (4 files + 7 tests) cannot run in the current environment. Actual verifiable passing tests: ~1314.

### 3.2 "CRITICAL-4: Data Pipeline Format Mismatch" — PARTIALLY FIXED, STILL BROKEN
The `build_recurrent_dataset.py` was fixed to output `champion_program` format. But the `_try_recurrent_feedback()` method in `evolution_engine.py` (the actual runtime path) writes to `milestone1/recurrent_data/recurrent_data.jsonl`, not `milestone1/recurrent_data.jsonl`. The training pipeline reads from `milestone1/recurrent_data.jsonl`. **The runtime data path is still wrong.**

### 3.3 "All Prior Audit Bugs Fixed" — VERIFIED TRUE
The 10 bugs listed in the prior audit (or True debug leftover, level_regression enum, parse_gradian norms, etc.) have all been confirmed fixed via grep.

### 3.4 HERMES_REFINEMENT_PROMPT.md — STALE
The refinement prompt references issues that have already been fixed:
- Task 0.1 (build_recurrent_dataset format) — FIXED
- Task 0.2 (path mismatch in build_recurrent_dataset) — FIXED  
- Task 2.1 (binary fitness) — FIXED
- Task 2.2 (quality-gated trace storage) — FIXED
- Task 5.1 (stateless MetaController) — FIXED
- Task 7.1 (barrier_tuner.py) — CREATED

But it misses issues that still exist:
- Retrain skip logic still too aggressive
- No retrain decision logging
- meta_controller_state.json empty
- Runtime data path mismatch (_try_recurrent_feedback writes to subdirectory)

---

## PART 4: Architectural Limitations (Still Present)

| ID | Issue | Impact | Status |
|----|-------|--------|--------|
| ARCH-1 | O(N²) diversity computation | Performance | Still present |
| ARCH-2 | Multi-level regression check quadratic | Performance | Still present |
| ARCH-3 | Novelty/QD selection bias at elite stage | Correctness | Still present |
| ARCH-4 | Meta-evolution optimizes wrong target | Effectiveness | Still present |
| ARCH-5 | Recurrent buffer eviction not quality-aware | Training quality | Still present |
| ARCH-6 | Task instance overfitting | Generalization | Partially mitigated (refresh every 25 gens) |
| ARCH-7 | GPU fallback masks kernel bugs | Debugging | Still present |
| ARCH-8 | ~80-100 trivially passing tests | Test quality | Still present |
| ARCH-9 | Zero tests for run_multi_cycle.py | Test coverage | Still present |
| ARCH-10 | Zero tests for gpu_vm_farm.py | Test coverage | Still present |
| ARCH-11 | Zero tests for v11_dashboard.py | Test coverage | Still present |

---

## PART 5: What Actually Needs To Be Done (Priority Order)

### P0 — Fix the Runtime Data Path (NEW)
**The `_try_recurrent_feedback()` method writes to `milestone1/recurrent_data/recurrent_data.jsonl` but training reads from `milestone1/recurrent_data.jsonl`.** This is the single most critical fix — without it, the autonomous evolution loop never feeds champion traces to the model.

1. Change `evolution_engine.py:_try_recurrent_feedback()` to write to `milestone1/recurrent_data.jsonl` (matching training's expected path)
2. OR change `run_multi_cycle.py` to pass `--recurrent-data milestone1/recurrent_data/recurrent_data.jsonl` to the training subprocess
3. Verify by checking that after a cycle, `milestone1/recurrent_data.jsonl` has new entries

### P1 — Fix Retrain Skip Logic
1. Remove the `last_retrain_epochs > 0` guard — always evaluate skip conditions after cycle 1
2. Make `force_retrain_interval` check independent of other skip conditions
3. Default `force_retrain_interval=3` (force retrain every 3 cycles regardless)
4. Add comprehensive retrain decision logging to `retrain_decisions.jsonl`

### P2 — Fix Test Environment
1. Add a `conftest.py` that skips torch-dependent tests when torch is not available
2. OR install torch in the system Python
3. Remove trivially passing tests or mark them as smoke tests (not regression)

### P3 — Clean Up Dead Code
1. Remove `_cleanup_stale_checkpoints()` body (`pass`) or implement it
2. Remove `_run_training_visibility_diagnostics()` (85 lines, never called)
3. Remove dead BarrierTuner diagnostic wiring

### P4 — Architecture Improvements
1. Sampled diversity (O(N²) → O(N))
2. Quality-aware buffer eviction
3. Elite selection at max level with raw_only=True
4. meta_controller_state.json persistence fix

### P5 — Verification
1. Run 3-cycle smoke test with `--gens 20`
2. Verify all 3 cycles show `retrained: true`
3. Verify `retrain_decisions.jsonl` is created
4. Verify checkpoint files exist
5. Verify `recurrent_data.jsonl` grows during run

---

## PART 6: Code Metrics

| File | Lines | Status |
|------|-------|--------|
| evolution_engine.py | 4,061 | Core — functional, needs data path fix |
| run_multi_cycle.py | 2,732 | Orchestrator — retrain skip logic still aggressive |
| mutation_proposer.py | 1,707 | Functional |
| tasks_rv32i.py | 1,439 | 18 levels, complete |
| autonomous_runner.py | 1,299 | Functional, MetaController fixed |
| meta_controller.py | 1,261 | Functional, state persistence broken |
| gpu_vm_farm.py | 1,219 | Functional |
| barrier_tuner.py | ~400 | NEW — created this session |
| meta_evolution.py | 950 | Suboptimal target |
| prefix_generator.py | 929 | Functional |
| **Total** | **~16,000** | Core files only |

---

## PART 7: Honest Assessment

### What Hermes Got Right
- Distance-weighted fitness scoring works correctly
- Quality-gated trace storage is properly implemented
- Checkpoint separation (cycle_N vs best) is correct
- MetaController persistence in autonomous_runner.py is fixed
- barrier_tuner.py was created and is functional

### What Hermes Got Wrong or Didn't Finish
- The runtime data path mismatch (`_try_recurrent_feedback` → subdirectory) was never fixed
- Retrain skip logic was patched but not fundamentally fixed
- No retrain decision logging was implemented
- meta_controller_state.json persistence is broken (empty)
- Dead code was never removed
- No end-to-end retrain test was written
- The HERMES_REFINEMENT_PROMPT.md is stale and references fixed issues

### The Real Blocker
The system cannot achieve compounding self-improvement because:
1. Champion traces go to the wrong file path (subdirectory vs root)
2. Training reads from the root path (no new data)
3. Retrain skips aggressively (59% of cycles in prior data)
4. No visibility into why retrain was skipped

**Fix the data path (P0) and the retrain skip logic (P1), and the system might actually start learning from its champions.**
