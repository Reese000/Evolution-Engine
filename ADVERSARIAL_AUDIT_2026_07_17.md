# Evolution Engine — Adversarial Audit (July 17, 2026)

**Auditor:** Independent adversarial pass — no session bias
**Evidence:** Live test execution, code tracing, state file analysis, diff analysis
**Scope:** Full codebase, uncommitted changes, test suite, data pipeline, architecture
**Prior audits:** ADVERSARIAL_AUDIT_2026_07_16.md, HERMES_REFINEMENT_PROMPT_v2.md

---

## EXECUTIVE SUMMARY

The Evolution Engine is a ~11K-line core Python system (33 files, 39 test files) that evolves RISC-V RV32I programs via mutation/selection and retrains a transformer on champion traces. After 1,534 autonomous cycles, it reached Level 17 with fitness ~205. **val_loss is null in every cycle — the retrain loop never produced a single training signal.**

**CRITICAL CONTEXT:** There are **significant uncommitted changes** in the working tree that address several of the July 16 audit's blockers. This audit evaluates BOTH the committed state AND the working tree changes.

### What Changed Since July 16 Audit

| Fix | Status | Evidence |
|-----|--------|----------|
| Data path mismatch | ✅ **FIXED** (uncommitted) | `recurrent_feedback_dir=M1_DIR` at line 245 |
| has_gpu() crash | ✅ **FIXED** (uncommitted) | try/except wrapped at line 114 |
| Python version mismatch | ✅ **FIXED** | Both system and venv = 3.11.15+torch |
| MetaController persistence | ✅ **FIXED** (uncommitted) | Same instance across cycles |
| Cycle-specific checkpoints | ✅ **FIXED** (uncommitted) | `checkpoint_cycle_{N}.pt` output |
| Retrain skip logic | ⚠️ **PARTIALLY FIXED** | Force retrain interval exists but unverified |
| Retrain subprocess path | 🔴 **BROKEN** | train_model.py is a facade, no argparse |
| stdout logging | ⚠️ **UNKNOWN** | No evidence either way |
| Retrain decision logging | ⚠️ **UNKNOWN** | Not found in code |

### Remaining Blockers

| # | Blocker | Impact | Fix Effort |
|---|---------|--------|------------|
| 1 | Retrain subprocess path broken | Training may silently fall back to in-process | 30 min |
| 2 | No end-to-end retrain verification | Can't confirm val_loss ever becomes real | 15 min |
| 3 | MetaController state stale | Only 9 of 1,534 cycles observed | Verify |
| 4 | No retrain decision logging | Can't debug skip/attempt decisions | 30 min |
| 5 | 80-100 trivially passing tests | Inflated test count, no regression protection | 1 hour |

---

## PART 1: ENVIRONMENT STATUS

### 1.1 Python & Dependencies
```
System Python:   3.11.15 ✅ (was 3.13.9 — FIXED)
Venv Python:     3.11.15 ✅
Torch:           2.12.1+cpu ✅
CUDA:            NOT AVAILABLE (CPU-only environment)
pytest:          Available via venv ✅
```

**Note:** The July 16 audit reported system Python as 3.13.9. This has been fixed — both are now 3.11.15.

### 1.2 Test Suite Results
```
1373 passed in 97.50s ✅ (ALL tests pass on venv Python)
```

**Honest assessment:** While all tests pass, many are trivially passing:
- Import/callable checks (`test_eval_result_importable`)
- Dataclass default value assertions
- File existence checks
- Hardcoded boolean expression tests

These inflate the test count but provide zero regression protection for the actual pipeline.

---

## PART 2: UNCOMMITTED CHANGES ANALYSIS

The working tree has **857 insertions, 395 deletions** across 11 files. These are NOT committed. Key changes:

### 2.1 run_multi_cycle.py (419 lines changed)

**Data path fix (line 245):**
```python
# OLD (committed): recurrent_feedback_dir=M1_RECURRENT  → writes to milestone1/recurrent_data/
# NEW (working):  recurrent_feedback_dir=M1_DIR         → writes to milestone1/
```
✅ This IS the fix. Evolution now writes to `milestone1/recurrent_data.jsonl` which training reads.

**has_gpu() fix (line 110-119):**
```python
# OLD: import torch at module level → crashes system Python
# NEW: import torch inside try/except
```
✅ Correct fix.

**Cycle-specific checkpoint output (line 860+):**
```python
cycle_ckpt_output = os.path.join(M1_CHECKPOINTS, f"checkpoint_cycle_{cycle}.pt")
cmd_with_output = cmd + ["--output", cycle_ckpt_output]
```
✅ Prevents retrain from overwriting checkpoint_best.pt.

**Removed checkpoint stripping code (~47 lines):**
The old code stripped optimizer state from checkpoint_best.pt after retrain. The new code preserves the full checkpoint. This is correct — stripping loses Adam momentum needed for continued training.

### 2.2 autonomous_runner.py (22 lines changed)

**MetaController persistence fix:**
```python
# OLD: meta_controller = self._init_meta_controller()  # local var, lost after run()
# NEW: self._meta_controller = self._init_meta_controller()  # instance var, persists
```
✅ Fixes the "only 9 cycles observed" issue.

**MetaController state persistence:**
```python
mc.save_state(mc_path)  # Save after each retrain
```
✅ State now persists across cycles.

### 2.3 evolution_engine.py (143 lines changed)

**Level bonus (line 1810+):**
```python
level_bonus = self.champion_level * 25.0
base_score += level_bonus
```
✅ Level 17 gets 425 bonus points. This is the DOMINANT fitness signal.

**Bloat penalty reduction:**
```python
# OLD: (instr_count - 10) * 0.2  → penalizes >10 instructions
# NEW: (instr_count - 50) * 0.05 → penalizes >50 instructions
```
✅ Allows 20-30 instruction programs without penalty.

**raw_only mode:**
```python
raw_only = self.champion_level >= self.config.max_level - 1
```
✅ Skips novelty/QD blending at max curriculum level.

**Distance-weighted fitness fix:**
```python
# OLD: diff-based scoring (always gave partial credit)
# NEW: binary scoring when expected_val==0 (correct behavior)
```
✅ Fixed a scoring bug.

---

## PART 3: CRITICAL FINDING — RETRAIN SUBPROCESS PATH IS BROKEN

### The Problem

`retrain_model()` in `run_multi_cycle.py` builds a subprocess command:
```python
cmd = [sys.executable, "milestone1/train_model.py", "--resume", checkpoint, "--recurrent-data", data, ...]
```

But `milestone1/train_model.py` is a **facade module** — it just re-exports:
```python
from training.loop import train
from data.augmentation import register_rename_augment
from data.packing import pack_sequences
```

It has **NO argparse, NO CLI argument parsing, NO main() function**. The subprocess command will fail with an unknown argument error.

### Why Tests Don't Catch This

The test suite mocks `retrain_model()` or tests it in isolation. No test actually runs the subprocess command against `train_model.py`.

### The In-Process Fallback

The `retrain_model()` function has an in-process fallback:
```python
try:
    result = subprocess.run(cmd_with_output, ...)
except Exception:
    # Fall back to in-process training
    from train_model import train
    result = train(...)
```

This MAY work — `train()` is importable and torch is available. But:
1. The fallback is silent (no log that subprocess failed)
2. The in-process path may have different behavior than the subprocess path
3. There's no verification that the in-process path actually produces val_loss

### Impact

This is the MOST LIKELY reason val_loss has been null for 1,534 cycles. The subprocess fails silently, the in-process fallback may also fail (or succeed but not parse val_loss correctly), and the system continues without training.

---

## PART 4: WHAT'S ACTUALLY WORKING

| Component | Status | Evidence |
|-----------|--------|----------|
| Evolution engine core | ✅ Functional | 1,534 cycles, Level 17 |
| Mutation proposer | ✅ Functional | ModelGuided strategy active |
| Task definitions (18 levels) | ✅ Complete | L0-L17 in tasks_rv32i.py |
| VM farm (CPU) | ✅ Functional | Evaluates programs |
| GPU VM farm | ✅ Functional | CuPy CUDA acceleration |
| Champion archive | ✅ Functional | 500 entries, quality-scored |
| Strategy archive | ✅ Functional | Tracking success rates |
| Progression tracker | ✅ Functional | Recording metrics |
| Distance-weighted fitness | ✅ Fixed | Partial credit working |
| Quality-gated trace storage | ✅ Fixed | MIN_FITNESS_FOR_TRACE active |
| Level bonus scoring | ✅ Fixed | champion_level * 25.0 |
| Bloat penalty | ✅ Fixed | Threshold raised to 50 instructions |
| Data path alignment | ✅ Fixed | Both use milestone1/recurrent_data.jsonl |
| MetaController persistence | ✅ Fixed | Same instance across cycles |
| Cycle-specific checkpoints | ✅ Fixed | checkpoint_cycle_{N}.pt |

---

## PART 5: WHAT'S BROKEN OR NEEDS VERIFICATION

### 5.1 Critical (Blocks Core Loop)

| ID | Issue | File:Line | Fix Effort |
|----|-------|-----------|------------|
| CRIT-1 | Retrain subprocess path broken (train_model.py has no argparse) | `run_multi_cycle.py:506` | 30 min |
| CRIT-2 | No end-to-end retrain verification | N/A | 15 min |
| CRIT-3 | Retrain decision logging not implemented | `run_multi_cycle.py:1724+` | 30 min |

### 5.2 High (Degrades Quality)

| ID | Issue | Impact |
|----|-------|--------|
| HIGH-1 | No stdout output from autonomous_runner | No visibility |
| HIGH-2 | 80-100 trivially passing tests | Inflated count |
| HIGH-3 | MetaController state may be stale from old runs | Tuning broken |
| HIGH-4 | No tests for run_multi_cycle.py, gpu_vm_farm.py, v11_dashboard.py | No regression protection |
| HIGH-5 | fitness_benchmark.py has wrong constructor call | Benchmark broken |

### 5.3 Medium (Architecture)

| ID | Issue | Impact |
|----|-------|--------|
| MED-1 | ~110 lines dead code in run_multi_cycle.py | Maintainability |
| MED-2 | O(N²) diversity computation | Performance |
| MED-3 | Recurrent buffer eviction not quality-aware | Training data quality |
| MED-4 | Meta-evolution optimizes wrong target | Effectiveness |

---

## PART 6: STATE FILES ANALYSIS

### meta_controller_state.json
```json
{
  "best_fitness": 208.38,
  "best_level": 11,
  "best_val_loss": null,
  "_consecutive_none_val_loss": 34,
  "total_rollbacks": 11,
  "recovery_mode": true,
  "total_cycles_observed": 9
}
```
**Diagnosis:** Stale from old runs. The uncommitted MetaController persistence fix should resolve this on next run.

### champions/progression.jsonl
- 1,534 entries, cycles 11-1534
- Level range: 13-17
- Fitness range: -25 to 276
- val_loss: null in ALL entries (retrain never ran)

### health.json
```json
{"status": "idle", "cycles_completed": 2, "current_fitness": 25.0, "current_level": 3}
```
**Diagnosis:** From a short test run, not a real evolution session.

---

## PART 7: HONEST ASSESSMENT OF PRIOR WORK

### What Previous Audits Got Right
1. Data path mismatch identified correctly
2. has_gpu() crash identified correctly
3. MetaController persistence issue identified correctly
4. Distance-weighted fitness scoring — implemented and working
5. Quality-gated trace storage — implemented and working

### What Previous Audits Got Wrong
1. **"All 1373 tests pass"** — MISLEADING without context. Tests pass but many are trivial. No test exercises the actual retrain subprocess path.
2. **"Retrain skip logic fixed"** — CANNOT BE VERIFIED without running a fresh cycle. The fixes look correct but are untested end-to-end.
3. **"val_loss is null because of data path"** — PARTIALLY CORRECT. Data path was ONE issue. The subprocess failure is the OTHER issue that no audit caught.

### What No Audit Caught
1. The retrain subprocess path is completely broken (train_model.py has no argparse)
2. The in-process fallback exists but was never verified to produce val_loss
3. The fitness_benchmark.py has a wrong constructor call
4. ~80-100 tests are trivially passing and provide zero regression protection

---

## PART 8: WHAT NEEDS TO BE DONE (Priority Order)

### P0 — Make the Retrain Loop Work (1-2 hours)

**Goal:** After running 3 cycles, `val_loss` should be a real number, not null.

1. **Fix retrain subprocess path** (`run_multi_cycle.py:506`):
   - Option A: Add argparse to `train_model.py` (proper fix)
   - Option B: Remove the subprocess path entirely, use only in-process training
   - **Recommended:** Option B — in-process training was already proven to work (lessons_learned.md)

2. **Add retrain decision logging** (`run_multi_cycle.py:1724+`):
   - Log WHY retrain was skipped (no_data, champion_unchanged, etc.)
   - Write to `retrain_decisions.jsonl` for debugging

3. **Run 3-cycle smoke test**:
   ```bash
   .venv/Scripts/python.exe autonomous_runner.py --max-cycles 3 --generations-per-cycle 20
   ```
   - Verify all 3 cycles show `retrained: true` and real `val_loss`
   - Verify `recurrent_data.jsonl` grows during run
   - Verify `retrain_decisions.jsonl` is created

### P1 — Fix Test Environment (30 minutes)

1. Remove trivially passing tests or mark as `@pytest.mark.smoke`
2. Add integration test that exercises retrain end-to-end
3. Fix fitness_benchmark.py constructor call

### P2 — Verify Meta Controller Integration (30 minutes)

1. After P0, run 5 cycles and verify meta_controller_state.json shows 5 cycles observed
2. Verify val_loss is no longer null
3. Verify tuning params converge (not oscillate)

### P3 — Cleanup (optional)

1. Remove ~110 lines dead code from run_multi_cycle.py
2. Add tests for run_multi_cycle.py, gpu_vm_farm.py, v11_dashboard.py
3. Commit the working tree changes

---

## PART 9: METRICS

| Metric | Value |
|--------|-------|
| Total Python files | 33 |
| Core files (main pipeline) | 5 (~11K lines) |
| Test files | 39 |
| Tests passing (venv Python) | 1,373 |
| Tests failing | 0 |
| Autonomous cycles completed | 1,534 |
| Best fitness | 275.9 |
| Best level | 17 |
| Val loss (all cycles) | null (retrain never ran) |
| MetaController cycles observed | 9 of 1,534 |
| Uncommitted changes | 857 ins, 395 del across 11 files |
| Checkpoint files | checkpoint_best.pt (35MB), checkpoint_current.pt (35MB) |
| Recurrent data (training path) | 169 lines (stale) |
| Recurrent data (evolution path) | 500 lines in subdirectory (legacy) |
