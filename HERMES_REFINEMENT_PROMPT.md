# Evolution Engine — Hermes Refinement Prompt (v2)
## Feed this entire document into a fresh Hermes session for the Evolution Engine project.

---

You are agent-evolution. Your task is to fix the Evolution Engine's remaining critical issues so the system achieves compounding self-improvement via the evolve→retrain loop.

**Read `ADVERSARIAL_AUDIT_2026_07_15.md` first** — it documents the current state with evidence. This prompt specifies exactly what to implement, based on verified findings (not assumptions).

**Working directory:** `C:\Users\reese\Projects\Evolution Engine`
**Python env:** System Python 3.13.9 (no torch) OR `.venv/Scripts/python.exe` (has torch+numpy)
**Test command:** `python -m pytest tests/ --ignore=tests/test_batch_prefix.py --ignore=tests/test_model_migrator.py --ignore=tests/test_warm_start.py --ignore=tests/test_scale_integration.py --ignore=tests/test_checkpoint_manager_integration.py -x --tb=short -q`

---

## Phase 0: Fix the Runtime Data Path (MOST CRITICAL)

The `_try_recurrent_feedback()` method in `evolution_engine.py` writes champion traces to the WRONG path. Training never sees the data.

### Task 0.1: Fix the Write Path
**File:** `evolution_engine.py`, method `_try_recurrent_feedback()` (around line 3608)
**Current code writes to:**
```python
recur_dir = self.config.recurrent_feedback_dir
if not recur_dir:
    recur_dir = os.path.join(MILESTONE1_DIR, "recurrent_data")
jsonl_path = os.path.join(recur_dir, "recurrent_data.jsonl")
# Results in: milestone1/recurrent_data/recurrent_data.jsonl
```
**Training reads from:**
```python
# run_multi_cycle.py passes this to the training subprocess:
M1_RECURRENT_DATA = os.path.join(MILESTONE1_DIR, "recurrent_data.jsonl")
# Results in: milestone1/recurrent_data.jsonl
```
**Fix:** Change the default `recur_dir` fallback to write directly to `MILESTONE1_DIR` (not a `recurrent_data` subdirectory):
```python
if not recur_dir:
    recur_dir = MILESTONE1_DIR  # NOT os.path.join(MILESTONE1_DIR, "recurrent_data")
```
OR set `self.config.recurrent_feedback_dir` to the correct path in the config.

### Task 0.2: Verify End-to-End Data Flow
After the fix, verify by running a single cycle and checking:
1. `milestone1/recurrent_data.jsonl` has new entries with `champion_program` key
2. The entries have `fitness`, `task_score`, `level`, `generation`, `save_reason` fields
3. `milestone1/data/loading.py` can parse these entries (test by importing and calling `load_champion_traces()`)

---

## Phase 1: Fix Retrain Skip Logic

The retrain skip logic is too aggressive. Even when traces exist, retrain gets skipped.

### Task 1.1: Remove the `last_retrain_epochs > 0` Guard
**File:** `run_multi_cycle.py` lines 1750-1780
**Problem:** The condition `cycle >= 3 and last_retrain_epochs > 0` means skip logic is NEVER evaluated if the first retrain fails. `last_retrain_epochs` stays 0 forever.
**Fix:** Change to `cycle >= 2` (evaluate skip after first cycle). The `last_retrain_epochs > 0` check is redundant — if no retrain happened, `cycles_since_last_retrain` will be large and force retrain anyway.

### Task 1.2: Make force_retrain_interval Independent
**File:** `run_multi_cycle.py` lines 1780-1810
**Problem:** `force_retrain_interval` only activates inside the `if fitness_same and level_same and words_same and no_new_traces` block. It's unreachable when any condition is False.
**Fix:** Move `force_retrain_interval` check OUTSIDE the inner `if` block:
```python
# Force retrain after N cycles regardless of champion state
if args.force_retrain_interval > 0 and cycles_since_last_retrain >= args.force_retrain_interval:
    should_retrain = True
    cycles_since_last_retrain = 0
    log(f"  Cycle {cycle}: Force retrain after {args.force_retrain_interval} cycles")
# THEN check champion-unchanged skip (only if force_retrain didn't fire)
elif cycle >= 2 and cycles_since_last_retrain >= min_cycles_between_retrains:
    # ... existing skip logic
```

### Task 1.3: Add Retrain Decision Logging
**File:** `run_multi_cycle.py`
**Create:** `retrain_decisions.jsonl` at the project root
**Implementation:**
- After every retrain decision (attempted, skipped, failed), append a JSON line:
```python
{"cycle": N, "decision": "attempted|skipped|failed", "reason": "...", "timestamp": "ISO", "fitness": F, "level": L, "traces": T}
```
- Create a helper function `_log_retrain_decision(cycle, decision, reason, **kwargs)` that appends to this file
- Call it at every decision point (skip reasons, retrain attempt, retrain success/failure)

---

## Phase 2: Fix meta_controller_state.json Persistence

The state file is empty despite 1534+ cycles.

### Task 2.1: Diagnose and Fix State Saving
**File:** `meta_controller.py`
**Check:**
1. Is `save_state()` actually called? Search for `save_state` calls in `run_multi_cycle.py` and `autonomous_runner.py`
2. Is the path correct? Check what `META_CONTROLLER_STATE` resolves to vs where `save_state()` writes
3. Is there a silent exception swallowing the save?

**Fix:** Ensure `save_state()` is called after every cycle in both `run_multi_cycle.py` AND `autonomous_runner.py`. Add error logging if save fails.

---

## Phase 3: Clean Up Dead Code

### Task 3.1: Remove Dead Code from run_multi_cycle.py
- Delete `_cleanup_stale_checkpoints()` (body is `pass`) and its call site
- Delete `_run_training_visibility_diagnostics()` (85 lines, never called)
- Delete dead BarrierTuner diagnostic wiring that depends on hardcoded `None`

### Task 3.2: Verify No Functional Code Is Removed
After deletion, run `python -m pytest tests/ --ignore=tests/test_batch_prefix.py --ignore=tests/test_model_migrator.py --ignore=tests/test_warm_start.py --ignore=tests/test_scale_integration.py --ignore=tests/test_checkpoint_manager_integration.py -x --tb=short -q` and confirm all tests still pass.

---

## Phase 4: Add Skip Logic to Torch-Dependent Tests

### Task 4.1: Mark Tests That Require Torch
**Files:** `tests/test_batch_prefix.py`, `tests/test_model_migrator.py`, `tests/test_warm_start.py`, `tests/test_scale_integration.py`, `tests/test_checkpoint_manager_integration.py`
**Fix:** Add to the top of each file:
```python
import pytest
torch = pytest.importorskip("torch", reason="Requires torch")
```
This way they're skipped (not errored) when torch is absent, and the test suite runs clean.

---

## Phase 5: End-to-End Verification

After all fixes, run:

### Test Suite
```bash
python -m pytest tests/ -x --tb=short -q
```
Expected: All tests pass (torch tests skipped cleanly).

### 3-Cycle Smoke Test
```bash
python run_multi_cycle.py --cycles 3 --gens 20 --force-retrain-interval 3
```
Verify:
1. All 3 cycles complete without errors
2. `retrain_decisions.jsonl` exists with 3+ entries
3. At least 2 of 3 cycles show `retrained: true`
4. `milestone1/recurrent_data.jsonl` grows during the run
5. No "Operation not permitted" errors
6. `checkpoint_cycle_1.pt`, `checkpoint_cycle_2.pt`, `checkpoint_cycle_3.pt` exist
7. `meta_controller_state.json` is NOT empty after the run

---

## Implementation Rules

1. **Read before writing:** Always read the full function before modifying
2. **Preserve all existing tests:** Run pytest after every change
3. **No new dependencies:** Use only stdlib + torch + numpy
4. **Log everything:** Every decision point should log what happened and why
5. **Verify before claiming done:** Run the smoke test and report actual output
6. **Don't re-fix what's already fixed:** Distance-weighted fitness, quality-gated storage, checkpoint separation, MetaController persistence, barrier_tuner.py — these are all DONE. Focus on the remaining issues.
