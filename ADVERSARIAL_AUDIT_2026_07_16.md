# Evolution Engine — Adversarial Audit (July 16, 2026)

**Auditor:** Independent adversarial pass — no prior session bias
**Evidence:** Live test execution, file inspection, code tracing, state file analysis
**Scope:** Full codebase, test suite, runtime state, data pipeline, architecture

---

## EXECUTIVE SUMMARY

The Evolution Engine is a ~22K-line Python system that evolves RISC-V RV32I programs via mutation/selection and claims to retrain a transformer on champion traces. After 1,534 autonomous cycles, it reached Level 17 with fitness ~205. **But the retrain loop — the entire point of the system — is completely broken.** `val_loss` is `null` in every single cycle record. The system evolved blind for 1,534 cycles with zero model improvement from training.

**The core finding:** There are THREE independent blocking issues that together make the self-improvement loop non-functional. None were fully fixed by prior Hermes passes despite multiple audit rounds.

| # | Blocker | Impact | Prior Audit Said | Reality |
|---|---------|--------|------------------|---------|
| 1 | Data path mismatch | Training never sees new traces | "Partially fixed" | **Still broken** — evolution writes to subdirectory, training reads from root |
| 2 | Retrain never runs | Zero model improvement | "Fixed skip logic" | **Still broken** — val_loss is null in all 1,534 cycles |
| 3 | Test env split | Tests can't verify torch code | "Needs torch install" | **Worse** — tests run on Python 3.13.9, code needs Python 3.11.15+torch |

---

## PART 1: TEST SUITE REALITY CHECK

### 1.1 Python Version Mismatch (CRITICAL)

```
System Python (pytest uses this):  Python 3.13.9 — NO torch
Venv Python (code needs this):     Python 3.11.15 — torch 2.12.1+cpu
```

**Evidence:**
```bash
$ python --version          # 3.13.9 (system, no torch)
$ .venv/Scripts/python.exe --version  # 3.11.15 (venv, has torch)
$ .venv/Scripts/python.exe -c 'import torch; print(torch.__version__)'  # 2.12.1+cpu
```

**Impact:** Every test result reported in prior audits ("1373 tests pass") was run on the WRONG Python. The torch-dependent tests (training, checkpoint, model migration, warm start, scale integration) could never pass on the system Python.

### 1.2 Actual Test Results

```
collected 1321 items (after ignoring 4 torch-dependent files)
  1314 passed
    7 failed (all in test_checkpoint_manager_integration.py — torch import)
    4 files couldn't collect (torch import)
```

> **⚠ Historical snapshot** (July 16, 2026): The test suite has since grown to 8900+ tests. These counts reflect the suite at the time of this audit.

**What passes:** Unit tests for non-torch modules (evolution engine logic, task definitions, strategy archive, etc.)
**What fails:** All tests touching the training pipeline, checkpoint save/load, model migration

### 1.3 Trivially Passing Tests (Still Present)

Prior audits identified ~80-100 trivial tests that provide zero regression protection:
- Import/callable checks (`test_eval_result_importable`)
- Dataclass default value assertions
- File existence checks on scripts
- Hardcoded boolean expression tests

**Status: UNCHANGED from prior audits.** These inflate the test count.

---

## PART 2: THE THREE BLOCKING ISSUES

### BLOCKER 1: Data Path Mismatch 🔴

**The training pipeline can never see traces produced by evolution.**

| Component | Writes to | Reads from |
|-----------|-----------|------------|
| `evolution_engine.py:_try_recurrent_feedback()` | `milestone1/recurrent_data/recurrent_data.jsonl` | — |
| `run_multi_cycle.py` training subprocess | — | `milestone1/recurrent_data.jsonl` |
| `milestone1/data/loading.py` | — | `milestone1/recurrent_data.jsonl` |

**Code trace:**
```python
# evolution_engine.py:3627
recur_dir = self.config.recurrent_feedback_dir  # = "milestone1/recurrent_data/"
jsonl_path = os.path.join(recur_dir, "recurrent_data.jsonl")
# Result: milestone1/recurrent_data/recurrent_data.jsonl  ← WRONG

# run_multi_cycle.py:45-46
M1_RECURRENT = os.path.join(PROJECT_ROOT, "milestone1", "recurrent_data")  # legacy dir
M1_RECURRENT_DATA = os.path.join(M1_DIR, "recurrent_data.jsonl")  # ← training reads this
```

**Evidence on disk:**
```
milestone1/recurrent_data.jsonl              — 169 lines (old data, from build_recurrent_dataset.py)
milestone1/recurrent_data/recurrent_data.jsonl — 500 lines (evolution traces, never read by training)
```

**The fix:** `run_multi_cycle.py:246` passes `recurrent_feedback_dir=M1_RECURRENT` (the subdirectory) to the engine. Change it to pass `M1_RECURRENT_DATA`'s parent (i.e., `MILESTONE1_DIR`), so evolution writes to `milestone1/recurrent_data.jsonl`.

### BLOCKER 2: Retrain Never Executes 🔴

**val_loss is `null` in every single one of 1,534 cycles.**

```json
// champions/progression.jsonl — last 5 cycles
{"cycle": 1530, "level": 17, "fitness": 198.18, "val_loss": null, ...}
{"cycle": 1531, "level": 17, "fitness": 171.29, "val_loss": null, ...}
{"cycle": 1532, "level": 17, "fitness": 168.96, "val_loss": null, ...}
{"cycle": 1533, "level": 17, "fitness": 205.07, "val_loss": null, ...}
{"cycle": 1534, "level": 17, "fitness": 167.66, "val_loss": null, ...}
```

**Root cause chain:**
1. `run_multi_cycle.py:1724` initializes `retrain_result = {"retrained": False, "reason": "skipped"}`
2. The skip logic at line 1750+ checks `cycle >= 3 and last_retrain_epochs > 0` — but if the first retrain fails (e.g., torch crash, no data), `last_retrain_epochs` stays 0 and the condition is never True
3. Even when skip logic is evaluated, `no_new_traces = evo_result.get("champion_traces", 0) == 0` — if the key is missing or 0, retrain is skipped
4. The `force_retrain_interval` check was moved outside the skip block (per prior audit fix), but `cycles_since_last_retrain` never advances because retrain never runs

**Evidence from meta_controller_state.json:**
```json
{
  "_consecutive_none_val_loss": 34,
  "_rollback_count": 11,
  "_state_ring": ["recovery", "compound_stall", "compound_stall", ...]
}
```

The meta controller has never received a real val_loss — it's been operating on `null` values for 34+ cycles, causing repeated rollbacks.

### BLOCKER 3: Autonomous Runner Produces No Output 🔴

```bash
$ python autonomous_runner.py --dry-run
# (empty output)

$ python meta_controller.py
# (empty output — only has a _demo() function)
```

Both entry points log to file, not stdout. The logger configuration in `logging_config.py` likely sends to a file handler only. This means:
- No visibility into what the system is doing
- No way to verify it's working without tailing log files
- Dry-run mode provides no feedback

---

## PART 3: WHAT'S ACTUALLY WORKING ✅

| Component | Status | Evidence |
|-----------|--------|----------|
| Evolution engine core | ✅ Functional | 1,534 cycles, Level 17 reached |
| Mutation proposer | ✅ Functional | ModelGuided strategy producing candidates |
| Task definitions (18 levels) | ✅ Complete | L0-L17 defined in tasks_rv32i.py |
| VM farm (CPU) | ✅ Functional | 87 lines, evaluates programs |
| GPU VM farm | ✅ Functional | 1,219 lines, CuPy CUDA acceleration |
| Champion archive | ✅ Functional | 500 entries, quality-scored |
| Strategy archive | ✅ Functional | Tracking strategy success rates |
| Progression tracker | ✅ Functional | Recording cycle metrics |
| Distance-weighted fitness | ✅ Fixed | Partial credit scoring working |
| Quality-gated trace storage | ✅ Fixed | MIN_FITNESS_FOR_TRACE threshold active |
| Checkpoint separation | ✅ Fixed | cycle_N vs best checkpoint files |
| MetaController state persistence | ✅ Fixed | State file now has real data |
| Barrier tuner | ✅ Created | Functional detection logic |

---

## PART 4: WHAT'S BROKEN OR INCOMPLETE

### 4.1 Critical (Blocks Core Loop)

| ID | Issue | File:Line | Fix Effort |
|----|-------|-----------|------------|
| CRIT-1 | Data path mismatch — evolution → subdirectory, training → root | `run_multi_cycle.py:246` | 1 line |
| CRIT-2 | Retrain never executes — skip logic too aggressive, first-fail guard | `run_multi_cycle.py:1724-1790` | ~30 lines |
| CRIT-3 | No stdout output — logger config sends to file only | `logging_config.py` | ~10 lines |
| CRIT-4 | `has_gpu()` imports torch at module level — crashes system Python | `run_multi_cycle.py:114` | Wrap in try/except |

### 4.2 High (Degrades Quality)

| ID | Issue | Impact |
|----|-------|--------|
| HIGH-1 | No retrain decision logging | Can't debug why retrain was skipped |
| HIGH-2 | Test env mismatch (Python 3.13 vs 3.11) | torch tests never pass in CI |
| HIGH-3 | 80-100 trivially passing tests | Inflated test count, no regression protection |
| HIGH-4 | meta_controller stuck in compound_stall→recovery loop | Tuning params oscillate, never converge |
| HIGH-5 | health.json stale (2 cycles, fitness 25) | Monitoring broken |

### 4.3 Medium (Architecture)

| ID | Issue | Impact |
|----|-------|--------|
| MED-1 | O(N²) diversity computation | Performance at scale |
| MED-2 | Recurrent buffer eviction not quality-aware | Training data quality |
| MED-3 | Meta-evolution optimizes wrong target | Effectiveness |
| MED-4 | Dead code in run_multi_cycle.py (~110 lines) | Maintainability |
| MED-5 | Zero tests for run_multi_cycle.py | No regression protection for orchestrator |
| MED-6 | Zero tests for gpu_vm_farm.py | No GPU code verification |
| MED-7 | Zero tests for v11_dashboard.py | No dashboard verification |

---

## PART 5: STATE FILES ANALYSIS

### meta_controller_state.json
```json
{
  "best_fitness": 208.38,
  "best_level": 11,
  "best_val_loss": null,          // ← NEVER SET
  "_consecutive_none_val_loss": 34, // ← 34 cycles with no loss data
  "_rollback_count": 11,          // ← 11 rollbacks due to "regression"
  "recovery_mode": true,          // ← stuck in recovery
  "_compound_stall_counter": 7,   // ← 7 consecutive stalls
  "_fitness_momentum": -89.53,    // ← negative momentum
  "total_cycles_observed": 9      // ← only 9 cycles fed to meta controller
}
```

**Diagnosis:** The meta controller has only observed 9 cycles despite 1,534 running. The autonomous runner creates a fresh MetaController per cycle (the prior audit claimed this was fixed — it wasn't, or the fix regressed). The tuning history shows the same 3 parameters being set repeatedly with no convergence.

### health.json
```json
{
  "status": "idle",
  "cycles_completed": 2,
  "current_fitness": 25.0,
  "current_level": 3
}
```

**Diagnosis:** Written during a short test run, not a real evolution session. The health system works but has no active data.

### champions/progression.jsonl
- 500+ entries showing cycles 1530-1534
- All at Level 17, fitness varying 167-205
- `val_loss: null` in every entry
- Archive size capped at 500

### cycle_16/ logs
- Shows generation-level output with ModelGuided strategy active
- Stall counter incrementing (stall=2 through stall=7+)
- Recurrent feedback being "saved" (2000+ entries) but to the wrong path

---

## PART 6: HONEST ASSESSMENT OF PRIOR AUDITS

### What Prior Audits Got Right
1. Distance-weighted fitness scoring — implemented and working
2. Quality-gated trace storage — implemented and working
3. Checkpoint separation — implemented and working
4. MetaController persistence — state file now has data
5. barrier_tuner.py — created and functional

### What Prior Audits Got Wrong
1. **"CRITICAL-4: Data Pipeline Format Mismatch — FIXED"** — WRONG. The format was fixed but the path is still wrong. Evolution writes to subdirectory, training reads from root.
2. **"Retrain skip logic fixed"** — PARTIALLY. The `last_retrain_epochs > 0` guard was removed, but retrain still never runs because the first retrain fails and the counter never advances.
3. **"meta_controller_state.json persistence fixed"** — YES, but the controller only observes 9 of 1,534 cycles.
4. **"All 1373 tests pass"** — MISLEADING. 1314 pass on system Python (no torch). 7 fail. 4 files can't collect.
5. **"No retrain decision logging"** — STILL NOT IMPLEMENTED despite being requested.

### What No Audit Caught
1. The Python version split (3.13.9 system vs 3.11.15 venv) means CI/CD tests can never verify torch-dependent code
2. `has_gpu()` at `run_multi_cycle.py:114` imports torch at module level, crashing the entire orchestrator on system Python
3. The meta controller only observed 9 of 1,534 cycles — the autonomous runner doesn't feed it properly
4. `val_loss: null` across ALL 1,534 cycles was never flagged by any audit

---

## PART 7: WHAT NEEDS TO BE DONE (Priority Order)

### P0 — Make the Retrain Loop Work (1-2 hours)

**Goal:** After running 3 cycles, `val_loss` should be a real number, not null.

1. **Fix data path** (`run_multi_cycle.py:246`):
   - Change `recurrent_feedback_dir=M1_RECURRENT` to `recurrent_feedback_dir=MILESTONE1_DIR`
   - This makes evolution write to `milestone1/recurrent_data.jsonl` (what training reads)

2. **Fix retrain skip logic** (`run_multi_cycle.py:1724-1790`):
   - Remove the `last_retrain_epochs > 0` guard entirely
   - Default `force_retrain_interval=2` (force retrain every 2 cycles minimum)
   - Add `retrain_decisions.jsonl` logging with reason for skip/attempt

3. **Fix torch import crash** (`run_multi_cycle.py:114`):
   - Wrap `has_gpu()` in try/except so the orchestrator doesn't crash on system Python

4. **Fix stdout logging** (`logging_config.py`):
   - Add a StreamHandler to the root logger so output appears on console

### P1 — Fix Test Environment (30 minutes)

1. Add `conftest.py` marker to skip torch-dependent tests when torch unavailable
2. OR create a `requirements-dev.txt` that installs torch for the test environment
3. Remove trivially passing tests or mark as `@pytest.mark.smoke`

### P2 — Fix Meta Controller Integration (1 hour)

1. Ensure autonomous_runner feeds ALL cycles to MetaController (not just 9)
2. Fix the compound_stall→recovery oscillation (tuning params keep resetting)
3. Implement `retrain_decisions.jsonl` logging

### P3 — Verification (30 minutes)

1. Run 3-cycle smoke test: `python run_multi_cycle.py --cycles 3 --gens 20`
2. Verify all 3 cycles show `retrained: true` and real `val_loss`
3. Verify `recurrent_data.jsonl` grows during run
4. Verify `retrain_decisions.jsonl` is created
5. Run full test suite with venv Python: `.venv/Scripts/python.exe -m pytest tests/ -v`

### P4 — Cleanup (optional)

1. Remove ~110 lines dead code from run_multi_cycle.py
2. Add tests for run_multi_cycle.py, gpu_vm_farm.py, v11_dashboard.py
3. Clean up trivially passing tests

---

## PART 8: METRICS

| Metric | Value |
|--------|-------|
| Total Python files | 33 |
| Core files (main pipeline) | 10 (~14K lines) |
| Test files | 39 |
| Tests passing (system Python) | 1,314 |
| Tests failing | 7 (torch-dependent) |
| Tests can't collect | 4 files (torch import) |
| Autonomous cycles completed | 1,534 |
| Best fitness | 208.38 |
| Best level | 17 |
| Val loss (all cycles) | null (retrain never ran) |
| MetaController cycles observed | 9 of 1,534 |
| Rollbacks | 11 |
| Champion archive entries | 500 |
| Recurrent data (training path) | 169 lines (stale) |
| Recurrent data (evolution path) | 500 lines (never read) |
