# Evolution Engine — Hermes Refinement Prompt
# Generated: 2026-07-16 | Based on independent adversarial audit

---

## Context

You are working on the Evolution Engine project at `C:\Users\reese\Projects\Evolution Engine\`.

This is a Python autonomous code evolution system that evolves RISC-V RV32I programs through mutation/selection cycles and retrains a transformer on champion traces. After 1,534 cycles it reached Level 17 with fitness ~208. **But the retrain loop — the entire point of the system — is completely broken.** `val_loss` is `null` in every cycle. The system evolved blind with zero model improvement from training.

There are THREE independent blocking issues. None were fully fixed by prior Hermes passes despite multiple audit rounds.

**Read `ADVERSARIAL_AUDIT_2026_07_16.md` first** for the full evidence. This prompt specifies the exact fixes needed.

---

## TASK 1: Fix Data Path Mismatch (CRITICAL — 1 line change)

**Problem:** Evolution writes champion traces to `milestone1/recurrent_data/recurrent_data.jsonl` (subdirectory) but training reads from `milestone1/recurrent_data.jsonl` (root). Training never sees new data.

**File:** `run_multi_cycle.py`

**Current code (line 246):**
```python
recurrent_feedback_dir=M1_RECURRENT,
```

Where `M1_RECURRENT = os.path.join(PROJECT_ROOT, "milestone1", "recurrent_data")` (the subdirectory).

**Fix:** Change to:
```python
recurrent_feedback_dir=MILESTONE1_DIR,
```

Where `MILESTONE1_DIR = os.path.join(PROJECT_ROOT, "milestone1")` (already defined in the file).

This makes `evolution_engine.py:_try_recurrent_feedback()` write to `milestone1/recurrent_data.jsonl` — matching what training reads.

**Verification:** After the fix, grep `evolution_engine.py` for `recur_dir` and confirm the resulting path matches `milestone1/recurrent_data.jsonl`.

---

## TASK 2: Fix Retrain Skip Logic (CRITICAL — ~30 lines)

**Problem:** Retrain never executes. `val_loss` is null in all 1,534 cycles. The skip logic is too aggressive and has a deadlock condition.

**File:** `run_multi_cycle.py`, lines ~1724-1790

**Root causes:**
1. `retrain_result = {"retrained": False, "reason": "skipped"}` is the default (line 1724)
2. The `last_retrain_epochs > 0` guard prevents skip evaluation if the first retrain fails
3. `force_retrain_interval` check was inside the skip block (now moved out per prior fix, but counter never advances)
4. No logging of why retrain was skipped

**Required changes:**

### 2a. Remove the `last_retrain_epochs > 0` guard
At line ~1750, change:
```python
if cycle >= 3 and last_retrain_epochs > 0 and cycles_since_last_retrain >= min_cycles_between_retrains:
```
To:
```python
if cycle >= 2 and cycles_since_last_retrain >= min_cycles_between_retrains:
```

### 2b. Make force_retrain_interval unconditional
Ensure the `force_retrain_interval` check (line ~1758) is evaluated EVERY cycle, not conditionally. The counter `cycles_since_last_retrain` must increment on EVERY cycle, not just skip cycles.

Move the increment to the top of the cycle loop:
```python
cycles_since_last_retrain += 1  # Count every cycle
```

### 2c. Default force_retrain_interval to 2
Change the argparse default from 3 to 2:
```python
parser.add_argument("--force-retrain-interval", type=int, default=2, ...)
```

### 2d. Add retrain decision logging
After the retrain skip/attempt decision, write to `retrain_decisions.jsonl`:
```python
import json, os
decision_path = os.path.join(PROJECT_ROOT, "retrain_decisions.jsonl")
decision = {
    "cycle": cycle,
    "decision": "attempted" if should_retrain else "skipped",
    "reason": retrain_result.get("reason", "unknown"),
    "fitness": current_fitness,
    "level": current_level,
    "traces": evo_result.get("champion_traces", 0),
    "cycles_since_last": cycles_since_last_retrain,
}
with open(decision_path, "a") as f:
    f.write(json.dumps(decision) + "\n")
```

---

## TASK 3: Fix torch Import Crash (HIGH — 5 lines)

**Problem:** `run_multi_cycle.py:114` has `has_gpu()` that imports torch at module level. On system Python (3.13.9 without torch), this crashes the entire orchestrator.

**File:** `run_multi_cycle.py`, function `has_gpu()`

**Fix:** Wrap in try/except:
```python
def has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
```

---

## TASK 4: Fix Stdout Logging (HIGH — 10 lines)

**Problem:** Both `autonomous_runner.py` and `meta_controller.py` produce zero stdout output. Logger sends to file only.

**File:** `logging_config.py`

**Fix:** Add a StreamHandler to the root logger:
```python
import sys

# After existing file handler setup, add:
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
logging.getLogger().addHandler(console_handler)
```

---

## TASK 5: Fix Meta Controller Integration (HIGH — 20 lines)

**Problem:** MetaController only observed 9 of 1,534 cycles. `val_loss` is always null. Tuning params oscillate in compound_stall→recovery loop.

**File:** `autonomous_runner.py`

**Fix:** Ensure the autonomous runner:
1. Passes each cycle's metrics to MetaController (including retrain results)
2. Feeds the actual `val_loss` from retrain (not null) to `mc.update_cycle()`
3. Does NOT create a fresh MetaController per cycle (verify `self._meta_controller` is used consistently)

Also in `run_multi_cycle.py`, ensure `retrain_result["val_loss"]` is passed to the meta controller:
```python
mc.update_cycle(
    cycle=cycle,
    fitness=current_fitness,
    level=current_level,
    val_loss=retrain_result.get("val_loss"),  # Must be a real number, not None
    diversity=diversity
)
```

---

## TASK 6: Verify the Fix (REQUIRED)

After making all changes:

### 6a. Run non-torch tests
```bash
python -m pytest tests/ --ignore=tests/test_batch_prefix.py --ignore=tests/test_model_migrator.py --ignore=tests/test_scale_integration.py --ignore=tests/test_warm_start.py -v
```
All 1314+ tests must pass.

### 6b. Run 3-cycle smoke test
```bash
.venv/Scripts/python.exe run_multi_cycle.py --cycles 3 --gens 20
```

### 6c. Verify the following after the smoke test:
1. `retrain_decisions.jsonl` exists and has 3 entries
2. `milestone1/recurrent_data.jsonl` has more than 169 lines (grew during run)
3. Each cycle shows `retrained: true` in stdout output
4. `val_loss` in progression.jsonl is a real number (not null)
5. No torch import crashes

### 6d. Run full test suite with venv Python
```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```
All tests including torch-dependent ones should pass.

---

## CONSTRAINTS

1. **Do NOT modify** `evolution_engine.py` fitness scoring — it's working correctly
2. **Do NOT modify** `tasks_rv32i.py` — task definitions are complete
3. **Do NOT modify** `milestone1/rv32i_model.py` — model architecture is correct
4. **Do NOT remove** existing tests — only add skips for torch-dependent ones when torch unavailable
5. **Do NOT change** the config/engine_config.json defaults — the current values are reasonable
6. **DO verify** every change with actual test execution, not just code inspection
7. **DO commit** each fix as a separate atomic commit with descriptive message

---

## EXPECTED OUTCOME

After this prompt is executed:
- The retrain loop works: `val_loss` is a real number after each cycle
- Evolution traces reach the training pipeline
- The orchestrator doesn't crash on system Python
- Console output is visible during runs
- All tests pass (with appropriate torch skips on system Python)
- A 3-cycle smoke test demonstrates the full evolve→retrain loop working
