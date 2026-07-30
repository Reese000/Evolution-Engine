# Evolution Engine — Hermes Refinement Prompt
# Generated: 2026-07-17 | Based on independent adversarial audit

---

## Context

You are working on the Evolution Engine project at `C:\Users\reese\Projects\Evolution Engine\`.

This is a Python autonomous code evolution system that evolves RISC-V RV32I programs through mutation/selection cycles and retrains a transformer on champion traces. After 1,534 cycles it reached Level 17 with fitness ~205. **But the retrain loop — the entire point of the system — never produced a single val_loss value.** The system evolved blind with zero model improvement from training.

**Read `ADVERSARIAL_AUDIT_2026_07_17.md` first** for full evidence. This prompt specifies exact fixes needed.

**IMPORTANT:** There are significant UNCOMMITTED changes in the working tree that fix several prior blockers (data path, has_gpu, MetaController persistence, cycle-specific checkpoints). These fixes look correct but are UNTESTED end-to-end. Do NOT re-implement them — verify they work.

---

## TASK 1: Fix Retrain Subprocess Path (CRITICAL — root cause of null val_loss)

**Problem:** `retrain_model()` in `run_multi_cycle.py` builds a subprocess command to run `milestone1/train_model.py`. But `train_model.py` is a facade module with NO argparse — it just re-exports `from training.loop import train`. The subprocess fails silently, and the in-process fallback may also fail.

**Evidence:**
```bash
$ head -30 milestone1/train_model.py
# Just re-exports — no argparse, no main(), no CLI handling
from training.loop import train
from data.augmentation import register_rename_augment
from data.packing import pack_sequences
```

**Fix (Option B — recommended):** Remove the subprocess path entirely. The in-process `train()` function was already proven to work (see `lessons_learned.md` entry "In-Process Training Fix for Hermes SIGTERM Kill"). Change `retrain_model()` to always use in-process training:

In `run_multi_cycle.py`, around line 850+, replace the subprocess call with:
```python
# Always use in-process training (subprocess path is broken — train_model.py has no argparse)
from train_model import train as _train_fn
import io, contextlib

# Redirect stdout/stderr to log file
log_path = os.path.join(PROJECT_ROOT, f"cycle_{cycle}", "retrain.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

old_stdout, old_stderr = sys.stdout, sys.stderr
with open(log_path, "w") as log_f:
    sys.stdout = log_f
    sys.stderr = log_f
    try:
        train_result = _train_fn(
            resume=checkpoint_path,
            recurrent_data=recurrent_data,
            model_size=model_size,
            lr=lr,
            output_path=cycle_ckpt_output,
            # ... other params from retrain_meta_params
        )
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
```

**Verify:** After the fix, run:
```bash
cd "C:\Users\reese\Projects\Evolution Engine"
.venv/Scripts/python.exe -c "
from run_multi_cycle import retrain_model
result = retrain_model(cycle=999, checkpoint_path='milestone1/checkpoints/checkpoint_current.pt')
print(result)
"
```
Expected: `{'retrained': True, 'val_loss': <float>, ...}` — NOT `{'retrained': False, 'reason': ...}`

---

## TASK 2: Add Retrain Decision Logging (HIGH — debugging essential)

**Problem:** When retrain is skipped, there's no log of WHY. This makes it impossible to debug the pipeline.

**File:** `run_multi_cycle.py`, around line 1724+

**Fix:** Add a `retrain_decisions.jsonl` log. After every retrain decision (skip or attempt), append:
```python
import json as _json

decision_record = {
    "cycle": cycle,
    "timestamp": time.time(),
    "decision": "attempted" if should_retrain else "skipped",
    "reason": retrain_result.get("reason", "unknown"),
    "retrained": retrain_result.get("retrained", False),
    "val_loss": retrain_result.get("val_loss"),
    "fitness": current_fitness,
    "level": current_level,
    "champion_traces": evo_result.get("champion_traces", 0),
    "cycles_since_last_retrain": cycles_since_last_retrain,
}
decisions_path = os.path.join(PROJECT_ROOT, "retrain_decisions.jsonl")
with open(decisions_path, "a") as f:
    f.write(_json.dumps(decision_record) + "\n")
```

**Verify:** After running 3 cycles, check `retrain_decisions.jsonl` has 3 entries with `"decision": "attempted"`.

---

## TASK 3: Run End-to-End Verification (CRITICAL — prove it works)

**Goal:** Prove the retrain loop produces real val_loss.

**Steps:**
1. Clean stale state:
   ```bash
   rm -f meta_controller_state.json runner_state.json barrier_tuner_state.json
   rm -f retrain_decisions.jsonl
   rm -f milestone1/recurrent_data.jsonl
   ```

2. Run 3-cycle smoke test:
   ```bash
   cd "C:\Users\reese\Projects\Evolution Engine"
   .venv/Scripts/python.exe autonomous_runner.py --max-cycles 3 --generations-per-cycle 20
   ```

3. Verify:
   ```bash
   # Check val_loss is real (not null)
   python -c "
   import json
   with open('champions/progression.jsonl') as f:
       entries = [json.loads(l) for l in f if l.strip()]
   recent = entries[-3:]
   for e in recent:
       print(f'C{e[\"cycle\"]}: L{e[\"level\"]} F={e[\"fitness\"]:.1f} val_loss={e.get(\"val_loss\")}')
   assert all(e.get('val_loss') is not None for e in recent), 'val_loss still null!'
   print('SUCCESS: val_loss is real')
   "

   # Check retrain decisions logged
   cat retrain_decisions.jsonl | python -m json.tool

   # Check recurrent data grew
   wc -l milestone1/recurrent_data.jsonl
   ```

4. If val_loss is still null, check `cycle_N/retrain.log` for errors and fix the root cause.

---

## TASK 4: Clean Up Stale State Files (REQUIRED before fresh run)

The meta_controller_state.json is stale from old runs (only 9 cycles observed, val_loss null). Clean it before fresh testing:

```bash
cd "C:\Users\reese\Projects\Evolution Engine"
rm -f meta_controller_state.json runner_state.json barrier_tuner_state.json
rm -f retrain_decisions.jsonl
# Keep milestone1/recurrent_data.jsonl if you want to preserve old training data
# Remove it for a completely fresh start:
# rm -f milestone1/recurrent_data.jsonl
```

---

## TASK 5: Fix fitness_benchmark.py Constructor (LOW)

**Problem:** `benchmarks/fitness_benchmark.py` has a wrong constructor call that's masked by try/except.

**File:** `benchmarks/fitness_benchmark.py`

**Fix:** Find the `EvolutionEngine(cfg, tasks)` call and change to `EvolutionEngine(config=cfg)`. The `tasks` parameter doesn't exist in the constructor.

---

## VERIFICATION CHECKLIST

After all tasks, verify:
- [ ] `python -m pytest tests/ -q` — all 1373 tests pass *(Note: historical count — suite now has 8900+ tests.)*
- [ ] 3-cycle smoke test produces real val_loss
- [ ] `retrain_decisions.jsonl` has entries
- [ ] `meta_controller_state.json` shows cycles observed > 0
- [ ] `recurrent_data.jsonl` grows during run
- [ ] No crashes or silent failures in `cycle_N/retrain.log`

---

## DO NOT

- Do NOT re-implement fixes already in the working tree (data path, has_gpu, MetaController persistence, cycle-specific checkpoints)
- Do NOT change the fitness function further without understanding the current level_bonus system
- Do NOT add new dependencies
- Do NOT modify test files unless fixing broken tests
- Do NOT commit changes — leave for manual review
