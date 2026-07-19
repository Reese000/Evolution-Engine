# Crash Bisection Experiment Matrix

## Date: 2026-07-19

## Control
- **Run**: 100-cycle GPU at commit 3d7ffe6 (pre-R48, pre-L26-L30)
- **Result**: 0 relaunches, 55 cyc/hr, 100/100 cycles completed
- **Stack**: N/A (no crashes)

## Arm A: L26-L30 Capped (EE_MAX_LEVEL=25)
- **Run**: 20-cycle GPU, current code minus L26-L30
- **Result**: 6/20 cycles completed, 10 relaunches (hit max)
- **Crash rate**: 10 crashes / 6 successful cycles = 1.67 crashes/cycle
- **Faulting stacks** (via faulthandler):
  - `augmentation.py:141 inject_synthetic_data`
  - `rv32i_mask.py:193 _split_imm_b → rv32i_mask.py:290 encode`
  - `synthetic_data.py:163 generate_all → 336 _gen_templates → 416 _gen_loop_sum`
  - All called from `training/loop.py train()` → `run_multi_cycle.py retrain_model()`

## Arm B: R48 Reverted (EE_SKIP_EVOLUTION_ON_REJECTION=1)
- **Run**: 20-cycle GPU, skip_evolution restored on gate rejection
- **Result**: 4/20 cycles completed, 10 relaunches (hit max)
- **Crash rate**: 10 crashes / 4 successful cycles = 2.50 crashes/cycle
- **Faulting stacks** (via faulthandler):
  - `augmentation.py:270 inject_reference_solutions`
  - Called from `training/loop.py:165 train()` → `run_multi_cycle.py:956 retrain_model()`

## Conclusions

### What IS NOT the trigger (eliminated by experiment)
1. **L26-L30 levels** — Arm A crashes at same rate without them
2. **R48 (skip_evolution removal)** — Arm B crashes at same rate with it reverted
3. **Evolution always running** — both arms crash whether evolution skips or not

### What the data shows
- Both arms crash at ~1.6-2.5 crashes/cycle, same as the full L30 run
- All faulting stacks are in the **retrain augmentation path**, not evolution
- The crash is ACCESS_VIOLATION (C-level, not Python)
- faulthandler shows Python frames where the thread WAS when the C crash fired
  (per PM: "augmentation may not be the actual culprit — faulting stacks may
  show main thread location while actual fault is in CUDA/native thread")
- The 100-cycle zero-crash control was at an earlier commit (3d7ffe6)
- Something changed between 3d7ffe6 and HEAD that introduced the crash

### What changed between control and current HEAD
Commits between 3d7ffe6 and HEAD (after removing R48 and L26-L30):
- R50 integrity gate (verify() calls in evolution_engine.py)
- verify() class-vs-instance fix
- In-process retrain CUDA cleanup
- Supervisor CUDA cleanup fix
- Various docs/fixes
- The augmentation code itself was NOT changed between 3d7ffe6 and HEAD

### Leading hypothesis
A torch/CUDA state corruption that accumulates during retrain's
in-process training, manifesting as ACCESS_VIOLATION when augmentation
code (encode_trace, synthetic data) runs in a corrupted CUDA context.

### Recommended fix (per PM)
Isolate synthetic-data generation + augmentation into a short-lived
subprocess (clean interpreter, no CUDA context) so retrain's data
prep can't collide with CUDA state. Resilient regardless of the exact
corruption source.
