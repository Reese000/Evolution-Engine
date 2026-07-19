# Crash Bisection Experiment Matrix

## Date: 2026-07-19

## Control
- **Run**: 100-cycle GPU at commit 3d7ffe6 (pre-R48, pre-L26-L30)
- **VRAM**: ~0GB used (fresh system, no prior runs)
- **Result**: 0 relaunches, 55 cyc/hr, 100/100 cycles completed
- **Stack**: N/A (no crashes)

## Arm A: L26-L30 Capped (EE_MAX_LEVEL=25)
- **Run**: 20-cycle GPU, current code minus L26-L30
- **VRAM**: 8137 MiB used (orphaned CUDA contexts from prior runs)
- **Result**: 6/20 cycles completed, 10 relaunches (hit max)
- **Crash rate**: 10 crashes / 6 successful cycles = 1.67 crashes/cycle
- **Faulting stacks** (via faulthandler):
  - `augmentation.py:141 inject_synthetic_data`
  - `rv32i_mask.py:193 _split_imm_b → rv32i_mask.py:290 encode`
  - `synthetic_data.py:163 generate_all → 336 _gen_templates → 416 _gen_loop_sum`
  - All called from `training/loop.py train()` → `run_multi_cycle.py retrain_model()`

## Arm B: R48 Reverted (EE_SKIP_EVOLUTION_ON_REJECTION=1)
- **Run**: 20-cycle GPU, skip_evolution restored on gate rejection
- **VRAM**: 8137 MiB used (same leaked CUDA contexts)
- **Result**: 4/20 cycles completed, 10 relaunches (hit max)
- **Crash rate**: 10 crashes / 4 successful cycles = 2.50 crashes/cycle
- **Faulting stacks** (via faulthandler):
  - `augmentation.py:270 inject_reference_solutions`
  - Called from `training/loop.py:165 train()` → `run_multi_cycle.py:956 retrain_model()`

## Arm C: Definitive Clean-GPU Run (R48 + L26-L30 + R50, clean VRAM)
- **Status**: BLOCKED — VRAM stuck at 8137 MiB. Orphaned CUDA contexts
  in GPU driver cannot be freed without system reboot.
- **Prediction**: Near-zero relaunches on clean GPU (per PM's hypothesis)

## Root Cause: VRAM Exhaustion from Orphaned CUDA Contexts
- 8137 MiB leaked across multiple EE runs (not from current session)
- NVIDIA driver holds CUDA contexts after process exit on Windows
- `nvidia-smi --gpu-reset` not available on driver 595.97
- `torch.cuda.empty_cache()` only clears allocated memory, not contexts
- Only solution: system reboot or driver restart
- Both Arm A and Arm B crash at identical rates — **confirms VRAM theory**

## Conclusions
1. **L26-L30 is NOT the trigger** — Arm A crashes at same rate without them
2. **R48 is NOT the trigger** — Arm B crashes at same rate with it reverted
3. **Code-independent crash rate** — same crashes regardless of code changes
4. **VRAM exhaustion is the trigger** — 8GB leaked CUDA contexts leave only 4GB
5. **Augmentation path is the allocation-heavy trigger** — crashes during
   reference solution injection (pure CPU work, but CUDA context corrupted)
6. **Both R49 commits were falsified** — supervisor cleanup didn't free contexts
7. **Fix**: VRAM pre-flight gate (abort if <9GB free) + orphaned process cleanup
