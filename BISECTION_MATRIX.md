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
  - `augmentation.py:217 inject_reference_solutions`
  - All called from `training/loop.py train()` → `run_multi_cycle.py retrain_model()`
- **Conclusion**: L26-L30 is NOT the trigger. Crash rate identical to full L30 run.

## Arm B: R48 Reverted (skip_evolution restored)
- **Status**: NOT YET RUN
- **Rationale**: Arm A already eliminated L26-L30. Next step is to test
  whether reverting R48 eliminates crashes (yesterday's 100-cycle control
  had zero relaunches).

## Key Findings
1. Crashes are in retrain augmentation path, NOT evolution
2. All stacks point to `rv32i_model.py` encoding → `synthetic_data.py` generation
3. ACCESS_VIOLATION is C-level (torch/CUDA), Python frames show thread location
4. Crash rate unchanged whether L26-L30 enabled or disabled
5. The 100-cycle zero-crash control was before R48 (skip_evolution removed)

## Hypothesis (pending Arm B test)
R48 (removing skip_evolution) causes more frequent retrain calls, which
increases exposure to a latent CUDA crash in the encode_trace/synthetic_data
path. The crash was always present but was masked by lower retrain frequency.
