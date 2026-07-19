# Crash Bisection Experiment Matrix

## Date: 2026-07-19

| # | Arm | Config | VRAM | Cycles | Relaunches | Crash Rate | Result |
|---|-----|--------|------|--------|------------|------------|--------|
| 1 | Control | Pre-R48, pre-L26-L30 | ~0GB used | 100/100 | **0** | 0 | Clean run |
| 2 | Arm A | EE_MAX_LEVEL=25 (L26-L30 off) | 8137 MiB | 6/20 | **10** | 1.67/cycle | Crash-storm |
| 3 | Arm B | EE_SKIP_EVOLUTION_ON_REJECTION=1 | 8137 MiB | 4/20 | **10** | 2.50/cycle | Crash-storm |
| 4 | Arm C | Full code (R48+L26-L30+R50+R51) | 8126 MiB (llama-server) | 1/10 | **2** | 2.0/cycle | Abort on gate |
| 5 | Arm D | Full code, clean VRAM | **1103 MiB** (10.6GB free) | 4/20 | **10** | 2.50/cycle | **Crash-storm** |

## Key Findings

### VRAM theory: FALSIFIED
- Arm D ran with 10.6GB free VRAM (llama-server killed, watchdog stopped)
- Still crash-stormed: 10 relaunches, 4/20 cycles completed
- Crash rate identical to arms with 8GB VRAM usage
- **VRAM exhaustion is NOT the root cause**

### What IS the trigger
- Both R48 and L26-L30 eliminated as triggers (Arms A, B)
- VRAM eliminated as trigger (Arm D)
- Crashes persist at identical rates regardless of code changes or VRAM state
- All faulting stacks point to retrain augmentation path (augmentation.py, synthetic_data.py, rv32i_mask.py)
- The crash is ACCESS_VIOLATION (C-level, not Python)
- The control (100 cycles, 0 crashes) was at a different commit
- **Something between commit 3d7ffe6 and HEAD introduced the crash**
- The crash is in the retrain subprocess, not in evolution

### What we know for certain
1. verify() gate (R50) works correctly — levels only advance on verified pass
2. F=0.000 rate: 0% across all arms (R48 fix works)
3. best val_loss: 0.0468 (Arm D)
4. Crash is ACCESS_VIOLATION in retrain augmentation path
5. Crash rate is code-independent — same rate with or without R48, L26-L30, or VRAM

### What remains unknown
- The exact code change between 3d7ffe6 and HEAD that introduced the crash
- Whether the crash is in Python code, torch, CUDA driver, or hardware
- Why faulthandler shows Python frames for a C-level fault (Windows async fault reporting)
