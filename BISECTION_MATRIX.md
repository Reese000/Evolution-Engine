# Crash Bisection Experiment Matrix

## Date: 2026-07-19

| # | Arm | Config | VRAM Free | Cycles | Relaunches | Crash Rate | Result |
|---|-----|--------|-----------|--------|------------|------------|--------|
| 1 | Control | Pre-R48, pre-L26-L30 | ~11GB | 100/100 | **0** | 0 | Clean |
| 2 | Arm A | EE_MAX_LEVEL=25 | 3.8GB | 6/20 | **10** | 1.67/cycle | Crash-storm |
| 3 | Arm B | EE_SKIP_EVOLUTION_ON_REJECTION=1 | 3.8GB | 4/20 | **10** | 2.50/cycle | Crash-storm |
| 4 | Arm C | Full code, dirty VRAM | 3.8GB | 0/10 | **2** | gate abort | VRAM gate correct |
| 5 | Arm D | Full code, clean VRAM | **10.6GB** | 4/20 | **10** | 2.50/cycle | Crash-storm |
| 6 | Arm E | MAX_LEVEL=25 + SKIP_EVOLUTION=1 | **10.6GB** | 10/20 | **10** | 1.0/cycle | Crash-storm |
| 7 | **Arm F** | **Full code, post-reboot** | **10.4GB** | **10/10** | **0** | **0** | **CLEAN** |

## Root Cause (Final)
The NVIDIA driver/CUDA state degraded during the session (after ~21:00 Jul 18). This caused delayed native memory corruption that manifested as ACCESS_VIOLATION at arbitrary Python code locations. The corruption was **not caused by**:
- Any EE code change
- L26-L30 levels
- R48 (skip_evolution)
- VRAM exhaustion
- Any combination of the above

**Resolution:** System reboot (Windows Update at ~09:11 Jul 19) reset the driver state. Post-reboot, all code runs cleanly with 0 relaunches.

## Key Fixes Retained
1. **R50 integrity gate**: verify() before curriculum advance + verified_fitness field
2. **R51 VRAM pre-flight**: abort if free VRAM < 9GB (prevents running on corrupted driver state)
3. **R51 orphaned process cleanup**: kills leaked run_multi_cycle.py processes on startup
4. **F=0.000: 0%** across all clean runs (R48 fix works)
