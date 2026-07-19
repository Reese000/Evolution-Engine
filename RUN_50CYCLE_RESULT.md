# 50-Cycle GPU Run — Post-Reboot Full Config

## Date: 2026-07-19, 10:22–15:00 UTC

## Setup
- System rebooted at 09:11 UTC (Windows Update)
- GPU fresh: 1249 MiB used / 10746 MiB free
- Full config: R48 + L26-L30 + R50 integrity gate + R51 VRAM gate (9GB)
- Model: xlarge (16L, d=768, d_ff=2048, ~115M params)
- Batch=8, grad_accum=2, amp=True, TF32=True

## Result: 50/50 CYCLES, 1 RELAUNCH

| Metric | Value |
|--------|-------|
| Cycles completed | 50/50 |
| Relaunches | 1 (crash at C37, recovered) |
| Crashes | 1 total |
| F=0.000 | 0/50 (0%) |
| Best val_loss | 0.0446 |
| Best verified fitness | 339.9 (C1, L3, v=1.00) |
| Max level reached | L13 |
| Avg cycle time | 76s (47.4 cyc/hr) |
| Total wall time | 1.1h |

## Per-Cycle Trajectory

| Cycle | Fitness | VF | Level | Words | Verify | Time |
|-------|---------|-----|-------|-------|--------|------|
| C1 | 339.9 | **339.9** | 3 | 5 | **1.00** | 80s |
| C2 | 339.9 | -339.9 | 3 | 5 | 0.00 | 74s |
| C3 | 357.0 | -357.0 | 4 | 6 | 0.00 | 72s |
| C4 | 353.7 | -353.7 | 4 | 7 | 0.00 | 71s |
| C5-C9 | 357.0 | -357.0 | 4 | 6 | 0.00 | 70-77s |
| C9* | 357.0 | -357.0 | 4 | 6 | 0.00 | 190s |
| C10 | 348.8 | -348.8 | 7 | 8 | 0.00 | 173s |
| C11 | 378.8 | -378.8 | 8 | 11 | 0.00 | 218s |
| C12 | 426.0 | -426.0 | 10 | 30 | 0.00 | 71s |
| C13 | 403.1 | -403.1 | 10 | 30 | 0.00 | 70s |
| C14 | 583.4 | -583.4 | 12 | 22 | 0.00 | 95s |
| C15-C50 | 466-527 | -xxx | 13 | 7-26 | 0.00 | 51-195s |

## L13 Wall Analysis
- Model reaches L13 at C15 (15/50 cycles) then stays there for 35 cycles
- No verify() passes after C1 — all subsequent champions at L13 have verify_score=0.00
- Model learns to *reach* higher levels but cannot *verify* them
- This suggests the model generates programs that complete at L13 but fail verification
- The verify() gate correctly prevents level advancement

## Key Observations
1. C1 verified (v=1.00) at L3 — the only verified champion in 50 cycles
2. Level progression L3→L4→L7→L8→L10→L12→L13 in first 15 cycles
3. Stalled at L13 for 35 cycles — no further progression
4. Fitness 466-583 range at L13 — model is generating but not verifying
5. GPU utilization: ~20% true (per MSI Afterburner measurement)

## GPU Utilization (PM-measured)
- Task Manager: 100% (misleading — busy time, not saturation)
- MSI Afterburner: ~16-20% true utilization, ~60W/285W TDP
- VRAM: 1.8GB/12GB during training
- **Optimization needed**: model too small to saturate RTX 4070 Ti
