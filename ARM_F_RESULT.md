# Crash Bisection — Arm F: Post-Reboot Full Config

## Date: 2026-07-19, 09:35 UTC

## Setup
- System rebooted at ~09:11 UTC (Windows Update) — driver state reset
- GPU fresh: 1249 MiB used / 10746 MiB free at launch
- llama-server stopped + watchdog chain killed
- Full current config: R48 + L26-L30 + R50 + R51 gate (EE_VRAM_MIN_GB=9)
- No caps, no skips — exact production configuration

## Result: CLEAN

| Metric | Value |
|--------|-------|
| Batch 1 | **10/10 cycles, exit=0, 0 relaunches** |
| Batch 2 | Running (in progress) |
| Crashes | **0** |
| Best val_loss | 0.0475 |
| F=0.000 | 0/10 (0%) |
| Avg cycle time | 85s (42.4 cyc/hr) |

## Key Observations
- Batch 1 completed in 1535s (25.6 min) without a single crash
- C1: verified (v=1.00), F=339.9 at L3
- Level progression: L3 → L4 (C3-C9) → L7 (C10)
- Crash rate: **0/10** (vs 10/10 in Arms A-E post-corruption)

## Conclusion
**The reboot fixed the crash.** The machine/driver state was degraded since ~21:00 Jul 18. Windows Update reboot at ~09:11 Jul 19 reset the driver state, and the crashes disappeared.

All code stays — R48, L26-L30, R50 gate, R51 gate are all correct.
