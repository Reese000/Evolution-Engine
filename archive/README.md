# Archive

Archived on **2026-07-29** as part of P2-1 dead code cleanup.

## Dead Modules (2,727 LOC)

These 10 modules had **zero cross-references** (never imported by any live code):

| Module | LOC | Purpose |
|--------|-----|---------|
| `cron_evolution_loop.py` | 154 | Cron-based evolution loop runner |
| `run_evolution_supervised.py` | 468 | Supervised evolution run script |
| `v9_monitor.py` | 111 | v9 monitoring/health-check script |
| `v9_to_v10_watcher.py` | 170 | v9→v10 migration watcher |
| `v11_dashboard.py` | 594 | v11 dashboard (HTML report generator) |
| `profile_phases.py` | 435 | Phase profiling/benchmarking |
| `benchmark_gpu.py` | 202 | GPU benchmark utility |
| `benchmark_step.py` | 173 | Step-level benchmark utility |
| `log_rotation.py` | 335 | Log rotation utility |
| `augmentation_subprocess.py` | 85 | Data augmentation subprocess launcher |

## Stale Cycle Directories

Cycle directories `cycle_1` through `cycle_100` and `cycle_999` were moved here.
These contain generated evolution cycle data (configs, results, checkpoints)
from earlier runs. The active cycle data lives in `cycle_data/` at the project root.

## Notes

- These files are **not deleted** — preserved for historical reference.
- Not tracked by git (see root `.gitignore` for `archive/cycle_*/`).
- Dead modules are tracked in git for audit trail.
