# Evolution Engine

Python autonomous code evolution system that iteratively improves its own codebase through mutation and selection cycles.

## Structure

- `milestone0/` through `milestone7/` — Complete self-contained versions per evolution milestone
- `simulation/` — Simulation support code
- `champions/` — Champion code snapshots from evolution runs

## Commands

```bash
python meta_controller.py              # Run single evolution cycle
python evolution_engine.py             # Run evolution engine directly
python run_multi_cycle.py              # Run multi-cycle evolution
python v11_dashboard.py                # Launch dashboard
python -m pytest tests/                # Run tests
```
