# Evolution Engine — Hermes Refinement Prompt

Use this prompt to feed into Hermes for a focused refinement pass on the Evolution Engine project.

---

## PROMPT

You are working on the Evolution Engine project at `C:\Users\reese\Projects\Evolution Engine`.

### Context
An adversarial audit has been completed. Read `ADVERSARIAL_AUDIT.md` for the full report. The summary:

- **1,373 tests pass, 108 Python files, 0 syntax errors** — code quality is high *(Note: historical snapshot — suite now has 8900+ tests.)*
- **The system does not evolve** — best fitness (320.51) was achieved in Cycle 1, Gen 4. After 1,534 cycles, no improvement. The system oscillates between 160-220 fitness.
- **Root cause: Fitness function rewards simplicity too aggressively** — trivial 3-5 instruction programs at level 0 beat complex programs at level 17
- **Benchmark bug**: `benchmarks/verify_improvements.py` line 211 passes `tasks` list as `rng` argument to `EvolutionEngine(cfg, tasks)` — should be `EvolutionEngine(cfg)`

### Your Task — 5 Steps, Execute All

#### Step 1: Fix the Benchmark Bug
Fix `benchmarks/verify_improvements.py` line 211. The `EvolutionEngine` constructor signature is:
```python
def __init__(self, config, rng=None, vm_farm=None, champion_anchor=None):
```
The benchmark passes `tasks` (a list) where `rng` (a `random.Random`) is expected. Change:
```python
engine = EvolutionEngine(cfg, tasks)
```
to:
```python
engine = EvolutionEngine(cfg)
```
Then verify the benchmark runs: `.venv/Scripts/python.exe benchmarks/verify_improvements.py`

#### Step 2: Analyze and Fix the Fitness Function
Read `evolution_engine.py` and find `_compute_fitness`. The fitness function currently penalizes program length so aggressively that simple programs always beat complex ones, even when complex programs solve harder curriculum levels.

**Requirements for the new fitness function:**
1. **Primary signal: Curriculum level** — a program at level N should always score higher than a program at level N-1, regardless of length
2. **Secondary: Task correctness** — within the same level, more correct outputs = higher fitness
3. **Tertiary: Efficiency** — within same level and correctness, fewer instructions = slightly higher fitness
4. **Bloat penalty: Mild only** — penalize programs over 200 instructions, but never let the penalty overcome a level advancement

The current fitness formula likely looks like: `fitness = task_score - bloat_penalty`. It needs to become: `fitness = level_bonus + task_score - mild_bloat_penalty`

#### Step 3: Verify Evolution Actually Progresses
After fixing the fitness function, run a 20-generation evolution at level 0:
```python
from evolution_engine import EvolutionEngine, EvolutionConfig
config = EvolutionConfig()
config.population_size = 20
config.max_generations = 20
config.curriculum_level = 0
config.num_eval_seeds = 1
engine = EvolutionEngine(config)
result = engine.run(num_generations=20)
print(f"Final fitness: {engine.champion_fitness}")
print(f"Champion length: {len(engine.champion)}")
```
**Success criteria:** Fitness should be > 320 (the current best) or the champion should have advanced to level 1+.

#### Step 4: Run Full Test Suite
```bash
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short
```
All 1,373 tests must still pass. If the fitness function change breaks tests, fix the tests to match the new behavior (the tests were verifying the broken behavior).

#### Step 5: Update ADVERSARIAL_AUDIT.md
Add a section at the bottom:
```markdown
## 8. Refinement Pass Results (Date)
- Benchmark bug: [fixed/not fixed]
- Fitness function: [what changed]
- Evolution test: [result — did fitness improve?]
- Test suite: [X/1373 pass]
- Remaining issues: [list]
```

### Constraints
- Do NOT refactor the codebase architecture — just fix the fitness function and benchmark
- Do NOT add new modules or reorganize files
- Do NOT change the test structure — only update assertions if the fitness function behavior changes
- Do NOT modify `run_multi_cycle.py` or `meta_controller.py` — focus on `evolution_engine.py` and the benchmark
- All changes must pass the existing test suite
