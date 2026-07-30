# Evolution Engine — Development Roadmap & Kanban

**Date:** 2026-07-30  
**Commit:** a6ba77a (chore: archive stale root files)  
**Test Suite:** 8842 passed / 6 failed / 76 skipped — 121.29s  
**Previous Analysis:** .hermes/analysis-2026-07-29.md (8911p/17f)

---

## Table of Contents

1. [Architecture Summary](#1-architecture-summary)
2. [Test State & Remaining Failures](#2-test-state--remaining-failures)
3. [Curriculum Expansion: Levels 14-30](#3-curriculum-expansion-levels-14-30)
4. [Feature Gaps vs. Constitution Goals](#4-feature-gaps-vs-constitution-goals)
5. [Prioritized Roadmap (P0/P1/P2)](#5-prioritized-roadmap-p0p1p2)
6. [Kanban Task List](#6-kanban-task-list)
7. [Algorithm & ML Enhancement Ideas](#7-algorithm--ml-enhancement-ideas)

---

## 1. Architecture Summary

### System Overview

Evolution Engine is a **self-evolving RV32I machine-code AI**. It maintains a champion program, generates challengers via mutation, evaluates them on a VM (CPU or GPU), and re-trains a transformer model on champion traces. The system is orchestrated by a MetaController that tunes hyperparameters across cycles.

### Module Map (~28,900 LOC active code)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS RUNNER (entrypoint)                    │
│                  autonomous_runner.py (1,305 LOC)                   │
│          Config-driven, --no-limit, systemd service, self-correct   │
└────────────┬────────────────────────────┬───────────────────────────┘
             │                            │
             ▼                            ▼
┌────────────────────────┐  ┌─────────────────────────────────────────┐
│  META CONTROLLER       │  │  MULTI-CYCLE RUNNER                     │
│  meta_controller.py    │  │  run_multi_cycle.py (2,961 LOC)        │
│  (1,698 LOC)           │  │  ├ BarrierTuner (393 LOC)              │
│                        │  │  ├ MetaEvolution (950 LOC)              │
│  • Checkpoint gating   │  │  └ MetaGenome (622 LOC)                │
│  • Telemetry tracking  │  └─────────────────────────────────────────┘
│  • Stagnation detect   │
│  • Parameter tuning    │
│  • Circuit breaker     │
│  • Recovery mode       │
└────────────┬───────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EVOLUTION ENGINE (core)                           │
│              evolution_engine.py (6,189 LOC — GOD OBJECT)           │
│                                                                     │
│  Classes: VMFarm, HallOfFame, EvolutionConfig, EvolutionEngine      │
│  7 mutation strategies: RANDOM, TASK_AWARE, DIRECTED,              │
│    PERTURBATION_CAPPED, GROW, HYBRID, MODEL_GUIDED                │
│  3 diversity archives: QD (MAP-Elites), Novelty (k-NN), Strategy   │
│  Fitness: 14-signal composite scoring (344-line method)             │
│  Curriculum: auto-advance with stall detection                      │
└──┬──────────┬────────────┬──────────────┬───────────────────────────┘
   │          │            │              │
   ▼          ▼            ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────────┐
│MUTATION│ │STRATEGY  │ │NOVELTY   │ │QD ARCHIVE                │
│PROPOSER│ │ARCHIVE   │ │ARCHIVE   │ │qd_archive.py (929 LOC)  │
│(2,078) │ │(935 LOC) │ │(1,303 LOC│ │MAP-Elites grid (20×20)   │
│        │ │UCB-based │ │k-NN via  │ │                          │
│4 mutate│ │strategy  │ │SciPy     │ │Behavioural axes:         │
│strategies│ │selection│ │KDTree   │ │  instr diversity,        │
└────────┘ └──────────┘ └──────────┘ │  register pressure       │
                                      └──────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                                │
│                                                                     │
│  milestone1/train_model.py → training/loop.py                      │
│  ├ data/loading.py, data/augmentation.py, data/packing.py          │
│  ├ training/model_factory.py → rv32i_model.py                     │
│  ├ training/schedule.py (cosine + warmup + SWA)                    │
│  └ training/checkpoint.py + training/scaling/ (6 modules)          │
│                                                                     │
│  Model: RV32ITransformer (~6M params)                               │
│  • Decoder-only, hidden=256, 8 layers, 8 heads, d_ff=1024          │
│  • SwiGLU FFN, RoPE, RMSNorm, tied embeddings                     │
│  • 2048-vocab field-level tokenization (47 inst + 32 regs + nibbles)│
│  • Logit masking for valid instruction field transitions            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    VM EVALUATION                                    │
│  vm_farm.py (shim) → gpu_vm_farm.py (1,219 LOC)                   │
│  CPU: sequential via Rust safety kernel                             │
│  GPU: CuPy CUDA kernel (batch_size=128, budget=5000)               │
│  30-300× speedup over CPU, bit-exact verification                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    CURRICULUM                                       │
│  tasks_rv32i.py (2,323 LOC) — 31 levels (L0-L30) + 3 secondary   │
│  LEVELS dict: {0: Level0, ..., 30: Level30}                       │
│  Config default: max_level = 30 (raised from 13 on 2026-07-29)    │
│  Secondary: BitManipulationTask, ArithmeticTask, MemoryAccessTask  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    SUPPORTING MODULES                               │
│  config.py (452) — Unified config (JSON + env + CLI)               │
│  champion_archive.py (852) — Persistent champion archive            │
│  champion_anchor.py (158) — Champion preservation/restart           │
│  progression_tracker.py (824) — Cross-cycle metrics + auto-tune     │
│  prefix_generator.py (929) — Prefix-guided model generation        │
│  data_rotation.py (510) — Checkpoint rotation                       │
│  metrics.py (383) + metrics_export.py (285) — Prometheus export    │
│  monitoring_alerts.py (310) — AlertManager thresholds              │
│  barrier_tuner.py (393) — Gradient barrier tuning for retrain      │
│  cycle_state.py (211) — Cycle state persistence                     │
│  checkpoint_manager.py (169) — Atomic config-tagged checkpoints     │
│  paths.py (83) — Path resolution                                    │
│  logging_config.py — Structured logging                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Per Cycle

```
1. autonomous_runner.py / run_multi_cycle.py starts cycle N
2. MetaController.tune_retrain_params() → LR, freeze layers, etc.
3. EvolutionEngine.run(num_generations):
   a. For each generation:
      - _generate_challengers() → mutations of champion
      - _evaluate_batch() → VM evaluation (CPU/GPU)
      - batch_compute_fitness() → 14-signal composite scoring
      - _select_elite() → tournament + QD selection
      - _update_champion() → verify gate → promote if better
      - _update_qd_archive(), _update_novelty_archive()
   b. _try_recurrent_feedback() → save champion traces
   c. _try_curriculum_advance() → check if stalled → advance level
4. MetaController.update_cycle() → record telemetry
5. MetaController.accept_checkpoint() → gate model checkpoint
6. Training pipeline (train_model.py):
   a. load_champion_traces() from recurrent_data.jsonl
   b. augment (register rename, synthetic, reference solutions)
   c. pack sequences → DataLoader
   d. Fine-tune transformer (position-aware weighted CE loss)
   e. Checkpoint gating: only accept if val_loss improves
7. Log cycle results, prune old cycle directories
```

### Fitness Computation (14-signal composite)

| Signal | Max Points | Description |
|--------|-----------|-------------|
| reg_match × 100 | 100 | Per-register distance-weighted matching |
| bloat_penalty | -∞ | Penalty for >50 instructions |
| exec_credit | +10 | Execution progress |
| solved_bonus | +50 | All registers match (reg_match ≥ 0.99) |
| level_bonus | +25/level | Level × 25 (dominant signal) |
| efficiency_score | +15 | Instruction efficiency |
| multi_task_score | +15 | Secondary task performance |
| time_credit | +20 | Time-to-solution |
| loop_bonus | +5 | Backward branch detection (L5+) |
| mem_bonus | +3 | Memory access credit (L7+) |
| div_bonus | +2 | Instruction diversity (L5+) |
| smc_bonus | +25 | Self-modifying code patterns (L12+) |
| graduated_bonus | +50 | Near-threshold bonus |
| novelty_bonus | variable | From novelty archive |
| qd_bonus | variable | From QD archive |

---

## 2. Test State & Remaining Failures

### Current Test Results (2026-07-30)

| Metric | Value |
|--------|-------|
| **Total tests** | 8,924 |
| **Passed** | 8,842 |
| **Failed** | 6 |
| **Skipped** | 76 |
| **Duration** | 121.29s |
| **Pass rate** | 99.93% |

### Improvement from Previous Session (2026-07-29)
- Previous: 8,911p / 17f / 48s
- Current: 8,842p / 6f / 76s
- **11 failures fixed** (batch eval, stale test, partial verify)
- 28 additional skips (Windows-broken tests)

### Remaining 6 Failures

#### Category A: Missing Schema File (4 failures)
```
tests/test_progress_report.py::TestProgressReportSchema::test_schema_file_exists
tests/test_progress_report.py::TestProgressReportSchema::test_schema_is_valid_json
tests/test_progress_report.py::TestProgressReportSchema::test_schema_has_required_top_level_keys
tests/test_progress_report.py::TestProgressReportSchema::test_schema_trend_stats_def_exists
```
**Root Cause:** `progress_report_schema.json` does not exist at project root. Tests expect a JSON schema file that was never created or was archived.  
**Fix:** Create `progress_report_schema.json` matching the structure expected by `_generate_progress_report()` in autonomous_runner.py, OR update the 4 schema tests to skip when file is absent.

#### Category B: Method Complexity Quality Gates (2 failures)
```
tests/test_wave19_quality_gates.py::TestMethodComplexity::test_methods_within_line_limit[evolution_engine.py]
tests/test_wave19_quality_gates.py::TestMethodComplexity::test_no_method_over_100_lines[evolution_engine.py]
```
**Root Cause:** `evolution_engine.py` (6,189 LOC) has 14 methods exceeding the 50-line soft limit and 10 methods exceeding the 100-line hard ceiling. Worst offenders:
- `_compute_fitness()`: 344 lines
- `batch_compute_fitness()`: 331 lines
- `get_fitness_diagnostic_summary()`: 299 lines
- `_choose_mutation_strategy()`: 267 lines
- `run()`: 271 lines
- `_try_recurrent_feedback()`: 213 lines
- `_replace_with_model_guided()`: 173 lines

**Fix:** Major refactoring of evolution_engine.py — extract fitness computation, strategy selection, and training data collection into separate modules. This is the #1 code quality issue.

---

## 3. Curriculum Expansion: Levels 14-30

### Curriculum Architecture
- **31 main levels** (L0-L30): Each level adds a NEW output register (x3 through x33)
- **3 secondary tasks**: BitManipulationTask, ArithmeticTask, MemoryAccessTask (for multi-task fitness)
- **Cumulative design**: Each level's verify() checks ALL previous output registers
- **Monotonic registers**: L0→x3, L1→x4, ..., L30→x33

### Level Categories

#### Foundation (L0-L6): Basic Arithmetic & Data Movement
| Level | Task | Key Operations |
|-------|------|----------------|
| L0 | x3 = x1 | ADDI (move) |
| L1 | x4 = 2*x1 | SLLI (shift) |
| L2 | x5 = x2 | ADDI (move) |
| L3 | x6 = 2*x2 | SLLI + ADD |
| L4 | x7 = x4+x6 | ADD chain |
| L5 | x8 = SLT(x4,x6) | Compare |
| L6 | x9 = x1+x2 | Cross-input ADD |

#### Memory & Loops (L7-L10): Memory Operations
| Level | Task | Key Operations |
|-------|------|----------------|
| L7 | x10 = mem[x1] | LW (load word) |
| L8 | x11 = mem[x1]+mem[x2] | Dual LW + ADD |
| L9 | x12 = sum(mem[x1..x1+x2]) | Loop + LW + ADD |
| L10 | x13 = max(mem[x1..x1+x2]) | Loop + LW + SLT |

#### Self-Modification (L11-L13): SMC Patterns
| Level | Task | Key Operations |
|-------|------|----------------|
| L11 | x14 = fib(x1) | Fibonacci (recursive loops) |
| L12 | x15 = x1+x2 (SMC) | Self-modifying addition |
| L13 | x16 = x1+100+x2 (SMC) | Multi-instruction SMC |

#### Newly Unlocked (L14-L30): Advanced Patterns ⭐
| Level | Task | Key Operations | Difficulty |
|-------|------|----------------|------------|
| L14 | x17 = 2*x1+x2+100 (multi-program) | Shared memory between programs | 550 |
| L15 | x18 = conditional (x2≤x1 ? x1+x2 : 0) | BEQ/SLT conditional branching | 792 |
| L16 | x19 = 4*x1+8*x2 | SLLI chains (shift-and-add) | 936 |
| L17 | x20 = x1+x2 (via JALR) | Meta-evolution: generate + execute sub-program | 1,372 |
| L18 | x21 = (x1^x2) & 0xFFFF | XOR + AND masking | 1,800 |
| L19 | x22 = BNE result | BNE/BLTU conditional branching | 1,980 |
| L20 | x23 = memory XOR chain | LW + XOR + SW + LW | 2,975 |
| L21 | x24 = ((x1^x2)<<2)^x1 | Chained XOR + shift pipeline | 3,240 |
| L22 | x25 = nested if-then-else | Nested branching (3-way) | 4,015 |
| L23 | x26 = sum(mem[0]+mem[4]) | Multi LW/SW with address compute | 4,800 |
| L24 | x27 = conditional memory store | Conditional execution + memory side effects | 5,535 |
| L25 | x28 = (x1^x2)+(x1<<1)+(x2&0xFF) | Full instruction mix | 6,930 |
| L26 | x29 = shift chain + masked add | SLL/SRL/SRA chained shifts | 7,442 |
| L27 | x30 = unsigned compare branch | BGEU/BLTU unsigned comparison | 8,640 |
| L28 | x31 = SB/LB round-trip ^ x2 | Byte memory operations | 10,000 |
| L29 | x28 = (x1+x2)<<1 (subroutine) | Inline subroutine call pattern | 11,442 |
| L30 | x29=abs_diff, x30=sorted_sum | Sort-and-compute (compare + branch) | 13,104 |

### Critical Finding: meta_evolution.py Still Has max_level=13

```python
# meta_evolution.py:70
_DEFAULT_MAX_LEVEL: int = 13  # ← STALE, should be 30
```

This means meta-evolution's fitness normalization still uses `final_level / 13`, so a champion at L20 would score `20/13 = 1.54` (over 1.0), distorting meta-fitness calculations.

---

## 4. Feature Gaps vs. Constitution Goals

### Constitution Phase Status

| Phase | Goal | Status | Evidence |
|-------|------|--------|----------|
| **Phase 0** | Foundation | ✅ COMPLETE | MetaController, GPU VM, self-calibrate, 31 tests |
| **Phase 1** | Knowledge Injection | ⚠️ PARTIAL | Reference solutions injected, but model still stuck at L4-L10 historically |
| **Phase 2** | Prefix-Guided Generation | ✅ IMPLEMENTED | prefix_generator.py + MODEL_GUIDED strategy exist |
| **Phase 3** | Self-Modifying Code Mastery | ⚠️ PARTIAL | SMC bonus exists, L12-L13 defined, but champion never reached independently |
| **Phase 4** | Full Autonomy | ⚠️ PARTIAL | autonomous_runner.py exists, systemd service exists, but 48-cycle runs not verified |

### Key Gaps

| Gap | Severity | Description |
|-----|----------|-------------|
| **G1: L4 plateau** | CRITICAL | Champion stuck at L4 for 3+ cycles; model can't learn memory+arithmetic patterns |
| **G2: L11+ independence** | HIGH | Constitution says "champion can't independently solve L11+" — this is still true |
| **G3: Model capacity** | HIGH | ~6M params may be insufficient for L20+ patterns; no scaling path defined |
| **G4: Training data poverty** | HIGH | Champion traces at L4+ don't contain correct patterns for higher levels |
| **G5: meta_evolution max_level** | MEDIUM | Still set to 13, distorts meta-fitness for levels 14-30 |
| **G6: No curriculum warm-start** | MEDIUM | New levels (14-30) have no pre-trained model that can solve them |
| **G7: evolution_engine.py monolith** | MEDIUM | 6,189 LOC god-object, 10 methods exceed 100-line ceiling |
| **G8: Missing progress_report_schema.json** | LOW | 4 test failures from missing schema file |
| **G9: README stale** | LOW | Still claims "12-level curriculum" and "1321/1321 passing" |
| **G10: Dead code ~3,200 LOC** | LOW | 10 archived modules still in codebase |

---

## 5. Prioritized Roadmap (P0/P1/P2)

### P0 — Critical (Blocks correctness, data integrity, or curriculum progress)

| ID | Task | Rationale | Effort |
|----|------|-----------|--------|
| **P0-1** | Fix meta_evolution.py `_DEFAULT_MAX_LEVEL` 13→30 | Meta-fitness normalization is wrong for L14-30; over-rewards levels above 13 | S |
| **P0-2** | Fix evolution_engine.py 10 methods exceeding 100-line ceiling | God-object prevents testing, maintenance, and evolution of the core algorithm. 2 test failures. | L |
| **P0-3** | Create progress_report_schema.json OR fix 4 test failures | 4 tests always fail; blocks clean CI. Either create the schema or mark tests as `skip` with reason. | S |
| **P0-4** | Validate L14-L30 reference solutions execute correctly on VM | 17 new levels were added but never run end-to-end. Verify all reference solutions produce correct output. | M |

### P1 — High (Functional gaps, curriculum advancement, ML improvements)

| ID | Task | Rationale | Effort |
|----|------|-----------|--------|
| **P1-1** | Break evolution_engine.py into modules | Extract fitness computation, strategy selection, QD/novelty updates, and training data collection into separate files | XL |
| **P1-2** | Curriculum warm-start for L14-30 | Inject reference solutions for L14-30 into training data so the model can learn these patterns before evolution tries to discover them | M |
| **P1-3** | Fix L4 plateau: adaptive fitness gradient | The step from L3→L4 requires memory+arithmetic patterns the model hasn't learned. Add curriculum-aware trace injection specifically for L4 bottleneck | M |
| **P1-4** | Model scaling study | Test if larger models (12M, 24M params) can learn L14+ patterns. Add `training/scaling/` support for model size ablation | M |
| **P1-5** | Prefix-guided generation integration testing | prefix_generator.py exists but may not be wired into the evolution loop properly. Verify MODEL_GUIDED strategy uses it correctly | S |
| **P1-6** | Update README to reflect current state | 31-level curriculum, 8842+ tests, current feature set | S |
| **P1-7** | Run 48-cycle autonomous test | Constitution Phase 4 goal. Run autonomous_runner.py for 48 cycles without intervention and verify net fitness improvement | L |

### P2 — Medium (Code quality, testing, operational)

| ID | Task | Rationale | Effort |
|----|------|-----------|--------|
| **P2-1** | Reorganize 156 test files into logical groups | Wave-based naming (test_wave1-wave78) is unmaintainable. Group by: unit, integration, quality-gate, e2e | M |
| **P2-2** | Remove sys.path.insert() hacks | Use proper packaging (pyproject.toml already exists) or consistent ensure_project_on_path() | S |
| **P2-3** | Archive remaining dead modules | ~3,200 LOC of dead code (cron_evolution_loop, run_evolution_supervised, v9_monitor, etc.) | S |
| **P2-4** | Re-run benchmarks and update results | benchmark_results.json is stale; verify 5.3× speedup claim | S |
| **P2-5** | Add integration test for L14-L30 | Currently no test verifies that the full pipeline (evolution + training) works at these levels | M |
| **P2-6** | Diversity measurement sensitivity | Diversity stuck at 0.3 across cycles; measurement may not capture real diversity changes | S |
| **P2-7** | Circuit breaker for new levels | If champion stalls at a new level (L14+) for N cycles, automatically adjust strategy or inject reference data | M |

---

## 6. Kanban Task List

### Backlog
- [ ] **KB-01** Model scaling study: test 12M/24M param variants (P1-4)
- [ ] **KB-02** 48-cycle autonomous run validation (P1-7)
- [ ] **KB-03** Reorganize 156 test files into logical groups (P2-1)
- [ ] **KB-04** Circuit breaker for new curriculum levels (P2-7)
- [ ] **KB-05** Diversity measurement sensitivity analysis (P2-6)
- [ ] **KB-06** Re-run benchmarks and update README claims (P2-4)

### Ready
- [ ] **KB-07** Fix meta_evolution.py `_DEFAULT_MAX_LEVEL` 13→30 (P0-1) — _one-line fix_
- [ ] **KB-08** Create progress_report_schema.json (P0-3) — _create schema matching autonomous_runner.py output_
- [ ] **KB-09** Validate L14-L30 reference solutions on VM (P0-4) — _run tasks_rv32i.py self-test_
- [ ] **KB-10** Update README curriculum section to "31 levels" (P1-6) — _documentation fix_

### In Progress
- [ ] **KB-11** Break evolution_engine.py into modules (P1-1) — _extract fitness, strategy, archives_
- [ ] **KB-12** Fix evolution_engine.py method complexity (P0-2) — _part of KB-11_

### Done (This Session)
- [x] **KB-13** Raise max_level 13→30 in config.py (commit 175b222)
- [x] **KB-14** Fix batch fitness divergence (P0-1 from previous session, commit e9702b5)
- [x] **KB-15** Archive ~3,200 LOC stale root files (commit a6ba77a)
- [x] **KB-16** Fix sys.path.insert() hacks with ensure_project_on_path() (commit 09424bb)
- [x] **KB-17** Skip tests for archived modules (commit 83bd9d8)

---

## 7. Algorithm & ML Enhancement Ideas

### Evolutionary Algorithm Improvements

1. **Adaptive Mutation Intensity by Level**
   - L0-L6: Random mutations work well (simple patterns)
   - L7-L10: Need memory-aware mutations (LW/SW adjacent to computation)
   - L11-L13: Need SMC-aware mutations (JALR + SW patterns)
   - L14-L30: Need multi-instruction sequence mutations
   - Currently: `_choose_mutation_strategy()` is a 267-line function that doesn't condition on level complexity

2. **Island Model / Population Structuring**
   - Current: single population of 50 individuals
   - Enhancement: 4 islands of 12 individuals each, with migration every 10 generations
   - Benefit: maintains diversity while allowing specialization per island

3. **Coevolution of Program + Hyperparameters**
   - MetaEvolution exists but is subprocess-based (slow)
   - Enhancement: in-process meta-evolution that shares the same VM evaluation

4. **Program Synthesis from Components**
   - QD archive stores best programs per behavioral cell
   - Enhancement: recombine programs from different QD cells (not just mutation, but structural crossover)

5. **Fitness Shaping for New Levels**
   - When curriculum advances to a new level, the fitness landscape changes dramatically
   - Enhancement: adaptive fitness weights that shift emphasis to the new level's registers

### ML Component Improvements

1. **Curriculum-Aware Training Data Injection**
   - Current: reference solutions are injected as synthetic data
   - Enhancement: weight training samples by curriculum level, with exponential decay for older levels
   - Goal: model focuses on learning patterns for the CURRENT level, not memorizing old ones

2. **Larger Model Capacity**
   - Current: ~6M params (hidden=256, 8 layers)
   - Enhancement: test hidden=512, 12 layers (~24M params) for L20+ pattern learning
   - Constraint: RTX 4070 Ti (12.9GB VRAM) limits practical model size

3. **Prefix-Guided Generation with Temperature Control**
   - Current: prefix_generator.py generates from prefix context
   - Enhancement: add temperature scheduling — high temp for exploration at new levels, low temp for refinement at mastered levels

4. **Recurrent Feedback Quality Filtering**
   - Current: traces saved when fitness > MIN_FITNESS_FOR_TRACE (0.1)
   - Enhancement: save traces from the TOP 10% of each generation, weighted by fitness
   - Also: add program-length normalization to prevent the model from learning to generate very long programs

5. **Multi-Task Training Signal**
   - Current: secondary tasks (Bitwise, Arithmetic, Memory) are evaluated but not used in training
   - Enhancement: add secondary task traces to training data so the model learns diverse instruction patterns

### Curriculum Design Extensions

1. **L31+: Function Calls (JAL/JALR)**
   - Currently L17 tests sub-program execution via JALR
   - Enhancement: proper function call convention with stack frames (x2/sp, save/restore)

2. **L32+: Recursive Function Calls**
   - Fibonacci (L11) is the only recursive pattern
   - Enhancement: general recursion with stack-based state management

3. **L33+: Multi-Function Programs**
   - Programs that call multiple subroutines, each with different behavior
   - Tests: program decomposition, register allocation, stack management

4. **L34+: Self-Modifying Code with Loops**
   - L12-L13 test basic SMC
   - Enhancement: programs that modify their own loop bodies based on runtime data

### Operational Improvements

1. **Real-Time Dashboard**
   - v11_dashboard.py was archived
   - Enhancement: web-based dashboard showing fitness trajectory, diversity, level progression, model loss curves

2. **Experiment Tracking**
   - Currently: JSON state files + metrics_export.py (Prometheus)
   - Enhancement: integrate with Weights & Biases or MLflow for experiment comparison

3. **Automated A/B Testing**
   - Compare different configurations (population size, mutation rates, model sizes) automatically
   - Run parallel evolution instances with different configs, compare results

---

## Appendix: Key Files Reference

| File | LOC | Purpose |
|------|-----|---------|
| evolution_engine.py | 6,189 | Core evolutionary loop (GOD OBJECT) |
| run_multi_cycle.py | 2,961 | Multi-cycle orchestrator |
| tasks_rv32i.py | 2,323 | 31-level curriculum definition |
| mutation_proposer.py | 2,078 | Mutation strategy generation |
| meta_controller.py | 1,698 | Dynamic parameterization |
| autonomous_runner.py | 1,305 | Entrypoint, self-correction |
| novelty.py | 1,303 | Novelty search archive |
| gpu_vm_farm.py | 1,219 | GPU-accelerated RV32I VM |
| meta_evolution.py | 950 | Meta-evolution of configs |
| strategy_archive.py | 935 | UCB strategy selection |
| prefix_generator.py | 929 | Prefix-guided generation |
| qd_archive.py | 929 | MAP-Elites archive |
| champion_archive.py | 852 | Persistent champion archive |
| progression_tracker.py | 824 | Cross-cycle metrics |
| meta_genome.py | 622 | Meta-evolution genome |
| data_rotation.py | 510 | Checkpoint rotation |
| config.py | 452 | Unified configuration |
| barrier_tuner.py | 393 | Gradient barrier tuning |
| metrics.py | 383 | MetricsCollector |
| monitoring_alerts.py | 310 | Alert thresholds |
| metrics_export.py | 285 | Prometheus export |
| cycle_state.py | 211 | Cycle state persistence |
| champion_anchor.py | 158 | Champion preservation |
| checkpoint_manager.py | 169 | Atomic checkpoints |

---

*Generated by Hermes Agent on 2026-07-30. Analysis only — no changes made.*
