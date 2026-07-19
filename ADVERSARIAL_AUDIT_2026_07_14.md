# Evolution Engine — Adversarial Audit Report
**Date:** 2026-07-14 | **Scope:** Full codebase, test suite, run history, prior audit findings

---

## Executive Summary

The Evolution Engine is a 22,664-line Python system that evolves RISC-V RV32I programs via mutation/selection and retrains a transformer on champion traces. **1373 tests pass**, all modules import successfully, and 132 multi-cycle runs + 1534 autonomous cycles exist in the data. However, the system has **critical functional gaps** that prevent it from achieving its core goal: compounding self-improvement via the evolve→retrain loop.

**Bottom line:** The system evolves programs well (L17 reached) but the retraining feedback loop is broken in practice — 59% of cycles skip retraining entirely. The system cannot compound improvements because the model never learns from its champions.

---

## PART 1: What Actually Works ✅

### 1.1 Import/Compilation: All Modules Load
Every Python module in the root imports successfully with the project venv (torch+numpy). The only missing module is `barrier_tuner.py` (referenced in `run_multi_cycle.py` line 84) — gracefully handled via try/except.

### 1.2 Test Suite: 1373 Tests Pass
All tests pass in 76 seconds. Test coverage spans integration, unit, edge cases, convergence, efficiency, scale, warm-start, and checkpoint management.

### 1.3 Evolution Engine Core
- `evolution_engine.py` (3,949 lines): Population-based evolution with 7 mutation strategies, QD/novelty archives, curriculum auto-advancement, secondary tasks, batch evaluation, GPU VM farm integration
- `meta_controller.py` (1,261 lines): Cross-cycle telemetry, stagnation detection, checkpoint gating, rollback, state persistence
- `mutation_proposer.py` (1,707 lines): Trace analysis, directed mutations, model-guided generation, reference solution injection
- `tasks_rv32i.py` (1,439 lines): 18 curriculum levels (L0-L17), 22 task classes, secondary tasks

### 1.4 Autonomous Runner
- `autonomous_runner.py` (1,287 lines): Self-correcting evolution loop with stagnation detection, self-correction attempts, health monitoring
- **1534 autonomous cycles completed** reaching L17 (peak fitness 222.00)

### 1.5 Data Artifacts
- `champions/archive.jsonl`: 500 champion entries with full metadata
- `champions/progression.jsonl`: 1534 cycle records
- `milestone1/recurrent_data.jsonl`: 100 training examples
- `milestone1/checkpoints/`: Best, current, and snapshot checkpoints present

---

## PART 2: Critical Functional Failures 🔴

### CRITICAL-1: Retrain Loop Broken (59% of Cycles Skip Retraining)
**Impact: HIGHEST — the core feedback loop doesn't work**

Of 132 multi-cycle results:
- 54 cycles retrained (41%)
- 78 cycles skipped (59%)
- Skip reasons: "skipped" (45), "unknown" (26), "Operation not permitted" (7)

The `retrain_result` defaults to `{"retrained": False, "reason": "skipped"}` (line 1755). Many cycles never reach the actual retrain call because:
1. **Champion unchanged skip** (line 1780-1806): If fitness, level, AND words are unchanged from last cycle, retrain is skipped. With volatile fitness scores, this check triggers unpredictably.
2. **First-cycle skip**: `last_retrain_epochs=0` means the champion-unchanged check at line 1781 (`cycle >= 2 and last_retrain_epochs > 0`) never fires for cycle 1, but the default "skipped" value persists if anything else prevents retrain.
3. **Operation not permitted** (7 cycles): File permission errors during subprocess execution.
4. **Unknown reason** (26 cycles): Cycle result dict doesn't contain a "reason" key — these were likely successful evolution but retrain status was never recorded.

**Evidence:** The `retrained: false, reason: "skipped"` appears in cycles 7-9 of the most recent run despite having 193-207 champion traces available.

### CRITICAL-2: Fitness Function Is Deceptive (Binary Exact-Match)
**Impact: HIGH — evolution cannot find gradient toward improvement**

`evolution_engine.py:1688+` `_compute_fitness()`: Register scoring is binary — 1.0 if exact match, 0.0 otherwise. A program computing `fib(10)=54` scores identically to one that crashes. This makes L9+ tasks (loops, Fibonacci) nearly impossible to evolve because near-correct programs get no credit.

**Evidence from AUDIT_REPORT.md:** "For L9+ tasks, a program can be *almost* correct and still receive the same fitness as random noise."

### CRITICAL-3: Training Signal Contamination
**Impact: HIGH — model learns from failures instead of successes**

`evolution_engine.py` `_try_recurrent_feedback()`: Champion traces are saved to the recurrent buffer regardless of champion quality. When the champion has negative fitness (common at L8+), the transformer trains on crash traces and wrong outputs. After ~10 cycles, the model generates programs biased toward known failure modes.

**Evidence:** `milestone1/recurrent_data.jsonl` has only 100 entries — far fewer than the 193-207 traces generated per cycle, suggesting most are filtered out, but the quality gate is not consistently applied.

### CRITICAL-4: Data Pipeline Format + Path Mismatch (NEW — Subagent Finding)
**Impact: CRITICAL — training data never reaches the model from champion traces**

**Breakage A — Format mismatch:** `milestone1/build_recurrent_dataset.py` saves entries as:
```python
{'input_sequence': prog[:i+1], 'label': prog[i+1], 'source': 'champion', ...}
```
But `milestone1/data/loading.py` (line 88) expects:
```python
entry.get("champion_program", [])  # list of hex strings or ints
# Also expects: fitness, task_score, save_reason, generation, level
```
The loader finds no `champion_program` key and **skips every line** produced by the builder.

**Breakage B — Path mismatch:** Builder saves to `milestone1/recurrent_data/recurrent_data.jsonl` but training reads from `milestone1/recurrent_data.jsonl` (different directory level). Even if formats matched, files wouldn't be found.

**Consequence:** The "champion traces → training data" pipeline described in the project docs does not work end-to-end. Training only works if pre-existing `recurrent_data.jsonl` files in the correct format are already present from prior manual runs.

### CRITICAL-5: Stateless MetaController in AutonomousRunner (NEW — Subagent Finding)
**Impact: HIGH — checkpoint gating never rejects bad models**

`autonomous_runner.py` `_run_retrain()` (lines 760-777) creates a **brand new MetaController** on every call:
```python
mc = MetaController(checkpoint_dir=str(M1_CHECKPOINTS))
```
Since `MetaController.__init__` sets `_best_val_loss = float("inf")`, `accept_checkpoint()` always accepts (because `val_loss <= inf * 1.05` is always True). The runner can overwrite a good checkpoint with a worse model without any gating protection.

Compare with `run_multi_cycle.py` where MetaController persists across cycles and properly tracks `_best_val_loss` history.

### CRITICAL-6: Retrain Pipeline Architecture Flaw
**Impact: HIGH — checkpoint rollback silently fails**

`run_multi_cycle.py` uses `checkpoint_best.pt` as both retrain input AND output. When the gate rejects:
1. Rollback tries `checkpoint_best.pt.regressed → checkpoint_best.pt`
2. If retrain wrote to `checkpoint_best.pt` directly, the "regressed" backup IS the current file
3. `shutil.copy2(src, dst)` with `src == dst` silently no-ops
4. The model is never actually restored

**Status:** Same-file guard added (meta_controller.py:823-873) but the underlying design flaw remains — retrain output and best checkpoint should be completely separate paths.

---

## PART 3: Architectural Limitations 🟡

### ARCH-1: O(N²) Diversity Computation
`evolution_engine.py:2493+` `_compute_population_diversity()`: Full pairwise Hamming distance every generation. With pop=50 and 200 gens/cycle × 48 cycles = 8.5M pairwise comparisons per run. Should be MinHash sketches (O(N)).

### ARCH-2: Multi-Level Regression Check Is Quadratic
`evolution_engine.py:1688+` `_update_champion()`: Each elite challenger evaluated against ALL previously mastered levels. At L10, that's 10 extra VM evaluations per elite candidate × ~10 candidates = 100 extra evaluations per generation. Should only check `champion_level - 1`.

### ARCH-3: Novelty/QD Selection Bias at Elite Stage
Stage 1 uses novelty-blended fitness for elite selection, but stage 2 champion update uses raw task score. Programs with high raw-task-score but low novelty may be filtered out of the elite set, never becoming champion.

### ARCH-4: Meta-Evolution Optimizes the Wrong Target
`meta_evolution.py` optimizes static config (pop_size, mutation_rate) using 20-generation base runs. The bottleneck is generation quality, not hyperparameters. 20-generation runs are too short to distinguish good configs from lucky ones.

### ARCH-5: Recurrent Buffer Eviction Is Not Quality-Aware
Buffer evicts oldest trace from most over-represented level, ignoring fitness. A zero-fitness L12 trace can survive while a high-fitness L5 trace is evicted.

### ARCH-6: Task Instance Overfitting
`evolution_engine.py` `_refresh_task_instance()`: Task instance only refreshed at curriculum advances, not between generations. Champion can hardcode answers for specific input values rather than implementing general algorithms.

### ARCH-7: GPU Fallback Masks Kernel Bugs
GPU VM farm silently falls back to CPU on any exception. If GPU kernel produces systematically wrong results for certain instruction sequences, those errors are silently corrected without logging.

### ARCH-8: Missing barrier_tuner.py Module
`run_multi_cycle.py:84` imports `BarrierTuner` from `barrier_tuner` — module does not exist. The import is gracefully handled (try/except), but the v11.5 barrier detection + self-tuning features are completely inoperative.

### ARCH-9: ~110 Lines Dead Code in run_multi_cycle.py (NEW — Subagent Finding)
- `_cleanup_stale_checkpoints()` (line 2324-2330): Called but body is just `pass` — no-op
- `_run_training_visibility_diagnostics()` (lines 2743-2828): 85-line function **never called** — intentionally disabled but never removed
- BarrierTuner diagnostic wiring (lines 2231-2256): Dead because `diagnostics` is hardcoded to `None`

### ARCH-10: ~80-100 Trivially Passing Tests (NEW — Subagent Finding)
- **30 tests** are import/callable checks (e.g., `test_eval_result_importable`)
- **40 tests** are file existence/string matching on Dockerfile/scripts
- **20 tests** are pure dataclass default value checks
- **5+ tests** are hardcoded boolean expression tests (testing logic, not production code)
- These provide zero regression protection and inflate the test count

### ARCH-11: Critical Untested Paths (NEW — Subagent Finding)
- `run_multi_cycle.py`: **Zero tests** — the entire multi-cycle orchestrator
- `gpu_vm_farm.py`: **1 trivial import test** — GPU evaluation path untested
- `v11_dashboard.py`: **Zero tests**
- `tasks_rv32i.py`: Level1-17 tasks have no dedicated tests
- `curriculum.py`: **Zero tests**
- `model_factory.py`: **Zero tests**

---

## PART 4: Implementation Bugs (from Prior Audit) 🟠

### BUG STATUS CHECK — Fixed vs Still Present

| Bug | Status | Evidence |
|-----|--------|----------|
| `or True` debug leftover | ✅ FIXED | grep finds 0 matches |
| `level_regression` enum missing | ✅ FIXED | grep finds 0 matches |
| `parse_grad_norms` missing | ✅ FIXED | Function exists at meta_controller.py:1160 |
| `skip_evolution` dead code | ✅ FIXED | Properly initialized at line 1496, checked at 1706 |
| `_last_stagnation` never updated | ✅ FIXED | Updated at lines 409 and 500 |
| `_encode_inst` opcode key mismatch | ✅ FIXED | Uses `inst.get("name", inst.get("opcode", ""))` |
| `status: "ok"` for empty-error crashes | ✅ FIXED | Now `"ok" if success else "crash"` |
| Strategy archive fitness=0.0 bug | ✅ FIXED | Uses `_last_challenger_fitnesses` from `_select_elite` |
| `use_snapshot` signature mismatch | ✅ FIXED | `execute_rollback` now accepts `use_snapshot` param |
| Ensemble fitness `-inf` cascade | ✅ FIXED | Uses `continue` instead of replacing entire list |
| SLTU → SLT for signed max | ✅ N/A | No SLTU found in tasks_rv32i.py |

**Conclusion:** The prior audit bugs have been addressed. The remaining issues are architectural (Part 3) and the retrain loop (Part 2).

---

## PART 5: Code Quality Assessment

### 5.1 Codebase Size
| File | Lines | Status |
|------|-------|--------|
| evolution_engine.py | 3,949 | Core — functional but needs fitness refactor |
| run_multi_cycle.py | 2,832 | Orchestrator — retrain loop broken |
| mutation_proposer.py | 1,707 | Functional — trace analysis + model-guided |
| tasks_rv32i.py | 1,439 | 18 levels, 22 task classes — complete |
| autonomous_runner.py | 1,287 | Self-correcting loop — functional |
| meta_controller.py | 1,261 | Telemetry + gating — functional |
| gpu_vm_farm.py | 1,219 | GPU evaluation — functional |
| meta_evolution.py | 950 | Meta-evolution — suboptimal target |
| prefix_generator.py | 929 | Model-guided generation — functional |
| strategy_archive.py | 622 | Strategy tracking — functional |
| **Total** | **22,664** | |

### 5.2 Test Coverage Assessment
- **1373 tests pass** across 40 test files
- **Unit tests dominate** — most test isolated functions with mocks
- **Integration tests present** but limited (test_integration.py, test_e2e_smoke.py)
- **Critical gap:** No end-to-end test that runs `run_multi_cycle.py --cycles 2` and verifies retraining actually happens
- **Critical gap:** No test for the retrain subprocess execution path
- **Critical gap:** No test for checkpoint rollback on gate rejection
- **Trivial tests present:** Some tests just check imports or dataclass defaults

### 5.3 Dependency Status
- `torch`, `numpy`: Required, available in project venv
- `cupy-cuda12x`, `triton`: Optional GPU deps (not installed, gracefully skipped)
- `barrier_tuner`: **MISSING** — referenced but never created
- `pytest`: Available for testing

---

## PART 6: Run History Analysis

### 6.1 Multi-Cycle Results (132 cycles)
- **Max level reached:** 17
- **Max fitness:** 222.00
- **Last 5 cycles:** Levels 5-6, fitness 63-162 (volatile)
- **Retraining success rate:** 41%
- **Pattern:** Early cycles reach high levels (L13-L17), later cycles regress to L5-6

### 6.2 Autonomous Runner (1534 cycles)
- **Peak:** Cycle 47, L17, fitness 189.51
- **Stabilization:** 1522 cycles at L17 (last 99% of run)
- **Pattern:** Rapid early progress, then stagnation at L17

### 6.3 The Compounding Self-Improvement Goal
**NOT ACHIEVED.** The system does not demonstrate compounding improvement across cycles. The last 5 multi-cycle results show levels 5-6 with inconsistent fitness, while the autonomous runner plateaued at L17. The retrain feedback loop (the mechanism for compounding) is broken.

---

## PART 7: What Needs to Be Done (Priority Order)

### P0 — Fix the Retrain Loop (CRITICAL)
1. **Separate retrain output from best checkpoint**: Retrain always writes to `checkpoint_cycle_N.pt`, gate acceptance copies to `checkpoint_best.pt`
2. **Remove champion-unchanged skip logic**: Always retrain when new traces exist (at minimum every N cycles)
3. **Add retrain status logging**: Every cycle must record whether retraining was attempted, succeeded, or skipped, with a reason
4. **Fix file permission errors**: 7 cycles failed with "Operation not permitted" — diagnose and fix

### P1 — Fix the Fitness Function
1. **Distance-weighted register scoring**: Replace binary exact-match with `score += 1.0 / (1.0 + abs(actual - expected) / max(1, expected))`
2. **Partial-progress metrics**: At L9+, add intermediate signals (loop body executed? correct memory address written?)
3. **Add fitness gate to recurrent buffer**: Only save traces where `champion_fitness > 0` or `raw_task_score >= 0.3 * max_possible`

### P2 — Fix Training Signal Quality
1. **Fitness-gated trace storage**: Discard traces from negative-fitness champions
2. **Recency decay**: Weight recent traces higher than old ones
3. **Separate failure archive**: Log failures for diagnostics but don't feed to retrain pipeline

### P3 — Architecture Improvements
1. **Task instance refresh**: Reseed every 25 generations to prevent overfitting
2. **Recurrent buffer quality eviction**: Evict lowest-fitness trace, not oldest
3. **Elite selection at max level**: Use `raw_only=True` for stage-1 selection
4. **O(N²) → MinHash diversity**: Approximate Hamming distance with sketches

### P4 — Operational Hardening
1. **`--clean-state` flag**: Delete all state files before starting a new run
2. **Atomic state writes**: Use tempfile + rename for meta_controller_state.json
3. **GPU fallback logging**: Count and warn on excessive CPU fallback
4. **Dashboard error boundary**: Handle corrupted input files gracefully
5. **Create barrier_tuner.py**: Implement the missing module or remove all references

### P5 — Long-Term
1. **Phase 1 Knowledge Injection**: Inject L11-L17 reference solutions into training data
2. **Larger model**: 90M-120M params at fp16 (current ~6M is too small for L9+)
3. **Meta-evolution repurpose**: Optimize reward signal composition, not static config
