# Evolution Engine — Audit Report
**Date:** 2026-06-17 | **Auditor:** Claude Sonnet 4.6 | **Scope:** Full codebase + run history

---

## Executive Summary

The Evolution Engine is a self-improving AI system that uses evolutionary algorithms to synthesize correct RISC-V RV32I machine-code programs through a closed feedback loop (evolve → collect traces → retrain → evolve). The system reaches L11+ when seeded with reference solutions but **cannot independently discover** L11+ solutions via generation quality alone.

**The core finding:** There are four compounding failure modes that together cause model degradation over long runs. The fitness function is deceptive, the training signal is contaminated by low-quality champions, strategy monoculture is a persistent risk, and the model capacity is saturated at L9+.

---

## Part 1 — Leading Causes of Model Degradation

### D1. Deceptive Fitness Landscape (Highest Impact)

**Location:** `evolution_engine.py:909–1035`, `_compute_fitness()`

**Problem:** The fitness function uses **binary exact-match scoring** for registers:
```python
for reg_idx, expected_val in expected_regs.items():
    actual_val = final_regs.get(reg_idx, 0)
    if actual_val == expected_val:
        program_scores[reg_idx] = 1.0  # all-or-nothing
```

A program that computes `fib(10) = 55` instead of `fib(10) = 55` scores 1.0. A program that computes `fib(10) = 54` scores 0.0 — identical to a program that crashes immediately. For L9+ tasks (loops, counters, Fibonacci), a program can be *almost* correct and still receive the same fitness as random noise. This makes the fitness landscape deceptive: evolution cannot find gradient toward improvement because small steps toward correctness are invisible.

**Cascading effect:** When fitness is flat across 90%+ of the search space, GROW dominates because it produces "non-crashing" programs that at least avoid the crash penalty (-25 to -50). This biases the training data toward short, non-functional programs.

**Path forward:**
- Replace exact-match with distance-weighted scoring:
  `score += 1.0 / (1.0 + abs(actual - expected) / max(1, expected))`
- For L9+ specifically, add partial-progress metrics: did the program execute the loop body at least once? Did it write to the correct memory address? These intermediate signals give evolution gradient without changing the final success criterion.

---

### D2. Training Signal Feedback Loop (High Impact)

**Location:** `evolution_engine.py:1954–2114`, `_try_recurrent_feedback()` and `_save_periodic_trace()`

**Problem:** The recurrent buffer saves champion traces regardless of champion quality. When the champion is stuck at negative fitness (common at L8+), the recurrent buffer fills with examples of programs that:
1. Crash with illegal instructions (crash_penalty = -50)
2. Reach budget timeout without computing anything (crash_penalty = -10)
3. Produce wrong register values for the task

The transformer then fine-tunes on these failure examples, learning to *generate programs that fail in similar ways*. This is the exact inverse of the intended behavior. The model should be learning from *successes*, not from *documented failures*.

**Evidence:** From `lessons_learned.md`: "The bottleneck is generation quality... The model needs to learn correct program patterns for high-level tasks before evolution can optimize them."

**Cascading effect:** After ~10 cycles, the model has been fine-tuned on hundreds of failing champions. Its program proposals at L8+ are biased toward known failure modes. The CROSSOVER and MODEL_GUIDED strategies produce worse challengers than random mutation because they learned from degraded examples.

**Path forward:**
- Add a fitness gate to `_try_recurrent_feedback()`: only save traces where `champion_fitness > 0` OR where `raw_task_score >= 0.3 * max_possible`. Traces below this threshold should be discarded even if the champion is the "best found so far."
- Add a separate "failure archive" that logs failures but is NOT fed to the retrain pipeline.
- Apply a recency decay: weight recent traces higher. Old traces from early cycles (when the model was worse) should count less than recent traces from a better-trained model.

---

### D3. Strategy Archive Positive Feedback Loop (High Impact)

**Location:** `evolution_engine.py:1570–1655`, `_choose_mutation_strategy()`; `strategy_archive.py`

**Problem:** The strategy archive records any outcome where a challenger's fitness > 0 as a "success." CROSSOVER and GROW dominate because:
- GROW produces slightly longer programs than short 2-word champions → fitness can increase trivially via growth bonus (`nop_count * 0.001`)
- CROSSOVER between moderately-fit programs can inherit a correct sub-sequence → gets recorded as successful even if the full program doesn't solve the task

Once a strategy builds a success rate lead, the archive's UCB exploration bonus still recommends it most often. The monoculture cap (`max_consecutive_same_strategy=30`) is too high — a strategy can dominate for 30 consecutive generations before the cap fires.

**Evidence:** This exact failure mode has been documented three times in `lessons_learned.md`:
1. v9: GROW 100% dominance (1515/1515 strategy uses)
2. v10: CROSSOVER 99.4% dominance
3. v10.1 C1: CROSSOVER still 20.6% even after fixes, but GROW 71%

**Path forward:**
- Reduce `max_consecutive_same_strategy` from 30 to 15 for non-GROW strategies at L5+.
- Add a level-advance archive flush: when the curriculum advances, decay the archive weights for the new level (start fresh with uniform priors). The archive for L7 should not carry over the CROSSOVER bias learned at L5.
- Track **generation-over-generation champion delta** per strategy, not just "did a challenger beat the current champion." A strategy that produces improvements every 50 generations should have lower weight than one producing improvements every 5 generations.

---

### D4. Model Capacity Saturation at L9+ (Medium-High Impact)

**Location:** `project_constitution.md:24`, `evolution_engine.py:1495–1568`

**Problem:** The seed model has 45M parameters. At L9+ (loops requiring branch instructions, conditional jumps, loop counter management), the correct program structures require 15-30 instruction words with specific dependency chains. The 45M model:
- Was trained primarily on L0-L8 programs (the levels evolution succeeds at)
- Has limited representational capacity for long-range dependencies needed in Fibonacci (L11) or self-modifying code (L12-L13)
- Produces programs of 8-20 words at L5+, but L11 requires 25-40 words with correct JALR/branch sequences

The model_guided_ratio at L8+ is set to 0.60 — meaning 60% of challengers come from this saturated model. When the model can't represent L11 patterns, 60% of challengers are guaranteed to be wrong.

**Path forward:**
- Profile actual VRAM usage at fp16. A 90M-120M model would still fit in 12.9GB with fp16 and the current batch size of 8.
- At minimum, increase `model_guided_ratio_high` from 0.60 to only 0.40 until the model is retrained on L11+ reference solutions (Phase 1 in project_constitution.md).
- Implement the Phase 1 Knowledge Injection: inject reference solutions for L11-L13 into the training data as fixed examples (not just champion traces). This teaches the model the correct program structure for high-level tasks regardless of what the champion has achieved.

---

### D5. Checkpoint Rollback Design Flaw (Medium Impact, Partially Fixed)

**Location:** `meta_controller.py`, documented in `lessons_learned.md:22–53`

**Problem:** The rollback architecture has a fundamental flaw: `checkpoint_best.pt` is used as both the *retrain input* and the *retrain output*. When the gate rejects a new checkpoint:
1. Rollback tries to copy `checkpoint_best.pt.regressed → checkpoint_best.pt`
2. If the retrain wrote to `checkpoint_best.pt` directly, the regressed backup *is* the current file
3. `shutil.copy2(src, dst)` with `src == dst` silently no-ops
4. The "rollback" completes with no effect — the model is not actually restored

The same-file guard was added (documented fix), but the underlying design remains: the retrain output and best checkpoint should be completely separate paths. Until the retrain pipeline is refactored to write to `checkpoint_cycle_N.pt` and only copy to `checkpoint_best.pt` after gate acceptance, this risk persists.

**Path forward:** Refactor `run_multi_cycle.py` so that:
- Retrain ALWAYS writes to `checkpoint_cycle_N.pt`
- Gate acceptance copies to `checkpoint_best.pt` AFTER validation
- The rollback path copies FROM `checkpoint_best.pt` (never `checkpoint_best.pt.regressed`)

---

## Part 2 — Architectural Limitations

### L1. Single Task Instance Per Generation — Overfitting Risk

**Location:** `evolution_engine.py:782–788`, `_refresh_task_instance()`

The task instance (`self.task_instance`) is only refreshed at curriculum advances, not between generations. If the task uses random inputs (registers, memory values), the same inputs are used for hundreds of generations. A champion can evolve to overfit to specific input values rather than implement the general algorithm.

For example, at L6 (sorted array), if `count = 5` for all 200 generations, the champion may hardcode the answer for length-5 arrays rather than implementing a general loop. This overfitting is invisible until the curriculum advances and the task changes.

**Fix:** Refresh `task_instance` every N generations (e.g., every 25). Use the same instance for an entire cycle but reseed it regularly. This forces the champion to generalize.

---

### L2. O(N²) Diversity Computation Every Generation

**Location:** `evolution_engine.py:1262–1280`, `_apply_diversity_maintenance()`

Pairwise Hamming distance is computed between all challengers every generation:
```python
for i in range(len(challengers)):
    for j in range(i + 1, len(challengers)):
        # per-word comparison
```
With population_size=49, this is 49×48/2 = 1,176 pairs per generation, each comparing variable-length program words. At 150 generations per cycle and 48 cycles per run, this is 8,467,200 pairwise comparisons per full run — for a metric that doesn't directly improve fitness.

**Fix:** Replace full pairwise computation with a MinHash sketch approximation. Compute a 32-bit sketch per program (XOR of instruction hashes), then diversity ≈ number of distinct sketches / population_size. O(N) instead of O(N²).

---

### L3. Multi-Level Regression Check Is Quadratically Expensive

**Location:** `evolution_engine.py:1682–1754`, `_update_champion()`

For each elite challenger, the champion anchor evaluates it against ALL previously mastered levels:
```python
for prev_level, (prev_fitness, _, prev_task) in self._level_scores.items():
    challenger_prev_fitness = self._evaluate_on_task(prog, prev_task)
```
At L10, this means each elite candidate requires 10 extra VM evaluations (L0-L9). With `elite_size = pop * elite_ratio * 2` ≈ 10 candidates, that's 100 extra VM calls per generation just for regression checking — on top of the regular 49 evaluations. This is 3× the evaluation budget.

**Fix:** Only check the immediately previous level (the current code does restrict to `prev_level == champion_level - 1` for the anchor violation, but still iterates all levels). Change the loop to only check `champion_level - 1` from the start, skipping older levels entirely.

---

### L4. Novelty/QD Selection Bias at Elite Stage

**Location:** `evolution_engine.py:1657–1680`, `_select_elite()`

Elite selection in stage 1 uses novelty/QD-blended fitness (the default). But champion update in stage 2 uses `raw_only=True` at max level. This creates a mismatch:

1. Stage 1: selects elite based on novelty score (behavioral distinctiveness)
2. Stage 2: champion is updated based on raw task score only

The best raw-task-score programs may not be in the elite set at all — they could have been filtered in stage 1 because they're behaviorally similar to existing archive entries (low novelty). The champion is never updated from these programs.

**Fix:** At max curriculum level, run stage 1 selection using `raw_only=True` to prevent the novelty bias from filtering out the strongest task-solving candidates.

---

### L5. MetaController max_per_level Overrides Cause Retrain Hangs

**Location:** `tasks.md:35–38`, `meta_controller.py`

The MetaController increases `max_per_level` as it detects stagnation (trying to give the model more data). But when it reaches `max_per_level=500`, the dataset grows to ~7,000 sequences (500 × 14 levels). At `batch_size=8`, that's 875 steps per epoch × 12 epochs = 10,500 gradient steps per retrain — roughly 4× longer than intended. This is documented as causing "Cycle 2 hang" in `tasks.md`.

The CLI override (`--retrain-max-per-level 200`) wins over the MetaController (verified at line 1595-1616 of `run_multi_cycle.py`), but this means the MetaController's adaptive behavior is permanently suppressed.

**Fix:** Separate MetaController's "stagnation countermeasure" from the per-level sample count. Instead of increasing max_per_level on stagnation, it should increase training epochs or learning rate — changes that don't grow the dataset unboundedly.

---

### L6. Meta-Evolution Optimizes the Wrong Target

**Location:** `meta_evolution.py:68–82`, `_compute_meta_fitness()`

Meta-evolution optimizes the static configuration (population_size, mutation_rate, elite_ratio, etc.) using 20-generation base runs. The project constitution explicitly states: "The bottleneck is generation quality, not curriculum mechanics or hyperparameters." Meta-evolution is solving the wrong problem.

The compute cost of meta-evolution (10 meta-genomes × 3 seeds × 20 gen × ~49 programs/gen evaluations) would be better spent on:
1. More training epochs on high-quality examples
2. Running Phase 1 knowledge injection (reference solution training)
3. Longer evolution cycles to find better champion traces

Additionally, 20-generation base runs are too short to distinguish good configs from lucky ones. Config D (the best known: pop=49, μ=0.584) was verified over 200 generations — meta-fitness from 20 generations cannot reliably identify it.

**Fix:** If meta-evolution is retained, increase base_generations to at least 100. Alternatively, repurpose the meta-evolution loop to optimize *reward signal composition* (novelty_weight, qd_bonus_weight, the fitness normalization scale) rather than evolutionary parameters.

---

### L7. Recurrent Buffer Eviction Is Not Quality-Aware

**Location:** `evolution_engine.py:1990–2009`

When the buffer reaches capacity, it evicts "the oldest trace from the most over-represented curriculum level." This can evict a high-fitness trace from L5 (most examples) in favor of keeping a zero-fitness trace from L12 (fewest examples). The quality of the evicted trace is never considered.

**Fix:** Evict the **lowest-fitness trace** from the most over-represented level:
```python
level_entries = [(i, ex) for i, ex in enumerate(existing) if ex.get("level") == most_level]
worst_idx = min(level_entries, key=lambda x: x[1].get("fitness", -999))[0]
existing.pop(worst_idx)
```

---

### L8. GPU Fallback Masks Kernel Bugs

**Location:** `evolution_engine.py:126–195`, `_evaluate_batch_gpu()`

The GPU multi-program kernel fallback silently switches to CPU on any exception:
```python
try:
    gpu_results = self._gpu_farm.evaluate_multi(programs, inputs_list, budget=budget)
    results = []
    for r in gpu_results:
        if r.get('success'):
            results.append(...)
        else:
            # Fallback to CPU for failed candidates  ← silent fallback
            result = _run_one_evaluation(...)
```

If the GPU kernel produces systematically wrong results for certain instruction sequences (e.g., off-by-one in shift operations, wrong sign extension), those results get reported as `success=False` and the CPU corrects them — but the logged fitness for the generation came from the GPU's pre-failure output. This creates a subtle fitness accounting error that accumulates across cycles.

**Fix:** Log and count GPU fallback events. If GPU fallback rate exceeds 5% in a generation, emit a warning and optionally disable GPU for that cycle while flagging for investigation.

---

## Part 3 — State Management Issues

### S1. No State Cleanup Between Runs

The strategy archive, novelty archive, QD archive, and MetaController state all persist as files between runs. A new run with different config picks up the old state, biasing the new run toward strategies that worked for the previous run's config. This was documented as causing v10.1 to stall at L2 for 32+ cycles.

The `run_v10.sh` script does NOT clean these files by default. The user must manually run `rm -f meta_controller_state.json strategy_archive_gen*.json` before a new run. This is fragile.

**Fix:** Add a `--clean-state` flag to `run_multi_cycle.py` that deletes all state files before starting. Make it warn (not error) if files don't exist.

---

### S2. Dashboard State Leaks Between Cycles

**Location:** `evolution_engine.py:299`, config field `dashboard_state_dir`

The dashboard state directory (`dashboard_state/`) accumulates per-generation JSON files. After 48 cycles × 300 generations each, this is 14,400 files. The cleanup in v11 (`_KEEP_CYCLE_CHECKPOINTS = 5`) only applies to checkpoint files, not dashboard state files.

**Fix:** Add dashboard state cleanup to `run_multi_cycle.py`'s end-of-cycle cleanup: keep only the last N generation state files.

---

## Part 4 — Correct Paths Forward (Priority Order)

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| P0 | **Fitness partial credit** (D1) — distance-weighted register scoring | Highest | Low |
| P0 | **Training signal quality gate** (D2) — only save fitness > 0 traces | Highest | Low |
| P1 | **Phase 1 Knowledge Injection** — inject L11-L13 reference solutions into training | High | Medium |
| P1 | **Task instance refresh** (L1) — reseed task every 25 generations | High | Low |
| P1 | **Recurrent buffer quality eviction** (L7) — evict lowest-fitness, not oldest | High | Low |
| P2 | **Retrain pipeline refactor** (D5) — separate retrain output from best checkpoint | High | Medium |
| P2 | **Fix elite selection bias** (L4) — use raw_only for stage-1 selection at max level | Medium | Low |
| P2 | **O(N²) → MinHash diversity** (L2) — approximate Hamming with sketches | Medium | Low |
| P3 | **MetaController max_per_level decoupling** (L5) — tune epochs/LR, not sample count | Medium | Medium |
| P3 | **GPU fallback logging** (L8) — count and warn on excessive CPU fallback | Medium | Low |
| P4 | **State cleanup flag** (S1) — `--clean-state` in `run_multi_cycle.py` | Low | Low |
| P4 | **Larger model** (D4) — 90M-120M parameter model at fp16 | High | High |

---

## Part 5 — Degradation Timeline (Reconstructed from Logs)

Based on the run history in `lessons_learned.md` and `v10_1_findings.md`:

**Cycles 1-5:** Model improves rapidly. Reference solutions seed good initial champions. Strategy diversity maintained (GROW + CROSSOVER + HYBRID). Positive fitness achievable.

**Cycles 6-15:** GROW monoculture begins forming. Short programs dominate elite. Model is retrained on increasingly short, simple programs. L8+ advancement slows.

**Cycles 15-30:** Strategy archive heavily biased toward GROW (or CROSSOVER). Curriculum stalls at L9-L11. Recurrent buffer fills with zero-fitness and negative-fitness traces. Model quality degrades.

**Cycles 30+:** Model generates programs that are stylistically similar to previous champions but rarely advance. MetaController detects stagnation and adjusts LR/freeze, but cannot compensate for degraded training data.

---

## Appendix: Quick Wins (Can Be Applied Today)

**1. Add fitness floor to recurrent buffer** — 5 lines in `_try_recurrent_feedback()`:
```python
if self.champion_fitness <= 0 and self.generations_stalled < 20:
    return False  # Don't save failing champions unless deeply stalled
```

**2. Refresh task instance more frequently** — 1 line in `run()` loop:
```python
if gen % 25 == 0:
    self._refresh_task_instance()
```

**3. State cleanup in run script** — Add to `run_v10.sh` / `v10_1_runner.sh` before starting:
```bash
rm -f /root/milestone2/meta_controller_state.json
rm -f /root/milestone2/strategy_archive_gen*.json
rm -f /root/milestone2/novelty_archive_gen*.json
```

**4. Reduce diversity computation cost** — In `_apply_diversity_maintenance()`, sample 20 random pairs instead of all N(N-1)/2:
```python
sample_pairs = [(i, j) for i, j in random.sample(all_pairs, min(20, len(all_pairs)))]
```

**5. Log GPU fallback rate** — Add counter at line 156 of `_evaluate_batch_gpu()`:
```python
self._gpu_fallback_count = getattr(self, '_gpu_fallback_count', 0) + 1
if self._gpu_fallback_count % 100 == 0:
    print(f"[GPU] {self._gpu_fallback_count} fallbacks to CPU so far")
```

---

## Supplemental Audit — 2026-06-22 (External Auditor)

### Additional Implementation Bugs Found

#### HIGH — `skip_evolution` variable referenced without assignment
- **File:** `evolution_engine.py` — main evolution loop
- **Bug:** The `skip_evolution` variable is referenced in conditional logic but may not be assigned in all code paths. If the variable is not defined when the condition is checked, Python raises `UnboundLocalError`, crashing the entire evolution cycle mid-execution.
- **Fix:** Initialize `skip_evolution = False` at the top of the function, or restructure the conditional logic.

#### HIGH — `execute_rollback` signature mismatch
- **File:** `meta_controller.py` / `evolution_engine.py`
- **Bug:** `execute_rollback` is called with arguments that don't match its function signature. If the number or type of arguments doesn't match, the rollback fails silently (caught by a broad except) or crashes the meta-controller, leaving the system in an inconsistent state with no way to recover.
- **Fix:** Audit all call sites against the function signature and fix the mismatch.

#### HIGH — Cycle state not persisted atomically
- **File:** `meta_controller.py` — `meta_controller_state.json` writes
- **Bug:** State is written directly to `meta_controller_state.json` without atomic writes (no temp file + rename). If the process crashes during write, the JSON file is corrupted and the entire evolution history is lost. With 48+ cycles of evolution, this is a catastrophic data loss scenario.
- **Fix:** Use `tempfile.mkstemp` + `shutil.move` for atomic state persistence.

#### HIGH — No recovery mechanism for failed cycles
- **File:** `run_multi_cycle.py` — multi-cycle orchestration
- **Bug:** If a cycle fails (exception, OOM, timeout), the orchestrator continues to the next cycle without checking the exit code or cleaning up partial state. The failed cycle's partial outputs remain in the cycle directory, potentially corrupting the next cycle's input.
- **Fix:** Check exit codes, clean up partial outputs on failure, and implement retry logic with backoff.

#### HIGH — Evolution engine has no maximum iteration guard
- **File:** `evolution_engine.py` — main loop
- **Bug:** The evolution loop has no hard maximum iteration count. If the convergence condition is never met (e.g., due to a fitness function bug that always returns the same score), the engine runs indefinitely.
- **Fix:** Add a `MAX_CYCLES` constant and enforce it.

#### MEDIUM — Mutation proposer has no validation on generated code
- **File:** `mutation_proposer.py`
- **Bug:** Mutations are proposed and applied without syntax validation. If the LLM generates code with syntax errors, the mutation is applied to the genome, causing the next evaluation to crash. The error is caught but the mutation is not rolled back.
- **Fix:** Validate proposed mutations with `py_compile` before applying.

#### MEDIUM — `meta_genome.py` — genome serialization may lose precision
- **File:** `meta_genome.py`
- **Bug:** Floating-point parameters in the genome are serialized to JSON. `json.dumps` may lose precision for very small or very large floats, causing genome parameters to drift across save/load cycles.
- **Fix:** Use `repr()` for float serialization or add precision-preserving encoding.

#### MEDIUM — Dashboard has no error boundary
- **File:** `v11_dashboard.py`
- **Bug:** The dashboard reads `multi_cycle_results.jsonl` and `meta_controller_state.json` directly. If either file is corrupted, the dashboard crashes with an unhandled exception — precisely when you need it most.
- **Fix:** Add try/except around file reads and display a degraded view.

#### MEDIUM — Strategy archive grows unboundedly
- **File:** `strategy_archive.py`
- **Bug:** Every strategy is archived without deduplication or eviction. Over many cycles, the archive grows without bound.
- **Fix:** Add deduplication (hash-based) and a maximum archive size with LRU eviction.

#### MEDIUM — No file locking on cycle directories
- **File:** `run_multi_cycle.py`, `meta_controller.py`
- **Bug:** If two instances of the engine run simultaneously, they can both write to the same cycle directory, corrupting results. There is no file lock or PID-based guard.
- **Fix:** Use a PID file or `fcntl.flock` to prevent concurrent execution.

---

## Deep File-by-File Audit — Additional Findings (2026-06-22)

### HIGH — Strategy archive records ALL strategies as equally successful (fitness always 0.0)
- **File:** `evolution_engine.py:1621-1661`
- **Bug:** `_record_strategy_outcomes` receives `results` from `_evaluate_batch`, which are trace dicts from `_eval_result_to_trace`. That function hardcodes `"fitness": 0.0` in every returned dict (line 751). Therefore `r.get("fitness", ...)` always returns 0.0, the threshold at line 1649 is 0.0, and `fit >= threshold` is always True. Every strategy is marked as a success. The strategy archive learns nothing.
- **Fix:** In `_record_strategy_outcomes`, compute fitness via `_compute_fitness(prog, trace)` instead of reading from the raw trace dict. Or pass the programs alongside the results and compute fitness there.

### HIGH — Checkpoint rejection rollback crashes with TypeError
- **File:** `run_multi_cycle.py:1543`
- **Bug:** `meta_controller.execute_rollback(checkpoint, use_snapshot=True)` — but `MetaController.execute_rollback` only accepts `(self, current_checkpoint: str)`. The `use_snapshot=True` keyword argument causes a `TypeError` every time a checkpoint is rejected. This means the rollback-on-rejection code path never works — it always crashes and falls through silently.
- **Fix:** Either remove the `use_snapshot=True` argument, or implement snapshot support in `MetaController.execute_rollback`.

### HIGH — Missing function `parse_grad_norms_from_retrain_log`
- **File:** `run_multi_cycle.py:2384`
- **Bug:** `from meta_controller import parse_grad_norms_from_retrain_log` — this function does not exist in `meta_controller.py`. The import is inside a try/except so it won't crash, but the gradient norms diagnostic is silently skipped every cycle.
- **Fix:** Implement `parse_grad_norms_from_retrain_log` in `meta_controller.py`, or remove the dead import.

### HIGH — Failed evaluations with empty error get status "ok" and full fitness scoring
- **File:** `evolution_engine.py:741`
- **Bug:** `"status": "ok" if success else ("crash" if error else "ok")` — when `success=False` and `error=""` (happens for some failure modes like VM timeout without an error string), the status is "ok". The fitness function then scores this as a normal evaluation. The champion could be replaced by a crashed program.
- **Fix:** Change to `"status": "ok" if success else "crash"`.

### HIGH — Seed evaluation clears ALL accumulated fitnesses on single-seed failure
- **File:** `meta_evolution.py:561`
- **Bug:** If any single seed in a genome's multi-seed ensemble fails to produce a summary, `ensemble_fitnesses` is replaced with `[-inf]`, discarding any successful seeds that already evaluated. A 3-seed ensemble where 2 seeds succeed and 1 fails gets scored as `-inf`.
- **Fix:** Use `continue` instead of replacing the entire list and breaking.

### HIGH — Hardcoded `/root/` paths will fail on non-root deployments
- **File:** `run_multi_cycle.py:50-53`
- **Bug:** `sys.path.insert(0, "/root/milestone0/logit-masking")` and similar paths. Also at line 1894: `_cleanup_stale_checkpoints` hardcodes `/root/milestone2/../milestone1/checkpoints`. On this user's machine (WSL2), these paths resolve to nothing.
- **Fix:** Use `os.path.join(SCRIPT_DIR, ...)` or resolve paths relative to the project root.

### HIGH — `skip_evolution` variable set but never checked
- **File:** `run_multi_cycle.py:1568`
- **Bug:** `skip_evolution = True` is set inside the "Reset-and-Relearn" block but the main loop never checks it before calling `run_evolution`. The flag is dead code, and the intended "skip evolution" behavior never happens.
- **Fix:** Add a check before the `run_evolution` call: `if not skip_evolution:`.

### MEDIUM — `detect_stagnation` never updates `_last_stagnation`
- **File:** `meta_controller.py:184,281-319`
- **Bug:** The field `self._last_stagnation` is initialized to `StagnationResult.NORMAL` but is never updated by `detect_stagnation()`. `get_state_summary` reads it and always returns "normal".
- **Fix:** Add `self._last_stagnation = result` before the return in `detect_stagnation`.

### MEDIUM — Retry config crash: `dict(SimpleNamespace)`
- **File:** `run_multi_cycle.py:1594`
- **Bug:** `retry_config = dict(merged_retrain_config)` — when `merged_retrain_config` is a `SimpleNamespace`, `dict(...)` raises `TypeError` because `SimpleNamespace` is not iterable over key-value pairs.
- **Fix:** Use `vars(merged_retrain_config)` instead.

### MEDIUM — `StagnationResult` has no "level_regression" value but code checks for it
- **File:** `run_multi_cycle.py:1681`
- **Bug:** `meta_controller.detect_stagnation().value == "level_regression"` — but `StagnationResult` has no "level_regression" value. This comparison is always False.
- **Fix:** Either add a LEVEL_REGRESSION enum value or remove the dead code.

### MEDIUM — Level 10 reference solution uses SLTU but task requires signed max
- **File:** `tasks_rv32i.py:628`
- **Bug:** The Level 10 reference solution uses `SLTU` (Set Less Than Unsigned) for max-finding. But the task requires finding the maximum of signed values. With `SLTU`, negative numbers appear "greater" than positive numbers, producing wrong results.
- **Fix:** Change `SLTU` to `SLT`.

### MEDIUM — `_encode_inst` uses "name" key but reference solutions use "name" key with lowercase check
- **File:** `mutation_proposer.py:1171`
- **Bug:** `inst.get("opcode", "").upper()` — but reference solution dicts use key `"name"` (e.g., `{"name": "ADDI", ...}`), not `"opcode"`. `_encode_inst` always returns None when called with reference solution dicts, meaning `_generate_initial_champion` and `_replace_with_model_guided` silently fail to encode reference solutions.
- **Fix:** Change to `inst.get("name", inst.get("opcode", ""))`.

### MEDIUM — `or True` debug leftover forces verbose logging every cycle
- **File:** `run_multi_cycle.py:1856`
- **Bug:** `if freed > 50 or True:` — the `or True` makes the condition always true, meaning GPU memory stats are logged every cycle.
- **Fix:** Remove `or True`.

### LOW — `v11_dashboard.py` --log argument parsing
- **File:** `v11_dashboard.py:305-308`
- **Bug:** The argument parsing uses `enumerate(sys.argv[1:])` which works correctly but is fragile and hard to maintain. Consider using `argparse` instead.
- **Fix:** Refactor to use `argparse`.
