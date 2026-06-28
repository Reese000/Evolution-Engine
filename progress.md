# Evolution Engine — Autonomous Runner Progress

**Date:** 2026-06-27  
**Commit:** dfb480d  
**Tests:** 42/42 passing

---

## Phase 1: Fix Foundations ✅

### STEP 1 — VM Farm Import Chain ✅
- **Status:** Already working — `vm_farm.py` (thin shim) exists at project root
- **Verified:** `python3 -c 'from vm_farm import EvalResult; print("OK")'` → OK
- **Architecture:** `vm_farm.py` defines `EvalResult` namedtuple + `_run_one_evaluation()` that delegates to `gpu_vm_farm._execute_vm()` (pure Python RV32I VM)
- **No changes needed** — the import chain is clean

### STEP 2 — Config Files ✅
- `config/engine_config.json`: Already existed with full sections for `evolution`, `retrain`, `meta_controller`, `autonomous_runner`
- `config/tool_library.json`: Already existed with complete RV32I ISA instruction definitions (37 instructions) and register/memory specs
- **Validated:** Both files parse as valid JSON with all required fields

### STEP 3 — Path Dependencies ✅
- **Created:** `paths.py` — centralized project path resolution module
  - `PROJECT_ROOT`, `MILESTONE0_DIR`, `MILESTONE1_DIR`, `MILESTONE3_DIR`
  - `MILESTONE0_LOGIT_MASKING`, `MILESTONE0_VM_FARM`, `MILESTONE0_GPU_VM`
  - `M1_CHECKPOINTS`, `M1_RECURRENT`, `M1_TRAIN_MODEL`
  - `ENGINE_CONFIG`, `TOOL_LIBRARY`, `RESULTS_FILE`, `META_CONTROLLER_STATE`
  - `ensure_project_on_path()` — idempotent sys.path setup
- **Updated:** `evolution_engine.py` — replaced 15-line sys.path.insert block with `from paths import ...; ensure_project_on_path()`
- **Updated:** `mutation_proposer.py` — replaced 4-line sys.path hack with `from paths import MILESTONE0_LOGIT_MASKING, ensure_project_on_path`

---

## Phase 2: Autonomous Runner ✅

### STEP 4 — Build autonomous_runner.py ✅
- **Created:** `autonomous_runner.py` (547 lines)

#### AutonomousRunner Class
```python
class AutonomousRunner:
    def run(self)              # Main loop: cycle → retrain → cycle
    def run_cycle(cycle)       # Single evolution + retrain cycle
    def save_state()           # Persist to runner_state.json
    def load_state()           # Resume from saved state
    def should_stop()          # Check max_cycles, target_fitness, stop file
    def attempt_recovery()     # Reload checkpoint on error
    def graceful_stop()        # Finish current cycle, save, exit
    def get_health_status()    # Return cycles_run, last_fitness, uptime, errors
    def log_error()            # Track error counts
    def _handle_oom()          # Reduce population 25% on MemoryError
```

#### HealthMonitor Class
- Writes `health.json` with status, uptime, cycles_completed, current_fitness, current_level, errors
- External tools can check `health.json` to verify system is alive

#### CLI Interface
```
python autonomous_runner.py                       # Run with defaults
python autonomous_runner.py --max-cycles 10       # Override max cycles
python autonomous_runner.py --target-fitness 200   # Stop at target
python autonomous_runner.py --resume               # Resume from saved state
python autonomous_runner.py --dry-run              # Validate config only
python autonomous_runner.py --status               # Show health status
```

#### Crash Recovery
- State saved to `runner_state.json` after every cycle
- On restart, loads last state and resumes from saved cycle number
- Corrupted state → clean restart with warning

#### Graceful Shutdown
- SIGINT/SIGTERM handlers set shutdown flag
- Current cycle completes before stopping
- State saved before exit

---

## Phase 5: Fitness Landscape Fixes (Verified Existing) ✅

### D1: Deceptive Fitness Landscape → Partial Credit
- **Location:** `evolution_engine.py:_compute_fitness()`
- **Mechanism:** Distance-weighted partial credit per register: `1.0 / (1.0 + diff/max_diff)`
- **Effect:** Program with wrong value gets partial score instead of 0/1 binary
- **Test:** `test_partial_credit_is_positive` ✅

### D2: Training Signal Contamination
- **Location:** `evolution_engine.py:_try_recurrent_feedback()`
- **Mechanism:** Filters negative-fitness traces from recurrent training buffer
- **Exception:** Deep stall (>=50 gen) allows low-quality traces to prevent starvation
- **Test:** `test_negative_fitness_filtered`, `test_deep_stall_allows_low_fitness` ✅

### D3: Strategy Monoculture
- **Locations:** `evolution_engine.py:_choose_mutation_strategy()`, `strategy_archive.py`
- **Mechanisms:**
  - `max_consecutive_same_strategy=30` (halved at L5+)
  - Archive decay on level advance: 50% of success records flipped to False
  - UCB exploration bonus in strategy selection
  - Skip non-GROW strategies after cap hit
- **Tests:** `test_record_and_recommend`, `test_skip_strategy`, `test_decay_level_records` ✅

---

## Test Results

```
42 passed in 1.65s

TestVMFarmImport (5 tests):      ✅ EvalResult, _run_one_evaluation, _execute_vm
TestPaths (4 tests):             ✅ PROJECT_ROOT, ensure_project_on_path, config paths
TestConfigFiles (4 tests):       ✅ Valid JSON, required fields, AR section
TestEvolutionEngineImport (4):   ✅ Config, engine, VMFarm, QD axes
TestMutationProposerImport (4):  ✅ Import, enum, trace analysis, perturbation
TestStrategyArchive (3 tests):   ✅ Record/recommend, skip, decay
TestFitnessLandscape (3 tests):  ✅ Partial credit, crash penalty, bloat
TestTrainingSignalFilter (3):    ✅ Negative filter, positive pass, deep stall
TestAutonomousRunner (9 tests):  ✅ Init, state, save/load, stop conditions,
                                  health monitor, shutdown signal, OOM
TestMetaController (2 tests):    ✅ Import, stagnation detection
TestChampionAnchor (1 test):     ✅ Save/load roundtrip
```

---

## Files Changed/Created

| File | Status | Lines |
|------|--------|-------|
| `paths.py` | **NEW** | 83 |
| `autonomous_runner.py` | **NEW** | 547 |
| `tests/__init__.py` | **NEW** | 0 |
| `tests/test_phase1_phase2.py` | **NEW** | 469 |
| `evolution_engine.py` | Modified | +6/-14 (path setup) |
| `mutation_proposer.py` | Modified | +3/-4 (path setup) |

---

## Next Steps (Phase 3+)

- [ ] Phase 3: Structured logging (replace print() with logger)
- [ ] Phase 4: GPU VM farm integration testing
- [ ] Phase 5: Cycle resilience (OOM/retrain failure recovery)
- [ ] Phase 6: Unified configuration (config.py with env var overrides)
- [ ] Phase 7: Dashboard update (read from health.json + metrics)
