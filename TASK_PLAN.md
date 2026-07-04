# Evolution Engine: Full Autonomy Task Plan

## Goal

Make the Evolution Engine run fully autonomously -- continuous evolution/retrain cycles with zero human intervention, automatic recovery from failures, and structured progress tracking.

---

## Phase 1: Fix Broken Foundations (Critical)

These must be resolved first because the engine cannot run at all without them.

### 1.1 Fix VM Farm Import Chain

**Problem**: `evolution_engine.py` imports `from vm_farm import EvalResult, _run_one_evaluation` but no `vm_farm.py` exists at the project root. The file referenced is `milestone0/vm-farm/vm_farm.py` (or similar), and the path setup in `evolution_engine.py` lines 25-39 adds several directories to `sys.path` but none of them resolve to a working `vm_farm` module.

**Action**:
- Locate the actual `vm_farm.py` module (check `milestone0/` subdirectories, `simulation/`, or any path that was previously used)
- Either symlink it to the project root or fix the path setup in `evolution_engine.py` to point to the correct location
- Verify `EvalResult` and `_run_one_evaluation` are importable
- If the CPU vm_farm is missing, create a minimal shim that delegates to `gpu_vm_farm.py` (which does exist)

**Files**: `evolution_engine.py`, `vm_farm.py` (new or moved)

### 1.2 Create Missing Config Files

**Problem**: `config/engine_config.json` and `config/tool_library.json` do not exist. The `CLAUDE.md` references them as part of the architecture.

**Action**:
- Create `config/` directory
- Create `config/engine_config.json` with sensible defaults extracted from hardcoded values in `evolution_engine.py` and `meta_controller.py`:
  - population_size, elite_ratio, mutation_rate, crossover_probability
  - max_generations_per_cycle, max_cycles
  - retrain params: learning_rate, epochs, batch_size, freeze_layers
  - curriculum: start_level, max_level
  - strategy weights
- Create `config/tool_library.json` with the standard endmill definitions used in evaluation

**Files**: `config/engine_config.json`, `config/tool_library.json`

### 1.3 Fix Path Dependencies

**Problem**: Multiple files use relative path hacks (`SCRIPT_DIR`, `sys.path.insert`) to find sibling modules. These are fragile and break when running from different working directories.

**Action**:
- Add a `paths.py` module at the project root that defines all project-relative paths consistently
- Replace scattered `sys.path.insert` calls with imports from `paths`
- Ensure all modules can be imported from any working directory

**Files**: `paths.py` (new), `evolution_engine.py`, `mutation_proposer.py`, `meta_controller.py`

---

## Phase 2: Autonomous Runner

### 2.1 Build `autonomous_runner.py`

**Problem**: Nothing in the project runs cycles continuously with proper error handling, state persistence, and recovery. `run_multi_cycle.py` does multi-cycle but lacks daemon mode, crash recovery, and health monitoring.

**Action** -- Create `autonomous_runner.py` with:

```
class AutonomousRunner:
    """Runs the evolution/retrain loop indefinitely until:
    - max_cycles reached
    - target fitness achieved
    - manual stop signal
    """

    def __init__(self, config_path="config/engine_config.json"):
        # Load config, initialize MetaController, load state

    def run(self):
        """Main loop: cycle -> retrain -> cycle -> ..."""
        while not self.should_stop():
            try:
                self.run_cycle()
            except OOMError:
                self.handle_oom()
            except KeyboardInterrupt:
                self.graceful_stop()
            except Exception as e:
                self.log_error(e)
                self.attempt_recovery()
            self.save_state()

    def run_cycle(self):
        """Single evolution cycle + retrain."""

    def save_state(self):
        """Persist runner state to disk after every cycle."""

    def load_state(self):
        """Resume from last saved state on restart."""

    def should_stop(self):
        """Check stop conditions: max cycles, target fitness, stop file."""

    def attempt_recovery(self):
        """Try to recover from errors: reset population, reload checkpoint."""

    def get_health_status(self):
        """Return current health: cycles_run, last_fitness, uptime, errors."""
```

**Key design decisions**:
- State file: `runner_state.json` with cycle number, last fitness, last level, error count, timestamps
- Stop conditions: configurable max_cycles, target_fitness, or `touch stop` file
- After each cycle: save state, update dashboard, write to metrics log
- Crash recovery: on restart, load state, check if last cycle completed, resume from there

**File**: `autonomous_runner.py`

### 2.2 Integrate with MetaController

**Action**:
- `AutonomousRunner` should use `MetaController` for:
  - Checkpoint gating (don't accept regressed models)
  - Retrain parameter tuning (LR, epochs, freeze layers per cycle)
  - Stagnation detection (switch strategies when stuck)
  - Circuit breaker rollback (revert to best checkpoint on regression)
- Feed telemetry from each cycle into `MetaController.update_cycle()`
- Respect `MetaController.tune_retrain_params()` for retrain configuration

**Files**: `autonomous_runner.py`, `meta_controller.py` (minor integration)

### 2.3 Add Graceful Shutdown

**Action**:
- Signal handler for SIGINT/SIGTERM that sets a shutdown flag
- Runner completes current cycle before stopping
- Save state before exit
- On next start, resume from saved state

**File**: `autonomous_runner.py`

---

## Phase 3: Proper Logging and Metrics

### 3.1 Structured Logging

**Problem**: Scattered `print()` statements throughout the codebase. No log levels, no timestamps, no file output.

**Action**:
- Create `logging_config.py` that sets up Python logging with:
  - Console handler (INFO level, colored)
  - File handler (`evolution.log`, DEBUG level, rotating)
  - Structured format: `[timestamp] [level] [component] message`
- Replace `print()` calls in key files with `logger.info()` / `logger.debug()` / `logger.error()`
- Priority files: `evolution_engine.py`, `meta_controller.py`, `run_multi_cycle.py`

**File**: `logging_config.py` (new), plus edits to existing files

### 3.2 Metrics Collection

**Problem**: No persistent tracking of fitness, level, diversity, val_loss across cycles. The meta-controller has telemetry but it's not persisted in a queryable format.

**Action**:
- Create `metrics.py` with a `MetricsCollector` class:

```python
class MetricsCollector:
    """Writes cycle metrics to JSONL for analysis."""

    def __init__(self, path="metrics/cycles.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record_cycle(self, cycle_data: dict):
        """Append one cycle's metrics as a JSON line."""

    def get_history(self, last_n=None):
        """Read back metrics history."""

    def get_trend(self, metric, window=10):
        """Compute rolling average trend for a metric."""
```

- Each cycle writes: cycle number, fitness, level, val_loss, diversity, strategy used, time taken, population size, champion length
- `AutonomousRunner` calls `metrics.record_cycle()` after each cycle
- Dashboard reads from metrics file for live display

**File**: `metrics.py` (new)

### 3.3 Health Monitoring

**Action**:
- Create `health.py` that writes a `health.json` file with:
  - `status`: "running" | "idle" | "error" | "stopped"
  - `last_cycle`: timestamp of last completed cycle
  - `uptime`: seconds since start
  - `cycles_completed`: count
  - `current_fitness`: latest fitness value
  - `current_level`: latest curriculum level
  - `errors`: recent error count
- Updated after each cycle by `AutonomousRunner`
- External tools can check `health.json` to verify the system is alive

**File**: `health.py` (new)

---

## Phase 4: GPU VM Integration

### 4.1 Wire GPU VM into Evolution Engine

**Problem**: `gpu_vm_farm.py` exists with CuPy CUDA acceleration but `evolution_engine.py` doesn't use it. The engine imports from the missing `vm_farm` module.

**Action**:
- Modify `evolution_engine.py` to:
  - Try importing `gpu_vm_farm.GpuVMFarm` first
  - Fall back to CPU `vm_farm` if CUDA unavailable
  - Use GPU batch evaluation for all candidate evaluation
- Create a unified `VMFactory` that returns the best available backend:

```python
class VMFactory:
    @staticmethod
    def create(use_gpu=True):
        if use_gpu and HAS_CUDA:
            return gpu_vm_farm.GpuVMFarm(...)
        return vm_farm.CpuVMFarm(...)
```

**Files**: `evolution_engine.py`, `vm_farm.py` (new shim or moved)

### 4.2 GPU Memory Management

**Action**:
- Add CUDA memory cleanup between evaluation batches
- Monitor VRAM usage and fall back to CPU if GPU is near capacity
- Add `torch.cuda.empty_cache()` calls after large batch evaluations
- Log GPU memory usage per cycle

**Files**: `evolution_engine.py`, `gpu_vm_farm.py`

---

## Phase 5: Cycle Resilience

### 5.1 State Persistence

**Problem**: `meta_controller_state.json` exists but is incomplete. Strategy archives from previous runs cause monoculture bugs (documented in `lessons_learned.md`).

**Action**:
- Define a complete `cycle_state.json` schema:

```json
{
  "version": 2,
  "current_cycle": 11,
  "best_fitness": -0.085,
  "best_level": 2,
  "best_checkpoint": "milestone1/checkpoints/checkpoint_best.pt",
  "champion_program": [...],
  "strategy_archive": {...},
  "diversity_score": 0.15,
  "consecutive_stalls": 0,
  "last_cycle_timestamp": "...",
  "error_count": 0
}
```

- `AutonomousRunner` saves this after every cycle
- On startup, validate state file integrity (checksum)
- Corrupted state triggers a clean restart from last known good state

**File**: `autonomous_runner.py`, new state schema

### 5.2 Automatic Cleanup Between Cycles

**Problem**: Stale state files from previous runs cause strategy monoculture (documented in `lessons_learned.md` as "Stale State Files Corrupt v10.1 Deployment").

**Action**:
- Add `cleanup_stale_state()` function called at startup:
  - Remove stale strategy archives older than the current cycle
  - Remove stale novelty/QD archives
  - Reset dashboard state files
  - Log what was cleaned up
- Never automatically delete `checkpoint_best.pt` (sacred per constitution)
- Archive old cycle data to `archive/` instead of deleting

**File**: `autonomous_runner.py`

### 5.3 OOM Recovery

**Problem**: GPU OOM is common with large populations and long programs. Currently no automatic recovery.

**Action**:
- Catch `torch.cuda.OutOfMemoryError` and `cupy.cuda.memory.OutOfMemoryError`
- On OOM:
  1. Reduce population size by 25%
  2. Reduce max sequence length if applicable
  3. Clear GPU cache
  4. Retry the cycle with smaller parameters
  5. If still OOM after 3 retries, fall back to CPU evaluation
  6. Log the OOM event with GPU memory stats
- If population drops below minimum viable size, flag for human review

**File**: `autonomous_runner.py`

### 5.4 Retrain Failure Recovery

**Problem**: Retrain can fail (loss divergence, NaN gradients, timeout). Currently no recovery.

**Action**:
- Wrap retrain in try/except with specific handlers:
  - NaN loss: reduce learning rate by 50%, retry
  - Loss divergence (3x increase): load best checkpoint, retry with lower LR
  - Timeout: kill training, use previous model
  - CUDA error: fall back to CPU training
- After 3 failed retrain attempts, skip retrain for this cycle and continue evolution with current model
- Log all retrain failures with diagnostic info

**File**: `autonomous_runner.py`

---

## Phase 6: Configuration and Startup

### 6.1 Unified Configuration

**Action**:
- Create `config.py` that loads and validates configuration:

```python
class Config:
    """Central configuration loaded from JSON + env vars + CLI args."""

    def __init__(self, config_path="config/engine_config.json"):
        self.data = self._load_defaults()
        self._load_file(config_path)
        self._apply_env_overrides()
        self._validate()

    def _load_defaults(self):
        """Hardcoded safe defaults."""

    def _apply_env_overrides(self):
        """ENV vars override file config: EE_POPULATION_SIZE, EE_MAX_CYCLES, etc."""

    def _validate(self):
        """Check constraints: population > 0, LR > 0, etc."""
```

- All config values accessible via `config.evolution.population_size` etc.
- Environment variable overrides for containerized deployment
- Validation at startup with clear error messages

**File**: `config.py` (new)

### 6.2 CLI Interface

**Action**:
- Add argparse to `autonomous_runner.py`:

```
python autonomous_runner.py                    # Run with defaults
python autonomous_runner.py --config custom.json  # Custom config
python autonomous_runner.py --max-cycles 50    # Override max cycles
python autonomous_runner.py --target-fitness 0.5  # Stop at target
python autonomous_runner.py --resume           # Resume from saved state
python autonomous_runner.py --dry-run          # Validate config without running
python autonomous_runner.py --status           # Show current health status
```

**File**: `autonomous_runner.py`

### 6.3 Startup Validation

**Action**:
- On startup, validate:
  - All required Python packages are importable
  - CUDA is available (or warn if falling back to CPU)
  - Checkpoint files exist and are loadable
  - Disk space is sufficient for checkpoints (~200MB per checkpoint)
  - Config file is valid JSON with required fields
- Fail fast with clear error messages if validation fails

**File**: `autonomous_runner.py`

---

## Phase 7: Monitoring and Observability

### 7.1 Dashboard Update

**Problem**: `v11_dashboard.py` exists but may not work with current engine state.

**Action**:
- Update dashboard to read from `health.json` and `metrics/cycles.jsonl`
- Display: current cycle, fitness trend, level progress, strategy distribution, GPU usage, uptime
- Make it work with the new `MetricsCollector` output format
- Add a simple mode that just tails `evolution.log` for remote monitoring

**File**: `v11_dashboard.py`

### 7.2 Progress Reports

**Action**:
- Generate a `progress_report.json` after each cycle with:
  - Summary statistics (best fitness, best level, cycles completed)
  - Trend analysis (improving / plateau / regressing)
  - Strategy effectiveness breakdown
  - Resource usage (GPU hours, disk space)
- Optionally generate a human-readable markdown report

**File**: `autonomous_runner.py`

---

## Implementation Order

Execute in this sequence -- each phase builds on the previous:

1. **Phase 1** (Fix Foundations) -- Nothing works without this
2. **Phase 3.1** (Logging) -- Needed for debugging everything else
3. **Phase 2** (Autonomous Runner) -- The core deliverable
4. **Phase 4** (GPU Integration) -- Performance requirement
5. **Phase 3.2-3.3** (Metrics + Health) -- Observability
6. **Phase 5** (Resilience) -- Production hardening
7. **Phase 6** (Configuration) -- Polishing
8. **Phase 7** (Monitoring) -- Final layer

## Estimated Scope

| Phase | Files Changed | New Files | Complexity |
|-------|--------------|-----------|------------|
| 1. Fix Foundations | 2-3 | 2 | Medium |
| 2. Autonomous Runner | 1 | 1 | High |
| 3. Logging/Metrics | 3-4 | 2 | Medium |
| 4. GPU Integration | 2 | 1 | Medium |
| 5. Resilience | 1 | 0 | Medium |
| 6. Configuration | 1 | 1 | Low |
| 7. Monitoring | 1 | 0 | Low |

## Success Criteria

The system is fully autonomous when:

- [ ] `python autonomous_runner.py` starts and runs cycles without intervention
- [ ] Cycles complete with fitness logged to `metrics/cycles.jsonl`
- [ ] Retrain runs between cycles using best champion traces
- [ ] MetaController tunes parameters based on telemetry
- [ ] Stagnation triggers strategy rotation
- [ ] Regression triggers checkpoint rollback
- [ ] OOM triggers automatic recovery (population reduction or CPU fallback)
- [ ] `health.json` reflects current system status
- [ ] System can be stopped (`touch stop` or SIGTERM) and resumed cleanly
- [ ] Stale state from previous runs is cleaned up on startup
