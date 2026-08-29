# Evolution Engine

A self-evolving machine-code AI that continuously improves its ability to generate correct RISC-V RV32I programs through a closed feedback loop: evolve, retrain, evolve.

The system combines an evolutionary algorithm with a transformer-based code generator. A champion program is maintained and iteratively improved through mutation and selection. A transformer model learns from champion traces to generate increasingly complex programs across a 12-level curriculum. A meta-controller orchestrates the entire process, tuning hyperparameters in real-time and detecting stagnation or regression.

> **Architecture origin:** This project is the production implementation of the design documents and simulation prototypes in the companion repo [SelfEvolving-MachineCode-AI](https://github.com/Reese000/SelfEvolving-MachineCode-AI). That repo contains the project constitution, TinyVM-SIM simulation, and milestone-by-milestone blueprints. This repo is the living system.

## Architecture

The system has four major components:

**Evolution Engine** (`evolution_engine.py`) -- Maintains a champion RV32I program, generates challengers via mutation strategies (RANDOM, TASK_AWARE, DIRECTED, PERTURBATION_CAPPED, GROW, HYBRID, MODEL_GUIDED), evaluates them on the VM farm, and promotes fitter individuals. Supports concurrent evaluation with configurable population sizes.

**Meta-Controller** (`meta_controller.py`) -- Cross-cycle orchestrator that monitors telemetry (fitness, level, validation loss, diversity) and dynamically tunes retrain parameters, evolution hyperparameters, and strategy weights. Provides checkpoint gating (only accept improved models), stagnation detection, circuit-breaker rollback, and recovery mode.

**Mutation Proposer** (`mutation_proposer.py`) -- Generates candidate mutations by modifying RV32I instruction sequences. Includes trace analysis (identifies where a program diverges from correct execution), directed mutations (targeted fixes at divergence points), stochastic perturbation (random valid-instruction modifications), and model-guided generation (prefix-guided completion from the transformer).

**RV32I Transformer** (`milestone1/rv32i_model.py`) -- Decoder-only transformer with field-level tokenization for the full RV32I ISA (47 instructions, 6 encoding formats, 32 registers). Features include RoPE positional encoding, SwiGLU feed-forward, RMSNorm pre-norm, logit masking for valid field transitions, and tied embedding/head weights. Default config is ~6M parameters (256 hidden, 8 layers, 8 heads).

## Project Structure

```
Evolution Engine/
├── evolution_engine.py          # Core evolution: population, selection, fitness eval
├── meta_controller.py           # Cross-cycle orchestration, checkpoint gating, telemetry
├── mutation_proposer.py         # Mutation strategies + trace-guided directed mutation
| champion_anchor.py           # Champion preservation and restart logic
├── champion_archive.py         # Persistent champion archive with tier sampling
├── checkpoint_manager.py       # Atomic config-tagged checkpoint management
├── progression_tracker.py      # Cross-cycle metrics tracking and auto-tuning
├── cron_evolution_loop.py      # Cron-driven evolution loop / saturation detector
├── run_multi_cycle.py           # Multi-cycle evolution + retrain orchestrator
├── prefix_generator.py          # Prefix-guided generation for MODEL_GUIDED strategy
├── strategy_archive.py          # Strategy success tracking and selection
├── meta_evolution.py            # Meta-evolution of controller parameters
├── meta_genome.py               # Genome representation for meta-evolution
├── tasks_rv32i.py               # 12-level curriculum definition (L0-L11)
├── vm_farm.py                   # CPU RV32I evaluation (safety kernel)
├── gpu_vm_farm.py               # GPU-accelerated RV32I evaluation (CuPy CUDA)
├── v11_dashboard.py             # Live terminal dashboard
├── v9_monitor.py                # Monitoring utilities
├── milestone1/
│   ├── rv32i_model.py           # RV32I transformer model definition
│   ├── rv32i_mask.py            # RV32I encoding table and validation
│   ├── train_model.py           # Training pipeline (data loading, optimizer, scheduler)
│   ├── train_model_engine.py    # Engine-integrated training
│   ├── build_recurrent_dataset.py  # Dataset builder from champion traces
│   ├── synthetic_data.py        # Synthetic training data generation
│   ├── weight_validator.py      # Checkpoint validation utility
│   ├── evaluation/              # Evaluation pipeline (quality, latency, reports)
│   ├── training/                # Training submodules (checkpoint, schedule, loop)
│   ├── data/                    # Data loading, packing, augmentation
│   ├── model/                   # Model server for inference
│   └── checkpoints/             # Model checkpoints (gitignored)
├── cycle_1/ through cycle_11/   # Per-cycle evolution data and champion snapshots
├── champions/                   # Champion code snapshots
├── config/
│   ├── engine_config.json       # Default engine configuration
│   └── tool_library.json        # Tool definitions
├── tests/
│   ├── test_integration.py             # Full pipeline integration tests (40 tests)
│   ├── test_e2e_smoke.py               # End-to-end smoke test
│   ├── test_phase8_production.py       # Production hardening tests
│   ├── test_champion_archive_integration.py  # Champion archive lifecycle tests
│   ├── test_checkpoint_manager_integration.py  # Checkpoint manager tests
│   ├── test_progression_tracker_integration.py  # Progression/auto-tuning tests
│   ├── test_scale_integration.py       # Model scaling integration tests
│   ├── test_scale_validator.py         # Scale validator tests
│   ├── test_scaling_types.py           # Scaling type dataclass tests
│   ├── test_warm_start.py              # Warm-start training tests
│   ├── test_model_migrator.py          # Weight migration tests
│   ├── test_batch_evaluation.py        # Batch evaluation tests
│   ├── test_batch_prefix.py            # Batch prefix tests
│   ├── test_recurrent_data_filter.py   # Recurrent data filtering tests
│   ├── test_progression_snapshot.py    # Progression snapshot tests
│   └── test_round*.py                  # Round-specific improvement tests
└── lessons_learned.md           # Post-mortem analysis of failure modes
```

## How It Works

1. **Evolution**: The engine generates challenger programs by mutating the current champion. Each challenger is evaluated on a 12-level curriculum (L0: register assignment through L11: complex loops). The best challenger replaces the champion if it achieves higher fitness.

2. **Retraining**: Champion instruction traces are collected and used to fine-tune the RV32I transformer. The model learns to predict the next instruction field given a program prefix. Training uses position-aware weighted cross-entropy loss (1.5x weight on instruction-type positions).

3. **Meta-Control**: Between cycles, the meta-controller analyzes telemetry to detect stagnation, regression, or diversity collapse. It adjusts retrain learning rate, evolution strategy weights, population size, and mutation rates. Checkpoint gating prevents accepting regressed models.

4. **Prefix-Guided Generation**: At higher curriculum levels, the MODEL_GUIDED strategy uses the transformer to generate program continuations from champion prefixes, replacing blind mutation with learned generation.

## Verification Status (as of July 2026)

### Benchmarks

| Metric | Result | Method |
|--------|--------|--------|
| Feature benchmarks | **5/5 passed** | `benchmarks/verify_v13_features.py` — MG scheduler, temperature annealing, tournament selection with elitism, MG ratio, full evolution cycle |
| Average fitness/generation vs baseline | **5.3× faster** | `benchmarks/verify_improvements.py` |
| Per-generation speed | **1.6× faster** | `benchmarks/verify_improvements.py` |
| Test suite | **1450 passing / 0 failing** | `python -m pytest tests/` (Aug 2026; was 1321 in Jul 2026) |
| Retrain loop (evolve→retrain) | **Verified — real, decreasing `val_loss`** | 6-cycle CPU run: `val_loss` 4.35→3.64 monotonically (see Honest Caveats) |

### Verified-But-Not-Yet-Activated Fixes

The following fixes have been **verified as correct in isolation** but require a restart of the autonomous evolution cycle to actually activate in the running system:

| Fix | What it does | Activation |
|-----|-------------|------------|
| SEP-as-EOS | Fixes end-of-sequence handling in model-guided generation | Requires new model checkpoint |
| EBREAK | Proper halt instruction at program end | Requires champion restart |
| Task score gate | Prevents accepting degenerate programs that pass fewer levels | Requires MetaController restart |
| Checkpoint chain | Better checkpoint management with cycle-specific files | Requires checkpoint dir reset |
| LR cap | Prevents learning rate from growing unboundedly during retrain | Requires retrain restart |

**Status:** The fixes are merged into the codebase and all tests pass, but the engine was last started before these changes landed. A fresh `python autonomous_runner.py --resume` will activate them.

### Honest Caveats

- **(Resolved, Aug 2026)** The Jul 17 adversarial audit found that `val_loss` had been null across all 1,534 historical cycles — the retrain loop's training signal was never realized. This is now fixed and verified end-to-end: after bootstrapping a seed checkpoint (`python init_checkpoint.py`), a multi-cycle run produces a real, monotonically-decreasing `val_loss` (6-cycle CPU run: 4.35 → 3.64), and the checkpoint gate accepts each improved model. Root causes addressed: the retrain now runs in-process (the subprocess path was broken because `milestone1/train_model.py` has no CLI), the `checkpoint_acceptance_tolerance` default was aligned to config (1.05), and an epoch-cap bug that ignored `--retrain-max-epochs` for L5+ champions was fixed.
- Many test files in `tests/test_round*.py` are round-specific regression tests that may pass trivially. The core integration tests (`test_integration.py`, `test_e2e_smoke.py`, `test_phase8_production.py`) provide the real regression safety net.
- GPU acceleration requires CUDA (CuPy + RTX 4070 Ti); CPU mode works but is significantly slower for batch evaluation.

## Curriculum Levels

| Level | Task | Registers Used |
|-------|------|----------------|
| L0 | Register assignment | x3 |
| L1 | Simple ALU operation | x3, x4 |
| L2 | Two-input computation | x3, x4, x5 |
| L3 | Memory load/store | x3-x6 |
| L4 | Conditional execution | x3-x7 |
| L5 | Loop with counter | x3-x8 |
| L6 | Nested operations | x3-x9 |
| L7 | Array processing | x3-x10 |
| L8 | Multi-pass algorithm | x3-x11 |
| L9 | Recursive pattern | x3-x12 |
| L10 | Complex data flow | x3-x13 |
| L11 | Self-modifying code | x3-x14+ |

Each level uses a distinct output register, and verification checks all previous levels' outputs remain correct (monotonic cumulative design).

## Commands

```bash
# Autonomous evolution (recommended)
python autonomous_runner.py                  # Run with config/engine_config.json
python autonomous_runner.py --no-limit       # Run indefinitely (no cycle cap)
python autonomous_runner.py --max-cycles 10  # Override max cycles
python autonomous_runner.py --target-fitness 200  # Stop at target fitness
python autonomous_runner.py --resume         # Resume from saved state
python autonomous_runner.py --resume --no-limit  # Resume and keep going forever
python autonomous_runner.py --dry-run        # Validate config only
python autonomous_runner.py --status         # Show health status from health.json

# Evolution
python meta_controller.py                    # Single evolution cycle (demo)
python run_multi_cycle.py                    # Multi-cycle evolution + retrain
python run_multi_cycle.py --cycles 5 --gens 100  # Custom cycles/generations
python run_multi_cycle.py --resume           # Resume from last completed cycle
python v11_dashboard.py --log engine_run.log # Launch live dashboard

# Training
# Retrain normally runs in-process via the runners above. For standalone
# training use train_model_engine.py; milestone1/train_model.py is an
# import-only facade (no CLI) and does nothing if run directly.
python milestone1/train_model_engine.py      # Standalone training CLI (see --help)
python milestone1/build_recurrent_dataset.py # Build dataset from champion traces

# Testing
python -m pytest tests/ -v                   # Run full test suite (1400+ tests)

# Deployment
./deploy.sh                                  # Deploy via Docker
./deploy.sh --dry-run                        # Dry-run deployment
./status.sh                                  # Check running status
./status.sh --json                           # JSON status output
./stop.sh                                    # Graceful stop (SIGTERM)
./stop.sh --force                            # Force stop (SIGKILL)
```

## Continuous Operation

The evolution engine can run indefinitely for unattended long-term evolution. The `--no-limit` flag disables the cycle cap so the engine runs until stopped externally.

### Running Indefinitely

```bash
# Start from scratch with no cycle limit
python autonomous_runner.py --no-limit

# Resume a previous session and keep going forever
python autonomous_runner.py --resume --no-limit

# Resume with a custom cycle cap
python autonomous_runner.py --resume --max-cycles 500
```

When `--no-limit` is active, the engine ignores `max_cycles` from `config/engine_config.json` and runs until one of:
- Target fitness is reached
- A stop file is created (`touch stop`)
- A SIGINT/SIGTERM signal is received
- Too many consecutive errors occur (`max_errors_before_stop`)

### Stopping Gracefully

```bash
# Option 1: Create a stop file (safest — completes current cycle first)
touch stop

# Option 2: Send SIGTERM (completes current cycle)
kill -SIGTERM <pid>

# Option 3: Send SIGINT (Ctrl+C — completes current cycle)
kill -SIGINT <pid>
```

### Disk Management

Old cycle directories (`cycle_*`) are automatically pruned after each cycle, keeping the most recent 10. This prevents unbounded disk usage during long-running sessions. The `archive/` directory is used by startup cleanup to preserve a history of older cycles.

### Systemd Service (Auto-restart)

A service file is included for running as a persistent background service:

```bash
# Install the service
sudo cp evolution-engine.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable evolution-engine
sudo systemctl start evolution-engine

# Manage the service
sudo systemctl status evolution-engine
sudo systemctl stop evolution-engine
sudo journalctl -u evolution-engine -f   # View logs
```

The service uses `--resume --no-limit` so it automatically picks up where it left off after a crash or reboot, with a 30-second restart delay to avoid rapid crash loops.

### Monitoring

```bash
# Quick health check
python autonomous_runner.py --status

# Watch health.json directly
watch -n 5 cat health.json

# Tail the log
tail -f runner.log
```

`health.json` reports: status, uptime, cycles completed, current fitness, current level, errors, and target fitness.

### Self-Correction & Stagnation

The engine monitors fitness progress across cycles. When stagnation is detected (via MetaController telemetry analysis):

1. **Self-correction** (up to 5 attempts):
   - Attempt 1: Inject diversity (prune low-fitness QD archive entries)
   - Attempt 2: Boost mutation rate (2x)
   - Attempt 3: Reset champion from archive
   - Attempt 4: Full archive reset (preserve champion)
   - Attempt 5: Log diagnostics

2. **Exit with report** (after 15 stagnant cycles):
   - Writes `stagnation_report.json` with full history and recommendation
   - Exits with code 0 (clean shutdown)

The stagnation counter resets whenever progress is detected. Self-correction state and history are persisted across restarts via `runner_state.json`.

To adjust thresholds: set `max_stagnation_cycles` and `max_self_correction_attempts` in `config/engine_config.json` under the `autonomous_runner` section.

## Requirements

- Python 3.11+
- PyTorch 2.6+ (with CUDA)
- CuPy 14+ (for GPU-accelerated VM evaluation)
- Rust toolchain (for safety kernel compilation)

```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
pytest>=7.0.0
```

## Hardware

- **GPU**: NVIDIA RTX 4070 Ti (12.9 GB VRAM)
- **CUDA**: 12.x
- The GPU VM farm achieves 30-300x speedup over CPU evaluation for batch candidate assessment.

## Key Design Decisions

- **Field-level tokenization**: Instructions are encoded as sequences of field tokens (opcode, funct3, registers, immediates) rather than raw 32-bit words. This lets the model learn structural patterns in the ISA.
- **Logit masking**: At inference time, the model's output logits are masked to only allow valid field transitions for the current instruction format. This guarantees syntactically correct output.
- **Strategy monoculture prevention**: Any single mutation strategy is capped at 8 consecutive wins before forced rotation, preventing the feedback loops observed in early runs.
- **In-process training**: The retrain pipeline runs within the parent process (not as a subprocess) to avoid SIGTERM issues in the Hermes agent environment.
