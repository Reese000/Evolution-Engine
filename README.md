# Evolution Engine

A self-evolving machine-code AI that continuously improves its ability to generate correct RISC-V RV32I programs through a closed feedback loop: evolve, retrain, evolve.

The system combines an evolutionary algorithm with a transformer-based code generator. A champion program is maintained and iteratively improved through mutation and selection. A transformer model learns from champion traces to generate increasingly complex programs across a 12-level curriculum. A meta-controller orchestrates the entire process, tuning hyperparameters in real-time and detecting stagnation or regression.

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
├── champion_anchor.py           # Champion preservation and restart logic
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
│   ├── test_audit_fixes.py      # Audit test suite
│   ├── test_deployment_scripts.py  # Deployment verification
│   └── test_meta_evolution_engine.py  # Meta-controller tests
└── lessons_learned.md           # Post-mortem analysis of failure modes
```

## How It Works

1. **Evolution**: The engine generates challenger programs by mutating the current champion. Each challenger is evaluated on a 12-level curriculum (L0: register assignment through L11: complex loops). The best challenger replaces the champion if it achieves higher fitness.

2. **Retraining**: Champion instruction traces are collected and used to fine-tune the RV32I transformer. The model learns to predict the next instruction field given a program prefix. Training uses position-aware weighted cross-entropy loss (1.5x weight on instruction-type positions).

3. **Meta-Control**: Between cycles, the meta-controller analyzes telemetry to detect stagnation, regression, or diversity collapse. It adjusts retrain learning rate, evolution strategy weights, population size, and mutation rates. Checkpoint gating prevents accepting regressed models.

4. **Prefix-Guided Generation**: At higher curriculum levels, the MODEL_GUIDED strategy uses the transformer to generate program continuations from champion prefixes, replacing blind mutation with learned generation.

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
# Evolution
python meta_controller.py                    # Single evolution cycle
python run_multi_cycle.py                    # Multi-cycle evolution + retrain
python v11_dashboard.py                      # Launch live dashboard

# Training
python milestone1/train_model.py             # Train model on accumulated traces
python milestone1/build_recurrent_dataset.py # Build dataset from champion traces

# Testing
python -m pytest tests/                      # Run test suite

# Benchmarks
python run_benchmark.py                      # Run performance benchmarks
```

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
