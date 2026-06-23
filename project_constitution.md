# Project Constitution — Self-Evolving Machine-Code AI

## Project Vision
Build a fully autonomous system that continuously improves its ability to generate correct RISC-V RV32I machine-code programs through a closed feedback loop: evolution → retrain → evolve. The system should discover novel programs for increasingly difficult curriculum levels, eventually reaching and mastering self-modifying code (L12/L13) without human intervention.

The user (Reese) has stated: *"Build a self-evolving machine-code AI that continuously improves via a closed feedback loop"* — the system runs autonomously, the user sets strategy, the agent executes.

## Agent Interpretation
The system architecture is an evolutionary algorithm wrapped around a transformer-based program generator:

- **Evolution engine** (`milestone2/evolution_engine.py`): Maintains a champion program, generates challengers via mutation/crossover, evaluates them on the RV32I VM farm, and promotes fitter individuals.
- **Multi-cycle runner** (`milestone2/run_multi_cycle.py`): Orchestrates N cycles of evolution → retrain → evolve. The retrain pipeline collects champion traces and fine-tunes the seed model.
- **MetaController** (`milestone2/meta_controller.py`): Dynamic parameterization — monitors telemetry (fitness, val_loss, diversity, level) and tunes hyperparameters in real-time.
- **GPU acceleration** (`milestone0/gpu-vm/`): CUDA RV32I emulator for batch evaluation (30-300x speedup over CPU safety kernel).

Key insight from failed runs: The system can reach L13 via reference-solution reseeding, but the champion cannot **independently solve** L11+ (Fibonacci, self-modifying code). The bottleneck is **generation quality**, not curriculum mechanics or hyperparameters. The model needs to learn correct program patterns for high-level tasks before evolution can optimize them.

## Directives

### Scope Boundaries
1. **RV32I only** — No other instruction set architectures. The VM farm, model, and curriculum are all RV32I-specific.
2. **Small model (45M params)** — No larger models unless GPU memory permits after optimization. torch.compile + TF32 for training speed.
3. **No external APIs** — Everything runs locally. No API keys, no cloud services, no secrets.
4. **CUDA GPU required** — RTX 4070 Ti (12.9GB VRAM). The retrain pipeline and GPU VM farm both require CUDA.

### Technology Constraints
1. **Python 3.11+** — The entire pipeline is Python. No C/C++ extensions beyond the Rust safety kernel.
2. **PyTorch 2.6+** — Model training uses torch with `weights_only=False` for checkpoint loading (required for config-embedded format).
3. **CuPy 14+** — GPU VM acceleration uses CuPy for CUDA kernel launch.
4. **Rust safety kernel** — The reference CPU evaluator. Must match GPU results bit-for-bit.

### Quality Standards
1. **Three-gate workflow** — Every change goes through DEBATE → EXECUTE → CRITIQUE (mcp_critic).
2. **31/31 tests must pass** — test_meta_controller.py (23) + test_calibration_e2e.py (8). GPU tests (test_gpu_vm.py, test_orphan_cleanup.py) optional but recommended.
3. **Checkpoint gating** — Never accept a model with val_loss > best_val_loss × 1.05. MetaController enforces this.
4. **Diversity injection** — Population diversity below 0.05 triggers random program injection. Self-calibrate adjusts mutation_rate in real-time.

### Things to Never Do
1. Never run without checkpoint gating (C18 regression must never repeat)
2. Never delete the best known checkpoint (checkpoint_best.pt is sacred)
3. Never skip the three-gate workflow (debate → execute → critique)
4. Never prompt the user mid-task for decisions — resolve ambiguity with tools
5. Never fabricate results — report real terminal output or blockers honestly

## Execution Context

### Tech Stack
| Component | Technology | Location |
|-----------|------------|----------|
| Evolution engine | Python 3.11, no deps | `/root/milestone2/evolution_engine.py` |
| Multi-cycle runner | Python 3.11 | `/root/milestone2/run_multi_cycle.py` |
| MetaController | Python 3.11, stdlib | `/root/milestone2/meta_controller.py` |
| Seed model | PyTorch 2.6+, RV32ITransformer | `/root/milestone1/rv32i_model.py` |
| Training pipeline | PyTorch, AdamW, CosineLR | `/root/milestone1/train_model.py` |
| GPU VM farm | CUDA, CuPy 14+ | `/root/milestone0/gpu-vm/` |
| CPU safety kernel | Rust | `/root/milestone0/safety-kernel/` |
| Curriculum | 14 levels (L0-L13) | `/root/milestone2/tasks_rv32i.py` |

### Environment
- **OS**: WSL2 (Windows Subsystem for Linux)
- **GPU**: NVIDIA RTX 4070 Ti (12.9GB VRAM)
- **CUDA**: 12.x
- **Disk**: ~20GB free (checkpoints at 179MB each)
- **Working directory**: `/root`

### Key Dependencies
- `torch` (PyTorch 2.6+) — model training, inference
- `cupy` (CuPy 14+) — GPU kernel launch
- `rustc` / `cargo` — safety kernel compilation
- No external network APIs required

### File Layout
```
/root/
├── milestone0/
│   ├── gpu-vm/              # GPU RV32I emulator
│   │   ├── gpu_vm_kernel.cu    # CUDA kernel (warp-per-candidate)
│   │   └── gpu_vm_farm.py      # Python CuPy wrapper
│   └── safety-kernel/          # Rust CPU evaluator
├── milestone1/
│   ├── train_model.py          # Training pipeline
│   ├── rv32i_model.py          # RV32ITransformer definition
│   └── checkpoints/            # All model checkpoints
├── milestone2/
│   ├── evolution_engine.py     # Core evolutionary loop
│   ├── run_multi_cycle.py      # Multi-cycle orchestrator
│   ├── meta_controller.py      # Dynamic parameterization
│   ├── tasks_rv32i.py          # 14-level curriculum definition
│   └── test_*.py               # Test suite
└── continuous_iteration/       # Run logs and artifacts
```

## Execution Plan (High-Level)

### Phase 0 — Foundation ✅ COMPLETE
- MetaController: checkpoint gating, stagnation detection, dynamic tuning
- GPU VM farm: SIMD RV32I evaluation in CUDA
- Self-calibrate: diversity-based mutation rate adjustment
- Test suite: 31 tests, integration verified

### Phase 1 — Knowledge Injection 🔜 CURRENT
- Inject reference solutions into retrain training data
- Teach the model correct patterns for L11+ (Fibonacci, self-modifying code)
- Expected impact: model generates valid programs at all levels, not just L0-L10
- Success metric: model-generated programs solve L11 without reference reseeding

### Phase 2 — Prefix-Guided Generation
- Replace random mutation with model-completed programs from champion prefixes
- The model generates the CONTINUATION from a champion prefix
- No more blind mutation operators — the model becomes the mutation
- Success metric: champion discovers novel L12/L13 solutions not in training data

### Phase 3 — Self-Modifying Code Mastery
- Champion can produce self-modifying programs (L12/L13) via prefix-guided generation
- Model learns temporal instruction patterns: load→modify→execute
- Evolution optimizes self-modifying programs for correctness and efficiency
- Success metric: champion achieves fitness > 0.5 on L12 and L13

### Phase 4 — Full Autonomy
- Continuous 48-cycle runs without human oversight
- Automatic recovery from any failure mode (OOM, regression, plateau)
- MetaController evolves its own tuning rules via meta-evolution
- System self-diagnoses curriculum bottlenecks and adjusts strategy
- Success metric: 48 cycles complete with net fitness improvement, no intervention
