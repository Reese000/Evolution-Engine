# P1-4: Model Scaling Study

**Date:** 2026-07-30  
**Status:** Analysis only — no code changes  
**Roadmap Reference:** P1-4 — "Model scaling study: test if larger models (12M, 24M params) help with L14+"

---

## 1. Current Architecture

### 1.1 RV32ITransformer Overview

The model is a **decoder-only transformer** with field-level tokenization for the RV32I ISA.

| Component | Implementation |
|-----------|---------------|
| Architecture | Decoder-only transformer (GPT-style) |
| Position encoding | Rotary Position Embeddings (RoPE) |
| Normalization | RMSNorm (pre-norm) |
| Feed-forward | SwiGLU (LLaMA/Mistral-style) |
| Attention | Multi-head causal self-attention with Flash Attention (SDPA) |
| Embeddings | Tied embedding/head weights |
| Tokenization | Field-level: 9 tokens per instruction |
| Vocabulary | 2,048 tokens (47 instructions + 32 registers + 8 funct3 + 128 funct7 + 48 immediate nibbles + specials) |

### 1.2 Current Model Configurations

From `milestone1/training/model_factory.py` → `MODEL_SIZE_CONFIGS`:

| Config | d_model | n_layers | n_heads | d_ff | Parameters | Status |
|--------|---------|----------|---------|------|------------|--------|
| `small` | 256 | 6 | 8 | 683 | **5.2M** | Available |
| `medium` | 256 | 8 | 8 | 1024 | **8.9M** | Available |
| `large` | 512 | 14 | 8 | 1365 | **45.1M** | Available |
| `xlarge` | 768 | 16 | 12 | 2048 | **114.8M** | Available |

**Production config:** The README states "~6M parameters" for the default config, which corresponds to `small` (5.2M). The `medium` config (8.9M) is used for most training runs based on the `model_size` parameter defaults.

### 1.3 Parameter Breakdown (Medium: 256d/8L/8.9M)

```
Token embedding:     2048 × 256 =     524,288 (5.9%)
Token type embedding:    6 × 256 =       1,536 (0.0%)
8 Transformer blocks:          =   8,392,704 (94.1%)
  ├─ qkv:         3 × 256² =     196,608/block
  ├─ out_proj:       256² =      65,536/block
  ├─ w1 (gate): 256 × 1024 =     262,144/block
  ├─ w2 (down): 1024 × 256 =     262,144/block
  ├─ w3 (up):   256 × 1024 =     262,144/block
  └─ RMSNorm:         2 × 256 =       512/block
Final lm_head_norm:          =         256
LM head (tied):              =           0
─────────────────────────────────────────────
TOTAL:                       =   8,918,784 (8.9M)
```

### 1.4 Key Architectural Constraints

1. **d_model must be divisible by n_heads** (enforced by `assert d_model % n_heads == 0`)
2. **d_ff is 2/3 of standard FFN** (SwiGLU uses 3 matrices instead of 2)
3. **Tied weights** (embedding = lm_head) reduce parameter count but tie capacity
4. **Max sequence length:** 2048 tokens (configurable)
5. **RoPE head_dim:** must be even (d_model / n_heads)

---

## 2. Scaling Analysis

### 2.1 Why Scale?

| Level | Pattern Complexity | Current Model Capability |
|-------|-------------------|-------------------------|
| L0–L6 | Simple arithmetic, register moves | ✅ Sufficient (5.2M learns these) |
| L7–L10 | Memory operations, loops | ⚠️ Marginal (needs 8.9M+) |
| L11–L13 | Self-modifying code, fibonacci | ❌ Insufficient (needs deeper model) |
| L14–L20 | Multi-program, conditionals, bitops | ❌ Insufficient (needs wider + deeper) |
| L21–L30 | Nested branching, sorting, subroutines | ❌ Insufficient (needs 24M+) |

The hypothesis is that **larger models can learn more complex instruction patterns** because:
- More layers → longer dependency chains (critical for L12+ SMC patterns)
- Wider hidden dim → richer intermediate representations (critical for L14+ multi-program)
- More attention heads → better register dependency tracking

### 2.2 Parameter Count Targets

For 12M and 24M variants, we need to find configurations that:
1. Maintain architectural compatibility (same tokenizer, same forward pass)
2. Fit in reasonable VRAM (≤4GB for training with AMP)
3. Use hardware-efficient dimensions (multiples of 64 for tensor cores)

#### ~12M Configurations

| Config | d_model | n_layers | n_heads | d_ff | Params | VRAM (est.) |
|--------|---------|----------|---------|------|--------|-------------|
| **12M-A** | 256 | 12 | 8 | 1024 | **13.1M** | ~1.3 GB |
| **12M-B** | 256 | 11 | 8 | 1024 | **12.1M** | ~1.2 GB |
| **12M-C** | 320 | 8 | 8 | 853 | **10.5M** | ~1.1 GB |
| **12M-D** | 384 | 8 | 12 | 1024 | **14.0M** | ~1.2 GB |

**Recommended: 12M-A (256d/12L)** — Keeps d_model=256 (same as current), adds 4 layers. Minimal code changes needed.

#### ~24M Configurations

| Config | d_model | n_layers | n_heads | d_ff | Params | VRAM (est.) |
|--------|---------|----------|---------|------|--------|-------------|
| **24M-A** | 256 | 24 | 8 | 1024 | **25.7M** | ~2.6 GB |
| **24M-B** | 256 | 23 | 8 | 1024 | **24.7M** | ~2.5 GB |
| **24M-C** | 384 | 12 | 12 | 1024 | **22.0M** | ~1.9 GB |
| **24M-D** | 320 | 16 | 8 | 853 | **20.3M** | ~1.8 GB |

**Recommended: 24M-A (256d/24L)** — Keeps d_model=256, doubles layers from 12→24. Maximum depth for dependency chain learning.

### 2.3 Scaling Trade-offs

| Dimension | 12M (256d/12L) | 24M (256d/24L) | Impact |
|-----------|----------------|----------------|--------|
| **Depth (n_layers)** | 12 | 24 | Deeper models learn longer dependency chains; critical for SMC (L12+) |
| **Width (d_model)** | 256 | 256 | Keeping width constant isolates depth effect |
| **FFN capacity** | 1024 | 1024 | Same per-layer capacity; more layers = more total capacity |
| **Attention heads** | 8 | 8 | 32 head_dim (256/8) — sufficient for register tracking |
| **Training speed** | ~1.3× slower than medium | ~2.7× slower than medium | More layers = more compute per forward pass |
| **Gradient checkpointing** | Not needed | Recommended | 24L may need activation memory optimization |

---

## 3. What Needs to Change for 12M/24M

### 3.1 Code Changes (Minimal)

The architecture is already parameterized — no structural changes needed.

#### 3.1.1 Add Configs to `model_factory.py`

```python
MODEL_SIZE_CONFIGS: dict[str, dict] = {
    "small":   dict(d_model=256, n_layers=6,  n_heads=8, d_ff=683),
    "medium":  dict(d_model=256, n_layers=8,  n_heads=8, d_ff=1024),
    "large":   dict(d_model=512, n_layers=14, n_heads=8, d_ff=1365),
    "xlarge":  dict(d_model=768, n_layers=16, n_heads=12, d_ff=2048),
    # NEW: Scaling study variants
    "12m":     dict(d_model=256, n_layers=12, n_heads=8, d_ff=1024),
    "24m":     dict(d_model=256, n_layers=24, n_heads=8, d_ff=1024),
}
```

#### 3.1.2 Update CLI Args

```python
# In train_model.py or training/loop.py
parser.add_argument("--model-size", choices=["small", "medium", "large", "xlarge", "12m", "24m"])
```

#### 3.1.3 Enable Gradient Checkpointing for 24M

```python
# In training/loop.py, after model creation
if model_size in ("24m", "large", "xlarge"):
    model.use_checkpointing = True
    print(f"[Train] Gradient checkpointing enabled for {model_size}")
```

### 3.2 Training Hyperparameter Adjustments

Larger models may need different training hyperparameters:

| Parameter | Medium (8.9M) | 12M | 24M | Rationale |
|-----------|---------------|-----|-----|-----------|
| Learning rate | 3e-5 | 2e-5 | 1e-5 | Larger models need smaller LR for stability |
| Batch size | 8 | 8 | 4-8 | May need smaller batch for VRAM |
| Weight decay | 0.01 | 0.01 | 0.02 | More regularization for larger models |
| Grad clip | None | 1.0 | 0.5 | Prevent gradient explosion in deep models |
| Warmup epochs | 1 | 1 | 2 | Longer warmup for larger models |
| Max epochs | 12 | 12 | 15 | May need more epochs to converge |

### 3.3 Checkpoint Compatibility

The checkpoint system already handles architecture mismatches:
- `load_or_create_model()` checks `config_name` in checkpoint
- `allow_mismatch=True` with `--migrate` flag loads weights even when dimensions differ
- Weight initialization uses GPT-2 style scaled initialization (already handles deep models)

**Migration path:** When scaling from medium→12m or 12m→24m:
1. Use `--migrate` flag to load old checkpoint
2. New layers get fresh initialization (existing layers keep weights)
3. This provides a warm start rather than training from scratch

### 3.4 Scaling Ladder Integration

The existing `training/scaling/` module has:
- `SaturationDetector`: Detects when model has plateaued
- `ScalingPlanner`: Determines when to scale up
- `ModelMigrator`: Handles weight migration between architectures
- `WarmStartTrainer`: Trains new architecture from old checkpoint

**Current scaling ladder** (from `scaling_planner.py`):
```python
SCALING_LADDER = [
    (0, 8, 256, 8, "9M", 4.0),     # ← These embed dims are wrong (should be d_model)
    (1, 14, 512, 8, "35M", 8.5),
    (2, 20, 768, 12, "80M", 11.5),
    (3, 26, 1024, 16, "150M", 12.5),
]
```

**Issue:** The `embed_dim` values (8, 14, 20, 26) appear to be incorrect — they're too small for transformer models. These should be `d_model` values (256, 384, 512, 768). The scaling ladder needs to be updated.

**Proposed corrected ladder:**
```python
SCALING_LADDER = [
    (0, 256, 1024, 8, "8.9M", 1.0),   # medium (current)
    (1, 256, 1024, 8, "13.1M", 1.5),   # 12M-A (12 layers)
    (2, 256, 1024, 8, "25.7M", 3.0),   # 24M-A (24 layers)
    (3, 512, 1365, 8, "45.1M", 6.0),   # large (existing)
    (4, 768, 2048, 12, "114.8M", 12.0), # xlarge (existing)
]
```

---

## 4. Expected Impact on Curriculum Levels

### 4.1 Level Capability Mapping

| Model | Params | Max Level (Expected) | Key Patterns Learned |
|-------|--------|---------------------|---------------------|
| small (5.2M) | 5.2M | L3–L4 | Basic register moves, simple ADD |
| medium (8.9M) | 8.9M | L6–L7 | Arithmetic chains, first memory ops |
| **12M (13.1M)** | 13.1M | L9–L10 | Loops, memory ranges, comparison |
| **24M (25.7M)** | 25.7M | L14–L16 | Self-modification, conditionals |
| large (45.1M) | 45.1M | L20–L25 | Multi-program, nested branching |
| xlarge (114.8M) | 114.8M | L30+ | Full curriculum |

### 4.2 Why Deeper Models Help

The key insight is that **instruction depth correlates with model depth**:

- **L0–L6 (6 instructions):** 6 layers sufficient (1 layer per instruction)
- **L7–L10 (10–20 instructions):** Need 10+ layers for attention span
- **L11–L13 (25–30 instructions, SMC):** Need 12+ layers for nested patterns
- **L14–L30 (30–50+ instructions):** Need 20+ layers for long-range dependencies

The model must attend to earlier instructions when generating later ones. With causal attention, the effective receptive field grows with depth. A 24-layer model can attend to instructions 24 positions back, which covers most L14+ programs.

### 4.3 Why Wider Models Help

Wider models (larger d_model) learn richer intermediate representations:

- **Register tracking:** More dimensions = better ability to track which register holds what value
- **Instruction encoding:** More dimensions = richer representation of instruction semantics
- **Pattern matching:** More dimensions = ability to distinguish similar but different patterns

However, **depth is more important than width** for this task because:
1. Instruction sequences are long (30–50 tokens for L14+)
2. Dependencies span many instructions (L12 SMC needs instruction 1 to influence instruction 30)
3. Depth increases the effective attention span

This is why the recommended configs keep d_model=256 and increase n_layers.

---

## 5. Implementation Plan

### Step 1: Add 12M and 24M Configs (S)

**Files to modify:**
- `milestone1/training/model_factory.py` — Add `"12m"` and `"24m"` to `MODEL_SIZE_CONFIGS`
- `milestone1/training/loop.py` — Add `--model-size` choices

**Effort:** 30 minutes

### Step 2: Enable Gradient Checkpointing for 24M (S)

**Files to modify:**
- `milestone1/training/loop.py` — Auto-enable `use_checkpointing` for 24m

**Effort:** 15 minutes

### Step 3: Fix Scaling Ladder (S)

**Files to modify:**
- `milestone1/training/scaling/scaling_planner.py` — Correct `SCALING_LADDER` embed dims

**Effort:** 30 minutes

### Step 4: Create Ablation Test Script (M)

Create a script that runs short training cycles with each model size and compares:
- Training loss convergence
- Validation loss
- Level advancement rate
- Training time per epoch

**Effort:** 2–3 hours

### Step 5: Run Ablation Study (L)

Run 5-cycle training with each model size:
1. medium (8.9M) — baseline
2. 12M (13.1M) — test depth scaling
3. 24M (25.7M) — test deeper scaling

Measure:
- Final validation loss
- Max level achieved
- Training time
- VRAM usage

**Effort:** 4–8 hours (depending on GPU availability)

### Step 6: Update Scaling Coordinator (M)

Wire the new configs into the automatic scaling system:
- Update `SaturationDetector` thresholds for 12M/24M
- Update `ScalingPlanner` to recommend 12M→24M progression
- Update `ModelMigrator` to handle 256d weight migration (12L→24L)

**Effort:** 2–3 hours

---

## 6. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| 24M too slow for iterative training | High | Use gradient checkpointing; consider 12M as primary |
| Overfitting on small training set | Medium | Increase weight decay; use more augmentation |
| Checkpoint migration loses knowledge | Medium | Test migration quality; keep old checkpoint as fallback |
| VRAM exceeded on target GPU | Low | All configs fit in 4GB; gradient checkpointing for safety |
| Deeper model doesn't help L4 plateau | Medium | L4 is a training data issue, not capacity — fix P1-3 first |

---

## 7. VRAM Budget Summary

| Config | Params | Model Memory | Activation Memory | Total VRAM | Fits 4GB? |
|--------|--------|-------------|-------------------|------------|-----------|
| medium (256d/8L) | 8.9M | 0.05 GB | 0.81 GB | **0.86 GB** | ✅ Yes |
| 12M (256d/12L) | 13.1M | 0.08 GB | 1.21 GB | **1.29 GB** | ✅ Yes |
| 24M (256d/24L) | 25.7M | 0.15 GB | 2.42 GB | **2.57 GB** | ✅ Yes |
| large (512d/14L) | 45.1M | 0.27 GB | 4.31 GB | **4.58 GB** | ⚠️ Tight |

All proposed configs fit comfortably in 4GB VRAM with AMP (mixed precision).

---

## 8. Migration Strategy

### 8.1 Warm Start from Medium → 12M

```bash
# Train medium model first (baseline)
python train_model.py --model-size medium --max-cycles 10

# Scale to 12M with migration
python train_model.py --model-size 12m --migrate --resume-path checkpoints/model_final.pt
```

The `--migrate` flag:
1. Loads medium checkpoint (256d/8L weights)
2. Creates 12M model (256d/12L)
3. Copies matching layers (8 layers → first 8 of 12)
4. Randomly initializes new layers (9–12)
5. Continues training with warm-started weights

### 8.2 Warm Start from 12M → 24M

```bash
# Scale from 12M to 24M
python train_model.py --model-size 24m --migrate --resume-path checkpoints/model_final.pt
```

Same migration logic: 12 layers → first 12 of 24, new layers (13–24) get fresh init.

### 8.3 Expected Migration Quality

- **Layers 1–8 (from medium):** Retain learned patterns (L0–L6 arithmetic)
- **Layers 9–12 (new in 12M):** Freshly initialized, need training to learn
- **Layers 13–24 (new in 24M):** Freshly initialized, need more training

Migration should preserve L0–L6 knowledge while allowing new layers to learn L7+ patterns.

---

## 9. Summary

### Current State
- **Production model:** 256d/8L (8.9M params) — sufficient for L0–L6
- **Gap:** L7+ requires more capacity; L14+ needs significantly more depth

### Proposed Scaling
- **12M variant:** 256d/12L (13.1M params) — targets L7–L10
- **24M variant:** 256d/24L (25.7M params) — targets L14–L16

### Key Design Decisions
1. **Keep d_model=256** — isolates depth effect, minimal code changes
2. **Increase n_layers** — depth is more important than width for instruction sequences
3. **Use gradient checkpointing** — enables 24L training on consumer GPUs
4. **Warm-start migration** — preserves learned knowledge across scales

### Implementation Effort
- Config changes: ~1 hour
- Ablation study: ~4–8 hours
- Scaling coordinator integration: ~3 hours
- **Total: ~1–2 days**

### Expected Impact
- 12M: Break through L7–L10 plateau
- 24M: Enable L14–L16 patterns
- Combined with P1-3 (L4 plateau fix): Full L0–L16 capability
