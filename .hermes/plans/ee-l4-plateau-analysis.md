# P1-3: L4 Plateau Analysis

**Date:** 2026-07-30  
**Status:** Analysis only — no code changes  
**Roadmap Reference:** P1-3 — "Fix L4 plateau: adaptive fitness gradient"

---

## 1. The Problem

The champion program stalls at **L4** for 3+ consecutive cycles. The model cannot learn the pattern required to advance from L3→L4, despite L4 being a seemingly simple arithmetic instruction (`ADD x7, x4, x6`).

### Why L4 Is Uniquely Hard

| Level | Task | Key Operation | Dependency |
|-------|------|---------------|------------|
| **L0** | `x3 = x1` | `ADDI x3, x1, 0` | Direct input only |
| **L1** | `x4 = 2*x1` | `ADD x4, x1, x1` | Direct input only |
| **L2** | `x5 = x2` | `ADDI x5, x2, 0` | Direct input only |
| **L3** | `x6 = 2*x2` | `ADD x6, x2, x2` | Direct input only |
| **L4** | `x7 = x4 + x6` | `ADD x7, x4, x6` | **Requires x4 (from L1) and x6 (from L3)** |

**Critical insight:** L0–L3 each operate directly on input registers (`x1`, `x2`). L4 is the **first level that requires intermediate register chaining** — it must read `x4` (written by L1) and `x6` (written by L3) as source operands. This is a qualitatively different pattern: the model must learn that previously written registers can be used as inputs to subsequent instructions.

### Reference Solution for L4 (cumulative)

```python
[
    {"name": "ADDI", "rd": 3,  "rs1": 1, "imm": 0},  # L0: x3 = x1
    {"name": "ADD",  "rd": 4,  "rs1": 1, "rs2": 1},  # L1: x4 = x1 + x1
    {"name": "ADDI", "rd": 5,  "rs1": 2, "imm": 0},  # L2: x5 = x2
    {"name": "ADD",  "rd": 6,  "rs1": 2, "rs2": 2},  # L3: x6 = x2 + x2
    {"name": "ADD",  "rd": 7,  "rs1": 4, "rs2": 6},  # L4: x7 = x4 + x6  ← THE BOTTLENECK
]
```

The L4 instruction uses `rs1=4` and `rs2=6` — registers that were **written by earlier instructions in the same program**. The model must learn this "write-then-read" register dependency pattern.

---

## 2. Root Cause Analysis

### 2.1 Training Data Gap

The training pipeline loads champion traces from `recurrent_data.jsonl`. When the champion is stuck at L4, the traces in the buffer are overwhelmingly L0–L3 programs that never use intermediate registers as source operands. The model sees:

- Many examples of `ADDI x3, x1, 0` (L0 pattern)
- Many examples of `ADD x4, x1, x1` (L1 pattern)
- Many examples of `ADD x5, x2, 0` (L2 pattern)
- Many examples of `ADD x6, x2, x2` (L3 pattern)
- **Very few or zero examples** of `ADD x7, x4, x6` (L4 pattern)

The model never learns the "read from previously written register" pattern because it never sees it in the training data.

### 2.2 Reference Solution Injection Coverage

`inject_reference_solutions()` in `milestone1/data/augmentation.py` injects reference solutions for L0–L13 with copy counts:

```python
ref_copies = {
    0: 5, 1: 5, 2: 5, 3: 5, 4: 5,   # L0-L4: 5 copies each
    5: 10, 6: 10, 7: 10,              # L5-L7: 10 copies
    8: 20, 9: 30, 10: 30, 11: 30,     # L8-L11: 20-30 copies
    12: 40, 13: 50,                    # L12-L13: 40-50 copies
}
```

**Problem:** L4 only gets 5 copies — the same as L0–L3. But L4 requires a more complex pattern (register chaining). With only 5 reference copies and no champion traces containing the L4 pattern, the model has insufficient signal to learn this pattern.

### 2.3 Verify Score Gradient

The `_verify_champion()` method stores `_champion_partial_verify` as a float score (0.0–1.0):

```python
# Partial verify scoring: store the float score even when < 1.0
# to create a fitness gradient for the L4 plateau escape.
self._champion_partial_verify = score
```

This creates a gradient: if a program gets 4/5 registers correct (L0–L3 correct, L4 wrong), it scores 0.8. But the fitness function uses `level_bonus = champion_level * 25.0`, which rewards the current level (L3→L4) regardless of whether L4 is actually solved. The champion gets bonus points for being at L4 even without solving it, masking the lack of progress.

### 2.4 Circuit Breaker and Stall Detection

`_try_curriculum_advance()` checks:
1. `generations_stalled >= curriculum_stall_threshold` (default 3)
2. Champion **verified** for current level (`verify_score >= 1.0`)
3. `_champion_raw_task_score >= 120.0`

If the champion hasn't solved L4 (verify < 1.0), curriculum advancement is **blocked** (R50). This is correct behavior but means the system loops indefinitely without solving L4.

### 2.5 Fitness Function Masking

The 14-signal composite fitness includes `level_bonus = champion_level * 25.0`. At L4, this adds 100 points (4 × 25). This dominates the base task score, making it appear that the champion is performing well even when it fails to produce the correct L4 output. The champion gets "paid" for being at level 4 without actually solving it.

---

## 3. What Patterns Are Missing

### 3.1 The "Write-Then-Read" Pattern

L4 is the first level requiring the model to:
1. Write an intermediate result to a register (`x4 = x1 + x1`)
2. Write another intermediate result (`x6 = x2 + x2`)
3. **Read both intermediate results** as source operands (`ADD x7, x4, x6`)

This is a fundamental programming pattern (temporaries/intermediate values) that the model has never seen in training data.

### 3.2 Register-to-Register Data Flow

L0–L3 use only `x1` and `x2` (inputs) as source operands. L4 introduces the pattern where a register written by one instruction becomes a source operand for a later instruction. This creates a **data flow dependency chain** that the model must learn to reason about.

### 3.3 Instruction Ordering Awareness

The model must learn that instruction order matters: the `ADD x4, x1, x1` must come **before** `ADD x7, x4, x6` because `x4` is a source operand in the latter. This requires understanding of instruction sequencing and data dependencies.

---

## 4. Proposed Solutions

### 4.1 Curriculum-Aware Trace Injection (Primary Fix)

**Approach:** Inject synthetic L4-pattern traces that specifically demonstrate the "write-then-read" register chaining pattern.

**Implementation sketch:**

```python
# In inject_reference_solutions() or a new inject_curriculum_aware_traces()

def inject_curriculum_aware_traces(tokenizer, sequences, sample_weights, rv32i_decode):
    """Inject traces specifically targeting the L4 bottleneck pattern."""
    
    # Generate variants of the L4 pattern with different register assignments
    # This teaches the model that ANY register can be used as a source,
    # not just x1 and x2.
    
    patterns = [
        # Core L4 pattern: two intermediates summed
        [ADDI(x3, x1, 0), ADD(x4, x1, x1), ADDI(x5, x2, 0), 
         ADD(x6, x2, x2), ADD(x7, x4, x6)],
        
        # Variant: different intermediate registers
        [ADDI(x3, x1, 0), ADD(x15, x1, x1), ADDI(x5, x2, 0),
         ADD(x16, x2, x2), ADD(x7, x15, x16)],
        
        # Chain: three intermediates
        [ADD(x3, x1, x2), ADD(x4, x1, x1), ADD(x5, x2, x2),
         ADD(x6, x3, x4), ADD(x7, x5, x6)],
        
        # Cross-dependency: x4 uses x2, x6 uses x1
        [ADDI(x3, x1, 0), ADD(x4, x2, x2), ADDI(x5, x2, 0),
         ADD(x6, x1, x1), ADD(x7, x4, x6)],
    ]
    
    for pattern in patterns:
        for _ in range(20):  # 20 copies of each variant
            tokens = tokenizer.encode_trace(pattern)
            sequences.append(torch.tensor(tokens, dtype=torch.long))
            sample_weights.append(1.0)
```

**Key design choices:**
- Generate 20+ copies of each pattern variant (vs. 5 for regular L4 refs)
- Use different register assignments to teach register invariance
- Include both the core L4 pattern and cross-dependency variants
- Weight at 1.0 (same as reference solutions) since these are correct programs

### 4.2 Enhanced Partial Verify Fitness Gradient

**Approach:** Make the fitness function reward partial progress toward L4 more explicitly.

**Current:** `_champion_partial_verify` is stored but only used for logging. The fitness function doesn't use it.

**Proposed:** Add a partial verify bonus to `_compute_fitness()`:

```python
# In _compute_fitness():
partial_verify = self._champion_partial_verify
if partial_verify > 0 and partial_verify < 1.0:
    # Reward partial progress: 0.8 → +40 points, 0.6 → +30 points
    partial_bonus = partial_verify * 50.0
    fitness += partial_bonus
```

This creates a gradient: a program that gets 4/5 registers right (L0–L3) gets +40 points, incentivizing the evolution to explore near-miss solutions.

### 4.3 Targeted Mutation Strategy for L4

**Approach:** Add a mutation strategy that specifically targets register chaining.

**Idea:** When the champion is at L4 and stalled, generate mutations that:
1. Take the champion's L0–L3 prefix
2. Append `ADD x7, xN, xM` where `xN` and `xM` are registers written earlier
3. This directly tests the "write-then-read" pattern

### 4.4 Level-Specific Training Data Cap

**Approach:** Ensure L4 reference solutions are not drowned out by L0–L3 data.

**Current:** `apply_level_cap()` caps per-level data. But L4 only has 5 reference copies plus whatever champion traces exist.

**Proposed:** Increase L4 reference copies to 30 (matching L8–L11):

```python
ref_copies = {
    0: 5, 1: 5, 2: 5, 3: 5,
    4: 30,  # Increased from 5 — L4 is a bottleneck level
    5: 10, 6: 10, 7: 10,
    8: 20, 9: 30, 10: 30, 11: 30,
    12: 40, 13: 50,
}
```

---

## 5. Implementation Steps

### Step 1: Add Curriculum-Aware Trace Injection
1. Create `inject_l4_bottleneck_traces()` in `milestone1/data/augmentation.py`
2. Generate 50+ synthetic traces demonstrating L4 register chaining patterns
3. Call this function in `training/loop.py` after reference solution injection
4. **Effort:** M (2-3 hours)

### Step 2: Increase L4 Reference Copies
1. Modify `ref_copies` dict in `inject_reference_solutions()` to set L4→30
2. Verify training data balance after the change
3. **Effort:** S (30 minutes)

### Step 3: Add Partial Verify Fitness Gradient
1. Modify `_compute_fitness()` in `evolution_engine.py` to use `_champion_partial_verify`
2. Add partial_bonus computation
3. Update fitness diagnostic summary to report partial verify contribution
4. **Effort:** S (1 hour)

### Step 4: Add L4-Aware Mutation Strategy
1. Modify `_choose_mutation_strategy()` in `evolution_engine.py`
2. When champion_level == 4 and stalled, bias toward TASK_AWARE mutations
3. Generate mutations that append register-chaining instructions
4. **Effort:** M (2-3 hours)

### Step 5: Validate End-to-End
1. Run single-cycle test with L4 curriculum
2. Verify model val_loss decreases with enhanced training data
3. Verify champion can advance past L4
4. **Effort:** M (1-2 hours)

---

## 6. Expected Impact

| Metric | Before | After (Expected) |
|--------|--------|-------------------|
| L4 solve rate | ~0% (stalled) | 50-80% within 3 cycles |
| Training data (L4 patterns) | 5 reference copies | 50+ synthetic + 30 reference |
| Partial verify gradient | None (binary pass/fail) | Continuous 0.0–1.0 |
| Fitness masking | level_bonus dominates | Partial verify bonus visible |

---

## 7. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Synthetic L4 traces may not generalize | Generate diverse register assignments and operand orderings |
| Partial verify bonus may cause premature promotion | Cap bonus at 50 points (below level_bonus threshold) |
| More L4 training data may cause L0–L3 regression | Use level-weighted loss to balance contributions |
| L4 fix may not unblock L5+ (different patterns) | L5 introduces SLT comparison — separate bottleneck analysis needed |

---

## 8. Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `milestone1/data/augmentation.py` | Add `inject_l4_bottleneck_traces()`, increase L4 ref_copies | ~50 lines |
| `milestone1/training/loop.py` | Call new injection function | ~5 lines |
| `evolution_engine.py` | Add partial verify bonus to `_compute_fitness()` | ~15 lines |

**Total estimated change:** ~70 lines of new/modified code.

---

## 9. Summary

The L4 plateau is caused by a **training data gap**: the model never sees the "write-then-read" register chaining pattern in its training data. L0–L3 programs only use input registers (`x1`, `x2`) as sources, so the model never learns that previously written registers can be used as operands. The fix is threefold:

1. **Inject curriculum-aware traces** that explicitly demonstrate L4 patterns (50+ synthetic traces)
2. **Increase L4 reference copies** from 5 to 30
3. **Add partial verify fitness gradient** to reward near-miss solutions

These changes require ~70 lines of code and should unblock L4 within 1–2 training cycles.
