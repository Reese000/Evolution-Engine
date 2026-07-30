# Curriculum Warm-Start: L14-L30 Reference Injection

**Status:** Design document — approach documented, not yet implemented.  
**Date:** 2026-07-30  
**Related P1 item:** P1-2

## Problem

The synthetic data generator (`milestone1/synthetic_data.py`) currently produces ~50K training sequences from 5 strategies:

| Strategy | Source | Levels covered |
|----------|--------|----------------|
| 1A: Single instructions | All 47 ISA types | All |
| 1B: Data flow chains | Random arithmetic chains | All |
| 1C: Subsequence extraction | Reference solutions L0-L11 | L0-L11 only |
| 1D: Template instantiation | Structural patterns from L8-L11 | L8-L11 patterns |
| 1E: Reference mutations | Perturbed L8-L11 solutions | L8-L11 only |

**Strategies 1C and 1E are limited to L0-L11.** L14-L30 reference solutions exist in `tasks_rv32i.py` (12-75 instructions each) but are not used for training data generation. This means the model never sees examples of higher-level program structures during training.

## Reference Solutions Available (L14-L30)

| Level | Task | Instructions |
|-------|------|-------------|
| L14 | Multi-program composition | 19 |
| L15 | Adaptive self-modification | 12 |
| L16 | Evolutionary self-reference | 10 |
| L17 | Meta-evolution (sub-program) | 19 |
| L18 | XOR bit manipulation | 22 |
| L19 | Conditional branching | 27 |
| L20 | Complex memory + all ops | 32 |
| L21 | XOR chain + shift pipeline | 35 |
| L22 | Multi-branch conditional | 46 |
| L23 | Memory array operations | 43 |
| L24 | Complex conditional + memory | 49 |
| L25 | Full program (all operations) | 55 |
| L26 | Shift chain | 56 |
| L27 | Unsigned comparison | 61 |
| L28 | Byte memory operations | 65 |
| L29 | Subroutine call | 67 |
| L30 | Sort and compute | 75 |

## Proposed Approach

### Phase 1: Extend Subsequence Extraction (Strategy 1C)

Extend `_gen_subsequences()` to cover L14-L30:

```python
# In synthetic_data.py, _gen_subsequences():
for level in range(31):  # was range(12)
    ref = get_ref_solution(level)
    # ... existing window extraction logic
```

**Risk:** L14-L30 programs are 10-75 instructions. Subsequence windows of length 2-5 from a 75-instruction program produce many more fragments. This could balloon dataset size.

**Mitigation:** Cap total subsequences per level at 2000 (vs unlimited for L0-L11).

### Phase 2: Extend Reference Mutations (Strategy 1E)

Extend `_gen_mutations()` to cover L14-L30:

```python
# In synthetic_data.py, _gen_mutations():
for level in range(14, 31):  # was range(8, 12)
    ref = get_ref_solution(level)
    # ... existing mutation logic (100 mutations per level)
```

**Risk:** Mutations on 75-instruction programs are more likely to produce degenerate output. The `_mutate_program()` helper handles this via None returns, but coverage may be lower.

**Mitigation:** Increase mutation attempts from 100 to 200 for L20+ to maintain yield.

### Phase 3: Full-Program Injection (New Strategy 1F)

Add a new strategy that injects the complete reference solutions for L14-L30, register-renamed 8x each:

```python
def _gen_full_reference(self) -> List[List[Dict]]:
    """Inject full L14-L30 reference solutions, register-renamed."""
    programs = []
    for level in range(14, 31):
        ref = get_ref_solution(level)
        ref_body = [d for d in ref if d["name"] != "EBREAK"]
        for _ in range(8):
            renamed = register_rename(ref_body, self.rng)
            programs.append(renamed + [self._ebreak])
    return programs
```

**Expected output:** 17 levels × 8 renames = 136 full programs. These serve as "golden examples" that teach the model complete higher-level program structures.

## Estimated Dataset Impact

| Strategy | Current (L0-L11) | After (L0-L30) | Delta |
|----------|-------------------|-----------------|-------|
| 1C: Subsequences | ~8K | ~20K | +12K |
| 1E: Mutations | ~5K | ~10K | +5K |
| 1F: Full reference | 0 | 136 | +136 |
| **Total** | **~50K** | **~62K** | **+12K** |

The 24% increase is manageable and should not significantly impact training time.

## Risks and Considerations

1. **Data leakage:** Reference solutions for L14-L30 are also used as evaluation targets. Training on them directly could inflate validation metrics. Mitigation: use register-renamed versions (different register assignments) to reduce direct memorization.

2. **Sequence length:** L20+ programs are 32-75 instructions. With 9 tokens per instruction, sequences are 288-675 tokens. The model's context window (1024 tokens for default config) can handle this, but training efficiency may decrease for very long sequences.

3. **Curriculum ordering:** The model currently trains on all levels simultaneously. Injecting L14-L30 data doesn't change the evaluation curriculum — the model still progresses L0→L30 through evolution. The warm-start just gives it a head start on recognizing higher-level patterns.

4. **Validation:** After implementation, run `python milestone1/synthetic_data.py` self-test to verify dataset size, then run a short training cycle to confirm loss convergence with the expanded dataset.

## Existing Warm-Start Infrastructure

The project already has a `WarmStartTrainer` in `milestone1/training/scaling/warm_start_trainer.py` that handles knowledge distillation after model scaling (teacher→student). This is a different warm-start mechanism — it's for architecture changes, not curriculum progression. The proposed approach is complementary: it enriches the training data rather than modifying the training procedure.
