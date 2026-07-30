# P0 Issue: Harvard vs Von Neumann Architecture Mismatch in VM

**Priority:** P0 — Blocking 14 of 17 curriculum levels (L12, L13, L17-L30)
**Date:** 2026-07-30
**Status:** Open — Requires architectural decision
**Related:** [Curriculum Validation Report](./ee-curriculum-validation-2026-07-30.md)

---

## Summary

The Evolution Engine's VM (`gpu_vm_farm._execute_vm()`) uses a **Harvard architecture** — separate address spaces for instructions (`program: list[int]`) and data (`memory: dict[int, int]`). The curriculum levels L12, L13, and L17-L30 (14 levels total) use **JALR-based meta-evolution**, which assumes a **von Neumann architecture** — a single unified address space where programs can store instruction words via `SW` and then execute them by jumping to that address via `JALR`. This fundamental mismatch causes all 14 levels to fail.

---

## The Problem

### What the VM does (Harvard)

```python
# Instruction fetch — fetches from program list, NOT data memory
word_offset = pc >> 2
if word_offset >= _prog_len:
    error_msg = f"PC out of range: {pc:#x}"
    break
word = _prog[word_offset]  # ← always from program list
```

```python
# JALR — sets PC to computed address
elif opcode == OP_JALR:
    imm = _se((word >> 20) & _MASK12, 12)
    if rd != 0:
        _regs[rd] = pc
    pc = ((v_rs1 + imm) & ~1) & _m32
    # Then next iteration: _prog[pc >> 2] is fetched
```

### What the curriculum expects (Von Neumann)

```
1. ADDI x20, x1, x2       # Build an instruction word
2. SW x20, x0, 0x40        # Store it to memory address 0x40
3. JALR x0, x23, 0         # Jump to PC=0x40
   Expected: Execute the instruction stored at memory[0x40]
   Actual:   VM fetches program[16] (a completely different instruction)
```

### Why it fails

When `JALR` sets `pc = 0x40`, the fetch loop computes `program[0x40 >> 2] = program[16]`. The instruction stored at `memory[0x40]` via `SW` is **never fetched**. The VM only reads instructions from the `program` list parameter — data memory is completely invisible to instruction fetch.

This means:
- `JALR` to data-memory addresses always executes the **wrong instruction**
- Programs either hit infinite loops (cycle budget exhausted) or execute unintended code paths
- Meta-evolution — building instructions at runtime and executing them — is impossible

---

## Affected Levels

| Level | Category | Failure Mode |
|-------|----------|--------------|
| L12 | Pre-existing | PC out of range at 0x40 |
| L13 | Pre-existing | 10000-cycle budget (infinite loop) |
| L17 | New (unlocked) | 10000-cycle budget; x20=0 |
| L18 | New (unlocked) | 10000-cycle budget; x20=0 |
| L19 | New (unlocked) | 10000-cycle budget; x20=0 |
| L20 | New (unlocked) | 10000-cycle budget; x20=0 |
| L21 | New (unlocked) | 10000-cycle budget; x20=0 |
| L22 | New (unlocked) | 10000-cycle budget; x20=0 |
| L23 | New (unlocked) | 10000-cycle budget; x20=0 |
| L24 | New (unlocked) | 10000-cycle budget; x20=0 |
| L25 | New (unlocked) | 10000-cycle budget; x20=0 |
| L26 | New (unlocked) | 10000-cycle budget; x20=0 |
| L27 | New (unlocked) | 10000-cycle budget; x20=0 |
| L28 | New (unlocked) | 10000-cycle budget; x20=0 |
| L29 | New (unlocked) | 10000-cycle budget; x20=0 |
| L30 | New (unlocked) | 10000-cycle budget; x20=0 |

**Note:** L14-L16 pass with EBREAK appended (trivial fix, unrelated to this issue).

---

## Proposed Fixes

### Option A: Unified Memory Space (Recommended)

**Approach:** Modify `_execute_vm()` to support a von Neumann memory model where instructions and data share a single address space.

**Implementation:**
```python
def _execute_vm(program, init_regs, init_mem, budget, *, unified_memory=False):
    # Initialize unified memory from program
    memory: dict[int, int] = {}
    if unified_memory:
        for i, word in enumerate(program):
            memory[i * 4] = word  # Store program in data memory
    
    # Instruction fetch reads from memory (unified) or program (Harvard)
    while cycles < budget:
        word_offset = pc >> 2
        if unified_memory:
            word = memory.get(pc, 0)  # ← fetch from data memory
        else:
            if word_offset >= _prog_len:
                error_msg = f"PC out of range: {pc:#x}"
                break
            word = _prog[word_offset]  # ← current behavior
        ...
```

**Pros:**
- Backward-compatible (existing levels work unchanged with `unified_memory=False`)
- Enables meta-evolution and self-modifying code
- Matches real RV32I behavior
- Single clean architectural change

**Cons:**
- Performance impact (dict lookup vs list index)
- Requires curriculum levels to opt-in via level config

### Option B: EBREAK Injection + Static Solutions

**Approach:** Rewrite L17-L30 reference solutions to avoid JALR-based meta-evolution entirely. Each level uses only sequential computation.

**Pros:**
- No VM changes
- Simpler level design

**Cons:**
- Fundamentally changes what the levels test (no more meta-evolution)
- Levels L17-L30 are specifically designed to test self-modifying code capability
- Defeats the purpose of advanced curriculum progression

### Option C: Dual-Mode VM

**Approach:** Add a `vm_mode` flag to levels. Harvard mode for L0-L16, Von Neumann mode for L17-L30.

**Pros:**
- Clean separation of concerns
- Both modes work optimally

**Cons:**
- Adds complexity to VM interface
- Still requires Option A's implementation
- Two code paths to maintain

---

## Recommendation

**Implement Option A (Unified Memory Space)** with the following plan:

1. **Phase 1 (Quick):** Add `unified_memory=False` parameter to `_execute_vm()`. Default preserves backward compatibility. Estimated: 30 lines of code.

2. **Phase 2 (Curriculum):** Add `vm_mode = "unified"` to L17-L30 level definitions. Pass flag to VM during execution.

3. **Phase 3 (Validation):** Re-run curriculum validation with unified mode. All 14 failing levels should pass.

4. **Phase 4 (Optimization):** If performance is critical, add a fast-path for Harvard mode that uses list indexing (current behavior).

**Estimated effort:** 2-4 hours for implementation + validation.

---

## Additional Fixes Required (Independent of This Issue)

| Issue | Levels | Fix |
|-------|--------|-----|
| Missing EBREAK in reference solutions | L14, L15, L16 | Append `{"name": "EBREAK"}` to `reference_solution()` methods |
| Reference logic bug (wrong LW address) | L9, L10 | Correct LW source register/immediate in reference solutions |

---

## Verification

After implementing the fix, re-run the validation script:

```bash
cd "/mnt/c/Users/reese/Projects/Evolution Engine"
python validate_l14_l30.py  # Expected: 17/17 PASS
```

---

## References

- Validation report: [ee-curriculum-validation-2026-07-30.md](./ee-curriculum-validation-2026-07-30.md)
- VM implementation: `gpu_vm_farm.py:306-363` (`_execute_vm` function)
- JALR handler: `gpu_vm_farm.py:515-519`
- EBREAK auto-append: `evolution_engine.py:2273`
