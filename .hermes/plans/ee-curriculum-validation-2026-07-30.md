# Evolution Engine L14-L30 Curriculum Validation Report

**Date:** 2026-07-30  
**Scope:** Validate reference solutions for 17 newly unlocked curriculum levels (L14-L30)  
**Method:** End-to-end VM execution — encode reference solutions, run through pure-Python RV32I VM, verify output registers and memory  
**Rule:** Validation only — no source code modifications

---

## Executive Summary

**0/17 levels pass end-to-end validation without modification.**

However, the failures fall into two clean categories:

| Category | Levels | Root Cause | Fix Complexity |
|----------|--------|------------|----------------|
| **Missing EBREAK** | L14, L15, L16 | Reference solutions lack EBREAK terminator; VM treats program-tail fallthrough as error | Trivial (append EBREAK) |
| **Harvard arch mismatch** | L12, L13, L17-L30 | Reference solutions use JALR to execute instructions from data memory, but VM has separate program/data address spaces | Requires reference solution rewrite or VM change |
| **Reference logic bug** | L9, L10 | Reference loads from wrong memory address vs. expected output | Reference solution fix needed |

With EBREAK appended (as the evolution engine does in practice):

| Status | Levels |
|--------|--------|
| **PASS** (5/5 seeds) | L0-L8, L11, L14, L15, L16 |
| **FAIL** | L9, L10, L12, L13, L17-L30 |

---

## Detailed Findings

### Category 1: Missing EBREAK (L14, L15, L16) — TRIVIAL FIX

All three levels execute correctly and produce correct register values, but the VM returns `success=False` with "PC out of range" because the programs lack an EBREAK terminator.

**Evidence (with EBREAK appended):**
- L14: 5/5 seeds PASS (19 instructions, 19 cycles)
- L15: 5/5 seeds PASS (12 instructions, 10 cycles — BEQ skips correctly)
- L16: 5/5 seeds PASS (10 instructions, 10 cycles)

**Root cause:** The `reference_solution()` methods don't include an EBREAK instruction. The VM's fetch loop treats falling off the end of the program list as an error. The evolution engine's `evolution_engine.py` (line 2273) appends EBREAK automatically before execution, so this works in production but the raw reference solutions fail standalone.

**Fix:** Append `{"name": "EBREAK"}` to each reference solution, or document that EBREAK is always appended by the runner.

### Category 2: Harvard Architecture Mismatch (L12, L13, L17-L30) — FUNDAMENTAL

14 levels use JALR to jump to data-memory addresses where they've stored constructed instructions. The VM's `_execute_vm()` fetches from a separate program list (`program[pc >> 2]`), not from the data memory dict. JALR to address 0x40 fetches `program[16]` instead of `memory[0x40]`.

**Evidence:**
- L12: PC out of range at 0x40 (JALR jumps to program[16] which is a different instruction)
- L13: Hits 10000-cycle budget (infinite loop from JALR misdirection)
- L17-L30: All hit 10000-cycle budget; x20=0 (meta-evolution instruction never executes)

**Root cause:** The reference solutions assume a von Neumann architecture (program and data share memory), but `_execute_vm()` in `gpu_vm_farm.py` uses Harvard architecture (separate program list and data dict).

**Affected pattern (L17 example):**
```
1. Build ADDI x20, x1, x2 instruction word
2. SW to mem[0x40]           # stores to data memory
3. JALR x0, x23, 0          # jumps to PC=0x40
   → VM fetches program[16], NOT memory[0x40]
   → Executes wrong instruction → infinite loop
```

**Impact:** L12, L13 are pre-existing failures (were in the original 13-level curriculum). L17-L30 are the newly unlocked levels — all share this fundamental incompatibility.

### Category 3: Reference Logic Bugs (L9, L10)

**L9:** Reference loads `LW x15, x2, 0` (from mem[count]), but expected output assumes loading from mem[MEMORY_SIZE-8] (dummy address 248). The LW loads from address `count` (2-6), which is always 0, not the dummy value.

**L10:** Same pattern — reference loads from mem[count] instead of the intended dummy address.

**Fix:** Change the LW source register or immediate to target the correct dummy address.

---

## Validation Methodology

For each level L14-L30:
1. Instantiate level class with test seed
2. Generate test instance (inputs + expected outputs)
3. Get reference solution (list of instruction dicts)
4. Encode to RV32I machine code via `rv32i_mask.encode()`
5. Execute through `gpu_vm_farm._execute_vm()` with budget=10000
6. Compare VM output registers/memory to expected values

**Multi-seed testing:** 5 seeds per level (42, 123, 456, 789, 1000)

---

## Files Created

- `/mnt/c/Users/reese/Projects/Evolution Engine/validate_l14_l30.py` — Validation script (can be deleted after review)
- `/mnt/c/Users/reese/Projects/Evolution Engine/.hermes/plans/ee-curriculum-validation-2026-07-30.md` — This report

---

## Recommendations

1. **Immediate:** Add EBREAK to L14-L16 reference solutions (trivial, unblocks these 3 levels)
2. **Decision needed:** For L17-L30, choose one of:
   - **Option A:** Modify `_execute_vm()` to use von Neumann memory model (program stored in data memory, fetched via `memory[pc]`)
   - **Option B:** Rewrite L17-L30 reference solutions to avoid JALR-based meta-evolution (use only sequential computation)
   - **Option C:** Accept that L17-L30 require a different VM mode for self-modifying code
3. **Fix L9/L10:** Correct the LW source address in reference solutions
4. **Clean up:** Delete `validate_l14_l30.py` after review
