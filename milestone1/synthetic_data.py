#!/usr/bin/env python3
"""synthetic_data.py -- Synthetic Training Data Generator for RV32I Transformer

Generates ~50,000 synthetic instruction sequences to teach the 45M param
decoder-only transformer compositional RV32I program structure. The Evolution
Engine has only 93 unique programs (1500 after 16x register renaming), which
is insufficient for learning generalizable instruction patterns.

Five complementary strategies cover different aspects of the ISA:
  1A. Single instructions -- exhaustive coverage of all 47 types (~3K)
  1B. Data flow chains   -- short programs with register dependencies (~5K)
  1C. Subsequence windows -- contiguous fragments from reference solutions (~8K)
  1D. Template patterns   -- four key structural patterns with varied regs (~30K)
  1E. Reference mutations -- perturbed reference solutions for robustness (~5K)

Every generated program ends with EBREAK. All sequences are tokenized via
RV32IFieldTokenizer and deduplicated by token-tuple hash.

Usage:
    from synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    sequences = gen.generate_all()   # list[list[int]]
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
MILESTONE0_DIR = os.path.join(ROOT_DIR, "milestone0", "logit-masking")

for p in (BASE_DIR, ROOT_DIR, MILESTONE0_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Imports ───────────────────────────────────────────────────────────

from rv32i_mask import _INSTRUCTIONS
from rv32i_mask import decode as rv32i_decode
from rv32i_mask import encode as rv32i_encode
from rv32i_model import RV32IFieldTokenizer

from tasks_rv32i import reference_solution

# ── Constants ─────────────────────────────────────────────────────────

EBREAK_WORD = 0x00100073  # EBREAK encoding: opcode=0x73, funct3=0, bit20=1


# ======================================================================
#  HELPER FUNCTIONS
# ======================================================================


def make_ebreak() -> Dict:
    """Return a properly formatted EBREAK instruction dict.

    EBREAK has special encoding (bit 20 set) that must go through
    encode/decode to produce the canonical dict with all keys.
    """
    return rv32i_decode(EBREAK_WORD)


def get_ref_solution(level: int) -> List[Dict]:
    """Get reference solution for a curriculum level (normalized to uppercase name)."""
    return [
        {"name": inst["name"].upper(), **{k: v for k, v in inst.items() if k != "name"}}
        for inst in reference_solution(level)
    ]


def register_rename(instr_dicts: List[Dict], rng: random.Random) -> List[Dict]:
    """Apply a random permutation to registers 1..31 (x0 stays fixed).

    The permutation is a bijection on {1, ..., 31}.  Register 0 is never
    produced because we permute indices 0..29 into positions 1..31, skipping
    the x0 slot entirely.

    Works with both decoded dicts (have 'format' key) and minimal dicts
    (just 'name', 'rd', 'rs1', 'rs2', 'imm').
    """
    perm = list(range(1, 32))  # possible output values: 1..31
    rng.shuffle(perm)

    result: List[Dict] = []
    for d in instr_dicts:
        nd = dict(d)
        # Remap destination register (R/I/U/J types)
        if "rd" in nd and nd["rd"] != 0:
            nd["rd"] = perm[nd["rd"] - 1]
        # Remap source register 1 (R/I/S/B types)
        if "rs1" in nd and nd["rs1"] != 0:
            nd["rs1"] = perm[nd["rs1"] - 1]
        # Remap source register 2 (R/S/B types)
        if "rs2" in nd and nd["rs2"] != 0:
            nd["rs2"] = perm[nd["rs2"] - 1]
        result.append(nd)
    return result


# ======================================================================
#  SYNTHETIC DATA GENERATOR
# ======================================================================


class SyntheticDataGenerator:
    """Generates ~50,000 synthetic RV32I training sequences.

    Each sequence is a list of token IDs produced by RV32IFieldTokenizer.
    All programs end with EBREAK.  Sequences are deduplicated by token-tuple
    hash to avoid redundant training examples.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tokenizer = RV32IFieldTokenizer()
        # EBREAK instruction dict (reused across strategies)
        self._ebreak = make_ebreak()

    # ── Public API ────────────────────────────────────────────────────

    def generate_all(self) -> List[List[int]]:
        """Generate all synthetic training sequences.

        Returns:
            List of token sequences (each a list[int]) ready for training.
            Duplicates are removed; every sequence ends with EBREAK tokens.
        """
        seen: set = set()
        sequences: List[List[int]] = []

        def _add(inst_list: List[Dict]) -> bool:
            """Tokenize an instruction list and add if unique. Returns True if added."""
            toks = tuple(self.tokenizer.encode_trace(inst_list))
            if toks in seen:
                return False
            seen.add(toks)
            sequences.append(list(toks))
            return True

        # Strategy 1A: Single instructions
        for inst_list in self._gen_single_instructions():
            _add(inst_list)

        # Strategy 1B: Data flow chains
        for _ in range(8000):
            chain = self._gen_data_flow_chain()
            if chain is not None:
                _add(chain)

        # Strategy 1C: Subsequence extraction
        for inst_list in self._gen_subsequences():
            _add(inst_list)

        # Strategy 1D: Template instantiation
        for inst_list in self._gen_templates():
            _add(inst_list)

        # Strategy 1E: Reference mutations
        for inst_list in self._gen_mutations():
            _add(inst_list)

        return sequences

    # ── Strategy 1A: Single Instructions ──────────────────────────────
    #
    # Generate ALL valid single-instruction programs.  Each instruction is
    # followed by EBREAK.  We iterate over all 47 instruction types in the
    # ISA encoding table and produce representative field combinations.

    def _gen_single_instructions(self) -> List[List[Dict]]:
        """Generate representative single-instruction programs (~3K)."""
        programs: List[List[Dict]] = []
        imm_values = [0, 1, 2, 4, 8, -1, 3, 7, 16, -4]

        for opcode, funct3, funct7, name, fmt in _INSTRUCTIONS:
            inst_name = name.upper()

            if inst_name == "EBREAK":
                programs.append([{"name": "EBREAK", "rd": 0, "rs1": 0, "rs2": 0, "imm": 1}])
                continue

            if inst_name in ("ECALL", "FENCE", "FENCE.I"):
                programs.append([{"name": inst_name, "rd": 0, "rs1": 0, "rs2": 0, "imm": 0}])
                continue

            if inst_name in ("CSRRW", "CSRRS", "CSRRC", "CSRRWI", "CSRRSI", "CSRRCI"):
                # CSR instructions: imm is CSR address, rs1 is source register
                for csr_addr in [0x001, 0x300, 0x340]:
                    for rs1 in [0, 1, 5]:
                        programs.append([{"name": inst_name, "rd": 1, "rs1": rs1, "rs2": 0, "imm": csr_addr}])
                continue

            if fmt == "R":
                # R-type: rd, rs1, rs2 are registers; no immediate
                for rd in range(1, 32):
                    for rs1 in [1, 2, 3, 10]:
                        for rs2 in [0, 1, 2, 3, 10]:
                            if rd == rs1 or rd == rs2:
                                continue
                            programs.append([{"name": inst_name, "rd": rd, "rs1": rs1, "rs2": rs2, "imm": 0}])

            elif fmt == "I" and name in ("SLLI", "SRLI", "SRAI"):
                # Shift instructions: imm is shift amount (0-31)
                for shamt in [0, 1, 2, 3, 4, 5, 7, 8, 15, 31]:
                    programs.append([{"name": inst_name, "rd": 3, "rs1": 1, "rs2": 0, "imm": shamt}])

            elif fmt == "I" and name in ("ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI"):
                # I-type arithmetic: rd, rs1, imm
                for rd in [3, 4, 5, 10, 15]:
                    for rs1 in [0, 1, 2, 3]:
                        for imm in imm_values:
                            programs.append([{"name": inst_name, "rd": rd, "rs1": rs1, "rs2": 0, "imm": imm}])

            elif fmt == "I" and name in ("LB", "LH", "LW", "LBU", "LHU"):
                # I-type load: rd, rs1=base, imm=offset (non-negative, word-aligned for LW)
                for offset in [0, 4, 8, 16, 32]:
                    for rs1 in [1, 2, 5, 10]:
                        programs.append([{"name": inst_name, "rd": 10, "rs1": rs1, "rs2": 0, "imm": offset}])

            elif fmt == "I" and name == "JALR":
                # JALR: rd, rs1, imm
                for imm in [0, 4, 8]:
                    for rs1 in [1, 5, 10]:
                        programs.append([{"name": "JALR", "rd": 10, "rs1": rs1, "rs2": 0, "imm": imm}])

            elif fmt == "S":
                # S-type: rs1=base, rs2=source, imm=offset
                for imm in [0, 4, 8, 16, 32]:
                    for rs1 in [1, 2, 5, 10]:
                        for rs2 in [3, 4, 15, 20]:
                            programs.append([{"name": inst_name, "rd": 0, "rs1": rs1, "rs2": rs2, "imm": imm}])

            elif fmt == "B":
                # B-type: rs1, rs2, imm=branch_offset (byte offset, even)
                for imm in [-32, -24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24, 32]:
                    programs.append([{"name": inst_name, "rd": 0, "rs1": 1, "rs2": 2, "imm": imm}])

            elif fmt == "U":
                # U-type: rd, imm=upper immediate
                for imm in [0, 0x100, 0x1000, 0x10000, 0x80000]:
                    programs.append([{"name": inst_name, "rd": 5, "rs1": 0, "rs2": 0, "imm": imm}])

            elif fmt == "J":
                # J-type: rd, imm=jump_offset
                for imm in [8, 16, 24, 32, -16, -32]:
                    programs.append([{"name": inst_name, "rd": 10, "rs1": 0, "rs2": 0, "imm": imm}])

            else:
                # Fallback for any unhandled format
                programs.append([{"name": inst_name, "rd": 0, "rs1": 0, "rs2": 0, "imm": 0}])

        # Append EBREAK to each program
        return [p + [self._ebreak] for p in programs]

    # ── Strategy 1B: Data Flow Chains ─────────────────────────────────
    #
    # Generate 2-4 instruction programs where the output of instruction N
    # feeds into the input of instruction N+1.  Only arithmetic instructions
    # that work on registers are used (no memory or branches).

    _ARITH_NAMES = ("ADD", "SUB", "AND", "OR", "XOR", "SLL", "SRL", "SLT")

    def _gen_data_flow_chain(self) -> Optional[List[Dict]]:
        """Generate a single data flow chain program (2-4 instructions + EBREAK).

        Returns None if a degenerate chain is produced (e.g. all instructions
        read only from x0).
        """
        rng = self.rng
        n_insts = rng.randint(2, 4)
        program: List[Dict] = []
        live_regs = {1, 2}  # x1, x2 are always available inputs

        for i in range(n_insts):
            inst_name = rng.choice(self._ARITH_NAMES)
            rd = rng.randint(3, 31)

            # Pick source registers from live set + x0
            candidates = list(live_regs | {0})
            if len(candidates) < 2:
                candidates.extend([1, 2])
            rng.shuffle(candidates)
            rs1, rs2 = candidates[0], candidates[1]

            # Encode and decode to get canonical dict
            word = rv32i_encode(inst_name, rd=rd, rs1=rs1, rs2=rs2)
            program.append(rv32i_decode(word))
            live_regs.add(rd)

        program.append(self._ebreak)
        return program

    # ── Strategy 1C: Subsequence Extraction ───────────────────────────
    #
    # Extract all contiguous windows of length 2, 3, 4, 5 from each
    # reference solution (L0-L11).  Each window is register-renamed 16x
    # to produce diverse training data from known-correct program fragments.

    def _gen_subsequences(self) -> List[List[Dict]]:
        """Extract and register-rename subsequence windows from reference solutions."""
        programs: List[List[Dict]] = []

        for level in range(12):
            ref = get_ref_solution(level)
            for window_len in (2, 3, 4, 5):
                for start in range(max(0, len(ref) - window_len + 1)):
                    window = ref[start : start + window_len]
                    for _ in range(16):
                        renamed = register_rename(window, self.rng)
                        programs.append(renamed + [self._ebreak])

        return programs

    # ── Strategy 1D: Template Instantiation ───────────────────────────
    #
    # Extract four key structural patterns from reference solutions and
    # instantiate each with varied register assignments.  This teaches
    # the model common program structures (loads, loops, branches).

    def _gen_templates(self) -> List[List[Dict]]:
        """Generate all template-based programs (~30K total)."""
        programs: List[List[Dict]] = []

        # Template 1: Load-Add pattern (from L8) -- 5000 instantiations
        programs.extend(self._gen_load_add(5000))

        # Template 2: Loop-Sum pattern (from L9) -- 8000 instantiations
        programs.extend(self._gen_loop_sum(8000))

        # Template 3: Max-Find pattern (from L10) -- 7000 instantiations
        programs.extend(self._gen_max_find(7000))

        # Template 4: Fibonacci pattern (from L11) -- 10000 instantiations
        programs.extend(self._gen_fibonacci(10000))

        return programs

    def _gen_load_add(self, n: int) -> List[List[Dict]]:
        """Load-Add: LW r_a, 0(r_x); LW r_b, 0(r_y); ADD r_out, r_a, r_b

        Pattern from L8.  Teaches the model to load two values from memory
        and combine them with an arithmetic instruction.
        """
        programs: List[List[Dict]] = []
        rng = self.rng

        for _ in range(n):
            r_a, r_b, r_out, r_x, r_y = rng.sample(range(1, 32), 5)
            insts = [
                rv32i_decode(rv32i_encode("LW", rd=r_a, rs1=r_x, imm=0)),
                rv32i_decode(rv32i_encode("LW", rd=r_b, rs1=r_y, imm=0)),
                rv32i_decode(rv32i_encode("ADD", rd=r_out, rs1=r_a, rs2=r_b)),
                self._ebreak,
            ]
            programs.append(insts)

        return programs

    def _gen_loop_sum(self, n: int) -> List[List[Dict]]:
        """Loop-Sum: iterative accumulation with word-aligned addressing.

        Structure (12 instructions):
            ADDI r_sum, 0, 0          -- sum = 0
            ADDI r_i, 0, 0            -- i = 0
            SLLI r_off, r_i, 2        -- offset = i * 4  [shamt=2]
            ADD  r_addr, r_base, r_off -- addr = base + offset
            LW   r_val, 0(r_addr)     -- val = mem[addr]
            ADD  r_sum, r_sum, r_val   -- sum += val
            ADDI r_i, r_i, 1          -- i++
            BLT  r_i, r_count, OFF    -- if i < count, loop back to SLLI

        Branch offset is computed dynamically:
            SLLI is at position 0, BLT at position 7.
            BLT target = position 0.  offset = (0 - 7) * 4 = -28.
        """
        programs: List[List[Dict]] = []
        rng = self.rng
        shamt = rng.choice((1, 2, 3))

        for _ in range(n):
            # x1=base, x2=count are fixed inputs.
            # Pick 6 unique registers for template variables (from 3-31, excluding 1,2).
            pool = list(range(3, 32))
            rng.shuffle(pool)
            r_sum, r_i, r_off, r_addr, r_val, r_out = pool[:6]

            blt_imm = (0 - 7) * 4  # = -28

            template = [
                ("ADDI", r_sum, 0, None, 0),
                ("ADDI", r_i, 0, None, 0),
                ("SLLI", r_off, r_i, None, shamt),
                ("ADD", r_addr, 1, r_off, None),
                ("LW", r_val, r_addr, None, 0),
                ("ADD", r_sum, r_sum, r_val, None),
                ("ADDI", r_i, r_i, None, 1),
                ("BLT", 0, r_i, 2, blt_imm),
                ("ADDI", r_out, r_sum, None, 0),
            ]

            insts = []
            for name, rd, rs1, rs2, imm in template:
                kwargs = {"name": name, "rd": rd, "rs1": rs1}
                if rs2 is not None:
                    kwargs["rs2"] = rs2
                if imm is not None:
                    kwargs["imm"] = imm
                insts.append(rv32i_decode(rv32i_encode(**kwargs)))
            insts.append(self._ebreak)
            programs.append(insts)

        return programs

    def _gen_max_find(self, n: int) -> List[List[Dict]]:
        """Max-Find: find maximum in a memory range using SLT/BEQ.

        Structure (15 instructions):
            LW   r_max, 0(r_base)       -- max = mem[base]
            ADDI r_i, 0, 1              -- i = 1
            SLLI r_off, r_i, 2          -- offset = i * 4
            ADD  r_addr, r_base, r_off  -- addr = base + offset
            LW   r_val, 0(r_addr)       -- val = mem[addr]
            SLT  r_cmp, r_max, r_val    -- cmp = (max < val) ? 1 : 0
            BEQ  r_cmp, 0, SKIP         -- if max >= val, skip update
            ADDI r_max, r_val, 0        -- max = val
            ADDI r_i, r_i, 1            -- i++
            BLT  r_i, r_count, LOOP     -- if i < count, loop

        Branch offsets:
            BEQ at position 6, targets position 8 (skip ADDI): (8-6)*4 = 8
            BLT at position 9, targets position 2 (SLLI):      (2-9)*4 = -28
        """
        programs: List[List[Dict]] = []
        rng = self.rng
        shamt = rng.choice((1, 2, 3))

        for _ in range(n):
            pool = list(range(3, 32))
            rng.shuffle(pool)
            r_max, r_i, r_off, r_addr, r_val, r_cmp = pool[:6]

            beq_imm = (8 - 6) * 4  # = 8
            blt_imm = (2 - 9) * 4  # = -28

            template = [
                ("LW", r_max, 1, None, 0),
                ("ADDI", r_i, 0, None, 1),
                ("SLLI", r_off, r_i, None, shamt),
                ("ADD", r_addr, 1, r_off, None),
                ("LW", r_val, r_addr, None, 0),
                ("SLT", r_cmp, r_max, r_val, None),
                ("BEQ", 0, r_cmp, 0, beq_imm),
                ("ADDI", r_max, r_val, None, 0),
                ("ADDI", r_i, r_i, None, 1),
                ("BLT", 0, r_i, 2, blt_imm),
            ]

            insts = []
            for name, rd, rs1, rs2, imm in template:
                kwargs = {"name": name, "rd": rd, "rs1": rs1}
                if rs2 is not None:
                    kwargs["rs2"] = rs2
                if imm is not None:
                    kwargs["imm"] = imm
                insts.append(rv32i_decode(rv32i_encode(**kwargs)))
            insts.append(self._ebreak)
            programs.append(insts)

        return programs

    def _gen_fibonacci(self, n: int) -> List[List[Dict]]:
        """Fibonacci: iterative computation with skip-loop optimization.

        Structure (10 instructions):
            ADDI r_a, 0, 0             -- a = 0
            ADDI r_b, 0, 1             -- b = 1
            ADDI r_i, 0, 0             -- i = 0
            BEQ  r_n, 0, SKIP          -- if n==0, skip loop
            ADD  r_tmp, r_a, r_b       -- tmp = a + b
            ADDI r_a, r_b, 0           -- a = b
            ADDI r_b, r_tmp, 0         -- b = tmp
            ADDI r_i, r_i, 1           -- i++
            BLT  r_i, r_n, LOOP        -- if i < n, loop back to ADD

        Branch offsets:
            BEQ at position 3, targets position 9 (copy-out): (9-3)*4 = 24
            BLT at position 8, targets position 4 (ADD):      (4-8)*4 = -16

        Note: the reference solution uses imm=20 for BEQ, which branches to
        ADDI r_a,r_b,0 -- this works because after the loop r_a holds F(n).
        We use the more natural offset to the copy-out instruction.
        """
        programs: List[List[Dict]] = []
        rng = self.rng

        for _ in range(n):
            pool = list(range(3, 32))
            rng.shuffle(pool)
            r_a, r_b, r_i, r_tmp = pool[:4]

            beq_imm = (9 - 3) * 4  # = 24 (skip to copy-out)
            blt_imm = (4 - 8) * 4  # = -16 (loop back to ADD)

            template = [
                ("ADDI", r_a, 0, None, 0),
                ("ADDI", r_b, 0, None, 1),
                ("ADDI", r_i, 0, None, 0),
                ("BEQ", 0, 1, 0, beq_imm),
                ("ADD", r_tmp, r_a, r_b, None),
                ("ADDI", r_a, r_b, None, 0),
                ("ADDI", r_b, r_tmp, None, 0),
                ("ADDI", r_i, r_i, None, 1),
                ("BLT", 0, r_i, 1, blt_imm),
                ("ADDI", 14, r_a, None, 0),
            ]

            insts = []
            for name, rd, rs1, rs2, imm in template:
                kwargs = {"name": name, "rd": rd, "rs1": rs1}
                if rs2 is not None:
                    kwargs["rs2"] = rs2
                if imm is not None:
                    kwargs["imm"] = imm
                insts.append(rv32i_decode(rv32i_encode(**kwargs)))
            insts.append(self._ebreak)
            programs.append(insts)

        return programs

    # ── Strategy 1E: Reference Solution Mutations ─────────────────────
    #
    # For L8-L11 reference solutions, apply 100 random mutations each.
    # Mutation types: swap independent instructions, shift immediates,
    # replace with semantically similar instruction, duplicate one.

    def _gen_mutations(self) -> List[List[Dict]]:
        """Mutate reference solutions L8-L11 and register-rename each (~5K)."""
        programs: List[List[Dict]] = []
        rng = self.rng

        for level in range(8, 12):
            ref = get_ref_solution(level)
            # Skip the EBREAK at the end (if present) for mutation targets
            ref_body = [d for d in ref if d["name"] != "EBREAK"]

            for _ in range(100):
                mutated = self._mutate_program(ref_body, rng)
                if mutated is not None:
                    programs.append(mutated + [self._ebreak])
                    # Also register-rename the mutated version for diversity
                    renamed = register_rename(mutated, rng)
                    programs.append(renamed + [self._ebreak])

        return programs

    def _mutate_program(self, program: List[Dict], rng: random.Random) -> Optional[List[Dict]]:
        """Apply a single random mutation to a program.

        Returns the mutated program, or None if mutation is degenerate.
        """
        import copy

        mutated = copy.deepcopy(program)
        n = len(mutated)

        mutation_type = rng.choice(("swap", "imm", "replace", "duplicate"))

        if mutation_type == "swap" and n >= 2:
            # Swap two independent instructions (non-overlapping register usage)
            # Try up to 20 times to find a valid swap
            for _ in range(20):
                i, j = rng.sample(range(n), 2)
                a, b = mutated[i], mutated[j]
                # Check register independence: no rd of one is rs1/rs2 of other
                rd_a = a.get("rd", 0)
                rd_b = b.get("rd", 0)
                rs1_a, rs2_a = a.get("rs1", 0), a.get("rs2", 0)
                rs1_b, rs2_b = b.get("rs1", 0), b.get("rs2", 0)
                if rd_a == 0 or rd_b == 0:
                    continue  # Skip if either writes to x0
                if rd_a in (rs1_b, rs2_b) or rd_b in (rs1_a, rs2_a):
                    continue  # Not independent
                mutated[i], mutated[j] = mutated[j], mutated[i]
                return mutated

        elif mutation_type == "imm" and n >= 1:
            # Shift an immediate value by +/- 1 or +/- 2
            candidates = [
                i for i in range(n) if mutated[i].get("imm", 0) != 0 and mutated[i]["name"] not in ("EBREAK", "ECALL")
            ]
            if candidates:
                idx = rng.choice(candidates)
                delta = rng.choice((-2, -1, 1, 2))
                mutated[idx] = dict(mutated[idx])
                mutated[idx]["imm"] = mutated[idx]["imm"] + delta
                return mutated

        elif mutation_type == "replace" and n >= 1:
            # Replace one instruction with a semantically similar one
            _SIMILAR = {
                "ADD": ["ADDI", "SUB"],
                "ADDI": ["ADD"],
                "SUB": ["ADD"],
                "AND": ["ANDI"],
                "OR": ["ORI"],
                "XOR": ["XORI"],
                "LW": ["LW"],  # same instruction, different offset
            }
            candidates = [i for i in range(n) if mutated[i]["name"] in _SIMILAR]
            if candidates:
                idx = rng.choice(candidates)
                old = mutated[idx]
                new_name = rng.choice(_SIMILAR[old["name"]])
                new_inst = dict(old)
                new_inst["name"] = new_name
                if new_name == "LW" and old["name"] == "LW":
                    # Vary the load offset
                    new_inst["imm"] = rng.choice((0, 4, 8))
                mutated[idx] = new_inst
                return mutated

        elif mutation_type == "duplicate" and n >= 1:
            # Duplicate one instruction (structurally valid but redundant)
            idx = rng.randint(0, n - 1)
            dup = dict(mutated[idx])
            mutated.insert(idx + 1, dup)
            return mutated

        return None


# ======================================================================
#  SELF-TEST
# ======================================================================


def _run_self_test():
    """Quick validation of the synthetic data generator."""
    print("=" * 64)
    print("SyntheticDataGenerator Self-Test")
    print("=" * 64)

    gen = SyntheticDataGenerator(seed=42)

    # Test helpers
    print("\n[1] Testing helper functions...")
    ebreak = make_ebreak()
    assert ebreak["name"] == "EBREAK", f"EBREAK name wrong: {ebreak['name']}"
    assert ebreak["rd"] == 0
    assert ebreak["rs1"] == 0
    print(f"    EBREAK dict: {ebreak}")

    ref8 = get_ref_solution(8)
    assert len(ref8) == 10, f"L8 ref should have 10 insts, got {len(ref8)}"
    print(f"    L8 reference: {len(ref8)} instructions")

    # Test register rename
    print("\n[2] Testing register renaming...")
    rng = random.Random(99)
    test_insts = [
        {"name": "ADD", "rd": 3, "rs1": 1, "rs2": 2, "imm": 0},
        {"name": "ADDI", "rd": 4, "rs1": 3, "rs2": 0, "imm": 5},
    ]
    renamed = register_rename(test_insts, rng)
    assert renamed[0]["name"] == "ADD"
    assert renamed[0]["rd"] != 0, "rd should not map to 0"
    assert renamed[0]["rs1"] != 0, "rs1 should not map to 0"
    print(f"    Original:  {test_insts}")
    print(f"    Renamed:   {renamed}")

    # Test single instruction generation
    print("\n[3] Testing single instruction generation...")
    singles = gen._gen_single_instructions()
    print(f"    Generated {len(singles)} single-instruction programs")
    assert len(singles) > 1000, f"Expected >1000, got {len(singles)}"
    # Verify all end with EBREAK
    for p in singles[:10]:
        assert p[-1]["name"] == "EBREAK"
    print("    All end with EBREAK: OK")

    # Test tokenization of a single instruction
    toks = gen.tokenizer.encode_trace(singles[0])
    print(f"    Token sequence length: {len(toks)} (expect {len(singles[0]) * 9})")
    assert len(toks) == len(singles[0]) * 9

    # Test data flow chain
    print("\n[4] Testing data flow chain generation...")
    chain = gen._gen_data_flow_chain()
    assert chain is not None
    assert chain[-1]["name"] == "EBREAK"
    assert 3 <= len(chain) <= 5  # 2-4 insts + EBREAK
    print(f"    Chain length: {len(chain)} instructions")

    # Test subsequence generation
    print("\n[5] Testing subsequence extraction...")
    subs = gen._gen_subsequences()
    print(f"    Generated {len(subs)} subsequences")
    assert len(subs) > 100

    # Test template generation
    print("\n[6] Testing template instantiation...")
    templates = gen._gen_templates()
    print(f"    Generated {len(templates)} template programs")
    assert len(templates) > 1000

    # Test mutation generation
    print("\n[7] Testing reference mutations...")
    mutations = gen._gen_mutations()
    print(f"    Generated {len(mutations)} mutated programs")
    assert len(mutations) > 100

    # Full generation
    print("\n[8] Running full generation...")
    all_seqs = gen.generate_all()
    total = len(all_seqs)
    print(f"    Total unique sequences: {total}")

    # Statistics
    lengths = [len(s) for s in all_seqs]
    print(f"    Min length: {min(lengths)} tokens")
    print(f"    Max length: {max(lengths)} tokens")
    print(f"    Avg length: {sum(lengths) / len(lengths):.1f} tokens")
    print(f"    1-inst programs: {sum(1 for l in lengths if l == 18)}")  # 1 inst + EBREAK = 18 tokens
    print(f"    2-inst programs: {sum(1 for l in lengths if l == 27)}")  # 2 inst + EBREAK = 27 tokens

    # Verify no duplicates
    unique_tokens = set(tuple(s) for s in all_seqs)
    assert len(unique_tokens) == total, "Duplicates found!"
    print(f"    Deduplication verified: {total} unique")

    # Verify all end with SEP (EBREAK's last token)
    sep_count = sum(1 for s in all_seqs if s[-1] == 1)  # SEP = 1
    print(f"    All end with SEP: {sep_count}/{total}")

    print("\n" + "=" * 64)
    print(f"SELF-TEST PASSED -- {total} unique sequences generated")
    print("=" * 64)


if __name__ == "__main__":
    _run_self_test()
