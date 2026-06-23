#!/usr/bin/env python3
"""
RV32I Mutation Proposer with Trace Analysis
=============================================
Port of simulation/evolution.py (DirectedMutator, TraceAnalyzer classes)
to real RV32I instruction set using rv32i_mask encoding.

Provides:
  - TraceAnalyzer: reads safety-kernel trace entries, identifies divergence
  - DirectedMutator: proposes targeted RV32I mutations based on trace analysis
  - stochastic_perturbation: 10% random valid-instruction perturbations
  - All mutations produce valid RV32I encodings (rv32i_mask validation)
  - Edge case handling: empty/single-instruction/no-EBREAK programs
"""

import copy
import random
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Mutation Strategy Enum ───────────────────────────────────────────

from enum import Enum


class MutationStrategy(str, Enum):
    """Mutation strategies for challenger generation.

    Values match string representations used in strategy archive logging.
    """
    RANDOM = "RANDOM"
    TASK_AWARE = "TASK_AWARE"
    DIRECTED = "DIRECTED"
    PERTURBATION_CAPPED = "PERTURBATION_CAPPED"
    GROW = "GROW"
    HYBRID = "HYBRID"
    MODEL_GUIDED = "MODEL_GUIDED"


# ── Import RV32I encoding from milestone0 ──────────────────────────────
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
_rv32i_mask_dir = os.path.join(_project_root, 'milestone0', 'logit-masking')
if _rv32i_mask_dir not in sys.path:
    sys.path.insert(0, _rv32i_mask_dir)

from rv32i_mask import (                             # type: ignore[import]
    encode as rv32i_encode,
    decode as rv32i_decode,
    is_valid,
    get_valid_combinations,
    get_encodings,
    RV32IEncoding,
    DC,
    Opcode,
)


# ── Constants ──────────────────────────────────────────────────────────

# Maximum program length (in 32-bit words)
MAX_PROGRAM_LEN = 256

# EBREAK instruction encoding
EBREAK = rv32i_encode(name='EBREAK', imm=1)

# A "NOP-like" instruction: ADDI x0, x0, 0 — safe no-op
NOP = rv32i_encode(name='ADDI', rd=0, rs1=0, imm=0)

# Valid instruction names for random generation (excluding system instrs)
INSTRUCTION_NAMES_RTYPE   = ['ADD', 'SUB', 'SLL', 'SLT', 'SLTU',
                              'XOR', 'SRL', 'SRA', 'OR', 'AND']
INSTRUCTION_NAMES_ITYPE   = ['ADDI', 'SLTI', 'SLTIU', 'XORI',
                              'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI']
INSTRUCTION_NAMES_LOAD    = ['LB', 'LH', 'LW', 'LBU', 'LHU']
INSTRUCTION_NAMES_STORE   = ['SB', 'SH', 'SW']
INSTRUCTION_NAMES_BRANCH  = ['BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU']
INSTRUCTION_NAMES_OTHER   = ['LUI', 'AUIPC', 'JAL', 'JALR',
                              'FENCE', 'FENCE.I']

ALL_INSTRUCTION_NAMES = (
    INSTRUCTION_NAMES_RTYPE +
    INSTRUCTION_NAMES_ITYPE +
    INSTRUCTION_NAMES_LOAD +
    INSTRUCTION_NAMES_STORE +
    INSTRUCTION_NAMES_BRANCH +
    INSTRUCTION_NAMES_OTHER
)

# Branch swap pairs (flip funct3 to get opposite branch condition)
BRANCH_SWAP_PAIRS: Dict[str, str] = {
    'BEQ': 'BNE', 'BNE': 'BEQ',
    'BLT': 'BGE', 'BGE': 'BLT',
    'BLTU': 'BGEU', 'BGEU': 'BLTU',
}

# ALU swap pairs (flip funct7 to swap ADD↔SUB, SRL↔SRA)
ALU_SWAP_PAIRS: Dict[str, str] = {
    'ADD': 'SUB', 'SUB': 'ADD',
    'SRL': 'SRA', 'SRA': 'SRL',
}


# ── Data Structures ────────────────────────────────────────────────────

@dataclass
class TraceEntry:
    """Matches safety-kernel TraceEntry from types.rs.

    Fields:
        regs:  32-element register file state (i32 values)
        pc:    Program counter (u32)
        cycle: Cycle/instruction count (u64)
    """
    regs: List[int]
    pc: int
    cycle: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TraceEntry':
        return cls(
            regs=list(d.get('regs', [0] * 32)),
            pc=d.get('pc', 0),
            cycle=d.get('cycle', 0),
        )


@dataclass
class TraceAnalysis:
    """Result of analyzing an execution trace.

    Fields:
        divergence_point: Step index where divergence first occurs
        divergence_type:  Category of failure
        final_regs:       Final register state (32 elements)
        trace_length:     Number of trace entries
        suggestion:       Mutation suggestion metadata dict
        divergence_pc:    PC at divergence point (if available)
        expected:         Expected output dict (pass-through)
        inputs:           Input state dict (pass-through)
        error:            Error string from safety-kernel (if any)
    """
    divergence_point: Optional[int]
    divergence_type: str
    final_regs: List[int]
    trace_length: int
    suggestion: Dict[str, Any]
    divergence_pc: Optional[int] = None
    expected: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ── TraceAnalyzer ──────────────────────────────────────────────────────

class TraceAnalyzer:
    """
    Analyzes RV32I safety-kernel execution traces to identify failures.

    Reads trace entries (regs[32], pc, cycle) and compares against
    expected output to determine divergence type and propose fixes.
    """

    @staticmethod
    def analyze(
        trace: List[Dict[str, Any]],
        expected: Dict[str, Any],
        inputs: Dict[str, Any],
        program: Optional[List[int]] = None,
        error: Optional[str] = None,
        budget: Optional[int] = None,
    ) -> TraceAnalysis:
        """Analyze a safety-kernel execution trace and return failure analysis.

        Args:
            trace:    List of trace entries from safety-kernel response.
                      Each entry has 'regs' (list[32]), 'pc' (int), 'cycle' (int).
            expected: Expected output dict (e.g. {'registers': {1: 42}}).
            inputs:   Input state dict (e.g. {'registers': {1: 10}}).
            program:  Original program (list of 32-bit instruction words).
            error:    Error string from safety-kernel response, if any.
            budget:   Instruction budget used for evaluation.

        Returns:
            TraceAnalysis with divergence info and mutation suggestion.
        """
        # ── Empty trace ─────────────────────────────────────
        if not trace:
            if error:
                return TraceAnalysis(
                    divergence_point=0,
                    divergence_type='illegal_instruction',
                    final_regs=[0] * 32,
                    trace_length=0,
                    suggestion={
                        'type': 'replace_program',
                        'reason': f'execution error: {error}',
                    },
                    error=error,
                    expected=expected,
                    inputs=inputs,
                )
            return TraceAnalysis(
                divergence_point=0,
                divergence_type='unknown',
                final_regs=[0] * 32,
                trace_length=0,
                suggestion={'type': 'regenerate', 'reason': 'empty trace'},
                expected=expected,
                inputs=inputs,
            )

        final_regs = trace[-1].get('regs', [0] * 32)

        # ── Check for illegal instruction error ─────────────
        if error and 'illegal instruction' in error.lower():
            div_pc = TraceAnalyzer._extract_pc_from_error(error)
            return TraceAnalysis(
                divergence_point=len(trace) - 1,
                divergence_type='illegal_instruction',
                final_regs=final_regs,
                trace_length=len(trace),
                suggestion={
                    'type': 'replace_instruction',
                    'pc': div_pc,
                    'reason': error,
                },
                divergence_pc=div_pc,
                error=error,
                expected=expected,
                inputs=inputs,
            )

        # ── Check for PC out of bounds error ────────────────
        if error and 'pc out of bounds' in error.lower():
            return TraceAnalysis(
                divergence_point=len(trace) - 1,
                divergence_type='pc_out_of_bounds',
                final_regs=final_regs,
                trace_length=len(trace),
                suggestion={
                    'type': 'add_terminator',
                    'reason': 'PC went out of bounds; program likely needs EBREAK',
                },
                error=error,
                expected=expected,
                inputs=inputs,
            )

        # ── Compare final register state against expected ───
        expected_regs: Dict = expected.get('registers', {})

        # Normalise keys to int
        norm_expected: Dict[int, int] = {}
        for k, v in expected_regs.items():
            if isinstance(k, str):
                try:
                    norm_expected[int(k)] = int(v)
                except (ValueError, TypeError):
                    continue
            else:
                norm_expected[int(k)] = int(v)

        wrong_regs: Dict[int, Dict[str, int]] = {}
        for reg_idx, expected_val in norm_expected.items():
            actual = final_regs[reg_idx] if reg_idx < len(final_regs) else 0
            if actual != expected_val:
                wrong_regs[reg_idx] = {'actual': actual, 'expected': expected_val}

        # ── Check for ran out of budget ─────────────────────
        ran_out = False
        if budget is not None and len(trace) >= budget:
            ran_out = True

        # ── Check if program has EBREAK ─────────────────────
        prog_has_ebreak = False
        if program:
            for word in program:
                try:
                    decoded = rv32i_decode(word, validate=False)
                    if decoded.get('name') == 'EBREAK':
                        prog_has_ebreak = True
                        break
                except Exception:
                    continue

        # ── Wrong register values detected ──────────────────
        if wrong_regs:
            primary_reg = min(wrong_regs.keys())
            primary_info = wrong_regs[primary_reg]
            return TraceAnalysis(
                divergence_point=len(trace) - 1,
                divergence_type='wrong_reg_value',
                final_regs=final_regs,
                trace_length=len(trace),
                suggestion={
                    'type': 'fix_register',
                    'reg': primary_reg,
                    'actual': primary_info['actual'],
                    'expected': primary_info['expected'],
                    'reason': (
                        f'reg[{primary_reg}] = {primary_info["actual"]}, '
                        f'expected {primary_info["expected"]}'
                    ),
                },
                expected=expected,
                inputs=inputs,
            )

        # ── Ran out of budget ───────────────────────────────
        if ran_out:
            return TraceAnalysis(
                divergence_point=len(trace) - 1,
                divergence_type='ran_out_of_budget',
                final_regs=final_regs,
                trace_length=len(trace),
                suggestion={
                    'type': 'reduce_complexity',
                    'reason': 'program exceeded instruction budget without halting',
                    'has_ebreak': prog_has_ebreak,
                },
                expected=expected,
                inputs=inputs,
            )

        # ── No EBREAK in program ────────────────────────────
        if not prog_has_ebreak:
            return TraceAnalysis(
                divergence_point=len(trace) - 1,
                divergence_type='no_terminator',
                final_regs=final_regs,
                trace_length=len(trace),
                suggestion={
                    'type': 'add_ebreak',
                    'reason': 'program has no EBREAK instruction',
                },
                expected=expected,
                inputs=inputs,
            )

        # ── Check for halted early (PC stuck) ───────────────
        for i in range(1, len(trace)):
            prev_pc = trace[i - 1].get('pc', 0)
            curr_pc = trace[i].get('pc', 0)
            if curr_pc == prev_pc and i < len(trace) - 1:
                return TraceAnalysis(
                    divergence_point=i,
                    divergence_type='halted_early',
                    final_regs=final_regs,
                    trace_length=len(trace),
                    suggestion={
                        'type': 'check_halt_position',
                        'reason': f'program halted at step {i}; may be too short',
                    },
                    divergence_pc=curr_pc,
                    expected=expected,
                    inputs=inputs,
                )

        # ── Default: unknown divergence → stochastic ────────
        return TraceAnalysis(
            divergence_point=len(trace) - 1,
            divergence_type='unknown',
            final_regs=final_regs,
            trace_length=len(trace),
            suggestion={
                'type': 'stochastic_perturbation',
                'reason': 'no specific divergence pattern detected',
            },
            expected=expected,
            inputs=inputs,
        )

    @staticmethod
    def _extract_pc_from_error(error: str) -> Optional[int]:
        """Try to extract a PC address from an error string."""
        import re
        match = re.search(r'PC\s+0x([0-9a-fA-F]+)', error)
        if match:
            return int(match.group(1), 16)
        match = re.search(r'at\s+PC\s+0x([0-9a-fA-F]+)', error)
        if match:
            return int(match.group(1), 16)
        return None


# ── Helper Functions for RV32I Instruction Construction ────────────────

def make_rv32i(name: str, rd: int = 0, rs1: int = 0,
               rs2: int = 0, imm: int = 0) -> int:
    """Safely create an RV32I instruction encoding, validating it.

    Args:
        name: Instruction mnemonic (e.g. 'ADD', 'ADDI', 'LW')
        rd:   Destination register (0-31)
        rs1:  Source register 1 (0-31)
        rs2:  Source register 2 (0-31)
        imm:  Immediate value

    Returns:
        32-bit encoded instruction.

    Raises:
        ValueError: If the encoding is not a valid RV32I instruction.
    """
    return rv32i_encode(name=name, rd=rd, rs1=rs1, rs2=rs2,
                        imm=imm, validate=True)


def make_random_rv32i(rng: random.Random) -> int:
    """Generate a random valid RV32I instruction.

    Uses all valid instruction names and randomly fills in register/immediate
    fields.  Returns a validated 32-bit encoding.  Falls back to NOP on
    validation failure.
    """
    name = rng.choice(ALL_INSTRUCTION_NAMES)
    rd = rng.randint(0, 31)
    rs1 = rng.randint(0, 31)
    rs2 = rng.randint(0, 31)

    # Plausible immediate based on instruction type
    if name in ('LUI', 'AUIPC'):
        imm = rng.randint(0, 0xFFFFF)           # 20-bit unsigned
    elif name == 'JAL':
        imm = rng.randint(-0x100000, 0xFFFFF)   # 21-bit signed
    elif name in ('BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU'):
        imm = rng.randint(-4096, 4095) & ~1     # 13-bit signed, even
    else:
        imm = rng.randint(-2048, 2047)          # 12-bit signed

    try:
        return rv32i_encode(name=name, rd=rd, rs1=rs1, rs2=rs2,
                            imm=imm, validate=True)
    except (ValueError, AssertionError):
        return NOP


def find_ebreak_index(program: List[int]) -> Optional[int]:
    """Find the index of the last EBREAK instruction in a program."""
    for i in range(len(program) - 1, -1, -1):
        try:
            decoded = rv32i_decode(program[i], validate=False)
            if decoded.get('name') == 'EBREAK':
                return i
        except Exception:
            continue
    return None


def has_ebreak(program: List[int]) -> bool:
    """Check whether the program contains at least one EBREAK instruction."""
    return find_ebreak_index(program) is not None


# ── DirectedMutator ────────────────────────────────────────────────────

class DirectedMutator:
    """
    Rule-based RV32I mutation proposer.

    Proposes targeted changes based on trace analysis:
      - Task-aware instruction injection (LUI/ADDI for reg init,
        ADD/SUB for computation, load/store for memory access)
      - Single-op funct3/funct7 swapping (ADD↔SUB, BEQ↔BNE, SRL↔SRA)
      - Insert / delete N instructions
      - Reorder contiguous blocks (3-10 instructions)
      - All mutations produce valid RV32I encodings
    """

    # ── Structural helpers ─────────────────────────────────

    @staticmethod
    def find_ebreak_index(program: List[int]) -> Optional[int]:
        """Find index of the last EBREAK in the program (delegates to module-level)."""
        return find_ebreak_index(program)

    @staticmethod
    def insert_before_ebreak(program: List[int], new_instr: int) -> List[int]:
        """Insert an instruction just before the last EBREAK."""
        prog = list(program)
        ebreak_idx = find_ebreak_index(prog)
        if ebreak_idx is not None:
            prog.insert(ebreak_idx, new_instr)
        else:
            prog.append(new_instr)
            prog.append(EBREAK)
        return prog[:MAX_PROGRAM_LEN]

    @staticmethod
    def insert_seq_before_ebreak(program: List[int],
                                  instrs: List[int]) -> List[int]:
        """Insert a sequence of instructions before the last EBREAK."""
        prog = list(program)
        ebreak_idx = find_ebreak_index(prog)
        if ebreak_idx is not None:
            for j, inst in enumerate(instrs):
                prog.insert(ebreak_idx + j, inst)
        else:
            prog.extend(instrs)
            prog.append(EBREAK)
        return prog[:MAX_PROGRAM_LEN]

    @staticmethod
    def insert_after_ebreak(program: List[int], new_instr: int) -> List[int]:
        """Insert an instruction after the last EBREAK (dead code territory)."""
        prog = list(program)
        ebreak_idx = find_ebreak_index(prog)
        if ebreak_idx is not None:
            prog.insert(ebreak_idx + 1, new_instr)
            # Keep a terminator at the very end
            if ebreak_idx + 2 >= len(prog) or prog[ebreak_idx + 2] != EBREAK:
                prog.insert(ebreak_idx + 2, EBREAK)
        else:
            prog.append(new_instr)
            prog.append(EBREAK)
        return prog[:MAX_PROGRAM_LEN]

    @staticmethod
    def ensure_ebreak(program: List[int]) -> List[int]:
        """Ensure the program has an EBREAK at the end."""
        prog = list(program)
        if not has_ebreak(prog):
            prog.append(EBREAK)
        else:
            ebreak_idx = find_ebreak_index(prog)
            # Move the last EBREAK to the very end if it isn't already
            if ebreak_idx is not None and ebreak_idx != len(prog) - 1:
                instr = prog.pop(ebreak_idx)
                prog.append(instr)
        return prog[:MAX_PROGRAM_LEN]

    # ── Main mutation entry point ──────────────────────────

    @staticmethod
    def mutate(
        program: List[int],
        trace: List[Dict[str, Any]],
        analysis: TraceAnalysis,
        rng: random.Random,
        task_level: Optional[int] = None,
    ) -> Tuple[List[int], str]:
        """Propose a targeted RV32I mutation based on trace analysis.

        Args:
            program:    Original program (list of 32-bit instruction words).
            trace:      Execution trace from safety-kernel.
            analysis:   TraceAnalysis from TraceAnalyzer.
            rng:        Random number generator.
            task_level: Optional curriculum task level for task-aware mutations.

        Returns:
            Tuple of (mutated_program, description_string).
        """
        # ── Empty program → seed ────────────────────────────
        if not program:
            seed = make_rv32i('ADDI', rd=10, rs1=0, imm=42)
            return [seed, EBREAK], 'regenerate_empty'

        prog = list(program)
        divergence_type = analysis.divergence_type
        suggestion = analysis.suggestion

        # ── Try task-aware mutation first when task_level is set ─
        if task_level is not None:
            return DirectedMutator._task_aware_mutation(prog, rng, task_level)

        # ── Illegal instruction ─────────────────────────────
        if divergence_type == 'illegal_instruction':
            return DirectedMutator._handle_illegal_instruction(
                prog, analysis, rng)

        # ── PC out of bounds / no terminator ────────────────
        if divergence_type in ('pc_out_of_bounds', 'no_terminator'):
            # Find the instruction that caused the crash (usually JAL/JALR or branch)
            div_pc = analysis.divergence_pc
            if div_pc is not None and div_pc >= 4:
                crash_idx = (div_pc // 4) - 1  # instruction before the out-of-bounds PC
                if 0 <= crash_idx < len(prog) and prog[crash_idx] != EBREAK:
                    # Replace it with NOP
                    prog[crash_idx] = NOP
                    return DirectedMutator.ensure_ebreak(prog), f'fix_crash_at_{crash_idx}'
            return DirectedMutator.ensure_ebreak(prog), 'add_ebreak'

        # ── Ran out of budget ───────────────────────────────
        if divergence_type == 'ran_out_of_budget':
            return DirectedMutator._handle_budget_exhausted(
                prog, analysis, rng)

        # ── Wrong register values ───────────────────────────
        if divergence_type == 'wrong_reg_value':
            return DirectedMutator._handle_wrong_reg(
                prog, analysis, rng)

        # ── Halted early ────────────────────────────────────
        if divergence_type == 'halted_early':
            return DirectedMutator._handle_halted_early(
                prog, analysis, rng)

        # ── Task-aware or fallback mutation ─────────────────
        if task_level is not None:
            return DirectedMutator._task_aware_mutation(prog, rng, task_level)

        return DirectedMutator._fallback_mutation(prog, rng)

    # ── Specific handlers ──────────────────────────────────

    @staticmethod
    def _handle_illegal_instruction(
        prog: List[int], analysis: TraceAnalysis, rng: random.Random
    ) -> Tuple[List[int], str]:
        """Replace an illegal instruction with a safe valid one."""
        div_pc = analysis.divergence_pc
        if div_pc is not None and len(prog) > 0:
            inst_idx = div_pc // 4
            if 0 <= inst_idx < len(prog):
                prog[inst_idx] = NOP
                return prog, f'replace_illegal_at_{inst_idx}'
        # Fallback: replace last non-EBREAK instruction
        for i in range(len(prog) - 1, -1, -1):
            if prog[i] != EBREAK:
                prog[i] = NOP
                return prog, f'replace_illegal_fallback_{i}'
        return [NOP, EBREAK], 'replace_illegal_new_program'

    @staticmethod
    def _handle_budget_exhausted(
        prog: List[int], analysis: TraceAnalysis, rng: random.Random
    ) -> Tuple[List[int], str]:
        """Handle a program that ran out of instruction budget."""
        if not has_ebreak(prog):
            return DirectedMutator.ensure_ebreak(prog), 'add_ebreak_budget'
        # Simplify: delete a random non-EBREAK instruction
        if len(prog) > 3:
            candidates = [i for i in range(len(prog)) if prog[i] != EBREAK]
            if candidates:
                idx = rng.choice(candidates)
                decoded = rv32i_decode(prog[idx], validate=False)
                deleted = decoded.get('name', '?')
                prog.pop(idx)
                return prog, f'delete_{deleted}_at_{idx}_budget'
        return DirectedMutator.insert_before_ebreak(
            prog, NOP), 'insert_nop_budget'

    @staticmethod
    def _handle_wrong_reg(
        prog: List[int], analysis: TraceAnalysis, rng: random.Random
    ) -> Tuple[List[int], str]:
        """Fix wrong register values by injecting LUI/ADDI or ALU sequences."""
        suggestion = analysis.suggestion
        reg = suggestion.get('reg', 10)
        expected = suggestion.get('expected', 0)

        # Normalise to signed 32-bit
        expected = expected & 0xFFFFFFFF
        if expected >= 0x80000000:
            expected -= 0x100000000

        if expected >= 0x800 or expected < -0x800:
            # Needs LUI + ADDI
            upper = (expected >> 12) & 0xFFFFF
            lower = expected & 0xFFF
            if lower >= 0x800:
                lower -= 0x1000
                upper = (upper + 1) & 0xFFFFF
            lui = make_rv32i('LUI', rd=reg, imm=upper)
            addi = make_rv32i('ADDI', rd=reg, rs1=reg, imm=lower)
            return (DirectedMutator.insert_seq_before_ebreak(prog, [lui, addi]),
                    f'inject_lui_addi_r{reg}_{expected}')
        # Single ADDI from x0 suffices
        addi = make_rv32i('ADDI', rd=reg, rs1=0, imm=expected & 0xFFF)
        return (DirectedMutator.insert_before_ebreak(prog, addi),
                f'inject_addi_r{reg}_{expected}')

    @staticmethod
    def _handle_halted_early(
        prog: List[int], analysis: TraceAnalysis, rng: random.Random
    ) -> Tuple[List[int], str]:
        """Add more instructions if the program halted too early."""
        new_instr = make_random_rv32i(rng)
        idx = rng.randint(0, len(prog) - 1) if len(prog) > 1 else 0
        prog.insert(idx, new_instr)
        return prog[:MAX_PROGRAM_LEN], f'insert_extra_instr_at_{idx}'

    # ── Task-aware mutation ────────────────────────────────

    @staticmethod
    def _task_aware_mutation(
        prog: List[int], rng: random.Random, task_level: int
    ) -> Tuple[List[int], str]:
        """Apply a curriculum-task-aware mutation.

        Uses the progressive register convention: output register = 3 + task_level.
        Injects instruction patterns matching the task domain.

        Task levels:
          L0: x3 = x1                    (ADDI x3, x1, 0)
          L1: x4 = x1 + x1               (ADD x4, x1, x1)
          L2: x5 = x2                    (ADDI x5, x2, 0)
          L3: x6 = x2 + x2               (ADD x6, x2, x2)
          L4: x7 = x4 + x6               (ADD x7, x4, x6)
          L5: x8 = (x4 > x6)            (SLT x8, x6, x4)
          L6: x9 = x1 + x2               (ADD x9, x1, x2)
          L7: x10 = mem[x1]              (LW x10, 0(x1))
          L8: x11 = mem[x1] + mem[x2]    (LW + ADD)
          L9: x12 = sum(mem[x1..])       (loop)
          L10: x13 = max(mem[x1..])      (loop + compare)
          L11: x14 = fibonacci(x1)       (loop)
        """
        out_reg = 3 + task_level
        roll = rng.random()

        if task_level == 0:
            # ADDI x3, x1, 0
            instr = make_rv32i('ADDI', rd=3, rs1=1, imm=0)
            return (DirectedMutator.insert_before_ebreak(prog, instr),
                    f'l0_addi_x3_x1_0')

        if task_level == 1:
            # ADD x4, x1, x1
            instr = make_rv32i('ADD', rd=4, rs1=1, rs2=1)
            return (DirectedMutator.insert_before_ebreak(prog, instr),
                    f'l1_add_x4_x1_x1')

        if task_level == 2:
            # ADDI x5, x2, 0
            instr = make_rv32i('ADDI', rd=5, rs1=2, imm=0)
            return (DirectedMutator.insert_before_ebreak(prog, instr),
                    f'l2_addi_x5_x2_0')

        if task_level == 3:
            # ADD x6, x2, x2
            instr = make_rv32i('ADD', rd=6, rs1=2, rs2=2)
            return (DirectedMutator.insert_before_ebreak(prog, instr),
                    f'l3_add_x6_x2_x2')

        if task_level == 4:
            # ADD x7, x4, x6
            instr = make_rv32i('ADD', rd=7, rs1=4, rs2=6)
            return (DirectedMutator.insert_before_ebreak(prog, instr),
                    f'l4_add_x7_x4_x6')

        if task_level == 5:
            # SLT x8, x6, x4  (x8 = 1 if x6 < x4, i.e. x4 > x6)
            instr = make_rv32i('SLT', rd=8, rs1=6, rs2=4)
            return (DirectedMutator.insert_before_ebreak(prog, instr),
                    f'l5_slt_x8_x6_x4')

        if task_level == 6:
            # ADD x9, x1, x2
            instr = make_rv32i('ADD', rd=9, rs1=1, rs2=2)
            return (DirectedMutator.insert_before_ebreak(prog, instr),
                    f'l6_add_x9_x1_x2')

        if task_level == 7:
            # LW x10, 0(x1)
            instr = make_rv32i('LW', rd=10, rs1=1, imm=0)
            return (DirectedMutator.insert_before_ebreak(prog, instr),
                    f'l7_lw_x10_0_x1')

        if task_level == 8:
            # L8: x11 = mem[x1] + mem[x2]
            # Need: LW x15, 0(x1); LW x16, 0(x2); ADD x11, x15, x16
            tmp1 = make_rv32i('LW', rd=15, rs1=1, imm=0)
            tmp2 = make_rv32i('LW', rd=16, rs1=2, imm=0)
            add = make_rv32i('ADD', rd=11, rs1=15, rs2=16)
            return (DirectedMutator.insert_seq_before_ebreak(prog, [tmp1, tmp2, add]),
                    f'l8_dual_load_add_x11')

        if task_level == 9:
            # L9: x12 = sum(mem[x1..x1+x2-1])
            # ADDI x15, x0, 0; ADDI x16, x0, 0; loop: ADD x17, x1, x16;
            # LW x18, 0(x17); ADD x15, x15, x18; ADDI x16, x16, 1;
            # BLT x16, x2, -20; ADDI x12, x15, 0
            init_s = make_rv32i('ADDI', rd=15, rs1=0, imm=0)
            init_i = make_rv32i('ADDI', rd=16, rs1=0, imm=0)
            addr_calc = make_rv32i('ADD', rd=17, rs1=1, rs2=16)
            load = make_rv32i('LW', rd=18, rs1=17, imm=0)
            sum_add = make_rv32i('ADD', rd=15, rs1=15, rs2=18)
            inc_i = make_rv32i('ADDI', rd=16, rs1=16, imm=1)
            branch = make_rv32i('BLT', rs1=16, rs2=2, imm=-20)
            store = make_rv32i('ADDI', rd=12, rs1=15, imm=0)
            return (DirectedMutator.insert_seq_before_ebreak(
                prog, [init_s, init_i, addr_calc, load, sum_add, inc_i, branch, store]),
                    f'l9_sum_loop_x12')

        if task_level == 10:
            # L10: x13 = max(mem[x1..x1+x2-1])
            # Similar to L9 but uses SLT + branch for max comparison
            init_max = make_rv32i('ADDI', rd=15, rs1=0, imm=0)  # max = 0
            init_i = make_rv32i('ADDI', rd=16, rs1=0, imm=0)    # i = 0
            addr_calc = make_rv32i('ADD', rd=17, rs1=1, rs2=16)
            load = make_rv32i('LW', rd=18, rs1=17, imm=0)       # val = mem[addr]
            # Compare val > max
            cmp = make_rv32i('SLT', rd=19, rs1=15, rs2=18)      # x19 = 1 if max < val
            beq_skip = make_rv32i('BEQ', rs1=19, rs2=0, imm=8)  # skip update if max >= val
            update = make_rv32i('ADDI', rd=15, rs1=18, imm=0)   # max = val
            inc_i = make_rv32i('ADDI', rd=16, rs1=16, imm=1)
            branch = make_rv32i('BLT', rs1=16, rs2=2, imm=-28)
            store = make_rv32i('ADDI', rd=13, rs1=15, imm=0)
            return (DirectedMutator.insert_seq_before_ebreak(
                prog, [init_max, init_i, addr_calc, load, cmp, beq_skip, update, inc_i, branch, store]),
                    f'l10_max_loop_x13')

        if task_level == 11:
            # L11: x14 = fibonacci(x1)
            # ADDI x15, x0, 0  (a = 0)
            # ADDI x16, x0, 1  (b = 1)
            # BEQ x1, x0, done (if n == 0, skip)
            # loop: ADD x17, x15, x16 (next = a + b)
            # ADDI x15, x16, 0 (a = b)
            # ADDI x16, x17, 0 (b = next)
            # ADDI x1, x1, -1   (n--)
            # BNE x1, x0, loop (if n != 0, continue)
            # done: ADDI x14, x15, 0
            init_a = make_rv32i('ADDI', rd=15, rs1=0, imm=0)
            init_b = make_rv32i('ADDI', rd=16, rs1=0, imm=1)
            check_n = make_rv32i('BEQ', rs1=1, rs2=0, imm=20)
            next_fib = make_rv32i('ADD', rd=17, rs1=15, rs2=16)
            shift_a = make_rv32i('ADDI', rd=15, rs1=16, imm=0)
            shift_b = make_rv32i('ADDI', rd=16, rs1=17, imm=0)
            dec_n = make_rv32i('ADDI', rd=1, rs1=1, imm=-1)
            loop_br = make_rv32i('BNE', rs1=1, rs2=0, imm=-20)
            store_fib = make_rv32i('ADDI', rd=14, rs1=15, imm=0)
            return (DirectedMutator.insert_seq_before_ebreak(
                prog, [init_a, init_b, check_n, next_fib, shift_a, shift_b, dec_n, loop_br, store_fib]),
                    f'l11_fibonacci_loop_x14')

        # Fallback to original generic task-aware logic
        if roll < 0.30:
            reg = rng.randint(5, 15)
            value = rng.randint(0, 0xFFFFFFFF) & 0xFFFFFFFF
            if value >= 0x80000000:
                value -= 0x100000000
            upper = (value >> 12) & 0xFFFFF
            lower = value & 0xFFF
            if lower >= 0x800:
                lower -= 0x1000
                upper = (upper + 1) & 0xFFFFF
            lui = make_rv32i('LUI', rd=reg, imm=upper)
            addi = make_rv32i('ADDI', rd=reg, rs1=reg, imm=lower)
            return (DirectedMutator.insert_seq_before_ebreak(prog, [lui, addi]),
                    f'generic_init_r{reg}_{value}')

        elif roll < 0.55:
            rd = rng.randint(5, 15)
            rs1 = rng.randint(0, 15)
            rs2 = rng.randint(0, 15)
            name = rng.choice(['ADD', 'SUB', 'XOR', 'OR', 'AND'])
            instr = make_rv32i(name, rd=rd, rs1=rs1, rs2=rs2)
            return (DirectedMutator.insert_before_ebreak(prog, instr),
                    f'generic_alu_{name}_r{rd}_r{rs1}_r{rs2}')

        elif roll < 0.75:
            mem_reg = rng.randint(0, 15)
            data_reg = rng.randint(5, 15)
            im_val = rng.randint(0, 255) & 0xFFF
            sw = make_rv32i('SW', rs1=mem_reg, rs2=data_reg, imm=im_val)
            lw = make_rv32i('LW', rd=data_reg, rs1=mem_reg, imm=im_val)
            return (DirectedMutator.insert_seq_before_ebreak(prog, [sw, lw]),
                    f'generic_mem_sw_lw_r{data_reg}')

        else:
            rs1 = rng.randint(0, 15)
            rs2 = rng.randint(0, 15)
            name = rng.choice(['BEQ', 'BNE', 'BLT', 'BGE'])
            instr = make_rv32i(name, rs1=rs1, rs2=rs2, imm=4)
            return (DirectedMutator.insert_before_ebreak(prog, instr),
                    f'generic_branch_{name}')

    # ── Fallback / generic mutations ───────────────────────

    @staticmethod
    def _fallback_mutation(
        prog: List[int], rng: random.Random
    ) -> Tuple[List[int], str]:
        """Apply a generic mutation when no specific analysis is available."""
        if len(prog) <= 1:
            return [make_random_rv32i(rng), EBREAK], 'fallback_new'

        mutation_type = rng.choice([
            'swap_op', 'insert_nop', 'delete_instr',
            'reorder_block', 'swap_reg',
        ])

        if mutation_type == 'swap_op' and len(prog) > 0:
            return DirectedMutator._swap_operation(prog, rng)

        if mutation_type == 'insert_nop':
            idx = rng.randint(0, len(prog) - 1)
            prog.insert(idx, NOP)
            return prog[:MAX_PROGRAM_LEN], f'insert_nop_at_{idx}'

        if mutation_type == 'delete_instr' and len(prog) > 2:
            candidates = [i for i in range(len(prog)) if prog[i] != EBREAK]
            if candidates:
                idx = rng.choice(candidates)
                decoded = rv32i_decode(prog[idx], validate=False)
                deleted = decoded.get('name', '?')
                prog.pop(idx)
                return prog, f'delete_{deleted}_at_{idx}'
            return prog, 'no_change_no_deletable'

        if mutation_type == 'reorder_block' and len(prog) >= 4:
            return DirectedMutator._reorder_block(prog, rng)

        if mutation_type == 'swap_reg' and len(prog) > 0:
            return DirectedMutator._swap_register(prog, rng)

        return prog, 'no_change_fallback'

    @staticmethod
    def _swap_operation(
        prog: List[int], rng: random.Random
    ) -> Tuple[List[int], str]:
        """Swap funct3/funct7 of an instruction (ADD↔SUB, BEQ↔BNE, etc.)."""
        candidates: List[Tuple[int, str, str]] = []
        for i, word in enumerate(prog):
            try:
                decoded = rv32i_decode(word, validate=False)
                name = decoded.get('name', '')
                if name in BRANCH_SWAP_PAIRS:
                    candidates.append((i, name, 'branch'))
                elif name in ALU_SWAP_PAIRS:
                    candidates.append((i, name, 'alu'))
            except Exception:
                continue

        if not candidates:
            idx = rng.randint(0, len(prog) - 1)
            prog.insert(idx, NOP)
            return prog[:MAX_PROGRAM_LEN], 'nop_swap_fallback'

        idx, name, swap_type = rng.choice(candidates)
        decoded = rv32i_decode(prog[idx], validate=False)

        if swap_type == 'branch':
            new_name = BRANCH_SWAP_PAIRS[name]
            new_instr = rv32i_encode(
                name=new_name,
                rd=0,  # branches have no rd
                rs1=decoded['rs1'],
                rs2=decoded['rs2'],
                imm=decoded['imm'],
                validate=True,
            )
        else:  # alu
            new_name = ALU_SWAP_PAIRS[name]
            new_instr = rv32i_encode(
                name=new_name,
                rd=decoded['rd'],
                rs1=decoded['rs1'],
                rs2=decoded['rs2'],
                validate=True,
            )
        prog[idx] = new_instr
        return prog, f'swap_{name}_to_{new_name}'

    @staticmethod
    def _swap_register(
        prog: List[int], rng: random.Random
    ) -> Tuple[List[int], str]:
        """Swap register indices within an instruction."""
        idx = rng.randint(0, len(prog) - 1)
        try:
            decoded = rv32i_decode(prog[idx], validate=False)
            name = decoded.get('name', '')
            fmt = decoded.get('fmt', '')

            if fmt == 'R' and decoded['rs1'] != decoded['rs2']:
                new_instr = rv32i_encode(
                    name=name,
                    rd=decoded['rd'],
                    rs1=decoded['rs2'],
                    rs2=decoded['rs1'],
                    validate=True,
                )
                prog[idx] = new_instr
                return prog, f'swap_rs1_rs2_{name}_at_{idx}'

            if fmt == 'I' and decoded.get('rs1', 0) > 0:
                new_rs1 = rng.randint(0, 31)
                new_instr = rv32i_encode(
                    name=name,
                    rd=decoded['rd'],
                    rs1=new_rs1,
                    imm=decoded['imm'],
                    validate=True,
                )
                prog[idx] = new_instr
                return prog, f'change_rs1_{name}_at_{idx}_to_{new_rs1}'
        except Exception:
            pass
        return prog, 'no_change_reg_swap'

    @staticmethod
    def _reorder_block(
        prog: List[int], rng: random.Random
    ) -> Tuple[List[int], str]:
        """Reorder a contiguous block of 3-10 instructions."""
        block_size = rng.randint(3, min(10, len(prog) - 1))
        max_start = len(prog) - block_size
        if max_start <= 0:
            return prog, 'no_change_too_small'

        start = rng.randint(0, max_start - 1)
        # Shrink block to avoid including EBREAK
        actual_end = start + block_size
        for i in range(start, min(actual_end, len(prog))):
            if prog[i] == EBREAK:
                actual_end = i
                break
        if actual_end - start < 3:
            return prog, 'no_change_block_too_small'

        block = prog[start:actual_end]
        # Split-and-swap reorder
        split_point = rng.randint(1, len(block) - 1)
        reordered = block[split_point:] + block[:split_point]
        prog[start:actual_end] = reordered
        return prog, f'reorder_{start}_{actual_end}'

    # ── Stochastic perturbation ────────────────────────────

    @staticmethod
    def stochastic_perturbation(
        program: List[int], rng: random.Random
    ) -> Tuple[List[int], str]:
        """Single-instruction stochastic perturbation (~10% of mutations).

        Picks one of: replace instruction, delete instruction, or insert
        instruction.  All replacements produce valid RV32I encodings.
        """
        if not program:
            return [make_random_rv32i(rng), EBREAK], 'stochastic_regenerate'

        prog = list(program)
        if not has_ebreak(prog):
            prog.append(EBREAK)

        mutation_type = rng.choice(['replace', 'delete', 'insert'])

        if mutation_type == 'replace':
            idx = rng.randint(0, len(prog) - 1)
            # Avoid replacing EBREAK unless it's the sole instruction
            if prog[idx] == EBREAK and len(prog) > 1:
                idx = (idx + 1) % len(prog)
                while idx < len(prog) and prog[idx] == EBREAK:
                    idx = (idx + 1) % len(prog)
            new_instr = make_random_rv32i(rng)
            prog[idx] = new_instr
            return prog, f'stochastic_replace_{idx}'

        if mutation_type == 'delete' and len(prog) > 2:
            candidates = [i for i in range(len(prog)) if prog[i] != EBREAK]
            if candidates:
                idx = rng.choice(candidates)
                prog.pop(idx)
                return prog, f'stochastic_delete_{idx}'
            return prog, 'stochastic_no_delete'

        # insert
        idx = rng.randint(0, len(prog))
        new_instr = make_random_rv32i(rng)
        prog.insert(idx, new_instr)
        return prog[:MAX_PROGRAM_LEN], f'stochastic_insert_{idx}'


# ── Convenience / High-Level API ──────────────────────────────────────

def propose_mutation(
    program: List[int],
    trace: List[Dict[str, Any]],
    expected: Dict[str, Any],
    inputs: Dict[str, Any],
    *,
    error: Optional[str] = None,
    budget: Optional[int] = None,
    rng: Optional[random.Random] = None,
    stochastic_prob: float = 0.10,
    task_level: Optional[int] = None,
    analysis: Any = None,
    strategy: Any = None,
    strategy_archive: Any = None,
) -> Tuple[List[int], str, TraceAnalysis]:
    """High-level entry point: analyse trace and propose a mutation.

    Supports strategy-based routing:
      - RANDOM: stochastic perturbation only (no trace analysis)
      - TASK_AWARE: task-level injection only
      - DIRECTED: trace-analysis-driven mutation
      - PERTURBATION_CAPPED: capped random instruction replacement
      - HYBRID: default blend (10% stochastic + 90% directed)

    Args:
        program:        Original program (list of 32-bit instruction words).
        trace:          Execution trace from safety-kernel.
        expected:       Expected output dict.
        inputs:         Input state dict.
        error:          Error string from safety-kernel (if any).
        budget:         Instruction budget used.
        rng:            Random number generator (created fresh if None).
        stochastic_prob: Probability of stochastic perturbation (default 0.10).
        task_level:     Optional curriculum task level.
        strategy:       MutationStrategy enum value (controls routing).
        strategy_archive: StrategyArchive for historical solutions.

    Returns:
        Tuple of (mutated_program, description_string, trace_analysis).
    """
    if rng is None:
        rng = random.Random()

    # ── Strategy-based routing ───────────────────────────────────────
    strat_val = getattr(strategy, "value", str(strategy)) if strategy is not None else "HYBRID"

    if strat_val == "RANDOM":
        # Pure stochastic: no trace analysis, maximise exploration
        mutated_prog, desc = DirectedMutator.stochastic_perturbation(
            program, rng)
        analysis = TraceAnalysis(
            divergence_point=None, divergence_type="random",
            final_regs=[], trace_length=0, suggestion={})
        return mutated_prog, f"random_{desc}", analysis

    if strat_val == "TASK_AWARE":
        # Direct task-level injection
        if task_level is not None:
            mutated_prog, desc = DirectedMutator._task_aware_mutation(
                program, rng, task_level)
        else:
            mutated_prog, desc = DirectedMutator._fallback_mutation(
                program, rng)
        analysis = TraceAnalysis(
            divergence_point=None, divergence_type="task_aware",
            final_regs=[], trace_length=0, suggestion={})
        return mutated_prog, f"task_aware_{desc}", analysis

    # Branches below need trace analysis
    analysis = TraceAnalyzer.analyze(
        trace=trace,
        expected=expected,
        inputs=inputs,
        program=program,
        error=error,
        budget=budget,
    )

    if strat_val == "DIRECTED":
        # Trace-analysis-driven (no stochastic component)
        mutated_prog, desc = DirectedMutator.mutate(
            program, trace, analysis, rng, task_level=task_level)
        return mutated_prog, f"directed_{desc}", analysis

    if strat_val == "PERTURBATION_CAPPED":
        # Capped perturbation: replace up to 3 random instructions with
        # valid RV32I encodings. The archive does not store programs, so
        # true parent blending is not possible; this provides bounded
        # exploration distinct from full stochastic perturbation.
        champion_prog = program
        if strategy_archive is not None and strategy_archive.total_records() > 10:
            if task_level is not None:
                _, hist_desc = DirectedMutator._task_aware_mutation(
                    list(champion_prog), rng, task_level)
                desc = f"crossover_archive_{hist_desc}"
            else:
                desc = "crossover_directed"
        else:
            desc = "crossover_directed"

        if len(champion_prog) >= 2:
            mutated_prog = list(champion_prog)
            n_swaps = min(3, max(1, len(champion_prog) // 4))
            for _ in range(n_swaps):
                idx = rng.randint(0, len(mutated_prog) - 1)
                if mutated_prog[idx] != EBREAK:
                    mutated_prog[idx] = make_random_rv32i(rng)
            if mutated_prog[-1] != EBREAK:
                mutated_prog.append(EBREAK)
            return mutated_prog[:MAX_PROGRAM_LEN], desc, analysis
        else:
            mutated_prog, desc2 = DirectedMutator._fallback_mutation(
                program, rng)
            return mutated_prog, f"crossover_fallback_{desc2}", analysis

    # ── Default: HYBRID (90% directed, 10% stochastic) ──────────────
    if rng.random() < stochastic_prob:
        mutated_prog, desc = DirectedMutator.stochastic_perturbation(
            program, rng)
    else:
        mutated_prog, desc = DirectedMutator.mutate(
            program, trace, analysis, rng, task_level=task_level)

    return mutated_prog, desc, analysis


def generate_seed_program(length: int = 5,
                          rng: Optional[random.Random] = None) -> List[int]:
    """Generate a random seed program of valid RV32I instructions."""
    if rng is None:
        rng = random.Random()
    prog = [make_random_rv32i(rng) for _ in range(max(length - 1, 1))]
    prog.append(EBREAK)
    return prog


def validate_mutation(mutated: List[int]) -> Tuple[bool, Optional[str]]:
    """Validate that every instruction in the program is a legal RV32I encoding.

    Returns:
        (is_valid, error_message_or_None)
    """
    if not mutated:
        return False, 'Empty program'
    for i, word in enumerate(mutated):
        try:
            rv32i_decode(word, validate=True)
        except ValueError as e:
            return False, f'Instruction {i} (0x{word:08x}): {e}'
    return True, None


# ── Self-Test / Demo ──────────────────────────────────────────────────

def _encode_inst(inst: dict) -> Optional[int]:
    """Encode an instruction dict to a 32-bit RV32I word.

    Converts the reference_solution instruction format:
        {'opcode': 'addi', 'rd': 1, 'rs1': 0, 'imm': 42}
    into a 32-bit machine code word using the rv32i_mask encoder.

    Args:
        inst: Instruction dict with keys: opcode, rd, rs1, rs2, imm.

    Returns:
        32-bit encoded instruction, or None if encoding fails.
    """
    opcode = inst.get("name", inst.get("opcode", "")).upper()
    rd = inst.get("rd", 0)
    rs1 = inst.get("rs1", 0)
    rs2 = inst.get("rs2", 0)
    imm = inst.get("imm", 0)
    try:
        return rv32i_encode(name=opcode, rd=rd, rs1=rs1, rs2=rs2, imm=imm)
    except Exception:
        return None


def _demo() -> None:
    """Run a quick self-test demonstrating the module."""
    print('=' * 72)
    print('RV32I Mutation Proposer — Self-Test')
    print('=' * 72)

    rng = random.Random(42)

    # ── Test 1: Empty program ──────────────────────────────
    print('\n--- Test 1: Empty program ---')
    empty_prog: List[int] = []
    analysis = TraceAnalyzer.analyze(
        trace=[], expected={}, inputs={}, program=empty_prog, error=None)
    print(f'  Divergence type: {analysis.divergence_type}')
    mutated, desc = DirectedMutator.mutate(empty_prog, [], analysis, rng)
    valid, err = validate_mutation(mutated)
    print(f'  Mutation: {desc}')
    print(f'  Valid encoded: {valid}, size={len(mutated)}')

    # ── Test 2: Single-instruction program (no EBREAK) ─────
    print('\n--- Test 2: Single-instruction (no EBREAK) ---')
    single = [make_rv32i('ADDI', rd=10, rs1=0, imm=42)]
    trace = [{'regs': [0] * 32, 'pc': 4, 'cycle': 1}]
    expected = {'registers': {10: 42}}
    inputs = {}
    analysis = TraceAnalyzer.analyze(
        trace, expected, inputs, program=single)
    print(f'  Divergence type: {analysis.divergence_type}')
    print(f'  Suggestion: {analysis.suggestion}')
    mutated, desc = DirectedMutator.mutate(single, trace, analysis, rng)
    valid, err = validate_mutation(mutated)
    print(f'  Mutation: {desc}')
    print(f'  Valid encoded: {valid}, size={len(mutated)}')
    if valid:
        d = rv32i_decode(mutated[-1], validate=False)
        print(f'  Last instr: {d["name"]}')

    # ── Test 3: Swap ADD → SUB ─────────────────────────────
    print('\n--- Test 3: Swap ADD → SUB ---')
    prog_add = [
        make_rv32i('ADD', rd=10, rs1=1, rs2=2),
        EBREAK,
    ]
    analysis = TraceAnalysis(
        divergence_point=0,
        divergence_type='wrong_reg_value',
        final_regs=[0] * 32,
        trace_length=1,
        suggestion={'type': 'swap_op'},
        expected={'registers': {10: 42}},
        inputs={},
    )
    mutated, desc = DirectedMutator.mutate(prog_add, [], analysis, rng)
    valid, err = validate_mutation(mutated)
    print(f'  Mutation: {desc}')
    if valid:
        d = rv32i_decode(mutated[0], validate=False)
        print(f'  Instruction 0: {d["name"]}')
    print(f'  Valid encoded: {valid}')

    # ── Test 4: Swap BEQ → BNE ─────────────────────────────
    print('\n--- Test 4: Swap BEQ → BNE ---')
    prog_beq = [
        make_rv32i('ADDI', rd=10, rs1=0, imm=5),
        make_rv32i('BEQ', rs1=10, rs2=11, imm=8),
        EBREAK,
    ]
    analysis = TraceAnalysis(
        divergence_point=1,
        divergence_type='wrong_reg_value',
        final_regs=[0] * 32,
        trace_length=2,
        suggestion={'type': 'swap_op'},
        expected={'registers': {10: 5}},
        inputs={},
    )
    mutated, desc = DirectedMutator.mutate(prog_beq, [], analysis, rng)
    valid, err = validate_mutation(mutated)
    print(f'  Mutation: {desc}')
    if valid:
        for i, w in enumerate(mutated):
            d = rv32i_decode(w, validate=False)
            print(f'    [{i}] {d["name"]:8s} rs1={d["rs1"]}, rs2={d["rs2"]}, imm={d["imm"]}')
    print(f'  Valid encoded: {valid}')

    # ── Test 5: Stochastic perturbation ────────────────────
    print('\n--- Test 5: Stochastic perturbation ---')
    prog_base = [make_rv32i('ADDI', rd=5, rs1=0, imm=10), EBREAK]
    for i in range(5):
        mutated, desc = DirectedMutator.stochastic_perturbation(
            prog_base, rng)
        valid, err = validate_mutation(mutated)
        print(f'  [{i}] {desc:30s} valid={valid}, size={len(mutated)}')
        if valid:
            for w in mutated:
                d = rv32i_decode(w, validate=False)
                print(f'         {d["name"]:8s} rd={d["rd"]}, rs1={d["rs1"]}, '
                      f'rs2={d["rs2"]}, imm={d["imm"]}')

    # ── Test 6: Reorder block ──────────────────────────────
    print('\n--- Test 6: Reorder block ---')
    prog_reorder = [
        make_rv32i('ADDI', rd=5, rs1=0, imm=10),
        make_rv32i('ADDI', rd=6, rs1=0, imm=20),
        make_rv32i('ADD',  rd=7, rs1=5, rs2=6),
        make_rv32i('ADDI', rd=8, rs1=0, imm=30),
        EBREAK,
    ]
    print('  Before:')
    for i, w in enumerate(prog_reorder):
        d = rv32i_decode(w, validate=False)
        print(f'    [{i}] {d["name"]:8s} rd={d["rd"]}, rs1={d["rs1"]}, '
              f'rs2={d["rs2"]}, imm={d["imm"]}')
    mutated, desc = DirectedMutator._reorder_block(prog_reorder, rng)
    valid, err = validate_mutation(mutated)
    print(f'  After  ({desc}):')
    for i, w in enumerate(mutated):
        d = rv32i_decode(w, validate=False)
        print(f'    [{i}] {d["name"]:8s} rd={d["rd"]}, rs1={d["rs1"]}, '
              f'rs2={d["rs2"]}, imm={d["imm"]}')
    print(f'  Valid encoded: {valid}')

    # ── Test 7: Task-aware mutation ────────────────────────
    print('\n--- Test 7: Task-aware mutation (inject register init) ---')
    prog_task = [EBREAK]
    for tl in range(3):
        mutated, desc = DirectedMutator._task_aware_mutation(
            prog_task, rng, tl)
        valid, err = validate_mutation(mutated)
        print(f'  [{tl}] {desc:30s} valid={valid}, size={len(mutated)}')
        if valid:
            for w in mutated:
                d = rv32i_decode(w, validate=False)
                print(f'         {d["name"]:8s} rd={d["rd"]}, rs1={d["rs1"]}, '
                      f'rs2={d["rs2"]}, imm={d["imm"]}')
        prog_task = mutated  # chain

    # ── Test 8: Full propose_mutation pipeline ─────────────
    print('\n--- Test 8: Full propose_mutation pipeline ---')
    prog = [make_rv32i('ADDI', rd=10, rs1=0, imm=99), EBREAK]
    trace = [{'regs': [0] * 32, 'pc': 4, 'cycle': 1}]
    expected = {'registers': {10: 42}}
    inputs = {}
    mutated, desc, analysis = propose_mutation(
        prog, trace, expected, inputs, rng=rng, task_level=0)
    valid, err = validate_mutation(mutated)
    print(f'  Analysis: {analysis.divergence_type}')
    print(f'  Mutation: {desc}')
    print(f'  Valid encoded: {valid}, size={len(mutated)}')
    if valid:
        for w in mutated:
            d = rv32i_decode(w, validate=False)
            print(f'    {d["name"]:8s} rd={d["rd"]}, rs1={d["rs1"]}, '
                  f'rs2={d["rs2"]}, imm={d["imm"]}')

    # ── Test 9: No EBREAK program ──────────────────────────
    print('\n--- Test 9: Program without EBREAK ---')
    prog_no_ebreak = [make_rv32i('ADDI', rd=5, rs1=0, imm=10)]
    trace = [{'regs': [0] * 32, 'pc': 4, 'cycle': 1}]
    analysis = TraceAnalyzer.analyze(
        trace, {}, {}, program=prog_no_ebreak)
    print(f'  Divergence type: {analysis.divergence_type}')
    mutated, desc = DirectedMutator.mutate(
        prog_no_ebreak, trace, analysis, rng)
    valid, err = validate_mutation(mutated)
    print(f'  Mutation: {desc}')
    print(f'  Valid encoded: {valid}, size={len(mutated)}')
    print(f'  Has EBREAK: {has_ebreak(mutated)}')

    # ── Test 10: validate_mutation on deliberately invalid ──
    print('\n--- Test 10: Mutation validation ---')
    prog_good = [make_rv32i('ADDI', rd=5, rs1=0, imm=10), EBREAK]
    ok, err = validate_mutation(prog_good)
    print(f'  Good program: valid={ok}')
    # Try an all-zeros instruction (not a valid RV32I)
    prog_bad = [0x00000000, EBREAK]
    ok, err = validate_mutation(prog_bad)
    print(f'  Bad program (0x00000000): valid={ok}, err={err}')

    print('\n--- Self-test complete ---')


if __name__ == '__main__':
    _demo()
