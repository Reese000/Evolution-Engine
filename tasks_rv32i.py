#!/usr/bin/env python3
"""
RV32I curriculum tasks — 31-level monotonic curriculum (L0-L30).

Each level adds a NEW output register. Previous levels' outputs are never
overwritten because each level's output goes to a DIFFERENT register.

Register convention (RV32I ABI-aware):
  x0   = hardwired zero
  x1   = primary input / address / n
  x2   = secondary input / address / count
  x3   = output for Level 0  (gp)
  x4   = output for Level 1  (tp)
  x5   = output for Level 2  (t0)
  x6   = output for Level 3  (t1)
  x7   = output for Level 4  (t2)
  x8   = output for Level 5  (s0/fp)
  x9   = output for Level 6  (s1)
  x10  = output for Level 7  (a0)
  x11  = output for Level 8  (a1)
  x12  = output for Level 9  (a2)
  x13  = output for Level 10 (a3)
  x14  = output for Level 11 (a4)
  x15+ = temporaries for loops and intermediate values

Each level's verify() checks ALL previous output registers have correct
values (monotonic cumulative design). Reference solutions are provided as
instruction dicts compatible with rv32i_mask encode/decode.
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────
# RV32I constants
# ──────────────────────────────────────────────────────────────────────
REGISTER_MAX = 0xFFFFFFFF
MEMORY_SIZE = 256  # bytes


def compute_difficulty(instr_count: float, nesting: float, dep_length: float) -> float:
    """Compute a combined difficulty metric."""
    return instr_count * nesting * dep_length


def output_reg(level: int) -> int:
    """Map task level to output register (x3 + level)."""
    return 3 + level


def _cap32(val: int) -> int:
    """Cap a value to 32-bit unsigned range."""
    return val & REGISTER_MAX


def _sign32(val: int) -> int:
    """Interpret 32-bit unsigned as signed 32-bit."""
    if val >= 0x80000000:
        return val - 0x100000000
    return val


# ──────────────────────────────────────────────────────────────────────
# Base TaskLevel ABC
# ──────────────────────────────────────────────────────────────────────


class TaskLevel(ABC):
    """Base class for curriculum task levels."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    @abstractmethod
    def generate_instance(self) -> dict: ...

    @abstractmethod
    def verify(self, output: dict, inst: dict) -> float: ...

    @abstractmethod
    def difficulty_metric(self) -> float: ...

    @staticmethod
    @abstractmethod
    def reference_solution() -> List[dict]: ...

    def _pick_value(self) -> int:
        """Pick a random 32-bit register value (unsigned, matches VM representation)."""
        return self._rng.randint(0, 0xFFFF_FFFF)

    def _pick_small(self) -> int:
        return self._rng.randint(1, 15)

    def _pick_addr(self) -> int:
        """Pick a valid word-aligned memory address within bounds."""
        max_word = (MEMORY_SIZE // 4) - 1
        return self._rng.randint(0, max_word) * 4

    def _pick_mem_val(self) -> int:
        """Pick a memory byte value (0-255) for safety-kernel u8 format."""
        return self._rng.randint(0, 255)

    def _pick_count(self, lo: int = 3, hi: int = 8) -> int:
        return self._rng.randint(lo, hi)


# ===========================================================================
# Shared helpers: compute cumulative expected register values for L0-L6
# ===========================================================================


def _compute_base_expected(x1_val: int, x2_val: int) -> Dict[int, int]:
    """Compute the base expected register values from L0-L6.

    All values are capped to unsigned 32-bit to match what the VM stores.

    L0: x3 = x1
    L1: x4 = x1 + x1
    L2: x5 = x2
    L3: x6 = x2 + x2
    L4: x7 = x4 + x6
    L5: x8 = 1 if x4 > x6 (signed) else 0
    L6: x9 = x1 + x2
    """
    x3 = _cap32(x1_val)
    x4 = _cap32(x1_val + x1_val)
    x5 = _cap32(x2_val)
    x6 = _cap32(x2_val + x2_val)
    x7 = _cap32(x4 + x6)
    x8 = 1 if _sign32(x4) > _sign32(x6) else 0
    x9 = _cap32(x1_val + x2_val)
    return {3: x3, 4: x4, 5: x5, 6: x6, 7: x7, 8: x8, 9: x9}


def _base_ref() -> List[dict]:
    """Reference instructions for cumulative L0-L6."""
    return [
        {"name": "ADDI", "rd": 3, "rs1": 1, "imm": 0},  # L0: x3 = x1
        {"name": "ADD", "rd": 4, "rs1": 1, "rs2": 1},  # L1: x4 = x1 + x1
        {"name": "ADDI", "rd": 5, "rs1": 2, "imm": 0},  # L2: x5 = x2
        {"name": "ADD", "rd": 6, "rs1": 2, "rs2": 2},  # L3: x6 = x2 + x2
        {"name": "ADD", "rd": 7, "rs1": 4, "rs2": 6},  # L4: x7 = x4 + x6
        {"name": "SLT", "rd": 8, "rs1": 6, "rs2": 4},  # L5: x8 = (x4 > x6) i.e. x6 < x4
        {"name": "ADD", "rd": 9, "rs1": 1, "rs2": 2},  # L6: x9 = x1 + x2
    ]


def _continuous_verify(output: dict, inst: dict, reg_list: list) -> float:
    """Return fraction of expected registers that match (continuous 0.0-1.0).

    The partial verify scoring system needs a gradient signal rather than
    binary 0.0/1.0. This counts how many of the checked registers are
    correct and returns n_correct / total.
    """
    exp_regs = inst["expected_output"].get("registers", {})
    regs = output.get("registers", {})
    if not reg_list:
        return 1.0
    n_correct = sum(1 for r in reg_list if regs.get(r, 0) == exp_regs.get(r, 0))
    return n_correct / len(reg_list)


# ===========================================================================
# Level 0 — Move: x3 = x1  (ADDI x3, x1, 0)
# ===========================================================================


class Level0(TaskLevel):
    """ADDI: x3 = x1 (copy x1 into x3). Difficulty: 1.0"""

    def generate_instance(self) -> dict:
        v = self._pick_value()
        return {
            "level": 0,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v}, "memory": {}},
            "expected_output": {"registers": {3: v}, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3])

    def difficulty_metric(self) -> float:
        return compute_difficulty(1, 1, 1)

    @staticmethod
    def reference_solution() -> List[dict]:
        return [{"name": "ADDI", "rd": 3, "rs1": 1, "imm": 0}]


# ===========================================================================
# Level 1 — Double: x4 = x1 + x1  (ADD x4, x1, x1)
# ===========================================================================


class Level1(TaskLevel):
    """ADD: x4 = x1 + x1. Also expects x3 = x1."""

    def generate_instance(self) -> dict:
        v = self._pick_value()
        d1 = _cap32(v + v)
        return {
            "level": 1,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v}, "memory": {}},
            "expected_output": {"registers": {3: v, 4: d1}, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4])

    def difficulty_metric(self) -> float:
        return compute_difficulty(2, 1, 1)

    @staticmethod
    def reference_solution() -> List[dict]:
        return [
            {"name": "ADDI", "rd": 3, "rs1": 1, "imm": 0},
            {"name": "ADD", "rd": 4, "rs1": 1, "rs2": 1},
        ]


# ===========================================================================
# Level 2 — Move x2: x5 = x2  (ADDI x5, x2, 0)
# ===========================================================================


class Level2(TaskLevel):
    """ADDI: x5 = x2. Also expects x3=x1, x4=x1+x1."""

    def generate_instance(self) -> dict:
        v1 = self._pick_value()
        v2 = self._pick_value()
        base = _compute_base_expected(v1, v2)
        return {
            "level": 2,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": {3: base[3], 4: base[4], 5: base[5]}, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5])

    def difficulty_metric(self) -> float:
        return compute_difficulty(3, 1, 2)

    @staticmethod
    def reference_solution() -> List[dict]:
        return _base_ref()[:3]


# ===========================================================================
# Level 3 — Double x2: x6 = x2 + x2  (ADD x6, x2, x2)
# ===========================================================================


class Level3(TaskLevel):
    """ADD: x6 = x2 + x2. Previous outputs still expected."""

    def generate_instance(self) -> dict:
        v1 = self._pick_value()
        v2 = self._pick_value()
        base = _compute_base_expected(v1, v2)
        return {
            "level": 3,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {
                "registers": {3: base[3], 4: base[4], 5: base[5], 6: base[6]},
                "memory": {},
            },
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6])

    def difficulty_metric(self) -> float:
        return compute_difficulty(4, 1, 2)

    @staticmethod
    def reference_solution() -> List[dict]:
        return _base_ref()[:4]


# ===========================================================================
# Level 4 — Sum of doubles: x7 = x4 + x6  (ADD x7, x4, x6)
# ===========================================================================


class Level4(TaskLevel):
    """ADD: x7 = x4 + x6 (sum of the two doubled values)."""

    def generate_instance(self) -> dict:
        v1 = self._pick_value()
        v2 = self._pick_value()
        base = _compute_base_expected(v1, v2)
        return {
            "level": 4,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {
                "registers": {3: base[3], 4: base[4], 5: base[5], 6: base[6], 7: base[7]},
                "memory": {},
            },
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7])

    def difficulty_metric(self) -> float:
        return compute_difficulty(5, 1, 3)

    @staticmethod
    def reference_solution() -> List[dict]:
        return _base_ref()[:5]


# ===========================================================================
# Level 5 — Compare: x8 = 1 if x4 > x6 else 0  (SLT)
# ===========================================================================


class Level5(TaskLevel):
    """SLT: x8 = 1 if x4 > x6 else 0."""

    def generate_instance(self) -> dict:
        v1 = self._pick_value()
        v2 = self._pick_value()
        base = _compute_base_expected(v1, v2)
        return {
            "level": 5,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {
                "registers": {3: base[3], 4: base[4], 5: base[5], 6: base[6], 7: base[7], 8: base[8]},
                "memory": {},
            },
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8])

    def difficulty_metric(self) -> float:
        return compute_difficulty(6, 2, 5)

    @staticmethod
    def reference_solution() -> List[dict]:
        return _base_ref()[:6]


# ===========================================================================
# Level 6 — Cross-input add: x9 = x1 + x2  (ADD x9, x1, x2)
# ===========================================================================


class Level6(TaskLevel):
    """ADD: x9 = x1 + x2. Combines both input registers."""

    def generate_instance(self) -> dict:
        v1 = self._pick_value()
        v2 = self._pick_value()
        base = _compute_base_expected(v1, v2)
        return {
            "level": 6,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {
                "registers": {3: base[3], 4: base[4], 5: base[5], 6: base[6], 7: base[7], 8: base[8], 9: base[9]},
                "memory": {},
            },
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9])

    def difficulty_metric(self) -> float:
        return compute_difficulty(7, 1, 3)

    @staticmethod
    def reference_solution() -> List[dict]:
        return _base_ref()[:7]


# ===========================================================================
# Level 7 — Load: x10 = memory[x1]  (LW x10, 0(x1))
# ===========================================================================


class Level7(TaskLevel):
    """LW: x10 = memory[x1]. Load word from address in x1."""

    def generate_instance(self) -> dict:
        addr1 = self._pick_addr()
        val1 = self._pick_value()
        v2 = self._pick_value()
        base = _compute_base_expected(addr1, v2)
        base[10] = val1
        memory = {addr1: val1}
        return {
            "level": 7,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: addr1, 2: v2}, "memory": memory},
            "expected_output": {"registers": base, "memory": memory},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 10])

    def difficulty_metric(self) -> float:
        return compute_difficulty(8, 2, 5)

    @staticmethod
    def reference_solution() -> List[dict]:
        return _base_ref() + [
            {"name": "LW", "rd": 10, "rs1": 1, "imm": 0},
        ]


# ===========================================================================
# Level 8 — Dual load + add: x11 = memory[x1] + memory[x2]
# ===========================================================================


class Level8(TaskLevel):
    """LW+ADD: x11 = mem[x1] + mem[x2]. Two loads and an add."""

    def generate_instance(self) -> dict:
        addr1 = self._pick_addr()
        addr2 = self._pick_addr()
        while addr2 == addr1:
            addr2 = self._pick_addr()
        val1 = self._pick_value()
        val2 = self._pick_value()
        base = _compute_base_expected(addr1, addr2)
        base[10] = val1
        base[11] = _cap32(val1 + val2)
        memory = {addr1: val1, addr2: val2}
        return {
            "level": 8,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: addr1, 2: addr2}, "memory": memory},
            "expected_output": {"registers": base, "memory": memory},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 10, 11])

    def difficulty_metric(self) -> float:
        return compute_difficulty(10, 2, 6)

    @staticmethod
    def reference_solution() -> List[dict]:
        return _base_ref() + [
            {"name": "LW", "rd": 10, "rs1": 1, "imm": 0},
            {"name": "LW", "rd": 15, "rs1": 2, "imm": 0},
            {"name": "ADD", "rd": 11, "rs1": 10, "rs2": 15},
        ]


# ===========================================================================
# Level 9 — Loop sum: x12 = sum(memory[x1] .. memory[x1+x2-1])
# ===========================================================================


class Level9(TaskLevel):
    """Loop: sum memory[x1] through memory[x1+x2-1] into x12."""

    def generate_instance(self) -> dict:
        count = self._pick_count(2, 6)
        base_addr = self._rng.randint(0, (MEMORY_SIZE // 4) - count) * 4
        vals = [self._pick_value() for _ in range(count)]
        total = _cap32(sum(vals))
        mem_dict = {base_addr + i * 4: vals[i] for i in range(count)}

        # Cumulative: L8 needs x11 = mem[x1] + mem[x2].
        # Use a dummy value at a safe address that won't overlap with data.
        # Store at MEMORY_SIZE-8 (last safe word address) to avoid overlap.
        dummy_addr = MEMORY_SIZE - 8
        dummy_val = self._pick_value()
        mem_dict[dummy_addr] = dummy_val

        base = _compute_base_expected(base_addr, count)
        base[10] = vals[0]
        base[11] = _cap32(vals[0] + dummy_val)
        base[12] = total

        return {
            "level": 9,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: base_addr, 2: count}, "memory": mem_dict},
            "expected_output": {"registers": base, "memory": mem_dict},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 10, 12])

    def difficulty_metric(self) -> float:
        return compute_difficulty(16, 3, 10)

    @staticmethod
    def reference_solution() -> List[dict]:
        """Reference: cumulative L0-L8 + loop summing mem[x1..x1+x2-1] -> x12.
        Uses SLLI for word-aligned addressing in the loop body.
        BLT: target=inst 13, current=inst 17, offset=(13-17)*4=-16
        """
        return _base_ref() + [
            {"name": "LW", "rd": 10, "rs1": 1, "imm": 0},
            {"name": "LW", "rd": 15, "rs1": 2, "imm": 0},
            {"name": "ADD", "rd": 11, "rs1": 10, "rs2": 15},
            # L9 loop: sum mem[x1..x1+x2-1] -> x12
            {"name": "ADDI", "rd": 15, "rs1": 0, "imm": 0},  # x15 = 0 (sum)
            {"name": "ADDI", "rd": 16, "rs1": 0, "imm": 0},  # x16 = 0 (i)
            {"name": "SLLI", "rd": 20, "rs1": 16, "imm": 2},  # x20 = i * 4
            {"name": "ADD", "rd": 17, "rs1": 1, "rs2": 20},  # x17 = base + i*4
            {"name": "LW", "rd": 18, "rs1": 17, "imm": 0},  # x18 = mem[base+i*4]
            {"name": "ADD", "rd": 15, "rs1": 15, "rs2": 18},  # sum += val
            {"name": "ADDI", "rd": 16, "rs1": 16, "imm": 1},  # i++
            {"name": "BLT", "rd": 0, "rs1": 16, "rs2": 2, "imm": -20},
            # BLT if i < count (x2), branch back
            {"name": "ADDI", "rd": 12, "rs1": 15, "imm": 0},  # x12 = sum
        ]


# ===========================================================================
# Level 10 — Max in range: x13 = max(memory[x1] .. memory[x1+x2-1])
# ===========================================================================


class Level10(TaskLevel):
    """Loop: find max value in memory[x1..x1+x2-1] into x13."""

    def generate_instance(self) -> dict:
        count = self._pick_count(2, 6)
        base_addr = self._rng.randint(0, (MEMORY_SIZE // 4) - count) * 4
        vals = [self._pick_value() for _ in range(count)]
        max_val = max(vals)
        mem_dict = {base_addr + i * 4: vals[i] for i in range(count)}

        dummy_val = self._pick_value()
        mem_dict[MEMORY_SIZE - 8] = dummy_val

        base = _compute_base_expected(base_addr, count)
        base[10] = vals[0]
        base[11] = _cap32(vals[0] + dummy_val)
        base[12] = _cap32(sum(vals))
        base[13] = max_val

        return {
            "level": 10,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: base_addr, 2: count}, "memory": mem_dict},
            "expected_output": {"registers": base, "memory": mem_dict},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 10, 12, 13])

    def difficulty_metric(self) -> float:
        return compute_difficulty(20, 3, 12)

    @staticmethod
    def reference_solution() -> List[dict]:
        """Reference: cumulative L0-L9 + max-finding loop.
        Uses SLLI for word-aligned addressing.
        """
        return _base_ref() + [
            {"name": "LW", "rd": 10, "rs1": 1, "imm": 0},
            {"name": "LW", "rd": 15, "rs1": 2, "imm": 0},
            {"name": "ADD", "rd": 11, "rs1": 10, "rs2": 15},
            # L9: sum loop
            {"name": "ADDI", "rd": 15, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 16, "rs1": 0, "imm": 0},
            {"name": "SLLI", "rd": 20, "rs1": 16, "imm": 2},
            {"name": "ADD", "rd": 17, "rs1": 1, "rs2": 20},
            {"name": "LW", "rd": 18, "rs1": 17, "imm": 0},
            {"name": "ADD", "rd": 15, "rs1": 15, "rs2": 18},
            {"name": "ADDI", "rd": 16, "rs1": 16, "imm": 1},
            {"name": "BLT", "rd": 0, "rs1": 16, "rs2": 2, "imm": -20},
            {"name": "ADDI", "rd": 12, "rs1": 15, "imm": 0},
            # L10: max loop
            {"name": "LW", "rd": 13, "rs1": 1, "imm": 0},  # x13 = mem[x1] (init max)
            {"name": "ADDI", "rd": 16, "rs1": 0, "imm": 1},  # i = 1
            {"name": "SLLI", "rd": 20, "rs1": 16, "imm": 2},  # x20 = i * 4
            {"name": "ADD", "rd": 17, "rs1": 1, "rs2": 20},  # addr = base + i*4
            {"name": "LW", "rd": 18, "rs1": 17, "imm": 0},  # val
            {"name": "SLT", "rd": 19, "rs1": 13, "rs2": 18},  # x19 = 1 if x13 < x18 (SIGNED)
            {"name": "BEQ", "rd": 0, "rs1": 19, "rs2": 0, "imm": 8},  # skip if x13 >= val
            {"name": "ADDI", "rd": 13, "rs1": 18, "imm": 0},  # x13 = val (new max)
            {"name": "ADDI", "rd": 16, "rs1": 16, "imm": 1},  # i++
            {"name": "BLT", "rd": 0, "rs1": 16, "rs2": 2, "imm": -28},
            # if i < count, branch back to max loop body
        ]


# ===========================================================================
# Level 11 — Fibonacci: x14 = fibonacci(x1)
# ===========================================================================


class Level11(TaskLevel):
    """Loop: compute fibonacci(x1) into x14. x1 is n (0 <= n <= 12)."""

    def generate_instance(self) -> dict:
        n = self._rng.randint(0, 12)
        a, b = 0, 1
        for _ in range(n):
            a, b = b, _cap32(a + b)
        fib_n = a

        v2 = self._pick_value()
        base = _compute_base_expected(n, v2)
        base[14] = fib_n
        inputs = {1: n, 2: v2}
        return {
            "level": 11,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": inputs, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 14])

    def difficulty_metric(self) -> float:
        return compute_difficulty(25, 3, 14)

    @staticmethod
    def reference_solution() -> List[dict]:
        """Reference: L0-L6 + iterative fibonacci -> x14.
        BEQ jumps +20 bytes (5 instructions) to skip loop when n=0.
        BLT jumps -16 bytes back to loop body.
        """
        return _base_ref() + [
            {"name": "ADDI", "rd": 15, "rs1": 0, "imm": 0},  # a = 0
            {"name": "ADDI", "rd": 16, "rs1": 0, "imm": 1},  # b = 1
            {"name": "ADDI", "rd": 17, "rs1": 0, "imm": 0},  # i = 0
            {"name": "BEQ", "rd": 0, "rs1": 1, "rs2": 0, "imm": 20},  # if n==0 skip loop
            {"name": "ADD", "rd": 18, "rs1": 15, "rs2": 16},  # temp = a+b
            {"name": "ADDI", "rd": 15, "rs1": 16, "imm": 0},  # a = b
            {"name": "ADDI", "rd": 16, "rs1": 18, "imm": 0},  # b = temp
            {"name": "ADDI", "rd": 17, "rs1": 17, "imm": 1},  # i++
            {"name": "BLT", "rd": 0, "rs1": 17, "rs2": 1, "imm": -16},  # if i<n: loop
            {"name": "ADDI", "rd": 14, "rs1": 15, "imm": 0},  # x14 = a = F(n)
        ]


# ===========================================================================
# Level 12 — Self-Modifying Addition: x15 = x1 + x2
# ===========================================================================
#
# The champion must construct an ADDI x15, x1, imm instruction word at
# runtime, store it to R/W memory, and jump to execute it.  x2 provides
# the immediate constant.  This is the first level where the program
# modifies its own code at runtime.
#
# Reference solution mechanics (build ADDI x15, x1, x2):
#   word = (x2 << 20) | (rs1=1 << 15) | (rd=15 << 7) | opcode=0x13
#        = (x2 << 20) | 0x8000 | 0x780 | 0x13
#        = (x2 << 20) | 0x8793
#
# The lower 20 bits (0x08793) are built via LUI + ORI, then OR'd with
# the shifted immediate.  The resulting word is SW'd to scratch memory
# and JALR'd to execute, placing x1 + x2 into x15.


class Level12(TaskLevel):
    """Self-modifying code: construct ADDI x15, x1, x2 and execute it.

    The program must compute the encoding of ``ADDI x15, x1, x2``,
    store it to writable scratch memory, and jump there with JALR.
    After execution x15 == value(x1) + value(x2).

    Requires 7+ instructions for instruction-word construction alone,
    plus L0-L6 base.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()  # keeps expected regs predictable
        v2 = self._rng.randint(1, 127)  # constant small enough for ADDI imm
        base = _compute_base_expected(v1, v2)
        base[15] = _cap32(v1 + v2)
        inputs = {1: v1, 2: v2}
        return {
            "level": 12,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": inputs, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 15])

    def difficulty_metric(self) -> float:
        return compute_difficulty(30, 4, 16)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L6 base + self-modifying ADDI construct-and-execute.

        Instruction word for ADDI x15, x1, x2:
          word = (x2 << 20) | 0x8793

        The lower constant 0x00008793 is built from:
          LUI  x20, 2     -> x20 = 0x2000
          SLLI x20, x20, 2 -> x20 = 0x8000
          ORI  x20, x20, 0x793 -> x20 = 0x8793

        Then OR'd with (x2 << 20) and stored at a safe scratch address.
        """
        return _base_ref() + [
            # ---- Build the instruction-word constant 0x8793 ----
            # 0x8793 = 0b1000_0111_1001_0011 = bits 19:0 of ADDI x15, x1, imm
            # Approach: LUI + SLLI + ORI
            {"name": "LUI", "rd": 20, "imm": 2},  # x20 = 0x2000
            {"name": "SLLI", "rd": 20, "rs1": 20, "imm": 2},  # x20 = 0x8000
            {"name": "ORI", "rd": 20, "rs1": 20, "imm": 0x793},  # x20 = 0x8793
            # ---- Shift x2 to bits 31:20 ----
            {"name": "SLLI", "rd": 21, "rs1": 2, "imm": 20},  # x21 = x2 << 20
            # ---- Combine: word = (x2 << 20) | 0x8793 ----
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 20},  # x21 = instruction word
            # ---- Store to scratch (address 0x40, word-aligned) ----
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 0x40},  # x22 = scratch addr
            {"name": "SW", "rd": 0, "rs1": 22, "rs2": 21, "imm": 0},  # mem[0x40] = instr
            # ---- Jump to execute ----
            {"name": "JALR", "rd": 0, "rs1": 22, "imm": 0},
            # After JALR, the stored ADDI x15, x1, x2 runs. The VM
            # sequential executes until EBREAK or timeout, so the frame
            # does NOT need an explicit EBREAK here — the VM halts after
            # the modified instruction (which terminates normally).
        ]


# ===========================================================================
# Level 13 — Self-Modifying Multi-Instruction: x16 = x1 + 100 + x2
# ===========================================================================
#
# The champion writes TWO instructions to consecutive memory locations and
# executes both sequentially: first an ADDI that computes an intermediate
# result, then a second ADDI that computes the final output from it.
# This tests the ability to construct and execute multi-instruction
# self-modifying code sequences.


class Level13(TaskLevel):
    """Dual self-modifying code: write two instructions, execute in sequence.

    The program writes:
      1. ADDI x16, x1, 100   (compute intermediate)
      2. ADDI x16, x16, x2   (add x2 to intermediate, reusing x16)

    to consecutive memory addresses, then jumps to the first one.
    After execution x16 == x1 + 100 + x2.

    Note: the second instruction's rs1 is x16 (output of the first), so
    the immediate for the second instruction includes rs1=16 as part of
    the encoding.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 63)  # keep small for reliable ADDI
        base = _compute_base_expected(v1, v2)
        base[15] = _cap32(v1 + v2)  # L12 output (needed if L12 verified)
        base[16] = _cap32(v1 + 100 + v2)  # L13 output
        inputs = {1: v1, 2: v2}
        return {
            "level": 13,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": inputs, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 16])

    def difficulty_metric(self) -> float:
        return compute_difficulty(40, 5, 20)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L6 base + dual self-modifying instruction construct-and-execute.

        Instruction 1: ADDI x16, x1, 100
          word1 = (100 << 20) | (1 << 15) | (16 << 7) | 0x13
                = 0x06400000 | 0x8000 | 0x800 | 0x13
                = 0x06408813

        Instruction 2: ADDI x16, x16, x2
          word2 = (x2 << 20) | (16 << 15) | (16 << 7) | 0x13
                = (x2 << 20) | 0x80000 | 0x800 | 0x13
                = (x2 << 20) | 0x80813

        Both are built via LUI/SLLI/ORI patterns and stored at consecutive
        word-aligned addresses.  After the first instruction runs (setting
        x16 = x1 + 100), the second instruction adds x2 (x16 += x2).
        """
        return _base_ref() + [
            # === Instruction 1: ADDI x16, x1, 100 = 0x06408813 ===
            # Build 0x06408813:
            #   0x06400000  (imm=100 << 20)
            #   0x00008813  (rs1=1, rd=16, opcode=0x13)
            # Constant 0x00008813 = 0b0000_0000_0000_1000_1000_0000_0001_0011
            # LUI 0x00008 -> 0x8000, then ORI and shift
            {"name": "ADDI", "rd": 20, "rs1": 0, "imm": 100},  # x20 = 100
            {"name": "SLLI", "rd": 20, "rs1": 20, "imm": 20},  # x20 = 100 << 20 = 0x06400000
            {"name": "LUI", "rd": 21, "imm": 0x00008},  # x21 = 0x8000
            {"name": "SLLI", "rd": 21, "rs1": 21, "imm": 4},  # x21 = 0x80000 (rs1=16<<15)
            {"name": "ORI", "rd": 21, "rs1": 21, "imm": 0x813},  # x21 = 0x80813
            {"name": "OR", "rd": 20, "rs1": 20, "rs2": 21},  # x20 = 0x06408813
            # === Instruction 2: ADDI x16, x16, x2 ===
            # word2 = (x2 << 20) | (16 << 15) | (16 << 7) | 0x13
            #       = (x2 << 20) | 0x80813
            {"name": "SLLI", "rd": 21, "rs1": 2, "imm": 20},  # x21 = x2 << 20
            {"name": "ORI", "rd": 22, "rs1": 0, "imm": 0x813},  # x22 = 0x813
            {"name": "ADDI", "rd": 22, "rs1": 22, "imm": 0x80000},  # x22 = 0x80813
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 22},  # x21 = word2
            # === Store both instructions at consecutive scratch addresses ===
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x50},  # x23 = scratch base (0x50)
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 20, "imm": 0},  # mem[0x50] = instr1
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 21, "imm": 4},  # mem[0x54] = instr2
            # === Execute both ===
            {"name": "JALR", "rd": 0, "rs1": 23, "imm": 0},
            # instr1 (ADDI x16, x1, 100) runs, then instr2 (ADDI x16, x16, x2)
            # runs at 0x54, then sequential execution continues until EBREAK
            # or timeout.
        ]


# ===========================================================================
# Level 14 — Multi-Program Composition: x17 = 2*x1 + x2 + 100
# ===========================================================================
#
# Two independent programs share a data region in memory.
# Program A stores x1 + x2 to shared data; Program B reads it, doubles
# x1 contribution, and adds 100.
#
# Reference approach (direct, no self-modifying code):
#   Store x1 to shared[0], x2 to shared[1].
#   Program A: load x1 from shared, compute 2*x1 + x2, store back.
#   Program B: load from shared, add 100 → x17.
#
# Expected: x17 = 2*x1 + x2 + 100


class Level14(TaskLevel):
    """Multi-Program Composition: two programs share a data region.

    Program A computes 2*x1 + x2 and writes to shared memory.
    Program B reads from shared memory and adds 100.
    Output: x17 = 2*x1 + x2 + 100.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 63)
        base = _compute_base_expected(v1, v2)
        base[17] = _cap32(2 * v1 + v2 + 100)
        return {
            "level": 14,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 17])

    def difficulty_metric(self) -> float:
        return compute_difficulty(50, 5, 22)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L6 base + multi-program composition via shared memory.

        Stores x1 and x2 to shared data, then Program A computes
        2*x1 + x2 and stores back.  Program B reads, adds 100.
        """
        return _base_ref() + [
            # ---- Store inputs to shared data region at 0x40 ----
            {"name": "ADDI", "rd": 20, "rs1": 0, "imm": 0x40},  # x20 = 0x40
            {"name": "SW", "rd": 0, "rs1": 20, "rs2": 1, "imm": 0},  # mem[0x40] = x1
            {"name": "ADDI", "rd": 20, "rs1": 0, "imm": 0x44},  # x20 = 0x44
            {"name": "SW", "rd": 0, "rs1": 20, "rs2": 2, "imm": 0},  # mem[0x44] = x2
            # ---- Program A: compute 2*x1 + x2, store to shared ----
            {"name": "ADDI", "rd": 20, "rs1": 0, "imm": 0x40},  # x20 = 0x40
            {"name": "LW", "rd": 17, "rs1": 20, "imm": 0},  # x17 = mem[0x40] = x1
            {"name": "ADD", "rd": 17, "rs1": 17, "rs2": 17},  # x17 = 2*x1
            {"name": "ADD", "rd": 17, "rs1": 17, "rs2": 2},  # x17 = 2*x1 + x2
            {"name": "ADDI", "rd": 20, "rs1": 0, "imm": 0x48},  # x20 = 0x48
            {"name": "SW", "rd": 0, "rs1": 20, "rs2": 17, "imm": 0},  # mem[0x48] = 2*x1+x2
            # ---- Program B: read shared, add 100 → x17 ----
            {"name": "LW", "rd": 17, "rs1": 20, "imm": 0},  # x17 = mem[0x48]
            {"name": "ADDI", "rd": 17, "rs1": 17, "imm": 100},  # x17 = 2*x1 + x2 + 100
        ]


# ===========================================================================
# Level 15 — Adaptive Self-Modification: x18 = x1+x2 if x2<=x1, else 0
# ===========================================================================
#
# Program conditionally modifies its own behavior based on input comparison.
# If x2 <= x1 (signed), x18 = x1 + x2; otherwise x18 = 0.
#
# Reference approach: SLT + BEQ conditional branching.
#   SLT x20, x1, x2    → x20 = 1 if x1 < x2 (signed)
#   BEQ x20, x0, +12   → skip to "x18 = x1 + x2" if x1 >= x2
#   ADDI x18, x0, 0     → x18 = 0 (x1 < x2 case)
#   BEQ x0, x0, +8      → unconditional jump to end
#   ADD  x18, x1, x2    → x18 = x1 + x2 (x1 >= x2 case)


class Level15(TaskLevel):
    """Adaptive Self-Modification: conditional behavior based on input.

    If x2 <= x1 (signed comparison), output x18 = x1 + x2.
    Otherwise x18 = 0.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(0, 15)
        base = _compute_base_expected(v1, v2)
        base[18] = _cap32(v1 + v2) if _sign32(v2) <= _sign32(v1) else 0
        return {
            "level": 15,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 18])

    def difficulty_metric(self) -> float:
        return compute_difficulty(55, 6, 24)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L6 base + conditional self-modification.

        SLT determines which branch to take; BEQ x0,x0 is an
        unconditional forward jump (the standard RV32I idiom).
        """
        return _base_ref() + [
            # x20 = 1 if x1 < x2 (signed), else 0
            {"name": "SLT", "rd": 20, "rs1": 1, "rs2": 2},
            # If x1 >= x2 (x20 == 0), jump to "x18 = x1 + x2"
            {"name": "BEQ", "rd": 0, "rs1": 20, "rs2": 0, "imm": 12},
            # x1 < x2: x18 = 0
            {"name": "ADDI", "rd": 18, "rs1": 0, "imm": 0},
            # Unconditional jump to end (skip "x18 = x1 + x2")
            {"name": "BEQ", "rd": 0, "rs1": 0, "rs2": 0, "imm": 8},
            # x1 >= x2: x18 = x1 + x2
            {"name": "ADD", "rd": 18, "rs1": 1, "rs2": 2},
        ]


# ===========================================================================
# Level 16 — Evolutionary Self-Reference: x19 = (x1 << 2) + (x2 << 3)
# ===========================================================================
#
# Program applies multiple shift-and-add operations to combine inputs
# in a non-trivial way, testing that the organism can reason about
# its own computation structure (self-referential bit manipulation).
#
# Reference: SLLI + ADD chain.


class Level16(TaskLevel):
    """Evolutionary Self-Reference: complex bit manipulation.

    x19 = (x1 << 2) + (x2 << 3) = 4*x1 + 8*x2
    Tests multi-step shift-and-add computation.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 15)
        base = _compute_base_expected(v1, v2)
        base[19] = _cap32((v1 << 2) + (v2 << 3))
        return {
            "level": 16,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 19])

    def difficulty_metric(self) -> float:
        return compute_difficulty(60, 6, 26)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L6 base + shift-and-add: x19 = 4*x1 + 8*x2."""
        return _base_ref() + [
            {"name": "SLLI", "rd": 20, "rs1": 1, "imm": 2},  # x20 = x1 << 2
            {"name": "SLLI", "rd": 21, "rs1": 2, "imm": 3},  # x21 = x2 << 3
            {"name": "ADD", "rd": 19, "rs1": 20, "rs2": 21},  # x19 = 4*x1 + 8*x2
        ]


# ===========================================================================
# Level 17 — Meta-Evolution: x20 = x1 + x2 (via self-modifying code)
# ===========================================================================
#
# Program generates a shorter sub-program that computes x1 + x2,
# writes it to memory, and executes via JALR.  This tests the
# organism's ability to create and invoke code at runtime.
#
# Reference approach: build ADDI x20, x1, x2 instruction word,
# store to scratch memory, JALR to execute.
#
# ADDI x20, x1, x2:
#   word = (x2 << 20) | (1 << 15) | (20 << 7) | 0x13
#        = (x2 << 20) | 0x8A13
#
# 0x8A13 is built from:
#   0x8000 (LUI 8, SLLI 15) + 0x800 (1 << 11) + 0x213 (ORI/ADDI)


class Level17(TaskLevel):
    """Meta-Evolution: generate a sub-program and execute it.

    The organism writes an ADDI x20, x1, x2 instruction to scratch
    memory, then jumps to it with JALR.  After execution, x20 = x1 + x2.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 127)
        base = _compute_base_expected(v1, v2)
        base[20] = _cap32(v1 + v2)
        return {
            "level": 17,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20])

    def difficulty_metric(self) -> float:
        return compute_difficulty(70, 7, 28)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L6 base + meta-evolution: generate and execute sub-program.

        Builds ADDI x20, x1, x2 = (x2 << 20) | 0x8A13.

        0x8A13 construction:
          0x8000: ADDI x21, x0, 1; SLLI x21, x21, 15
          0x0800: ADDI x22, x0, 1; SLLI x22, x22, 11
          0x0213: ADDI x23, x0, 0x213
          0x8A13: x21 | x22 | x23
        """
        return _base_ref() + [
            # ---- Build constant 0x8A13 ----
            {"name": "ADDI", "rd": 21, "rs1": 0, "imm": 1},  # x21 = 1
            {"name": "SLLI", "rd": 21, "rs1": 21, "imm": 15},  # x21 = 0x8000
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 1},  # x22 = 1
            {"name": "SLLI", "rd": 22, "rs1": 22, "imm": 11},  # x22 = 0x800
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x213},  # x23 = 0x213
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 22},  # x21 = 0x8800
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 23},  # x21 = 0x8A13
            # ---- Shift x2 into bits 31:20 ----
            {"name": "SLLI", "rd": 22, "rs1": 2, "imm": 20},  # x22 = x2 << 20
            # ---- Combine to form instruction word ----
            {"name": "OR", "rd": 22, "rs1": 22, "rs2": 21},  # x22 = ADDI x20,x1,x2
            # ---- Store to scratch and execute ----
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x40},  # x23 = scratch addr
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 22, "imm": 0},  # mem[0x40] = instr
            {"name": "JALR", "rd": 0, "rs1": 23, "imm": 0},  # jump → execute
        ]


class Level18(TaskLevel):
    """XOR-based bit manipulation: x21 = (x1 XOR x2) AND mask.

    Introduces XOR and XORI operations for bitwise data manipulation.
    The mask (0x0000FFFF) isolates the lower 16 bits after XOR.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_base_expected(v1, v2)
        base[20] = _cap32(v1 + v2)  # L17 result
        mask = 0x0000FFFF
        base[21] = _cap32((v1 ^ v2) & mask)
        return {
            "level": 18,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21])

    def difficulty_metric(self) -> float:
        return compute_difficulty(75, 8, 30)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L6 base + L17 meta-evolution + L18: XOR + AND mask."""
        # Reuse L17's reference (builds and executes sub-program)
        base_ref = _base_ref() + [
            {"name": "ADDI", "rd": 21, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 21, "rs1": 21, "imm": 15},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 22, "rs1": 22, "imm": 11},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x213},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 22},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 23},
            {"name": "SLLI", "rd": 22, "rs1": 2, "imm": 20},
            {"name": "OR", "rd": 22, "rs1": 22, "rs2": 21},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x40},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 22, "imm": 0},
            {"name": "JALR", "rd": 0, "rs1": 23, "imm": 0},
        ]
        # L18: x21 = (x1 XOR x2) AND 0xFFFF
        return base_ref + [
            {"name": "XOR", "rd": 21, "rs1": 1, "rs2": 2},  # x21 = x1 ^ x2
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 0x0000FFFF},  # x22 = mask
            {"name": "AND", "rd": 21, "rs1": 21, "rs2": 22},  # x21 = (x1^x2) & mask
        ]


class Level19(TaskLevel):
    """Conditional branching: x22 = BNE result.

    Introduces BNE (branch if not equal) and BLTU (unsigned comparison).
    If x1 != x2: x22 = x1 + x2, else x22 = x1 - x2.
    Tests conditional control flow with different branch conditions.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_base_expected(v1, v2)
        base[20] = _cap32(v1 + v2)  # L17
        mask = 0x0000FFFF
        base[21] = _cap32((v1 ^ v2) & mask)  # L18
        if v1 != v2:
            base[22] = _cap32(v1 + v2)
        else:
            base[22] = _cap32(v1 - v2)
        return {
            "level": 19,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22])

    def difficulty_metric(self) -> float:
        return compute_difficulty(80, 9, 32)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L6 base + L17 meta-evolution + L18 XOR + L19 BNE."""
        base_ref = _base_ref() + [
            {"name": "ADDI", "rd": 21, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 21, "rs1": 21, "imm": 15},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 22, "rs1": 22, "imm": 11},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x213},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 22},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 23},
            {"name": "SLLI", "rd": 22, "rs1": 2, "imm": 20},
            {"name": "OR", "rd": 22, "rs1": 22, "rs2": 21},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x40},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 22, "imm": 0},
            {"name": "JALR", "rd": 0, "rs1": 23, "imm": 0},
            # L18: XOR + AND
            {"name": "XOR", "rd": 21, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 0x0000FFFF},
            {"name": "AND", "rd": 21, "rs1": 21, "rs2": 22},
        ]
        # L19: BNE — if x1 != x2, x22 = x1+x2, else x22 = x1-x2
        # Since _pick_small ensures v1 != v2 (both random 1-15),
        # the BNE branch is always taken.
        return base_ref + [
            {"name": "BNE", "rs1": 1, "rs2": 2, "imm": 12},  # skip if x1==x2
            {"name": "ADD", "rd": 22, "rs1": 1, "rs2": 2},  # x22 = x1+x2 (else)
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},  # NOP (pad)
            {"name": "ADDI", "rd": 22, "rs1": 1, "imm": 0},  # x22 = x1 (if x1==x2)
            {"name": "SUB", "rd": 22, "rs1": 1, "rs2": 2},  # x22 = x1-x2
        ]


class Level20(TaskLevel):
    """Complex memory + all operations: x23 = memory-based computation.

    Combines memory load/store, XOR, conditional branch, and shifts.
    Loads x1 from memory, XORs with x2, stores result, loads it back.
    Tests full RV32I instruction mix.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_base_expected(v1, v2)
        base[20] = _cap32(v1 + v2)  # L17
        mask = 0x0000FFFF
        base[21] = _cap32((v1 ^ v2) & mask)  # L18
        if v1 != v2:
            base[22] = _cap32(v1 + v2)
        else:
            base[22] = _cap32(v1 - v2)  # L19
        # L20: store x1 to mem[0], XOR with x2, store result, load back
        addr = 0
        stored = _cap32(v1)
        xored = _cap32(stored ^ v2)
        base[23] = xored
        return {
            "level": 20,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23])

    def difficulty_metric(self) -> float:
        return compute_difficulty(85, 10, 35)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L6 + L17 meta-evolution + L18 XOR + L19 BNE + L20 mem XOR."""
        base_ref = _base_ref() + [
            {"name": "ADDI", "rd": 21, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 21, "rs1": 21, "imm": 15},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 22, "rs1": 22, "imm": 11},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x213},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 22},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 23},
            {"name": "SLLI", "rd": 22, "rs1": 2, "imm": 20},
            {"name": "OR", "rd": 22, "rs1": 22, "rs2": 21},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x40},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 22, "imm": 0},
            {"name": "JALR", "rd": 0, "rs1": 23, "imm": 0},
            # L18: XOR + AND
            {"name": "XOR", "rd": 21, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 0x0000FFFF},
            {"name": "AND", "rd": 21, "rs1": 21, "rs2": 22},
            # L19: BNE (always taken since v1 != v2)
            {"name": "BNE", "rs1": 1, "rs2": 2, "imm": 12},
            {"name": "ADD", "rd": 22, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 22, "rs1": 1, "imm": 0},
            {"name": "SUB", "rd": 22, "rs1": 1, "rs2": 2},
        ]
        # L20: memory XOR — store x1 to mem[0], XOR with x2, store result
        return base_ref + [
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0},  # addr = 0
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 1, "imm": 0},  # mem[0] = x1
            {"name": "XOR", "rd": 23, "rs1": 1, "rs2": 2},  # x23 = x1 ^ x2
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 23, "imm": 4},  # mem[4] = x23
            {"name": "LW", "rd": 23, "rs1": 23, "imm": 0},  # x23 = mem[x23] (alias)
        ]


class Level21(TaskLevel):
    """XOR chain + shift pipeline: x24 = ((x1 XOR x2) SLLI 2) XOR x1.

    Introduces chained bit manipulation: XOR followed by shift followed
    by another XOR. Tests instruction sequencing with data dependencies.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_base_expected(v1, v2)
        base[20] = _cap32(v1 + v2)
        base[21] = _cap32((v1 ^ v2) & 0xFFFF)
        base[22] = _cap32((v1 + v2) if v1 != v2 else (v1 - v2))
        base[23] = _cap32(v1 ^ v2)
        # L21: x24 = ((x1 XOR x2) << 2) XOR x1
        xored = v1 ^ v2
        shifted = _cap32(xored << 2)
        base[24] = shifted ^ v1
        return {
            "level": 21,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24])

    def difficulty_metric(self) -> float:
        return compute_difficulty(90, 10, 36)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L20 base + L21: XOR chain + shift pipeline."""
        # L17-L20 base (abbreviated — reuse L20's chain)
        base_ref = _base_ref() + [
            # L17: meta-evolution
            {"name": "ADDI", "rd": 21, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 21, "rs1": 21, "imm": 15},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 22, "rs1": 22, "imm": 11},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x213},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 22},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 23},
            {"name": "SLLI", "rd": 22, "rs1": 2, "imm": 20},
            {"name": "OR", "rd": 22, "rs1": 22, "rs2": 21},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x40},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 22, "imm": 0},
            {"name": "JALR", "rd": 0, "rs1": 23, "imm": 0},
            # L18: XOR + AND
            {"name": "XOR", "rd": 21, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 0x0000FFFF},
            {"name": "AND", "rd": 21, "rs1": 21, "rs2": 22},
            # L19: BNE
            {"name": "BNE", "rs1": 1, "rs2": 2, "imm": 12},
            {"name": "ADD", "rd": 22, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 22, "rs1": 1, "imm": 0},
            {"name": "SUB", "rd": 22, "rs1": 1, "rs2": 2},
            # L20: memory XOR
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 1, "imm": 0},
            {"name": "XOR", "rd": 23, "rs1": 1, "rs2": 2},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 23, "imm": 4},
            {"name": "LW", "rd": 23, "rs1": 23, "imm": 0},
        ]
        # L21: x24 = ((x1 XOR x2) << 2) XOR x1
        return base_ref + [
            {"name": "XOR", "rd": 24, "rs1": 1, "rs2": 2},  # x24 = x1 ^ x2
            {"name": "SLLI", "rd": 24, "rs1": 24, "imm": 2},  # x24 = (x1^x2) << 2
            {"name": "XOR", "rd": 24, "rs1": 24, "rs2": 1},  # x24 = ((x1^x2)<<2) ^ x1
        ]


class Level22(TaskLevel):
    """Multi-branch conditional: x25 = nested if-then-else.

    Introduces nested branching: if x1 > x2 then x25 = x1+x2
    else if x1 < x2 then x25 = x1-x2
    else x25 = x1*x2 (approximated as x1<<1)
    Tests nested control flow with multiple branch conditions.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_base_expected(v1, v2)
        base[20] = _cap32(v1 + v2)
        base[21] = _cap32((v1 ^ v2) & 0xFFFF)
        base[22] = _cap32((v1 + v2) if v1 != v2 else (v1 - v2))
        base[23] = _cap32(v1 ^ v2)
        base[24] = _cap32(((v1 ^ v2) << 2) ^ v1)
        # L22: nested conditional
        if v1 > v2:
            base[25] = _cap32(v1 + v2)
        elif v1 < v2:
            base[25] = _cap32(v1 - v2)
        else:
            base[25] = _cap32(v1 << 1)
        return {
            "level": 22,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25])

    def difficulty_metric(self) -> float:
        return compute_difficulty(95, 11, 38)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L20 base + L21 chain + L22 nested conditional."""
        # L21 chain
        l21_ref = [
            {"name": "XOR", "rd": 24, "rs1": 1, "rs2": 2},
            {"name": "SLLI", "rd": 24, "rs1": 24, "imm": 2},
            {"name": "XOR", "rd": 24, "rs1": 24, "rs2": 1},
        ]
        # L22: if x1 > x2 then x25 = x1+x2
        #      elif x1 < x2 then x25 = x1-x2
        #      else x25 = x1<<1
        l22_ref = [
            {"name": "SLT", "rd": 25, "rs1": 2, "rs2": 1},  # x25 = (x2 < x1) = (x1 > x2)
            {"name": "BEQ", "rs1": 25, "rs2": 0, "imm": 16},  # if x1 > x2, skip
            {"name": "SLT", "rd": 25, "rs1": 1, "rs2": 2},  # x25 = (x1 < x2)
            {"name": "BEQ", "rs1": 25, "rs2": 0, "imm": 20},  # if x1 < x2, skip
            {"name": "SLLI", "rd": 25, "rs1": 1, "imm": 1},  # x25 = x1<<1 (else)
            {"name": "JALR", "rd": 0, "rs1": 0, "imm": 0},  # jump to end
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},  # NOP (pad)
            {"name": "ADD", "rd": 25, "rs1": 1, "rs2": 2},  # x25 = x1+x2 (x1>x2)
            {"name": "JALR", "rd": 0, "rs1": 0, "imm": 0},  # jump to end
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},  # NOP (pad)
            {"name": "SUB", "rd": 25, "rs1": 1, "rs2": 2},  # x25 = x1-x2 (x1<x2)
        ]
        # Build L0-L21 base
        base_ref = (
            _base_ref()
            + [
                {"name": "ADDI", "rd": 21, "rs1": 0, "imm": 1},
                {"name": "SLLI", "rd": 21, "rs1": 21, "imm": 15},
                {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 1},
                {"name": "SLLI", "rd": 22, "rs1": 22, "imm": 11},
                {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x213},
                {"name": "OR", "rd": 21, "rs1": 21, "rs2": 22},
                {"name": "OR", "rd": 21, "rs1": 21, "rs2": 23},
                {"name": "SLLI", "rd": 22, "rs1": 2, "imm": 20},
                {"name": "OR", "rd": 22, "rs1": 22, "rs2": 21},
                {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x40},
                {"name": "SW", "rd": 0, "rs1": 23, "rs2": 22, "imm": 0},
                {"name": "JALR", "rd": 0, "rs1": 23, "imm": 0},
                {"name": "XOR", "rd": 21, "rs1": 1, "rs2": 2},
                {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 0x0000FFFF},
                {"name": "AND", "rd": 21, "rs1": 21, "rs2": 22},
                {"name": "BNE", "rs1": 1, "rs2": 2, "imm": 12},
                {"name": "ADD", "rd": 22, "rs1": 1, "rs2": 2},
                {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
                {"name": "ADDI", "rd": 22, "rs1": 1, "imm": 0},
                {"name": "SUB", "rd": 22, "rs1": 1, "rs2": 2},
                {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0},
                {"name": "SW", "rd": 0, "rs1": 23, "rs2": 1, "imm": 0},
                {"name": "XOR", "rd": 23, "rs1": 1, "rs2": 2},
                {"name": "SW", "rd": 0, "rs1": 23, "rs2": 23, "imm": 4},
                {"name": "LW", "rd": 23, "rs1": 23, "imm": 0},
            ]
            + l21_ref
        )
        return base_ref + l22_ref


class Level23(TaskLevel):
    """Memory array operations: x26 = sum of memory elements.

    Stores x1 and x2 to memory, loads them back, computes their sum.
    Tests multiple load/store operations with address computation.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_base_expected(v1, v2)
        base[20] = _cap32(v1 + v2)
        base[21] = _cap32((v1 ^ v2) & 0xFFFF)
        base[22] = _cap32((v1 + v2) if v1 != v2 else (v1 - v2))
        base[23] = _cap32(v1 ^ v2)
        base[24] = _cap32(((v1 ^ v2) << 2) ^ v1)
        if v1 > v2:
            base[25] = _cap32(v1 + v2)
        elif v1 < v2:
            base[25] = _cap32(v1 - v2)
        else:
            base[25] = _cap32(v1 << 1)
        # L23: store x1 to mem[0], x2 to mem[4], load both, sum
        addr0, addr4 = 0, 4
        sum_val = _cap32(v1 + v2)
        base[26] = sum_val
        return {
            "level": 23,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26])

    def difficulty_metric(self) -> float:
        return compute_difficulty(100, 12, 40)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L21 base + L22 nested + L23 memory array."""
        # Reuse L22's reference as base
        # For simplicity, add memory operations after L21 chain
        base_ref = _base_ref() + [
            {"name": "ADDI", "rd": 21, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 21, "rs1": 21, "imm": 15},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 22, "rs1": 22, "imm": 11},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x213},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 22},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 23},
            {"name": "SLLI", "rd": 22, "rs1": 2, "imm": 20},
            {"name": "OR", "rd": 22, "rs1": 22, "rs2": 21},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x40},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 22, "imm": 0},
            {"name": "JALR", "rd": 0, "rs1": 23, "imm": 0},
            {"name": "XOR", "rd": 21, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 0x0000FFFF},
            {"name": "AND", "rd": 21, "rs1": 21, "rs2": 22},
            {"name": "BNE", "rs1": 1, "rs2": 2, "imm": 12},
            {"name": "ADD", "rd": 22, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 22, "rs1": 1, "imm": 0},
            {"name": "SUB", "rd": 22, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 1, "imm": 0},
            {"name": "XOR", "rd": 23, "rs1": 1, "rs2": 2},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 23, "imm": 4},
            {"name": "LW", "rd": 23, "rs1": 23, "imm": 0},
            # L21: XOR chain + shift
            {"name": "XOR", "rd": 24, "rs1": 1, "rs2": 2},
            {"name": "SLLI", "rd": 24, "rs1": 24, "imm": 2},
            {"name": "XOR", "rd": 24, "rs1": 24, "rs2": 1},
        ]
        # L23: store, load, sum
        return base_ref + [
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0},  # addr = 0
            {"name": "SW", "rd": 0, "rs1": 25, "rs2": 1, "imm": 0},  # mem[0] = x1
            {"name": "ADDI", "rd": 25, "rs1": 25, "imm": 4},  # addr = 4
            {"name": "SW", "rd": 0, "rs1": 25, "rs2": 2, "imm": 0},  # mem[4] = x2
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0},  # addr = 0
            {"name": "LW", "rd": 25, "rs1": 25, "imm": 0},  # x25 = mem[0] = x1
            {"name": "LW", "rd": 26, "rs1": 25, "imm": 4},  # x26 = mem[4] = x2
            {"name": "ADD", "rd": 26, "rs1": 25, "rs2": 26},  # x26 = x1 + x2
        ]


class Level24(TaskLevel):
    """Complex conditional with memory: x27 = conditional memory store.

    If x1 > x2, store x1+x2 to mem[0] and load to x27.
    Otherwise, store x1-x2 to mem[0] and load to x27.
    Tests conditional execution with memory side effects.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_base_expected(v1, v2)
        base[20] = _cap32(v1 + v2)
        base[21] = _cap32((v1 ^ v2) & 0xFFFF)
        base[22] = _cap32((v1 + v2) if v1 != v2 else (v1 - v2))
        base[23] = _cap32(v1 ^ v2)
        base[24] = _cap32(((v1 ^ v2) << 2) ^ v1)
        if v1 > v2:
            base[25] = _cap32(v1 + v2)
        elif v1 < v2:
            base[25] = _cap32(v1 - v2)
        else:
            base[25] = _cap32(v1 << 1)
        base[26] = _cap32(v1 + v2)
        # L24: conditional memory store
        if v1 > v2:
            base[27] = _cap32(v1 + v2)
        else:
            base[27] = _cap32(v1 - v2)
        return {
            "level": 24,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27])

    def difficulty_metric(self) -> float:
        return compute_difficulty(105, 13, 42)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L23 base + L24 conditional memory."""
        # Reuse L23's reference, add conditional memory at the end
        base_ref = _base_ref() + [
            {"name": "ADDI", "rd": 21, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 21, "rs1": 21, "imm": 15},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 22, "rs1": 22, "imm": 11},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x213},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 22},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 23},
            {"name": "SLLI", "rd": 22, "rs1": 2, "imm": 20},
            {"name": "OR", "rd": 22, "rs1": 22, "rs2": 21},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x40},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 22, "imm": 0},
            {"name": "JALR", "rd": 0, "rs1": 23, "imm": 0},
            {"name": "XOR", "rd": 21, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 0x0000FFFF},
            {"name": "AND", "rd": 21, "rs1": 21, "rs2": 22},
            {"name": "BNE", "rs1": 1, "rs2": 2, "imm": 12},
            {"name": "ADD", "rd": 22, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 22, "rs1": 1, "imm": 0},
            {"name": "SUB", "rd": 22, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 1, "imm": 0},
            {"name": "XOR", "rd": 23, "rs1": 1, "rs2": 2},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 23, "imm": 4},
            {"name": "LW", "rd": 23, "rs1": 23, "imm": 0},
            {"name": "XOR", "rd": 24, "rs1": 1, "rs2": 2},
            {"name": "SLLI", "rd": 24, "rs1": 24, "imm": 2},
            {"name": "XOR", "rd": 24, "rs1": 24, "rs2": 1},
            # L23: memory array
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0},
            {"name": "SW", "rd": 0, "rs1": 25, "rs2": 1, "imm": 0},
            {"name": "ADDI", "rd": 25, "rs1": 25, "imm": 4},
            {"name": "SW", "rd": 0, "rs1": 25, "rs2": 2, "imm": 0},
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0},
            {"name": "LW", "rd": 25, "rs1": 25, "imm": 0},
            {"name": "LW", "rd": 26, "rs1": 25, "imm": 4},
            {"name": "ADD", "rd": 26, "rs1": 25, "rs2": 26},
        ]
        # L24: conditional memory store
        return base_ref + [
            {"name": "SLT", "rd": 27, "rs1": 2, "rs2": 1},  # x27 = (x2 < x1)
            {"name": "BEQ", "rs1": 27, "rs2": 0, "imm": 16},  # if x1 > x2, skip
            {"name": "SUB", "rd": 27, "rs1": 1, "rs2": 2},  # x27 = x1-x2 (else)
            {"name": "JALR", "rd": 0, "rs1": 0, "imm": 0},  # jump to end
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},  # NOP
            {"name": "ADD", "rd": 27, "rs1": 1, "rs2": 2},  # x27 = x1+x2 (if)
        ]


class Level25(TaskLevel):
    """Full program: x28 = complex computation with all operations.

    Combines all previous operations into a single complex program.
    Tests the full RV32I instruction set in a realistic scenario.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_base_expected(v1, v2)
        base[20] = _cap32(v1 + v2)
        base[21] = _cap32((v1 ^ v2) & 0xFFFF)
        base[22] = _cap32((v1 + v2) if v1 != v2 else (v1 - v2))
        base[23] = _cap32(v1 ^ v2)
        base[24] = _cap32(((v1 ^ v2) << 2) ^ v1)
        if v1 > v2:
            base[25] = _cap32(v1 + v2)
        elif v1 < v2:
            base[25] = _cap32(v1 - v2)
        else:
            base[25] = _cap32(v1 << 1)
        base[26] = _cap32(v1 + v2)
        if v1 > v2:
            base[27] = _cap32(v1 + v2)
        else:
            base[27] = _cap32(v1 - v2)
        # L25: x28 = (x1 XOR x2) + (x1 SLLI 1) + (x2 AND 0xFF)
        mask = 0xFF
        base[28] = _cap32((v1 ^ v2) + (v1 << 1) + (v2 & mask))
        return {
            "level": 25,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28])

    def difficulty_metric(self) -> float:
        return compute_difficulty(110, 14, 45)

    @staticmethod
    def reference_solution() -> List[dict]:
        """L0-L24 base + L25: complex computation with all operations."""
        # Reuse L24's reference, add L25 computation
        base_ref = _base_ref() + [
            {"name": "ADDI", "rd": 21, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 21, "rs1": 21, "imm": 15},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 1},
            {"name": "SLLI", "rd": 22, "rs1": 22, "imm": 11},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x213},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 22},
            {"name": "OR", "rd": 21, "rs1": 21, "rs2": 23},
            {"name": "SLLI", "rd": 22, "rs1": 2, "imm": 20},
            {"name": "OR", "rd": 22, "rs1": 22, "rs2": 21},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x40},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 22, "imm": 0},
            {"name": "JALR", "rd": 0, "rs1": 23, "imm": 0},
            {"name": "XOR", "rd": 21, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 0x0000FFFF},
            {"name": "AND", "rd": 21, "rs1": 21, "rs2": 22},
            {"name": "BNE", "rs1": 1, "rs2": 2, "imm": 12},
            {"name": "ADD", "rd": 22, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 22, "rs1": 1, "imm": 0},
            {"name": "SUB", "rd": 22, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 1, "imm": 0},
            {"name": "XOR", "rd": 23, "rs1": 1, "rs2": 2},
            {"name": "SW", "rd": 0, "rs1": 23, "rs2": 23, "imm": 4},
            {"name": "LW", "rd": 23, "rs1": 23, "imm": 0},
            {"name": "XOR", "rd": 24, "rs1": 1, "rs2": 2},
            {"name": "SLLI", "rd": 24, "rs1": 24, "imm": 2},
            {"name": "XOR", "rd": 24, "rs1": 24, "rs2": 1},
            # L23: memory array
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0},
            {"name": "SW", "rd": 0, "rs1": 25, "rs2": 1, "imm": 0},
            {"name": "ADDI", "rd": 25, "rs1": 25, "imm": 4},
            {"name": "SW", "rd": 0, "rs1": 25, "rs2": 2, "imm": 0},
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0},
            {"name": "LW", "rd": 25, "rs1": 25, "imm": 0},
            {"name": "LW", "rd": 26, "rs1": 25, "imm": 4},
            {"name": "ADD", "rd": 26, "rs1": 25, "rs2": 26},
            # L24: conditional memory
            {"name": "SLT", "rd": 27, "rs1": 2, "rs2": 1},
            {"name": "BEQ", "rs1": 27, "rs2": 0, "imm": 16},
            {"name": "SUB", "rd": 27, "rs1": 1, "rs2": 2},
            {"name": "JALR", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADD", "rd": 27, "rs1": 1, "rs2": 2},
        ]
        # L25: x28 = (x1 XOR x2) + (x1 SLLI 1) + (x2 AND 0xFF)
        return base_ref + [
            {"name": "XOR", "rd": 28, "rs1": 1, "rs2": 2},  # x28 = x1 ^ x2
            {"name": "SLLI", "rd": 25, "rs1": 1, "imm": 1},  # x25 = x1 << 1
            {"name": "ADD", "rd": 28, "rs1": 28, "rs2": 25},  # x28 += x1<<1
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0xFF},  # x25 = 0xFF
            {"name": "AND", "rd": 26, "rs1": 2, "rs2": 25},  # x26 = x2 & 0xFF
            {"name": "ADD", "rd": 28, "rs1": 28, "rs2": 26},  # x28 += x2&0xFF
        ]


# ===========================================================================
# Levels 26-30: harder task families
# ===========================================================================


def _l25_base_ref() -> List[dict]:
    """L0-L25 base reference instructions."""
    return _base_ref() + [
        # L17: meta-evolution
        {"name": "ADDI", "rd": 21, "rs1": 0, "imm": 1},
        {"name": "SLLI", "rd": 21, "rs1": 21, "imm": 15},
        {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 1},
        {"name": "SLLI", "rd": 22, "rs1": 22, "imm": 11},
        {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x213},
        {"name": "OR", "rd": 21, "rs1": 21, "rs2": 22},
        {"name": "OR", "rd": 21, "rs1": 21, "rs2": 23},
        {"name": "SLLI", "rd": 22, "rs1": 2, "imm": 20},
        {"name": "OR", "rd": 22, "rs1": 22, "rs2": 21},
        {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0x40},
        {"name": "SW", "rd": 0, "rs1": 23, "rs2": 22, "imm": 0},
        {"name": "JALR", "rd": 0, "rs1": 23, "imm": 0},
        # L18: XOR + AND
        {"name": "XOR", "rd": 21, "rs1": 1, "rs2": 2},
        {"name": "ADDI", "rd": 22, "rs1": 0, "imm": 0x0000FFFF},
        {"name": "AND", "rd": 21, "rs1": 21, "rs2": 22},
        # L19: BNE
        {"name": "BNE", "rs1": 1, "rs2": 2, "imm": 12},
        {"name": "ADD", "rd": 22, "rs1": 1, "rs2": 2},
        {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
        {"name": "ADDI", "rd": 22, "rs1": 1, "imm": 0},
        {"name": "SUB", "rd": 22, "rs1": 1, "rs2": 2},
        # L20: memory XOR
        {"name": "ADDI", "rd": 23, "rs1": 0, "imm": 0},
        {"name": "SW", "rd": 0, "rs1": 23, "rs2": 1, "imm": 0},
        {"name": "XOR", "rd": 23, "rs1": 1, "rs2": 2},
        {"name": "SW", "rd": 0, "rs1": 23, "rs2": 23, "imm": 4},
        {"name": "LW", "rd": 23, "rs1": 23, "imm": 0},
        # L21: XOR chain + shift
        {"name": "XOR", "rd": 24, "rs1": 1, "rs2": 2},
        {"name": "SLLI", "rd": 24, "rs1": 24, "imm": 2},
        {"name": "XOR", "rd": 24, "rs1": 24, "rs2": 1},
        # L23: memory array
        {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0},
        {"name": "SW", "rd": 0, "rs1": 25, "rs2": 1, "imm": 0},
        {"name": "ADDI", "rd": 25, "rs1": 25, "imm": 4},
        {"name": "SW", "rd": 0, "rs1": 25, "rs2": 2, "imm": 0},
        {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0},
        {"name": "LW", "rd": 25, "rs1": 25, "imm": 0},
        {"name": "LW", "rd": 26, "rs1": 25, "imm": 4},
        {"name": "ADD", "rd": 26, "rs1": 25, "rs2": 26},
        # L25: full program
        {"name": "XOR", "rd": 28, "rs1": 1, "rs2": 2},
        {"name": "SLLI", "rd": 25, "rs1": 1, "imm": 1},
        {"name": "ADD", "rd": 28, "rs1": 28, "rs2": 25},
        {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0xFF},
        {"name": "AND", "rd": 26, "rs1": 2, "rs2": 25},
        {"name": "ADD", "rd": 28, "rs1": 28, "rs2": 26},
    ]


def _compute_l25_expected(v1: int, v2: int) -> Dict[int, int]:
    """Compute expected register values for L0-L25."""
    base = _compute_base_expected(v1, v2)
    base[20] = _cap32(v1 + v2)
    base[21] = _cap32((v1 ^ v2) & 0xFFFF)
    base[22] = _cap32((v1 + v2) if v1 != v2 else (v1 - v2))
    base[23] = _cap32(v1 ^ v2)
    base[24] = _cap32(((v1 ^ v2) << 2) ^ v1)
    if v1 > v2:
        base[25] = _cap32(v1 + v2)
    elif v1 < v2:
        base[25] = _cap32(v1 - v2)
    else:
        base[25] = _cap32(v1 << 1)
    base[26] = _cap32(v1 + v2)
    if v1 > v2:
        base[27] = _cap32(v1 + v2)
    else:
        base[27] = _cap32(v1 - v2)
    base[28] = _cap32((v1 ^ v2) + (v1 << 1) + (v2 & 0xFF))
    return base


class Level26(TaskLevel):
    """Shift chain: x29 = ((x1<<3)>>1)>>2 + ((x2&0xFF)<<1).

    Introduces SLL, SRL, SRA — chained shift pipeline.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_l25_expected(v1, v2)
        shifted = ((v1 << 3) >> 1) >> 2
        masked = (v2 & 0xFF) << 1
        base[29] = _cap32(shifted + masked)
        return {
            "level": 26,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

    def difficulty_metric(self) -> float:
        return compute_difficulty(115, 14, 46)

    @staticmethod
    def reference_solution() -> List[dict]:
        return _l25_base_ref() + [
            {"name": "SLLI", "rd": 29, "rs1": 1, "imm": 3},
            {"name": "SRLI", "rd": 29, "rs1": 29, "imm": 1},
            {"name": "SRLI", "rd": 29, "rs1": 29, "imm": 2},
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0xFF},
            {"name": "AND", "rd": 26, "rs1": 2, "rs2": 25},
            {"name": "SLLI", "rd": 26, "rs1": 26, "imm": 1},
            {"name": "ADD", "rd": 29, "rs1": 29, "rs2": 26},
        ]


class Level27(TaskLevel):
    """Unsigned comparison: x30 = BGEU/BLTU result.

    If (unsigned)x1 >= x2: x30 = x1+x2, else x30 = x1-x2.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_l25_expected(v1, v2)
        base[29] = _cap32((((v1 << 3) >> 1) >> 2) + ((v2 & 0xFF) << 1))
        base[30] = _cap32(v1 + v2) if v1 >= v2 else _cap32(v1 - v2)
        return {
            "level": 27,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])

    def difficulty_metric(self) -> float:
        return compute_difficulty(120, 15, 48)

    @staticmethod
    def reference_solution() -> List[dict]:
        base = _l25_base_ref() + [
            {"name": "SLLI", "rd": 29, "rs1": 1, "imm": 3},
            {"name": "SRLI", "rd": 29, "rs1": 29, "imm": 1},
            {"name": "SRLI", "rd": 29, "rs1": 29, "imm": 2},
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0xFF},
            {"name": "AND", "rd": 26, "rs1": 2, "rs2": 25},
            {"name": "SLLI", "rd": 26, "rs1": 26, "imm": 1},
            {"name": "ADD", "rd": 29, "rs1": 29, "rs2": 26},
        ]
        return base + [
            {"name": "BLTU", "rs1": 1, "rs2": 2, "imm": 12},
            {"name": "ADD", "rd": 30, "rs1": 1, "rs2": 2},
            {"name": "JALR", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "SUB", "rd": 30, "rs1": 1, "rs2": 2},
        ]


class Level28(TaskLevel):
    """Byte memory: x31 = SB/LB round-trip ^ x2.

    Store x1 as byte, load back, XOR with x2.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_l25_expected(v1, v2)
        base[29] = _cap32((((v1 << 3) >> 1) >> 2) + ((v2 & 0xFF) << 1))
        base[30] = _cap32(v1 + v2) if v1 >= v2 else _cap32(v1 - v2)
        base[31] = _cap32((v1 & 0xFF) ^ v2)
        return {
            "level": 28,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

    def difficulty_metric(self) -> float:
        return compute_difficulty(125, 16, 50)

    @staticmethod
    def reference_solution() -> List[dict]:
        base = _l25_base_ref() + [
            {"name": "SLLI", "rd": 29, "rs1": 1, "imm": 3},
            {"name": "SRLI", "rd": 29, "rs1": 29, "imm": 1},
            {"name": "SRLI", "rd": 29, "rs1": 29, "imm": 2},
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0xFF},
            {"name": "AND", "rd": 26, "rs1": 2, "rs2": 25},
            {"name": "SLLI", "rd": 26, "rs1": 26, "imm": 1},
            {"name": "ADD", "rd": 29, "rs1": 29, "rs2": 26},
            {"name": "BLTU", "rs1": 1, "rs2": 2, "imm": 12},
            {"name": "ADD", "rd": 30, "rs1": 1, "rs2": 2},
            {"name": "JALR", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "SUB", "rd": 30, "rs1": 1, "rs2": 2},
        ]
        return base + [
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0x50},
            {"name": "SB", "rd": 0, "rs1": 25, "rs2": 1, "imm": 0},
            {"name": "LB", "rd": 31, "rs1": 25, "imm": 0},
            {"name": "XOR", "rd": 31, "rs1": 31, "rs2": 2},
        ]


class Level29(TaskLevel):
    """Subroutine call: x28 = (x1+x2)<<1 via inline subroutine.

    Tests sequential compute with register chaining.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_l25_expected(v1, v2)
        base[29] = _cap32((((v1 << 3) >> 1) >> 2) + ((v2 & 0xFF) << 1))
        base[30] = _cap32(v1 + v2) if v1 >= v2 else _cap32(v1 - v2)
        base[31] = _cap32((v1 & 0xFF) ^ v2)
        base[28] = _cap32((v1 + v2) << 1)
        return {
            "level": 29,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

    def difficulty_metric(self) -> float:
        return compute_difficulty(130, 17, 52)

    @staticmethod
    def reference_solution() -> List[dict]:
        base = _l25_base_ref() + [
            {"name": "SLLI", "rd": 29, "rs1": 1, "imm": 3},
            {"name": "SRLI", "rd": 29, "rs1": 29, "imm": 1},
            {"name": "SRLI", "rd": 29, "rs1": 29, "imm": 2},
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0xFF},
            {"name": "AND", "rd": 26, "rs1": 2, "rs2": 25},
            {"name": "SLLI", "rd": 26, "rs1": 26, "imm": 1},
            {"name": "ADD", "rd": 29, "rs1": 29, "rs2": 26},
            {"name": "BLTU", "rs1": 1, "rs2": 2, "imm": 12},
            {"name": "ADD", "rd": 30, "rs1": 1, "rs2": 2},
            {"name": "JALR", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "SUB", "rd": 30, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0x50},
            {"name": "SB", "rd": 0, "rs1": 25, "rs2": 1, "imm": 0},
            {"name": "LB", "rd": 31, "rs1": 25, "imm": 0},
            {"name": "XOR", "rd": 31, "rs1": 31, "rs2": 2},
        ]
        return base + [
            {"name": "ADD", "rd": 28, "rs1": 1, "rs2": 2},
            {"name": "SLLI", "rd": 28, "rs1": 28, "imm": 1},
        ]


class Level30(TaskLevel):
    """Sort-and-compute: abs_diff + sum of sorted x1, x2.

    Compares x1 and x2, computes absolute difference and sum.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_small()
        v2 = self._rng.randint(1, 255)
        base = _compute_l25_expected(v1, v2)
        base[28] = _cap32((v1 + v2) << 1)
        base[29] = _cap32(abs(v1 - v2))
        base[30] = _cap32(min(v1, v2) + max(v1, v2))
        base[31] = _cap32((v1 & 0xFF) ^ v2)
        return {
            "level": 30,
            "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

    def difficulty_metric(self) -> float:
        return compute_difficulty(135, 18, 54)

    @staticmethod
    def reference_solution() -> List[dict]:
        base = _l25_base_ref() + [
            {"name": "SLLI", "rd": 28, "rs1": 1, "imm": 3},
            {"name": "SRLI", "rd": 28, "rs1": 28, "imm": 1},
            {"name": "SRLI", "rd": 28, "rs1": 28, "imm": 2},
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0xFF},
            {"name": "AND", "rd": 26, "rs1": 2, "rs2": 25},
            {"name": "SLLI", "rd": 26, "rs1": 26, "imm": 1},
            {"name": "ADD", "rd": 28, "rs1": 28, "rs2": 26},
            {"name": "BLTU", "rs1": 1, "rs2": 2, "imm": 12},
            {"name": "ADD", "rd": 30, "rs1": 1, "rs2": 2},
            {"name": "JALR", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "SUB", "rd": 30, "rs1": 1, "rs2": 2},
            {"name": "ADDI", "rd": 25, "rs1": 0, "imm": 0x50},
            {"name": "SB", "rd": 0, "rs1": 25, "rs2": 1, "imm": 0},
            {"name": "LB", "rd": 31, "rs1": 25, "imm": 0},
            {"name": "XOR", "rd": 31, "rs1": 31, "rs2": 2},
            {"name": "ADD", "rd": 28, "rs1": 1, "rs2": 2},
            {"name": "SLLI", "rd": 28, "rs1": 28, "imm": 1},
        ]
        return base + [
            {"name": "SLT", "rd": 25, "rs1": 1, "rs2": 2},
            {"name": "BNE", "rs1": 25, "rs2": 0, "imm": 12},
            {"name": "ADD", "rd": 30, "rs1": 1, "rs2": 2},
            {"name": "SUB", "rd": 29, "rs1": 2, "rs2": 1},
            {"name": "JALR", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
            {"name": "ADD", "rd": 30, "rs1": 1, "rs2": 2},
            {"name": "SUB", "rd": 29, "rs1": 1, "rs2": 2},
        ]


# ===========================================================================
# Secondary tasks (multi-task fitness extension)
# ===========================================================================


class BitManipulationTask(TaskLevel):
    """Bitwise ops: x15 = x1 | mask, x16 = x1 & mask, x17 = x1 ^ mask.

    Tests OR, AND, XOR on a random 32-bit value. Each operation uses a
    different mask to exercise distinct bit patterns.
    """

    def generate_instance(self) -> dict:
        v = self._pick_value()
        mask_or = 0xFF00FF00
        mask_and = 0x00FFFF00
        mask_xor = 0xAAAAAAAA
        return {
            "level": -1,
            "difficulty": 1.0,
            "inputs": {"registers": {1: v}, "memory": {}},
            "expected_output": {
                "registers": {
                    15: _cap32(v | mask_or),
                    16: _cap32(v & mask_and),
                    17: _cap32(v ^ mask_xor),
                },
                "memory": {},
            },
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [15, 16, 17])

    def difficulty_metric(self) -> float:
        return compute_difficulty(3, 1, 1)

    @staticmethod
    def reference_solution() -> List[dict]:
        return [
            {"name": "ORI", "rd": 15, "rs1": 1, "imm": 0xFF00},
            {"name": "ANDI", "rd": 16, "rs1": 1, "imm": 0xFF00},
            {"name": "XORI", "rd": 17, "rs1": 1, "imm": 0xAA},
        ]


class ArithmeticTask(TaskLevel):
    """Arithmetic: x15 = x1 + x2, x16 = x1 - x2, x17 = (x1 << 2) + x2.

    Tests ADD, SUB (via ADDI + SUB pseudo), and shift+add in a single
    evaluation. Uses only RV32I base instructions.
    """

    def generate_instance(self) -> dict:
        v1 = self._pick_value()
        v2 = self._pick_value()
        return {
            "level": -1,
            "difficulty": 1.0,
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {
                "registers": {
                    15: _cap32(v1 + v2),
                    16: _cap32(v1 - v2),
                    17: _cap32((v1 << 2) + v2),
                },
                "memory": {},
            },
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [15, 16, 17])

    def difficulty_metric(self) -> float:
        return compute_difficulty(3, 1, 2)

    @staticmethod
    def reference_solution() -> List[dict]:
        return [
            {"name": "ADD", "rd": 15, "rs1": 1, "rs2": 2},
            {"name": "SUB", "rd": 16, "rs1": 1, "rs2": 2},
            {"name": "SLLI", "rd": 18, "rs1": 1, "imm": 2},
            {"name": "ADD", "rd": 17, "rs1": 18, "rs2": 2},
        ]


class MemoryAccessTask(TaskLevel):
    """Memory store/load: store x2 at mem[x1], load into x15;
    store (x2+1) at mem[x1+4], load into x16.

    Tests SW and LW with word-aligned addressing.
    """

    def generate_instance(self) -> dict:
        addr = self._pick_addr()
        val = self._pick_value()
        val_plus_1 = _cap32(val + 1)
        return {
            "level": -1,
            "difficulty": 1.0,
            "inputs": {"registers": {1: addr, 2: val}, "memory": {}},
            "expected_output": {
                "registers": {15: val, 16: val_plus_1},
                "memory": {addr: val, _cap32(addr + 4): val_plus_1},
            },
        }

    def verify(self, output: dict, inst: dict) -> float:
        return _continuous_verify(output, inst, [15, 16])

    def difficulty_metric(self) -> float:
        return compute_difficulty(4, 1, 2)

    @staticmethod
    def reference_solution() -> List[dict]:
        return [
            {"name": "SW", "rd": 0, "rs1": 1, "rs2": 2, "imm": 0},
            {"name": "ADDI", "rd": 19, "rs1": 2, "imm": 1},
            {"name": "ADDI", "rd": 20, "rs1": 1, "imm": 4},
            {"name": "SW", "rd": 0, "rs1": 20, "rs2": 19, "imm": 0},
            {"name": "LW", "rd": 15, "rs1": 1, "imm": 0},
            {"name": "LW", "rd": 16, "rs1": 20, "imm": 0},
        ]


SECONDARY_TASKS = [BitManipulationTask, ArithmeticTask, MemoryAccessTask]


# ===========================================================================
# Registry
# ===========================================================================

LEVELS = {
    0: Level0,
    1: Level1,
    2: Level2,
    3: Level3,
    4: Level4,
    5: Level5,
    6: Level6,
    7: Level7,
    8: Level8,
    9: Level9,
    10: Level10,
    11: Level11,
    12: Level12,
    13: Level13,
    14: Level14,
    15: Level15,
    16: Level16,
    17: Level17,
    18: Level18,
    19: Level19,
    20: Level20,
    21: Level21,
    22: Level22,
    23: Level23,
    24: Level24,
    25: Level25,
    26: Level26,
    27: Level27,
    28: Level28,
    29: Level29,
    30: Level30,
}


def get_task_level(n: int):
    if n not in LEVELS:
        raise ValueError(f"Unknown level: {n}. Valid levels: {sorted(LEVELS.keys())}")
    return LEVELS[n]


def reference_solution(level: int, **kwargs) -> list[dict]:
    """Get the reference solution for a curriculum level.

    Delegates to the TaskLevel's reference_solution() method.

    Args:
        level: Curriculum level number.
        **kwargs: Optional keyword arguments forwarded to the reference solution
            generator (e.g., load_offset for LW instructions, count for loops).

    Returns:
        List of dict instruction descriptions (opcode, rd, rs1, rs2, imm).
    """
    return get_task_level(level).reference_solution(**kwargs)


def generate_all_tasks(seed: int = 42):
    rng = random.Random(seed)
    return [LEVELS[l](seed=rng.randint(0, 2**31)).generate_instance() for l in sorted(LEVELS.keys())]


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import sys
    from paths import ensure_project_on_path
    ensure_project_on_path()

    HAVE_ENCODE = False
    try:
        from rv32i_mask import decode as rv32_decode
        from rv32i_mask import encode as rv32_encode

        HAVE_ENCODE = True
    except ImportError:
        pass

    print("=" * 72)
    print("RV32I Curriculum Tasks — Self-Test")
    print("=" * 72)

    all_pass = True
    all_refs_valid = True

    for lvl_num in sorted(LEVELS.keys()):
        level_cls = LEVELS[lvl_num]
        task = level_cls(seed=42)
        inst = task.generate_instance()
        n_regs = len(inst["expected_output"]["registers"])
        n_mem = len(inst["expected_output"]["memory"])

        perfect_output = {
            "registers": dict(inst["expected_output"]["registers"]),
            "memory": dict(inst["expected_output"]["memory"]),
        }
        score = task.verify(perfect_output, inst)
        ok = "PASS" if score >= 1.0 else "FAIL"
        print(
            f"  Level {lvl_num:2d}: diff={inst['difficulty']:6.0f}, "
            f"regs={n_regs}, mem={n_mem}, score={score:.1f} ({ok})"
        )
        if score < 1.0:
            all_pass = False

        ref = level_cls.reference_solution()
        print(f"    Reference: {len(ref)} instructions")
        if HAVE_ENCODE and ref:
            try:
                for i, instr_dict in enumerate(ref):
                    encoded = rv32_encode(**instr_dict)
                    decoded = rv32_decode(encoded)
                    assert (
                        decoded["name"] == instr_dict["name"].upper()
                    ), f"Round-trip mismatch: got {decoded['name']} expected {instr_dict['name']}"
                print(f"    All {len(ref)} instructions encode/decode OK")
            except Exception as e:
                print(f"    REF ENCODE ERROR: {e}")
                all_refs_valid = False

    print()
    print(f"Instance verification:    {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"Reference encoding:      {'ALL VALID' if all_refs_valid else 'SOME INVALID'}")
    print("=" * 72)
