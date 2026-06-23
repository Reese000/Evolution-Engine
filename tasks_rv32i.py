#!/usr/bin/env python3
"""
RV32I curriculum tasks — 12-level monotonic curriculum ported from
simulation/tasks_monotonic.py.

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
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

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
    def generate_instance(self) -> dict:
        ...

    @abstractmethod
    def verify(self, output: dict, inst: dict) -> float:
        ...

    @abstractmethod
    def difficulty_metric(self) -> float:
        ...

    @staticmethod
    @abstractmethod
    def reference_solution() -> List[dict]:
        ...

    def _pick_value(self) -> int:
        """Pick a random 32-bit register value in signed i32 range."""
        v = self._rng.randint(0, 0xFFFF_FFFF)
        # Convert to signed i32: values > 0x7FFFFFFF become negative
        if v >= 0x80000000:
            v -= 0x1_0000_0000
        return v

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

    L0: x3 = x1
    L1: x4 = x1 + x1
    L2: x5 = x2
    L3: x6 = x2 + x2
    L4: x7 = x4 + x6
    L5: x8 = 1 if x4 > x6 (signed) else 0
    L6: x9 = x1 + x2
    """
    x3 = x1_val
    x4 = _cap32(x1_val + x1_val)
    x5 = x2_val
    x6 = _cap32(x2_val + x2_val)
    x7 = _cap32(x4 + x6)
    x8 = 1 if _sign32(x4) > _sign32(x6) else 0
    x9 = _cap32(x1_val + x2_val)
    return {3: x3, 4: x4, 5: x5, 6: x6, 7: x7, 8: x8, 9: x9}


def _base_ref() -> List[dict]:
    """Reference instructions for cumulative L0-L6."""
    return [
        {"name": "ADDI", "rd": 3, "rs1": 1, "imm": 0},   # L0: x3 = x1
        {"name": "ADD", "rd": 4, "rs1": 1, "rs2": 1},    # L1: x4 = x1 + x1
        {"name": "ADDI", "rd": 5, "rs1": 2, "imm": 0},   # L2: x5 = x2
        {"name": "ADD", "rd": 6, "rs1": 2, "rs2": 2},    # L3: x6 = x2 + x2
        {"name": "ADD", "rd": 7, "rs1": 4, "rs2": 6},    # L4: x7 = x4 + x6
        {"name": "SLT", "rd": 8, "rs1": 6, "rs2": 4},    # L5: x8 = (x4 > x6) i.e. x6 < x4
        {"name": "ADD", "rd": 9, "rs1": 1, "rs2": 2},    # L6: x9 = x1 + x2
    ]


# ===========================================================================
# Level 0 — Move: x3 = x1  (ADDI x3, x1, 0)
# ===========================================================================


class Level0(TaskLevel):
    """ADDI: x3 = x1 (copy x1 into x3). Difficulty: 1.0"""

    def generate_instance(self) -> dict:
        v = self._pick_value()
        return {
            "level": 0, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v}, "memory": {}},
            "expected_output": {"registers": {3: v}, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        return 1.0 if regs.get(3, 0) == exp["registers"].get(3, 0) else 0.0

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
            "level": 1, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v}, "memory": {}},
            "expected_output": {"registers": {3: v, 4: d1}, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        return 1.0 if (regs.get(3, 0) == exp["registers"].get(3, 0)
                       and regs.get(4, 0) == exp["registers"].get(4, 0)) else 0.0

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
            "level": 2, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {"registers": {3: base[3], 4: base[4], 5: base[5]}, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        for r in [3, 4, 5]:
            if regs.get(r, 0) != exp["registers"].get(r, 0):
                return 0.0
        return 1.0

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
            "level": 3, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {
                "registers": {3: base[3], 4: base[4], 5: base[5], 6: base[6]},
                "memory": {},
            },
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        for r in [3, 4, 5, 6]:
            if regs.get(r, 0) != exp["registers"].get(r, 0):
                return 0.0
        return 1.0

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
            "level": 4, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {
                "registers": {3: base[3], 4: base[4], 5: base[5],
                              6: base[6], 7: base[7]},
                "memory": {},
            },
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        for r in range(3, 8):
            if regs.get(r, 0) != exp["registers"].get(r, 0):
                return 0.0
        return 1.0

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
            "level": 5, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {
                "registers": {3: base[3], 4: base[4], 5: base[5],
                              6: base[6], 7: base[7], 8: base[8]},
                "memory": {},
            },
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        for r in range(3, 9):
            if regs.get(r, 0) != exp["registers"].get(r, 0):
                return 0.0
        return 1.0

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
            "level": 6, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: v1, 2: v2}, "memory": {}},
            "expected_output": {
                "registers": {3: base[3], 4: base[4], 5: base[5],
                              6: base[6], 7: base[7], 8: base[8], 9: base[9]},
                "memory": {},
            },
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        for r in range(3, 10):
            if regs.get(r, 0) != exp["registers"].get(r, 0):
                return 0.0
        return 1.0

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
            "level": 7, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: addr1, 2: v2}, "memory": memory},
            "expected_output": {"registers": base, "memory": memory},
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        for r in range(3, 11):
            if regs.get(r, 0) != exp["registers"].get(r, 0):
                return 0.0
        return 1.0

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
            "level": 8, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: addr1, 2: addr2}, "memory": memory},
            "expected_output": {"registers": base, "memory": memory},
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        for r in range(3, 12):
            if regs.get(r, 0) != exp["registers"].get(r, 0):
                return 0.0
        return 1.0

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
            "level": 9, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: base_addr, 2: count}, "memory": mem_dict},
            "expected_output": {"registers": base, "memory": mem_dict},
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        # Check L0-L7 (x3-x10) and L9 (x12).
        # Skip x11 (L8) — L8 uses x2 as address, but L9 uses x2 as count.
        # The champion passes L8 standalone where x2 is an address.
        for r in range(3, 11):
            if regs.get(r, 0) != exp["registers"].get(r, 0):
                return 0.0
        if regs.get(12, 0) != exp["registers"].get(12, 0):
            return 0.0
        return 1.0

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
            {"name": "ADDI", "rd": 15, "rs1": 0, "imm": 0},   # x15 = 0 (sum)
            {"name": "ADDI", "rd": 16, "rs1": 0, "imm": 0},   # x16 = 0 (i)
            {"name": "SLLI", "rd": 20, "rs1": 16, "imm": 2},  # x20 = i * 4
            {"name": "ADD", "rd": 17, "rs1": 1, "rs2": 20},   # x17 = base + i*4
            {"name": "LW", "rd": 18, "rs1": 17, "imm": 0},    # x18 = mem[base+i*4]
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
            "level": 10, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": {1: base_addr, 2: count}, "memory": mem_dict},
            "expected_output": {"registers": base, "memory": mem_dict},
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        # Check L0-L7 (x3-x10), L9 (x12), L10 (x13).
        # Skip x11 (L8) — same x2 semantic change as L9.
        for r in range(3, 11):
            if regs.get(r, 0) != exp["registers"].get(r, 0):
                return 0.0
        for r in [12, 13]:
            if regs.get(r, 0) != exp["registers"].get(r, 0):
                return 0.0
        return 1.0

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
            {"name": "LW", "rd": 13, "rs1": 1, "imm": 0},     # x13 = mem[x1] (init max)
            {"name": "ADDI", "rd": 16, "rs1": 0, "imm": 1},   # i = 1
            {"name": "SLLI", "rd": 20, "rs1": 16, "imm": 2},  # x20 = i * 4
            {"name": "ADD", "rd": 17, "rs1": 1, "rs2": 20},   # addr = base + i*4
            {"name": "LW", "rd": 18, "rs1": 17, "imm": 0},    # val
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
            "level": 11, "difficulty": self.difficulty_metric(),
            "inputs": {"registers": inputs, "memory": {}},
            "expected_output": {"registers": base, "memory": {}},
        }

    def verify(self, output: dict, inst: dict) -> float:
        exp = inst["expected_output"]
        regs = output.get("registers", {})
        for r in range(3, 10):  # L0-L6
            if regs.get(r, 0) != exp["registers"].get(r, 0):
                return 0.0
        if regs.get(14, 0) != exp["registers"].get(14, 0):  # L11
            return 0.0
        return 1.0

    def difficulty_metric(self) -> float:
        return compute_difficulty(25, 3, 14)

    @staticmethod
    def reference_solution() -> List[dict]:
        """Reference: L0-L6 + iterative fibonacci -> x14.
        BEQ jumps +20 bytes (5 instructions) to skip loop when n=0.
        BLT jumps -16 bytes back to loop body.
        """
        return _base_ref() + [
            {"name": "ADDI", "rd": 15, "rs1": 0, "imm": 0},   # a = 0
            {"name": "ADDI", "rd": 16, "rs1": 0, "imm": 1},   # b = 1
            {"name": "ADDI", "rd": 17, "rs1": 0, "imm": 0},   # i = 0
            {"name": "BEQ", "rd": 0, "rs1": 1, "rs2": 0, "imm": 20},  # if n==0 skip loop
            {"name": "ADD", "rd": 18, "rs1": 15, "rs2": 16},  # temp = a+b
            {"name": "ADDI", "rd": 15, "rs1": 16, "imm": 0},  # a = b
            {"name": "ADDI", "rd": 16, "rs1": 18, "imm": 0},  # b = temp
            {"name": "ADDI", "rd": 17, "rs1": 17, "imm": 1},  # i++
            {"name": "BLT", "rd": 0, "rs1": 17, "rs2": 1, "imm": -16},  # if i<n: loop
            {"name": "ADDI", "rd": 14, "rs1": 15, "imm": 0},  # x14 = a = F(n)
        ]


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
    return [
        LEVELS[l](seed=rng.randint(0, 2**31)).generate_instance()
        for l in sorted(LEVELS.keys())
    ]


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import sys
    HAVE_ENCODE = False
    try:
        sys.path.insert(0, "../milestone0/logit-masking")
        from rv32i_mask import encode as rv32_encode, decode as rv32_decode
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
        print(f"  Level {lvl_num:2d}: diff={inst['difficulty']:6.0f}, "
              f"regs={n_regs}, mem={n_mem}, score={score:.1f} ({ok})")
        if score < 1.0:
            all_pass = False

        ref = level_cls.reference_solution()
        print(f"    Reference: {len(ref)} instructions")
        if HAVE_ENCODE and ref:
            try:
                for i, instr_dict in enumerate(ref):
                    encoded = rv32_encode(**instr_dict)
                    decoded = rv32_decode(encoded)
                    assert decoded["name"] == instr_dict["name"].upper(), \
                        f"Round-trip mismatch: got {decoded['name']} expected {instr_dict['name']}"
                print(f"    All {len(ref)} instructions encode/decode OK")
            except Exception as e:
                print(f"    REF ENCODE ERROR: {e}")
                all_refs_valid = False

    print()
    print(f"Instance verification:    {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"Reference encoding:      {'ALL VALID' if all_refs_valid else 'SOME INVALID'}")
    print("=" * 72)
