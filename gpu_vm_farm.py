#!/usr/bin/env python3
"""gpu_vm_farm.py — GPU-accelerated RV32I VM batch evaluator.

Replaces per-program subprocess spawning with in-process execution.
When CuPy is available, uses GPU for batch register/memory operations.
Falls back to pure Python when CUDA is unavailable.

Key speedup: avoids 50+ subprocess spawns per generation (~2750/cycle).
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from typing import Any

import numpy as np

# EE-W5: record VM wall time at the SOURCE.  Direct callers of
# ``_execute_vm`` (terminal/scan-floor gate scoring calls this primitive
# without going through ``VMFarm.evaluate``/``evaluate_batch``) would
# otherwise be booked as orchestration time, which is exactly the
# VM-vs-orchestration question the phase split exists to answer.  The
# guard in ``timed_vm_eval`` is re-entrancy safe, so a call that already
# runs inside a timed VMFarm batch is counted once, not twice.  The
# nullcontext fallback keeps the evaluator usable if the instrumentation
# module cannot be imported.
try:
    from evolution.phase_timing import timed_vm_eval
except Exception as _exc:  # pragma: no cover -- instrumentation must never block eval
    timed_vm_eval = nullcontext  # type: ignore[assignment]
    import sys as _sys  # local import: only used on this failure path

    _sys.stderr.write(f"[gpu_vm_farm] phase_timing unavailable, VM timing off: {_exc!r}\n")

# Lazy CuPy import — only import when GPU is actually needed to avoid
# grabbing ~8GB VRAM at module load time, which blocks PyTorch training.
HAS_CUDA: bool = False
cp = None  # type: ignore


def _ensure_cupy():
    """Import CuPy on first use. Sets HAS_CUDA and cp module-level vars."""
    global HAS_CUDA, cp
    if cp is not None:
        return HAS_CUDA
    try:
        import cupy as _cp

        cp = _cp
        HAS_CUDA = _cp.is_available()
    except ImportError:
        HAS_CUDA = False
    return HAS_CUDA


# ══════════════════════════════════════════════════════════════════════
#  GPU Memory Management
# ══════════════════════════════════════════════════════════════════════


def get_gpu_vram_info() -> dict[str, Any]:
    """Return GPU VRAM usage statistics.

    Returns dict with:
        available: bool — whether CUDA GPU is present
        total_mb: float — total VRAM in MB
        used_mb: float — currently allocated VRAM in MB
        free_mb: float — free VRAM in MB
        utilization: float — fraction used (0.0–1.0)
        device_name: str — GPU name or "N/A"
    """
    info: dict[str, Any] = {
        "available": False,
        "total_mb": 0.0,
        "used_mb": 0.0,
        "free_mb": 0.0,
        "utilization": 0.0,
        "device_name": "N/A",
    }

    # Try PyTorch first
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(dev).total_memory / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(dev) / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(dev) / (1024 * 1024)
            free = total - reserved
            info.update(
                {
                    "available": True,
                    "total_mb": round(total, 1),
                    "used_mb": round(allocated, 1),
                    "free_mb": round(free, 1),
                    "utilization": round(allocated / max(total, 1.0), 3),
                    "device_name": torch.cuda.get_device_name(dev),
                }
            )
            return info
    except (ImportError, RuntimeError):
        pass

    # Try CuPy next
    _ensure_cupy()
    if HAS_CUDA and cp is not None:
        try:
            cp.get_default_memory_pool()
            total = cp.cuda.Device().mem_info[1] / (1024 * 1024)
            used = cp.cuda.Device().mem_info[0] / (1024 * 1024)
            free = total - used
            info.update(
                {
                    "available": True,
                    "total_mb": round(total, 1),
                    "used_mb": round(used, 1),
                    "free_mb": round(free, 1),
                    "utilization": round(used / max(total, 1.0), 3),
                    "device_name": str(cp.cuda.Device().attributes.get("DeviceName", "CuPy GPU")),
                }
            )
            return info
        except (OSError, RuntimeError, ValueError):
            pass

    return info


def clear_gpu_memory() -> dict[str, Any]:
    """Clear GPU memory from both PyTorch and CuPy.

    Returns dict with:
        torch_cleared: bool
        cupy_cleared: bool
        freed_mb: float — approximate MB freed
    """
    result: dict[str, Any] = {"torch_cleared": False, "cupy_cleared": False, "freed_mb": 0.0}

    # Measure before
    before = get_gpu_vram_info()
    before_used = before.get("used_mb", 0.0)

    # PyTorch
    try:
        import gc

        gc.collect()
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            result["torch_cleared"] = True
    except (ImportError, RuntimeError):
        pass

    # CuPy
    try:
        import cupy

        mem = cupy.get_default_memory_pool()
        mem.free_all_blocks()
        result["cupy_cleared"] = True
    except (ImportError, AttributeError):
        pass

    # Measure after
    after = get_gpu_vram_info()
    after_used = after.get("used_mb", 0.0)
    result["freed_mb"] = max(0.0, round(before_used - after_used, 1))

    return result


def check_gpu_capacity(threshold: float = 0.90) -> dict[str, Any]:
    """Check if GPU has capacity for more work.

    Args:
        threshold: Maximum utilization fraction before fallback (default 90%).

    Returns dict with:
        has_capacity: bool — True if GPU can handle more work
        vram_info: dict — full VRAM stats
        reason: str — explanation if capacity is low
    """
    vram = get_gpu_vram_info()
    result: dict[str, Any] = {
        "has_capacity": True,
        "vram_info": vram,
        "reason": "",
    }

    if not vram["available"]:
        result["has_capacity"] = False
        result["reason"] = "No CUDA GPU available"
        return result

    if vram["utilization"] >= threshold:
        result["has_capacity"] = False
        result["reason"] = (
            f"GPU utilization {vram['utilization']:.0%} exceeds "
            f"threshold {threshold:.0%} "
            f"({vram['used_mb']:.0f}/{vram['total_mb']:.0f} MB used)"
        )
        return result

    if vram["free_mb"] < 512:
        result["has_capacity"] = False
        result["reason"] = f"Only {vram['free_mb']:.0f} MB free VRAM (minimum 512 MB required)"
        return result

    return result


# ══════════════════════════════════════════════════════════════════════
#  RV32I Opcode Constants
# ══════════════════════════════════════════════════════════════════════

# Opcode field (bits [6:0])
OP_LUI = 0b0110111
OP_AUIPC = 0b0010111
OP_JAL = 0b1101111
OP_JALR = 0b1100111
OP_BRANCH = 0b1100011
OP_LOAD = 0b0000011
OP_STORE = 0b0100011
OP_OP_IMM = 0b0010011
OP_OP = 0b0110011
OP_FENCE = 0b0001111
OP_ECALL = 0b1110011

# funct3 values
F3_ADD_SUB = 0b000
F3_SLL = 0b001
F3_SLT = 0b010
F3_SLTU = 0b011
F3_XOR = 0b100
F3_SRL_SRA = 0b101
F3_OR = 0b110
F3_AND = 0b111

# Branch funct3
F3_BEQ = 0b000
F3_BNE = 0b001
F3_BLT = 0b100
F3_BGE = 0b101
F3_BLTU = 0b110
F3_BGEU = 0b111

# Load funct3
F3_LB = 0b000
F3_LH = 0b001
F3_LW = 0b010
F3_LBU = 0b100
F3_LHU = 0b101

# Store funct3
F3_SB = 0b000
F3_SH = 0b001
F3_SW = 0b010

# EBREAK
EBREAK = 0x00100073

# Mask constants
MASK8 = 0xFF
MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

# Mask constants for sign-extend widths
_MASK12 = 0xFFF
_MASK20 = 0xFFFFF

# Pre-computed sign-extend masks and thresholds for common widths.
# Avoids recomputing (1 << bits) - 1 and (1 << (bits-1)) on every call.
_SE_MASKS = {
    8: (0xFF, 0x80),
    12: (0xFFF, 0x800),
    13: (0x1FFF, 0x1000),
    16: (0xFFFF, 0x8000),
    20: (0xFFFFF, 0x80000),
    21: (0x1FFFFF, 0x100000),
}


def _sign_extend(val: int, bits: int) -> int:
    """Sign-extend an integer from `bits` width to Python int.

    Uses pre-computed mask/threshold pairs for common widths (8,12,13,16,20,21)
    to avoid the (1 << bits) - 1 computation on every call. Falls back to
    general path for uncommon widths.
    """
    pair = _SE_MASKS.get(bits)
    if pair is not None:
        mask, threshold = pair
        val = val & mask
        if val & threshold:
            val -= 1 << bits
        return val
    mask = (1 << bits) - 1
    val = val & mask
    if val & (1 << (bits - 1)):
        val -= 1 << bits
    return val


def _sign_extend32(val: int) -> int:
    """Sign-extend a 32-bit unsigned to signed Python int."""
    val = val & MASK32
    if val & 0x80000000:
        val -= 0x100000000
    return val


def _decode_insn(word: int) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Precompute opcode fields and the immediate used by this opcode."""
    if word == EBREAK:
        return (0, 0, 0, 0, 0, 0, 0, 0, word)
    opcode = word & 0x7F
    rd = (word >> 7) & 0x1F
    funct3 = (word >> 12) & 0x07
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    funct7 = (word >> 25) & 0x7F
    if opcode in (OP_OP_IMM, OP_LOAD, OP_JALR):
        imm = _sign_extend((word >> 20) & _MASK12, 12)
    elif opcode == OP_STORE:
        imm = _sign_extend(((word >> 25) & 0x7F) << 5 | ((word >> 7) & 0x1F), 12)
    elif opcode == OP_BRANCH:
        imm = _sign_extend(
            (word >> 31) << 12 | ((word >> 7) & 1) << 11 | ((word >> 25) & 0x3F) << 5 | ((word >> 8) & 0xF) << 1,
            13,
        )
    elif opcode == OP_JAL:
        imm = _sign_extend(
            ((word >> 31) & 1) << 20
            | ((word >> 12) & 0xFF) << 12
            | ((word >> 20) & 1) << 11
            | ((word >> 21) & 0x3FF) << 1,
            21,
        )
    elif opcode in (OP_LUI, OP_AUIPC):
        imm = ((word >> 12) & _MASK20) << 12
    else:
        imm = 0
    return (1, opcode, rd, funct3, rs1, rs2, funct7, imm, word)


@lru_cache(maxsize=512)
def _decode_program(prog: tuple[int, ...]) -> tuple[tuple[int, int, int, int, int, int, int, int, int], ...]:
    return tuple(_decode_insn(int(w)) for w in prog)


def _vm_result_from_parts(
    success: bool,
    registers: tuple[tuple[int, int], ...],
    memory: tuple[tuple[int, int], ...],
    cycles: int,
    error: str | None,
    blocked_writes: int,
) -> dict[str, Any]:
    return {
        "success": success,
        "registers": dict(registers),
        "memory": dict(memory),
        "trace": [],
        "cycles": cycles,
        "error": error,
        "blocked_writes": blocked_writes,
    }


@lru_cache(maxsize=1024)
def _execute_vm_cached(
    program: tuple[int, ...],
    init_regs: tuple[tuple[int, int], ...],
    init_mem: tuple[tuple[int, int], ...],
    budget: int,
) -> tuple:
    result = _execute_vm(list(program), dict(init_regs), dict(init_mem), budget)
    return (
        bool(result["success"]),
        tuple(sorted((int(k), int(v)) for k, v in (result.get("registers") or {}).items())),
        tuple(sorted((int(k), int(v)) for k, v in (result.get("memory") or {}).items())),
        int(result["cycles"]),
        result.get("error"),
        int(result.get("blocked_writes") or 0),
    )


# ══════════════════════════════════════════════════════════════════════
#  Pure-Python RV32I VM
# ══════════════════════════════════════════════════════════════════════


def _execute_vm(
    program: list[int],
    init_regs: dict[int, int],
    init_mem: dict[int, int],
    budget: int,
) -> dict[str, Any]:
    """Execute a single RV32I program in pure Python.

    Wall time is recorded here, at the source (EE-W5): callers that reach
    the VM without going through ``VMFarm.evaluate`` / ``evaluate_batch``
    (terminal and scan-floor gate scoring) are then still accounted as
    ``vm_eval`` instead of the surrounding orchestration phase, and the
    re-entrancy guard keeps a call that is already inside a timed VMFarm
    batch counted exactly once.

    Note the coverage boundary: the batch/CuPy implementations
    (``_execute_vm_batch_numpy``, ``VMFactory``) are deliberately NOT
    wrapped here -- they are only accounted because every production batch
    entry point is timed at ``VMFarm.evaluate_batch``.  A future direct
    caller of those would book batch VM time as orchestration again.
    """
    with timed_vm_eval():
        return _execute_vm_inner(program, init_regs, init_mem, budget)


def _execute_vm_inner(
    program: list[int],
    init_regs: dict[int, int],
    init_mem: dict[int, int],
    budget: int,
) -> dict[str, Any]:
    """Execute a single RV32I program in pure Python.

    Returns dict with EvalResult-compatible keys:
        success, registers, memory, cycles, error, blocked_writes

    Performance optimizations vs. prior version:
      - Removed per-iteration regs[0]=0: all write paths already guard rd != 0.
      - Removed redundant word_offset < 0 check: pc is always >= 0 (starts at 0,
        masked with & _m32 on every update).
      - Removed redundant word & MASK32: programs are int lists, no masking needed.
      - Hoisted regs, memory, program references into local variables for inner loop.
      - Uses local function aliases for _sign_extend / _sign_extend32.
      - Uses local dict .get method alias for memory lookups.
      - Pre-computed _MASK12, _MASK16, _MASK20, _MASK8 as module-level constants.
    """
    # Register file: x0-x31, x0 is hardwired to 0
    regs = [0] * 32
    for r, v in init_regs.items():
        if 1 <= r <= 31:
            regs[r] = v & MASK32

    # Memory: byte-addressed dict
    memory: dict[int, int] = {}
    for addr, val in init_mem.items():
        memory[addr & MASK32] = val & MASK32

    pc = 0
    cycles = 0
    blocked_writes = 0
    error_msg: str | None = None

    # Cache locals for inner-loop speed (avoids repeated global lookups
    # on every iteration -- CPython resolves globals via dict lookup each time).
    _regs = regs
    _mem = memory
    _decoded = _decode_program(tuple(int(w) for w in program))
    _prog_len = len(_decoded)
    _m32 = MASK32
    _se = _sign_extend
    _se32 = _sign_extend32
    _mem_get = _mem.get

    stuck_window = 32
    pc_ring = [0] * stuck_window
    regs_mark: list[int] | None = None
    try:
        while cycles < budget:
            # Instruction fetch -- removed redundant word_offset < 0 check
            # (pc is always >= 0 since it starts at 0 and is masked with & _m32).
            word_offset = pc >> 2
            if word_offset >= _prog_len:
                error_msg = f"PC out of range: {pc:#x}"
                break

            kind, opcode, rd, funct3, rs1, rs2, funct7, imm, word = _decoded[word_offset]
            if kind == 0:
                break

            cycles += 1
            pc_ring[cycles % stuck_window] = word_offset
            if cycles == stuck_window:
                regs_mark = _regs[:]
            elif regs_mark is not None and cycles >= stuck_window * 2 and (cycles % stuck_window) == 0:
                nuniq = len(set(pc_ring))
                if nuniq <= 8 and _regs == regs_mark:
                    cycles = budget
                    break
                regs_mark = _regs[:]
            pc += 4

            v_rs1 = _regs[rs1]
            v_rs2 = _regs[rs2]

            if opcode == OP_OP_IMM:
                if funct3 == F3_ADD_SUB:
                    result = (v_rs1 + imm) & _m32
                elif funct3 == F3_SLL:
                    result = (v_rs1 << (imm & 0x1F)) & _m32
                elif funct3 == F3_SLT:
                    result = 1 if _se32(v_rs1) < imm else 0
                elif funct3 == F3_SLTU:
                    result = 1 if v_rs1 < (imm & _m32) else 0
                elif funct3 == F3_XOR:
                    result = (v_rs1 ^ imm) & _m32
                elif funct3 == F3_SRL_SRA:
                    shamt = imm & 0x1F
                    if word & 0x40000000:  # SRA
                        result = (_se32(v_rs1) >> shamt) & _m32
                    else:  # SRL
                        result = (v_rs1 >> shamt) & _m32
                elif funct3 == F3_OR:
                    result = (v_rs1 | imm) & _m32
                elif funct3 == F3_AND:
                    result = (v_rs1 & imm) & _m32
                else:
                    error_msg = f"Unknown OP_IMM funct3={funct3}"
                    break
                if rd != 0:
                    _regs[rd] = result

            elif opcode == OP_OP:
                if funct3 == F3_ADD_SUB:
                    if funct7 & 0x20:  # SUB
                        result = (v_rs1 - v_rs2) & _m32
                    else:  # ADD
                        result = (v_rs1 + v_rs2) & _m32
                elif funct3 == F3_SLL:
                    result = (v_rs1 << (v_rs2 & 0x1F)) & _m32
                elif funct3 == F3_SLT:
                    result = 1 if _se32(v_rs1) < _se32(v_rs2) else 0
                elif funct3 == F3_SLTU:
                    result = 1 if v_rs1 < v_rs2 else 0
                elif funct3 == F3_XOR:
                    result = (v_rs1 ^ v_rs2) & _m32
                elif funct3 == F3_SRL_SRA:
                    shamt = v_rs2 & 0x1F
                    if funct7 & 0x20:  # SRA
                        result = (_se32(v_rs1) >> shamt) & _m32
                    else:  # SRL
                        result = (v_rs1 >> shamt) & _m32
                elif funct3 == F3_OR:
                    result = (v_rs1 | v_rs2) & _m32
                elif funct3 == F3_AND:
                    result = (v_rs1 & v_rs2) & _m32
                else:
                    error_msg = f"Unknown OP funct3={funct3}"
                    break
                if rd != 0:
                    _regs[rd] = result

            elif opcode == OP_LOAD:
                addr = (v_rs1 + imm) & _m32
                val = _mem_get(addr, 0)
                if funct3 == F3_LW:
                    result = val
                elif funct3 == F3_LH:
                    result = _se(val & MASK16, 16)
                elif funct3 == F3_LB:
                    result = _se(val & MASK8, 8)
                elif funct3 == F3_LHU:
                    result = val & MASK16
                elif funct3 == F3_LBU:
                    result = val & MASK8
                else:
                    error_msg = f"Unknown LOAD funct3={funct3}"
                    break
                if rd != 0:
                    _regs[rd] = result & _m32

            elif opcode == OP_STORE:
                addr = (v_rs1 + imm) & _m32
                val = v_rs2 & _m32
                if funct3 == F3_SW:
                    _mem[addr] = val
                elif funct3 == F3_SH:
                    old = _mem_get(addr, 0) & _m32
                    _mem[addr] = (old & 0xFFFF0000) | (val & MASK16)
                elif funct3 == F3_SB:
                    old = _mem_get(addr, 0) & _m32
                    _mem[addr] = (old & 0xFFFFFF00) | (val & MASK8)
                else:
                    error_msg = f"Unknown STORE funct3={funct3}"
                    break

            elif opcode == OP_BRANCH:
                branch = False
                if funct3 == F3_BEQ:
                    branch = v_rs1 == v_rs2
                elif funct3 == F3_BNE:
                    branch = v_rs1 != v_rs2
                elif funct3 == F3_BLT:
                    branch = _se32(v_rs1) < _se32(v_rs2)
                elif funct3 == F3_BGE:
                    branch = _se32(v_rs1) >= _se32(v_rs2)
                elif funct3 == F3_BLTU:
                    branch = v_rs1 < v_rs2
                elif funct3 == F3_BGEU:
                    branch = v_rs1 >= v_rs2
                else:
                    error_msg = f"Unknown BRANCH funct3={funct3}"
                    break
                if branch:
                    pc = (pc - 4 + imm) & _m32

            elif opcode == OP_JAL:
                if rd != 0:
                    _regs[rd] = pc  # already incremented
                pc = (pc - 4 + imm) & _m32

            elif opcode == OP_JALR:
                if rd != 0:
                    _regs[rd] = pc
                pc = ((v_rs1 + imm) & ~1) & _m32

            elif opcode == OP_LUI:
                if rd != 0:
                    _regs[rd] = imm & _m32

            elif opcode == OP_AUIPC:
                if rd != 0:
                    _regs[rd] = (pc - 4 + imm) & _m32

            elif opcode == OP_ECALL:
                break

            elif opcode == OP_FENCE:
                pass  # NOP in single-threaded VM

            else:
                error_msg = f"Unknown opcode={opcode:#04x} at PC={pc-4:#x}"
                break

            # x0 is never written: all write paths already guard rd != 0.

    except (RuntimeError, ValueError, KeyError) as e:
        error_msg = f"VM exception: {e}"

    # Build register dict (only non-zero registers).
    # Dict comprehension with walrus operator avoids double array lookup.
    reg_dict: dict[int, int] = {i: v for i in range(1, 32) if (v := regs[i])}

    return {
        "success": error_msg is None,
        "registers": reg_dict,
        "memory": memory,
        "trace": [],
        "cycles": cycles,
        "error": error_msg,
        "blocked_writes": blocked_writes,
    }


def _vector_spin_cap(programs: list[list[int]], budget: int) -> int:
    """Numpy cycle cap from compact programs only.

    A long diversity mutant must not raise vectorized cycles for the whole
    batch. Leftover lanes still run the scalar VM at the full budget.
    """
    spin_len_cap = 96
    typical = 0
    for prog in programs:
        n = len(prog)
        if n <= spin_len_cap and n > typical:
            typical = n
    if typical <= 0:
        typical = spin_len_cap
    return min(int(budget), max(320, 5 * int(typical)))


def _freeze_vm_input(inp: dict[str, Any]) -> tuple:
    regs = inp.get("registers") or {}
    mem = inp.get("memory") or {}
    return (
        tuple(sorted((int(k), int(v)) for k, v in regs.items())),
        tuple(sorted((int(k), int(v)) for k, v in mem.items())),
    )


def _dedupe_vm_batch(
    programs: list[list[int]], inputs_list: list[dict[str, Any]]
) -> tuple[list[int], list[list[int]], list[dict[str, Any]]] | None:
    """Map each slot to a unique (program, input) index, or None if already unique."""
    first: dict[tuple, int] = {}
    remap: list[int] = []
    uniq_p: list[list[int]] = []
    uniq_i: list[dict[str, Any]] = []
    for i, prog in enumerate(programs):
        inp = inputs_list[i] if i < len(inputs_list) else {}
        key = (tuple(int(w) for w in prog), _freeze_vm_input(inp))
        slot = first.get(key)
        if slot is None:
            slot = len(uniq_p)
            first[key] = slot
            uniq_p.append(prog)
            uniq_i.append(inp)
        remap.append(slot)
    if len(uniq_p) == len(programs):
        return None
    return remap, uniq_p, uniq_i


def _scatter_unique_results(remap: list[int], unique_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for slot in remap:
        src = unique_results[slot]
        out.append(
            {
                **src,
                "registers": dict(src.get("registers") or {}),
                "memory": dict(src.get("memory") or {}),
            }
        )
    return out


def _execute_vm_batch_numpy(
    programs: list[list[int]],
    inputs_list: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    """Execute N RV32I programs simultaneously using NumPy vectorized operations.

    Returns list of result dicts with the same format as _execute_vm.
    Memory operations use per-program dicts since memory access is sparse.
    """
    N = len(programs)
    if N == 0:
        return []
    if N > 1:
        deduped = _dedupe_vm_batch(programs, inputs_list)
        if deduped is not None:
            remap, uniq_p, uniq_i = deduped
            return _scatter_unique_results(remap, _execute_vm_batch_numpy(uniq_p, uniq_i, budget))

    # Find max program length for array padding
    # Long programs skip the vector path and use leftover scalar at full budget
    # so they cannot widen prog_arr or raise numpy cost for compact lanes.
    vector_len_cap = 96
    compact_lens = [len(p) for p in programs if len(p) <= vector_len_cap]
    max_len = max(compact_lens) if compact_lens else 1
    long_lane = [len(p) > vector_len_cap for p in programs]

    # Pad programs into 2D array (N, max_len), using EBREAK for padding
    prog_arr = np.full((N, max_len), EBREAK, dtype=np.int64)
    for i, prog in enumerate(programs):
        if long_lane[i]:
            continue
        prog_arr[i, : len(prog)] = prog

    # Initialize register files: (N, 32) array
    regs = np.zeros((N, 32), dtype=np.int64)

    # Initialize per-program memory dicts
    memory_list: list[dict[int, int]] = []
    for inp in inputs_list:
        mem = {}
        for addr, val in inp.get("memory", {}).items():
            mem[addr & MASK32] = val & MASK32
        memory_list.append(mem)

    # Initialize registers from inputs
    for i, inp in enumerate(inputs_list):
        init_regs = inp.get("registers", {})
        for r, v in init_regs.items():
            if 1 <= r <= 31:
                regs[i, r] = v & MASK32

    # Per-program state arrays
    pcs = np.zeros(N, dtype=np.int64)
    active = np.ones(N, dtype=bool)  # True = still executing
    for i, is_long in enumerate(long_lane):
        if is_long:
            active[i] = False
    cycle_counts = np.zeros(N, dtype=np.int64)
    errors: list[str | None] = [None] * N

    # Mask for 32-bit operations
    M32 = MASK32

    def _se_np(val, bits):
        """Sign-extend numpy array from 'bits' width."""
        mask = (1 << bits) - 1
        val = val & mask
        return np.where(val & (1 << (bits - 1)), val - (1 << bits), val)

    def _wb(mask, results):
        """Write back results to registers for ops matching mask."""
        masked_idx = idx[mask]
        rd_m = rd[mask]
        targets = rd_m != 0
        if np.any(targets):
            regs[masked_idx[targets], rd_m[targets]] = results[targets] & M32

    # Vectorized stepping has high per-cycle Python overhead. Programs that
    # halt quickly (typical L30 straight-line) finish here. Remaining spinners
    # are finished with the scalar VM from scratch at the same budget so
    # traces match a full-budget run without dragging the whole batch.
    # Cap length used for spin_cap so one long diversity mutant cannot raise
    # numpy cycles for every lane (leftover scalar still uses full budget).
    spin_cap = _vector_spin_cap(programs, budget)
    stuck_window = 32
    pc_ring = np.zeros((stuck_window, N), dtype=np.int64)
    regs_mark = None
    for cycle in range(budget):
        # Check if any programs are still active
        if not np.any(active):
            break
        if cycle >= spin_cap:
            break

        # Get indices of active programs
        idx = np.where(active)[0]
        if len(idx) == 0:
            break

        # Batch fetch: get word offset for each active program
        word_offsets = pcs[idx] >> 2

        # Check for out-of-range PCs
        oor_mask = word_offsets >= prog_arr.shape[1]
        if np.any(oor_mask):
            for j, ii in enumerate(idx):
                if oor_mask[j]:
                    errors[ii] = f"PC out of range: {pcs[ii]:#x}"
                    active[ii] = False
            continue

        # Fetch instructions
        words = prog_arr[idx, word_offsets]

        # Check for EBREAK
        ebreak_mask = words == EBREAK
        if np.any(ebreak_mask):
            for j, ii in enumerate(idx):
                if ebreak_mask[j]:
                    active[ii] = False
            idx = idx[~ebreak_mask]
            words = words[~ebreak_mask]
            if len(idx) == 0:
                break

        # Update cycle counts for active programs
        cycle_counts[idx] += 1

        # Advance PCs
        pcs[idx] = (pcs[idx] + 4) & M32

        # Decode all instructions at once
        opcode = words & 0x7F
        rd = (words >> 7) & 0x1F
        funct3 = (words >> 12) & 0x07
        rs1 = (words >> 15) & 0x1F
        rs2 = (words >> 20) & 0x1F
        funct7 = (words >> 25) & 0x7F

        # Batch register read
        v_rs1 = regs[idx, rs1]
        v_rs2 = regs[idx, rs2]

        # ===== OP_OP_IMM =====
        mask_op_imm = opcode == OP_OP_IMM
        if np.any(mask_op_imm):
            m = mask_op_imm
            imm = _se_np((words[m] >> 20) & 0xFFF, 12)
            f3 = funct3[m]
            rs1_val = v_rs1[m]
            results = np.zeros(np.sum(m), dtype=np.int64)

            # ADDI
            addi_mask = f3 == F3_ADD_SUB
            results[addi_mask] = (rs1_val[addi_mask] + imm[addi_mask]) & M32

            # SLLI
            slli_mask = f3 == F3_SLL
            results[slli_mask] = (rs1_val[slli_mask] << (imm[slli_mask] & 0x1F)) & M32

            # SLTI
            slti_mask = f3 == F3_SLT
            rs1_s = np.where(rs1_val & 0x80000000, rs1_val - 0x100000000, rs1_val)
            results[slti_mask] = np.where(rs1_s[slti_mask] < imm[slti_mask], 1, 0)

            # SLTIU
            sltiu_mask = f3 == F3_SLTU
            imm_u = imm & M32
            results[sltiu_mask] = np.where(rs1_val[sltiu_mask] < imm_u[sltiu_mask], 1, 0)

            # XORI
            xori_mask = f3 == F3_XOR
            results[xori_mask] = (rs1_val[xori_mask] ^ imm[xori_mask]) & M32

            # SRLI/SRAI
            srai_mask = f3 == F3_SRL_SRA
            if np.any(srai_mask):
                shamt = imm[srai_mask] & 0x1F
                is_srai = (words[m][srai_mask] & 0x40000000) != 0
                rs1_s_srl = np.where(
                    rs1_val[srai_mask] & 0x80000000, rs1_val[srai_mask] - 0x100000000, rs1_val[srai_mask]
                )
                srl_result = (rs1_val[srai_mask] >> shamt) & M32
                sra_result = (rs1_s_srl >> shamt) & M32
                results[srai_mask] = np.where(is_srai, sra_result, srl_result)

            # ORI
            ori_mask = f3 == F3_OR
            results[ori_mask] = (rs1_val[ori_mask] | imm[ori_mask]) & M32

            # ANDI
            andi_mask = f3 == F3_AND
            results[andi_mask] = (rs1_val[andi_mask] & imm[andi_mask]) & M32

            # Write back
            _wb(m, results)

        # ===== OP_OP (R-type) =====
        mask_op = opcode == OP_OP
        if np.any(mask_op):
            m = mask_op
            f3 = funct3[m]
            rs1_val = v_rs1[m]
            rs2_val = v_rs2[m]
            results = np.zeros(np.sum(m), dtype=np.int64)

            # ADD/SUB
            addsub_mask = f3 == F3_ADD_SUB
            if np.any(addsub_mask):
                is_sub = (funct7[m][addsub_mask] & 0x20) != 0
                add_res = (rs1_val[addsub_mask] + rs2_val[addsub_mask]) & M32
                sub_res = (rs1_val[addsub_mask] - rs2_val[addsub_mask]) & M32
                results[addsub_mask] = np.where(is_sub, sub_res, add_res)

            # SLL
            sll_mask = f3 == F3_SLL
            results[sll_mask] = (rs1_val[sll_mask] << (rs2_val[sll_mask] & 0x1F)) & M32

            # SLT
            slt_mask = f3 == F3_SLT
            rs1_s = np.where(rs1_val & 0x80000000, rs1_val - 0x100000000, rs1_val)
            rs2_s = np.where(rs2_val & 0x80000000, rs2_val - 0x100000000, rs2_val)
            results[slt_mask] = np.where(rs1_s[slt_mask] < rs2_s[slt_mask], 1, 0)

            # SLTU
            sltu_mask = f3 == F3_SLTU
            results[sltu_mask] = np.where(rs1_val[sltu_mask] < rs2_val[sltu_mask], 1, 0)

            # XOR
            xor_mask = f3 == F3_XOR
            results[xor_mask] = (rs1_val[xor_mask] ^ rs2_val[xor_mask]) & M32

            # SRL/SRA
            sra_mask = f3 == F3_SRL_SRA
            if np.any(sra_mask):
                shamt = rs2_val[sra_mask] & 0x1F
                is_sra = (funct7[m][sra_mask] & 0x20) != 0
                rs1_s_srl = np.where(rs1_val[sra_mask] & 0x80000000, rs1_val[sra_mask] - 0x100000000, rs1_val[sra_mask])
                srl_result = (rs1_val[sra_mask] >> shamt) & M32
                sra_result = (rs1_s_srl >> shamt) & M32
                results[sra_mask] = np.where(is_sra, sra_result, srl_result)

            # OR
            or_mask = f3 == F3_OR
            results[or_mask] = (rs1_val[or_mask] | rs2_val[or_mask]) & M32

            # AND
            and_mask = f3 == F3_AND
            results[and_mask] = (rs1_val[and_mask] & rs2_val[and_mask]) & M32

            # Write back
            _wb(m, results)

        # ===== OP_LOAD =====
        mask_load = opcode == OP_LOAD
        if np.any(mask_load):
            m = mask_load
            imm = _se_np((words[m] >> 20) & 0xFFF, 12)
            addrs = (v_rs1[m] + imm) & M32
            f3 = funct3[m]
            results = np.zeros(np.sum(m), dtype=np.int64)

            # Per-program memory access (sparse, can't vectorize)
            masked_idx = idx[m]
            for j, ii in enumerate(masked_idx):
                addr = int(addrs[j])
                val = memory_list[ii].get(addr, 0)
                ft = int(f3[j])
                if ft == F3_LW:
                    results[j] = val
                elif ft == F3_LH:
                    results[j] = _sign_extend(val & MASK16, 16)
                elif ft == F3_LB:
                    results[j] = _sign_extend(val & MASK8, 8)
                elif ft == F3_LHU:
                    results[j] = val & MASK16
                elif ft == F3_LBU:
                    results[j] = val & MASK8
                else:
                    errors[ii] = f"Unknown LOAD funct3={ft}"
                    active[ii] = False

            # Write back
            _wb(m, results)

        # ===== OP_STORE =====
        mask_store = opcode == OP_STORE
        if np.any(mask_store):
            m = mask_store
            imm = _se_np(((words[m] >> 25) & 0x7F) << 5 | ((words[m] >> 7) & 0x1F), 12)
            addrs = (v_rs1[m] + imm) & M32
            f3 = funct3[m]

            # Per-program memory access
            masked_idx = idx[m]
            rs2_m = v_rs2[m]
            for j, ii in enumerate(masked_idx):
                addr = int(addrs[j])
                val = int(rs2_m[j]) & M32
                ft = int(f3[j])
                old = memory_list[ii].get(addr, 0) & M32
                if ft == F3_SW:
                    memory_list[ii][addr] = val
                elif ft == F3_SH:
                    memory_list[ii][addr] = (old & 0xFFFF0000) | (val & MASK16)
                elif ft == F3_SB:
                    memory_list[ii][addr] = (old & 0xFFFFFF00) & M32 | (val & MASK8)
                else:
                    errors[ii] = f"Unknown STORE funct3={ft}"
                    active[ii] = False

        # ===== OP_BRANCH =====
        mask_branch = opcode == OP_BRANCH
        if np.any(mask_branch):
            m = mask_branch
            imm = _se_np(
                (words[m] >> 31) << 12
                | ((words[m] >> 7) & 1) << 11
                | ((words[m] >> 25) & 0x3F) << 5
                | ((words[m] >> 8) & 0xF) << 1,
                13,
            )
            f3 = funct3[m]
            rs1_val = v_rs1[m]
            rs2_val = v_rs2[m]

            # Compute branch conditions
            branch_taken = np.zeros(np.sum(m), dtype=bool)

            # BEQ
            beq_mask = f3 == F3_BEQ
            branch_taken[beq_mask] = rs1_val[beq_mask] == rs2_val[beq_mask]

            # BNE
            bne_mask = f3 == F3_BNE
            branch_taken[bne_mask] = rs1_val[bne_mask] != rs2_val[bne_mask]

            # BLT
            blt_mask = f3 == F3_BLT
            rs1_s = np.where(rs1_val & 0x80000000, rs1_val - 0x100000000, rs1_val)
            rs2_s = np.where(rs2_val & 0x80000000, rs2_val - 0x100000000, rs2_val)
            branch_taken[blt_mask] = rs1_s[blt_mask] < rs2_s[blt_mask]

            # BGE
            bge_mask = f3 == F3_BGE
            branch_taken[bge_mask] = rs1_s[bge_mask] >= rs2_s[bge_mask]

            # BLTU
            bltu_mask = f3 == F3_BLTU
            branch_taken[bltu_mask] = rs1_val[bltu_mask] < rs2_val[bltu_mask]

            # BGEU
            bgeu_mask = f3 == F3_BGEU
            branch_taken[bgeu_mask] = rs1_val[bgeu_mask] >= rs2_val[bgeu_mask]

            # Update PCs for taken branches
            if np.any(branch_taken):
                old_pcs = pcs[idx[m]]  # PCs were already advanced by 4
                pcs[idx[m][branch_taken]] = (old_pcs[branch_taken] - 4 + imm[branch_taken]) & M32

        # ===== OP_JAL =====
        mask_jal = opcode == OP_JAL
        if np.any(mask_jal):
            m = mask_jal
            imm = _se_np(
                ((words[m] >> 31) & 1) << 20
                | ((words[m] >> 12) & 0xFF) << 12
                | ((words[m] >> 20) & 1) << 11
                | ((words[m] >> 21) & 0x3FF) << 1,
                21,
            )
            # Write return address
            targets = m & (rd != 0)
            if np.any(targets):
                regs[idx[targets], rd[targets]] = pcs[idx[targets]]  # Already incremented

            # Update PC
            pcs[idx[m]] = (pcs[idx[m]] - 4 + imm) & M32

        # ===== OP_JALR =====
        mask_jalr = opcode == OP_JALR
        if np.any(mask_jalr):
            m = mask_jalr
            imm = _se_np((words[m] >> 20) & 0xFFF, 12)
            # Write return address
            targets = m & (rd != 0)
            if np.any(targets):
                regs[idx[targets], rd[targets]] = pcs[idx[targets]]  # Already incremented

            # Update PC
            pcs[idx[m]] = ((v_rs1[m] + imm) & ~1) & M32

        # ===== OP_LUI =====
        mask_lui = opcode == OP_LUI
        if np.any(mask_lui):
            m = mask_lui
            imm = ((words[m] >> 12) & 0xFFFFF) << 12
            targets = m & (rd != 0)
            if np.any(targets):
                rd_m = rd[m]
                targets_actual = rd_m != 0
                if np.any(targets_actual):
                    regs[idx[m][targets_actual], rd_m[targets_actual]] = imm[targets_actual] & M32

        # ===== OP_AUIPC =====
        mask_auipc = opcode == OP_AUIPC
        if np.any(mask_auipc):
            m = mask_auipc
            imm = ((words[m] >> 12) & 0xFFFFF) << 12
            targets = m & (rd != 0)
            if np.any(targets):
                # PC was already advanced, need to subtract 4
                result = (pcs[idx[m]] - 4 + imm) & M32
                rd_m = rd[m]
                targets_actual = rd_m != 0
                if np.any(targets_actual):
                    regs[idx[m][targets_actual], rd_m[targets_actual]] = result[targets_actual]

        # ===== OP_ECALL =====
        mask_ecall = opcode == OP_ECALL
        if np.any(mask_ecall):
            for ii in idx[mask_ecall]:
                active[ii] = False

        # ===== OP_FENCE =====
        # NOP in single-threaded VM, no action needed

        # x0 is never written: all write paths already guard rd != 0

        # Deactivate programs that hit errors
        for ii in idx:
            if errors[ii] is not None:
                active[ii] = False

        pc_ring[cycle % stuck_window] = pcs
        if cycle == spin_cap - stuck_window:
            regs_mark = regs.copy()

    # Build results
    results_list = []
    leftover_cache: dict[tuple, dict] = {}
    for i in range(N):
        if bool(active[i]) or long_lane[i]:
            if bool(active[i]) and not long_lane[i]:
                nuniq = len({int(p) for p in pc_ring[:, i]})
                regs_frozen = regs_mark is not None and np.array_equal(regs[i], regs_mark[i])
                if nuniq <= 8 and regs_frozen:
                    # Tight control-flow spin with unchanged registers: same as a
                    # full-budget timeout. Loops that still write regs stay scalar.
                    reg_dict = {int(r): int(regs[i, r]) for r in range(1, 32) if regs[i, r] != 0}
                    results_list.append(
                        {
                            "success": errors[i] is None,
                            "registers": reg_dict,
                            "memory": memory_list[i],
                            "trace": [],
                            "cycles": int(budget),
                            "error": errors[i],
                            "blocked_writes": 0,
                        }
                    )
                    continue
            inp = inputs_list[i] if i < len(inputs_list) else {}
            frozen_regs = tuple(sorted((int(k), int(v)) for k, v in (inp.get("registers") or {}).items()))
            frozen_mem = tuple(sorted((int(k), int(v)) for k, v in (inp.get("memory") or {}).items()))
            key = (tuple(int(w) for w in programs[i]), frozen_regs, frozen_mem, int(budget))
            cached = leftover_cache.get(key)
            if cached is None:
                cached = _vm_result_from_parts(*_execute_vm_cached(*key))
                leftover_cache[key] = cached
            results_list.append(
                {
                    **cached,
                    "registers": dict(cached.get("registers") or {}),
                    "memory": dict(cached.get("memory") or {}),
                }
            )
            continue
        # Build register dict (only non-zero)
        reg_dict = {int(r): int(regs[i, r]) for r in range(1, 32) if regs[i, r] != 0}

        results_list.append(
            {
                "success": errors[i] is None,
                "registers": reg_dict,
                "memory": memory_list[i],
                "trace": [],
                "cycles": int(cycle_counts[i]),
                "error": errors[i],
                "blocked_writes": 0,
            }
        )

    return results_list


# ══════════════════════════════════════════════════════════════════════
#  GpuVMFarm — Batch evaluator matching the expected interface
# ══════════════════════════════════════════════════════════════════════


class GpuVMFarm:
    """Batch RV32I VM evaluator.

    Interface matches what evolution_engine.py expects:
        evaluate_batch(programs, inputs_list, budget) -> list[dict]
        evaluate_multi(programs, inputs_list, budget) -> list[dict]
    """

    def __init__(self, batch_size: int = 128, budget: int = 5000):
        self.batch_size = batch_size
        self.budget = budget
        _ensure_cupy()
        self._use_cupy = HAS_CUDA

    def evaluate_batch(
        self, programs: list[int], inputs_list: list[dict[str, Any]], budget: int | None = None
    ) -> list[dict[str, Any]]:
        """Evaluate a single program (or list) against inputs."""
        budget = budget or self.budget
        if isinstance(programs, list) and programs and isinstance(programs[0], int):
            # Single program case: programs is the program itself
            inputs = inputs_list[0] if inputs_list else {}
            return [_execute_vm(programs, inputs.get("registers", {}), inputs.get("memory", {}), budget)]
        # Batch case
        if len(programs) <= 1:
            return [
                _execute_vm(prog, inp.get("registers", {}), inp.get("memory", {}), budget)
                for prog, inp in zip(programs, inputs_list, strict=False)
            ]
        return _execute_vm_batch_numpy(programs, inputs_list, budget)

    def evaluate_multi(
        self, programs: list[list[int]], inputs_list: list[dict[str, Any]], budget: int | None = None
    ) -> list[dict[str, Any]]:
        """Evaluate a batch of programs. Main entry point for GPU-accelerated eval."""
        budget = budget or self.budget
        if len(programs) <= 1:
            return [
                _execute_vm(prog, inp.get("registers", {}), inp.get("memory", {}), budget)
                for prog, inp in zip(programs, inputs_list, strict=False)
            ]
        if self._use_cupy and cp is not None:
            try:
                return self._evaluate_batch_cupy(programs, inputs_list, budget)
            except (RuntimeError, ValueError, TypeError):
                pass
        # NumPy vectorized path (always available)
        return _execute_vm_batch_numpy(programs, inputs_list, budget)

    def _evaluate_batch_cupy(
        self, programs: list[list[int]], inputs_list: list[dict[str, Any]], budget: int
    ) -> list[dict[str, Any]]:
        """Batch evaluation using NumPy vectorized execution.

        Falls back to per-program Python for very small batches where
        NumPy overhead exceeds the per-program loop cost.
        """
        if len(programs) <= 1:
            return [
                _execute_vm(prog, inp.get("registers", {}), inp.get("memory", {}), budget)
                for prog, inp in zip(programs, inputs_list, strict=False)
            ]
        return _execute_vm_batch_numpy(programs, inputs_list, budget)


# ══════════════════════════════════════════════════════════════════════
#  VMFactory — Unified factory for best available backend
# ══════════════════════════════════════════════════════════════════════


class VMFactory:
    """Factory that returns the best available VM backend.

    Resolution order:
      1. GPU (GpuVMFarm) when CUDA is available and VRAM has capacity
      2. CPU (GpuVMFarm in pure-Python mode) as fallback

    Usage:
        farm = VMFactory(use_gpu=True)
        farm.evaluate_batch(programs, inputs_list, budget)
        farm.cleanup()  # clear GPU memory after batch
    """

    # VRAM utilization threshold before falling back to CPU
    VRAM_CAPACITY_THRESHOLD: float = 0.90

    # Minimum free VRAM in MB required for GPU evaluation
    MIN_FREE_VRAM_MB: float = 512.0

    def __init__(
        self,
        use_gpu: bool = True,
        batch_size: int = 128,
        budget: int = 5000,
        vram_threshold: float | None = None,
    ):
        """
        Args:
            use_gpu: Request GPU acceleration (may fall back to CPU).
            batch_size: Batch size for GPU evaluation.
            budget: Default instruction budget per evaluation.
            vram_threshold: Override VRAM capacity threshold (0.0–1.0).
        """
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.budget = budget
        self._vram_threshold = vram_threshold or self.VRAM_CAPACITY_THRESHOLD
        self._backend: GpuVMFarm | None = None
        self._backend_type: str = "uninitialized"
        self._gpu_available: bool = False
        self._fallback_reason: str = ""

        self._initialize()

    def _initialize(self) -> None:
        """Resolve the best available backend."""
        if not self.use_gpu:
            self._init_cpu()
            return

        # Check CUDA availability
        _ensure_cupy()
        if not HAS_CUDA:
            # Also check PyTorch CUDA
            try:
                import torch

                torch_available = torch.cuda.is_available()
            except (ImportError, RuntimeError):
                torch_available = False

            if not torch_available:
                self._fallback_reason = "No CUDA runtime available (CuPy + PyTorch)"
                self._init_cpu()
                return

        # Check VRAM capacity before allocating
        capacity = check_gpu_capacity(self._vram_threshold)
        if not capacity["has_capacity"]:
            self._fallback_reason = capacity["reason"]
            self._init_cpu()
            return

        # Initialize GPU backend
        try:
            self._backend = GpuVMFarm(batch_size=self.batch_size, budget=self.budget)
            self._gpu_available = True
            self._backend_type = "gpu"
        except (RuntimeError, ValueError, OSError) as e:
            self._fallback_reason = f"GPU init failed: {e}"
            self._init_cpu()

    def _init_cpu(self) -> None:
        """Initialize CPU-only backend."""
        self._backend = GpuVMFarm(batch_size=self.batch_size, budget=self.budget)
        self._backend._use_cupy = False  # Force pure Python
        self._gpu_available = False
        self._backend_type = "cpu"

    @property
    def backend_type(self) -> str:
        """Returns 'gpu', 'cpu', or 'uninitialized'."""
        return self._backend_type

    @property
    def is_gpu(self) -> bool:
        """True when using GPU acceleration."""
        return self._gpu_available and self._backend_type == "gpu"

    @property
    def fallback_reason(self) -> str:
        """Reason for CPU fallback (empty if GPU is active)."""
        return self._fallback_reason

    def evaluate_batch(
        self, programs: list[list[int]], inputs_list: list[dict[str, Any]], budget: int | None = None
    ) -> list[dict[str, Any]]:
        """Evaluate a batch of programs using the resolved backend.

        If GPU is active and VRAM drops below capacity mid-evaluation,
        automatically falls back to CPU for subsequent calls.
        """
        budget = budget or self.budget

        if self._backend is None:
            raise RuntimeError("VMFactory not initialized")

        # Pre-batch VRAM check (skip if CPU)
        if self._gpu_available:
            capacity = check_gpu_capacity(self._vram_threshold)
            if not capacity["has_capacity"]:
                self._fallback_to_cpu(capacity["reason"])

        try:
            return self._backend.evaluate_multi(programs, inputs_list, budget=budget)
        except Exception as e:
            if self._gpu_available:
                self._fallback_to_cpu(f"GPU eval failed: {e}")
                return self._backend.evaluate_multi(programs, inputs_list, budget=budget)
            raise

    def evaluate_single(self, program: list[int], inputs: dict[str, Any], budget: int | None = None) -> dict[str, Any]:
        """Evaluate a single program. Convenience wrapper."""
        budget = budget or self.budget
        results = self.evaluate_batch([program], [inputs], budget=budget)
        return results[0] if results else {}

    def _fallback_to_cpu(self, reason: str) -> None:
        """Transition from GPU to CPU backend."""
        self._fallback_reason = reason
        self._init_cpu()

    def get_status(self) -> dict[str, Any]:
        """Return factory status for monitoring."""
        status: dict[str, Any] = {
            "backend": self._backend_type,
            "gpu_requested": self.use_gpu,
            "gpu_active": self._gpu_available,
            "fallback_reason": self._fallback_reason,
        }
        if self._gpu_available:
            vram = get_gpu_vram_info()
            status["vram"] = vram
        return status

    def cleanup(self) -> dict[str, Any]:
        """Clear GPU memory and return cleanup stats."""
        return clear_gpu_memory()

    def __repr__(self) -> str:
        return f"<VMFactory backend={self._backend_type} gpu={self._gpu_available}>"
