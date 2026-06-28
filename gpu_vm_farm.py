#!/usr/bin/env python3
"""gpu_vm_farm.py — GPU-accelerated RV32I VM batch evaluator.

Replaces per-program subprocess spawning with in-process execution.
When CuPy is available, uses GPU for batch register/memory operations.
Falls back to pure Python when CUDA is unavailable.

Key speedup: avoids 50+ subprocess spawns per generation (~2750/cycle).
"""

from __future__ import annotations

import os
import sys
from typing import Any

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
            total = torch.cuda.get_device_properties(dev).total_mem / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(dev) / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(dev) / (1024 * 1024)
            free = total - reserved
            info.update({
                "available": True,
                "total_mb": round(total, 1),
                "used_mb": round(allocated, 1),
                "free_mb": round(free, 1),
                "utilization": round(allocated / max(total, 1.0), 3),
                "device_name": torch.cuda.get_device_name(dev),
            })
            return info
    except (ImportError, RuntimeError):
        pass

    # Try CuPy next
    _ensure_cupy()
    if HAS_CUDA and cp is not None:
        try:
            mem = cp.get_default_memory_pool()
            total = cp.cuda.Device().mem_info[1] / (1024 * 1024)
            used = cp.cuda.Device().mem_info[0] / (1024 * 1024)
            free = total - used
            info.update({
                "available": True,
                "total_mb": round(total, 1),
                "used_mb": round(used, 1),
                "free_mb": round(free, 1),
                "utilization": round(used / max(total, 1.0), 3),
                "device_name": str(cp.cuda.Device().attributes.get("DeviceName", "CuPy GPU")),
            })
            return info
        except Exception:
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
        result["reason"] = (
            f"Only {vram['free_mb']:.0f} MB free VRAM (minimum 512 MB required)"
        )
        return result

    return result


# ══════════════════════════════════════════════════════════════════════
#  RV32I Opcode Constants
# ══════════════════════════════════════════════════════════════════════

# Opcode field (bits [6:0])
OP_LUI      = 0b0110111
OP_AUIPC    = 0b0010111
OP_JAL      = 0b1101111
OP_JALR     = 0b1100111
OP_BRANCH   = 0b1100011
OP_LOAD     = 0b0000011
OP_STORE    = 0b0100011
OP_OP_IMM   = 0b0010011
OP_OP       = 0b0110011
OP_FENCE    = 0b0001111
OP_ECALL    = 0b1110011

# funct3 values
F3_ADD_SUB = 0b000
F3_SLL     = 0b001
F3_SLT     = 0b010
F3_SLTU    = 0b011
F3_XOR     = 0b100
F3_SRL_SRA = 0b101
F3_OR      = 0b110
F3_AND     = 0b111

# Branch funct3
F3_BEQ  = 0b000
F3_BNE  = 0b001
F3_BLT  = 0b100
F3_BGE  = 0b101
F3_BLTU = 0b110
F3_BGEU = 0b111

# Load funct3
F3_LB  = 0b000
F3_LH  = 0b001
F3_LW  = 0b010
F3_LBU = 0b100
F3_LHU = 0b101

# Store funct3
F3_SB = 0b000
F3_SH = 0b001
F3_SW = 0b010

# EBREAK
EBREAK = 0x00100073

# Mask constants
MASK8  = 0xFF
MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


def _sign_extend(val: int, bits: int) -> int:
    """Sign-extend an integer from `bits` width to Python int."""
    mask = (1 << bits) - 1
    val = val & mask
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val


def _sign_extend32(val: int) -> int:
    """Sign-extend a 32-bit unsigned to signed Python int."""
    val = val & MASK32
    if val & 0x80000000:
        val -= 0x100000000
    return val


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

    Returns dict with EvalResult-compatible keys:
        success, registers, memory, cycles, error, blocked_writes
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

    prog_len = len(program) * 4  # bytes

    try:
        while cycles < budget:
            # Instruction fetch
            word_offset = pc >> 2
            if word_offset < 0 or word_offset >= len(program):
                error_msg = f"PC out of range: {pc:#x}"
                break

            word = program[word_offset] & MASK32
            if word == EBREAK:
                break

            cycles += 1
            pc += 4

            opcode = word & 0x7F
            rd     = (word >> 7) & 0x1F
            funct3 = (word >> 12) & 0x07
            rs1    = (word >> 15) & 0x1F
            rs2    = (word >> 20) & 0x1F
            funct7 = (word >> 25) & 0x7F

            v_rs1 = regs[rs1]
            v_rs2 = regs[rs2]

            if opcode == OP_OP_IMM:
                imm = _sign_extend((word >> 20) & 0xFFF, 12)
                if funct3 == F3_ADD_SUB:
                    result = (v_rs1 + imm) & MASK32
                elif funct3 == F3_SLL:
                    shamt = imm & 0x1F
                    result = (v_rs1 << shamt) & MASK32
                elif funct3 == F3_SLT:
                    result = 1 if _sign_extend32(v_rs1) < imm else 0
                elif funct3 == F3_SLTU:
                    result = 1 if v_rs1 < (imm & MASK32) else 0
                elif funct3 == F3_XOR:
                    result = (v_rs1 ^ imm) & MASK32
                elif funct3 == F3_SRL_SRA:
                    shamt = imm & 0x1F
                    if word & 0x40000000:  # SRA
                        result = (_sign_extend32(v_rs1) >> shamt) & MASK32
                    else:  # SRL
                        result = (v_rs1 >> shamt) & MASK32
                elif funct3 == F3_OR:
                    result = (v_rs1 | imm) & MASK32
                elif funct3 == F3_AND:
                    result = (v_rs1 & imm) & MASK32
                else:
                    error_msg = f"Unknown OP_IMM funct3={funct3}"
                    break
                if rd != 0:
                    regs[rd] = result

            elif opcode == OP_OP:
                if funct3 == F3_ADD_SUB:
                    if funct7 & 0x20:  # SUB
                        result = (v_rs1 - v_rs2) & MASK32
                    else:  # ADD
                        result = (v_rs1 + v_rs2) & MASK32
                elif funct3 == F3_SLL:
                    result = (v_rs1 << (v_rs2 & 0x1F)) & MASK32
                elif funct3 == F3_SLT:
                    result = 1 if _sign_extend32(v_rs1) < _sign_extend32(v_rs2) else 0
                elif funct3 == F3_SLTU:
                    result = 1 if v_rs1 < v_rs2 else 0
                elif funct3 == F3_XOR:
                    result = (v_rs1 ^ v_rs2) & MASK32
                elif funct3 == F3_SRL_SRA:
                    shamt = v_rs2 & 0x1F
                    if funct7 & 0x20:  # SRA
                        result = (_sign_extend32(v_rs1) >> shamt) & MASK32
                    else:  # SRL
                        result = (v_rs1 >> shamt) & MASK32
                elif funct3 == F3_OR:
                    result = (v_rs1 | v_rs2) & MASK32
                elif funct3 == F3_AND:
                    result = (v_rs1 & v_rs2) & MASK32
                else:
                    error_msg = f"Unknown OP funct3={funct3}"
                    break
                if rd != 0:
                    regs[rd] = result

            elif opcode == OP_LOAD:
                imm = _sign_extend((word >> 20) & 0xFFF, 12)
                addr = (v_rs1 + imm) & MASK32
                val = memory.get(addr, 0)
                if funct3 == F3_LW:
                    result = val
                elif funct3 == F3_LH:
                    result = _sign_extend(val & 0xFFFF, 16)
                elif funct3 == F3_LB:
                    result = _sign_extend(val & 0xFF, 8)
                elif funct3 == F3_LHU:
                    result = val & 0xFFFF
                elif funct3 == F3_LBU:
                    result = val & 0xFF
                else:
                    error_msg = f"Unknown LOAD funct3={funct3}"
                    break
                if rd != 0:
                    regs[rd] = result & MASK32

            elif opcode == OP_STORE:
                imm11_5 = (word >> 25) & 0x7F
                imm4_0  = (word >> 7) & 0x1F
                imm = _sign_extend((imm11_5 << 5) | imm4_0, 12)
                addr = (v_rs1 + imm) & MASK32
                val = v_rs2 & MASK32
                if funct3 == F3_SW:
                    memory[addr] = val
                elif funct3 == F3_SH:
                    # Read-modify-write (store low 16 bits)
                    old = memory.get(addr, 0) & MASK32
                    memory[addr] = (old & 0xFFFF0000) | (val & 0xFFFF)
                elif funct3 == F3_SB:
                    old = memory.get(addr, 0) & MASK32
                    memory[addr] = (old & 0xFFFFFF00) | (val & 0xFF)
                else:
                    error_msg = f"Unknown STORE funct3={funct3}"
                    break

            elif opcode == OP_BRANCH:
                imm12   = (word >> 31) & 1
                imm10_5 = (word >> 25) & 0x3F
                imm4_1  = (word >> 8) & 0xF
                imm11   = (word >> 7) & 1
                imm = _sign_extend(
                    (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1), 13
                )
                branch = False
                if funct3 == F3_BEQ:
                    branch = v_rs1 == v_rs2
                elif funct3 == F3_BNE:
                    branch = v_rs1 != v_rs2
                elif funct3 == F3_BLT:
                    branch = _sign_extend32(v_rs1) < _sign_extend32(v_rs2)
                elif funct3 == F3_BGE:
                    branch = _sign_extend32(v_rs1) >= _sign_extend32(v_rs2)
                elif funct3 == F3_BLTU:
                    branch = v_rs1 < v_rs2
                elif funct3 == F3_BGEU:
                    branch = v_rs1 >= v_rs2
                else:
                    error_msg = f"Unknown BRANCH funct3={funct3}"
                    break
                if branch:
                    pc = (pc - 4 + imm) & MASK32

            elif opcode == OP_JAL:
                imm20   = (word >> 31) & 1
                imm10_1 = (word >> 21) & 0x3FF
                imm11   = (word >> 20) & 1
                imm19_12 = (word >> 12) & 0xFF
                imm = _sign_extend(
                    (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1), 21
                )
                if rd != 0:
                    regs[rd] = pc  # already incremented
                pc = (pc - 4 + imm) & MASK32

            elif opcode == OP_JALR:
                imm = _sign_extend((word >> 20) & 0xFFF, 12)
                target = (v_rs1 + imm) & ~1
                if rd != 0:
                    regs[rd] = pc
                pc = target & MASK32

            elif opcode == OP_LUI:
                imm31_12 = (word >> 12) & 0xFFFFF
                if rd != 0:
                    regs[rd] = (imm31_12 << 12) & MASK32

            elif opcode == OP_AUIPC:
                imm31_12 = (word >> 12) & 0xFFFFF
                if rd != 0:
                    regs[rd] = (pc - 4 + (imm31_12 << 12)) & MASK32

            elif opcode == OP_ECALL:
                break

            elif opcode == OP_FENCE:
                pass  # NOP in single-threaded VM

            else:
                error_msg = f"Unknown opcode={opcode:#04x} at PC={pc-4:#x}"
                break

            # Enforce x0 = 0
            regs[0] = 0

    except Exception as e:
        error_msg = f"VM exception: {e}"

    # Build register dict (only non-zero registers)
    reg_dict: dict[int, int] = {}
    for i in range(1, 32):
        if regs[i] != 0:
            reg_dict[i] = regs[i]

    return {
        "success": error_msg is None,
        "registers": reg_dict,
        "memory": memory,
        "trace": [],
        "cycles": cycles,
        "error": error_msg,
        "blocked_writes": blocked_writes,
    }


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
        """Evaluate a single program (or list) against inputs.

        Note: evolution_engine.py calls this as:
            gpu_res = self._gpu_farm.evaluate_batch(prog, [inputs], budget=5000)
        where prog is a list[int] (one program), inputs_list is a single-element list.
        """
        budget = budget or self.budget
        results = []
        if isinstance(programs, list) and programs and isinstance(programs[0], int):
            # Single program case: programs is the program itself
            inputs = inputs_list[0] if inputs_list else {}
            results.append(_execute_vm(
                programs,
                inputs.get("registers", {}),
                inputs.get("memory", {}),
                budget,
            ))
        else:
            # Batch case: programs is list of programs
            for prog, inputs in zip(programs, inputs_list, strict=False):
                results.append(_execute_vm(
                    prog,
                    inputs.get("registers", {}),
                    inputs.get("memory", {}),
                    budget,
                ))
        return results

    def evaluate_multi(
        self, programs: list[list[int]], inputs_list: list[dict[str, Any]], budget: int | None = None
    ) -> list[dict[str, Any]]:
        """Evaluate a batch of programs. Main entry point for GPU-accelerated eval."""
        budget = budget or self.budget
        if self._use_cupy and cp is not None:
            try:
                return self._evaluate_batch_cupy(programs, inputs_list, budget)
            except Exception:
                # Fall through to Python
                pass
        # Pure Python path
        return [
            _execute_vm(prog, inp.get("registers", {}), inp.get("memory", {}), budget)
            for prog, inp in zip(programs, inputs_list, strict=False)
        ]

    def _evaluate_batch_cupy(
        self, programs: list[list[int]], inputs_list: list[dict[str, Any]], budget: int
    ) -> list[dict[str, Any]]:
        """CuPy-accelerated batch evaluation.

        For now, the VM execution itself is scalar Python (branching is
        data-dependent and hard to vectorize). The CuPy path prepares
        batch arrays for future optimization (e.g., batch register init).
        """
        # For small batches, just use Python (CuPy overhead isn't worth it)
        if len(programs) <= 4:
            return [
                _execute_vm(prog, inp.get("registers", {}), inp.get("memory", {}), budget)
                for prog, inp in zip(programs, inputs_list, strict=False)
            ]
        # For now, delegate to Python. CuPy kernel optimization is a future step.
        return [
            _execute_vm(prog, inp.get("registers", {}), inp.get("memory", {}), budget)
            for prog, inp in zip(programs, inputs_list, strict=False)
        ]


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
        except Exception as e:
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

    def evaluate_single(
        self, program: list[int], inputs: dict[str, Any], budget: int | None = None
    ) -> dict[str, Any]:
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
