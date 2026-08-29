#!/usr/bin/env python3
"""vm_farm.py — Thin shim providing EvalResult and _run_one_evaluation.

This module re-exports EvalResult and _run_one_evaluation so that
evolution_engine.py can import them directly without depending on the
milestone0/vm-farm directory layout at import time.
"""

from __future__ import annotations

from collections import namedtuple

# ──────────────────────────────────────────────────────────────────────
#  EvalResult namedtuple
# ──────────────────────────────────────────────────────────────────────

EvalResult = namedtuple(
    "EvalResult",
    ["success", "registers", "memory", "trace", "cycles", "error", "blocked_writes"],
)

# Provide defaults so EvalResult can be constructed with keyword args only.
EvalResult.__new__.__defaults__ = (False, {}, {}, [], 0, None, [])


# ──────────────────────────────────────────────────────────────────────
#  Delegation to gpu_vm_farm._execute_vm
# ──────────────────────────────────────────────────────────────────────

def _run_one_evaluation(
    kernel_path: str,
    program: bytes,
    budget: int,
    init_regs: dict[int, int] | None = None,
    init_mem: dict[int, int] | None = None,
) -> EvalResult:
    """Execute a single RV32I program via the GPU VM farm.

    Parameters
    ----------
    kernel_path : str
        Path to the safety-kernel RV32I binary.
    program : bytes
        The raw RV32I machine code to evaluate.
    budget : int
        Maximum number of cycles to execute.
    init_regs : dict, optional
        Initial register state {reg_num: value}.
    init_mem : dict, optional
        Initial memory state {address: value}.

    Returns
    -------
    EvalResult
        A namedtuple containing success, registers, memory, trace,
        cycles, error, and blocked_writes.
    """
    from gpu_vm_farm import _execute_vm  # local import to avoid circular deps

    # _execute_vm is the core evaluator in gpu_vm_farm; delegate directly.
    # Note: kernel_path is ignored -- the VM is now pure Python.
    result = _execute_vm(
        program=program,
        budget=budget,
        init_regs=init_regs or {},
        init_mem=init_mem or {},
    )

    # Normalise the result into an EvalResult namedtuple.
    if isinstance(result, EvalResult):
        return result

    # If _execute_vm returns a plain tuple/dict, wrap it.
    if isinstance(result, dict):
        return EvalResult(
            success=result.get("success", False),
            registers=result.get("registers", {}),
            memory=result.get("memory", {}),
            trace=result.get("trace", []),
            cycles=result.get("cycles", 0),
            error=result.get("error", None),
            blocked_writes=result.get("blocked_writes", []),
        )

    # Last-resort: positional unpacking (must match field order).
    return EvalResult(*result)
