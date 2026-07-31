#!/usr/bin/env python3
"""Verify all curriculum reference solutions L0-L16 through the RV32I VM."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_vm_farm import _execute_vm, EBREAK
from milestone1.rv32i_mask import encode
from tasks_rv32i import (
    Level0, Level1, Level2, Level3, Level4, Level5, Level6, Level7,
    Level8, Level9, Level10, Level11, Level12, Level13, Level14,
    Level15, Level16,
)

LEVEL_CLASSES = [
    Level0, Level1, Level2, Level3, Level4, Level5, Level6, Level7,
    Level8, Level9, Level10, Level11, Level12, Level13, Level14,
    Level15, Level16,
]

# Maps level -> registers that verify() checks
VERIFY_REGS = {
    0: [3],
    1: [3, 4],
    2: [3, 4, 5],
    3: [3, 4, 5, 6],
    4: [3, 4, 5, 6, 7],
    5: [3, 4, 5, 6, 7, 8],
    6: [3, 4, 5, 6, 7, 8, 9],
    7: [3, 4, 5, 6, 7, 8, 9, 10],
    8: [3, 4, 5, 6, 7, 8, 9, 10, 11],
    9: [3, 4, 5, 6, 7, 8, 9, 10, 12],
    10: [3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
    11: [3, 4, 5, 6, 7, 8, 9, 14],
    12: [3, 4, 5, 6, 7, 8, 9, 15],
    13: [3, 4, 5, 6, 7, 8, 9, 16],
    14: [3, 4, 5, 6, 7, 8, 9, 17],
    15: [3, 4, 5, 6, 7, 8, 9, 18],
    16: [3, 4, 5, 6, 7, 8, 9, 19],
}


def encode_program(instructions, add_ebreak=True):
    """Encode a list of instruction dicts to a list of 32-bit words."""
    words = []
    for inst in instructions:
        name = inst["name"]
        if name == "EBREAK":
            words.append(EBREAK)
            continue
        w = encode(
            name=name,
            rd=inst.get("rd", 0),
            rs1=inst.get("rs1", 0),
            rs2=inst.get("rs2", 0),
            imm=inst.get("imm", 0),
            validate=True,
        )
        words.append(w)
    if add_ebreak and (not words or words[-1] != EBREAK):
        words.append(EBREAK)
    return words


def run_level(level_cls, seed=42):
    """Run a single level's reference solution through the VM and verify."""
    level_num = int(level_cls.__name__.replace("Level", ""))
    inst_obj = level_cls(seed=seed)
    inst = inst_obj.generate_instance()
    ref = level_cls.reference_solution()

    init_regs = inst["inputs"]["registers"]
    init_mem = inst["inputs"]["memory"]
    expected_regs = inst["expected_output"]["registers"]
    verify_regs = VERIFY_REGS[level_num]

    # Encode program (always append EBREAK to avoid PC-out-of-range)
    words = encode_program(ref, add_ebreak=True)

    budget = max(500, len(words) * 200)

    result = _execute_vm(
        program=words,
        init_regs=init_regs,
        init_mem=init_mem,
        budget=budget,
    )

    # Check only the registers that verify() cares about
    mismatches = {}
    for reg in verify_regs:
        exp_val = expected_regs.get(reg, 0)
        actual_val = result["registers"].get(reg, 0)
        if actual_val != exp_val:
            mismatches[reg] = {"expected": exp_val, "actual": actual_val}

    passed = result["success"] and len(mismatches) == 0

    # For self-modifying levels (L12+), also note if JALR failed
    jalr_failed = False
    if not result["success"] and level_num >= 12:
        jalr_failed = "PC out of range" in (result.get("error") or "")

    return {
        "level": level_num,
        "pass": passed,
        "vm_success": result["success"],
        "error": result.get("error"),
        "cycles": result["cycles"],
        "expected": {r: expected_regs.get(r, 0) for r in verify_regs},
        "actual": {r: result["registers"].get(r, 0) for r in verify_regs},
        "mismatches": mismatches,
        "jalr_failed": jalr_failed,
    }


def main():
    print("=" * 72)
    print("  EE VM VERIFICATION: Curriculum Reference Solutions L0-L16")
    print("=" * 72)

    results = []

    for cls in LEVEL_CLASSES:
        level_num = int(cls.__name__.replace("Level", ""))
        seed = 42 + level_num
        try:
            r = run_level(cls, seed=seed)
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                "level": level_num,
                "pass": False,
                "vm_success": False,
                "error": str(e),
                "cycles": 0,
                "mismatches": {},
                "jalr_failed": False,
            })

    # Print summary table
    pass_count = sum(1 for r in results if r["pass"])
    fail_count = len(results) - pass_count

    print(f"\n{'Level':>5} | {'Status':>6} | {'VM OK':>5} | {'Cycles':>6} | Notes")
    print("-" * 72)
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        vm_ok = "Yes" if r.get("vm_success") else "No"
        notes = []
        if r["pass"]:
            out_reg = 3 + r["level"]
            val = r["actual"].get(out_reg, "N/A")
            notes.append(f"x{out_reg}={val}")
        else:
            if r.get("error"):
                err = r["error"]
                if len(err) > 50:
                    err = err[:50] + "..."
                notes.append(err)
            if r.get("jalr_failed"):
                notes.append("[JALR to scratch mem - VM can't exec data mem]")
            if r.get("mismatches"):
                notes.append(f"MISMATCH: {list(r['mismatches'].keys())}")
        print(f"  L{r['level']:<2} | {status:>6} | {vm_ok:>5} | {r['cycles']:>6} | {'; '.join(notes)}")

    print("-" * 72)
    print(f"\n  TOTAL: {pass_count}/{len(results)} PASS, {fail_count} FAIL\n")

    # Detailed failure analysis
    failures = [r for r in results if not r["pass"]]
    if failures:
        print("FAILURE ANALYSIS:")
        for r in failures:
            print(f"\n  L{r['level']}:")
            if r.get("error"):
                print(f"    VM error: {r['error']}")
            if r.get("jalr_failed"):
                print(f"    -> Self-modifying code: JALR targets data memory")
                print(f"       which is not part of the VM's instruction array.")
                print(f"       This is a known limitation for L12/L13.")
            if r.get("mismatches"):
                print(f"    Register mismatches:")
                for reg, m in r["mismatches"].items():
                    print(f"      x{reg}: expected={m['expected']}, actual={m['actual']}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
