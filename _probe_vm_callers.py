"""Attribute _execute_vm calls to their call sites during the two rebuilds."""
from __future__ import annotations

import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import gpu_vm_farm  # noqa: E402

_counts: dict[str, int] = {}
_orig = gpu_vm_farm._execute_vm


def _wrapper(program, init_regs, init_mem, budget):
    f = sys._getframe(1)
    key = f"{f.f_code.co_name}@{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"
    _counts[key] = _counts.get(key, 0) + 1
    return _orig(program, init_regs, init_mem, budget)


gpu_vm_farm._execute_vm = _wrapper  # type: ignore[assignment]


def load_champion() -> list[int]:
    with open("multi_cycle_results.jsonl", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("cycle") == 7595:
                return [
                    int(x, 16) if isinstance(x, str) else int(x)
                    for x in rec["champion_program"]
                ]
    raise SystemExit("champion not found")


def report(label: str, before: dict) -> None:
    delta = {k: v - before.get(k, 0) for k, v in _counts.items()}
    delta = {k: v for k, v in delta.items() if v > 0}
    total = sum(delta.values())
    print(f"\n--- {label}: {total} _execute_vm calls ---", flush=True)
    for k, v in sorted(delta.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {v:9d}  {k}", flush=True)


def main() -> None:
    champ = load_champion()
    print(f"champion words={len(champ)}", flush=True)

    from evolution.mutation_helpers import collect_scan_floor_growth_crossover_fills
    from evolution.terminal_stagnation import select_terminal_probes_for_batch

    snap = dict(_counts)
    t0 = time.perf_counter()
    collect_scan_floor_growth_crossover_fills(list(champ))
    print(f"growth fills: {time.perf_counter() - t0:.2f}s", flush=True)
    report("growth fills", snap)

    snap = dict(_counts)
    t0 = time.perf_counter()
    select_terminal_probes_for_batch(list(champ), 30, random.Random(7596), 12, pop_size=50)
    print(f"probe pool: {time.perf_counter() - t0:.2f}s", flush=True)
    report("probe pool", snap)


if __name__ == "__main__":
    main()
