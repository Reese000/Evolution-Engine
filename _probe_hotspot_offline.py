"""Offline hotspot probe: measure VM calls + wall time per scan-floor probe helper.

Runs in a fresh process (cold caches, like generation 0 of a cycle).
Read-only with respect to project state (only reads the results JSONL for the
champion).  Usage: python _probe_hotspot_offline.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from evolution.phase_timing import vm_eval_totals  # noqa: E402

RESULTS = os.path.join(ROOT, "multi_cycle_results.jsonl")


def load_champion() -> list[int]:
    best = None
    with open(RESULTS, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("cycle") in (7595, 7594, 7593):
                best = rec
                if rec["cycle"] == 7595:
                    break
    prog = best["champion_program"]
    return [int(x, 16) if isinstance(x, str) else int(x) for x in prog]


def measure(label: str, fn) -> None:
    c0, = [vm_eval_totals()[1]]
    s0 = vm_eval_totals()[0]
    t0 = time.perf_counter()
    try:
        out = fn()
        note = "" if out is None else f" -> {type(out).__name__} len={len(out) if hasattr(out, '__len__') else '?'}"
        err = ""
    except Exception as exc:  # noqa: BLE001
        note, err = "", f"  ERR {type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    s1, c1 = vm_eval_totals()
    print(
        f"{label:52s} wall={wall:8.2f}s vm_calls={c1 - c0:9d} vm_s={s1 - s0:8.2f}{note}{err}",
        flush=True,
    )


def main() -> None:
    champ = load_champion()
    print(f"champion words={len(champ)}", flush=True)
    rng = random.Random(7596)
    cycle_no = 7596

    from evolution.terminal_stagnation import get_scan_floor_gate

    measure("get_scan_floor_gate", lambda: get_scan_floor_gate(list(champ)))
    gate = get_scan_floor_gate(list(champ))

    from evolution.terminal_stagnation import (
        scan_floor_compression_hot,
        select_terminal_probes_for_batch,
        terminal_cycles_at_level,
        terminal_growth_material_locus_probe_variants,
        terminal_wide_imm_perturb_probe_variants,
        terminal_wide_imm_sweep_probe_variants,
    )

    measure("terminal_cycles_at_level", lambda: terminal_cycles_at_level())
    measure("scan_floor_compression_hot", lambda: scan_floor_compression_hot(list(champ), rng))
    measure(
        "terminal_wide_imm_sweep_probe_variants(max_evals=96)",
        lambda: terminal_wide_imm_sweep_probe_variants(
            list(champ), max_evals=96, start_offset=0, gate=gate
        ),
    )
    measure(
        "terminal_wide_imm_perturb_probe_variants",
        lambda: terminal_wide_imm_perturb_probe_variants(
            list(champ), random.Random(1), max_evals=96, max_attempts=2000, gate=gate
        ),
    )
    measure(
        "terminal_growth_material_locus_probe_variants",
        lambda: terminal_growth_material_locus_probe_variants(list(champ), list(champ)),
    )

    from evolution.terminal_stagnation import ranked_scan_floor_compression_variants

    measure(
        "ranked_scan_floor_compression_variants",
        lambda: ranked_scan_floor_compression_variants(list(champ), cycle_no=cycle_no),
    )

    from evolution.mutation_helpers import collect_scan_floor_growth_crossover_fills

    measure(
        "collect_scan_floor_growth_crossover_fills",
        lambda: collect_scan_floor_growth_crossover_fills(list(champ)),
    )

    measure(
        "select_terminal_probes_for_batch(no parents)",
        lambda: select_terminal_probes_for_batch(
            list(champ), 30, rng, 12, pop_size=50
        ),
    )

    from evolution_engine import EvolutionEngine  # noqa: F401  (import cost only)

    print("done", flush=True)


if __name__ == "__main__":
    main()
