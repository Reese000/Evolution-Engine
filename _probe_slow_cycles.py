"""Parse multi_cycle_results.jsonl: per-cycle elapsed + phase breakdown.

Usage: python _probe_slow_cycles.py [first_cycle] [last_cycle]
"""
from __future__ import annotations

import json
import sys

PATH = "multi_cycle_results.jsonl"

KEYS = [
    "elapsed_seconds",
    "phase_evolution_seconds",
    "phase_retrain_seconds",
    "gen_phase_generations",
    "gen_phase_seconds",
    "gen_phase_vm_eval_seconds",
    "gen_phase_mutation_propagation_seconds",
    "gen_phase_selection_archives_seconds",
    "gen_phase_diagnostics_logging_seconds",
    "gen_phase_unattributed_seconds",
    "gen_phase_run_init_seconds",
    "gen_phase_run_finalize_seconds",
    "vm_gpu_batches",
    "generations_completed",
]


def main() -> None:
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    rows = []
    with open(PATH, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = rec.get("cycle")
            if not isinstance(c, int) or not (lo <= c <= hi):
                continue
            rows.append(rec)
    # keep last record per cycle (retries append)
    by_cycle: dict[int, dict] = {}
    for r in rows:
        by_cycle[r["cycle"]] = r

    hdr = (
        "cyc  elapsed  evol  retrain  gens  gphase  vm  mutprop  seldiag  diag  unattr "
        "init  fin  batches"
    )
    print(hdr)
    for c in sorted(by_cycle):
        r = by_cycle[c]

        def g(k):
            v = r.get(k)
            return v if isinstance(v, (int, float)) else 0

        print(
            f"{c:5d} {g('elapsed_seconds'):7.1f} {g('phase_evolution_seconds'):6.1f} "
            f"{g('phase_retrain_seconds'):7.1f} {g('gen_phase_generations'):5.0f} "
            f"{g('gen_phase_seconds'):7.1f} {g('gen_phase_vm_eval_seconds'):5.1f} "
            f"{g('gen_phase_mutation_propagation_seconds'):8.1f} "
            f"{g('gen_phase_selection_archives_seconds'):7.1f} "
            f"{g('gen_phase_diagnostics_logging_seconds'):5.1f} "
            f"{g('gen_phase_unattributed_seconds'):6.1f} "
            f"{g('gen_phase_run_init_seconds'):5.1f} "
            f"{g('gen_phase_run_finalize_seconds'):4.1f} "
            f"{g('vm_gpu_batches'):7.0f}"
        )


if __name__ == "__main__":
    main()
