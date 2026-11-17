"""N consecutive probe-pool builds at a configurable memo-cache size.

EE_PROBE_CACHE=1024 reproduces production behaviour; a large value shows the
memoization head-room.  Each build uses a different rng seed (stand-in for a
different cycle).
"""
from __future__ import annotations

import functools
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
SIZE = int(os.environ.get("EE_PROBE_CACHE", "1024"))
N = int(os.environ.get("EE_PROBE_N", "4"))


def load_champion() -> list[int]:
    with open(RESULTS, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("cycle") == 7595:
                prog = rec["champion_program"]
                return [int(x, 16) if isinstance(x, str) else int(x) for x in prog]
    raise SystemExit("champion not found")


def set_cache_size() -> None:
    mod = __import__("evolution.terminal_stagnation", fromlist=["*"])
    fn = getattr(mod, "_curriculum_verify_mean_cached")
    if hasattr(fn, "__wrapped__"):
        mod._curriculum_verify_mean_cached = functools.lru_cache(maxsize=SIZE)(fn.__wrapped__)


def measure(label: str, fn) -> None:
    _, c0 = vm_eval_totals()
    s0 = vm_eval_totals()[0]
    t0 = time.perf_counter()
    try:
        out = fn()
        note = f" len={len(out) if hasattr(out, '__len__') else '?'}"
    except Exception as exc:  # noqa: BLE001
        note, out = f"  ERR {type(exc).__name__}: {exc}", None
    wall = time.perf_counter() - t0
    s1, c1 = vm_eval_totals()
    print(
        f"{label:44s} wall={wall:8.2f}s vm_calls={c1 - c0:9d} vm_s={s1 - s0:8.2f}{note}",
        flush=True,
    )


def main() -> None:
    champ = load_champion()
    print(f"champion words={len(champ)} cache_size={SIZE} builds={N}", flush=True)
    set_cache_size()
    from evolution.terminal_stagnation import select_terminal_probes_for_batch
    from evolution.mutation_helpers import collect_scan_floor_growth_crossover_fills

    for i in range(N):
        seed = 7596 + i
        measure(
            f"probe_pool build #{i + 1} (seed {seed})",
            lambda s=seed: select_terminal_probes_for_batch(
                list(champ), 30, random.Random(s), 12, pop_size=50
            ),
        )
        measure(
            f"growth fills #{i + 1} (cycle {7596 + i})",
            lambda c=7596 + i: collect_scan_floor_growth_crossover_fills(
                list(champ), cycle_no=c
            )
            if "cycle_no" in collect_scan_floor_growth_crossover_fills.__code__.co_varnames
            else collect_scan_floor_growth_crossover_fills(list(champ)),
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
