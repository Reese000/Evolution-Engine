"""Cross-cycle reuse test: does scoring repeat when inputs/offsets change?

Three back-to-back builds of the terminal probe pool with different rng
seeds (stand-ins for consecutive cycles) under enlarged memo caches.
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
BIG = int(os.environ.get("EE_PROBE_BIG_CACHE", "1000000"))


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


def enlarge_caches() -> None:
    for mod_name, names in (
        (
            "evolution.terminal_stagnation",
            ["_curriculum_verify_mean_cached", "_heldout_l30_verify_mean_cached"],
        ),
        ("evolution.mutation_helpers", ["_curriculum_l10_cached"]),
        ("evolution.fitness_helpers", ["_compact_prom_raw_cached"]),
    ):
        mod = __import__(mod_name, fromlist=["*"])
        for name in names:
            fn = getattr(mod, name, None)
            if fn is None or not hasattr(fn, "cache_info") or not hasattr(fn, "__wrapped__"):
                continue
            setattr(mod, name, functools.lru_cache(maxsize=BIG)(fn.__wrapped__))


def measure(label: str, fn) -> None:
    _, c0 = vm_eval_totals()
    s0 = vm_eval_totals()[0]
    t0 = time.perf_counter()
    try:
        out = fn()
        note = f" len={len(out) if hasattr(out, '__len__') else '?'}"
        err = ""
    except Exception as exc:  # noqa: BLE001
        note, err = "", f"  ERR {type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    s1, c1 = vm_eval_totals()
    print(
        f"{label:46s} wall={wall:8.2f}s vm_calls={c1 - c0:9d} vm_s={s1 - s0:8.2f}{note}{err}",
        flush=True,
    )


def main() -> None:
    champ = load_champion()
    print(f"champion words={len(champ)} big_cache={BIG}", flush=True)
    enlarge_caches()
    from evolution.terminal_stagnation import select_terminal_probes_for_batch, terminal_cycles_at_level
    from evolution.mutation_helpers import collect_scan_floor_growth_crossover_fills

    print("terminal_cycles_at_level =", terminal_cycles_at_level(), flush=True)
    for i, seed in enumerate((7596, 7597, 7598), start=1):
        measure(
            f"probe_pool build #{i} (rng seed {seed})",
            lambda s=seed: select_terminal_probes_for_batch(
                list(champ), 30, random.Random(s), 12, pop_size=50
            ),
        )
    measure(
        "growth crossover fills (2nd)",
        lambda: collect_scan_floor_growth_crossover_fills(list(champ)),
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
