"""Does enlarging the curriculum-score memo caches remove the gen0 cost?

Same measurements as _probe_hotspot_offline.py, but with the per-program
scoring LRU caches re-created with a much larger capacity (pure memoization:
identical values, identical gates -- only the cache size changes).
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
    want = 7595
    with open(RESULTS, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("cycle") == want:
                prog = rec["champion_program"]
                return [int(x, 16) if isinstance(x, str) else int(x) for x in prog]
    raise SystemExit("champion not found")


def enlarge_caches() -> list[str]:
    patched = []
    for mod_name, names in (
        (
            "evolution.terminal_stagnation",
            [
                "_curriculum_verify_mean_cached",
                "_heldout_l30_verify_mean_cached",
            ],
        ),
        ("evolution.mutation_helpers", ["_curriculum_l10_cached"]),
        ("evolution.fitness_helpers", ["_compact_prom_raw_cached"]),
    ):
        mod = __import__(mod_name, fromlist=["*"])
        for name in names:
            fn = getattr(mod, name, None)
            if fn is None or not hasattr(fn, "cache_info"):
                continue
            if not hasattr(fn, "__wrapped__"):
                continue
            setattr(
                mod,
                name,
                functools.lru_cache(maxsize=BIG)(fn.__wrapped__),
            )
            patched.append(f"{mod_name}.{name}")
    return patched


def measure(label: str, fn) -> None:
    _, c0 = vm_eval_totals()
    s0 = vm_eval_totals()[0]
    t0 = time.perf_counter()
    try:
        out = fn()
        note = "" if out is None else f" len={len(out) if hasattr(out, '__len__') else '?'}"
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
    print(f"champion words={len(champ)}  big_cache={BIG}", flush=True)
    patched = enlarge_caches()
    print("patched caches:", ", ".join(patched), flush=True)
    rng = random.Random(7596)

    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = get_scan_floor_gate(list(champ))
    from evolution.terminal_stagnation import select_terminal_probes_for_batch
    from evolution.mutation_helpers import collect_scan_floor_growth_crossover_fills

    measure(
        "select_terminal_probes_for_batch (warm caches)",
        lambda: select_terminal_probes_for_batch(list(champ), 30, rng, 12, pop_size=50),
    )
    measure(
        "select_terminal_probes_for_batch (2nd call)",
        lambda: select_terminal_probes_for_batch(list(champ), 30, rng, 12, pop_size=50),
    )
    measure(
        "collect_scan_floor_growth_crossover_fills (warm)",
        lambda: collect_scan_floor_growth_crossover_fills(list(champ)),
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
