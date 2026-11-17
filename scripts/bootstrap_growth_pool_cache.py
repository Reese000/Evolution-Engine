#!/usr/bin/env python3
"""Bootstrap .agents/scan_floor_growth_pool_cache.json from disk-cached growth ranks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    seeds_path = ROOT / ".agents" / "scan_floor_eq_refine_seeds.json"
    if not seeds_path.is_file():
        print("No eq-refine seeds file; nothing to bootstrap")
        return 1
    payload = json.loads(seeds_path.read_text(encoding="utf-8"))
    champ = list(payload.get("anchor") or [])
    if not champ:
        print("Empty anchor in eq-refine seeds")
        return 1

    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.mutation_helpers import (
        collect_scan_floor_growth_crossover_fills,
        load_scan_floor_eq_refine_seeds,
        save_scan_floor_growth_pool_cache,
        scan_floor_growth_inject_rank,
    )
    from evolution.terminal_stagnation import (
        get_scan_floor_gate,
        ranked_scan_floor_growth_fills,
        scan_floor_min_raw_gain,
    )

    gate = get_scan_floor_gate(champ)
    champ_pr = float(compact_l30_promotion_raw_fitness(champ))
    min_gain = float(scan_floor_min_raw_gain(len(champ)))
    pool: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def _add(prog: list[int]) -> None:
        key = tuple(int(x) for x in prog)
        if key in seen:
            return
        seen.add(key)
        pool.append([int(x) for x in prog])

    for prog in ranked_scan_floor_growth_fills(champ):
        _add(list(prog))
    for prog in collect_scan_floor_growth_crossover_fills(champ) or []:
        _add(list(prog))
    for prog in load_scan_floor_eq_refine_seeds(champ, max_seeds=8, probe_material=False):
        _add(list(prog))
    for prog in payload.get("seeds") or []:
        if isinstance(prog, list):
            _add([int(x) for x in prog])

    pool.sort(
        key=lambda p: scan_floor_growth_inject_rank(
            champ,
            p,
            gate=gate,
            champ_pr=champ_pr,
            min_gain=min_gain,
            stalled=True,
        ),
        reverse=True,
    )
    if not pool:
        print("No pool candidates assembled")
        return 1

    cycle = int(payload.get("cycle") or 0)
    save_scan_floor_growth_pool_cache(champ, pool, cross_keys=set(), cycle=cycle)
    out = ROOT / ".agents" / "scan_floor_growth_pool_cache.json"
    print(f"Wrote {len(pool)} programs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
