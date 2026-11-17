"""cProfile the probe-pool rebuild after the memo fix (top functions + lines)."""
from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import random
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


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


def main() -> None:
    champ = load_champion()
    from evolution.terminal_stagnation import select_terminal_probes_for_batch

    # warm pass so the profiled pass is the steady-state cost
    select_terminal_probes_for_batch(list(champ), 30, random.Random(1), 12, pop_size=50)

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    select_terminal_probes_for_batch(list(champ), 30, random.Random(7596), 12, pop_size=50)
    prof.disable()
    wall = time.perf_counter() - t0
    print(f"wall={wall:.2f}s", flush=True)
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf).sort_stats("cumulative")
    stats.print_stats(28)
    print(buf.getvalue(), flush=True)
    buf2 = io.StringIO()
    stats2 = pstats.Stats(prof, stream=buf2).sort_stats("tottime")
    stats2.print_stats(22)
    print(buf2.getvalue(), flush=True)


if __name__ == "__main__":
    main()
