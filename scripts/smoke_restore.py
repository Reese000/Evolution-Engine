#!/usr/bin/env python3
"""Smoke test restore; writes results to .agents/smoke_restore.txt"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("EE_USE_GPU", "0")

OUT = os.path.join(ROOT, ".agents", "smoke_restore.txt")


def main() -> int:
    from evolution_engine import EvolutionConfig, EvolutionEngine

    with open(os.path.join(ROOT, "multi_cycle_results.jsonl"), encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    bad = json.loads(lines[-1])
    bad_prog = [int(x, 16) if isinstance(x, str) else int(x) for x in bad["champion_program"]]

    eng = EvolutionEngine(EvolutionConfig(curriculum_level=30, max_generations=1))
    eng.champion_level = 30
    eng.champion = bad_prog
    eng._refresh_task_instance()
    eng._refresh_eval_budget()
    eng.champion_trace = eng._evaluate(bad_prog)
    eng.champion_fitness = eng._compute_fitness(bad_prog, eng.champion_trace)
    _, vs = eng._verify_champion()
    eng._champion_verify_score = float(vs)

    before = (len(bad_prog), float(eng.champion_fitness), float(vs))
    ok = eng._maybe_restore_best_certified_champion()
    after = (
        len(eng.champion),
        float(eng.champion_fitness),
        float(getattr(eng, "_champion_verify_score", 0.0)),
    )

    summary = {"before": before, "restore_ok": ok, "after": after}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary))
    return 0 if ok and after[0] == 73 else 1


if __name__ == "__main__":
    raise SystemExit(main())
