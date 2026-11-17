#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# fitness_helpers: add selection_fitness before reporting_champion_fitness
fh = (ROOT / "evolution" / "fitness_helpers.py").read_text(encoding="utf-8")
marker = "def reporting_champion_fitness("
insert = '''def selection_fitness(
    engine,
    program: list[int],
    trace: dict,
    *,
    verify_score: float | None = None,
) -> float:
    """Fitness for champion promotion when verify is saturated at terminal depth."""
    verify_fn = getattr(engine, "_verify_score_from_trace", None)
    verify = float(verify_score if verify_score is not None else (verify_fn(trace) if callable(verify_fn) else 0.0))
    champ_verify = float(getattr(engine, "_champion_verify_score", 0.0))
    if max(verify, champ_verify) >= 0.99:
        return reporting_champion_fitness(engine, program, trace, max(verify, champ_verify))
    return float(compute_fitness(engine, program, trace))


'''
if "def selection_fitness(" not in fh:
    if marker not in fh:
        raise SystemExit("reporting_champion_fitness not found")
    fh = fh.replace(marker, insert + marker, 1)
    (ROOT / "evolution" / "fitness_helpers.py").write_text(fh, encoding="utf-8")
    print("fitness_helpers: added selection_fitness")
else:
    print("fitness_helpers: selection_fitness already present")

# selection_mixin: use selection_fitness in key places
sm = (ROOT / "evolution" / "selection_mixin.py").read_text(encoding="utf-8")
replacements = [
    (
        "            fitness = self._compute_fitness(prog, trace)\n            if not candidate_beats_champion_on_verify(",
        "            from evolution.fitness_helpers import selection_fitness\n\n            fitness = selection_fitness(self, prog, trace)\n            if not candidate_beats_champion_on_verify(",
    ),
    (
        "            fitness = _compute(prog, trace)\n\n            rank = fitness",
        "            from evolution.fitness_helpers import selection_fitness\n\n            fitness = selection_fitness(self, prog, trace)\n\n            rank = fitness",
    ),
    (
        "            self.champion_fitness = float(self._compute_fitness(self.champion, fresh_trace))",
        "            from evolution.fitness_helpers import selection_fitness\n\n            self.champion_fitness = float(selection_fitness(self, self.champion, fresh_trace))",
    ),
]
for old, new in replacements:
    if old in sm and new not in sm:
        sm = sm.replace(old, new, 1)
        print("selection_mixin: patched block")
    elif new in sm:
        print("selection_mixin: block already patched")
    else:
        print("selection_mixin: WARN block not found")
(ROOT / "evolution" / "selection_mixin.py").write_text(sm, encoding="utf-8")

# evolution_engine: _should_replace_champion use selection fitness at saturation
ee = (ROOT / "evolution_engine.py").read_text(encoding="utf-8")
old_should = """        if candidate_fitness <= self.champion_fitness:

            return False

        delta = candidate_fitness - self.champion_fitness"""
new_should = """        champ_verify = float(getattr(self, "_champion_verify_score", 0.0))
        if champ_verify >= 0.99 and self.champion is not None and self.champion_trace is not None:
            from evolution.fitness_helpers import selection_fitness

            champion_fitness = float(selection_fitness(self, self.champion, self.champion_trace))
        else:
            champion_fitness = float(self.champion_fitness)

        if candidate_fitness <= champion_fitness:

            return False

        delta = candidate_fitness - champion_fitness"""
if new_should not in ee:
    if old_should not in ee:
        raise SystemExit("_should_replace_champion block not found")
    ee = ee.replace(old_should, new_should, 1)
    (ROOT / "evolution_engine.py").write_text(ee, encoding="utf-8")
    print("evolution_engine: patched _should_replace_champion")
else:
    print("evolution_engine: already patched")

# deploy tests
dr = (ROOT / "scripts" / "_deploy_run.py").read_text(encoding="utf-8")
if "test_reporting_champion_fitness" not in dr:
    dr = dr.replace(
        '"tests/unit/test_clean_depth_escape.py", "-q"], "pytest")',
        '"tests/unit/test_clean_depth_escape.py",\n'
        '     "tests/unit/test_reporting_champion_fitness.py",\n'
        '     "tests/unit/test_partial_verify_fitness.py", "-q"], "pytest")',
        1,
    )
    (ROOT / "scripts" / "_deploy_run.py").write_text(dr, encoding="utf-8")
    print("_deploy_run: added tests")

# progress_trajectory: auto-backfill hook on rebuild
pt = (ROOT / "scripts" / "progress_trajectory.py").read_text(encoding="utf-8")
hook = '''def _maybe_correct_stale_fitness_rows(rows: list[dict[str, Any]]) -> None:
    """Fix logged fitness for verified champions when jsonl predates reporting fix."""
    try:
        from scripts.backfill_champion_fitness import recompute_fitness
    except ImportError:
        return
    for row in rows:
        verify = float(row.get("champion_verify_score") or 0.0)
        fit = float(row.get("final_fitness") or 0.0)
        if verify >= 1.0 - 1e-6 and fit < 200.0 and row.get("champion_program"):
            try:
                live = recompute_fitness(row["champion_program"], int(row.get("final_level") or 0))
                if live > fit + 1.0:
                    row["final_fitness"] = live
            except (RuntimeError, TypeError, ValueError, AttributeError, KeyError):
                pass


'''
if "_maybe_correct_stale_fitness_rows" not in pt:
    pt = pt.replace(
        "def _build_trajectory_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:",
        hook + "def _build_trajectory_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:",
        1,
    )
    pt = pt.replace(
        "    rows = _dedupe_rows_by_cycle(rows)",
        "    _maybe_correct_stale_fitness_rows(rows)\n    rows = _dedupe_rows_by_cycle(rows)",
        1,
    )
    (ROOT / "scripts" / "progress_trajectory.py").write_text(pt, encoding="utf-8")
    print("progress_trajectory: added stale fitness correction")
else:
    print("progress_trajectory: hook already present")

print("done")
