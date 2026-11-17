"""EE-W3d regression: engine auto-record must not crowd L10-lift seeds out of cap=8.

`record_scan_floor_eq_refine_seeds` merges the prior payload with the incoming
record and keeps the top ``EE_SCAN_FLOOR_EQ_REFINE_SEED_CAP`` (8) candidates
ranked by `scan_floor_eq_inject_sort_key`, which is PROM-first at the scan
floor.  Same-prom plateau ties (prom 325.8257, L10 0.800 = champion) therefore
outrank every L10-lift seed (prom 318.8319, L10 0.825), and once the merged
pool holds >=7 distinct ties no lift seed survives the cap:
`warm_eq_refine_disk_seeds_sufficient` then reports the payload COLD and
`load_scan_floor_eq_refine_seeds` falls through to the heavy projection path.

Measured threshold (pre-fix, HEAD 740bb2d): exactly 6 ties + 2 lift still
passes the warm check (the 2 reserved-by-luck lift seeds survive), so the
failing case needs >=7 distinct ties in the merged pool.
"""

import json
import os

import pytest

os.environ.setdefault("EE_USE_GPU", "0")


_CHAMP = list(range(1, 27)) + [0x00100073]
_TIE_PROM = 325.8257
_LIFT_PROM = 318.8319
_CHAMP_L10 = 0.800
_LIFT_L10 = 0.825


def _ties(n: int) -> list[list[int]]:
    out = []
    for i in range(n):
        prog = list(_CHAMP)
        prog[i] = 900 + i
        out.append(prog)
    return out


def _lifts(n: int) -> list[list[int]]:
    out = []
    for i in range(n):
        prog = list(_CHAMP)
        prog[10 + i] = 800 + i
        out.append(prog)
    return out


@pytest.fixture
def rank_env(monkeypatch, tmp_path):
    """Production-shaped PROM-first ranking isolated inside tmp_path."""
    from evolution.mutation_helpers import _curriculum_l10_cycle_cache

    _curriculum_l10_cycle_cache.clear()
    agents = tmp_path / ".agents"
    agents.mkdir(exist_ok=True)
    payload_path = agents / "scan_floor_eq_refine_seeds.json"
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_eq_refine_seeds_path",
        lambda: str(payload_path),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda _c: list(_CHAMP),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    lift_keys = {tuple(p) for p in _lifts(8)}

    def _prom(prog):
        return _LIFT_PROM if tuple(prog) in lift_keys else _TIE_PROM

    def _l10(prog, level=10):
        return _LIFT_L10 if tuple(prog) in lift_keys else _CHAMP_L10

    monkeypatch.setattr("evolution.fitness_helpers.compact_l30_promotion_raw_fitness", _prom)
    monkeypatch.setattr("evolution.terminal_stagnation.curriculum_verify_mean", _l10)
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _n: True,
    )
    # Keep load()'s fallbacks hermetic: no heavy projection, and no bootstrap
    # lift-seed injection either — a COLD payload must stay visibly cold so the
    # load assertions cannot be rescued by recomputed seeds (and so a test run
    # in the shared checkout can never write the production bootstrap cache).
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_growth_material_projection_seeds",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_l10_lift_eq_refine_bootstrap_seeds",
        lambda *a, **k: [],
    )
    yield payload_path
    _curriculum_l10_cycle_cache.clear()


def _split(seeds, ties, lifts):
    tie_t = {tuple(p) for p in ties}
    lift_t = {tuple(p) for p in lifts}
    n_ties = sum(1 for s in seeds if tuple(s) in tie_t)
    n_lifts = sum(1 for s in seeds if tuple(s) in lift_t)
    return n_ties, n_lifts


def test_engine_tie_rerecord_keeps_warm_invariant(rank_env):
    """AC1/AC2: a payload seeded with 6 ties + 2 lift stays warm after the
    engine auto-records >=6 same-prom ties onto it."""
    from evolution.mutation_helpers import (
        load_scan_floor_eq_refine_seeds,
        record_scan_floor_eq_refine_seeds,
        warm_eq_refine_disk_seeds_sufficient,
    )

    ties = _ties(7)
    lifts = _lifts(8)

    # Card-style re-warm: 6 same-prom ties + 2 L10-lift seeds on disk.
    record_scan_floor_eq_refine_seeds(_CHAMP, ties[:6] + lifts[:2], cycle=1)
    payload = json.loads(rank_env.read_text())
    n_ties, n_lifts = _split(payload["seeds"], ties, lifts)
    assert (n_ties, n_lifts) == (6, 2), "seed state was %s" % ((n_ties, n_lifts),)

    # External run_multi_cycle.py auto-record: its eq pool is all ties.
    record_scan_floor_eq_refine_seeds(_CHAMP, ties, cycle=7596)

    payload = json.loads(rank_env.read_text())
    stored = payload["seeds"]
    n_ties, n_lifts = _split(stored, ties, lifts)
    assert len(stored) == 8
    assert (
        n_lifts >= 2
    ), "engine tie re-record left only %d L10-lift seeds " "(%d ties + %d lift) -> payload is COLD" % (
        n_lifts,
        n_ties,
        n_lifts,
    )
    assert warm_eq_refine_disk_seeds_sufficient(_CHAMP, stored, seed_cap=8)

    loaded = load_scan_floor_eq_refine_seeds(_CHAMP, max_seeds=8)
    assert len(loaded) == 8
    _l_ties, loaded_lifts = _split(loaded, ties, lifts)
    assert loaded_lifts >= 2, "load returned only %d L10-lift seeds" % (loaded_lifts,)
    assert warm_eq_refine_disk_seeds_sufficient(_CHAMP, loaded, seed_cap=8)


def test_six_ties_plus_lift_rerecord_stays_warm_characterization(rank_env):
    """The card's literal AC2 recipe (record lift seeds onto 6 ties + 2 lift).

    This is the MARGINAL case: exactly 6 ties let 2 lift seeds survive even on
    pre-fix code, so it passes before and after.  Kept as a characterization of
    the 6-vs-7 threshold; `test_engine_tie_rerecord_keeps_warm_invariant` is
    the case that actually fails pre-fix.
    """
    from evolution.mutation_helpers import record_scan_floor_eq_refine_seeds, warm_eq_refine_disk_seeds_sufficient

    ties = _ties(7)
    lifts = _lifts(8)
    record_scan_floor_eq_refine_seeds(_CHAMP, ties[:6] + lifts[:2], cycle=1)
    record_scan_floor_eq_refine_seeds(_CHAMP, lifts, cycle=2)

    payload = json.loads(rank_env.read_text())
    n_ties, n_lifts = _split(payload["seeds"], ties, lifts)
    assert len(payload["seeds"]) == 8
    assert (n_ties, n_lifts) == (6, 2)
    assert warm_eq_refine_disk_seeds_sufficient(_CHAMP, payload["seeds"], seed_cap=8)


def test_record_without_lift_candidates_keeps_prom_ranking(rank_env):
    """No lift seed anywhere: reserve must not change the historical ranking."""
    from evolution.mutation_helpers import record_scan_floor_eq_refine_seeds

    ties = _ties(10)
    record_scan_floor_eq_refine_seeds(_CHAMP, ties, cycle=3)

    payload = json.loads(rank_env.read_text())
    stored = payload["seeds"]
    assert len(stored) == 8
    assert [tuple(p) for p in stored] == [tuple(p) for p in ties[:8]]


def test_lift_reserve_applies_at_reduced_cap(rank_env, monkeypatch):
    """cap=4 -> reserve min(2, 4) = 2 lift slots, ties take the rest."""
    from evolution.mutation_helpers import record_scan_floor_eq_refine_seeds, warm_eq_refine_disk_seeds_sufficient

    monkeypatch.setenv("EE_SCAN_FLOOR_EQ_REFINE_SEED_CAP", "4")
    ties = _ties(7)
    lifts = _lifts(8)
    record_scan_floor_eq_refine_seeds(_CHAMP, ties + lifts, cycle=4)

    payload = json.loads(rank_env.read_text())
    stored = payload["seeds"]
    n_ties, n_lifts = _split(stored, ties, lifts)
    assert len(stored) == 4
    assert n_lifts == 2
    assert n_ties == 2
    assert warm_eq_refine_disk_seeds_sufficient(_CHAMP, stored, seed_cap=4)
    # survivors stay PROM-ranked: ties (higher prom) still come first
    assert n_ties == sum(1 for s in stored[:2] if tuple(s) in {tuple(t) for t in ties})


def test_no_reserve_when_champion_l10_saturated(rank_env, monkeypatch):
    """warm_eq_refine_disk_seeds_sufficient short-circuits at champ L10 >= 0.85,
    so reserving lift slots there would be dead weight."""
    from evolution.mutation_helpers import record_scan_floor_eq_refine_seeds, warm_eq_refine_disk_seeds_sufficient

    def _saturated(prog, level=10):
        return 0.9

    monkeypatch.setattr("evolution.terminal_stagnation.curriculum_verify_mean", _saturated)
    ties = _ties(7)
    lifts = _lifts(8)
    record_scan_floor_eq_refine_seeds(_CHAMP, ties + lifts, cycle=5)

    payload = json.loads(rank_env.read_text())
    stored = payload["seeds"]
    assert len(stored) == 8
    # unchanged historical behaviour: PROM-ranked, no slots held open
    assert [tuple(p) for p in stored[: len(ties)]] == [tuple(p) for p in ties]
    assert warm_eq_refine_disk_seeds_sufficient(_CHAMP, stored, seed_cap=8)


def test_lift_reserve_at_floor_cap_2(rank_env, monkeypatch):
    """record()'s cap floor is max(2, env): at cap=2 both slots go to lift seeds."""
    from evolution.mutation_helpers import record_scan_floor_eq_refine_seeds, warm_eq_refine_disk_seeds_sufficient

    monkeypatch.setenv("EE_SCAN_FLOOR_EQ_REFINE_SEED_CAP", "2")
    ties = _ties(7)
    lifts = _lifts(8)
    record_scan_floor_eq_refine_seeds(_CHAMP, ties + lifts, cycle=6)

    payload = json.loads(rank_env.read_text())
    stored = payload["seeds"]
    n_ties, n_lifts = _split(stored, ties, lifts)
    assert len(stored) == 2
    assert (n_ties, n_lifts) == (0, 2)
    assert warm_eq_refine_disk_seeds_sufficient(_CHAMP, stored, seed_cap=2)
