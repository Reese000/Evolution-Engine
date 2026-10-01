"""Certified champion rows must pass CPU side/held-out floors."""

import json
import os

os.environ.setdefault("EE_USE_GPU", "0")


def test_compact_l30_certified_row_valid_rejects_side_regression():
    from evolution.champion_rollback import compact_l30_certified_row_valid, find_best_certified_champion

    rows = {}
    for line in open("multi_cycle_results.jsonl"):
        d = json.loads(line)
        if d["cycle"] in (7448, 7452):
            rows[d["cycle"]] = d
    if 7448 not in rows or 7452 not in rows:
        return
    assert compact_l30_certified_row_valid(rows[7448])
    assert not compact_l30_certified_row_valid(rows[7452])

    best = find_best_certified_champion(
        "multi_cycle_results.jsonl",
        30,
        min_verify=0.99,
        min_fitness=100.0,
    )
    assert best is not None
    assert int(best["cycle"]) != 7452


def test_growth_material_load_prunes_side_regression():
    import json
    from pathlib import Path

    from evolution.champion_rollback import _load_growth_material_entries

    row = json.loads(Path("multi_cycle_results.jsonl").read_text().splitlines()[-1])
    champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
    material_path = Path(".agents/scan_floor_growth_material.json")
    if not material_path.exists():
        return
    before = json.loads(material_path.read_text())
    _anchor, entries = _load_growth_material_entries(champ)
    assert entries
    assert all(len(e.get("program") or []) <= len(champ) + 25 for e in entries)
    after = json.loads(material_path.read_text())
    # Prune should drop the highest-prom side-broken rows from the live bucket.
    assert sum(len(v) for v in after.values()) <= sum(len(v) for v in before.values())


def test_growth_material_record_rejects_side_regression(monkeypatch):
    from evolution.champion_rollback import _growth_material_capability_qualifies, record_scan_floor_growth_material

    champ = [1, 2, 3, 0x00100073]
    good = champ + [4, 5]
    bad = champ + [6, 7]
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_heldout_qualifies",
        lambda _a, _b: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _a, prog, **_k: prog == good,
    )
    assert not _growth_material_capability_qualifies(champ, bad)
    assert _growth_material_capability_qualifies(champ, good)
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._compact_prom_raw_cached",
        lambda _p: 325.0,
    )
    assert not record_scan_floor_growth_material(champ, bad, cycle=99999)


def test_ranked_growth_fills_reject_side_regression(monkeypatch):
    from evolution.terminal_stagnation import _ranked_scan_floor_growth_fills_cached, ranked_scan_floor_growth_fills

    champ = [1, 2, 3, 0x00100073]
    good = champ + [4, 5, 6]
    bad = champ + [7, 8, 9]
    _ranked_scan_floor_growth_fills_cached.cache_clear()
    monkeypatch.setattr(
        "evolution.terminal_stagnation.l30_easy_fill_variants",
        lambda _p: [good, bad],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.heldout_l30_verify_mean",
        lambda *_a, **_k: 0.99,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _a, prog, **_k: prog == good,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 326.0 if prog == bad else 325.0,
    )
    ranked = ranked_scan_floor_growth_fills(champ)
    assert ranked == [good]


def test_growth_completion_rejects_side_regression_at_length():
    import json

    from evolution.mutation_helpers import scan_floor_growth_completion_qualifies

    rows = {}
    for line in open("multi_cycle_results.jsonl"):
        d = json.loads(line)
        if d["cycle"] in (7448, 7450):
            rows[d["cycle"]] = d
    if 7448 not in rows or 7450 not in rows:
        return

    def dec(row):
        return [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]

    champ = dec(rows[7448])
    grown = dec(rows[7450])
    assert len(grown) > len(champ)
    assert not scan_floor_growth_completion_qualifies(champ, grown)
