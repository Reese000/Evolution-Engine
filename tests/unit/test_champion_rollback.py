"""Tests for evolution.champion_rollback."""

import json

from evolution.champion_rollback import (
    find_best_certified_champion,
    find_best_honest_compact_l30,
    find_last_clean_champion,
    flag_verify_regression_rollback,
)


def test_find_last_clean_skips_corrupt_tail(tmp_path):
    results = tmp_path / "multi_cycle_results.jsonl"
    rows = [
        {
            "cycle": 1,
            "final_level": 21,
            "champion_verify_score": 0.75,
            "champion_words": 36,
            "final_fitness": 80.0,
            "champion_program": ["0x00000013"] * 36 + ["0x00100073"],
        },
        {
            "cycle": 2,
            "final_level": 21,
            "champion_verify_score": 0.167,
            "champion_words": 5,
            "final_fitness": 47.5,
            "champion_program": ["0x000BAB83"] * 5,
        },
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    clean = find_last_clean_champion(str(results), 21, before_cycle=3)
    assert clean is not None
    assert clean["cycle"] == 1
    assert clean["champion_words"] == 36


def test_flag_verify_regression_rollback(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text("{}", encoding="utf-8")
    prog = ["0x00000013"] * 36 + ["0x00100073"]
    assert flag_verify_regression_rollback(
        str(tmp_path),
        level=30,
        prev_verify=0.579,
        prev_program=prog,
        prev_cycle=3444,
        current_verify=0.526,
    )
    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert tick.get("apply_champion_rollback") is True
    assert tick.get("champion_rollback_cycle") == 3444
    assert tick.get("champion_rollback_program") == prog


def test_find_best_certified_prefers_high_fitness(tmp_path, monkeypatch):
    """Cycle 10 (fitness 264) beats cycle 11 (fitness 26) when both verify=1.0."""
    monkeypatch.setattr(
        "evolution.champion_rollback.compact_l30_certified_row_valid",
        lambda _row: True,
    )
    results = tmp_path / "multi_cycle_results.jsonl"
    prog = ["0x00000013"] * 36 + ["0x00100073"]
    rows = [
        {
            "cycle": 10,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 37,
            "final_fitness": 264.0,
            "champion_program": prog,
        },
        {
            "cycle": 11,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 37,
            "final_fitness": 26.0,
            "champion_program": prog,
        },
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    best = find_best_certified_champion(str(results), 30, min_verify=1.0)
    assert best is not None
    assert best["cycle"] == 10
    assert best["final_fitness"] == 264.0


def test_find_best_certified_incremental_append(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evolution.champion_rollback.compact_l30_certified_row_valid",
        lambda _row: True,
    )
    results = tmp_path / "multi_cycle_results.jsonl"
    prog = ["0x00000013"] * 36 + ["0x00100073"]
    first = {
        "cycle": 10,
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 37,
        "final_fitness": 100.0,
        "champion_program": prog,
    }
    results.write_text(json.dumps(first) + "\n", encoding="utf-8")
    best = find_best_certified_champion(str(results), 30, min_verify=1.0)
    assert best is not None and best["cycle"] == 10
    second = {
        "cycle": 11,
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 37,
        "final_fitness": 400.0,
        "champion_program": prog,
    }
    with results.open("a", encoding="utf-8") as f:
        f.write(json.dumps(second) + "\n")
    best = find_best_certified_champion(str(results), 30, min_verify=1.0)
    assert best is not None
    assert best["cycle"] == 11
    assert best["final_fitness"] == 400.0


def test_find_best_certified_compact_prefers_promotion_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evolution.champion_rollback.compact_l30_certified_row_valid",
        lambda _row: True,
    )
    results = tmp_path / "multi_cycle_results.jsonl"
    stale = ["0x00000013"] * 59 + ["0x00100073"]
    better = list(stale)
    better[39] = "0xFE008CE3"
    rows = [
        {
            "cycle": 10,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 60,
            "final_fitness": 384.9,
            "task_fitness": 302.06,
            "promotion_raw_fitness": 312.81,
            "champion_program": stale,
        },
        {
            "cycle": 11,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 60,
            "final_fitness": 349.4,
            "task_fitness": 296.0,
            "promotion_raw_fitness": 313.31,
            "champion_program": better,
        },
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    best = find_best_certified_champion(str(results), 30, min_verify=1.0)
    assert best is not None
    assert best["cycle"] == 11
    assert best["promotion_raw_fitness"] == 313.31


def test_find_best_certified_allows_compact_l30(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evolution.champion_rollback.compact_l30_certified_row_valid",
        lambda _row: True,
    )
    results = tmp_path / "multi_cycle_results.jsonl"
    rows = [
        {
            "cycle": 10,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 22,
            "final_fitness": 936.0,
            "champion_program": ["0x00000013"] * 21 + ["0x00100073"],
        },
        {
            "cycle": 11,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 16,
            "final_fitness": 970.0,
            "champion_program": ["0x00000013"] * 15 + ["0x00100073"],
        },
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    best = find_best_certified_champion(str(results), 30, min_verify=1.0)
    assert best is not None
    assert best["cycle"] == 11
    assert best["champion_words"] == 16


def test_find_best_certified_skips_l30_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evolution.champion_rollback.compact_l30_certified_row_valid",
        lambda _row: True,
    )
    results = tmp_path / "multi_cycle_results.jsonl"
    rows = [
        {
            "cycle": 11,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 13,
            "final_fitness": 1848.0,
            "champion_program": ["0x00000013"] * 12 + ["0x00100073"],
        },
        {
            "cycle": 12,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 7,
            "final_fitness": 1934.0,
            "champion_program": ["0x00000013"] * 6 + ["0x00100073"],
        },
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    best = find_best_certified_champion(str(results), 30, min_verify=1.0)
    assert best is not None
    assert best["cycle"] == 11
    assert best["champion_words"] == 13


def test_find_best_certified_skips_corrupt_promotion_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evolution.champion_rollback.compact_l30_certified_row_valid",
        lambda _row: True,
    )
    results = tmp_path / "multi_cycle_results.jsonl"
    good_prog = ["0x00008193"] * 56 + ["0x00100073"]
    rows = [
        {
            "cycle": 6070,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 50,
            "final_fitness": 365.0,
            "promotion_raw_fitness": -25.0,
            "champion_program": good_prog,
        },
        {
            "cycle": 6061,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 57,
            "final_fitness": 371.98,
            "promotion_raw_fitness": 321.645,
            "champion_program": good_prog,
        },
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    best = find_best_certified_champion(str(results), 30, min_verify=1.0)
    assert best is not None
    assert best["cycle"] == 6061


def test_find_best_honest_compact_skips_trusted_one(tmp_path):
    results = tmp_path / "multi_cycle_results.jsonl"
    rows = [
        {
            "cycle": 4876,
            "final_level": 30,
            "champion_verify_score": 0.737,
            "champion_words": 24,
            "final_fitness": 167.8,
            "champion_program": ["0x00000013"] * 23 + ["0x00100073"],
        },
        {
            "cycle": 4878,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 22,
            "final_fitness": 3288.0,
            "champion_program": ["0x00000013"] * 21 + ["0x00100073"],
        },
        {
            "cycle": 4870,
            "final_level": 30,
            "champion_verify_score": 0.263,
            "champion_words": 24,
            "final_fitness": 50.0,
            "champion_program": ["0x00000013"] * 23 + ["0x00100073"],
        },
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    best = find_best_honest_compact_l30(str(results), 30)
    assert best is not None
    assert best["cycle"] == 4876
    assert abs(float(best["champion_verify_score"]) - 0.737) < 1e-6


def test_growth_material_record_and_collect(tmp_path, monkeypatch):
    from evolution.champion_rollback import (
        _growth_material_path,
        collect_growth_material_parents,
        record_scan_floor_growth_material,
    )

    champ = [1, 2, 3, 0x00100073]
    grown = [1, 2, 3, 4, 5, 0x00100073]
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(tmp_path / "growth_material.json"),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_capability_qualifies",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 323.5 if len(prog) > len(champ) else 324.0,
    )

    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    assert record_scan_floor_growth_material(champ, grown, cycle=10) is True
    parents = collect_growth_material_parents(champ, max_parents=4)
    assert parents == [grown]
    assert record_scan_floor_growth_material(champ, champ, cycle=11) is False
    assert _growth_material_path().endswith("growth_material.json")


def test_growth_material_anchor_uses_certified_floor_when_live_longer(tmp_path, monkeypatch):
    from evolution.champion_rollback import (
        _growth_material_storage_anchor,
        collect_growth_material_parents,
        record_scan_floor_growth_material,
    )

    floor_champ = [1, 2, 3, 4, 0x00100073]
    live_longer = floor_champ + [9, 10, 11]
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_plateau_armed",
        lambda: (True, len(floor_champ)),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.find_best_certified_champion",
        lambda *_a, **_k: {"champion_program": list(floor_champ), "cycle": 1},
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_capability_qualifies",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 324.0 if len(prog) == len(floor_champ) else 324.5,
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(tmp_path / "growth_material.json"),
    )

    assert _growth_material_storage_anchor(live_longer) == floor_champ
    grown = floor_champ + [20, 21]
    assert record_scan_floor_growth_material(live_longer, grown, cycle=12) is True
    parents = collect_growth_material_parents(live_longer, max_parents=4)
    assert parents == [grown]


def test_record_growth_material_strips_duplicate_trailing_ebreak(tmp_path, monkeypatch):
    from evolution.champion_rollback import _growth_material_path, record_scan_floor_growth_material

    champ = [1, 2, 3, 0x00100073]
    padded = champ + [9, 10, 0x00100073, 0x00100073, 0x00100073]
    path = tmp_path / "growth_material.json"
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(path),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_key",
        lambda _c: str(tuple(champ)),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_capability_qualifies",
        lambda _a, v: len(v) >= len(champ) + 1,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 325.5,
    )

    assert record_scan_floor_growth_material(champ, padded, cycle=9) is True
    stored = json.loads(path.read_text(encoding="utf-8"))
    prog = stored[str(tuple(champ))][0]["program"]
    assert prog == champ + [9, 10, 0x00100073]


def test_collect_growth_material_compress_parents_prefers_champ_prom_ties(tmp_path, monkeypatch):
    from evolution.champion_rollback import collect_growth_material_compress_parents

    champ = [1, 2, 3, 0x00100073]
    growth_phase = [1, 2, 3, 4, 5, 0x00100073]
    tie = [1, 2, 3, 4, 6, 7, 0x00100073]
    path = tmp_path / "growth_material.json"
    path.write_text(
        json.dumps(
            {
                str(tuple(champ)): [
                    {"program": tie, "cycle": 2, "prom": 325.5},
                    {"program": growth_phase, "cycle": 1, "prom": 325.03},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(path),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_key",
        lambda _c: str(tuple(champ)),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 325.5 if prog == champ else (325.03 if prog == growth_phase else 325.5),
    )

    parents = collect_growth_material_compress_parents(champ, max_parents=4)
    assert parents[0] == tie
    assert growth_phase in parents


def test_collect_growth_material_compress_parents_prefers_growth_phase_without_ties(tmp_path, monkeypatch):
    from evolution.champion_rollback import collect_growth_material_compress_parents

    champ = [1, 2, 3, 0x00100073]
    growth_phase = [1, 2, 3, 4, 5, 0x00100073]
    other_growth = [1, 2, 3, 4, 6, 0x00100073]
    path = tmp_path / "growth_material.json"
    path.write_text(
        json.dumps(
            {
                str(tuple(champ)): [
                    {"program": other_growth, "cycle": 2, "prom": 325.1},
                    {"program": growth_phase, "cycle": 1, "prom": 325.03},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(path),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_key",
        lambda _c: str(tuple(champ)),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 325.5 if prog == champ else (325.03 if prog == growth_phase else 325.1),
    )

    parents = collect_growth_material_compress_parents(champ, max_parents=4)
    assert parents[0] == other_growth
    assert growth_phase in parents


def test_collect_growth_material_skips_stale_sub_champion_prom(tmp_path, monkeypatch):
    from evolution.champion_rollback import collect_growth_material_parents

    champ = [1, 2, 3, 0x00100073]
    stale = [1, 2, 3, 4, 5, 0x00100073]
    fresh = [1, 2, 3, 4, 6, 0x00100073]
    path = tmp_path / "growth_material.json"
    path.write_text(
        json.dumps(
            {
                str(tuple(champ)): [
                    {"program": stale, "cycle": 1, "prom": 324.0},
                    {"program": fresh, "cycle": 2, "prom": 325.5},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(path),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_key",
        lambda _c: str(tuple(champ)),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_plateau_armed",
        lambda: (True, len(champ)),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 325.0 if prog == champ else 324.0,
    )

    parents = collect_growth_material_parents(champ, max_parents=4)
    assert parents == [fresh]


def test_collect_growth_material_parents_prefers_l10_lift_below_prom_floor(tmp_path, monkeypatch):
    from evolution.champion_rollback import collect_growth_material_parents

    champ = [1, 2, 3, 0x00100073]
    high_prom = [1, 2, 3, 4, 5, 0x00100073]
    l10_lift = [1, 2, 3, 9, 10, 11, 12, 0x00100073]
    path = tmp_path / "growth_material.json"
    path.write_text(
        json.dumps(
            {
                str(tuple(champ)): [
                    {"program": high_prom, "cycle": 1, "prom": 325.5},
                    {"program": l10_lift, "cycle": 2, "prom": 321.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(path),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_key",
        lambda _c: str(tuple(champ)),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_capability_qualifies",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_plateau_armed",
        lambda: (True, len(champ)),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 325.0 if prog == champ else (325.5 if prog == high_prom else 321.0),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda prog, level: 1.0 if level == 10 and prog == l10_lift else 0.8,
    )

    parents = collect_growth_material_parents(champ, max_parents=4)
    assert parents and parents[0] == l10_lift


def test_growth_material_records_l10_lift_below_prom_floor(tmp_path, monkeypatch):
    from evolution.champion_rollback import record_scan_floor_growth_material
    from evolution.mutation_helpers import clear_scan_floor_inject_caches

    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 0x00100073]
    grown = [1, 2, 3, 4, 5, 6, 7, 0x00100073]
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(tmp_path / "growth_material.json"),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_capability_qualifies",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_l10_lift_qualifies",
        lambda _a, _v, **kwargs: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 325.0 if prog == champ else 321.0,
    )
    assert record_scan_floor_growth_material(champ, grown, cycle=10) is True


def test_growth_material_retains_l10_lift_when_recording_higher_prom(tmp_path, monkeypatch):
    from evolution.champion_rollback import _load_growth_material_store, record_scan_floor_growth_material
    from evolution.mutation_helpers import clear_scan_floor_inject_caches

    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 0x00100073]
    l10_lift = [1, 2, 3, 4, 5, 6, 7, 8, 0x00100073]
    high_prom = [1, 2, 3, 4, 5, 6, 0x00100073]
    path = tmp_path / "growth_material.json"
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(path),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_capability_qualifies",
        lambda *_a, **_k: True,
    )

    def _l10_lift(anchor, variant, **kwargs):
        return variant == l10_lift

    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_l10_lift_qualifies",
        _l10_lift,
    )

    def _prom(prog):
        if prog == champ:
            return 325.0
        if prog == l10_lift:
            return 321.0
        return 325.5

    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        _prom,
    )
    assert record_scan_floor_growth_material(champ, l10_lift, cycle=10) is True
    assert record_scan_floor_growth_material(champ, high_prom, cycle=11) is True
    store = _load_growth_material_store()
    bucket = next(iter(store.values()))
    programs = [e["program"] for e in bucket]
    assert l10_lift in programs
    assert high_prom in programs


def test_growth_material_rejects_low_prom_junk(tmp_path, monkeypatch):
    from evolution.champion_rollback import record_scan_floor_growth_material
    from evolution.mutation_helpers import clear_scan_floor_inject_caches

    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 0x00100073]
    grown = [1, 2, 3, 4, 5, 0x00100073]
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(tmp_path / "growth_material.json"),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_capability_qualifies",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 324.0 if prog == champ else -25.0,
    )
    assert record_scan_floor_growth_material(champ, grown, cycle=10) is False


def test_growth_material_record_skips_duplicate_program(tmp_path, monkeypatch):
    from evolution.champion_rollback import record_scan_floor_growth_material
    from evolution.mutation_helpers import clear_scan_floor_inject_caches

    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 0x00100073]
    grown = [1, 2, 3, 4, 5, 0x00100073]
    prom_calls = {"n": 0}

    def _prom(prog):
        prom_calls["n"] += 1
        return 324.0 if prog == champ else 323.5

    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(tmp_path / "growth_material.json"),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_capability_qualifies",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        _prom,
    )
    assert record_scan_floor_growth_material(champ, grown, cycle=10) is True
    before = prom_calls["n"]
    assert record_scan_floor_growth_material(champ, grown, cycle=11) is False
    assert prom_calls["n"] == before


def test_growth_material_l10_lift_weak_margin(monkeypatch):
    from evolution.champion_rollback import _growth_material_l10_lift_qualifies

    anchor = [1, 2, 3, 4, 0x00100073]
    variant = anchor + [99, 88, 77, 66, 55]

    def _l10(prog, level, n_seeds=8):
        if int(level) != 10:
            return 0.99
        if tuple(prog) == tuple(anchor):
            return 0.8
        return 0.938

    monkeypatch.setattr("evolution.terminal_stagnation.curriculum_verify_mean", _l10)
    assert _growth_material_l10_lift_qualifies(anchor, variant) is True
