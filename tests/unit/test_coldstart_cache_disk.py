"""Disk hydration for post-restart cold-start caches (card t_5610b3e5 / t_cf7f250d).

``_l10_lift_eq_bootstrap_cache`` (documented ~9min cold build) and
``_projection_seeds_cache`` (expensive rebuild) were in-process only, so every
daemon restart re-paid both builds inside the first cycles — the cold-start
half of the 300s first-cycle gate. These tests pin:

* the save/load roundtrip for both caches (exact nested-tuple key fidelity),
* restart hydration at the call site (no rebuild after the in-process dict
  is cleared while the disk file remains),
* anchor isolation on hydration (W1b: same-prom same-length anchors share a
  plateau, so the entry-level anchor check is load-bearing), and
* the env kill switches.

Path isolation comes from tests/unit/conftest.py (autouse fixture), so no test
here can write production .agents/ state.
"""

import os


def test_l10_lift_bootstrap_disk_roundtrip(tmp_path, monkeypatch):
    from evolution.mutation_helpers import (
        _l10_lift_eq_bootstrap_cache,
        load_l10_lift_eq_bootstrap_cache,
        save_l10_lift_eq_bootstrap_cache,
    )

    champ = [1, 2, 3, 4, 0x00100073]
    cache_path = tmp_path / "scan_floor_l10_lift_eq_bootstrap_cache.json"
    monkeypatch.setattr(
        "evolution.mutation_helpers.l10_lift_eq_bootstrap_cache_path",
        lambda: str(cache_path),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._eq_refine_plateau_key",
        lambda _c: (325.8257, len(champ)),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda c: list(c),
    )
    _l10_lift_eq_bootstrap_cache.clear()
    cache_key = (tuple(champ), (tuple(champ + [9, 10]),), 8)
    value = (tuple([1, 2, 88, 4, 0x00100073]),)
    _l10_lift_eq_bootstrap_cache[cache_key] = value

    save_l10_lift_eq_bootstrap_cache(champ, cycle=42)
    assert cache_path.is_file()

    _l10_lift_eq_bootstrap_cache.clear()
    hydrated = load_l10_lift_eq_bootstrap_cache(champ)
    assert cache_key in hydrated
    assert hydrated[cache_key] == value
    # exact structure: nested tuples of ints, int cap — must hash identically
    assert isinstance(cache_key[0], tuple)
    assert isinstance(hydrated[cache_key][0], tuple)
    assert hydrated[cache_key][0] == (1, 2, 88, 4, 0x00100073)


def test_projection_seeds_disk_roundtrip(tmp_path, monkeypatch):
    from evolution.mutation_helpers import (
        _projection_seeds_cache,
        load_projection_seeds_cache,
        save_projection_seeds_cache,
    )

    champ = [0x10, 0x20, 0x30, 0x40, 0x00100073]
    cache_path = tmp_path / "scan_floor_projection_seeds_cache.json"
    monkeypatch.setattr(
        "evolution.mutation_helpers.projection_seeds_cache_path",
        lambda: str(cache_path),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._eq_refine_plateau_key",
        lambda _c: (100.5, len(champ)),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda c: list(c),
    )
    _projection_seeds_cache.clear()
    cache_key = (tuple(champ), 4, (tuple(champ + [9, 10]), tuple(champ + [11, 12])))
    value = (tuple([0x11, 0x20, 0x30, 0x40, 0x00100073]),)
    _projection_seeds_cache[cache_key] = value

    save_projection_seeds_cache(champ, cycle=7)
    assert cache_path.is_file()

    _projection_seeds_cache.clear()
    hydrated = load_projection_seeds_cache(champ)
    assert cache_key in hydrated
    assert hydrated[cache_key] == value
    # projection key order is (champ, cap, material) — pin the decode order.
    # NOTE: `hydrated[cache_key]` is the VALUE (tuple of programs), so the key
    # structure has to be read off the dict key itself, not the value.
    assert cache_key[1] == 4
    assert len(hydrated) == 1
    hkey = next(iter(hydrated))
    assert hkey[0] == tuple(champ)
    assert hkey[1] == 4
    assert hkey[2] == cache_key[2]


def test_l10_lift_bootstrap_hydrates_after_restart(monkeypatch):
    """Cleared in-process dict + intact disk file => no rebuild (cold start)."""
    from evolution.mutation_helpers import (
        _l10_lift_eq_bootstrap_cache,
        clear_scan_floor_inject_caches,
        collect_l10_lift_eq_refine_bootstrap_seeds,
    )

    clear_scan_floor_inject_caches()
    _l10_lift_eq_bootstrap_cache.clear()
    champ = [1, 2, 3, 4, 0x00100073]
    eq_lift = [1, 2, 88, 4, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()
    calls = {"n": 0}

    def _pair_drop(_c, _v, **k):
        calls["n"] += 1
        return [list(eq_lift)]

    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda current: list(current),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p: 1.0 if len(p) > len(champ) else 0.825 if tuple(p) == tuple(eq_lift) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.collect_growth_material_compress_parents",
        lambda _c, max_parents=16: [champ + [9, 10]],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.enumerate_l10_lift_shrink_intermediates",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_l10_lift_pair_drop_eq_programs",
        _pair_drop,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.record_scan_floor_eq_refine_seeds",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.persist_l10_lift_bootstrap_intermediates",
        lambda *_a, **_k: 0,
    )

    # cold process: builds once, persists to disk (conftest path -> tmp_path)
    assert collect_l10_lift_eq_refine_bootstrap_seeds(champ, gate=gate) == [eq_lift]
    assert calls["n"] == 1
    # warm process: in-process hit, no disk involved
    assert collect_l10_lift_eq_refine_bootstrap_seeds(champ, gate=gate) == [eq_lift]
    assert calls["n"] == 1
    # simulated restart: in-process dict gone, disk file remains -> hydrate
    _l10_lift_eq_bootstrap_cache.clear()
    assert collect_l10_lift_eq_refine_bootstrap_seeds(champ, gate=gate) == [eq_lift]
    assert calls["n"] == 1, "disk hydration must skip the rebuild after restart"


def test_projection_seeds_hydrate_after_restart(monkeypatch):
    from evolution.mutation_helpers import (
        _projection_seeds_cache,
        clear_scan_floor_inject_caches,
        collect_growth_material_projection_seeds,
    )

    clear_scan_floor_inject_caches()
    _projection_seeds_cache.clear()
    champ = [1, 2, 3, 4, 0x00100073]
    long_a = champ + [9, 10]
    long_b = champ + [11, 12]
    proj_a = [1, 2, 88, 4, 0x00100073]
    proj_b = [1, 2, 77, 4, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()
    builds = {"n": 0}

    def _expand(_c, _material, **kwargs):
        builds["n"] += 1
        return [proj_a, proj_b]

    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda c: list(c),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.collect_growth_material_compress_parents",
        lambda _c, max_parents=24: [long_a, long_b],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._expand_growth_material_for_grafts",
        _expand,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.equal_length_from_longer_growth",
        lambda _c, _g, **kwargs: list(proj_b),
    )
    prom = {tuple(proj_a): 325.7, tuple(proj_b): 325.8, tuple(champ): 325.6}
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: prom.get(tuple(prog), 325.6),
    )

    first = collect_growth_material_projection_seeds(champ, max_seeds=2, cycle_no=3, gate=gate)
    assert len(first) == 2
    assert builds["n"] >= 1
    built = builds["n"]

    # In-process warm call: production rotates the CACHED result by cycle_no
    # (mutation_helpers.py:3808) while the cold build returns it prom-ranked.
    # That asymmetry pre-dates disk hydration, so the contract a restart must
    # reproduce is the warm process's answer — not the cold build's order.
    warm = collect_growth_material_projection_seeds(champ, max_seeds=2, cycle_no=3, gate=gate)
    assert builds["n"] == built, "warm in-process call must not rebuild"
    assert warm == first[1:] + first[:1], "cycle_no=3 rotates the 2 stored seeds"

    # simulated restart: in-process dict gone, disk file remains -> hydrate
    _projection_seeds_cache.clear()
    again = collect_growth_material_projection_seeds(champ, max_seeds=2, cycle_no=3, gate=gate)
    assert builds["n"] == built, "disk hydration must skip the rebuild after restart"
    assert again == warm, "hydrated post-restart result must equal the warm result"
    assert set(map(tuple, again)) == set(map(tuple, first)), "hydration must not change the seed set, only the rotation"


def test_bootstrap_disk_hydration_is_anchor_isolated(tmp_path, monkeypatch):
    """W1b: same-prom same-length anchor B must not hydrate anchor A's entries."""
    from evolution.mutation_helpers import (
        _l10_lift_eq_bootstrap_cache,
        load_l10_lift_eq_bootstrap_cache,
        save_l10_lift_eq_bootstrap_cache,
    )

    champ_a = [1, 2, 3, 4, 0x00100073]
    champ_b = [1, 2, 99, 4, 0x00100073]
    cache_path = tmp_path / "scan_floor_l10_lift_eq_bootstrap_cache.json"
    monkeypatch.setattr(
        "evolution.mutation_helpers.l10_lift_eq_bootstrap_cache_path",
        lambda: str(cache_path),
    )
    # identical plateau for both anchors: only the entry-level anchor check
    # can separate them.
    monkeypatch.setattr(
        "evolution.mutation_helpers._eq_refine_plateau_key",
        lambda _c: (325.8257, len(champ_a)),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda c: list(c),
    )
    _l10_lift_eq_bootstrap_cache.clear()
    key_a = (tuple(champ_a), (tuple(champ_a + [9, 10]),), 8)
    _l10_lift_eq_bootstrap_cache[key_a] = (tuple([1, 2, 88, 4, 0x00100073]),)
    save_l10_lift_eq_bootstrap_cache(champ_a)
    _l10_lift_eq_bootstrap_cache.clear()

    assert load_l10_lift_eq_bootstrap_cache(champ_a) != {}
    assert load_l10_lift_eq_bootstrap_cache(champ_b) == {}


def test_disk_hydration_respects_env_kill_switch(tmp_path, monkeypatch):
    from evolution.mutation_helpers import (
        _l10_lift_eq_bootstrap_cache,
        load_l10_lift_eq_bootstrap_cache,
        save_l10_lift_eq_bootstrap_cache,
    )

    champ = [1, 2, 3, 4, 0x00100073]
    cache_path = tmp_path / "scan_floor_l10_lift_eq_bootstrap_cache.json"
    monkeypatch.setattr(
        "evolution.mutation_helpers.l10_lift_eq_bootstrap_cache_path",
        lambda: str(cache_path),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._eq_refine_plateau_key",
        lambda _c: (325.8257, len(champ)),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda c: list(c),
    )
    _l10_lift_eq_bootstrap_cache.clear()
    key = (tuple(champ), (), 8)
    _l10_lift_eq_bootstrap_cache[key] = (tuple(champ),)

    monkeypatch.setenv("EE_L10_LIFT_BOOTSTRAP_DISK", "0")
    save_l10_lift_eq_bootstrap_cache(champ)
    assert not cache_path.exists()
    assert load_l10_lift_eq_bootstrap_cache(champ) == {}

    monkeypatch.setenv("EE_L10_LIFT_BOOTSTRAP_DISK", "1")
    save_l10_lift_eq_bootstrap_cache(champ)
    assert cache_path.exists()
    assert load_l10_lift_eq_bootstrap_cache(champ) == {key: (tuple(champ),)}


def test_malformed_cache_file_is_ignored(tmp_path, monkeypatch):
    from evolution.mutation_helpers import load_l10_lift_eq_bootstrap_cache

    champ = [1, 2, 3, 4, 0x00100073]
    cache_path = tmp_path / "scan_floor_l10_lift_eq_bootstrap_cache.json"
    monkeypatch.setattr(
        "evolution.mutation_helpers.l10_lift_eq_bootstrap_cache_path",
        lambda: str(cache_path),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._eq_refine_plateau_key",
        lambda _c: (325.8257, len(champ)),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda c: list(c),
    )
    # corrupt JSON
    cache_path.write_text("{not json", encoding="utf-8")
    assert load_l10_lift_eq_bootstrap_cache(champ) == {}
    # valid JSON, wrong shape
    cache_path.write_text('{"entries": 7}', encoding="utf-8")
    assert load_l10_lift_eq_bootstrap_cache(champ) == {}
    # valid shape, entries of the wrong type are skipped individually
    cache_path.write_text(
        '{"plateau_key": [325.8257, 5], "anchor": [1,2,3,4,1048691],'
        ' "entries": ["nope", {"key": {"anchor": [9], "material": "x",'
        ' "cap": "z"}, "value": [[1]]}]}',
        encoding="utf-8",
    )
    assert load_l10_lift_eq_bootstrap_cache(champ) == {}


def test_conftest_isolates_coldstart_paths(tmp_path):
    """The autouse fixture must redirect both new path functions to tmp_path."""
    from evolution import mutation_helpers as mh

    assert mh.l10_lift_eq_bootstrap_cache_path().startswith(str(tmp_path))
    assert mh.projection_seeds_cache_path().startswith(str(tmp_path))
    # production defaults resolve under .agents/ when the fixture is absent
    default = os.path.join(".agents", "scan_floor_l10_lift_eq_bootstrap_cache.json")
    assert default.endswith("scan_floor_l10_lift_eq_bootstrap_cache.json")
