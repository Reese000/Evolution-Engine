"""Throughput guards for scan-floor stall paths."""

import os

import pytest

os.environ.setdefault("EE_USE_GPU", "0")


def test_compact_prom_raw_cached_amortizes_within_cycle(monkeypatch):
    from evolution.mutation_helpers import (
        _compact_prom_raw_cached,
        clear_scan_floor_inject_caches,
        scan_floor_growth_inject_rank,
    )

    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 4, 0x00100073]
    cand = [1, 2, 99, 4, 0x00100073]
    calls = {"n": 0}

    def _prom(prog):
        calls["n"] += 1
        return 325.627 if tuple(prog) == tuple(cand) else 325.0

    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        _prom,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )

    scan_floor_growth_inject_rank(champ, cand, stalled=True)
    scan_floor_growth_inject_rank(champ, cand, stalled=True)
    _compact_prom_raw_cached(cand)
    assert calls["n"] == 2


def test_record_scan_floor_eq_refine_seeds_prefers_higher_prom(monkeypatch, tmp_path):
    from evolution.mutation_helpers import record_scan_floor_eq_refine_seeds

    champ = [1, 2, 3, 4, 0x00100073]
    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_eq_refine_seeds_path",
        lambda: str(agents / "scan_floor_eq_refine_seeds.json"),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    prom_map = {
        tuple([1, 2, 3, 4, 0x00100073]): 325.0,
        tuple([1, 2, 88, 4, 0x00100073]): 325.9,
        tuple([1, 99, 3, 4, 0x00100073]): 325.5,
    }
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: prom_map.get(tuple(prog), 325.0),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda prog, level: 0.9,
    )
    # 5-word test champion sits below the production persist floor (20 words);
    # disable the size guard so this test exercises seed ranking itself.
    monkeypatch.setenv("EE_EQ_REFINE_PERSIST_MIN_WORDS", "0")
    record_scan_floor_eq_refine_seeds(
        champ,
        [[1, 2, 99, 4, 0x00100073], [1, 2, 88, 4, 0x00100073]],
        cycle=1,
    )
    import json

    payload = json.loads((agents / "scan_floor_eq_refine_seeds.json").read_text())
    assert payload["seeds"][0] == [1, 2, 88, 4, 0x00100073]


def test_eq_refine_bootstrap_survives_cycle_cache_clear(monkeypatch, tmp_path):
    from evolution.mutation_helpers import (
        _eq_refine_bootstrapped,
        clear_scan_floor_inject_caches,
        load_scan_floor_eq_refine_seeds,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_eq_refine_seeds_path",
        lambda: str(agents / "scan_floor_eq_refine_seeds.json"),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: False})(),
    )
    graft_calls = {"n": 0}

    def _graft(*_a, **_k):
        graft_calls["n"] += 1
        return []

    monkeypatch.setattr(
        "evolution.mutation_helpers.growth_material_diff_graft_variants",
        _graft,
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.collect_growth_material_compress_parents",
        lambda *_a, **_k: [[1, 2, 3, 4, 0x00100073]],
    )

    champ = [1, 2, 3, 4, 0x00100073]
    _eq_refine_bootstrapped.clear()
    load_scan_floor_eq_refine_seeds(champ, max_seeds=4)
    clear_scan_floor_inject_caches()
    load_scan_floor_eq_refine_seeds(champ, max_seeds=4)
    assert graft_calls["n"] == 1


def test_archive_parents_skip_when_compression_plateau_exhausted(monkeypatch):
    from evolution_engine import EvolutionConfig, EvolutionEngine

    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion_level = 30
    eng.champion = [0x00008193] * 39 + [0x00100073]
    rank_flags: list[bool] = []

    def _collect(*_a, **kwargs):
        rank_flags.append(bool(kwargs.get("rank_by_promotion_raw")))
        return []

    monkeypatch.setattr(
        "evolution.champion_rollback.collect_jsonl_archive_parents",
        _collect,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _n: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_compression_hot",
        lambda _p: False,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._skip_empty_compression_rescan",
        lambda _p, _w: True,
    )

    eng._terminal_archive_parents()
    assert rank_flags == [False]


def test_locus_probe_ranks_by_prom_raw(monkeypatch):
    from evolution.terminal_stagnation import terminal_growth_material_locus_probe_variants

    champ = [10, 20, 30, 40, 0x00100073]
    material_prog = [10, 20, 99, 40, 0x00100073, 0x11111111]
    prom_map = {
        tuple(champ): 325.0,
        (10, 20, 99, 40, 0x00100073): 325.9,
        (10, 88, 30, 40, 0x00100073): 325.5,
    }

    monkeypatch.setattr(
        "evolution.champion_rollback._load_growth_material_entries",
        lambda _c: (
            champ,
            [
                {"program": material_prog, "prom": 325.627},
                {"program": [10, 88, 30, 40, 0x00100073, 0x22222222], "prom": 325.6},
            ],
        ),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: prom_map.get(tuple(prog), 325.0),
    )

    out = terminal_growth_material_locus_probe_variants(champ, max_evals=8, start_offset=0)
    assert out
    assert float(prom_map[tuple(out[0])]) >= float(prom_map[tuple(out[1])])


def test_locus_probe_uses_projected_material_body_diffs(monkeypatch):
    from evolution.terminal_stagnation import terminal_growth_material_locus_probe_variants

    champ = [10, 20, 30, 40, 0x00100073]
    tail_only = champ + [9, 10]
    projected = [10, 20, 99, 40, 0x00100073]

    monkeypatch.setattr(
        "evolution.champion_rollback._load_growth_material_entries",
        lambda _c: (champ, [{"program": tail_only, "prom": 325.627}]),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._expand_growth_material_for_grafts",
        lambda _c, material, **_: material + [projected],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )

    out = terminal_growth_material_locus_probe_variants(champ, max_evals=4, start_offset=0)
    assert projected in out


def test_locus_probe_triple_site_combos(monkeypatch):
    from evolution.terminal_stagnation import clear_locus_probe_cache, terminal_growth_material_locus_probe_variants

    clear_locus_probe_cache()
    champ = [10, 20, 30, 40, 50, 0x00100073]
    monkeypatch.setattr(
        "evolution.champion_rollback._load_growth_material_entries",
        lambda _c: (
            champ,
            [
                {"program": [10, 88, 30, 40, 50, 0x00100073], "prom": 325.6},
                {"program": [10, 20, 77, 40, 50, 0x00100073], "prom": 325.6},
                {"program": [10, 20, 30, 99, 50, 0x00100073], "prom": 325.6},
            ],
        ),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._expand_growth_material_for_grafts",
        lambda _c, material, **_: material,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    monkeypatch.setenv("EE_SCAN_FLOOR_GROWTH_LOCUS_PAIR_BUDGET", "0")
    monkeypatch.setenv("EE_SCAN_FLOOR_GROWTH_LOCUS_TRIPLE_BUDGET", "4")

    out = terminal_growth_material_locus_probe_variants(champ, max_evals=12, start_offset=0)
    assert [10, 88, 77, 99, 50, 0x00100073] in out


def test_growth_material_locus_probe_grafts_diff_site(monkeypatch):
    from evolution.terminal_stagnation import terminal_growth_material_locus_probe_variants

    champ = [10, 20, 30, 40, 0x00100073]
    material_prog = [10, 20, 99, 40, 0x00100073, 0x11111111]

    monkeypatch.setattr(
        "evolution.champion_rollback._load_growth_material_entries",
        lambda _c: (champ, [{"program": material_prog, "prom": 325.627}]),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )

    out = terminal_growth_material_locus_probe_variants(champ, max_evals=4, start_offset=0)
    assert [10, 20, 99, 40, 0x00100073] in out


def test_locus_probe_cache_amortizes_within_cycle(monkeypatch):
    from evolution.terminal_stagnation import clear_locus_probe_cache, terminal_growth_material_locus_probe_variants

    clear_locus_probe_cache()
    champ = [10, 20, 30, 40, 0x00100073]
    material_prog = [10, 20, 99, 40, 0x00100073, 0x11111111]
    load_calls = {"n": 0}

    def _load(_c):
        load_calls["n"] += 1
        return champ, [{"program": material_prog, "prom": 325.627}]

    monkeypatch.setattr(
        "evolution.champion_rollback._load_growth_material_entries",
        _load,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )

    terminal_growth_material_locus_probe_variants(champ, max_evals=4, start_offset=0)
    terminal_growth_material_locus_probe_variants(champ, max_evals=4, start_offset=0)
    assert load_calls["n"] == 1


def test_eq_refine_alt_cache_key_ignores_refine_budget():
    from evolution.terminal_stagnation import _eq_refine_alt_probe_cache_key

    champ = [1, 2, 3, 4, 0x00100073]
    a = _eq_refine_alt_probe_cache_key(champ, 3, 24)
    b = _eq_refine_alt_probe_cache_key(champ, 3, 96)
    assert a == b


def test_eq_refine_uses_capped_prom_rank_when_stalled(monkeypatch):
    from evolution.terminal_stagnation import terminal_equal_length_refine_variants

    champ = [10, 20, 30, 40, 0x00100073]
    rank_calls: list[int] = []

    def _rank(_champion, picked, *, budget=None):
        rank_calls.append(int(budget or 0))
        return list(picked)[:12]

    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.compact_curriculum_side_score",
        lambda _p: 1.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._rank_scan_floor_picked_probes",
        _rank,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_branch_flip_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_overlay_keep_length_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.register_repair.incremental_repair_variant",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_imm_perturb_probe_variants",
        lambda *_a, **_k: [[10, 21, 30, 40, 0x00100073]],
    )
    for name in (
        "terminal_instr_swap_probe_variants",
        "terminal_deterministic_instr_swap_probe_variants",
        "terminal_growth_material_locus_probe_variants",
        "terminal_tail_wide_imm_probe_variants",
        "terminal_dual_imm_probe_variants",
        "terminal_chained_imm_perturb_probe_variants",
        "terminal_block_reorder_probe_variants",
        "terminal_chained_reorder_imm_probe_variants",
        "terminal_crossover_probe_variants",
    ):
        monkeypatch.setattr(
            f"evolution.terminal_stagnation.{name}",
            lambda *_a, **_k: [],
        )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )

    import random

    terminal_equal_length_refine_variants(champ, 30, random.Random(0), max_evals=12)
    assert rank_calls
    assert rank_calls[0] >= 12


def test_dual_word_diff_uses_multiple_archive_parents(monkeypatch):
    from evolution.terminal_stagnation import terminal_dual_word_diff_probe_variants

    champ = [10, 20, 30, 40, 0x00100073]
    parent_a = [10, 99, 30, 40, 0x00100073]
    parent_b = [10, 20, 88, 40, 0x00100073]

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_gate_repair_variant",
        lambda _p, v, _g: list(v),
    )

    out = terminal_dual_word_diff_probe_variants(
        champ,
        [parent_a, parent_b],
        max_evals=8,
        start_offset=0,
    )
    assert [10, 99, 88, 40, 0x00100073] in out


def test_stalled_model_guided_injection_helper():
    from evolution.mutation_helpers import (
        scan_floor_model_guided_injection_generation,
        scan_floor_stalled_model_guided_injection_generation,
    )

    assert scan_floor_stalled_model_guided_injection_generation(0, 30)
    assert scan_floor_stalled_model_guided_injection_generation(29, 30)
    assert not scan_floor_stalled_model_guided_injection_generation(15, 30)
    assert scan_floor_model_guided_injection_generation(29, 30)
    assert not scan_floor_model_guided_injection_generation(0, 30)


def test_mg_stall_collect_cap_env_default():
    import os

    assert int(os.environ.get("EE_SCAN_FLOOR_MG_STALL_COLLECT_CAP", "16")) >= 8
    assert int(os.environ.get("EE_SCAN_FLOOR_MG_STALL_PROM_SCORE_CAP", "4")) >= 4


def test_should_rescan_false_when_compression_plateau_exhausted(monkeypatch, tmp_path):
    from evolution.terminal_stagnation import should_rescan_terminal_efficiency_compression

    agents = tmp_path / ".agents"
    agents.mkdir()
    miss = agents / "scan_floor_compression_miss.json"
    miss.write_text(
        '{"deadbeef": {"cycle": 1, "hits": 0, "words": 40}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_compression_miss_path",
        lambda: str(miss),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda w: int(w) == 40,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._read_progress_state",
        lambda: {"terminal_efficiency_scan_floor_cycle": 100},
    )
    assert should_rescan_terminal_efficiency_compression(103, 40) is False


def test_should_rescan_terminal_efficiency_compression_warm_disk_interval(tmp_path, monkeypatch):
    import json

    from evolution.terminal_stagnation import should_rescan_terminal_efficiency_compression

    agents = tmp_path / ".agents"
    agents.mkdir()
    rank_path = agents / "scan_floor_compression_rank.json"
    miss_path = agents / "scan_floor_compression_miss.json"
    champ = [0x00008193, 0x002084B3, 0x00100073]
    rank_path.write_text(
        json.dumps(
            {
                "00008193,002084B3,00100073": [[0x00008193, 0x00100073]],
            }
        ),
        encoding="utf-8",
    )
    miss_path.write_text(
        json.dumps(
            {
                "00008193,002084B3,00100073": {"cycle": 100, "hits": 5, "words": 3},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_compression_rank_disk_path",
        lambda: str(rank_path),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_compression_miss_path",
        lambda: str(miss_path),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda w: int(w) == 3,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda w: True,
    )
    monkeypatch.setenv("EE_SCAN_FLOOR_STALL_COMPRESSION_RESCAN_INTERVAL", "3")
    monkeypatch.setenv("EE_SCAN_FLOOR_STALL_COMPRESSION_RESCAN_WARM_INTERVAL", "12")
    state = {"terminal_efficiency_scan_floor_cycle": 100}

    def _read():
        return dict(state)

    def _patch(updates: dict) -> None:
        state.update(updates)

    monkeypatch.setattr("evolution.terminal_stagnation._read_progress_state", _read)
    monkeypatch.setattr("evolution.terminal_stagnation._patch_progress_state", _patch)
    assert should_rescan_terminal_efficiency_compression(103, 3, champ) is False
    assert should_rescan_terminal_efficiency_compression(112, 3, champ) is True


def test_fresh_compress_core_cache_survives_cycle_change(monkeypatch):
    from evolution.mutation_helpers import _fresh_compress_core_cache, collect_scan_floor_fresh_compression_variants

    champ = [1, 2, 3, 4, 0x00100073]
    calls = {"n": 0}

    def _core(*_a, **_k):
        calls["n"] += 1
        return [[1, 2, 88, 4, 0x00100073]]

    monkeypatch.setattr(
        "evolution.mutation_helpers._collect_fresh_compression_core",
        _core,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._merge_fresh_compress_long_sources",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_compression_promote_qualifies",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda _p: 325.8257,
    )
    _fresh_compress_core_cache.clear()
    collect_scan_floor_fresh_compression_variants(champ, cycle_no=10)
    collect_scan_floor_fresh_compression_variants(champ, cycle_no=11)
    assert calls["n"] == 1


def test_eq_refine_late_merged_disk_cache_roundtrip(tmp_path, monkeypatch):
    from evolution.mutation_helpers import load_eq_refine_late_merged_cache, save_eq_refine_late_merged_cache

    champ = [1, 2, 3, 4, 0x00100073]
    merged = [[1, 2, 88, 4, 0x00100073], [1, 88, 3, 4, 0x00100073]]
    late_key = ((325.8257, 5), True, 8, False)
    cache_path = tmp_path / "scan_floor_eq_refine_late_merged_cache.json"
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_eq_refine_late_merged_cache_path",
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
    save_eq_refine_late_merged_cache(champ, late_key, merged, cycle=42)
    loaded = load_eq_refine_late_merged_cache(champ, late_key)
    assert loaded is not None
    assert [list(p) for p in loaded] == merged


def test_eq_refine_late_inject_skips_perturb_when_warm_disk(monkeypatch):
    from evolution.mutation_helpers import inject_scan_floor_eq_refine_seed_challengers

    champ = [10, 20, 30, 40, 0x00100073]
    seeds = [[10, 20, 99, 40, 0x00100073]] * 8
    challengers = [list(champ) for _ in range(4)]
    strategies = ["MUT"] * 4
    calls = {"perturb": 0}

    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [list(s) for s in seeds],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.warm_eq_refine_disk_seeds_sufficient",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_eq_refine_late_merged_cache",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )

    def _perturb(*_a, **_k):
        calls["perturb"] += 1
        return []

    monkeypatch.setattr(
        "evolution.mutation_helpers.refine_eq_seed_one_step_perturb_variants",
        _perturb,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.record_scan_floor_eq_refine_seeds",
        lambda *a, **k: None,
    )
    monkeypatch.setenv("EE_EQ_REFINE_WARM_SKIP_PERTURB", "1")

    inject_scan_floor_eq_refine_seed_challengers(
        champ,
        challengers,
        __import__("random").Random(0),
        cycle_no=3,
        generation=48,
        growth_inject_generation=48,
        mg_inject_generation=49,
    )
    assert calls["perturb"] == 0


def test_record_scan_floor_preserves_floor_cycle_at_same_length(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        record_terminal_efficiency_scan_floor,
        should_rescan_terminal_efficiency_compression,
    )

    state: dict = {}
    monkeypatch.setattr(
        "evolution.terminal_stagnation._read_progress_state",
        lambda: dict(state),
    )

    def _patch(updates: dict) -> None:
        state.update(updates)

    monkeypatch.setattr("evolution.terminal_stagnation._patch_progress_state", _patch)
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    record_terminal_efficiency_scan_floor(100, 27, 1.0)
    assert state["terminal_efficiency_scan_floor_cycle"] == 100
    record_terminal_efficiency_scan_floor(101, 27, 1.0)
    assert state["terminal_efficiency_scan_floor_cycle"] == 100
    assert should_rescan_terminal_efficiency_compression(101, 27) is False
    record_terminal_efficiency_scan_floor(102, 28, 1.0)
    assert state["terminal_efficiency_scan_floor_cycle"] == 102


def test_mg_stall_collect_skipped_when_late_merged_cache_warm(monkeypatch):
    from evolution.mutation_helpers import should_skip_mg_stall_collect_for_warm_eq_refine

    champ = [10, 20, 30, 40, 0x00100073]
    late_key = ((100.0, 5), False, 8, False)
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [list(champ)],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.warm_eq_refine_disk_seeds_sufficient",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_eq_refine_late_merged_cache",
        lambda _c, key: ((1, 2, 3, 0x00100073),) if key == late_key else None,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._eq_refine_plateau_key",
        lambda _c: (100.0, 5),
    )
    assert should_skip_mg_stall_collect_for_warm_eq_refine(champ) is True


def test_eq_refine_late_merged_cache_survives_clear(monkeypatch):
    from evolution.mutation_helpers import _eq_refine_late_merged_cache, clear_scan_floor_inject_caches

    key = ((100.5, 27), True, 8, False)
    _eq_refine_late_merged_cache[key] = ((1, 2, 3, 0x00100073),)
    clear_scan_floor_inject_caches()
    assert key in _eq_refine_late_merged_cache


def test_growth_prefix_tail_splice_uses_true_lcp(monkeypatch):
    from evolution.terminal_stagnation import terminal_growth_prefix_tail_splice_probe_variants

    champ = [10, 20, 30, 40, 50, 0x00100073]
    parent = [10, 20, 30, 99, 88, 0x00100073]

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_gate_repair_variant",
        lambda _p, v, _g: None,
    )

    out = terminal_growth_prefix_tail_splice_probe_variants(champ, [parent], max_evals=8, start_offset=0)
    assert [10, 20, 30, 99, 88, 0x00100073] in out


def test_growth_anchor_lcp_is_true_prefix_not_inflated():
    from evolution.mutation_helpers import growth_anchor_lcp_prefix

    champ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0x00100073]
    anchor = [1, 2, 3, 99, 5, 6, 7, 8, 9, 10, 11, 0x00100073]
    assert growth_anchor_lcp_prefix(champ, anchor) == 3


def test_repair_compressed_growth_probe_keeps_equal_length_gate_repair(monkeypatch):
    from evolution.mutation_helpers import _repair_compressed_growth_probe

    champ = [1, 2, 3, 4, 0x00100073]
    near = [1, 2, 99, 4, 0x00100073]
    repaired = [1, 2, 88, 4, 0x00100073]
    seen_prefix: list[int] = []

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(repaired)

    def _repair(_prog, _gate, **kwargs):
        seen_prefix.append(int(kwargs.get("prefix_count", -1)))
        return repaired

    monkeypatch.setattr(
        "evolution.mutation_helpers.repair_model_guided_for_gate",
        _repair,
    )
    out = _repair_compressed_growth_probe(champ, near, gate=_Gate(), prefix_count=26)
    assert out == repaired
    assert seen_prefix == [26]


def test_score_stall_collect_early_exit_on_breakthrough(monkeypatch):
    from evolution.mutation_helpers import clear_scan_floor_inject_caches, score_stall_collect_gate_programs

    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 4, 0x00100073]
    tie = [1, 2, 99, 4, 0x00100073]
    breakthrough = [1, 2, 88, 4, 0x00100073]
    calls = {"n": 0}

    def _prom(prog):
        calls["n"] += 1
        if tuple(prog) == tuple(breakthrough):
            return 325.64
        return 325.627

    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        _prom,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_min_raw_gain",
        lambda _n: 0.01,
    )
    ranked = score_stall_collect_gate_programs(
        champ,
        [tie, breakthrough],
        cycle_no=0,
        score_cap=4,
    )
    assert ranked[0] == breakthrough
    assert calls["n"] <= 3


def test_equal_length_from_longer_growth_prefers_projection(monkeypatch):
    from evolution.mutation_helpers import equal_length_from_longer_growth

    champ = [10] * 26 + [20, 30] + [0x00100073] * 12
    longer = [10] * 26 + [99, 88, 77] + [0x00100073] * 14
    projected = list(champ)
    projected[26] = 99
    projected[27] = 88
    compress_calls = {"n": 0}

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(projected)

    monkeypatch.setattr(
        "evolution.mutation_helpers.growth_mg_equal_length_projection",
        lambda _c, _g, **kwargs: projected,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda *_a, **_k: (compress_calls.__setitem__("n", compress_calls["n"] + 1), champ)[1],
    )

    out = equal_length_from_longer_growth(champ, longer, gate=_Gate(), max_drop_trials=4, cycle_no=1)
    assert compress_calls["n"] == 0
    assert out == projected


def test_compress_pool_prefers_projection_before_peephole(monkeypatch):
    from evolution.mutation_helpers import compress_growth_probe_pool

    champ = [10] * 26 + [20, 30] + [0x00100073] * 12
    longer = [10] * 26 + [99, 88, 77] + [0x00100073] * 14
    projected = list(champ)
    projected[26] = 99
    projected[27] = 88
    compress_calls = {"n": 0}

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(projected)

    monkeypatch.setattr(
        "evolution.mutation_helpers.growth_mg_equal_length_projection",
        lambda _c, _g, **kwargs: projected,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda *_a, **_k: (compress_calls.__setitem__("n", compress_calls["n"] + 1), champ)[1],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, variant, gate=None, **kwargs: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 325.64 if tuple(prog) == tuple(projected) else 325.0,
    )

    pool = compress_growth_probe_pool(champ, [longer], gate=_Gate(), cycle_no=0, max_trials=1)
    assert compress_calls["n"] == 0
    assert pool[0] == projected


def test_select_diverse_growth_material_prefers_spread():
    from evolution.mutation_helpers import select_diverse_growth_material_programs

    champ = [1] * 30 + [0x00100073] * 10
    near = champ + [2, 3, 4, 0x00100073]
    far = champ + [9, 8, 7, 0x00100073]
    picked = select_diverse_growth_material_programs(champ, [near, far], max_select=1)
    assert picked == [far]


def test_flush_growth_mg_material_candidates_records_diverse_subset(monkeypatch, tmp_path):
    from evolution.mutation_helpers import flush_growth_mg_material_candidates

    champ = [1, 2, 3, 0x00100073]
    a = [1, 2, 3, 4, 5, 0x00100073]
    b = [1, 2, 3, 9, 8, 0x00100073]
    recorded: list[list[int]] = []

    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_path",
        lambda: str(tmp_path / "growth_material.json"),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback._growth_material_storage_anchor",
        lambda _c: list(champ),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_heldout_qualifies",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 324.0 if prog == champ else 323.5,
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.record_scan_floor_growth_material",
        lambda _c, prog, **kwargs: recorded.append(list(prog)) or True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda _prog, _level: 0.8,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_scan_floor_l10_lift_growth_candidates",
        lambda _c, **kwargs: [],
    )

    n = flush_growth_mg_material_candidates(champ, [a, a, b], cycle=7, record_cap=2)
    assert n == 2
    assert len(recorded) == 2
    assert recorded[0] != recorded[1]


def test_flush_growth_mg_material_seeds_l10_lift_when_weak(monkeypatch):
    from evolution.mutation_helpers import flush_growth_mg_material_candidates

    champ = [1, 2, 3, 4, 0x00100073]
    lift = [1, 2, 3, 4, 9, 10, 11, 0x00100073]
    recorded: list[list[int]] = []

    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda prog, level: 1.0 if level == 10 and len(prog) > len(champ) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_scan_floor_l10_lift_growth_candidates",
        lambda _c, **kwargs: [list(lift)],
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.record_scan_floor_growth_material",
        lambda _c, prog, **kwargs: recorded.append(list(prog)) or True,
    )

    n = flush_growth_mg_material_candidates(champ, [], cycle=9, record_cap=2)
    assert n >= 1
    assert lift in recorded


def test_resolve_growth_mg_completion_tries_lcp_deltas_on_projection(monkeypatch):
    from evolution.mutation_helpers import resolve_growth_mg_completion

    champ = [10] * 26 + [20, 30] + [0x00100073] * 12
    longer = [10] * 26 + [99, 88, 77] + [0x00100073] * 14
    projected = list(champ)
    projected[27] = 88
    calls: list[int] = []

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(projected)

    def _proj(_c, _g, **kwargs):
        calls.append(int(kwargs.get("prefix_count", 0)))
        return projected if int(kwargs.get("prefix_count", 0)) == 27 else None

    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda _c, _g, **kwargs: list(champ),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.growth_mg_equal_length_projection",
        _proj,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, gate=None: True,
    )

    out, attempted = resolve_growth_mg_completion(champ, longer, gate=_Gate(), prefix_count=26, allow_projection=True)
    assert attempted is True
    assert out == projected
    assert 27 in calls


def test_resolve_growth_mg_completion_uses_projection_on_compress_tie(monkeypatch):
    from evolution.mutation_helpers import resolve_growth_mg_completion

    champ = [10] * 26 + [20, 30] + [0x00100073] * 12
    longer = [10] * 26 + [99, 88, 77] + [0x00100073] * 14
    projected = list(champ)
    projected[26] = 99
    projected[27] = 88

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(projected)

    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda _c, _g, **kwargs: list(champ),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.growth_mg_equal_length_projection",
        lambda _c, _g, **kwargs: projected,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, gate=None: True,
    )

    out, attempted = resolve_growth_mg_completion(champ, longer, gate=_Gate(), prefix_count=26, allow_projection=True)
    assert attempted is True
    assert out == projected


def test_resolve_growth_mg_completion_skips_projection_when_disallowed(monkeypatch):
    from evolution.mutation_helpers import resolve_growth_mg_completion

    champ = [10] * 26 + [20, 30] + [0x00100073] * 12
    longer = [10] * 26 + [99, 88, 77] + [0x00100073] * 14
    proj_calls = {"n": 0}

    class _Gate:
        def qualifies(self, variant):
            return False

    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda _c, _g, **kwargs: list(champ),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.growth_mg_equal_length_projection",
        lambda *_a, **_k: (proj_calls.__setitem__("n", proj_calls["n"] + 1), None)[1],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, gate=None: True,
    )

    out, attempted = resolve_growth_mg_completion(champ, longer, gate=_Gate(), prefix_count=26, allow_projection=False)
    assert attempted is False
    assert proj_calls["n"] == 0
    assert out == longer


def test_growth_mg_equal_length_projection_rejects_gate_fail():
    from evolution.mutation_helpers import growth_mg_equal_length_projection

    champ = [10] * 26 + [20, 30] + [0x00100073] * 12
    grown = [10] * 26 + [99, 88, 77] + [0x00100073] * 14

    class _Gate:
        def qualifies(self, _variant):
            return False

    assert growth_mg_equal_length_projection(champ, grown, prefix_count=26, gate=_Gate()) is None


def test_growth_mg_equal_length_projection_locks_prefix():
    from evolution.mutation_helpers import growth_mg_equal_length_projection

    champ = [10] * 26 + [20, 30] + [0x00100073] * 12
    grown = [10] * 26 + [99, 88, 77] + [0x00100073] * 14
    expected = list(champ)
    for i in range(26, len(champ)):
        expected[i] = grown[i]

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) == tuple(expected)

    out = growth_mg_equal_length_projection(champ, grown, prefix_count=26, gate=_Gate())
    assert out == expected


def test_expand_growth_anchor_prefix_counts_rotates_lcp_deltas():
    from evolution.mutation_helpers import expand_growth_anchor_prefix_counts_stalled

    champ = [10] * 26 + [20, 30] + [0x00100073] * 12
    anchor = [10] * 26 + [99, 88] + [0x00100073] * 16
    assert len(champ) == 40
    out0 = expand_growth_anchor_prefix_counts_stalled(champ, anchor, cycle_offset=0, max_counts=3)
    out1 = expand_growth_anchor_prefix_counts_stalled(champ, anchor, cycle_offset=1, max_counts=3)
    assert out0 == [24, 25, 26]
    assert out1 == [25, 26, 27]
    assert all(1 <= pc < len(champ) for pc in out0)


def test_expand_growth_anchor_prefix_counts_inject_gen_wide_tail(monkeypatch):
    from evolution.mutation_helpers import expand_growth_anchor_prefix_counts_stalled

    champ = [10] * 26 + [20, 30] + [0x00100073] * 12
    anchor = [10] * 26 + [99, 88] + [0x00100073] * 16
    monkeypatch.setenv("EE_SCAN_FLOOR_GROWTH_INJECT_MG_TAIL_WORDS", "18")

    out = expand_growth_anchor_prefix_counts_stalled(
        champ,
        anchor,
        cycle_offset=0,
        max_counts=5,
        generation=49,
        inject_generation=49,
    )
    assert out[0] == 22
    assert 22 in out
    assert all(1 <= pc < len(champ) for pc in out)


def test_eq_refine_inject_merges_locus_grafts(monkeypatch):
    from evolution.mutation_helpers import inject_scan_floor_eq_refine_seed_challengers

    champ = [10, 20, 30, 40, 0x00100073]
    locus_prog = [10, 20, 99, 40, 0x00100073]
    challengers = [list(champ) for _ in range(6)]
    strategies = ["MUT"] * 6

    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_growth_material_projection_seeds",
        lambda _c, **kwargs: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_growth_material_locus_probe_variants",
        lambda _c, **kwargs: [locus_prog],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.refine_eq_seed_one_step_perturb_variants",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.refine_eq_seed_two_step_perturb_variants",
        lambda *a, **k: [],
    )

    n = inject_scan_floor_eq_refine_seed_challengers(
        champ,
        challengers,
        __import__("random").Random(0),
        cycle_no=1,
        generation=48,
        growth_inject_generation=48,
        mg_inject_generation=49,
        strategies=strategies,
    )
    assert n >= 1
    assert locus_prog in challengers
    assert "EQ_REFINE_SEED" in strategies


def test_scan_floor_growth_mg_max_words_caps_between_inject_gens():
    from evolution.mutation_helpers import scan_floor_growth_mg_max_words

    assert scan_floor_growth_mg_max_words(40, generation=10, inject_generation=49, stalled=True) == 48
    assert scan_floor_growth_mg_max_words(40, generation=49, inject_generation=49, stalled=True) == 52
    assert scan_floor_growth_mg_max_words(40, generation=10, inject_generation=49, stalled=False) == 72


def test_scan_floor_growth_mg_max_words_scales_with_shorter_champion():
    from evolution.mutation_helpers import scan_floor_growth_mg_max_words

    # 30w floor: relative overlength budget must beat the legacy 48/52 absolutes.
    assert scan_floor_growth_mg_max_words(30, generation=10, inject_generation=49, stalled=True) == 42
    assert scan_floor_growth_mg_max_words(30, generation=49, inject_generation=49, stalled=True) == 44


def test_eq_refine_warm_inject_skips_non_late_generations(monkeypatch):
    from evolution.mutation_helpers import inject_scan_floor_eq_refine_seed_challengers

    champ = [10, 20, 30, 40, 0x00100073]
    seed = [10, 20, 99, 40, 0x00100073]
    challengers = [list(champ) for _ in range(4)]
    strategies = ["MUT"] * 4
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [list(seed)],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.should_skip_mg_stall_collect_for_warm_eq_refine",
        lambda _c: True,
    )
    n = inject_scan_floor_eq_refine_seed_challengers(
        champ,
        challengers,
        __import__("random").Random(0),
        cycle_no=1,
        generation=0,
        growth_inject_generation=48,
        mg_inject_generation=49,
        strategies=strategies,
    )
    assert n == 0


def test_eq_refine_inject_skips_material_probe_off_late_gen(monkeypatch):
    from evolution.mutation_helpers import inject_scan_floor_eq_refine_seed_challengers

    champ = [10, 20, 30, 40, 0x00100073]
    seed = [10, 20, 99, 40, 0x00100073]
    challengers = [list(champ) for _ in range(4)]
    strategies = ["MUT"] * 4
    calls = {"projection": 0, "locus": 0}

    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [list(seed)],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.should_skip_mg_stall_collect_for_warm_eq_refine",
        lambda _c: False,
    )

    def _proj(*_a, **_k):
        calls["projection"] += 1
        return []

    def _locus(*_a, **_k):
        calls["locus"] += 1
        return []

    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_growth_material_projection_seeds",
        _proj,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_growth_material_locus_probe_variants",
        _locus,
    )

    n = inject_scan_floor_eq_refine_seed_challengers(
        champ,
        challengers,
        __import__("random").Random(0),
        cycle_no=1,
        generation=0,
        growth_inject_generation=48,
        mg_inject_generation=49,
        strategies=strategies,
    )
    assert n >= 1
    assert calls == {"projection": 0, "locus": 0}


def test_eq_refine_late_inject_reuses_merged_pool_within_cycle(monkeypatch):
    from evolution.mutation_helpers import clear_scan_floor_inject_caches, inject_scan_floor_eq_refine_seed_challengers

    clear_scan_floor_inject_caches()
    champ = [10, 20, 30, 40, 0x00100073]
    seeds = [
        [10, 20, 99, 40, 0x00100073],
        [10, 88, 30, 40, 0x00100073],
        [10, 20, 30, 88, 0x00100073],
        [88, 20, 30, 40, 0x00100073],
        [10, 20, 88, 40, 0x00100073],
        [10, 20, 30, 40, 88],
        [10, 88, 99, 40, 0x00100073],
        [88, 88, 30, 40, 0x00100073],
    ]
    calls = {"locus": 0, "perturb": 0}
    # Disable the warm-disk shortcuts so this test isolates the within-cycle
    # merged-pool cache (first call probes, second reuses). The shared caches
    # are cleared/hydrated explicitly: earlier tests populate the same plateau
    # key, and the merged cache is intentionally retained across cycles.
    from evolution.mutation_helpers import _eq_refine_late_merged_cache

    _eq_refine_late_merged_cache.clear()
    monkeypatch.setenv("EE_EQ_REFINE_WARM_SKIP_LOCUS", "0")
    monkeypatch.setenv("EE_EQ_REFINE_WARM_SKIP_PERTURB", "0")
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_eq_refine_late_merged_cache",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.save_eq_refine_late_merged_cache",
        lambda *_a, **_k: None,
    )

    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [list(s) for s in seeds],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.warm_eq_refine_disk_seeds_sufficient",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p: 0.9 if list(p) != list(champ) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )

    def _locus(*_a, **_k):
        calls["locus"] += 1
        return []

    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_growth_material_locus_probe_variants",
        _locus,
    )

    def _perturb(*_a, **_k):
        calls["perturb"] += 1
        return []

    monkeypatch.setattr(
        "evolution.mutation_helpers.refine_eq_seed_one_step_perturb_variants",
        _perturb,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.refine_eq_seed_two_step_perturb_variants",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.record_scan_floor_eq_refine_seeds",
        lambda *a, **k: None,
    )

    for gen in (48, 49):
        inject_scan_floor_eq_refine_seed_challengers(
            champ,
            [list(champ) for _ in range(4)],
            __import__("random").Random(gen),
            cycle_no=7,
            generation=gen,
            growth_inject_generation=48,
            mg_inject_generation=49,
        )
    assert calls["locus"] == 1
    assert calls["perturb"] == 1


def test_should_skip_scan_floor_enrich_when_warm_disk(monkeypatch):
    from evolution.mutation_helpers import should_skip_scan_floor_enrich_equal_length_pool

    champ = [1, 2, 3, 4, 0x00100073]
    seeds = [[10, 20, 99, 40, 0x00100073]] * 8
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [list(s) for s in seeds],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.warm_eq_refine_disk_seeds_sufficient",
        lambda *_a, **_k: True,
    )
    assert should_skip_scan_floor_enrich_equal_length_pool(champ) is True


def test_should_skip_scan_floor_gen0_growth_inject(monkeypatch):
    from evolution.mutation_helpers import should_skip_scan_floor_gen0_growth_inject

    champ = [1, 2, 3, 4, 0x00100073]
    monkeypatch.setenv("EE_SCAN_FLOOR_WARM_SKIP_GEN0_GROWTH", "1")
    monkeypatch.setattr(
        "evolution.mutation_helpers.should_skip_scan_floor_enrich_equal_length_pool",
        lambda *_a, **_k: True,
    )
    assert should_skip_scan_floor_gen0_growth_inject(champ) is True
    monkeypatch.setenv("EE_SCAN_FLOOR_WARM_SKIP_GEN0_GROWTH", "0")
    assert should_skip_scan_floor_gen0_growth_inject(champ) is False


def test_repair_model_guided_deep_uses_eq_refine_seeds(monkeypatch):
    from evolution.mutation_helpers import repair_model_guided_for_gate

    champ = [1, 2, 3, 4, 0x00100073]
    near = [1, 2, 99, 4, 0x00100073]
    eq_seed = [1, 88, 3, 4, 0x00100073]
    repaired = [1, 77, 3, 4, 0x00100073]

    class _Gate:
        def qualifies(self, variant):
            return tuple(variant) in {tuple(repaired), tuple(eq_seed)}

    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda *_a, **_k: [list(eq_seed)],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.expand_l10_lift_eq_champion_grafts",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_wide_imm_sweep_probe_variants",
        lambda *_a, **_k: [list(repaired)],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_imm_perturb_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_tail_wide_imm_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_dual_imm_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_compact_tail_min_position",
        lambda _n: 1,
    )

    def _prom(prog):
        if tuple(prog) == tuple(repaired):
            return 325.84
        if tuple(prog) == tuple(eq_seed):
            return 325.63
        if tuple(prog) == tuple(champ):
            return 325.82
        return 325.60

    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        _prom,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_min_raw_gain",
        lambda _n: 0.01,
    )

    out = repair_model_guided_for_gate(
        near,
        _Gate(),
        max_evals=8,
        champion=champ,
        prefix_count=2,
    )
    assert out == repaired


def test_growth_crossover_pool_cache_survives_cycle_clear(monkeypatch):
    from evolution.mutation_helpers import build_scan_floor_growth_crossover_pool, clear_scan_floor_inject_caches

    champ = [10, 20, 30, 40, 0x00100073]
    fill = [10, 99, 30, 40, 0x00100073]
    hybrid = [10, 77, 30, 40, 0x00100073]
    calls = {"n": 0}

    class _Gate:
        def qualifies(self, _v):
            return True

    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: _Gate(),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda _p: 325.8257,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_min_raw_gain",
        lambda _n: 0.01,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._growth_crossover_candidate",
        lambda *_a, **_k: (calls.__setitem__("n", calls["n"] + 1), list(hybrid))[1],
    )

    clear_scan_floor_inject_caches()
    rng = __import__("random").Random(0)
    first = build_scan_floor_growth_crossover_pool(champ, [fill], rng, cycle_no=10, max_attempts=4)
    assert first == [hybrid]
    built = calls["n"]
    assert built > 0
    clear_scan_floor_inject_caches()
    second = build_scan_floor_growth_crossover_pool(champ, [fill], rng, cycle_no=11, max_attempts=4)
    assert second == [hybrid]
    assert calls["n"] == built


def test_prime_scan_floor_growth_pool_cache_retains_plateau(tmp_path, monkeypatch):
    from evolution.mutation_helpers import prime_scan_floor_growth_pool_cache

    champ = [1, 2, 3, 4, 0x00100073]
    entry = ((100.5, 5), ((1, 2, 3, 4, 0x00100073),), frozenset())
    monkeypatch.setattr(
        "evolution.mutation_helpers._eq_refine_plateau_key",
        lambda _c: (100.5, 5),
    )
    assert prime_scan_floor_growth_pool_cache(champ, entry) == entry


def test_growth_pool_disk_cache_roundtrip(tmp_path, monkeypatch):
    from evolution.mutation_helpers import load_scan_floor_growth_pool_cache, save_scan_floor_growth_pool_cache

    champ = [0x10, 0x20, 0x30, 0x40, 0x00100073]
    pool = [[0x11, 0x20, 0x30, 0x40, 0x00100073], [0x10, 0x21, 0x30, 0x40, 0x00100073]]
    cache_path = tmp_path / "scan_floor_growth_pool_cache.json"
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_pool_cache_path",
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
    save_scan_floor_growth_pool_cache(champ, pool, cross_keys={tuple(pool[0])}, cycle=42)
    loaded = load_scan_floor_growth_pool_cache(champ)
    assert loaded is not None
    assert loaded[0] == (100.5, len(champ))
    assert [list(p) for p in loaded[1]] == pool
    assert tuple(pool[0]) in loaded[2]


def test_eq_refine_late_inject_skips_locus_when_warm_disk(monkeypatch):
    from evolution.mutation_helpers import clear_scan_floor_inject_caches, inject_scan_floor_eq_refine_seed_challengers

    clear_scan_floor_inject_caches()
    champ = [10, 20, 30, 40, 0x00100073]
    seeds = [[10, 20, 99, 40, 0x00100073]] * 8
    calls = {"locus": 0}

    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [list(s) for s in seeds],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.warm_eq_refine_disk_seeds_sufficient",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )

    def _locus(*_a, **_k):
        calls["locus"] += 1
        return []

    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_growth_material_locus_probe_variants",
        _locus,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.refine_eq_seed_one_step_perturb_variants",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.record_scan_floor_eq_refine_seeds",
        lambda *a, **k: None,
    )

    inject_scan_floor_eq_refine_seed_challengers(
        champ,
        [list(champ) for _ in range(4)],
        __import__("random").Random(0),
        cycle_no=3,
        generation=48,
        growth_inject_generation=48,
        mg_inject_generation=49,
    )
    assert calls["locus"] == 0


def test_eq_refine_late_inject_skips_projection_when_warm_disk_seeds(monkeypatch):
    from evolution.mutation_helpers import inject_scan_floor_eq_refine_seed_challengers

    champ = [10, 20, 30, 40, 0x00100073]
    seeds = [
        [10, 20, 99, 40, 0x00100073],
        [10, 88, 30, 40, 0x00100073],
        [10, 20, 30, 88, 0x00100073],
        [88, 20, 30, 40, 0x00100073],
        [10, 20, 88, 40, 0x00100073],
        [10, 20, 30, 40, 88],
        [10, 88, 99, 40, 0x00100073],
        [88, 88, 30, 40, 0x00100073],
    ]
    challengers = [list(champ) for _ in range(8)]
    strategies = ["MUT"] * 8
    calls = {"projection": 0}

    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [list(s) for s in seeds],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p: 0.9 if list(p) != list(champ) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.get_scan_floor_gate",
        lambda _c: type("_G", (), {"qualifies": lambda self, v: True})(),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_growth_material_locus_probe_variants",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.refine_eq_seed_one_step_perturb_variants",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.refine_eq_seed_two_step_perturb_variants",
        lambda *a, **k: [],
    )

    def _proj(*_a, **_k):
        calls["projection"] += 1
        return []

    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_growth_material_projection_seeds",
        _proj,
    )

    n = inject_scan_floor_eq_refine_seed_challengers(
        champ,
        challengers,
        __import__("random").Random(0),
        cycle_no=1,
        generation=48,
        growth_inject_generation=48,
        mg_inject_generation=49,
        strategies=strategies,
    )
    assert n >= 1
    assert calls["projection"] == 0


def test_collect_growth_material_projection_seeds_prom_ranked(monkeypatch):
    from evolution.mutation_helpers import collect_growth_material_projection_seeds

    champ = [1, 2, 3, 4, 0x00100073]
    long_a = champ + [9, 10, 11]
    long_b = champ + [12, 13]
    proj_a = [1, 2, 88, 4, 0x00100073]
    proj_b = [1, 2, 77, 4, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()

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
        lambda _c, material, **kwargs: [proj_a, proj_b],
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

    out = collect_growth_material_projection_seeds(champ, max_seeds=2, cycle_no=3, gate=gate)
    assert len(out) == 2
    assert out[0] == proj_b
    assert out[1] == proj_a


def test_scan_floor_stalled_challenger_max_words_matches_mg_cap():
    from evolution.mutation_helpers import scan_floor_stalled_challenger_max_words

    assert scan_floor_stalled_challenger_max_words(40, generation=0, mg_inject_generation=49) == 48
    assert scan_floor_stalled_challenger_max_words(40, generation=49, mg_inject_generation=49) == 52


def test_filter_scan_floor_growth_programs_by_max_words():
    from evolution.mutation_helpers import filter_scan_floor_growth_programs_by_max_words

    pool = [[1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7]]
    out = filter_scan_floor_growth_programs_by_max_words(pool, max_words=5)
    assert out == [[1, 2, 3], [1, 2, 3, 4, 5]]


def test_cap_scan_floor_challenger_lengths_projects_long(monkeypatch):
    from evolution.mutation_helpers import cap_scan_floor_challenger_lengths

    champ = [1, 2, 3, 4, 0x00100073]
    long_prog = champ + [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    projected = [1, 2, 99, 4, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()

    monkeypatch.setattr(
        "evolution.mutation_helpers.shrink_scan_floor_challenger_to_max_words",
        lambda prog, _c, **_: list(projected) if len(prog) > 5 else list(prog),
    )

    out = cap_scan_floor_challenger_lengths(
        [champ, long_prog],
        champ,
        max_words=7,
        gate=gate,
        cycle_no=1,
    )
    assert out[0] == champ
    assert out[1] == projected


def test_select_diverse_l10_lift_inject_programs_prefers_near_target(monkeypatch):
    from evolution.mutation_helpers import select_diverse_l10_lift_inject_programs

    champ = [1, 2, 3, 4, 0x00100073]
    long_prog = champ + [9, 10, 11, 12, 13, 14, 15, 16]
    near_prog = champ + [9, 10]
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p: (1.0 if len(p) == len(long_prog) else 0.95 if len(p) > len(champ) else 0.8),
    )
    out = select_diverse_l10_lift_inject_programs(champ, [long_prog, near_prog], n_take=2, cycle_no=0)
    assert len(out) == 2
    assert out[0] == long_prog
    assert out[1] == near_prog


def test_select_growth_inject_slot_variants_seeds_l10_lift_when_weak(monkeypatch):
    from evolution.mutation_helpers import select_growth_inject_slot_variants

    champ = [1, 2, 3, 4, 0x00100073]
    eq_a = [1, 2, 88, 4, 0x00100073]
    l10_lift = champ + [9, 10, 11, 12, 13, 14, 15, 16]
    pool = [eq_a, l10_lift]
    monkeypatch.setenv("EE_SCAN_FLOOR_GROWTH_LONG_INJECT_CAP", "2")
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda prog, level: 1.0 if level == 10 and len(prog) > len(champ) else 0.8,
    )
    out = select_growth_inject_slot_variants(champ, pool, n_slots=4, cycle_no=0, stalled=True, max_long_words=20)
    assert any(len(p) > len(champ) for p in out)
    assert out[0] == l10_lift


def test_select_growth_inject_slot_variants_caps_long_when_stalled(monkeypatch):
    from evolution.mutation_helpers import select_growth_inject_slot_variants

    champ = [1, 2, 3, 4, 0x00100073]
    eq_a = [1, 2, 88, 4, 0x00100073]
    eq_b = [1, 2, 77, 4, 0x00100073]
    long_a = champ + [9, 10, 11]
    long_b = champ + [12, 13, 14, 15]
    pool = [eq_a, eq_b, long_a, long_b]
    monkeypatch.setenv("EE_SCAN_FLOOR_GROWTH_LONG_INJECT_CAP", "1")

    out = select_growth_inject_slot_variants(champ, pool, n_slots=5, cycle_no=0, stalled=True, max_long_words=6)
    assert len(out) == 5
    assert sum(len(p) > len(champ) for p in out) <= 1
    assert all(len(p) == len(champ) for p in out[:2])
    assert all(len(p) <= 6 for p in out)


def test_record_l10_lift_growth_discoveries(monkeypatch):
    from evolution.mutation_helpers import record_l10_lift_growth_discoveries

    champ = [1, 2, 3, 4, 0x00100073]
    l10_prog = champ + [9, 10, 11, 12]
    recorded: list[list[int]] = []
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_l10_lift_preserve_slot",
        lambda _c, prog, **k: len(prog) > len(champ),
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.record_scan_floor_growth_material",
        lambda _c, prog, cycle=0: recorded.append(list(prog)) or True,
    )
    state: dict = {}
    n = record_l10_lift_growth_discoveries(champ, [list(champ), list(l10_prog)], cycle=7, state=state, record_cap=2)
    assert n == 1
    assert recorded == [l10_prog]
    assert state == {"cycle": 7, "recorded": 1}


def test_enumerate_l10_lift_shrink_intermediates(monkeypatch):
    from evolution.mutation_helpers import enumerate_l10_lift_shrink_intermediates

    champ = [1, 2, 3, 4, 0x00100073]
    long_prog = champ + [9, 10, 11, 12, 13]
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p: 0.95 if len(p) > len(champ) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_compress_drop_qualifies",
        lambda _c, _v, **k: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, **k: True,
    )
    mids = enumerate_l10_lift_shrink_intermediates(champ, long_prog, max_steps=3)
    assert mids
    assert all(len(p) > len(champ) for p in mids)
    assert len(mids[0]) < len(long_prog)


def test_maybe_l10_lift_post_mutation_compress(monkeypatch):
    from evolution.mutation_helpers import maybe_l10_lift_post_mutation_compress

    champ = [1, 2, 3, 4, 0x00100073]
    parent = champ + [9, 10, 11, 12]
    mutated = parent + [13, 14]
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p: 1.0 if len(p) > len(champ) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda _c, _v, **k: list(parent),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, **k: True,
    )
    out = maybe_l10_lift_post_mutation_compress(champ, parent, mutated)
    assert out == parent


def test_record_l10_lift_growth_discoveries_check_cap(monkeypatch):
    from evolution.mutation_helpers import record_l10_lift_growth_discoveries

    champ = [1, 2, 3, 4, 0x00100073]
    long1 = champ + [9, 10, 11]
    long2 = champ + [12, 13, 14, 15]
    long3 = champ + [16, 17]
    checked: list[list[int]] = []
    monkeypatch.setenv("EE_L10_LIFT_DISCOVERY_CHECK_CAP", "2")
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda _p: 0.8,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_l10_lift_preserve_slot",
        lambda _c, prog, **k: checked.append(list(prog)) or True,
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.record_scan_floor_growth_material",
        lambda _c, prog, cycle=0: True,
    )
    state: dict = {}
    record_l10_lift_growth_discoveries(
        champ,
        [list(long3), list(long2), list(long1)],
        cycle=8,
        state=state,
        record_cap=4,
    )
    assert len(checked) == 2
    assert all(len(p) > len(champ) for p in checked)


def test_swap_inject_overlength_for_eq_seeds(monkeypatch):
    from evolution.mutation_helpers import swap_inject_overlength_for_eq_seeds

    champ = [10, 20, 30, 40, 0x00100073]
    seed = [10, 20, 99, 40, 0x00100073]
    long_prog = champ + [11, 12, 13, 14]
    challengers = [list(champ), list(long_prog), list(champ)]
    strategies = ["M", "M", "M"]
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [list(seed)],
    )
    n = swap_inject_overlength_for_eq_seeds(
        champ,
        challengers,
        cycle_no=0,
        generation=48,
        growth_inject_generation=48,
        mg_inject_generation=49,
        strategies=strategies,
    )
    assert n == 1
    assert challengers[1] == seed
    assert strategies[1] == "EQ_LENGTH_SWAP"


def test_swap_inject_overlength_preserves_l10_lift_long(monkeypatch):
    from evolution.mutation_helpers import swap_inject_overlength_for_eq_seeds

    champ = [10, 20, 30, 40, 0x00100073]
    seed = [10, 20, 99, 40, 0x00100073]
    l10_lift = champ + [11, 12, 13, 14, 15, 16]
    l10_lift_t = tuple(l10_lift)
    challengers = [list(l10_lift), list(champ) + [1, 2], list(champ)]
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [list(seed)],
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda prog, level: 1.0 if level == 10 and tuple(prog) == l10_lift_t else 0.8,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, v, **k: tuple(v) == l10_lift_t,
    )
    n = swap_inject_overlength_for_eq_seeds(
        champ,
        challengers,
        cycle_no=0,
        generation=48,
        growth_inject_generation=48,
        category_saturated=True,
    )
    assert challengers[0] == l10_lift
    assert n == 1
    assert challengers[1] == seed


def test_swap_inject_overlength_swaps_all_when_category_saturated(monkeypatch):
    from evolution.mutation_helpers import swap_inject_overlength_for_eq_seeds

    champ = [10, 20, 30, 40, 0x00100073]
    seed_a = [10, 20, 99, 40, 0x00100073]
    seed_b = [10, 88, 30, 40, 0x00100073]
    challengers = [
        list(champ) + [1, 2],
        list(champ) + [3, 4],
        list(champ),
    ]
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_scan_floor_eq_refine_seeds",
        lambda _c, max_seeds=8, probe_material=True: [seed_a, seed_b],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, **k: False,
    )
    n = swap_inject_overlength_for_eq_seeds(
        champ,
        challengers,
        cycle_no=0,
        generation=49,
        growth_inject_generation=48,
        mg_inject_generation=49,
        category_saturated=True,
    )
    assert n == 2
    assert all(len(p) == len(champ) for p in challengers[:2])


def test_dedupe_programs_for_eval_maps_results():
    from evolution.mutation_helpers import dedupe_programs_for_eval, expand_eval_results

    a = [1, 2, 3]
    b = [4, 5, 6]
    programs = [a, b, list(a), b]
    unique, index_map = dedupe_programs_for_eval(programs)
    assert unique == [a, b]
    assert index_map == [0, 1, 0, 1]
    results = [{"fit": 1.0}, {"fit": 2.0}]
    expanded = expand_eval_results(results, index_map)
    assert expanded == [{"fit": 1.0}, {"fit": 2.0}, {"fit": 1.0}, {"fit": 2.0}]


def test_select_growth_mg_anchor_fill_prefers_tail_diff_on_inject():
    from evolution.mutation_helpers import select_growth_mg_anchor_fill

    champ = [10] * 26 + [20, 30] + [0x00100073] * 12
    low_diff = list(champ)
    low_diff[27] = 99
    high_diff = list(champ)
    for i in range(27, 31):
        high_diff[i] = 50 + i
    high_diff = high_diff + [11, 12, 13, 14]
    fills = [low_diff, high_diff]
    pick = select_growth_mg_anchor_fill(
        champ,
        fills,
        cycle_no=0,
        generation=49,
        inject_generation=49,
    )
    assert pick == high_diff
    pick_rot = select_growth_mg_anchor_fill(
        champ,
        fills,
        cycle_no=1,
        generation=49,
        inject_generation=49,
    )
    assert pick_rot in fills
    pick_second = select_growth_mg_anchor_fill(
        champ,
        fills,
        cycle_no=0,
        generation=49,
        inject_generation=49,
        rank_offset=1,
    )
    assert pick_second == low_diff


def test_scan_floor_late_inject_generation_growth_and_mg():
    from evolution.mutation_helpers import scan_floor_late_inject_generation

    assert scan_floor_late_inject_generation(48, growth_inject_generation=48, mg_inject_generation=49)
    assert scan_floor_late_inject_generation(49, growth_inject_generation=48, mg_inject_generation=49)
    assert not scan_floor_late_inject_generation(0, growth_inject_generation=48, mg_inject_generation=49)
    assert not scan_floor_late_inject_generation(25, growth_inject_generation=48, mg_inject_generation=49)


def test_collect_l10_lift_pair_drop_eq_programs(monkeypatch):
    from evolution.mutation_helpers import collect_l10_lift_pair_drop_eq_programs

    champ = [1, 2, 3, 4, 0x00100073]
    eq_lift = [1, 2, 88, 4, 0x00100073]
    variant = [1, 2, 88, 4, 9, 10, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p: (0.95 if tuple(p) == tuple(eq_lift) else 1.0 if len(p) > len(champ) else 0.8),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_compress_drop_qualifies",
        lambda _c, v, **k: tuple(v) == tuple(eq_lift),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, **k: True,
    )
    out = collect_l10_lift_pair_drop_eq_programs(champ, variant, gate=gate)
    assert out == [eq_lift]


def test_collect_l10_lift_pair_drop_respects_trial_cap(monkeypatch):
    from evolution.mutation_helpers import collect_l10_lift_pair_drop_eq_programs

    champ = [1, 2, 3, 4, 0x00100073]
    variant = [1, 2, 88, 4, 9, 10, 11, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()
    calls = {"n": 0}

    def _l10(p):
        calls["n"] += 1
        return 0.95 if len(p) == len(champ) else 1.0

    monkeypatch.setenv("EE_L10_PAIR_DROP_TRIAL_CAP", "8")
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        _l10,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_compress_drop_qualifies",
        lambda _c, _v, **k: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, **k: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda _p: 320.0,
    )
    out = collect_l10_lift_pair_drop_eq_programs(champ, variant, gate=gate)
    assert calls["n"] <= 12
    assert len(out) <= 8


def test_enrich_growth_inject_adds_pair_drop_eq_when_l10_weak(monkeypatch):
    from evolution.mutation_helpers import enrich_growth_inject_equal_length_pool

    champ = [1, 2, 3, 4, 0x00100073]
    near = champ + [9, 10]
    eq_lift = [1, 2, 88, 4, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p: 1.0 if len(p) > len(champ) else 0.825 if list(p) == eq_lift else 0.8,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_l10_lift_pair_drop_eq_programs",
        lambda _c, _v, **k: [list(eq_lift)],
    )
    out = enrich_growth_inject_equal_length_pool(champ, [near], gate=gate, cycle_no=0)
    assert eq_lift in out
    assert out.index(eq_lift) < out.index(near)


def test_enrich_growth_inject_equal_length_pool_adds_projections(monkeypatch):
    from evolution.mutation_helpers import enrich_growth_inject_equal_length_pool

    champ = [1, 2, 3, 4, 0x00100073]
    long_prog = champ + [9, 10]
    projected = [1, 2, 99, 4, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()

    monkeypatch.setattr(
        "evolution.mutation_helpers.equal_length_from_longer_growth",
        lambda _c, _g, **_: list(projected),
    )
    # Patch the memoized wrapper directly: the per-cycle curriculum cache is
    # module-level and shared across tests in one session, so patching only
    # curriculum_verify_mean lets stale cached values from earlier tests leak in.
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p, level=10: 0.9 if level == 10 and list(p) == projected else 0.8,
    )

    out = enrich_growth_inject_equal_length_pool(champ, [long_prog], gate=gate, cycle_no=3)
    assert projected in out
    assert long_prog in out
    assert out.index(projected) < out.index(long_prog)


def test_expand_l10_lift_eq_champion_grafts(monkeypatch):
    from evolution.mutation_helpers import expand_l10_lift_eq_champion_grafts

    champ = [1, 2, 3, 4, 0x00100073]
    base = [1, 2, 88, 99, 0x00100073]
    grafted = [1, 2, 3, 99, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p: 0.9 if list(p) == grafted else 0.85 if tuple(p) == tuple(base) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda p: 321.0 if list(p) == grafted else 318.0,
    )
    out = expand_l10_lift_eq_champion_grafts(champ, base, gate=gate, min_l10_margin=0.02)
    assert grafted in out


def test_crossbreed_l10_prom_eq_refine_variants(monkeypatch):
    from evolution.mutation_helpers import crossbreed_l10_prom_eq_refine_variants

    champ = [1, 2, 3, 4, 0x00100073]
    high_l10 = [1, 2, 88, 99, 0x00100073]
    prom_seed = [1, 2, 77, 99, 0x00100073]
    hybrid = [1, 2, 88, 99, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p: (0.9 if list(p) in (high_l10, hybrid) else 0.825 if list(p) == prom_seed else 0.8),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda p: 322.0 if list(p) == hybrid else 318.0 if list(p) == prom_seed else -25.0,
    )
    out = crossbreed_l10_prom_eq_refine_variants(champ, [high_l10, prom_seed], gate=gate, max_variants=4)
    assert hybrid in out


def test_l10_lift_eq_bootstrap_cache(monkeypatch):
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
    # gap-2 parent: bootstrap skips gap>=3 sources when shrink intermediates
    # are disabled (production pair-drops the gap-1/2 intermediates instead).
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
    assert collect_l10_lift_eq_refine_bootstrap_seeds(champ, gate=gate) == [eq_lift]
    assert collect_l10_lift_eq_refine_bootstrap_seeds(champ, gate=gate) == [eq_lift]
    assert calls["n"] == 1


def test_l10_lift_bootstrap_cache_anchor_isolation(monkeypatch):
    """Same prom plateau, different anchor genomes must not share bootstrap cache.

    Two anchors on the same prom plateau pull from the SAME shared growth-material
    pool (material selection is prom/length/capability based, not genome based), so
    ``material_sig`` is identical for both. A cache keyed on material/plateau only
    would then let anchor B read anchor A's seeds -- the anchor tuple must be the
    primary cache key (W1b).
    """
    from evolution.mutation_helpers import (
        _l10_lift_eq_bootstrap_cache,
        clear_scan_floor_inject_caches,
        collect_l10_lift_eq_refine_bootstrap_seeds,
    )

    clear_scan_floor_inject_caches()
    _l10_lift_eq_bootstrap_cache.clear()
    champ_a = [1, 2, 3, 4, 0x00100073]
    champ_b = [1, 2, 99, 4, 0x00100073]
    eq_a = [1, 2, 88, 4, 0x00100073]
    eq_b = [1, 2, 77, 4, 0x00100073]
    # Shared growth material: production buckets growth material by the certified
    # scan-floor champion (champion_rollback._growth_material_storage_key), so
    # distinct same-prom anchors really do read one shared pool -- model that
    # aliasing worst case here instead of letting material_sig separate the keys.
    shared_material = [[1, 2, 3, 4, 9, 10]]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()
    calls = {"a": 0, "b": 0}

    def _pair_drop(champ, _src, **k):
        if tuple(champ) == tuple(champ_a):
            calls["a"] += 1
            return [list(eq_a)]
        calls["b"] += 1
        return [list(eq_b)]

    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_anchor_program",
        lambda current: list(current),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda p: (
            1.0 if len(p) > len(champ_a) else 0.825 if tuple(p) == tuple(eq_a) or tuple(p) == tuple(eq_b) else 0.8
        ),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._eq_refine_plateau_key",
        lambda _anchor: (325.0, len(champ_a)),
    )
    # gap-2 parents: bootstrap skips gap>=3 sources when shrink intermediates
    # are disabled (production pair-drops the gap-1/2 intermediates instead).
    # Same-prom anchors share one growth-material pool, so material_sig matches.
    monkeypatch.setattr(
        "evolution.champion_rollback.collect_growth_material_compress_parents",
        lambda _champ, max_parents=16: [list(p) for p in shared_material],
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
    anchor_a = tuple(int(x) for x in champ_a)
    anchor_b = tuple(int(x) for x in champ_b)
    out_a = collect_l10_lift_eq_refine_bootstrap_seeds(champ_a, gate=gate)
    # Step-wise (W1b): right after anchor A the cache holds EXACTLY one entry
    # and that entry is keyed by A's genome (membership, not position).
    keys_after_a = list(_l10_lift_eq_bootstrap_cache)
    assert len(keys_after_a) == 1, f"expected exactly one entry after call A: {keys_after_a!r}"
    assert anchor_a in keys_after_a[0], f"call A entry not keyed by anchor A: {keys_after_a!r}"
    out_b = collect_l10_lift_eq_refine_bootstrap_seeds(champ_b, gate=gate)
    assert out_a == [eq_a]
    assert out_b == [eq_b]
    assert calls["a"] == 1
    assert calls["b"] == 1
    # Contract (W1b): the cache key must carry the anchor genome itself (spec:
    # cache_key = (anchor_key, material_sig, cap)). A material/plateau-only key
    # would alias these two same-prom anchors; assert presence, not position, so a
    # harmless key re-ordering is not what this test polices.
    cached_keys = list(_l10_lift_eq_bootstrap_cache)
    assert cached_keys, "bootstrap cache must hold an entry per anchor"
    assert any(anchor_a in key for key in cached_keys), "anchor A not cache-keyed"
    assert any(anchor_b in key for key in cached_keys), "anchor B not cache-keyed"
    for key in cached_keys:
        assert anchor_a in key or anchor_b in key, f"unanchored bootstrap cache key: {key!r}"
    # Hardening (W1b): exact one-entry-per-anchor cardinality -- no sharing, no
    # eviction -- with the two keys differing ONLY by the anchor (material_sig
    # and cap are held identical by the shared-pool premise above).
    assert len(cached_keys) == 2, f"expected 2 anchor-isolated entries: {cached_keys!r}"
    key_a = next(k for k in cached_keys if anchor_a in k)
    key_b = next(k for k in cached_keys if anchor_b in k)
    assert (set(key_a) - {anchor_a}) == (
        set(key_b) - {anchor_b}
    ), "shared-pool premise: the two cache keys must differ only by the anchor"
    # Full key contract (spec): cache_key = (anchor_key, material_sig, cap).
    # Pin the SHAPE as well as anchor membership: a key of (anchor,) alone would
    # still isolate these two anchors yet serve stale seeds the moment the
    # growth-material pool or cap changes, so that must not pass as a "fix".
    assert (
        len(key_a) == 3 and len(key_b) == 3
    ), f"cache_key must be (anchor_key, material_sig, cap): {key_a!r} {key_b!r}"
    shared_a = set(key_a) - {anchor_a}
    shared_b = set(key_b) - {anchor_b}
    assert (
        len(shared_a) == 2 and shared_a == shared_b
    ), f"keys must share exactly material_sig + cap: {key_a!r} {key_b!r}"
    assert any(isinstance(part, int) for part in shared_a), "cap (int) missing from key"
    assert any(
        isinstance(part, tuple) and part and isinstance(part[0], tuple) for part in shared_a
    ), f"material_sig (tuple of program tuples) missing from key: {key_a!r}"
    # Value isolation: each anchor's entry holds only its own seeds.
    assert _l10_lift_eq_bootstrap_cache[key_a] == (tuple(eq_a),)
    assert _l10_lift_eq_bootstrap_cache[key_b] == (tuple(eq_b),)


def test_collect_l10_lift_eq_refine_bootstrap_seeds(monkeypatch):
    from evolution.mutation_helpers import collect_l10_lift_eq_refine_bootstrap_seeds

    champ = [1, 2, 3, 4, 0x00100073]
    near = [1, 2, 88, 4, 9, 10, 0x00100073]
    eq_lift = [1, 2, 88, 4, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()
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
        lambda _c, max_parents=16: [list(near)],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.collect_l10_lift_pair_drop_eq_programs",
        lambda _c, _v, **k: [list(eq_lift)],
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.record_scan_floor_eq_refine_seeds",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.persist_l10_lift_bootstrap_intermediates",
        lambda *_a, **_k: 0,
    )
    out = collect_l10_lift_eq_refine_bootstrap_seeds(champ, gate=gate, max_seeds=4)
    assert out == [eq_lift]


def test_scan_floor_eq_inject_sort_key_prefers_prom_at_scan_floor(monkeypatch):
    from evolution.mutation_helpers import clear_scan_floor_inject_caches, scan_floor_eq_inject_sort_key

    clear_scan_floor_inject_caches()
    champ = [1] * 26 + [0x00100073]
    high_l10 = list(champ)
    high_l10[20] = 88
    high_prom = list(champ)
    high_prom[21] = 99
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda words: int(words) == len(champ),
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda prog: 326.0 if prog[21] == 99 else 325.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda prog, level: 0.9 if level == 10 and prog[20] == 88 else 0.8,
    )
    assert scan_floor_eq_inject_sort_key(champ, high_prom) > scan_floor_eq_inject_sort_key(champ, high_l10)


def test_scan_floor_eq_inject_sort_key_prefers_l10_on_prom_tie(monkeypatch):
    from evolution.mutation_helpers import clear_scan_floor_inject_caches, scan_floor_eq_inject_sort_key

    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 4, 0x00100073]
    a = [1, 2, 3, 4, 0x00100073]
    b = [1, 2, 88, 4, 0x00100073]
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda _p: 325.627,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda prog, level: 0.8 if level == 10 and prog[2] == 88 else 0.75,
    )
    assert scan_floor_eq_inject_sort_key(champ, b) > scan_floor_eq_inject_sort_key(champ, a)


def test_compress_mg_prefers_champion_match_drop_positions(monkeypatch):
    from evolution.mutation_helpers import compress_model_guided_growth_candidate

    champ = [1, 2, 3, 4, 0x00100073]
    long = [1, 2, 3, 4, 99, 10, 11, 12, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: len(v) == len(champ)})()
    drops: list[int] = []

    def _drop_qual(_c, trial):
        drops.append(len(trial))
        return True

    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda prog, level: 0.9 if len(prog) > len(champ) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, **k: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_compress_drop_qualifies",
        _drop_qual,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.semantic_compress",
        lambda p: list(p),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _n: False,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        lambda _p: 325.627,
    )
    compress_model_guided_growth_candidate(champ, long, gate=gate, max_drop_trials=4)
    assert drops and drops[0] == len(long) - 1


def test_collect_l10_lift_growth_candidates(monkeypatch):
    from evolution.mutation_helpers import collect_scan_floor_l10_lift_growth_candidates

    champ = [1, 2, 3, 4, 0x00100073]
    bounded = champ + [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda prog, level: 0.95 if level == 10 and len(prog) > len(champ) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.l30_order_sensitive_variant",
        lambda _p: None,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.l30_l8_mem_add_variant",
        lambda _p: None,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.l30_l10_bounded_sum_variant",
        lambda _p: list(bounded),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.l30_l10_max_in_sum_loop_variant",
        lambda _p: None,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_heldout_qualifies",
        lambda _c, _v, **k: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, **k: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.enumerate_l10_lift_shrink_intermediates",
        lambda *_a, **_k: [],
    )
    out = collect_scan_floor_l10_lift_growth_candidates(champ)
    assert out and out[0] == bounded


def test_l10_lift_loop_prefix_semantic_compress_shrinks_preloop():
    import json
    import os

    os.environ.setdefault("EE_USE_GPU", "0")
    from evolution.fitness_helpers import decode_program_words
    from evolution.terminal_stagnation import (
        curriculum_verify_mean,
        l10_lift_loop_preserving_semantic_compress,
        l30_l10_sum_max_preloop_insert_variant,
    )

    row = json.loads(open("multi_cycle_results.jsonl").readlines()[-1])
    champ = decode_program_words(row["champion_program"])
    preloop = l30_l10_sum_max_preloop_insert_variant(list(champ))
    if not preloop:
        pytest.skip("preloop unavailable on live champion")
    shrunk = l10_lift_loop_preserving_semantic_compress(list(preloop))
    assert shrunk is not None
    assert len(shrunk) < len(preloop)
    assert float(curriculum_verify_mean(shrunk, 10)) >= 0.99


def test_l10_sum_max_preloop_insert_prefers_mid_ebreak(monkeypatch):
    import json
    import os

    os.environ.setdefault("EE_USE_GPU", "0")
    from evolution.fitness_helpers import decode_program_words
    from evolution.terminal_stagnation import (
        _l30_bounded_sum_max_loop_seq,
        curriculum_verify_mean,
        l30_l10_sum_max_preloop_insert_variant,
    )

    row = json.loads(open("multi_cycle_results.jsonl").readlines()[-1])
    champ = decode_program_words(row["champion_program"])
    if len(champ) < 24:
        pytest.skip("champion too short for preloop insert probe")
    base_l10 = float(curriculum_verify_mean(champ, 10))
    variant = l30_l10_sum_max_preloop_insert_variant(list(champ))
    assert variant is not None
    assert len(variant) > len(champ)
    assert float(curriculum_verify_mean(variant, 10)) > base_l10 + 0.05
    # Replace-tail at mid-body EBREAK should beat insert+keep-dead-tail on length.
    assert len(variant) <= len(champ) + len(_l30_bounded_sum_max_loop_seq()) + 1


def test_expand_growth_material_for_grafts_seeds_preloop_when_l10_weak(monkeypatch):
    from evolution.mutation_helpers import _expand_growth_material_for_grafts

    champ = [1, 2, 3, 0x00100073]
    preloop = [1, 2, 3, 9, 10, 11, 12, 0x00100073]
    gate = type("_G", (), {"qualifies": lambda self, v: True})()
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda prog, level: 1.0 if level == 10 and len(prog) > len(champ) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.l30_l10_sum_max_preloop_insert_variant",
        lambda _p: list(preloop),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.equal_length_from_longer_growth",
        lambda *_a, **_k: None,
    )
    out = _expand_growth_material_for_grafts(champ, [], gate=gate, cycle_no=1)
    assert preloop in out


def test_l10_lift_compress_keeps_higher_l10_on_live_champion():
    import json
    import os

    os.environ.setdefault("EE_USE_GPU", "0")
    from evolution.fitness_helpers import decode_program_words
    from evolution.mutation_helpers import compress_model_guided_growth_candidate
    from evolution.terminal_stagnation import (
        curriculum_verify_mean,
        get_scan_floor_gate,
        l30_l10_sum_max_preloop_insert_variant,
    )

    row = json.loads(open("multi_cycle_results.jsonl").readlines()[-1])
    champ = decode_program_words(row["champion_program"])
    if len(champ) < 24:
        pytest.skip("champion too short for L10 compress probe")
    grown = l30_l10_sum_max_preloop_insert_variant(list(champ))
    assert grown is not None
    gate = get_scan_floor_gate(champ)
    from evolution.mutation_helpers import scan_floor_growth_completion_qualifies

    out = compress_model_guided_growth_candidate(champ, grown, gate=gate, max_drop_trials=64)
    assert len(out) > len(champ)
    assert scan_floor_growth_completion_qualifies(champ, out, gate=gate)
    assert float(curriculum_verify_mean(out, 10)) >= 0.99
    assert float(curriculum_verify_mean(grown, 10)) >= 0.99


def test_collect_l10_lift_includes_compressed_shorter(monkeypatch):
    from evolution.mutation_helpers import collect_scan_floor_l10_lift_growth_candidates

    champ = [1, 2, 3, 4, 0x00100073]
    long = champ + [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    short = champ + [9, 10, 11, 12, 13, 14]
    monkeypatch.setattr(
        "evolution.terminal_stagnation.curriculum_verify_mean",
        lambda prog, level: 0.95 if level == 10 and len(prog) > len(champ) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.l30_l10_sum_max_preloop_insert_variant",
        lambda _p: list(long),
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.l30_order_sensitive_variant",
        lambda _p: None,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.l30_l8_mem_add_variant",
        lambda _p: None,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.l30_l10_bounded_sum_variant",
        lambda _p: None,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.l30_l10_max_in_sum_loop_variant",
        lambda _p: None,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_heldout_qualifies",
        lambda _c, _v, **k: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_growth_completion_qualifies",
        lambda _c, _v, **k: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.compress_model_guided_growth_candidate",
        lambda _c, variant, **k: list(short) if len(variant) == len(long) else list(variant),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.enumerate_l10_lift_shrink_intermediates",
        lambda *_a, **_k: [],
    )
    out = collect_scan_floor_l10_lift_growth_candidates(champ)
    assert out and out[0] == short


def test_l10_lift_compress_protects_sum_max_loop():
    from evolution.mutation_helpers import _l10_lift_compress_protected_positions
    from evolution.terminal_stagnation import _l30_bounded_sum_max_loop_body

    body = _l30_bounded_sum_max_loop_body()
    variant = [1, 2, 3] + body + [0x00100073]
    protected = _l10_lift_compress_protected_positions(variant)
    assert protected == set(range(3, 3 + len(body)))


def test_refine_eq_seed_two_step_perturb_chains(monkeypatch):
    from evolution.mutation_helpers import refine_eq_seed_two_step_perturb_variants

    champ = [1, 2, 3, 4, 0x00100073]
    calls = {"n": 0}

    def _one_step(seeds, rng, max_evals=8, gate=None, champion=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return [[1, 2, 99, 4, 0x00100073]]
        return [[1, 2, 99, 77, 0x00100073]]

    monkeypatch.setattr(
        "evolution.mutation_helpers.refine_eq_seed_one_step_perturb_variants",
        _one_step,
    )
    gate = type("_G", (), {"qualifies": lambda self, v: True})()
    out = refine_eq_seed_two_step_perturb_variants([champ], __import__("random").Random(0), max_evals=2, gate=gate)
    assert calls["n"] == 2
    assert out == [[1, 2, 99, 77, 0x00100073]]


def test_rank_picked_probes_prefers_l10_when_champion_weak(monkeypatch):
    from evolution.terminal_stagnation import _rank_scan_floor_picked_probes

    champ = [1, 2, 3, 0x00100073]
    low = [1, 2, 4, 0x00100073]
    high = [1, 2, 5, 0x00100073]

    def _l10(prog, level, n_seeds=8):
        if int(level) != 10:
            return 0.99
        if tuple(prog) == tuple(champ):
            return 0.8
        if tuple(prog) == tuple(low):
            return 0.82
        if tuple(prog) == tuple(high):
            return 0.95
        return 0.8

    monkeypatch.setattr("evolution.terminal_stagnation.curriculum_verify_mean", _l10)
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_capability_score",
        lambda _c, _v: 325.0,
    )
    ranked = _rank_scan_floor_picked_probes(champ, [low, high], budget=2)
    assert ranked[0] == high


def test_probe_rank_key_prefers_l10_when_champion_weak(monkeypatch):
    from evolution.terminal_stagnation import (
        _clear_scan_floor_probe_rank_champion,
        _scan_floor_probe_rank_key,
        _set_scan_floor_probe_rank_champion,
    )

    champ = [1, 2, 3, 0x00100073]
    low = [1, 2, 4, 0x00100073]
    high = [1, 2, 5, 0x00100073]

    def _l10(prog, level, n_seeds=8):
        if int(level) != 10:
            return 0.99
        if tuple(prog) == tuple(champ):
            return 0.8
        if tuple(prog) == tuple(low):
            return 0.82
        if tuple(prog) == tuple(high):
            return 0.95
        return 0.8

    monkeypatch.setattr("evolution.terminal_stagnation.curriculum_verify_mean", _l10)
    monkeypatch.setattr(
        "evolution.terminal_stagnation.heldout_l30_verify_mean",
        lambda *_a, **_k: 0.99,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.compact_curriculum_side_score",
        lambda _p: 0.87,
    )
    _set_scan_floor_probe_rank_champion(champ)
    assert _scan_floor_probe_rank_key(low) < _scan_floor_probe_rank_key(high)
    _clear_scan_floor_probe_rank_champion()


def test_persist_l10_lift_bootstrap_intermediates(monkeypatch):
    from evolution.mutation_helpers import persist_l10_lift_bootstrap_intermediates

    recorded: list[list[int]] = []

    champ = [1, 2, 3, 4, 0x00100073]

    def _l10(prog):
        return 0.94 if len(prog) > len(champ) else 0.8

    monkeypatch.setattr("evolution.mutation_helpers._curriculum_l10_cached", _l10)
    monkeypatch.setattr(
        "evolution.champion_rollback.record_scan_floor_growth_material",
        lambda _c, prog, **_: recorded.append(list(prog)) or True,
    )
    n = persist_l10_lift_bootstrap_intermediates(
        champ,
        [champ + [99], champ + [88, 77], champ],
        cycle=1,
        max_record=2,
    )
    assert n == 2
    assert len(recorded) == 2
