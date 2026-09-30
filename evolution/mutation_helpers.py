"""Mutation strategy helpers extracted from EvolutionEngine.

Functions for strategy selection, model-guided generation, and mutation
biasing. Each function that depended on ``self`` receives the engine
instance as the first parameter.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Pure / near-pure helpers (no or minimal engine state)
# ------------------------------------------------------------------

_compress_mg_cache: dict[tuple[tuple[int, ...], tuple[int, ...], int], tuple[int, ...]] = {}
_fresh_compress_source_union: dict[tuple[int, tuple[int, ...]], set[tuple[int, ...]]] = {}
_fresh_compress_core_cache: dict[tuple, tuple[int, tuple[tuple[int, ...], ...]]] = {}
_fresh_compress_last_cycle: int = -1
_fresh_compress_plateau_key: tuple[float, int] | None = None
_growth_crossover_pool_cache: dict[tuple, tuple[tuple[int, ...], ...]] = {}
_growth_compress_pool_cache: dict[tuple, tuple[tuple[int, ...], ...]] = {}
_GROWTH_CROSSOVER_CACHE_MAX = 16
_GROWTH_COMPRESS_CACHE_MAX = 16
_graft_cycle_evals: dict[tuple[int, float, int], int] = {}
_graft_cycle_results: dict[tuple[int, float, int], tuple[tuple[int, ...], ...]] = {}
_graft_projection_cache: dict[tuple, tuple[tuple[int, ...], ...]] = {}
_eq_refine_bootstrapped: set[tuple[float, int]] = set()
_l10_lift_eq_bootstrap_cache: dict[tuple, tuple[tuple[int, ...], ...]] = {}
_projection_seeds_cache: dict[tuple, tuple[tuple[int, ...], ...]] = {}
_eq_refine_late_merged_cache: dict[tuple, tuple[tuple[int, ...], ...]] = {}
_compact_prom_raw_cycle_cache: dict[tuple[int, ...], float] = {}
_curriculum_l10_cycle_cache: dict[tuple[int, ...], float] = {}
_COMPRESS_MG_CACHE_MAX = 256


def clear_scan_floor_inject_caches() -> None:
    """Reset per-cycle fresh-compress and model-guided compress memoization."""
    global _fresh_compress_last_cycle
    _compress_mg_cache.clear()
    # _fresh_compress_*: plateau-keyed, retained across cycles on same prom plateau
    # _growth_crossover_pool_cache / _growth_compress_pool_cache: plateau-keyed, retained
    _graft_cycle_evals.clear()
    _graft_cycle_results.clear()
    _graft_projection_cache.clear()
    # _l10_lift_eq_bootstrap_cache intentionally retained: plateau-stable, ~9min cold build
    # _projection_seeds_cache intentionally retained: material_sig-stable, expensive rebuild
    _compact_prom_raw_cycle_cache.clear()
    _curriculum_l10_cycle_cache.clear()
    # _eq_refine_late_merged_cache: plateau-keyed, retained across cycles
    try:
        from evolution.terminal_stagnation import clear_locus_probe_cache

        clear_locus_probe_cache()
    except ImportError:
        pass


def _compact_prom_raw_cached(program: list[int]) -> float:
    """Memoize promotion_raw within a cycle (growth-inject sort calls this heavily)."""
    key = tuple(int(x) for x in program)
    cached = _compact_prom_raw_cycle_cache.get(key)
    if cached is not None:
        return float(cached)
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    val = float(compact_l30_promotion_raw_fitness(list(program)))
    _compact_prom_raw_cycle_cache[key] = val
    return val


def _curriculum_l10_cached(program: list[int], level: int = 10) -> float:
    """Memoize curriculum_verify_mean within a cycle (inject/graft ranking hot path)."""
    key = (int(level),) + tuple(int(x) for x in program)
    cached = _curriculum_l10_cycle_cache.get(key)
    if cached is not None:
        return float(cached)
    from evolution.terminal_stagnation import curriculum_verify_mean

    val = float(curriculum_verify_mean(list(program), int(level)))
    _curriculum_l10_cycle_cache[key] = val
    if len(_curriculum_l10_cycle_cache) > 8192:
        _curriculum_l10_cycle_cache.pop(next(iter(_curriculum_l10_cycle_cache)))
    return val


def score_stall_collect_gate_programs(
    champion: list[int],
    collected: list[list[int]],
    *,
    cycle_no: int = 0,
    score_cap: int = 2,
) -> list[list[int]]:
    """Rank gate-qualified stall-collect MG outputs; early-exit prom scoring on breakthrough."""
    if not champion or not collected:
        return []
    cap = max(2, int(score_cap))
    pool = [list(p) for p in collected]
    if len(pool) > cap:
        offset = int(cycle_no) % len(pool)
        pool = [pool[(offset + i) % len(pool)] for i in range(cap)]
    from evolution.terminal_stagnation import scan_floor_min_raw_gain

    champ_pr = _compact_prom_raw_cached(list(champion))
    prom_bar = champ_pr + float(scan_floor_min_raw_gain(len(champion)))
    scored: list[list[int]] = []
    for prog in pool:
        pr = _compact_prom_raw_cached(prog)
        if pr + 1e-6 >= champ_pr:
            scored.append(list(prog))
        if pr > prom_bar:
            break
    scored.sort(key=_compact_prom_raw_cached, reverse=True)
    return scored


def _compress_mg_plateau_key(champion: list[int]) -> tuple[float, int]:
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    pr = float(compact_l30_promotion_raw_fitness(list(champion)))
    return (round(pr, 4), len(champion))


def _fresh_compress_core_cache_key(
    _cycle_key: int,
    champion: list[int],
) -> tuple[float, int]:
    """Plateau-stable key so fresh-compress core survives across cycles."""
    return _compress_mg_plateau_key(list(champion))


def _graft_cycle_key(cycle_no: int, champion: list[int]) -> tuple[int, float, int]:
    return _fresh_compress_core_cache_key(int(cycle_no), champion)


def _merge_fresh_compress_long_sources(
    cycle_key: int,
    champion: list[int],
    champ_n: int,
    long_sources: list[list[int]] | None,
) -> set[tuple[int, ...]]:
    union_key = _fresh_compress_core_cache_key(int(cycle_key), list(champion))
    union = _fresh_compress_source_union.setdefault(union_key, set())
    if long_sources:
        for prog in long_sources:
            if prog and len(prog) > champ_n:
                union.add(tuple(int(w) for w in prog))
    return union


def _collect_fresh_compression_core(
    champion: list[int],
    sorted_long: list[list[int]],
    *,
    cycle_no: int = 0,
) -> list[list[int]]:
    """Model-guided compress + diff-site grafts (expensive; cached per cycle union)."""
    if not sorted_long:
        return []
    champ_n = len(champion)
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = {tuple(champion)}
    try:
        from evolution.terminal_stagnation import (
            _scan_floor_probe_stall_mode,
            get_scan_floor_gate,
            scan_floor_compression_promote_qualifies,
        )

        if not _scan_floor_probe_stall_mode(champ_n):
            return []
        import os

        gate = get_scan_floor_gate(list(champion))
        drop_trials = max(
            24,
            int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_DROP_TRIALS", "48")),
        )
        for src in sorted_long[:3]:
            comp = compress_model_guided_growth_candidate(
                list(champion),
                list(src),
                gate=gate,
                max_drop_trials=drop_trials,
            )
            if not comp or len(comp) > champ_n:
                continue
            if not scan_floor_compression_promote_qualifies(list(champion), list(comp)):
                continue
            key = tuple(comp)
            if key in seen:
                continue
            seen.add(key)
            out.append(list(comp))
        for graft in growth_material_diff_graft_variants(
            list(champion),
            sorted_long,
            max_variants=8,
            cycle_no=int(cycle_no),
        ):
            key = tuple(graft)
            if key in seen:
                continue
            if not scan_floor_compression_promote_qualifies(list(champion), list(graft)):
                continue
            seen.add(key)
            out.insert(0, list(graft))
    except ImportError:
        return []
    return out


def skip_model_guided_at_terminal(level: int, verify: float) -> bool:
    """True when model-guided GPU completion would steal VM eval bandwidth."""
    return int(level) >= 29 and float(verify) >= 0.99 - 1e-6


def scan_floor_injection_generation(
    generation: int,
    *,
    inject_generation: int | None = None,
    max_generations: int = 30,
    once_per_cycle: bool = True,
) -> bool:
    """Whether scan-floor model-guided slots fire on this generation index."""
    import os

    if not once_per_cycle:
        return True
    if inject_generation is not None:
        return int(generation) == int(inject_generation)
    inject_on_last = os.environ.get("EE_SCAN_FLOOR_GROWTH_LAST_GEN", "1") != "0"
    target = max(0, int(max_generations) - 1) if inject_on_last else 0
    return int(generation) == target


def scan_floor_growth_injection_generation(
    generation: int,
    *,
    inject_generation: int | None = None,
    max_generations: int = 30,
    once_per_cycle: bool = True,
) -> bool:
    """Whether scan-floor growth inject fires (default penultimate gen for breeding)."""
    import os

    if not once_per_cycle:
        return True
    if inject_generation is not None:
        return int(generation) == int(inject_generation)
    offset = max(1, int(os.environ.get("EE_SCAN_FLOOR_GROWTH_INJECT_OFFSET", "2")))
    target = max(0, int(max_generations) - offset)
    return int(generation) == target


def scan_floor_compression_injection_generation(
    generation: int,
    *,
    inject_generation: int | None = None,
    max_generations: int = 30,
    once_per_cycle: bool = True,
) -> bool:
    """Whether scan-floor compression inject fires (default gen 0 for immediate eval)."""
    if not once_per_cycle:
        return True
    if inject_generation is not None:
        return int(generation) == int(inject_generation)
    return int(generation) == 0


def scan_floor_stalled_growth_injection_generation(
    generation: int,
    *,
    inject_generation: int | None = None,
    max_generations: int = 30,
    once_per_cycle: bool = True,
) -> bool:
    """Growth inject at gen 0 and late gen when locally stalled (54w breeding window)."""
    if not once_per_cycle:
        return True
    if inject_generation is not None:
        return int(generation) == int(inject_generation)
    if int(generation) == 0:
        return True
    return scan_floor_growth_injection_generation(generation, max_generations=max_generations, once_per_cycle=True)


def scan_floor_stalled_compression_injection_generation(
    generation: int,
    *,
    inject_generation: int | None = None,
    max_generations: int = 30,
    once_per_cycle: bool = True,
) -> bool:
    """Compression inject at gen 0 and late gen when locally stalled (C7118 path)."""
    if not once_per_cycle:
        return True
    if inject_generation is not None:
        return int(generation) == int(inject_generation)
    if scan_floor_compression_injection_generation(generation, max_generations=max_generations, once_per_cycle=True):
        return True
    import os

    offset = max(1, int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_STALL_OFFSET", "2")))
    target = max(0, int(max_generations) - offset)
    return int(generation) == target


def _expand_growth_material_for_grafts(
    champion: list[int],
    material: list[list[int]],
    *,
    gate,
    cycle_no: int | None = None,
) -> list[list[int]]:
    """Add equal-length projections of longer MG material for graft diff discovery."""
    import os

    if not champion:
        return []
    champ_n = len(champion)
    champ_t = tuple(int(x) for x in champion)
    cap = int(os.environ.get("EE_GROWTH_MATERIAL_GRAFT_PROJECTION_CAP", "12"))
    drop_trials = max(
        12,
        int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_DROP_TRIALS", "48")),
    )
    cycle_rot = int(cycle_no or 0) % 3
    material_sig = tuple(tuple(int(w) for w in prog) for prog in material[:16] if prog)
    cache_key = (champ_t, material_sig, cycle_rot, cap)
    cached = _graft_projection_cache.get(cache_key)
    if cached is not None:
        expanded = [list(prog) for prog in cached]
    else:
        expanded = []
    from evolution.terminal_stagnation import curriculum_verify_mean, l30_l10_sum_max_preloop_insert_variant

    def _seed_l10_preloop(target: list[list[int]]) -> None:
        if _curriculum_l10_cached(list(champion)) >= 0.85 - 1e-6:
            return
        preloop = l30_l10_sum_max_preloop_insert_variant(list(champion))
        if not preloop:
            return
        seeds: list[list[int]] = [list(preloop)]
        try:
            compressed = compress_model_guided_growth_candidate(
                list(champion),
                list(preloop),
                gate=gate,
                max_drop_trials=max(
                    16,
                    int(os.environ.get("EE_SCAN_FLOOR_L10_LIFT_COMPRESS_TRIALS", "48")),
                ),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
            compressed = None
        if (
            compressed
            and len(compressed) > len(champion)
            and _curriculum_l10_cached(list(compressed)) > _curriculum_l10_cached(list(champion)) + 0.01
            and scan_floor_growth_completion_qualifies(list(champion), list(compressed), gate=gate)
        ):
            seeds.insert(0, list(compressed))
        seen_local = {tuple(int(x) for x in prog) for prog in target}
        for seed in seeds:
            seed_t = tuple(int(x) for x in seed)
            if seed_t in seen_local:
                continue
            seen_local.add(seed_t)
            target.insert(0, list(seed))

    seen: set[tuple[int, ...]] = {tuple(int(x) for x in prog) for prog in expanded}
    if cached is not None:
        _seed_l10_preloop(expanded)
        return expanded

    projected = 0
    for src in material:
        if not src or len(src) < champ_n:
            continue
        src_list = [int(x) for x in src]
        src_t = tuple(src_list)
        if src_t not in seen:
            seen.add(src_t)
            expanded.append(src_list)
        if _curriculum_l10_cached(list(champion)) < 0.85 - 1e-6 and len(src_list) > champ_n:
            for mid in enumerate_l10_lift_shrink_intermediates(list(champion), src_list, gate=gate):
                mid_t = tuple(int(x) for x in mid)
                if mid_t in seen:
                    continue
                seen.add(mid_t)
                expanded.append(list(mid))
        if len(src_list) <= champ_n or projected >= cap:
            continue
        projected += 1
        eq = equal_length_from_longer_growth(
            list(champion),
            src_list,
            gate=gate,
            max_drop_trials=drop_trials,
            cycle_no=cycle_no,
        )
        if len(eq) != champ_n:
            continue
        eq_t = tuple(int(x) for x in eq)
        if eq_t == champ_t or eq_t in seen:
            continue
        if not gate.qualifies(eq):
            continue
        seen.add(eq_t)
        expanded.append(list(eq))
    _seed_l10_preloop(expanded)
    stored = tuple(tuple(int(w) for w in prog) for prog in expanded)
    _graft_projection_cache[cache_key] = stored
    if len(_graft_projection_cache) > 32:
        _graft_projection_cache.pop(next(iter(_graft_projection_cache)))
    return expanded


def growth_material_diff_graft_variants(
    champion: list[int],
    material: list[list[int]],
    *,
    max_variants: int = 12,
    max_diff_sites: int = 4,
    max_pair_combos: int | None = None,
    cycle_no: int | None = None,
) -> list[list[int]]:
    """Equal-length grafts at growth-material diff sites (index-24-style breakthrough loci)."""
    import itertools
    import os

    if not champion or not material:
        return []
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.terminal_stagnation import get_scan_floor_gate

    raw_material_key = tuple(tuple(int(w) for w in prog) for prog in material[:12] if prog)
    pair_cap = int(
        max_pair_combos if max_pair_combos is not None else os.environ.get("EE_GROWTH_MATERIAL_GRAFT_PAIR_BUDGET", "8")
    )
    cross_parent_cap = int(os.environ.get("EE_GROWTH_MATERIAL_GRAFT_CROSS_PARENT_BUDGET", "8"))
    triple_cap = int(os.environ.get("EE_GROWTH_MATERIAL_GRAFT_TRIPLE_BUDGET", "6"))
    projection_cap = int(os.environ.get("EE_GROWTH_MATERIAL_GRAFT_PROJECTION_CAP", "12"))
    eval_budget = max(
        1,
        int(os.environ.get("EE_GROWTH_MATERIAL_GRAFT_EVAL_BUDGET", "20")),
    )
    champ_pr = float(compact_l30_promotion_raw_fitness(list(champion)))
    cycle_key = _graft_cycle_key(int(cycle_no), champion) if cycle_no is not None else None
    if cycle_key is not None:
        used = int(_graft_cycle_evals.get(cycle_key, 0))
        eval_budget = max(0, int(eval_budget) - used)
        if eval_budget <= 0:
            prior = _graft_cycle_results.get(cycle_key)
            if prior:
                return [list(prog) for prog in prior[: max(1, int(max_variants))]]
            return []
    graft_key = (
        round(champ_pr, 4),
        len(champion),
        raw_material_key,
        int(max_variants),
        int(max_diff_sites),
        pair_cap,
        cross_parent_cap,
        triple_cap,
        projection_cap,
        int(cycle_no or -1) % 3,
        eval_budget,
    )
    cached = getattr(growth_material_diff_graft_variants, "_result_cache", {}).get(graft_key)
    if cached is not None:
        result = [list(prog) for prog in cached]
        if cycle_key is not None and result:
            _graft_cycle_evals[cycle_key] = int(_graft_cycle_evals.get(cycle_key, 0)) + min(
                int(eval_budget), len(result)
            )
            prior = _graft_cycle_results.get(cycle_key, ())
            merged_cached: list[tuple[int, ...]] = []
            seen_cached: set[tuple[int, ...]] = set()
            for prog in list(prior) + [tuple(int(w) for w in p) for p in result]:
                if prog in seen_cached:
                    continue
                seen_cached.add(prog)
                merged_cached.append(prog)
            _graft_cycle_results[cycle_key] = tuple(merged_cached)
        return result

    gate = get_scan_floor_gate(list(champion))
    material = _expand_growth_material_for_grafts(
        list(champion),
        material,
        gate=gate,
        cycle_no=cycle_no,
    )
    if not material:
        return []

    champ = list(champion)
    champ_n = len(champ)
    champ_t = tuple(int(x) for x in champ)
    ranked: list[tuple[float, list[int]]] = []
    seen: set[tuple[int, ...]] = {champ_t}
    material_sites: list[tuple[int, list[int], list[int]]] = []
    for src in material:
        if not src or len(src) < champ_n:
            continue
        diffs = [i for i in range(champ_n) if int(src[i]) != int(champ[i])]
        if not diffs or len(diffs) > 8:
            continue
        material_sites.append((len(diffs), list(src), diffs))
    material_sites.sort(key=lambda item: item[0])
    pair_trials = 0
    prom_evals = 0
    for _diff_n, src, diffs in material_sites[:8]:
        for idx in diffs[: max(1, int(max_diff_sites))]:
            if prom_evals >= eval_budget:
                break
            graft = list(champ)
            graft[idx] = int(src[idx])
            key = tuple(graft)
            if key in seen:
                continue
            if not gate.qualifies(graft):
                continue
            seen.add(key)
            prom_evals += 1
            try:
                prom = float(compact_l30_promotion_raw_fitness(graft))
            except (RuntimeError, TypeError, ValueError, OSError):
                continue
            ranked.append((prom, graft))
        if prom_evals >= eval_budget:
            break
        if 2 <= len(diffs) <= 6 and pair_cap > 0 and pair_trials < pair_cap:
            for a, b in itertools.combinations(diffs[:6], 2):
                if pair_trials >= pair_cap or prom_evals >= eval_budget:
                    break
                pair_trials += 1
                graft = list(champ)
                graft[a] = int(src[a])
                graft[b] = int(src[b])
                key = tuple(graft)
                if key in seen:
                    continue
                if not gate.qualifies(graft):
                    continue
                seen.add(key)
                prom_evals += 1
                try:
                    prom = float(compact_l30_promotion_raw_fitness(graft))
                except (RuntimeError, TypeError, ValueError, OSError):
                    continue
                ranked.append((prom, graft))
    index_words: dict[int, set[int]] = {}
    for _diff_n, src, diffs in material_sites:
        for idx in diffs:
            index_words.setdefault(int(idx), set()).add(int(src[int(idx)]))
    ranked_indices = sorted(
        index_words.keys(),
        key=lambda i: (-len(index_words[i]), int(i)),
    )[:6]
    if cross_parent_cap > 0 and len(ranked_indices) >= 2:
        cross_trials = 0
        for i_idx, j_idx in itertools.combinations(ranked_indices, 2):
            if cross_trials >= cross_parent_cap or prom_evals >= eval_budget:
                break
            words_i = sorted(index_words[i_idx])
            words_j = sorted(index_words[j_idx])
            for wi in words_i[:3]:
                for wj in words_j[:3]:
                    if cross_trials >= cross_parent_cap or prom_evals >= eval_budget:
                        break
                    cross_trials += 1
                    graft = list(champ)
                    graft[i_idx] = int(wi)
                    graft[j_idx] = int(wj)
                    key = tuple(graft)
                    if key in seen:
                        continue
                    if not gate.qualifies(graft):
                        continue
                    seen.add(key)
                    prom_evals += 1
                    try:
                        prom = float(compact_l30_promotion_raw_fitness(graft))
                    except (RuntimeError, TypeError, ValueError, OSError):
                        continue
                    ranked.append((prom, graft))
    if triple_cap > 0 and len(ranked_indices) >= 3:
        triple_trials = 0
        for i_idx, j_idx, k_idx in itertools.combinations(ranked_indices[:5], 3):
            if triple_trials >= triple_cap or prom_evals >= eval_budget:
                break
            words_i = sorted(index_words[i_idx])[:2]
            words_j = sorted(index_words[j_idx])[:2]
            words_k = sorted(index_words[k_idx])[:2]
            for wi in words_i:
                for wj in words_j:
                    for wk in words_k:
                        if triple_trials >= triple_cap or prom_evals >= eval_budget:
                            break
                        triple_trials += 1
                        graft = list(champ)
                        graft[i_idx] = int(wi)
                        graft[j_idx] = int(wj)
                        graft[k_idx] = int(wk)
                        key = tuple(graft)
                        if key in seen:
                            continue
                        if not gate.qualifies(graft):
                            continue
                        seen.add(key)
                        prom_evals += 1
                        try:
                            prom = float(compact_l30_promotion_raw_fitness(graft))
                        except (RuntimeError, TypeError, ValueError, OSError):
                            continue
                        ranked.append((prom, graft))
    from evolution.terminal_stagnation import curriculum_verify_mean

    champ_l10 = _curriculum_l10_cached(champ)
    if champ_l10 < 0.85 - 1e-6:
        ranked.sort(
            key=lambda item: (
                _curriculum_l10_cached(item[1]),
                item[0],
            ),
            reverse=True,
        )
    else:
        ranked.sort(key=lambda item: item[0], reverse=True)
    result = [prog for _, prog in ranked[: max(1, int(max_variants))]]
    cache = getattr(growth_material_diff_graft_variants, "_result_cache", None)
    if cache is None:
        cache = {}
        growth_material_diff_graft_variants._result_cache = cache
    cache[graft_key] = tuple(tuple(int(w) for w in prog) for prog in result)
    if len(cache) > 64:
        cache.pop(next(iter(cache)))
    if cycle_key is not None:
        _graft_cycle_evals[cycle_key] = int(_graft_cycle_evals.get(cycle_key, 0)) + int(prom_evals)
        prior = _graft_cycle_results.get(cycle_key, ())
        merged: list[tuple[int, ...]] = []
        seen_cycle: set[tuple[int, ...]] = set()
        for prog in list(prior) + [tuple(int(w) for w in p) for p in result]:
            if prog in seen_cycle:
                continue
            seen_cycle.add(prog)
            merged.append(prog)
        _graft_cycle_results[cycle_key] = tuple(merged)
    return result


def collect_scan_floor_fresh_compression_variants(
    champion: list[int],
    *,
    long_sources: list[list[int]] | None = None,
    cycle_no: int = 0,
    gen_idx: int = 0,
    stall_seed_offset: int = 0,
    max_evals_per_source: int | None = None,
) -> list[list[int]]:
    """Exploratory terminal-efficiency drops for compression inject when stalled."""
    import os
    import random

    from evolution.champion_bloat_guard import level_word_cap
    from evolution.terminal_stagnation import (
        scan_floor_compression_promote_qualifies,
        terminal_efficiency_drop_span,
        terminal_efficiency_variants,
    )

    if not champion:
        return []
    global _fresh_compress_core_cache, _fresh_compress_last_cycle, _fresh_compress_plateau_key
    champ_key = tuple(int(x) for x in champion)
    cycle_key = int(cycle_no)
    plateau_key = _compress_mg_plateau_key(list(champion))
    if cycle_key != _fresh_compress_last_cycle:
        if _fresh_compress_plateau_key != plateau_key:
            _fresh_compress_source_union.clear()
            _fresh_compress_core_cache.clear()
            _graft_cycle_evals.clear()
            _graft_cycle_results.clear()
        _fresh_compress_plateau_key = plateau_key
        _fresh_compress_last_cycle = cycle_key
    champ_n = len(champion)
    word_cap = level_word_cap(30)
    span = terminal_efficiency_drop_span(champ_n, word_cap)
    budget = max(
        4,
        int(max_evals_per_source or os.environ.get("EE_SCAN_FLOOR_COMPRESS_FRESH_BUDGET", "12")),
    )
    union = _merge_fresh_compress_long_sources(cycle_key, list(champion), champ_n, long_sources)
    long_pool = [list(prog) for prog in union]
    try:
        from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

        stalled = _scan_floor_probe_stall_mode(champ_n)
    except ImportError:
        stalled = False
    if stalled and long_pool:
        sorted_long = scan_floor_growth_compress_sources(
            list(champion),
            long_pool,
            stalled=True,
            max_trials=6,
        )
    else:
        sorted_long = sorted(long_pool, key=len)
    sources: list[list[int]] = [list(champion)]
    sources.extend(sorted_long[:4])
    core_key = _fresh_compress_core_cache_key(cycle_key, list(champion))
    core_cached = _fresh_compress_core_cache.get(core_key)
    if core_cached is not None:
        cached_union_size, cached_core = core_cached
        if len(union) <= int(cached_union_size):
            out = [list(prog) for prog in cached_core]
        else:
            out = _collect_fresh_compression_core(
                list(champion),
                sorted_long,
                cycle_no=cycle_key,
            )
            _fresh_compress_core_cache[core_key] = (
                len(union),
                tuple(tuple(int(w) for w in prog) for prog in out),
            )
    else:
        out = _collect_fresh_compression_core(
            list(champion),
            sorted_long,
            cycle_no=cycle_key,
        )
        _fresh_compress_core_cache[core_key] = (
            len(union),
            tuple(tuple(int(w) for w in prog) for prog in out),
        )
    seen: set[tuple[int, ...]] = {tuple(prog) for prog in out}
    seen.add(tuple(champion))
    for idx, source in enumerate(sources):
        seed = (cycle_no * 997 + gen_idx * 31 + stall_seed_offset * 13 + idx * 1009) & 0x7FFFFFFF
        rng = random.Random(seed)
        per_source = budget if idx == 0 else max(4, budget // 2)
        for variant in terminal_efficiency_variants(list(source), rng, max_evals=per_source, max_drop_span=span):
            if not variant or len(variant) >= len(source):
                continue
            if len(variant) > champ_n:
                continue
            if not scan_floor_compression_promote_qualifies(champion, variant):
                continue
            key = tuple(variant)
            if key in seen:
                continue
            seen.add(key)
            out.append(list(variant))
    return out


def scan_floor_model_guided_injection_generation(
    generation: int,
    max_generations: int,
    *,
    inject_generation: int | None = None,
) -> bool:
    """Generation index for the once-per-cycle scan-floor model-guided slot."""
    import os

    once_per_cycle = os.environ.get("EE_SCAN_FLOOR_MODEL_GUIDED_ONCE_PER_CYCLE", "1") != "0"
    if not once_per_cycle:
        cadence = max(1, int(os.environ.get("EE_SCAN_FLOOR_MODEL_GUIDED_CADENCE", "10")))
        return int(generation) % cadence == 0
    return scan_floor_injection_generation(
        int(generation),
        inject_generation=inject_generation,
        max_generations=int(max_generations),
        once_per_cycle=True,
    )


def scan_floor_stalled_model_guided_injection_generation(
    generation: int,
    max_generations: int,
    *,
    inject_generation: int | None = None,
    once_per_cycle: bool = True,
) -> bool:
    """MG at gen 0 (fresh challengers) and last gen when capability is stalled."""
    if not once_per_cycle:
        return True
    if inject_generation is not None:
        return int(generation) == int(inject_generation)
    if int(generation) == 0:
        return True
    return scan_floor_injection_generation(
        int(generation),
        inject_generation=inject_generation,
        max_generations=int(max_generations),
        once_per_cycle=True,
    )


def normalize_model_guided_tail_growth(
    program: list[int],
    champion: list[int],
    *,
    prefix_count: int,
    max_words: int,
) -> list[int] | None:
    """Lock champion prefix; allow a longer tail ending in EBREAK (scan-floor growth)."""
    if not program or not champion:
        return None
    ebreak = 0x00100073
    out = [int(w) for w in program if w is not None]
    if not out:
        return None
    if out[-1] != ebreak:
        out.append(ebreak)
    pc = min(int(prefix_count), len(champion))
    if len(out) <= pc:
        return None
    tail = list(out[pc:])
    if not tail:
        return None
    if tail[-1] != ebreak:
        tail.append(ebreak)
    merged = list(champion[:pc]) + tail
    cap = max(int(max_words), len(champion) + 1)
    if len(merged) > cap:
        merged = merged[: cap - 1] + [ebreak]
    if len(merged) <= len(champion):
        return None
    if merged[-1] != ebreak:
        merged[-1] = ebreak
    return merged


def growth_mg_equal_length_projection(
    champion: list[int],
    grown: list[int],
    *,
    prefix_count: int,
    gate,
) -> list[int] | None:
    """Project a longer MG growth completion to champion length at the active prefix lock."""
    if not champion or not grown or len(grown) <= len(champion) or gate is None:
        return None
    pc = min(int(prefix_count), len(champion))
    if pc >= len(champion):
        return None
    proj = list(champion)
    for i in range(pc, len(champion)):
        if i < len(grown):
            proj[i] = int(grown[i])
    if tuple(proj) == tuple(champion):
        return None
    if gate.qualifies(proj):
        return proj
    return None


def collect_scan_floor_l10_lift_growth_candidates(
    champion: list[int],
    *,
    gate=None,
) -> list[list[int]]:
    """Held-out-safe longer programs that lift L10 for MG compress (gate may fail at length)."""
    import os

    if not champion:
        return []
    from evolution.terminal_stagnation import (
        curriculum_verify_mean,
        get_scan_floor_gate,
        l30_l8_mem_add_variant,
        l30_l10_bounded_sum_variant,
        l30_l10_max_in_sum_loop_variant,
        l30_order_sensitive_variant,
    )

    gate = gate or get_scan_floor_gate(list(champion))
    champ = list(champion)
    champ_l10 = _curriculum_l10_cached(champ)
    if champ_l10 >= 0.99 - 1e-6:
        return []
    from evolution.terminal_stagnation import l30_l10_sum_max_preloop_insert_variant

    builders = (
        l30_l10_sum_max_preloop_insert_variant,
        l30_order_sensitive_variant,
        l30_l8_mem_add_variant,
        l30_l10_bounded_sum_variant,
        l30_l10_max_in_sum_loop_variant,
    )
    cap = max(
        1,
        int(os.environ.get("EE_SCAN_FLOOR_L10_LIFT_GROWTH_CANDIDATES", "4")),
    )
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = {tuple(int(x) for x in champ)}
    bases: list[list[int]] = [champ]
    for builder in builders:
        for base in list(bases):
            try:
                variant = builder(list(base))
            except (AttributeError, TypeError, ValueError, KeyError):
                continue
            if not variant or len(variant) <= len(champ):
                continue
            key = tuple(int(x) for x in variant)
            if key in seen:
                continue
            if _curriculum_l10_cached(list(variant)) <= champ_l10 + 1e-6:
                continue
            if not scan_floor_growth_heldout_qualifies(champ, list(variant)):
                continue
            if not scan_floor_growth_completion_qualifies(champ, list(variant), gate=gate):
                continue
            seen.add(key)
            out.append(list(variant))
            bases.append(list(variant))
            try:
                compressed = compress_model_guided_growth_candidate(
                    champ,
                    list(variant),
                    gate=gate,
                    max_drop_trials=max(
                        16,
                        int(os.environ.get("EE_SCAN_FLOOR_L10_LIFT_COMPRESS_TRIALS", "48")),
                    ),
                )
            except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
                compressed = None
            if compressed and len(compressed) > len(champ):
                comp_key = tuple(int(x) for x in compressed)
                if (
                    comp_key not in seen
                    and _curriculum_l10_cached(list(compressed)) > champ_l10 + 1e-6
                    and scan_floor_growth_heldout_qualifies(champ, list(compressed))
                    and scan_floor_growth_completion_qualifies(champ, list(compressed), gate=gate)
                ):
                    seen.add(comp_key)
                    out.append(list(compressed))
                    bases.append(list(compressed))
            for seed_prog in (list(variant), list(compressed) if compressed else None):
                if not seed_prog or len(seed_prog) <= len(champ):
                    continue
                for mid in enumerate_l10_lift_shrink_intermediates(champ, seed_prog, gate=gate):
                    mid_key = tuple(int(x) for x in mid)
                    if mid_key in seen:
                        continue
                    seen.add(mid_key)
                    out.append(list(mid))
                    if len(out) >= cap * 2:
                        break
            if len(out) >= cap:
                break
    out.sort(
        key=lambda prog: (
            _curriculum_l10_cached(list(prog)),
            -len(prog),
        ),
        reverse=True,
    )
    return out[:cap]


def scan_floor_growth_heldout_qualifies(
    champion: list[int],
    variant: list[int],
    *,
    margin: float = 0.01,
    n_seeds: int = 8,
) -> bool:
    """Held-out-safe growth filter (no same-length side gate)."""
    if not champion or not variant or len(variant) <= len(champion):
        return False
    from evolution.terminal_stagnation import heldout_l30_verify_mean

    champ_h = float(heldout_l30_verify_mean(list(champion), int(n_seeds)))
    return float(heldout_l30_verify_mean(list(variant), int(n_seeds))) + margin >= champ_h


def scan_floor_growth_completion_qualifies(
    champion: list[int],
    variant: list[int],
    *,
    gate=None,
    margin: float = 0.01,
    n_seeds: int = 8,
) -> bool:
    """Qualify MG growth outputs at any length relative to the champion."""
    if not champion or not variant:
        return False
    champ_n = len(champion)
    var_n = len(variant)
    from evolution.terminal_stagnation import get_scan_floor_gate, scan_floor_compression_probe_qualifies

    gate = gate or get_scan_floor_gate(champion)
    if var_n > champ_n:
        from evolution.terminal_stagnation import compact_curriculum_side_score, heldout_l30_verify_mean

        champ_h = float(heldout_l30_verify_mean(list(champion), int(n_seeds)))
        cand_h = float(heldout_l30_verify_mean(list(variant), int(n_seeds)))
        if cand_h + 1e-6 < champ_h:
            return False
        champ_side = float(compact_curriculum_side_score(list(champion)))
        return compact_curriculum_side_score(list(variant)) + 1e-6 >= champ_side
    if var_n == champ_n:
        return bool(gate.qualifies(variant))
    return scan_floor_compression_probe_qualifies(champion, variant)


def _l10_lift_compress_protected_positions(variant: list[int]) -> set[int]:
    """Indices of an embedded bounded sum+max loop that L10-lift compress must keep."""
    from evolution.terminal_stagnation import _l30_bounded_sum_max_loop_body

    body = _l30_bounded_sum_max_loop_body()
    n = len(body)
    if not variant or len(variant) < n:
        return set()
    protected: set[int] = set()
    for i in range(len(variant) - n + 1):
        if list(variant[i : i + n]) == body:
            protected.update(range(i, i + n))
    return protected


def enumerate_l10_lift_shrink_intermediates(
    champion: list[int],
    variant: list[int],
    *,
    gate=None,
    max_steps: int = 4,
) -> list[list[int]]:
    """Held-out-safe single-word drops toward champion length while preserving L10 lift."""
    if not champion or not variant or len(variant) <= len(champion):
        return []
    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = gate or get_scan_floor_gate(list(champion))
    champ = list(champion)
    champ_l10 = _curriculum_l10_cached(champ)
    source_l10 = _curriculum_l10_cached(list(variant))
    if source_l10 <= champ_l10 + 0.01:
        return []
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    cur = list(variant)
    steps = max(1, int(max_steps))
    for _ in range(steps):
        if len(cur) <= len(champ):
            break
        best_drop: list[int] | None = None
        best_l10 = -1.0
        best_len = len(cur) + 1
        for pos in range(1, len(cur) - 1):
            trial = cur[:pos] + cur[pos + 1 :]
            if not scan_floor_growth_compress_drop_qualifies(champ, trial):
                continue
            if not scan_floor_growth_completion_qualifies(champ, trial, gate=gate):
                continue
            trial_l10 = _curriculum_l10_cached(trial)
            if trial_l10 + 1e-6 < champ_l10 + 0.01:
                continue
            key = tuple(int(x) for x in trial)
            if key in seen:
                continue
            if trial_l10 > best_l10 + 1e-6 or (abs(trial_l10 - best_l10) < 1e-6 and len(trial) < best_len):
                best_drop = list(trial)
                best_l10 = trial_l10
                best_len = len(trial)
        if best_drop is None:
            break
        seen.add(tuple(int(x) for x in best_drop))
        out.append(list(best_drop))
        cur = best_drop
    out.sort(
        key=lambda prog: (
            _curriculum_l10_cached(list(prog)),
            -len(prog),
        ),
        reverse=True,
    )
    return out


def _iter_l10_lift_pair_drop_trials(prog: list[int], champ_n: int, gap: int):
    """Yield equal-length drop trials without materializing the full combinatorial set."""
    if gap == 1:
        for pos in range(1, len(prog) - 1):
            yield prog[:pos] + prog[pos + 1 :]
        return
    for drop_count in range(2, min(gap + 1, 4)):
        if drop_count == 2:
            for i in range(1, len(prog) - 2):
                for j in range(i + 1, len(prog) - 1):
                    yield prog[:i] + prog[i + 1 : j] + prog[j + 1 :]
        elif drop_count == 3 and gap >= 3:
            for i in range(1, len(prog) - 3):
                for j in range(i + 1, len(prog) - 2):
                    for k in range(j + 1, len(prog) - 1):
                        yield (prog[:i] + prog[i + 1 : j] + prog[j + 1 : k] + prog[k + 1 :])


def collect_l10_lift_pair_drop_eq_programs(
    champion: list[int],
    variant: list[int],
    *,
    gate=None,
    max_candidates: int = 8,
) -> list[list[int]]:
    """Equal-length programs from near-target L10-lift parents via pair/single drops."""
    import os

    if not champion or not variant:
        return []
    champ = list(champion)
    champ_n = len(champ)
    prog = list(variant)
    if len(prog) <= champ_n:
        return []
    gap = len(prog) - champ_n
    if gap < 1 or gap > 3:
        return []
    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = gate or get_scan_floor_gate(champ)
    champ_l10 = _curriculum_l10_cached(champ)
    if _curriculum_l10_cached(prog) <= champ_l10 + 0.01:
        return []
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    trial_cap = max(
        16,
        int(os.environ.get("EE_L10_PAIR_DROP_TRIAL_CAP", "512")),
    )
    candidates: list[tuple[float, float, list[int]]] = []
    seen_trial: set[tuple[int, ...]] = set()
    trials_seen = 0
    for trial in _iter_l10_lift_pair_drop_trials(prog, champ_n, gap):
        trials_seen += 1
        if trials_seen > trial_cap:
            break
        if len(trial) != champ_n:
            continue
        key = tuple(int(x) for x in trial)
        if key in seen_trial:
            continue
        seen_trial.add(key)
        if not scan_floor_growth_compress_drop_qualifies(champ, trial):
            continue
        if not scan_floor_growth_completion_qualifies(champ, trial, gate=gate):
            continue
        trial_l10 = _curriculum_l10_cached(trial)
        if trial_l10 <= champ_l10 + 0.01:
            continue
        if not gate.qualifies(trial):
            continue
        prom = float(compact_l30_promotion_raw_fitness(trial))
        candidates.append((trial_l10, prom, list(trial)))
        if len(candidates) >= max(1, int(max_candidates)) * 4:
            break
    if not candidates:
        return []
    if champ_l10 < 0.85 - 1e-6:
        candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
    else:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    out: list[list[int]] = []
    seen_out: set[tuple[int, ...]] = set()
    for _l10, _prom, trial in candidates:
        key = tuple(int(x) for x in trial)
        if key in seen_out:
            continue
        seen_out.add(key)
        out.append(list(trial))
        if len(out) >= max(1, int(max_candidates)):
            break
    return out


def maybe_l10_lift_post_mutation_compress(
    champion: list[int],
    parent: list[int],
    mutated: list[int],
    *,
    gate=None,
) -> list[int]:
    """Shrink long offspring from an L10-lift parent when compress preserves L10."""
    if not champion or not parent or not mutated:
        return list(mutated) if mutated else mutated
    if len(parent) <= len(champion) or len(mutated) <= len(champion):
        return list(mutated)
    champ_l10 = _curriculum_l10_cached(list(champion))
    parent_l10 = _curriculum_l10_cached(list(parent))
    if parent_l10 <= champ_l10 + 0.01:
        return list(mutated)
    if len(mutated) <= len(parent):
        return list(mutated)
    import os

    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = gate or get_scan_floor_gate(list(champion))
    compressed = compress_model_guided_growth_candidate(
        list(champion),
        list(mutated),
        gate=gate,
        max_drop_trials=max(
            12,
            int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_DROP_TRIALS", "48")),
        ),
    )
    if (
        compressed
        and len(compressed) < len(mutated)
        and _curriculum_l10_cached(list(compressed)) + 1e-6 >= champ_l10 + 0.01
        and scan_floor_growth_completion_qualifies(list(champion), list(compressed), gate=gate)
    ):
        return list(compressed)
    return list(mutated)


def scan_floor_growth_compress_drop_qualifies(
    champion: list[int],
    variant: list[int],
    *,
    margin: float = 0.01,
    n_seeds: int = 8,
) -> bool:
    """Held-out-safe step when shrinking a growth completion toward champion length."""
    if not champion or not variant:
        return False
    champ_n = len(champion)
    var_n = len(variant)
    if var_n < champ_n:
        from evolution.terminal_stagnation import scan_floor_compression_probe_qualifies

        return scan_floor_compression_probe_qualifies(champion, variant)
    if var_n == champ_n:
        from evolution.terminal_stagnation import get_scan_floor_gate

        return get_scan_floor_gate(list(champion)).qualifies(list(variant))
    from evolution.terminal_stagnation import heldout_l30_verify_mean

    champ_h = float(heldout_l30_verify_mean(list(champion), int(n_seeds)))
    return float(heldout_l30_verify_mean(list(variant), int(n_seeds))) + margin >= champ_h


def lock_model_guided_champion_prefix(
    program: list[int],
    champion: list[int],
    prefix_count: int,
) -> list[int] | None:
    """Keep the champion prefix fixed; only the tail may differ."""
    if not program or not champion:
        return None
    champ_n = len(champion)
    pc = max(0, min(int(prefix_count), champ_n - 1))
    ebreak = 0x00100073
    if len(program) != champ_n:
        return None
    out = list(champion[:pc]) + list(program[pc:champ_n])
    out[-1] = ebreak
    return out


def normalize_model_guided_program(
    program: list[int],
    champion: list[int],
) -> list[int] | None:
    """Force same length as champion with a trailing EBREAK."""
    if not program or not champion:
        return None
    champ_n = len(champion)
    ebreak = 0x00100073
    out = [int(w) for w in program if w is not None]
    if not out:
        return None
    if out[-1] != ebreak:
        out.append(ebreak)
    if len(out) > champ_n:
        out = out[: champ_n - 1] + [ebreak]
    elif len(out) < champ_n:
        prefix_len = len(out) - 1
        out = out[:prefix_len] + [int(w) for w in champion[prefix_len:champ_n]]
    if len(out) != champ_n:
        return None
    if out[-1] != ebreak:
        out[-1] = ebreak
    return out


def repair_model_guided_for_gate(
    program: list[int],
    gate,
    *,
    max_evals: int = 12,
    champion: list[int] | None = None,
    prefix_count: int | None = None,
    eval_cap: int | None = None,
) -> list[int] | None:
    """Nudge a near-miss model completion through the scan-floor gate."""
    import os
    import random

    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.terminal_stagnation import (
        scan_floor_compact_tail_min_position,
        terminal_dual_imm_probe_variants,
        terminal_imm_perturb_probe_variants,
        terminal_tail_wide_imm_probe_variants,
        terminal_wide_imm_sweep_probe_variants,
    )

    if not program or gate is None:
        return None
    champ_t = tuple(champion) if champion else None
    champ_pr = float(compact_l30_promotion_raw_fitness(list(champion))) if champion else -1.0
    min_gain = 0.0
    if champion:
        from evolution.terminal_stagnation import scan_floor_min_raw_gain

        min_gain = float(scan_floor_min_raw_gain(len(champion)))

    def _accept_candidate(best_local: list[int] | None, best_pr_local: float) -> list[int] | None:
        if best_local is None:
            return None
        if best_pr_local > champ_pr + min_gain:
            return best_local
        if best_pr_local + 1e-6 >= champ_pr:
            return best_local
        return None

    if gate.qualifies(program) and (champ_t is None or tuple(program) != champ_t):
        direct_pr = float(compact_l30_promotion_raw_fitness(list(program)))
        accepted = _accept_candidate(list(program), direct_pr)
        if accepted is not None:
            return accepted

    stalled = False
    try:
        from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

        stalled = bool(_scan_floor_probe_stall_mode(len(champion) if champion else None))
    except ImportError:
        pass
    budget = max(4, int(max_evals))
    if eval_cap is not None:
        budget = min(budget, int(eval_cap))
    elif stalled:
        budget = max(
            budget,
            int(os.environ.get("EE_SCAN_FLOOR_GATE_REPAIR_EVALS", "48")),
        )

    seed = list(program)
    if champion and prefix_count is not None:
        locked = lock_model_guided_champion_prefix(seed, champion, int(prefix_count))
        if locked is not None:
            seed = locked

    def _pick_best(candidates: list[list[int]]) -> list[int] | None:
        best_local: list[int] | None = None
        best_pr_local = -1.0
        seen_local: set[tuple[int, ...]] = set()
        for cand in candidates:
            key = tuple(cand)
            if key in seen_local:
                continue
            seen_local.add(key)
            if champ_t is not None and key == champ_t:
                continue
            if not gate.qualifies(cand):
                continue
            pr = float(compact_l30_promotion_raw_fitness(cand))
            if pr > best_pr_local:
                best_pr_local = pr
                best_local = list(cand)
        return best_local

    diff_candidates: list[list[int]] = [seed]
    if champion and prefix_count is not None and champ_t is not None:
        pc = int(prefix_count)
        diff_positions = [i for i in range(pc, len(champion)) if i < len(seed) and int(seed[i]) != int(champion[i])]
        from milestone1.rv32i_mask import decode as rv32i_decode
        from milestone1.rv32i_mask import encode as rv32i_encode

        for pos in diff_positions[:8]:
            trial = list(champion)
            trial[pos] = int(seed[pos])
            diff_candidates.append(trial)
            try:
                decoded = rv32i_decode(int(seed[pos]), validate=False)
                if "imm" not in decoded:
                    continue
                for delta in (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16):
                    trial2 = list(champion)
                    trial2[pos] = int(
                        rv32i_encode(
                            name=str(decoded["name"]),
                            rd=int(decoded.get("rd", 0)),
                            rs1=int(decoded.get("rs1", 0)),
                            rs2=int(decoded.get("rs2", 0)),
                            imm=int(decoded.get("imm", 0)) + int(delta),
                            validate=True,
                        )
                    )
                    diff_candidates.append(trial2)
            except (ValueError, TypeError, KeyError):
                continue

    best = _pick_best(diff_candidates)
    if best is not None:
        best_pr = float(compact_l30_promotion_raw_fitness(best))
        accepted = _accept_candidate(best, best_pr)
        if accepted is not None:
            return accepted
        deep_enabled = os.environ.get("EE_SCAN_FLOOR_GATE_REPAIR_DEEP", "0") == "1" or stalled
        if not deep_enabled:
            return None

    start = hash(tuple(seed)) & 0x7FFFFFFF
    deep_candidates: list[list[int]] = list(diff_candidates)
    deep_candidates.extend(
        terminal_wide_imm_sweep_probe_variants(seed, max_evals=budget, start_offset=start % 97, gate=gate)
    )
    rng = random.Random(start)
    deep_candidates.extend(terminal_imm_perturb_probe_variants(seed, rng, max_evals=max(4, budget // 2), gate=gate))
    if stalled and champion and len(seed) == len(champion):
        min_pos = max(
            int(prefix_count) if prefix_count is not None else 0,
            scan_floor_compact_tail_min_position(len(champion)),
        )
        deep_candidates.extend(
            terminal_tail_wide_imm_probe_variants(
                seed,
                min_position=min_pos,
                max_evals=budget,
                start_offset=start % 131,
                gate=gate,
            )
        )
        deep_candidates.extend(
            terminal_dual_imm_probe_variants(
                seed,
                min_position=min_pos,
                max_evals=max(8, budget // 2),
                start_offset=start % 89,
                gate=gate,
            )
        )
        try:
            eq_seeds = load_scan_floor_eq_refine_seeds(
                list(champion),
                max_seeds=4,
                probe_material=False,
            )
            champ_list = list(champion)
            pc = int(prefix_count) if prefix_count is not None else min_pos
            for eq_seed in eq_seeds:
                if len(eq_seed) != len(champion):
                    continue
                deep_candidates.append(list(eq_seed))
                for pos in range(pc, len(champion)):
                    if int(eq_seed[pos]) == int(champ_list[pos]):
                        continue
                    blend = list(champ_list)
                    blend[pos] = int(eq_seed[pos])
                    deep_candidates.append(blend)
                for graft in expand_l10_lift_eq_champion_grafts(
                    champ_list,
                    list(eq_seed),
                    gate=gate,
                    max_variants=4,
                ):
                    deep_candidates.append(list(graft))
        except (ImportError, TypeError, ValueError):
            pass
    deep_best = _pick_best(deep_candidates)
    if deep_best is None:
        return None
    deep_pr = float(compact_l30_promotion_raw_fitness(deep_best))
    return _accept_candidate(deep_best, deep_pr)


def compress_model_guided_growth_candidate(
    champion: list[int],
    variant: list[int],
    *,
    gate=None,
    max_drop_trials: int = 16,
) -> list[int]:
    """Peephole-compress growth completions and try held-out-safe interior drops."""
    if not champion or not variant:
        return list(variant)
    mg_key = (
        _compress_mg_plateau_key(list(champion)),
        tuple(int(x) for x in variant),
        int(max_drop_trials),
    )
    cached_mg = _compress_mg_cache.get(mg_key)
    if cached_mg is not None:
        return list(cached_mg)

    def _store_compress_mg(result: list[int]) -> list[int]:
        _compress_mg_cache[mg_key] = tuple(int(x) for x in result)
        if len(_compress_mg_cache) > _COMPRESS_MG_CACHE_MAX:
            _compress_mg_cache.pop(next(iter(_compress_mg_cache)))
        return list(result)

    from evolution.terminal_stagnation import (
        get_scan_floor_gate,
        scan_floor_compact_tail_min_position,
        scan_floor_compression_probe_qualifies,
        semantic_compress,
        terminal_efficiency_at_scan_floor,
    )

    gate = gate or get_scan_floor_gate(champion)
    champ_n = len(champion)
    from evolution.terminal_stagnation import curriculum_verify_mean

    champ_l10 = _curriculum_l10_cached(list(champion))
    source_l10 = _curriculum_l10_cached(list(variant))
    l10_lift_compress = source_l10 > champ_l10 + 0.01
    drop_start = 1
    if terminal_efficiency_at_scan_floor(champ_n):
        import os

        # Tail-lock only for near-length shrinks (50w@325.627 side-breakers). Long growth
        # parents (54w+, C7118 path) need interior drops — first diff vs 7117 was index 17.
        tail_lock_margin = max(
            2,
            int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_TAIL_LOCK_MARGIN", "2")),
        )
        if len(variant) <= champ_n + tail_lock_margin:
            drop_start = max(1, int(scan_floor_compact_tail_min_position(champ_n)))
    best = list(variant)
    compressed = semantic_compress(list(variant))
    if (
        compressed
        and len(compressed) <= len(best)
        and scan_floor_growth_completion_qualifies(champion, compressed, gate=gate)
    ):
        best = list(compressed)
    if l10_lift_compress:
        from evolution.terminal_stagnation import l10_lift_loop_preserving_semantic_compress

        loop_cur = list(best)
        for _ in range(6):
            loop_prefix = l10_lift_loop_preserving_semantic_compress(loop_cur)
            if (
                not loop_prefix
                or len(loop_prefix) >= len(loop_cur)
                or not scan_floor_growth_completion_qualifies(champion, loop_prefix, gate=gate)
            ):
                break
            loop_cur = list(loop_prefix)
            if len(loop_cur) < len(best):
                best = list(loop_cur)
    if len(best) == champ_n and gate.qualifies(best):
        return _store_compress_mg(best)
    if len(best) <= champ_n:
        return _store_compress_mg(best)
    trials_cap = max(4, int(max_drop_trials))
    candidates: list[list[int]] = []
    cur = list(best)
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    def _drop_rank(trial: list[int]) -> tuple[float, ...]:
        from evolution.terminal_stagnation import curriculum_verify_mean

        trial_n = len(trial)
        lcp = sum(1 for a, b in zip(champion, trial) if int(a) == int(b))
        l10 = _curriculum_l10_cached(trial)
        prom = float(compact_l30_promotion_raw_fitness(trial))
        side_tier = 0
        if trial_n < champ_n:
            if scan_floor_compression_probe_qualifies(champion, trial):
                side_tier = 1
        elif trial_n == champ_n and gate.qualifies(trial):
            side_tier = 2
        elif trial_n > champ_n and scan_floor_growth_completion_qualifies(champion, trial, gate=gate):
            side_tier = 1
        if l10_lift_compress:
            return (l10, prom, -trial_n, side_tier, lcp)
        return (side_tier, -trial_n, prom, l10, lcp)

    # Greedy single-word drops: reach champ_n when a monotonic shrink path exists.
    for _ in range(trials_cap):
        if len(cur) <= champ_n:
            break
        best_drop: list[int] | None = None
        best_rank: tuple[int, int, float, float, int] | None = None
        drop_positions = list(range(drop_start, len(cur) - 1))
        loop_protected: set[int] = set()
        if l10_lift_compress:
            loop_protected = _l10_lift_compress_protected_positions(cur)
            drop_positions.sort(
                key=lambda p: (
                    1 if p in loop_protected else 0,
                    0 if p < len(champion) and int(cur[p]) == int(champion[p]) else 1,
                    p,
                )
            )
        for pos in drop_positions:
            trial = cur[:pos] + cur[pos + 1 :]
            if not scan_floor_growth_compress_drop_qualifies(champion, trial):
                continue
            if l10_lift_compress:
                trial_l10 = _curriculum_l10_cached(trial)
                if trial_l10 + 1e-6 < source_l10 - 0.01:
                    continue
            rank = _drop_rank(trial)
            if best_rank is None or rank > best_rank:
                best_drop = trial
                best_rank = rank
        if best_drop is None:
            break
        candidates.append(list(best_drop))
        cur = best_drop
    if len(cur) > champ_n and len(cur) - champ_n <= 4:
        pair_cap = max(32, trials_cap * 8)
        tried = 0
        for i in range(drop_start, len(cur) - 2):
            for j in range(i + 1, len(cur) - 1):
                if tried >= pair_cap:
                    break
                tried += 1
                trial = cur[:i] + cur[i + 1 : j] + cur[j + 1 :]
                if not scan_floor_growth_compress_drop_qualifies(champion, trial):
                    continue
                if l10_lift_compress:
                    trial_l10 = _curriculum_l10_cached(trial)
                    if trial_l10 + 1e-6 < source_l10 - 0.01:
                        continue
                candidates.append(list(trial))
                if len(trial) <= champ_n:
                    cur = trial
    if not candidates:
        return _store_compress_mg(best)
    equal_len = [trial for trial in candidates if len(trial) == champ_n and gate.qualifies(trial)]
    if equal_len:
        equal_len.sort(
            key=lambda prog: scan_floor_eq_inject_sort_key(champion, list(prog)),
            reverse=True,
        )
        return _store_compress_mg(list(equal_len[0]))
    if l10_lift_compress:
        from evolution.terminal_stagnation import curriculum_verify_mean

        pool: list[list[int]] = []
        seen_pool: set[tuple[int, ...]] = set()
        for prog in (*candidates, best, list(variant), cur):
            key = tuple(int(x) for x in prog)
            if key in seen_pool:
                continue
            seen_pool.add(key)
            pool.append(list(prog))
        qualifying = [
            list(prog) for prog in pool if scan_floor_growth_completion_qualifies(champion, list(prog), gate=gate)
        ]
        if qualifying:
            qualifying.sort(
                key=lambda prog: (
                    _curriculum_l10_cached(list(prog)),
                    -len(prog),
                    float(compact_l30_promotion_raw_fitness(list(prog))),
                ),
                reverse=True,
            )
            return _store_compress_mg(list(qualifying[0]))
        if scan_floor_growth_completion_qualifies(champion, best, gate=gate):
            return _store_compress_mg(list(best))
        return _store_compress_mg(list(variant))
    return _store_compress_mg(min(candidates, key=len))


def _repair_compressed_growth_probe(
    champion: list[int],
    compressed: list[int],
    *,
    gate,
    prefix_count: int | None = None,
    eval_cap: int | None = None,
) -> list[int] | None:
    """Gate-repair a 48w compress near-miss that differs structurally from champion."""
    if not champion or not compressed or gate is None:
        return None
    champ_n = len(champion)
    if len(compressed) != champ_n or tuple(compressed) == tuple(champion):
        return None
    if gate.qualifies(compressed):
        return list(compressed)
    from evolution.terminal_stagnation import scan_floor_compact_tail_min_position

    prefix = int(prefix_count) if prefix_count is not None else int(scan_floor_compact_tail_min_position(champ_n))
    repaired = repair_model_guided_for_gate(
        list(compressed),
        gate,
        max_evals=24,
        champion=list(champion),
        prefix_count=prefix,
        eval_cap=eval_cap,
    )
    if repaired is None or not gate.qualifies(repaired):
        return None
    if len(repaired) > len(champion) and not scan_floor_growth_heldout_qualifies(champion, repaired):
        return None
    return list(repaired)


def equal_length_from_longer_growth(
    champion: list[int],
    grown: list[int],
    *,
    gate,
    max_drop_trials: int,
    cycle_no: int | None = None,
    repair_eval_cap: int | None = None,
) -> list[int]:
    """Project or compress a longer growth candidate to champion length."""
    import os

    champ_n = len(champion)
    prog_list = list(grown)
    if len(prog_list) <= champ_n:
        return prog_list
    from evolution.terminal_stagnation import curriculum_verify_mean

    champ_l10 = _curriculum_l10_cached(list(champion))
    source_l10 = _curriculum_l10_cached(prog_list)
    l10_lift = source_l10 > champ_l10 + 0.01
    comp: list[int] | None = None
    lcp = growth_anchor_lcp_prefix(champion, prog_list)
    if l10_lift:
        comp = list(
            compress_model_guided_growth_candidate(
                champion,
                prog_list,
                gate=gate,
                max_drop_trials=max_drop_trials,
            )
        )
        comp_l10 = _curriculum_l10_cached(comp)
        if comp_l10 > champ_l10 + 0.01 and scan_floor_growth_completion_qualifies(champion, comp, gate=gate):
            return list(comp)
        comp = None
    if cycle_no is not None:
        deltas = (-1, 0, 1)
        pc = max(1, min(champ_n - 1, lcp + deltas[int(cycle_no) % 3]))
    else:
        pc = lcp
    projected = growth_mg_equal_length_projection(
        champion,
        prog_list,
        prefix_count=pc,
        gate=gate,
    )
    if projected is not None:
        if not l10_lift or _curriculum_l10_cached(projected) > champ_l10 + 0.01:
            comp = list(projected)
    if comp is None:
        comp = compress_model_guided_growth_candidate(
            champion,
            prog_list,
            gate=gate,
            max_drop_trials=max_drop_trials,
        )
    if len(comp) == champ_n and tuple(comp) != tuple(champion) and not gate.qualifies(comp):
        cap = repair_eval_cap
        if cap is None:
            cap = max(
                8,
                int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_REPAIR_EVALS", "12")),
            )
        repaired = _repair_compressed_growth_probe(
            champion,
            comp,
            gate=gate,
            prefix_count=lcp,
            eval_cap=cap,
        )
        if repaired is not None:
            comp = repaired
    return list(comp)


def _growth_material_tail_score(
    champion: list[int],
    variant: list[int],
    *,
    start: int = 24,
) -> tuple[int, int]:
    """Tail divergence score for growth-material diversity (count, magnitude)."""
    if not champion or not variant:
        return (0, 0)
    tail_start = max(0, min(int(start), len(champion)))
    end = len(variant)
    count = 0
    magnitude = 0
    for i in range(tail_start, end):
        champ_word = int(champion[i]) if i < len(champion) else 0
        var_word = int(variant[i])
        if var_word != champ_word:
            count += 1
            magnitude += abs(var_word - champ_word)
    return (count, magnitude)


def select_diverse_growth_material_programs(
    champion: list[int],
    candidates: list[list[int]],
    *,
    max_select: int,
) -> list[list[int]]:
    """Greedy max-min tail diversity selection for MG material recording."""
    if not champion or not candidates or max_select <= 0:
        return []
    champ = list(champion)
    pool = [list(prog) for prog in candidates if prog and len(prog) > len(champ)]
    if not pool:
        return []
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for prog in pool:
        key = tuple(int(x) for x in prog)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(prog))
    selected: list[list[int]] = []
    while unique and len(selected) < int(max_select):
        best_idx = 0
        best_score = (-1, -1, -1)
        for idx, prog in enumerate(unique):
            champ_count, champ_mag = _growth_material_tail_score(champ, prog)
            if not selected:
                score = (champ_count, champ_mag, champ_count)
            else:
                min_sep = min(
                    sum(int(prog[i]) != int(other[i]) for i in range(len(champ), min(len(prog), len(other))))
                    for other in selected
                )
                score = (min_sep, champ_mag, champ_count)
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(unique.pop(best_idx))
    return selected


def flush_growth_mg_material_candidates(
    champion: list[int],
    candidates: list[list[int]],
    *,
    cycle: int,
    record_cap: int = 4,
) -> int:
    """Persist the most diverse held-out-safe MG growth tails for this cycle."""
    if not champion:
        return 0
    from evolution.champion_rollback import record_scan_floor_growth_material
    from evolution.terminal_stagnation import curriculum_verify_mean, get_scan_floor_gate

    champ = list(champion)
    augmented = [list(prog) for prog in (candidates or []) if prog]
    mandatory: list[list[int]] = []
    champ_l10 = _curriculum_l10_cached(champ)
    if champ_l10 < 0.85 - 1e-6:
        gate = get_scan_floor_gate(champ)
        l10_seeds = collect_scan_floor_l10_lift_growth_candidates(champ, gate=gate)
        if l10_seeds:
            l10_seeds.sort(
                key=lambda prog: (
                    _curriculum_l10_cached(list(prog)),
                    -len(prog),
                ),
                reverse=True,
            )
            mandatory = [list(l10_seeds[0])]
            augmented.extend(list(l10_seeds))
    if not augmented and not mandatory:
        return 0

    cap = max(1, int(record_cap))
    seen_mandatory = {tuple(int(x) for x in mandatory[0])} if mandatory else set()
    remaining = max(0, cap - len(mandatory))
    pool = [list(prog) for prog in augmented if tuple(int(x) for x in prog) not in seen_mandatory]
    picked = list(mandatory) + select_diverse_growth_material_programs(champ, pool, max_select=remaining)
    recorded = 0
    for prog in picked:
        if record_scan_floor_growth_material(champ, list(prog), cycle=int(cycle)):
            recorded += 1
    return recorded


def resolve_growth_mg_completion(
    champion: list[int],
    grown: list[int],
    *,
    gate,
    prefix_count: int,
    max_drop_trials: int = 12,
    allow_projection: bool = True,
) -> tuple[list[int] | None, bool]:
    """Normalize a growth-mode MG completion (compress, optional equal-length projection).

    Returns ``(candidate, projection_attempted)``.
    """
    champ = list(champion)
    grown_list = list(grown)
    if not champ or not grown_list or len(grown_list) <= len(champ):
        return None, False
    projection_attempted = False
    compressed = compress_model_guided_growth_candidate(
        champ,
        grown_list,
        gate=gate,
        max_drop_trials=max_drop_trials,
    )
    if tuple(compressed) != tuple(champ):
        candidate = list(compressed)
    else:
        candidate = list(grown_list)
        if allow_projection:
            projection_attempted = True
            champ_n = len(champ)
            base_pc = int(prefix_count)
            for delta in (0, -1, 1, -2, 2):
                try_pc = max(1, min(champ_n - 1, base_pc + int(delta)))
                projected = growth_mg_equal_length_projection(
                    champ,
                    grown_list,
                    prefix_count=try_pc,
                    gate=gate,
                )
                if projected is not None:
                    candidate = list(projected)
                    break
    if tuple(candidate) == tuple(champ):
        return None, projection_attempted
    if not scan_floor_growth_completion_qualifies(champ, candidate, gate=gate):
        return None, projection_attempted
    return candidate, projection_attempted


def enrich_growth_inject_equal_length_pool(
    champion: list[int],
    pool: list[list[int]],
    *,
    gate,
    cycle_no: int,
    max_projections: int | None = None,
) -> list[list[int]]:
    """Project top long growth inject candidates to champion length (stalled path)."""
    import os

    if not champion or not pool:
        return list(pool)
    champ_n = len(champion)
    champ_t = tuple(int(x) for x in champion)
    cap = int(
        max_projections
        if max_projections is not None
        else os.environ.get("EE_SCAN_FLOOR_GROWTH_INJECT_PROJECTION_CAP", "6")
    )
    drop_trials = max(
        12,
        int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_DROP_TRIALS", "48")),
    )
    seen = {tuple(int(w) for w in p) for p in pool}
    additions: list[list[int]] = []
    long_sorted = sorted(
        [list(p) for p in pool if len(p) > champ_n],
        key=lambda p: (len(p), -_compact_prom_raw_cached(list(p))),
    )
    from evolution.terminal_stagnation import curriculum_verify_mean

    champ_l10 = _curriculum_l10_cached(list(champion))
    champ_l10_weak = champ_l10 < 0.85 - 1e-6
    for src in long_sorted[: max(0, cap)]:
        src_l10 = _curriculum_l10_cached(list(src))
        if champ_l10_weak and src_l10 > champ_l10 + 0.01:
            for eq in collect_l10_lift_pair_drop_eq_programs(
                list(champion),
                list(src),
                gate=gate,
            ):
                eq_t = tuple(int(x) for x in eq)
                if eq_t == champ_t or eq_t in seen:
                    continue
                seen.add(eq_t)
                additions.append(list(eq))
            continue
        if src_l10 > champ_l10 + 0.01:
            continue
        eq = equal_length_from_longer_growth(
            list(champion),
            src,
            gate=gate,
            max_drop_trials=drop_trials,
            cycle_no=int(cycle_no),
        )
        if len(eq) != champ_n:
            continue
        if _curriculum_l10_cached(list(eq)) <= champ_l10 + 1e-6:
            continue
        eq_t = tuple(int(x) for x in eq)
        if eq_t == champ_t or eq_t in seen:
            continue
        if not gate.qualifies(eq):
            continue
        seen.add(eq_t)
        additions.append(list(eq))
    if not additions:
        return list(pool)
    return additions + list(pool)


def select_diverse_l10_lift_inject_programs(
    champion: list[int],
    programs: list[list[int]],
    n_take: int,
    *,
    cycle_no: int = 0,
) -> list[list[int]]:
    """Length-diverse L10-lift injects: peak L10 plus near-target-length breeders."""
    import os

    if not champion or not programs or n_take <= 0:
        return []
    champ_n = len(champion)
    champ_l10 = _curriculum_l10_cached(list(champion))
    target = champ_n + max(
        2,
        int(os.environ.get("EE_L10_LIFT_INJECT_TARGET_EXTRA", "2")),
    )
    min_l10 = champ_l10 + 0.01
    pool = [list(p) for p in programs if p and len(p) > champ_n and _curriculum_l10_cached(list(p)) > min_l10]
    if not pool:
        return []
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def _add(prog: list[int]) -> bool:
        key = tuple(int(x) for x in prog)
        if key in seen:
            return False
        seen.add(key)
        out.append(list(prog))
        return True

    by_l10 = sorted(
        pool,
        key=lambda p: _curriculum_l10_cached(list(p)),
        reverse=True,
    )
    _add(by_l10[0])
    near = sorted(
        pool,
        key=lambda p: (
            -_curriculum_l10_cached(list(p)),
            abs(len(p) - target),
            len(p),
        ),
    )
    for prog in near:
        if len(out) >= int(n_take):
            break
        _add(prog)
    high_short = sorted(
        [list(p) for p in pool if _curriculum_l10_cached(list(p)) >= 0.9 - 1e-6],
        key=len,
    )
    for prog in high_short:
        if len(out) >= int(n_take):
            break
        _add(prog)
    offset = int(cycle_no) % max(1, len(pool))
    idx = 0
    while len(out) < int(n_take) and idx < len(pool) * 2:
        _add(pool[(offset + idx) % len(pool)])
        idx += 1
    return out[: int(n_take)]


def select_growth_inject_slot_variants(
    champion: list[int],
    pool: list[list[int]],
    *,
    n_slots: int,
    cycle_no: int,
    stalled: bool = False,
    max_long_words: int | None = None,
) -> list[list[int]]:
    """Pick growth inject challengers; when stalled, favor equal-length eval throughput."""
    import os

    if not pool or n_slots <= 0:
        return []
    if not stalled:
        return [pool[(int(cycle_no) + i) % len(pool)] for i in range(int(n_slots))]
    from evolution.terminal_stagnation import curriculum_verify_mean

    champ_n = len(champion)
    champ_l10 = _curriculum_l10_cached(list(champion))
    champ_l10_weak = champ_l10 < 0.85 - 1e-6
    long_cap = max(
        1,
        int(os.environ.get("EE_SCAN_FLOOR_GROWTH_LONG_INJECT_CAP", "2")),
    )
    if champ_l10_weak:
        long_cap = max(
            long_cap,
            int(os.environ.get("EE_SCAN_FLOOR_L10_LIFT_LONG_INJECT_CAP", "3")),
        )
    eq_pool = [p for p in pool if len(p) == champ_n]
    long_pool = [p for p in pool if len(p) > champ_n]
    if max_long_words is not None:
        long_pool = [p for p in long_pool if len(p) <= int(max_long_words)]
    short_pool = [p for p in pool if len(p) < champ_n]
    l10_lift_long: list[list[int]] = []
    if champ_l10_weak:
        l10_lift_long = [list(p) for p in long_pool if _curriculum_l10_cached(list(p)) > champ_l10 + 0.01]
        l10_lift_long.sort(
            key=lambda prog: (
                _curriculum_l10_cached(list(prog)),
                -len(prog),
            ),
            reverse=True,
        )
    primary = eq_pool if eq_pool else short_pool
    out: list[list[int]] = []
    seen_out: set[tuple[int, ...]] = set()
    if champ_l10_weak and l10_lift_long:
        l10_take = min(int(n_slots), long_cap, len(l10_lift_long))
        for prog in select_diverse_l10_lift_inject_programs(
            list(champion),
            l10_lift_long,
            l10_take,
            cycle_no=int(cycle_no),
        ):
            key = tuple(int(x) for x in prog)
            if key in seen_out:
                continue
            seen_out.add(key)
            out.append(list(prog))
    remaining = int(n_slots) - len(out)
    if primary and remaining > 0:
        added = 0
        idx = 0
        while added < remaining and idx < len(primary) * 2:
            prog = primary[(int(cycle_no) + idx) % len(primary)]
            idx += 1
            key = tuple(int(x) for x in prog)
            if key in seen_out:
                continue
            seen_out.add(key)
            out.append(list(prog))
            added += 1
        remaining = int(n_slots) - len(out)
    long_other = [
        list(p) for p in long_pool if tuple(int(x) for x in p) not in seen_out and list(p) not in l10_lift_long
    ]
    if remaining > 0 and long_other:
        long_take = min(remaining, long_cap)
        for i in range(long_take):
            prog = long_other[(int(cycle_no) + i) % len(long_other)]
            key = tuple(int(x) for x in prog)
            if key in seen_out:
                continue
            seen_out.add(key)
            out.append(list(prog))
        remaining -= long_take
    if remaining > 0 and primary:
        for i in range(remaining):
            out.append(primary[(int(cycle_no) + len(out) + i) % len(primary)])
    elif remaining > 0:
        for i in range(remaining):
            out.append(pool[(int(cycle_no) + i) % len(pool)])
    return out[: int(n_slots)]


def scan_floor_growth_inject_rank(
    champion: list[int],
    prog: list[int],
    *,
    gate=None,
    champ_pr: float | None = None,
    min_gain: float | None = None,
    stalled: bool = False,
) -> tuple[int, float, int]:
    """Rank growth-inject candidates: prom-beating growth/compress first."""
    import os

    from evolution.terminal_stagnation import get_scan_floor_gate, scan_floor_min_raw_gain

    if not champion or not prog:
        return (0, -1.0, 0)
    champ = list(champion)
    champ_n = len(champ)
    gate = gate or get_scan_floor_gate(champ)
    pr = _compact_prom_raw_cached(list(prog))
    if champ_pr is None:
        champ_pr = _compact_prom_raw_cached(champ)
    if min_gain is None:
        min_gain = float(scan_floor_min_raw_gain(champ_n))
    prog_n = len(prog)
    if stalled and prog_n > champ_n and pr + 1e-6 < float(champ_pr):
        if scan_floor_growth_completion_qualifies(champ, prog, gate=gate):
            prog_l10 = _curriculum_l10_cached(list(prog))
            champ_l10 = _curriculum_l10_cached(champ)
            extra_words = int(os.environ.get("EE_GROWTH_COMPRESS_TARGET_EXTRA_WORDS", "4"))
            if champ_l10 < 0.85 - 1e-6:
                extra_words = int(os.environ.get("EE_GROWTH_COMPRESS_TARGET_EXTRA_WORDS_WEAK_L10", "2"))
            target_len = champ_n + max(2, extra_words)
            if prog_l10 > champ_l10 + 0.01:
                return (5, prog_l10, -abs(prog_n - target_len), pr)
            return (4, -abs(prog_n - target_len), pr)
    if prog_n == champ_n:
        tier = 3 if gate.qualifies(prog) else 2
        return (tier, pr, 0)
    if prog_n > champ_n and pr > float(champ_pr) + float(min_gain) - 1e-6:
        if scan_floor_growth_completion_qualifies(champ, prog, gate=gate):
            return (3, pr, -prog_n)
    return (1, pr, -prog_n)


def scan_floor_growth_crossover_max_words(champ_words: int) -> int:
    """Max word count for growth-crossover injects above the scan-floor champion."""
    import os

    margin = max(2, int(os.environ.get("EE_SCAN_FLOOR_GROWTH_CROSSOVER_MAX_WORDS", "6")))
    return int(champ_words) + margin


def _scan_floor_mg_overlength_budget(champ_words: int, *, inject: bool) -> int:
    """Champion-relative overlength ceiling (prevents 52w cap on 30w champs)."""
    import os

    if inject:
        margin = max(4, int(os.environ.get("EE_SCAN_FLOOR_GROWTH_MG_INJECT_OVERLENGTH", "14")))
    else:
        margin = max(4, int(os.environ.get("EE_SCAN_FLOOR_GROWTH_MG_STALLED_OVERLENGTH", "12")))
    return int(champ_words) + margin


def scan_floor_growth_mg_max_words(
    champ_words: int,
    *,
    generation: int,
    inject_generation: int,
    default_max: int = 72,
    stalled: bool = False,
) -> int:
    """Cap growth-mode MG tail length between inject generations when stalled."""
    import os

    extra = max(2, int(os.environ.get("EE_GROWTH_COMPRESS_TARGET_EXTRA_WORDS", "4")))
    if not stalled:
        return int(default_max)
    if int(generation) == int(inject_generation):
        inject_cap = max(
            int(champ_words) + extra + 2,
            int(os.environ.get("EE_SCAN_FLOOR_GROWTH_MG_MAX_WORDS_INJECT", "52")),
        )
        inject_cap = min(
            inject_cap,
            _scan_floor_mg_overlength_budget(int(champ_words), inject=True),
        )
        return min(int(default_max), inject_cap)
    cap = max(
        int(champ_words) + extra,
        int(os.environ.get("EE_SCAN_FLOOR_GROWTH_MG_MAX_WORDS_STALLED", "48")),
    )
    cap = min(cap, _scan_floor_mg_overlength_budget(int(champ_words), inject=False))
    return min(int(default_max), cap)


def scan_floor_stalled_challenger_max_words(
    champ_words: int,
    *,
    generation: int,
    mg_inject_generation: int,
    default_max: int = 72,
    stalled: bool = True,
) -> int:
    """Per-generation challenger word cap when scan-floor breeding is stalled."""
    return scan_floor_growth_mg_max_words(
        int(champ_words),
        generation=int(generation),
        inject_generation=int(mg_inject_generation),
        default_max=int(default_max),
        stalled=bool(stalled),
    )


def filter_scan_floor_growth_programs_by_max_words(
    programs: list[list[int]],
    *,
    max_words: int,
) -> list[list[int]]:
    """Drop growth-pool programs that exceed the stalled inject/MG word cap."""
    cap = int(max_words)
    if cap <= 0 or not programs:
        return list(programs)
    return [list(p) for p in programs if p and len(p) <= cap]


_SHRINK_CHALLENGER_MISS = object()
_shrink_challenger_cache: dict[tuple[tuple[int, ...], tuple[int, ...], int, int], list[int] | object] = {}


def clear_shrink_challenger_cache() -> None:
    """Drop per-generation shrink memoization (call at each evolution generation)."""
    _shrink_challenger_cache.clear()


def shrink_scan_floor_challenger_to_max_words(
    program: list[int],
    champion: list[int],
    *,
    max_words: int,
    gate,
    cycle_no: int = 0,
    prefix_count: int | None = None,
) -> list[int] | None:
    """Shrink an over-length challenger before GPU eval (projection, compress, tail cap)."""
    import os

    if not program or not champion or gate is None:
        return None
    prog = list(program)
    champ = list(champion)
    cap = int(max_words)
    if len(prog) <= cap:
        return prog
    cache_key = (
        tuple(int(x) for x in champ),
        tuple(int(x) for x in prog),
        cap,
        int(cycle_no),
    )
    if cache_key in _shrink_challenger_cache:
        hit = _shrink_challenger_cache[cache_key]
        if hit is _SHRINK_CHALLENGER_MISS:
            return None
        return list(hit)

    def _finish(result: list[int] | None) -> list[int] | None:
        if result is None:
            _shrink_challenger_cache[cache_key] = _SHRINK_CHALLENGER_MISS
        else:
            _shrink_challenger_cache[cache_key] = list(result)
        if len(_shrink_challenger_cache) > 256:
            _shrink_challenger_cache.pop(next(iter(_shrink_challenger_cache)))
        return result

    champ_n = len(champ)
    if len(prog) <= champ_n:
        return _finish(None)
    drop_trials = max(
        12,
        int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_DROP_TRIALS", "48")),
    )
    eq = equal_length_from_longer_growth(
        champ,
        prog,
        gate=gate,
        max_drop_trials=drop_trials,
        cycle_no=int(cycle_no),
    )
    if len(eq) == champ_n and gate.qualifies(eq):
        return _finish(list(eq))
    compressed = compress_model_guided_growth_candidate(
        champ,
        prog,
        gate=gate,
        max_drop_trials=drop_trials,
    )
    if compressed and len(compressed) <= cap:
        if len(compressed) == champ_n and gate.qualifies(compressed):
            return _finish(list(compressed))
        if len(compressed) > champ_n and scan_floor_growth_completion_qualifies(champ, list(compressed), gate=gate):
            return _finish(list(compressed))
    pc = int(prefix_count) if prefix_count is not None else growth_anchor_lcp_prefix(champ, prog)
    normalized = normalize_model_guided_tail_growth(
        prog,
        champ,
        prefix_count=pc,
        max_words=cap,
    )
    if (
        normalized
        and len(normalized) <= cap
        and len(normalized) > champ_n
        and scan_floor_growth_heldout_qualifies(champ, normalized)
    ):
        return _finish(list(normalized))
    return _finish(None)


def scan_floor_l10_lift_preserve_slot(
    champion: list[int],
    program: list[int],
    *,
    gate=None,
) -> bool:
    """True when a long challenger must survive diversity/swap overwrites."""
    if not champion or not program or len(program) <= len(champion):
        return False
    champ_l10 = _curriculum_l10_cached(list(champion))
    if champ_l10 >= 0.85 - 1e-6:
        return False
    prog_l10 = _curriculum_l10_cached(list(program))
    if prog_l10 <= champ_l10 + 0.01:
        return False
    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = gate or get_scan_floor_gate(list(champion))
    return bool(scan_floor_growth_completion_qualifies(list(champion), list(program), gate=gate))


def record_l10_lift_growth_discoveries(
    champion: list[int],
    challengers: list[list[int]],
    *,
    cycle: int,
    state: dict,
    record_cap: int | None = None,
) -> int:
    """Persist novel held-out-safe L10-lift programs seen during a cycle."""
    import os

    if not champion or not challengers:
        return 0
    champ_n = len(champion)
    champ_l10 = _curriculum_l10_cached(list(champion))
    if champ_l10 >= 0.85 - 1e-6:
        return 0
    cap = max(
        1,
        int(record_cap if record_cap is not None else os.environ.get("EE_L10_LIFT_DISCOVERY_RECORD_CAP", "4")),
    )
    check_cap = max(
        1,
        int(os.environ.get("EE_L10_LIFT_DISCOVERY_CHECK_CAP", "4")),
    )
    cycle_key = int(cycle)
    if int(state.get("cycle", -1)) != cycle_key:
        state["cycle"] = cycle_key
        state["recorded"] = 0
    recorded = int(state.get("recorded", 0))
    if recorded >= cap:
        return 0
    from evolution.champion_rollback import record_scan_floor_growth_material

    seen: set[tuple[int, ...]] = {tuple(int(x) for x in champion)}
    added = 0
    long_candidates = [
        list(prog) for prog in challengers if prog and len(prog) > champ_n and tuple(int(x) for x in prog) not in seen
    ]
    if not long_candidates:
        return 0
    long_candidates.sort(key=len)
    for prog in long_candidates[:check_cap]:
        key = tuple(int(x) for x in prog)
        if not scan_floor_l10_lift_preserve_slot(list(champion), list(prog)):
            continue
        if record_scan_floor_growth_material(list(champion), list(prog), cycle=cycle_key):
            added += 1
            recorded += 1
            seen.add(key)
            if recorded >= cap:
                break
    state["recorded"] = recorded
    return added


def scan_floor_l10_lift_preserve_indices(
    champion: list[int],
    challengers: list[list[int]],
    *,
    gate=None,
) -> set[int]:
    if not champion or not challengers:
        return set()
    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = gate or get_scan_floor_gate(list(champion))
    return {
        int(i)
        for i, prog in enumerate(challengers)
        if prog and scan_floor_l10_lift_preserve_slot(champion, list(prog), gate=gate)
    }


def swap_inject_overlength_for_eq_seeds(
    champion: list[int],
    challengers: list[list[int]],
    *,
    cycle_no: int = 0,
    generation: int = 0,
    growth_inject_generation: int | None = None,
    mg_inject_generation: int | None = None,
    strategies: list | None = None,
    category_saturated: bool = False,
) -> int:
    """On late inject gens, swap over-length slots to eq-length seeds before GPU eval."""
    import os

    if not champion or not challengers:
        return 0
    try:
        from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

        if not _scan_floor_probe_stall_mode(len(champion)):
            return 0
    except ImportError:
        return 0
    if not scan_floor_late_inject_generation(
        int(generation),
        growth_inject_generation=growth_inject_generation,
        mg_inject_generation=mg_inject_generation,
    ):
        return 0
    seeds = load_scan_floor_eq_refine_seeds(list(champion), max_seeds=8, probe_material=False)
    if not seeds:
        return 0
    champ_n = len(champion)
    champ_t = tuple(int(x) for x in champion)
    max_swap = max(
        0,
        int(os.environ.get("EE_SCAN_FLOOR_INJECT_EQ_LENGTH_SWAP", "4")),
    )
    long_slots = [i for i, prog in enumerate(challengers) if prog and len(prog) > champ_n]
    if (
        category_saturated
        and long_slots
        and os.environ.get("EE_SCAN_FLOOR_INJECT_EQ_LENGTH_SWAP_SATURATED", "1") != "0"
    ):
        max_swap = max(max_swap, len(long_slots))
    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = get_scan_floor_gate(list(champion))
    swapped = 0
    for slot in long_slots:
        if swapped >= max_swap:
            break
        prog = challengers[slot]
        if scan_floor_l10_lift_preserve_slot(list(champion), list(prog), gate=gate):
            continue
        seed = seeds[(int(cycle_no) + swapped) % len(seeds)]
        if tuple(int(x) for x in seed) == champ_t:
            continue
        challengers[slot] = list(seed)
        if strategies is not None and slot < len(strategies):
            strategies[slot] = "EQ_LENGTH_SWAP"
        swapped += 1
    return swapped


def dedupe_programs_for_eval(
    programs: list[list[int]],
) -> tuple[list[list[int]], list[int]]:
    """Collapse identical programs for one batch eval; map results back to slots."""
    unique: list[list[int]] = []
    index_map: list[int] = []
    key_to_idx: dict[tuple[int, ...], int] = {}
    for prog in programs:
        if not prog:
            key: tuple[int, ...] = ()
        else:
            key = tuple(int(x) for x in prog)
        idx = key_to_idx.get(key)
        if idx is None:
            idx = len(unique)
            key_to_idx[key] = idx
            unique.append(list(prog) if prog else prog)
        index_map.append(idx)
    return unique, index_map


def expand_eval_results(results: list, index_map: list[int]) -> list:
    """Restore per-slot eval rows after ``dedupe_programs_for_eval``."""
    if not index_map:
        return list(results)
    return [results[i] for i in index_map]


def cap_scan_floor_challenger_lengths(
    challengers: list[list[int]],
    champion: list[int],
    *,
    max_words: int,
    gate,
    cycle_no: int = 0,
) -> list[list[int]]:
    """Ensure no challenger exceeds the stalled scan-floor word cap before eval."""
    if not challengers or not champion or int(max_words) <= 0:
        return challengers
    cap = int(max_words)
    champ = list(champion)
    champ_t = tuple(int(x) for x in champ)
    out: list[list[int]] = []
    for prog in challengers:
        if not prog:
            out.append(prog)
            continue
        if len(prog) <= cap:
            out.append(list(prog))
            continue
        shrunk = shrink_scan_floor_challenger_to_max_words(
            list(prog),
            champ,
            max_words=cap,
            gate=gate,
            cycle_no=int(cycle_no),
        )
        if shrunk is not None and tuple(shrunk) != champ_t:
            out.append(list(shrunk))
        else:
            out.append(list(champ))
    return out


def scan_floor_eq_refine_seeds_path() -> str:
    import os

    return os.path.join(_evolution_project_root(), ".agents", "scan_floor_eq_refine_seeds.json")


def _eq_refine_plateau_key(champion: list[int]) -> tuple[float, int]:
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    pr = float(compact_l30_promotion_raw_fitness(list(champion)))
    return (round(pr, 4), len(champion))


def _read_eq_refine_seeds_payload() -> dict:
    import json
    import os

    path = scan_floor_eq_refine_seeds_path()
    if not os.path.isfile(path):
        return {}
    try:
        payload = json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _eq_refine_payload_matches_plateau(
    payload: dict,
    plateau_key: tuple[float, int],
    anchor: list[int],
) -> bool:
    stored_key = payload.get("plateau_key")
    if isinstance(stored_key, list) and len(stored_key) == 2:
        try:
            if round(float(stored_key[0]), 4) == plateau_key[0] and int(stored_key[1]) == plateau_key[1]:
                return True
        except (TypeError, ValueError):
            pass
    stored = payload.get("anchor")
    if isinstance(stored, list) and len(stored) == len(anchor):
        return tuple(int(x) for x in stored) == tuple(int(x) for x in anchor)
    return False


def scan_floor_eq_inject_sort_key(
    champion: list[int],
    variant: list[int],
) -> tuple[float, float, float, float]:
    """Rank EQ inject candidates; prom-first at scan floor, else L10-first when weak."""
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.terminal_stagnation import curriculum_verify_mean

    prom = float(compact_l30_promotion_raw_fitness(list(variant)))
    l10 = float(curriculum_verify_mean(list(variant), 10))
    l8 = float(curriculum_verify_mean(list(variant), 8))
    blend = (l10 + l8) * 0.5
    if _curriculum_l10_cached(list(champion)) < 0.85 - 1e-6:
        try:
            from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor

            if terminal_efficiency_at_scan_floor(len(champion)):
                return (prom, l10, l8, blend)
        except ImportError:
            pass
        return (l10, prom, l8, blend)
    return (prom, l10, l8, blend)


def expand_l10_lift_eq_champion_grafts(
    champion: list[int],
    program: list[int],
    *,
    gate=None,
    max_variants: int = 16,
    min_l10_margin: float = 0.02,
) -> list[list[int]]:
    """Graft champion words onto an L10-lift eq program at diff sites without losing L10."""
    if not champion or not program or len(program) != len(champion):
        return []
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = gate or get_scan_floor_gate(list(champion))
    anchor = list(champion)
    base = list(program)
    champ_l10 = _curriculum_l10_cached(anchor)
    diffs = [i for i in range(len(anchor)) if int(base[i]) != int(anchor[i])]
    if not diffs:
        return []
    cap = min(len(diffs), 8)
    diffs = diffs[:cap]
    candidates: list[tuple[float, float, list[int]]] = []
    for mask in range(1, 1 << len(diffs)):
        trial = list(base)
        for bit, idx in enumerate(diffs):
            if mask & (1 << bit):
                trial[idx] = int(anchor[idx])
        if not gate.qualifies(trial):
            continue
        trial_l10 = _curriculum_l10_cached(trial)
        if trial_l10 + 1e-6 < champ_l10 + float(min_l10_margin):
            continue
        prom = float(compact_l30_promotion_raw_fitness(trial))
        candidates.append((trial_l10, prom, list(trial)))
    if not candidates:
        return []
    if champ_l10 < 0.85 - 1e-6:
        candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
    else:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for _l10, _prom, trial in candidates:
        key = tuple(int(x) for x in trial)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(trial))
        if len(out) >= max(1, int(max_variants)):
            break
    return out


def _eq_donor_words_by_site(
    anchor: list[int],
    base: list[int],
    donors: list[list[int]],
    *,
    per_site_cap: int = 8,
) -> dict[int, list[int]]:
    """Donor words available at each diff site (champion word first)."""
    out: dict[int, list[int]] = {}
    champ_n = len(anchor)
    for i in range(champ_n):
        if int(base[i]) == int(anchor[i]):
            continue
        words: list[int] = [int(anchor[i])]
        for donor in donors:
            if not donor or i >= len(donor):
                continue
            word = int(donor[i])
            if word == int(base[i]) or word in words:
                continue
            words.append(word)
            if len(words) >= max(2, int(per_site_cap)):
                break
        out[i] = words
    return out


def expand_l10_lift_eq_donor_grafts(
    champion: list[int],
    program: list[int],
    *,
    donors: list[list[int]] | None = None,
    gate=None,
    max_variants: int = 8,
    min_l10_margin: float = 0.02,
    max_evals: int | None = None,
) -> list[list[int]]:
    """Two-stage eq-length grafts over champion + growth-material donor words.

    Stage 1 collects single-word *bridges* (the champion word at one diff
    site) without applying the L10 margin, so an edit that on its own drops
    below the margin can still be combined with a second edit.  Stage 2
    accepts a candidate only when it keeps the L10 margin **and** passes the
    eq-length gate, then ranks accepted candidates by prom.  Every returned
    program therefore satisfies exactly the acceptance that
    :func:`expand_l10_lift_eq_champion_grafts` demands - no gate is relaxed
    - but the search may cross a margin-dropping bridge to get there.

    Donors default to the persisted growth-material bucket plus the champion,
    which is where long near-champion extensions (single-word graft sources
    that are neither the champion nor an eq seed) live.
    """
    import os

    if not champion or not program or len(program) != len(champion):
        return []
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.terminal_stagnation import get_scan_floor_gate

    anchor = [int(x) for x in champion]
    base = [int(x) for x in program]
    gate = gate or get_scan_floor_gate(list(anchor))
    champ_n = len(anchor)
    champ_l10 = _curriculum_l10_cached(anchor)
    margin = float(min_l10_margin)
    diffs = [i for i in range(champ_n) if base[i] != anchor[i]]
    if not diffs:
        return []

    if donors is None:
        donors = [list(anchor)]
        from evolution.champion_rollback import load_growth_material_programs

        try:
            donors.extend(load_growth_material_programs(list(anchor)))
        except (OSError, TypeError, ValueError):
            pass
    donors = [list(d) for d in donors if d]

    budget = int(max_evals) if max_evals is not None else int(os.environ.get("EE_L10_DONOR_GRAFT_EVAL_CAP", "2500"))
    budget = max(16, budget)
    evals = 0
    accepted: dict[tuple[int, ...], tuple[float, float, list[int]]] = {}

    def _l10_of(prog: list[int]) -> float:
        nonlocal evals
        evals += 1
        return float(_curriculum_l10_cached(list(prog)))

    def _accept(trial: list[int]) -> None:
        """Keep a trial only at full acceptance: L10 margin AND eq gate."""
        if len(trial) != champ_n:
            return
        key = tuple(int(x) for x in trial)
        if key in accepted:
            return
        trial_l10 = _l10_of(list(key))
        if trial_l10 + 1e-6 < champ_l10 + margin:
            return
        if not gate.qualifies(list(key)):
            return
        prom = float(compact_l30_promotion_raw_fitness(list(key)))
        accepted[key] = (prom, trial_l10, list(key))

    site_cap = max(2, int(os.environ.get("EE_L10_DONOR_GRAFT_SITE_CAP", "12")))
    # safe sites first (their single champion graft already keeps the margin)
    safe_first: list[int] = []
    unsafe: list[int] = []
    for i in diffs:
        if evals >= budget:
            break
        trial = list(base)
        trial[i] = anchor[i]
        if _l10_of(trial) + 1e-6 >= champ_l10 + margin:
            safe_first.append(i)
        else:
            unsafe.append(i)
    sites = (safe_first + unsafe)[:site_cap]

    # Stage 1: bridges = base + one champion word at a diff site (no margin).
    bridges: list[list[int]] = [list(base)]
    for i in sites:
        trial = list(base)
        trial[i] = anchor[i]
        bridges.append(trial)

    # Stage 2a: accept the bridges themselves.
    for bridge in bridges:
        if evals >= budget:
            break
        _accept(bridge)

    # Stage 2b: champion/donor word applied on top of a bridge.
    words_by_site = _eq_donor_words_by_site(anchor, base, donors)
    for bridge in bridges:
        if evals >= budget:
            break
        bridge_diffs = [i for i in sites if bridge[i] != anchor[i]]
        for i in bridge_diffs:
            for word in words_by_site.get(i, [anchor[i]]):
                if evals >= budget:
                    break
                if word == bridge[i]:
                    continue
                trial = list(bridge)
                trial[i] = int(word)
                _accept(trial)

    # Stage 2c: subsets of margin-safe champion grafts (prom-lift ladder).
    subset_cap = max(0, int(os.environ.get("EE_L10_DONOR_GRAFT_SAFE_SITES", "10")))
    subset_sites = safe_first[: min(len(safe_first), subset_cap)]
    if len(subset_sites) >= 2:
        for mask in range(1, 1 << len(subset_sites)):
            if evals >= budget:
                break
            if mask & (mask - 1) == 0:
                continue  # singletons were already accepted in stage 2a/2b
            trial = list(base)
            for bit, idx in enumerate(subset_sites):
                if mask & (1 << bit):
                    trial[idx] = anchor[idx]
            _accept(trial)

    if not accepted:
        return []
    ranked = list(accepted.values())
    if champ_l10 < 0.85 - 1e-6:
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    else:
        ranked.sort(key=lambda item: (item[1], item[0]), reverse=True)
    out: list[list[int]] = []
    for _prom, _l10, trial in ranked:
        out.append(list(trial))
        if len(out) >= max(1, int(max_variants)):
            break
    return out


def crossbreed_l10_prom_eq_refine_variants(
    champion: list[int],
    seeds: list[list[int]],
    *,
    gate=None,
    max_variants: int = 8,
    min_l10_margin: float = 0.02,
) -> list[list[int]]:
    """Blend high-L10 and prom-friendly eq seeds at shared diff sites."""
    if not champion or not seeds:
        return []
    anchor = list(champion)
    champ_n = len(anchor)
    champ_l10 = _curriculum_l10_cached(anchor)
    if champ_l10 >= 0.85 - 1e-6:
        return []
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = gate or get_scan_floor_gate(anchor)
    champ_t = tuple(int(x) for x in anchor)
    high_l10: list[list[int]] = []
    prom_seeds: list[list[int]] = []
    for seed in seeds:
        if not seed or len(seed) != champ_n:
            continue
        key = tuple(int(x) for x in seed)
        if key == champ_t:
            continue
        l10 = _curriculum_l10_cached(list(seed))
        prom = float(compact_l30_promotion_raw_fitness(list(seed)))
        if l10 > champ_l10 + 0.05:
            high_l10.append(list(seed))
        elif l10 > champ_l10 + float(min_l10_margin) and prom > 100.0:
            prom_seeds.append(list(seed))
    if not high_l10 or not prom_seeds:
        return []
    high_l10.sort(
        key=lambda prog: _curriculum_l10_cached(list(prog)),
        reverse=True,
    )
    prom_seeds.sort(
        key=lambda prog: float(compact_l30_promotion_raw_fitness(list(prog))),
        reverse=True,
    )
    import os

    pos_cap = max(4, int(os.environ.get("EE_L10_PROM_CROSSBREED_POS_CAP", "6")))
    candidates: list[tuple[float, float, list[int]]] = []
    seen: set[tuple[int, ...]] = {champ_t}
    for hi in high_l10[:2]:
        for prom_seed in prom_seeds[:3]:
            diff_positions = [
                i for i in range(champ_n) if int(hi[i]) != int(anchor[i]) and int(prom_seed[i]) != int(anchor[i])
            ]
            if not diff_positions:
                continue
            if len(diff_positions) > pos_cap:
                diff_positions = diff_positions[-pos_cap:]
            for mask in range(1, 1 << len(diff_positions)):
                trial = list(prom_seed)
                for bit, pos in enumerate(diff_positions):
                    if mask & (1 << bit):
                        trial[pos] = int(hi[pos])
                key = tuple(int(x) for x in trial)
                if key in seen:
                    continue
                if not gate.qualifies(trial):
                    continue
                trial_l10 = _curriculum_l10_cached(trial)
                if trial_l10 + 1e-6 < champ_l10 + float(min_l10_margin):
                    continue
                seen.add(key)
                prom = float(compact_l30_promotion_raw_fitness(trial))
                candidates.append((prom, trial_l10, list(trial)))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    out: list[list[int]] = []
    for _prom, _l10, trial in candidates:
        out.append(list(trial))
        if len(out) >= max(1, int(max_variants)):
            break
    return out


def persist_l10_lift_bootstrap_intermediates(
    champion: list[int],
    programs: list[list[int]],
    *,
    cycle: int = 0,
    max_record: int = 6,
) -> int:
    """Persist held-out-safe L10-lift shrink paths (e.g. 29w@0.94) for compress breeding."""
    import os

    if not champion or not programs:
        return 0
    champ_n = len(champion)
    champ_l10 = _curriculum_l10_cached(list(champion))
    if champ_l10 >= 0.85 - 1e-6:
        return 0
    from evolution.champion_rollback import record_scan_floor_growth_material

    recorded = 0
    seen: set[tuple[int, ...]] = set()
    target_len = champ_n + max(2, int(os.environ.get("EE_GROWTH_COMPRESS_TARGET_EXTRA_WORDS_WEAK_L10", "2")))
    ranked = sorted(
        [list(p) for p in programs if p],
        key=lambda prog: (
            1 if len(prog) == target_len else 0,
            _curriculum_l10_cached(list(prog)),
            -len(prog),
        ),
        reverse=True,
    )
    cap = int(max_record)
    if cap <= 0:
        return 0
    cap = max(1, cap)
    for prog in ranked:
        if len(prog) <= champ_n:
            continue
        if _curriculum_l10_cached(list(prog)) <= champ_l10 + 0.05:
            continue
        key = tuple(int(x) for x in prog)
        if key in seen:
            continue
        seen.add(key)
        if record_scan_floor_growth_material(list(champion), list(prog), cycle=int(cycle)):
            recorded += 1
            if recorded >= cap:
                break
    return recorded


def collect_l10_lift_eq_refine_bootstrap_seeds(
    champion: list[int],
    *,
    gate=None,
    max_seeds: int = 8,
) -> list[list[int]]:
    """Equal-length L10-lift programs from near-target growth material pair-drops."""
    import os

    anchor = scan_floor_anchor_program(list(champion))
    if not anchor:
        return []
    champ_l10 = _curriculum_l10_cached(list(anchor))
    if champ_l10 >= 0.85 - 1e-6:
        return []
    from evolution.champion_rollback import collect_growth_material_compress_parents
    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = gate or get_scan_floor_gate(list(anchor))
    champ_n = len(anchor)
    cap = max(1, int(max_seeds))
    material = collect_growth_material_compress_parents(list(anchor), max_parents=16)
    material_sig = tuple(tuple(int(w) for w in prog) for prog in material[:12] if prog)
    anchor_key = tuple(int(x) for x in anchor)
    cache_key = (anchor_key, material_sig, cap)
    cached = _l10_lift_eq_bootstrap_cache.get(cache_key)
    if cached is None:
        # Post-restart cold start: hydrate this anchor's persisted entries
        # (disk read only happens on an in-process miss).
        for key, value in load_l10_lift_eq_bootstrap_cache(list(anchor)).items():
            _l10_lift_eq_bootstrap_cache.setdefault(key, value)
        cached = _l10_lift_eq_bootstrap_cache.get(cache_key)
    if cached is not None:
        return [list(prog) for prog in cached]
    source_cap = max(
        2,
        int(os.environ.get("EE_L10_LIFT_BOOTSTRAP_SOURCES", "6")),
    )
    l10_material = [
        list(src)
        for src in material
        if src and len(src) > champ_n and _curriculum_l10_cached(list(src)) > champ_l10 + 0.01
    ]
    l10_material.sort(
        key=lambda prog: (
            _curriculum_l10_cached(list(prog)),
            -len(prog),
        ),
        reverse=True,
    )
    seen: set[tuple[int, ...]] = {tuple(int(x) for x in anchor)}
    candidates: list[list[int]] = []
    shrink_seeds: list[list[int]] = []
    for src in l10_material[:source_cap]:
        shrink_seeds.append(list(src))
        for mid in enumerate_l10_lift_shrink_intermediates(list(anchor), list(src), gate=gate):
            shrink_seeds.append(list(mid))
    material_cap = int(os.environ.get("EE_L10_LIFT_BOOTSTRAP_MATERIAL_CAP", "4"))
    if material_cap > 0:
        try:
            persist_l10_lift_bootstrap_intermediates(
                list(anchor),
                shrink_seeds,
                cycle=0,
                max_record=material_cap,
            )
        except (ImportError, TypeError, ValueError, OSError):
            pass
    graft_cap = max(
        1,
        int(os.environ.get("EE_L10_LIFT_EQ_GRAFT_PAIR_DROP_CAP", "2")),
    )
    grafted_pair_drops = 0
    for src in shrink_seeds:
        if len(src) - champ_n >= 3:
            continue
        if _curriculum_l10_cached(list(src)) <= champ_l10 + 0.01:
            continue
        for eq in collect_l10_lift_pair_drop_eq_programs(list(anchor), list(src), gate=gate):
            key = tuple(int(x) for x in eq)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(list(eq))
            if grafted_pair_drops >= graft_cap:
                continue
            grafted_pair_drops += 1
            for grafted in expand_l10_lift_eq_champion_grafts(
                list(anchor),
                list(eq),
                gate=gate,
                max_variants=max(
                    4,
                    int(os.environ.get("EE_L10_LIFT_EQ_GRAFT_VARIANT_CAP", "8")),
                ),
            ):
                graft_key = tuple(int(x) for x in grafted)
                if graft_key in seen:
                    continue
                seen.add(graft_key)
                candidates.append(list(grafted))
    candidates.sort(
        key=lambda prog: scan_floor_eq_inject_sort_key(list(anchor), list(prog)),
        reverse=True,
    )
    out = candidates[:cap]
    _l10_lift_eq_bootstrap_cache[cache_key] = tuple(tuple(int(x) for x in prog) for prog in out)
    if len(_l10_lift_eq_bootstrap_cache) > 8:
        _l10_lift_eq_bootstrap_cache.pop(next(iter(_l10_lift_eq_bootstrap_cache)))
    save_l10_lift_eq_bootstrap_cache(list(anchor))
    if out and len(list(anchor)) >= int(os.environ.get("EE_EQ_REFINE_PERSIST_MIN_WORDS", "20")):
        try:
            record_scan_floor_eq_refine_seeds(list(champion), [list(p) for p in out], cycle=0)
        except (ImportError, TypeError, ValueError, OSError):
            pass
    return [list(prog) for prog in out]


def record_scan_floor_eq_refine_seeds(
    champion: list[int],
    seeds: list[list[int]],
    *,
    cycle: int = 0,
) -> None:
    """Persist gate-qualified equal-length crossover ties for alt refine probes."""
    import json
    import os

    anchor = scan_floor_anchor_program(list(champion))
    if not anchor or not seeds:
        return
    min_words = int(os.environ.get("EE_EQ_REFINE_PERSIST_MIN_WORDS", "20"))
    if len(anchor) < min_words:
        return
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = get_scan_floor_gate(anchor)
    champ_t = tuple(int(x) for x in anchor)
    plateau_key = _eq_refine_plateau_key(anchor)
    cap = max(2, int(os.environ.get("EE_SCAN_FLOOR_EQ_REFINE_SEED_CAP", "8")))
    candidates: list[tuple[float, int, list[int]]] = []
    seen: set[tuple[int, ...]] = {champ_t}

    def _add_candidate(prog: list[int]) -> None:
        if not prog or len(prog) != len(anchor):
            return
        key = tuple(int(x) for x in prog)
        if key in seen or not gate.qualifies(list(prog)):
            return
        seen.add(key)
        diff = sum(1 for a, b in zip(anchor, prog) if int(a) != int(b))
        sort_key = scan_floor_eq_inject_sort_key(anchor, list(prog))
        candidates.append((sort_key, diff, [int(x) for x in prog]))

    prior_payload = _read_eq_refine_seeds_payload()
    if _eq_refine_payload_matches_plateau(prior_payload, plateau_key, anchor):
        prior_seeds = prior_payload.get("seeds")
        if isinstance(prior_seeds, list):
            for prog in prior_seeds:
                if isinstance(prog, list):
                    _add_candidate([int(x) for x in prog])
    for prog in seeds:
        _add_candidate(list(prog))
    candidates.sort(
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            item[0][3],
            -item[1],
        ),
        reverse=True,
    )
    kept = [prog for _, _, prog in candidates[:cap]]
    if not kept:
        return
    path = scan_floor_eq_refine_seeds_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "plateau_key": [plateau_key[0], plateau_key[1]],
        "anchor": [int(x) for x in anchor],
        "cycle": int(cycle),
        "seeds": kept,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def should_skip_scan_floor_enrich_equal_length_pool(
    champion: list[int],
    *,
    seed_cap: int = 8,
) -> bool:
    """True when warm disk eq-refine seeds make growth-pool enrich redundant."""
    if not champion:
        return False
    cap = max(1, int(seed_cap))
    seeds = load_scan_floor_eq_refine_seeds(
        list(champion),
        max_seeds=cap,
        probe_material=False,
    )
    return warm_eq_refine_disk_seeds_sufficient(
        list(champion),
        [list(s) for s in seeds],
        seed_cap=cap,
    )


def should_skip_scan_floor_gen0_growth_inject(
    champion: list[int],
    *,
    seed_cap: int = 8,
) -> bool:
    """Skip gen-0 growth crossover build when warm disk seeds cover eq-refine inject."""
    import os

    if os.environ.get("EE_SCAN_FLOOR_WARM_SKIP_GEN0_GROWTH", "1") != "1":
        return False
    return should_skip_scan_floor_enrich_equal_length_pool(
        champion,
        seed_cap=seed_cap,
    )


def scan_floor_growth_pool_cache_path() -> str:
    import os

    return os.path.join(_evolution_project_root(), ".agents", "scan_floor_growth_pool_cache.json")


def load_scan_floor_growth_pool_cache(
    champion: list[int],
) -> tuple[tuple[float, int], tuple[tuple[int, ...], ...], frozenset[tuple[int, ...]]] | None:
    """Load persisted growth-inject pool for this plateau (cold-start throughput)."""
    import json
    import os

    if os.environ.get("EE_SCAN_FLOOR_GROWTH_POOL_DISK", "1") == "0":
        return None
    anchor = scan_floor_anchor_program(list(champion))
    if not anchor:
        return None
    plateau_key = _eq_refine_plateau_key(anchor)
    path = scan_floor_growth_pool_cache_path()
    if not os.path.isfile(path):
        return None
    try:
        payload = json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    # v2 payloads keep one entry per plateau so a transient growth-champion
    # save cannot orphan the stable champion's pool (cold-start rebuild);
    # legacy single-payload files are still read.
    candidates: list[dict] = []
    if isinstance(payload.get("entries"), list):
        candidates.extend(entry for entry in payload["entries"] if isinstance(entry, dict))
    if "pool" in payload:
        candidates.append(payload)
    for candidate in candidates:
        if not _eq_refine_payload_matches_plateau(candidate, plateau_key, anchor):
            continue
        pool_raw = candidate.get("pool")
        if not isinstance(pool_raw, list) or not pool_raw:
            continue
        pool: list[tuple[int, ...]] = []
        for item in pool_raw:
            if not isinstance(item, list):
                pool = []
                break
            pool.append(tuple(int(x) for x in item))
        if not pool:
            continue
        cross_raw = candidate.get("cross_keys")
        cross_keys: set[tuple[int, ...]] = set()
        if isinstance(cross_raw, list):
            for item in cross_raw:
                if isinstance(item, list):
                    cross_keys.add(tuple(int(x) for x in item))
        return (plateau_key, tuple(pool), frozenset(cross_keys))
    return None


def save_scan_floor_growth_pool_cache(
    champion: list[int],
    pool: list[list[int]],
    *,
    cross_keys: set[tuple[int, ...]] | frozenset[tuple[int, ...]] | None = None,
    cycle: int = 0,
) -> None:
    """Persist ranked growth-inject pool so the first post-restart cycle stays warm."""
    import json
    import os

    if os.environ.get("EE_SCAN_FLOOR_GROWTH_POOL_DISK", "1") == "0":
        return
    anchor = scan_floor_anchor_program(list(champion))
    if not anchor or not pool:
        return
    plateau_key = _eq_refine_plateau_key(anchor)
    path = scan_floor_growth_pool_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "plateau_key": [plateau_key[0], plateau_key[1]],
        "anchor": [int(x) for x in anchor],
        "cycle": int(cycle),
        "pool": [[int(x) for x in prog] for prog in pool],
        "cross_keys": [[int(x) for x in key] for key in (cross_keys or ())],
    }
    # One entry per plateau: without this a transient growth-champion save
    # overwrote the stable champion's pool, so the first post-restart cycle
    # always missed the disk cache and rebuilt it inside generation 0.
    entries: list[dict] = []
    if os.path.isfile(path):
        try:
            prior = json.loads(open(path, encoding="utf-8").read())
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            prior = None
        if isinstance(prior, dict):
            if isinstance(prior.get("entries"), list):
                entries = [entry for entry in prior["entries"] if isinstance(entry, dict)]
            elif "pool" in prior:
                entries = [prior]
    kept: list[dict] = []
    for entry in entries:
        try:
            same_identity = _eq_refine_payload_matches_plateau(entry, plateau_key, anchor)
        except (TypeError, ValueError):
            same_identity = False
        if not same_identity:
            kept.append(entry)
    kept.append(payload)
    if len(kept) > 8:
        kept.sort(key=lambda e: int(e.get("cycle") or 0))
        kept = kept[-8:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 2, "entries": kept}, f)


def l10_lift_eq_bootstrap_cache_path() -> str:
    import os

    return os.path.join(
        _evolution_project_root(),
        ".agents",
        "scan_floor_l10_lift_eq_bootstrap_cache.json",
    )


def projection_seeds_cache_path() -> str:
    import os

    return os.path.join(
        _evolution_project_root(),
        ".agents",
        "scan_floor_projection_seeds_cache.json",
    )


def _encode_bootstrap_entry_key(cache_key: tuple) -> dict | None:
    """Serialize an ``(anchor, material_sig, cap)`` bootstrap cache key."""
    try:
        anchor_key, material_sig, cap = cache_key
        return {
            "anchor": [int(x) for x in anchor_key],
            "material": [[int(x) for x in prog] for prog in material_sig],
            "cap": int(cap),
        }
    except (TypeError, ValueError):
        return None


def _decode_bootstrap_entry_key(raw: object) -> tuple | None:
    if not isinstance(raw, dict):
        return None
    try:
        anchor = raw.get("anchor")
        material = raw.get("material")
        cap = raw.get("cap")
        if not isinstance(anchor, list) or not isinstance(material, list):
            return None
        return (
            tuple(int(x) for x in anchor),
            tuple(tuple(int(x) for x in prog) for prog in material),
            int(cap),
        )
    except (TypeError, ValueError):
        return None


def _encode_projection_entry_key(cache_key: tuple) -> dict | None:
    """Serialize a ``(champ, max_seeds, material_sig)`` projection cache key."""
    try:
        champ_t, max_seeds, material_sig = cache_key
        return {
            "anchor": [int(x) for x in champ_t],
            "material": [[int(x) for x in prog] for prog in material_sig],
            "cap": int(max_seeds),
        }
    except (TypeError, ValueError):
        return None


def _decode_projection_entry_key(raw: object) -> tuple | None:
    if not isinstance(raw, dict):
        return None
    try:
        anchor = raw.get("anchor")
        material = raw.get("material")
        cap = raw.get("cap")
        if not isinstance(anchor, list) or not isinstance(material, list):
            return None
        return (
            tuple(int(x) for x in anchor),
            int(cap),
            tuple(tuple(int(x) for x in prog) for prog in material),
        )
    except (TypeError, ValueError):
        return None


def _load_anchor_keyed_cache(
    path: str,
    champion: list[int],
    decode_key,
    *,
    env_switch: str,
) -> dict:
    """Read one anchor-keyed cache file; entries must match this anchor (W1b).

    Plateau match alone is not enough: two same-prom, same-length anchors share
    a plateau, so every entry's anchor tuple is re-checked before hydration.
    """
    import json
    import os

    if os.environ.get(env_switch, "1") == "0":
        return {}
    anchor = scan_floor_anchor_program(list(champion))
    if not anchor:
        return {}
    plateau_key = _eq_refine_plateau_key(anchor)
    if not os.path.isfile(path):
        return {}
    try:
        payload = json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if not _eq_refine_payload_matches_plateau(payload, plateau_key, anchor):
        return {}
    entries_raw = payload.get("entries")
    if not isinstance(entries_raw, list):
        return {}
    anchor_t = tuple(int(x) for x in anchor)
    hydrated: dict = {}
    for entry in entries_raw:
        if not isinstance(entry, dict):
            continue
        key = decode_key(entry.get("key"))
        if key is None or tuple(int(x) for x in key[0]) != anchor_t:
            continue
        value_raw = entry.get("value")
        if not isinstance(value_raw, list):
            continue
        try:
            value = tuple(tuple(int(x) for x in prog) for prog in value_raw)
        except (TypeError, ValueError):
            continue
        hydrated[key] = value
    return hydrated


def _save_anchor_keyed_cache(
    path: str,
    cache: dict,
    champion: list[int],
    encode_key,
    *,
    env_switch: str,
    cycle: int = 0,
) -> None:
    import json
    import os

    if os.environ.get(env_switch, "1") == "0":
        return
    anchor = scan_floor_anchor_program(list(champion))
    if not anchor:
        return
    anchor_t = tuple(int(x) for x in anchor)
    entries = []
    for key, value in cache.items():
        enc = encode_key(key)
        if enc is None or tuple(int(x) for x in enc["anchor"]) != anchor_t:
            continue
        try:
            entries.append(
                {
                    "key": enc,
                    "value": [[int(x) for x in prog] for prog in value],
                }
            )
        except (TypeError, ValueError):
            continue
    if not entries:
        return
    plateau_key = _eq_refine_plateau_key(anchor)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "plateau_key": [plateau_key[0], plateau_key[1]],
        "anchor": [int(x) for x in anchor],
        "cycle": int(cycle),
        "entries": entries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_l10_lift_eq_bootstrap_cache(champion: list[int]) -> dict:
    """Hydrate persisted l10-lift bootstrap seeds (post-restart cold start).

    ``_l10_lift_eq_bootstrap_cache`` is documented as a ~9min cold build; disk
    hydration lets a fresh process skip it when the plateau/anchor still match.
    """
    return _load_anchor_keyed_cache(
        l10_lift_eq_bootstrap_cache_path(),
        list(champion),
        _decode_bootstrap_entry_key,
        env_switch="EE_L10_LIFT_BOOTSTRAP_DISK",
    )


def save_l10_lift_eq_bootstrap_cache(champion: list[int], *, cycle: int = 0) -> None:
    """Persist current l10-lift bootstrap entries for this anchor."""
    _save_anchor_keyed_cache(
        l10_lift_eq_bootstrap_cache_path(),
        _l10_lift_eq_bootstrap_cache,
        list(champion),
        _encode_bootstrap_entry_key,
        env_switch="EE_L10_LIFT_BOOTSTRAP_DISK",
        cycle=int(cycle),
    )


def load_projection_seeds_cache(champion: list[int]) -> dict:
    """Hydrate persisted projection seeds (post-restart cold start)."""
    return _load_anchor_keyed_cache(
        projection_seeds_cache_path(),
        list(champion),
        _decode_projection_entry_key,
        env_switch="EE_PROJECTION_SEEDS_DISK",
    )


def save_projection_seeds_cache(champion: list[int], *, cycle: int = 0) -> None:
    """Persist current projection-seed entries for this anchor."""
    _save_anchor_keyed_cache(
        projection_seeds_cache_path(),
        _projection_seeds_cache,
        list(champion),
        _encode_projection_entry_key,
        env_switch="EE_PROJECTION_SEEDS_DISK",
        cycle=int(cycle),
    )


def scan_floor_eq_refine_late_merged_cache_path() -> str:
    import os

    return os.path.join(
        _evolution_project_root(),
        ".agents",
        "scan_floor_eq_refine_late_merged_cache.json",
    )


def _normalize_late_cache_key(key: list | tuple) -> tuple:
    normalized: list = []
    for item in key:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                normalized.append((round(float(item[0]), 4), int(item[1])))
            except (TypeError, ValueError):
                normalized.append(item)
        else:
            normalized.append(item)
    return tuple(normalized)


def load_eq_refine_late_merged_cache(
    champion: list[int],
    late_cache_key: tuple,
) -> tuple[tuple[int, ...], ...] | None:
    """Load persisted late-inject merged seed pool (survives daemon restart)."""
    import json
    import os

    if os.environ.get("EE_EQ_REFINE_LATE_MERGED_DISK", "1") == "0":
        return None
    anchor = scan_floor_anchor_program(list(champion))
    if not anchor:
        return None
    plateau_key = _eq_refine_plateau_key(anchor)
    path = scan_floor_eq_refine_late_merged_cache_path()
    if not os.path.isfile(path):
        return None
    try:
        payload = json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if not _eq_refine_payload_matches_plateau(payload, plateau_key, anchor):
        return None
    stored_key = payload.get("late_cache_key")
    if not isinstance(stored_key, list):
        return None
    if _normalize_late_cache_key(stored_key) != _normalize_late_cache_key(late_cache_key):
        return None
    seeds_raw = payload.get("merged")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        return None
    merged: list[tuple[int, ...]] = []
    for item in seeds_raw:
        if not isinstance(item, list):
            return None
        merged.append(tuple(int(x) for x in item))
    return tuple(merged)


def save_eq_refine_late_merged_cache(
    champion: list[int],
    late_cache_key: tuple,
    merged: list[list[int]],
    *,
    cycle: int = 0,
) -> None:
    """Persist late-inject merged seeds so post-restart cycles skip heavy probes."""
    import json
    import os

    if os.environ.get("EE_EQ_REFINE_LATE_MERGED_DISK", "1") == "0":
        return
    if not merged:
        return
    anchor = scan_floor_anchor_program(list(champion))
    if not anchor:
        return
    plateau_key = _eq_refine_plateau_key(anchor)
    path = scan_floor_eq_refine_late_merged_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "plateau_key": [plateau_key[0], plateau_key[1]],
        "anchor": [int(x) for x in anchor],
        "late_cache_key": [
            [plateau_key[0], plateau_key[1]],
            late_cache_key[1],
            late_cache_key[2],
            late_cache_key[3],
        ],
        "cycle": int(cycle),
        "merged": [[int(x) for x in prog] for prog in merged],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def should_skip_mg_stall_collect_for_warm_eq_refine(champion: list[int]) -> bool:
    """Skip bulk model-guided stall collect when late eq-refine pool is already warm."""
    import os

    if os.environ.get("EE_SCAN_FLOOR_MG_STALL_COLLECT_WARM_SKIP", "1") != "1":
        return False
    if not champion:
        return False
    seed_cap = max(2, int(os.environ.get("EE_SCAN_FLOOR_EQ_REFINE_SEED_CAP", "8")))
    seeds = load_scan_floor_eq_refine_seeds(list(champion), max_seeds=seed_cap, probe_material=False)
    if not seeds:
        return False
    warm = warm_eq_refine_disk_seeds_sufficient(list(champion), [list(s) for s in seeds], seed_cap=seed_cap)
    force_late = os.environ.get("EE_EQ_REFINE_FORCE_LATE_PROBE", "0") == "1"
    late_key = (
        _eq_refine_plateau_key(list(champion)),
        bool(warm),
        int(seed_cap),
        bool(force_late),
    )
    if _eq_refine_late_merged_cache.get(late_key):
        return True
    return load_eq_refine_late_merged_cache(list(champion), late_key) is not None


def warm_eq_refine_disk_seeds_sufficient(
    champion: list[int],
    seeds: list[list[int]],
    *,
    seed_cap: int = 8,
) -> bool:
    """True when persisted eq-refine seeds cover late inject without heavy projection."""
    if not champion or not seeds:
        return False
    cap = max(1, int(seed_cap))
    if len(seeds) < cap:
        return False
    champ_l10 = _curriculum_l10_cached(list(champion))
    if champ_l10 >= 0.85 - 1e-6:
        return True
    l10_lift = sum(1 for prog in seeds if _curriculum_l10_cached(list(prog)) > champ_l10 + 0.02)
    return l10_lift >= min(2, cap)


def collect_growth_material_projection_seeds(
    champion: list[int],
    *,
    max_seeds: int = 8,
    cycle_no: int = 0,
    gate=None,
) -> list[list[int]]:
    """Prom-ranked equal-length projections from recorded growth material (CPU-only)."""
    import os

    from evolution.champion_rollback import collect_growth_material_compress_parents
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.terminal_stagnation import curriculum_verify_mean, get_scan_floor_gate

    anchor = scan_floor_anchor_program(list(champion))
    if not anchor:
        return []
    champ = list(anchor)
    champ_t = tuple(int(x) for x in champ)
    gate = gate or get_scan_floor_gate(champ)
    parent_cap = max(
        4,
        int(os.environ.get("EE_PROJECTION_MATERIAL_PARENT_CAP", "12")),
    )
    material = collect_growth_material_compress_parents(champ, max_parents=parent_cap)
    if not material:
        return []
    material_sig = tuple(tuple(int(w) for w in prog) for prog in material[:16] if prog)
    cache_key = (champ_t, int(max_seeds), material_sig)
    cached = _projection_seeds_cache.get(cache_key)
    if cached is None:
        # Post-restart cold start: hydrate this anchor's persisted entries.
        for key, value in load_projection_seeds_cache(list(champ)).items():
            _projection_seeds_cache.setdefault(key, value)
        cached = _projection_seeds_cache.get(cache_key)
    if cached is not None:
        out_cached = [list(prog) for prog in cached]
        if out_cached:
            start = int(cycle_no) % len(out_cached)
            out_cached = out_cached[start:] + out_cached[:start]
        return out_cached[: max(1, int(max_seeds))]
    drop_trials = max(
        24,
        int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_DROP_TRIALS", "48")),
    )
    rot_cap = max(
        1,
        int(os.environ.get("EE_PROJECTION_MATERIAL_ROTATIONS", "2")),
    )
    expanded: list[list[int]] = []
    seen_exp: set[tuple[int, ...]] = {champ_t}
    for rot in range(rot_cap):
        try:
            projected = _expand_growth_material_for_grafts(
                champ,
                material,
                gate=gate,
                cycle_no=int(cycle_no) + rot,
            )
        except (ImportError, TypeError, ValueError, OSError):
            projected = []
        for eq_src in projected:
            if not eq_src or len(eq_src) != len(champ):
                continue
            key = tuple(int(x) for x in eq_src)
            if key in seen_exp:
                continue
            if not gate.qualifies(list(eq_src)):
                continue
            seen_exp.add(key)
            expanded.append(list(eq_src))
        for src in material:
            if not src or len(src) <= len(champ):
                continue
            eq = equal_length_from_longer_growth(
                champ,
                list(src),
                gate=gate,
                max_drop_trials=drop_trials,
                cycle_no=int(cycle_no) + rot,
            )
            if len(eq) != len(champ):
                continue
            key = tuple(int(x) for x in eq)
            if key in seen_exp:
                continue
            if not gate.qualifies(eq):
                continue
            seen_exp.add(key)
            expanded.append(list(eq))
    champ_l10_weak = _curriculum_l10_cached(champ) < 0.85 - 1e-6
    ranked: list[tuple[float, ...]] = []
    for prog in expanded:
        try:
            prom = float(compact_l30_promotion_raw_fitness(list(prog)))
        except (RuntimeError, TypeError, ValueError, OSError):
            continue
        if champ_l10_weak:
            ranked.append(
                (
                    prom,
                    _curriculum_l10_cached(list(prog)),
                    list(prog),
                )
            )
        else:
            ranked.append((prom, list(prog)))
    if champ_l10_weak:
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    else:
        ranked.sort(key=lambda item: item[0], reverse=True)
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = {champ_t}
    for item in ranked:
        prog = item[-1]
        key = tuple(int(x) for x in prog)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(prog))
        if len(out) >= max(1, int(max_seeds)):
            break
    stored = tuple(tuple(int(w) for w in prog) for prog in out)
    _projection_seeds_cache[cache_key] = stored
    if len(_projection_seeds_cache) > 32:
        _projection_seeds_cache.pop(next(iter(_projection_seeds_cache)))
    save_projection_seeds_cache(list(champ))
    return out


def scan_floor_late_inject_generation(
    generation: int,
    *,
    growth_inject_generation: int | None,
    mg_inject_generation: int | None,
) -> bool:
    """True on growth/MG inject gens where heavy CPU material probes are allowed."""
    gen = int(generation)
    if growth_inject_generation is not None and gen == int(growth_inject_generation):
        return True
    if mg_inject_generation is not None and gen == int(mg_inject_generation):
        return True
    return False


def inject_scan_floor_eq_refine_seed_challengers(
    champion: list[int],
    challengers: list,
    rng,
    *,
    cycle_no: int = 0,
    generation: int = 0,
    growth_inject_generation: int | None = None,
    mg_inject_generation: int | None = None,
    strategies: list | None = None,
) -> int:
    """Inject cached equal-length crossover seeds into challenger slots when stalled."""
    import os

    if not champion or not challengers:
        return 0
    try:
        from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

        if not _scan_floor_probe_stall_mode(len(champion)):
            return 0
    except ImportError:
        return 0
    seed_cap = max(1, int(os.environ.get("EE_SCAN_FLOOR_EQ_REFINE_INJECT_SLOTS", "8")))
    late_probe = scan_floor_late_inject_generation(
        int(generation),
        growth_inject_generation=growth_inject_generation,
        mg_inject_generation=mg_inject_generation,
    )
    if (
        not late_probe
        and os.environ.get("EE_EQ_REFINE_WARM_INJECT_LATE_ONLY", "1") == "1"
        and should_skip_mg_stall_collect_for_warm_eq_refine(list(champion))
    ):
        return 0
    seeds = load_scan_floor_eq_refine_seeds(
        champion,
        max_seeds=seed_cap,
        probe_material=late_probe,
    )
    if not late_probe:
        if not seeds:
            return 0
        n_slots = min(len(seeds), len(challengers), seed_cap)
        slots = rng.sample(range(len(challengers)), n_slots)
        rotated = [seeds[(int(cycle_no) + int(generation) + i) % len(seeds)] for i in range(n_slots)]
        for slot, variant in zip(slots, rotated, strict=False):
            challengers[slot] = list(variant)
            if strategies is not None and slot < len(strategies):
                strategies[slot] = "EQ_REFINE_SEED"
        return n_slots
    try:
        from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
        from evolution.terminal_stagnation import get_scan_floor_gate, terminal_growth_material_locus_probe_variants

        gate = get_scan_floor_gate(list(champion))
        warm_disk = warm_eq_refine_disk_seeds_sufficient(
            list(champion),
            [list(s) for s in seeds],
            seed_cap=seed_cap,
        )
        force_late = os.environ.get("EE_EQ_REFINE_FORCE_LATE_PROBE", "0") == "1"
        anchor = scan_floor_anchor_program(list(champion))
        late_cache_key = (
            _eq_refine_plateau_key(list(champion)),
            bool(warm_disk),
            int(seed_cap),
            bool(force_late),
        )
        cached_merged = _eq_refine_late_merged_cache.get(late_cache_key)
        if cached_merged is None:
            disk_merged = load_eq_refine_late_merged_cache(list(champion), late_cache_key)
            if disk_merged is not None:
                cached_merged = disk_merged
                _eq_refine_late_merged_cache[late_cache_key] = disk_merged
        if cached_merged is not None:
            seeds = [list(prog) for prog in cached_merged]
        else:
            locus_budget = max(
                8,
                int(os.environ.get("EE_SCAN_FLOOR_GROWTH_LOCUS_INJECT", "12")),
            )
            if warm_disk and not force_late:
                locus_budget = min(
                    locus_budget,
                    max(4, int(os.environ.get("EE_EQ_REFINE_WARM_LOCUS_BUDGET", "6"))),
                )
            skip_locus = warm_disk and not force_late and os.environ.get("EE_EQ_REFINE_WARM_SKIP_LOCUS", "1") == "1"
            if skip_locus:
                locus = []
            else:
                locus = terminal_growth_material_locus_probe_variants(
                    list(champion),
                    max_evals=locus_budget,
                    start_offset=int(cycle_no),
                    gate=gate,
                )
            projections: list[list[int]] = []
            if not warm_disk or force_late:
                projections = collect_growth_material_projection_seeds(
                    list(champion),
                    max_seeds=seed_cap,
                    cycle_no=int(cycle_no),
                    gate=gate,
                )
            champ_t = tuple(int(x) for x in champion)
            seen: set[tuple[int, ...]] = {champ_t}
            merged: list[list[int]] = []
            for prog in list(projections) + list(locus) + list(seeds):
                key = tuple(int(x) for x in prog)
                if key in seen:
                    continue
                seen.add(key)
                merged.append([int(x) for x in prog])
            if merged:
                champ_anchor = list(champion)
                merged.sort(
                    key=lambda variant: scan_floor_eq_inject_sort_key(champ_anchor, list(variant)),
                    reverse=True,
                )
                skip_perturb = (
                    warm_disk
                    and not force_late
                    and os.environ.get("EE_EQ_REFINE_WARM_SKIP_PERTURB", "1") == "1"
                    and warm_eq_refine_disk_seeds_sufficient(
                        list(champion),
                        [list(s) for s in seeds],
                        seed_cap=seed_cap,
                    )
                )
                perturb_budget = (
                    0
                    if skip_perturb
                    else max(
                        0,
                        int(os.environ.get("EE_SCAN_FLOOR_EQ_REFINE_ONE_STEP_PERTURB", "8")),
                    )
                )
                if perturb_budget > 0:
                    try:
                        from evolution.terminal_stagnation import curriculum_verify_mean

                        if _curriculum_l10_cached(list(champion)) < 0.85 - 1e-6:
                            perturb_budget = max(
                                perturb_budget,
                                int(os.environ.get("EE_SCAN_FLOOR_L10_WEAK_EQ_PERTURB", "12")),
                            )
                    except ImportError:
                        pass
                if perturb_budget > 0:
                    perturbed = refine_eq_seed_one_step_perturb_variants(
                        merged[: max(4, seed_cap)],
                        rng,
                        max_evals=perturb_budget,
                        gate=gate,
                        champion=list(champion),
                    )
                    two_step_budget = max(
                        0,
                        int(os.environ.get("EE_SCAN_FLOOR_EQ_REFINE_TWO_STEP_PERTURB", "4")),
                    )
                    if two_step_budget > 0 and perturbed:
                        perturbed.extend(
                            refine_eq_seed_two_step_perturb_variants(
                                perturbed[: max(2, two_step_budget)],
                                rng,
                                max_evals=two_step_budget,
                                gate=gate,
                                champion=list(champion),
                            )
                        )
                    if perturbed:
                        seen = {tuple(int(x) for x in p) for p in merged}
                        for prog in perturbed:
                            key = tuple(int(x) for x in prog)
                            if key in seen:
                                continue
                            seen.add(key)
                            merged.append(list(prog))
                        merged.sort(
                            key=lambda variant: scan_floor_eq_inject_sort_key(champ_anchor, list(variant)),
                            reverse=True,
                        )
                seeds = merged
                record_scan_floor_eq_refine_seeds(
                    anchor,
                    [list(p) for p in seeds[: max(seed_cap, 8)]],
                    cycle=int(cycle_no),
                )
            cache_tuple = tuple(tuple(int(x) for x in prog) for prog in seeds)
            _eq_refine_late_merged_cache[late_cache_key] = cache_tuple
            save_eq_refine_late_merged_cache(
                list(champion),
                late_cache_key,
                [list(prog) for prog in seeds],
                cycle=int(cycle_no),
            )
    except ImportError:
        pass
    if not seeds:
        return 0
    n_slots = min(
        len(seeds),
        len(challengers),
        seed_cap,
    )
    slots = rng.sample(range(len(challengers)), n_slots)
    rotated = [seeds[(int(cycle_no) + int(generation) + i) % len(seeds)] for i in range(n_slots)]
    for slot, variant in zip(slots, rotated, strict=False):
        challengers[slot] = list(variant)
        if strategies is not None and slot < len(strategies):
            strategies[slot] = "EQ_REFINE_SEED"
    return n_slots


def load_scan_floor_eq_refine_seeds(
    champion: list[int],
    *,
    max_seeds: int = 8,
    probe_material: bool = True,
) -> list[list[int]]:
    """Load alternate equal-length programs for refine probes when locally stalled."""
    anchor = scan_floor_anchor_program(list(champion))
    if not anchor:
        return []
    plateau_key = _eq_refine_plateau_key(anchor)
    payload = _read_eq_refine_seeds_payload()
    if not _eq_refine_payload_matches_plateau(payload, plateau_key, anchor):
        payload = {}
    seeds = payload.get("seeds")
    if not isinstance(seeds, list):
        seeds = []
    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = get_scan_floor_gate(list(champion))
    champ_t = tuple(int(x) for x in champion)
    champ_l10 = _curriculum_l10_cached(list(anchor))
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = {champ_t}
    for prog in seeds:
        if not isinstance(prog, list) or len(prog) != len(anchor):
            continue
        key = tuple(int(x) for x in prog)
        if key in seen or not gate.qualifies(list(prog)):
            continue
        seen.add(key)
        out.append([int(x) for x in prog])
    if champ_l10 < 0.85 - 1e-6:
        l10_lift_disk = sum(1 for prog in out if _curriculum_l10_cached(list(prog)) > champ_l10 + 0.02)
        need_bootstrap = l10_lift_disk < min(2, max(1, int(max_seeds)))
        if need_bootstrap:
            for prog in collect_l10_lift_eq_refine_bootstrap_seeds(
                list(champion),
                gate=gate,
                max_seeds=max(4, int(max_seeds)),
            ):
                key = tuple(int(x) for x in prog)
                if key in seen:
                    continue
                seen.add(key)
                out.append(list(prog))
                if len(out) >= max(1, int(max_seeds)):
                    break
    if len(out) >= max(1, int(max_seeds)):
        return out[: max(1, int(max_seeds))]
    if probe_material and len(out) < max(1, int(max_seeds)):
        for prog in collect_growth_material_projection_seeds(
            list(champion),
            max_seeds=int(max_seeds),
            cycle_no=0,
            gate=gate,
        ):
            key = tuple(int(x) for x in prog)
            if key in seen:
                continue
            seen.add(key)
            out.append(list(prog))
            if len(out) >= max(1, int(max_seeds)):
                break
    if len(out) < max(1, int(max_seeds)):
        try:
            from evolution.champion_rollback import _load_growth_material_entries

            _anchor, entries = _load_growth_material_entries(list(champion))
            material_ranked: list[tuple[float, int, list[int]]] = []
            for entry in entries:
                prog = entry.get("program")
                if not isinstance(prog, list) or len(prog) < len(anchor):
                    continue
                prom = float(entry.get("prom") or 0.0)
                if len(prog) == len(anchor):
                    key = tuple(int(x) for x in prog)
                    if key in seen or not gate.qualifies(list(prog)):
                        continue
                    diff = sum(1 for a, b in zip(anchor, prog) if int(a) != int(b))
                    material_ranked.append((prom, diff, list(prog)))
                    continue
                diffs = [i for i in range(len(anchor)) if int(prog[i]) != int(anchor[i])]
                if not diffs or len(diffs) > 8:
                    continue
                for idx in diffs[: max(1, int(max_seeds))]:
                    graft = list(anchor)
                    graft[idx] = int(prog[idx])
                    key = tuple(graft)
                    if key in seen or not gate.qualifies(graft):
                        continue
                    material_ranked.append((prom, 1, graft))
            material_ranked.sort(key=lambda item: (-item[0], item[1]))
            for _prom, _diff, prog in material_ranked:
                key = tuple(int(x) for x in prog)
                if key in seen:
                    continue
                seen.add(key)
                out.append(list(prog))
                if len(out) >= max(1, int(max_seeds)):
                    break
        except ImportError:
            pass
    if out:
        return out
    if plateau_key in _eq_refine_bootstrapped:
        return []
    _eq_refine_bootstrapped.add(plateau_key)
    try:
        from evolution.champion_rollback import collect_growth_material_compress_parents

        material = collect_growth_material_compress_parents(list(champion), max_parents=8)
        if material:
            boot = growth_material_diff_graft_variants(
                list(champion),
                material,
                max_variants=max(4, int(max_seeds)),
                max_pair_combos=4,
            )
            for prog in boot:
                key = tuple(prog)
                if key in seen or not gate.qualifies(list(prog)):
                    continue
                seen.add(key)
                out.append(list(prog))
                if len(out) >= max(1, int(max_seeds)):
                    break
            if out:
                record_scan_floor_eq_refine_seeds(list(champion), out, cycle=0)
    except ImportError:
        return []
    return out


def _growth_crossover_candidate(
    champ: list[int],
    hybrid: list[int],
    *,
    gate,
    champ_t: tuple[int, ...],
    champ_pr: float,
    min_gain: float,
    drop_trials: int,
    cycle_no: int | None = None,
) -> list[int] | None:
    """Normalize a crossover hybrid into a held-out-safe growth pool candidate."""
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    if not hybrid or tuple(hybrid) == champ_t:
        return None
    if not scan_floor_growth_heldout_qualifies(champ, hybrid):
        return None
    candidate = list(hybrid)
    max_words = scan_floor_growth_crossover_max_words(len(champ))
    if len(candidate) > len(champ):
        from evolution.terminal_stagnation import curriculum_verify_mean

        hybrid_l10 = _curriculum_l10_cached(candidate)
        champ_l10 = _curriculum_l10_cached(champ)
        if (
            hybrid_l10 > champ_l10 + 0.01
            and len(candidate) <= max_words
            and scan_floor_growth_completion_qualifies(champ, candidate, gate=gate)
        ):
            if candidate[-1] != 0x0010_0073:
                candidate.append(0x0010_0073)
            return list(candidate)
        compressed = equal_length_from_longer_growth(
            champ,
            candidate,
            gate=gate,
            max_drop_trials=drop_trials,
            cycle_no=cycle_no,
        )
        if len(compressed) == len(champ) and tuple(compressed) != champ_t:
            if gate.qualifies(compressed) and tuple(compressed) != champ_t:
                comp_pr = float(compact_l30_promotion_raw_fitness(compressed))
                if (
                    comp_pr > champ_pr + min_gain - 1e-6 or comp_pr + 1e-6 >= champ_pr
                ) and scan_floor_growth_completion_qualifies(champ, compressed, gate=gate):
                    candidate = list(compressed)
    if tuple(candidate) == champ_t or len(candidate) > max_words:
        return None
    if len(candidate) != len(champ) and not scan_floor_growth_heldout_qualifies(champ, candidate):
        return None
    if candidate[-1] != 0x0010_0073:
        candidate.append(0x0010_0073)
    return list(candidate)


def build_scan_floor_growth_crossover_pool(
    champion: list[int],
    fills: list[list[int]],
    rng,
    *,
    cycle_no: int = 0,
    max_attempts: int = 6,
) -> list[list[int]]:
    """Build held-out-safe champion×growth hybrids once per cycle (amortize eval cost)."""
    import os

    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.terminal_stagnation import get_scan_floor_gate, scan_floor_min_raw_gain
    from mutation_proposer import _crossover_mutation

    if not champion or not fills:
        return []
    global _growth_crossover_pool_cache
    champ = list(champion)
    champ_t = tuple(champ)
    cycle_key = int(cycle_no)
    attempts = max(1, int(max_attempts))
    champ_pr = float(compact_l30_promotion_raw_fitness(champ))
    cross_cache_key = (round(champ_pr, 4), len(champ), attempts)
    cross_cached = _growth_crossover_pool_cache.get(cross_cache_key)
    if cross_cached is not None:
        return [list(prog) for prog in cross_cached]
    gate = get_scan_floor_gate(champ)
    min_gain = float(scan_floor_min_raw_gain(len(champ)))
    probs = (0.35, 0.5, 0.65)
    ordered_fills = sorted(
        fills,
        key=lambda f: (abs(len(f) - len(champ)), len(f)),
    )
    stalled = False
    try:
        from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

        stalled = _scan_floor_probe_stall_mode(len(champ))
    except ImportError:
        stalled = False
    drop_trials = max(24, 8)
    if stalled:
        drop_trials = max(
            drop_trials,
            int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_DROP_TRIALS", "48")),
        )
    pool: list[list[int]] = []
    seen: set[tuple[int, ...]] = {champ_t}
    for step in range(attempts):
        fill = ordered_fills[step % len(ordered_fills)]
        if tuple(fill) == champ_t:
            continue
        prob = probs[step % len(probs)]
        cross_pairs = [(list(fill), champ, prob)]
        if stalled:
            cross_pairs.append((champ, list(fill), prob))
        for program_parent, champion_parent, cross_prob in cross_pairs:
            hybrid = _crossover_mutation(
                program=list(program_parent),
                task_level=30,
                rng=rng,
                champion=list(champion_parent),
                champion_crossover_prob=cross_prob,
            )
            candidate = _growth_crossover_candidate(
                champ,
                list(hybrid) if hybrid else [],
                gate=gate,
                champ_t=champ_t,
                champ_pr=champ_pr,
                min_gain=min_gain,
                drop_trials=drop_trials,
                cycle_no=cycle_key,
            )
            if candidate is None:
                continue
            key = tuple(candidate)
            if key in seen:
                continue
            seen.add(key)
            pool.append(list(candidate))

    from evolution.terminal_stagnation import curriculum_verify_mean

    champ_l10 = _curriculum_l10_cached(champ)
    champ_l10_weak = champ_l10 < 0.85 - 1e-6

    def _pool_rank(prog: list[int]) -> tuple[float, ...]:
        prog_l10 = _curriculum_l10_cached(prog)
        eq_gate = len(prog) == len(champ) and gate.qualifies(prog)
        if champ_l10_weak and prog_l10 > champ_l10 + 0.01:
            pr = float(compact_l30_promotion_raw_fitness(prog))
            return (3.0, prog_l10, pr, -len(prog))
        if eq_gate:
            pr = float(compact_l30_promotion_raw_fitness(prog))
            return (2.0, pr, prog_l10, 0.0)
        if scan_floor_growth_heldout_qualifies(champ, prog):
            return (1.0, prog_l10, 0.0, -float(len(prog)))
        return (0.0, prog_l10, -1.0, 0.0)

    pool.sort(key=_pool_rank, reverse=True)
    equal_gate = [p for p in pool if len(p) == len(champ) and gate.qualifies(p)]
    if equal_gate:
        if champ_l10_weak:
            l10_eq = [p for p in equal_gate if _curriculum_l10_cached(p) > champ_l10 + 0.01]
            if l10_eq:
                rest = [p for p in pool if p not in l10_eq]
                pool = l10_eq + rest
        else:
            rest = [p for p in pool if p not in equal_gate]
            pool = equal_gate + rest
    _growth_crossover_pool_cache[cross_cache_key] = tuple(tuple(int(w) for w in prog) for prog in pool)
    if len(_growth_crossover_pool_cache) > _GROWTH_CROSSOVER_CACHE_MAX:
        for stale in list(_growth_crossover_pool_cache.keys())[:-12]:
            if stale != cross_cache_key:
                _growth_crossover_pool_cache.pop(stale, None)
    return pool


def scan_floor_growth_crossover_variants(
    champion: list[int],
    fills: list[list[int]],
    rng,
    *,
    cycle_no: int = 0,
    gen_idx: int = 0,
    max_attempts: int = 6,
) -> list[int] | None:
    """Pick one cached-style hybrid from a growth-fill crossover pool."""
    pool = build_scan_floor_growth_crossover_pool(
        champion,
        fills,
        rng,
        cycle_no=cycle_no,
        max_attempts=max_attempts,
    )
    if not pool:
        return None
    return list(pool[(cycle_no + gen_idx) % len(pool)])


def scan_floor_growth_compress_sources(
    champion: list[int],
    long_candidates: list[list[int]],
    *,
    stalled: bool,
    max_trials: int | None = None,
) -> list[list[int]]:
    """Pick long growth parents worth compressing (near-length unless ALL mode)."""
    import os

    if not champion or not long_candidates or not stalled:
        return []
    champ_n = len(champion)
    cap = max(1, int(max_trials or os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_CAP", "6")))
    long_pool = [list(p) for p in long_candidates if p and len(p) > champ_n]
    if stalled:
        from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

        try:
            champ_pr = float(compact_l30_promotion_raw_fitness(list(champion)))
        except (RuntimeError, TypeError, ValueError, OSError):
            champ_pr = 0.0
        target_len = champ_n + max(
            2,
            int(os.environ.get("EE_GROWTH_COMPRESS_TARGET_EXTRA_WORDS", "4")),
        )

        def _compress_source_rank(prog: list[int]) -> tuple:
            prog_pr = float(compact_l30_promotion_raw_fitness(list(prog)))
            growth_phase = prog_pr + 1e-6 < champ_pr
            at_champ_prom = prog_pr + 1e-6 >= champ_pr
            if at_champ_prom:
                tier = 0
            elif growth_phase:
                tier = 1
            else:
                tier = 2
            return (tier, abs(len(prog) - target_len), -prog_pr, len(prog))

        sorted_long = sorted(long_pool, key=_compress_source_rank)
    else:
        sorted_long = sorted(long_pool, key=len)
    if os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_ALL", "0") == "1":
        return sorted_long[:cap]
    word_margin = max(
        4,
        int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_WORD_MARGIN", "8")),
    )
    return [p for p in sorted_long if len(p) <= champ_n + word_margin][:cap]


def compress_growth_probe_pool(
    champion: list[int],
    candidates: list[list[int]],
    *,
    gate=None,
    max_trials: int | None = None,
    max_drop_trials: int | None = None,
    cycle_no: int | None = None,
) -> list[list[int]]:
    """Shrink growth probes toward champion length; front-load gate-qualified 48w hits."""
    import os

    from evolution.terminal_stagnation import get_scan_floor_gate

    if not champion or not candidates:
        return list(candidates)
    gate = gate or get_scan_floor_gate(champion)
    champ_n = len(champion)
    champ_pr = float(_compact_prom_raw_cached(list(champion)))
    cap = max(1, int(max_trials or len(candidates)))
    drop_trials = max(
        4,
        int(
            max_drop_trials
            if max_drop_trials is not None
            else os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_DROP_TRIALS", "12")
        ),
    )
    pool_key: tuple | None = None
    if cycle_no is not None:
        global _growth_compress_pool_cache
        pool_key = (
            round(champ_pr, 4),
            champ_n,
            cap,
            drop_trials,
        )
        cached_pool = _growth_compress_pool_cache.get(pool_key)
        if cached_pool is not None:
            return [list(prog) for prog in cached_pool]
    repair_eval_cap = max(
        8,
        int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_REPAIR_EVALS", "12")),
    )
    front: list[list[int]] = []
    rest: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for prog in candidates[:cap]:
        prog_list = list(prog)
        comp = (
            equal_length_from_longer_growth(
                champion,
                prog_list,
                gate=gate,
                max_drop_trials=drop_trials,
                cycle_no=cycle_no,
                repair_eval_cap=repair_eval_cap,
            )
            if len(prog_list) > champ_n
            else prog_list
        )
        if not scan_floor_growth_completion_qualifies(champion, comp, gate=gate):
            comp = list(prog)
        key = tuple(comp)
        if key in seen:
            continue
        seen.add(key)
        comp_pr = float(_compact_prom_raw_cached(comp)) if len(comp) == champ_n else 0.0
        if len(comp) == champ_n and gate.qualifies(comp) and comp_pr + 1e-6 >= champ_pr:
            front.append(list(comp))
            if comp_pr > champ_pr + 1e-6:
                import logging

                logging.getLogger("EvolutionEngine").info(
                    "[GrowthCompress] prom_raw breakthrough candidate %.4f > champion %.4f",
                    comp_pr,
                    champ_pr,
                )
        else:
            rest.append(list(comp))
    for prog in candidates[cap:]:
        key = tuple(prog)
        if key in seen:
            continue
        seen.add(key)
        rest.append(list(prog))
    result = front + rest
    if pool_key is not None:
        _growth_compress_pool_cache[pool_key] = tuple(tuple(int(w) for w in prog) for prog in result)
        if len(_growth_compress_pool_cache) > _GROWTH_COMPRESS_CACHE_MAX:
            for stale in list(_growth_compress_pool_cache.keys())[:-12]:
                if stale != pool_key:
                    _growth_compress_pool_cache.pop(stale, None)
    return result


def prime_scan_floor_growth_pool_cache(
    champion: list[int],
    existing: object | None,
) -> tuple[tuple[float, int], tuple[tuple[int, ...], ...], frozenset[tuple[int, ...]]] | None:
    """Retain in-memory or disk growth pool across cycles on the same plateau."""
    if not champion:
        return None
    plateau_key = _eq_refine_plateau_key(list(champion))
    if existing is not None and isinstance(existing, tuple) and len(existing) == 3 and existing[0] == plateau_key:
        return existing
    return load_scan_floor_growth_pool_cache(list(champion))


def select_growth_mg_anchor_fill(
    champion: list[int],
    fills: list[list[int]],
    *,
    cycle_no: int = 0,
    generation: int | None = None,
    inject_generation: int | None = None,
    rank_offset: int = 0,
) -> list[int]:
    """Pick growth MG anchor; on inject gen prefer tail-divergent material fills."""
    import os

    if not fills:
        return list(champion)
    if len(fills) == 1:
        return list(fills[0])
    if generation is not None and inject_generation is not None and int(generation) == int(inject_generation):
        anchor = scan_floor_anchor_program(list(champion))
        champ = list(champion)
        lcp = growth_anchor_lcp_prefix(champ, anchor)
        scored: list[tuple[tuple[int, int, int], list[int]]] = []
        for fill in fills:
            prog = list(fill)
            tail_diff = sum(
                1 for i in range(min(lcp, len(prog)), min(len(prog), len(champ))) if int(prog[i]) != int(champ[i])
            )
            ext = max(0, len(prog) - len(champ))
            scored.append(((tail_diff, ext, len(prog)), prog))
        scored.sort(key=lambda item: item[0], reverse=True)
        top_k = max(
            1,
            min(
                len(scored),
                int(os.environ.get("EE_GROWTH_MG_ANCHOR_TAIL_TOPK", "8")),
            ),
        )
        pool = [prog for _, prog in scored[:top_k]]
        idx = (int(cycle_no) + int(rank_offset)) % len(pool)
        return list(pool[idx])
    idx = (int(cycle_no) + int(rank_offset)) % len(fills)
    return list(fills[idx])


def growth_anchor_lcp_prefix(champion: list[int], anchor: list[int]) -> int:
    """Shared prefix length between champion and a longer growth near-miss.

    Returns the true first-divergence index (not inflated). Inflating to
    ``len(champion)-1`` broke window grafts and growth-tail MG when material
    diverged before the champion tail (e.g. 40w champ / 44w fill at index 26).
    """
    if not champion or not anchor:
        return 0
    n = 0
    for a, b in zip(champion, anchor):
        if int(a) != int(b):
            break
        n += 1
    return max(1, n)


def expand_growth_anchor_prefix_counts_stalled(
    champion: list[int],
    anchor: list[int],
    *,
    cycle_offset: int = 0,
    max_counts: int = 3,
    deltas: tuple[int, ...] = (-2, -1, 0, 1, 2),
    generation: int | None = None,
    inject_generation: int | None = None,
) -> list[int]:
    """Rotate LCP±δ prefix locks for stalled growth-mode model-guided completion."""
    import os

    lcp = growth_anchor_lcp_prefix(list(champion), list(anchor))
    champ_n = len(champion)
    variants: list[int] = []
    if generation is not None and inject_generation is not None and int(generation) == int(inject_generation):
        tail_words = max(
            8,
            int(os.environ.get("EE_SCAN_FLOOR_GROWTH_INJECT_MG_TAIL_WORDS", "18")),
        )
        wide_pc = max(1, min(champ_n - 1, champ_n - tail_words))
        for extra in (wide_pc, wide_pc - 2, wide_pc + 2, wide_pc - 4):
            pc = max(1, min(champ_n - 1, int(extra)))
            if pc not in variants:
                variants.append(pc)
    for delta in deltas:
        pc = max(1, min(champ_n - 1, lcp + int(delta)))
        if pc not in variants:
            variants.append(pc)
    if not variants:
        return [max(1, champ_n - 1)]
    start = int(cycle_offset) % len(variants)
    cap = max(1, int(max_counts))
    out: list[int] = []
    for i in range(min(cap, len(variants))):
        pc = variants[(start + i) % len(variants)]
        if pc not in out:
            out.append(pc)
    return out


def refine_eq_seed_two_step_perturb_variants(
    seeds: list[list[int]],
    rng,
    *,
    max_evals: int = 4,
    gate=None,
    champion: list[int] | None = None,
) -> list[list[int]]:
    """Chain two immediate nudges from one-step neighbors (gate unchanged)."""
    if not seeds or int(max_evals) <= 0:
        return []
    first = refine_eq_seed_one_step_perturb_variants(
        seeds,
        rng,
        max_evals=max(1, int(max_evals)) * 2,
        gate=gate,
        champion=champion,
    )
    if not first:
        return []
    return refine_eq_seed_one_step_perturb_variants(
        first,
        rng,
        max_evals=int(max_evals),
        gate=gate,
        champion=champion,
    )


def refine_eq_seed_one_step_perturb_variants(
    seeds: list[list[int]],
    rng,
    *,
    max_evals: int = 8,
    gate=None,
    champion: list[int] | None = None,
) -> list[list[int]]:
    """Single immediate nudge from EQ-refine seeds (multi-step search without gate weaken)."""
    from mutation_proposer import DirectedMutator

    if not seeds or int(max_evals) <= 0:
        return []
    from evolution.terminal_stagnation import get_scan_floor_gate

    gate = gate or get_scan_floor_gate(list(seeds[0]))
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    budget = max(1, int(max_evals))
    anchor = list(champion or seeds[0])
    if champion and _curriculum_l10_cached(list(champion)) < 0.85 - 1e-6:
        graft_budget = min(budget, max(4, budget // 2))
        cross_budget = min(graft_budget, max(2, graft_budget // 3))
        for cross in crossbreed_l10_prom_eq_refine_variants(
            list(champion),
            [list(s) for s in seeds],
            gate=gate,
            max_variants=cross_budget,
        ):
            key = tuple(int(x) for x in cross)
            if key in seen:
                continue
            seen.add(key)
            out.append(list(cross))
        import os

        donor_budget = max(2, int(os.environ.get("EE_L10_DONOR_GRAFT_VARIANTS", "4")))
        donor_limit = graft_budget + donor_budget
        for seed in seeds[:4]:
            for graft in expand_l10_lift_eq_donor_grafts(
                list(champion),
                list(seed),
                gate=gate,
                max_variants=donor_budget,
            ):
                key = tuple(int(x) for x in graft)
                if key in seen:
                    continue
                seen.add(key)
                out.append(list(graft))
            if len(out) >= donor_limit:
                break
        for seed in seeds[:4]:
            for graft in expand_l10_lift_eq_champion_grafts(
                list(champion),
                list(seed),
                gate=gate,
                max_variants=graft_budget,
            ):
                key = tuple(int(x) for x in graft)
                if key in seen:
                    continue
                seen.add(key)
                out.append(list(graft))
                if len(out) >= graft_budget:
                    break
            if len(out) >= graft_budget:
                break
    per_seed = max(1, (budget - len(out)) // max(1, min(len(seeds), 8)))
    for seed in seeds[:8]:
        champ_n = len(seed)
        for _ in range(per_seed * 2):
            if len(out) >= budget:
                break
            try:
                variant, _ = DirectedMutator.immediate_perturbation(list(seed), rng)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if len(variant) != champ_n:
                continue
            key = tuple(variant)
            if key in seen:
                continue
            seen.add(key)
            if gate.qualifies(variant):
                out.append(list(variant))
    if out:
        out.sort(
            key=lambda prog: scan_floor_eq_inject_sort_key(anchor, list(prog)),
            reverse=True,
        )
    return out[:budget]


def scan_floor_model_guided_inference_params(
    champion_hex: list[str],
    archive_parents_hex: list[list[str]],
    *,
    cycle_offset: int,
    generation: int,
    tail_budget: int = 1,
    growth_mode: bool = False,
    growth_anchor_hex: list[str] | None = None,
) -> tuple[list[str], list[int], str]:
    """Lock almost all of champion/archive; model completes only the tail at scan floor."""
    if not champion_hex:
        return [], [4], "empty-champion"
    champ_n = len(champion_hex)
    if growth_mode:
        import os

        extra = max(1, int(os.environ.get("EE_SCAN_FLOOR_MODEL_GUIDED_TAIL_BUDGET", "12")))
        champ_words = [int(h, 16) for h in champion_hex]
        if growth_anchor_hex and len(growth_anchor_hex) > champ_n:
            anchor_words = [int(h, 16) for h in growth_anchor_hex]
            prefix_count = growth_anchor_lcp_prefix(champ_words, anchor_words)
            return (
                growth_anchor_hex,
                [prefix_count],
                (f"growth-anchor-tail({len(growth_anchor_hex)}w→≤{champ_n + extra}w," f"pc={prefix_count})"),
            )
        prefix_count = max(1, champ_n - 1)
        return (
            champion_hex,
            [prefix_count],
            f"champion-tail-growth({champ_n}w→≤{champ_n + extra}w,pc={prefix_count})",
        )
    import os

    tail_budget_i = max(1, int(tail_budget))
    ref_lock = int(os.environ.get("EE_TERMINAL_MODEL_GUIDED_MIN_LOCK", "56"))
    compact_tail = max(
        tail_budget_i,
        int(os.environ.get("EE_SCAN_FLOOR_MODEL_GUIDED_TAIL_BUDGET", "8")),
    )
    if champ_n < ref_lock:
        tail_opts = [6, 8, 10, 12, 14, 16]
        rotated = tail_opts[(int(cycle_offset) + int(generation)) % len(tail_opts)]
        tail = min(max(compact_tail, rotated), champ_n - 2)
        prefix_count = max(1, champ_n - tail)
    else:
        prefix_count = max(ref_lock, champ_n - tail_budget_i)
    prefix_count = min(prefix_count, champ_n - 1)
    parents = [p for p in archive_parents_hex if p and len(p) == champ_n]
    slot = (int(cycle_offset) + int(generation)) % 3
    if parents and slot == 1:
        parent_hex = parents[(int(cycle_offset) // 3) % len(parents)]
        return (
            parent_hex,
            [prefix_count],
            f"archive-tail-complete({champ_n}w,pc={prefix_count})",
        )
    if parents and slot == 2:
        parent_hex = parents[((int(cycle_offset) + 1) // 3) % len(parents)]
        blend_n = max(prefix_count, champ_n - int(tail_budget))
        half = blend_n // 2
        blended = list(champion_hex[:half]) + list(parent_hex[half:blend_n])
        return (
            blended,
            [min(prefix_count, len(blended))],
            f"champion-archive-blend({len(blended)}w,pc={prefix_count})",
        )
    return (
        champion_hex,
        [prefix_count],
        f"champion-tail-complete({champ_n}w,pc={prefix_count})",
    )


def _evolution_project_root() -> str:
    import os

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scan_floor_plateau_armed() -> tuple[bool, int]:
    """Return (armed, floor_words) from progress controller scan-floor state."""
    from evolution.terminal_stagnation import _read_progress_state

    state = _read_progress_state()
    floor_w = int(state.get("terminal_efficiency_scan_floor_words", 0))
    floor_h = float(state.get("terminal_efficiency_scan_floor_heldout", 0.0))
    armed = floor_w > 0 and floor_h >= 0.99 - 1e-6
    return armed, floor_w


def recent_scan_floor_growth_challenger_fills(
    champion: list[int],
    *,
    max_fills: int = 4,
    lookback_cycles: int | None = None,
    word_margin: int | None = None,
) -> list[list[int]]:
    """Near-length champions from recent cycles for scan-floor crossover breeding."""
    import json
    import os

    armed, floor_w = scan_floor_plateau_armed()
    if not armed or not champion:
        return []
    anchor = scan_floor_anchor_program(list(champion))
    if not anchor:
        return []
    anchor_n = len(anchor)
    if anchor_n != int(floor_w):
        return []
    if os.environ.get("EE_SCAN_FLOOR_CHALLENGER_FILLS", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return []

    lb = (
        lookback_cycles
        if lookback_cycles is not None
        else int(os.environ.get("EE_SCAN_FLOOR_CHALLENGER_FILL_CYCLES", "60"))
    )
    max_words = int(word_margin) if word_margin is not None else scan_floor_growth_crossover_max_words(anchor_n)
    results = os.path.join(_evolution_project_root(), "multi_cycle_results.jsonl")
    if not os.path.isfile(results):
        return []
    try:
        with open(results, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
    except OSError:
        return []

    champ_t = tuple(int(x) for x in anchor)
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    seen_cycles = 0
    for line in reversed(lines):
        if seen_cycles >= lb:
            break
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("final_level", row.get("level", 0))) < 30:
            continue
        seen_cycles += 1
        words = int(row.get("champion_words", row.get("words", 0)))
        if words <= anchor_n or words > max_words:
            continue
        prog_hex = row.get("champion_program")
        if not prog_hex:
            continue
        prog = [int(x, 16) for x in prog_hex]
        key = tuple(prog)
        if key in seen:
            continue
        if not scan_floor_growth_completion_qualifies(anchor, prog):
            continue
        seen.add(key)
        out.append(list(prog))
        if len(out) >= max_fills:
            break
    out.sort(key=lambda p: (len(p) - anchor_n, len(p)))
    return out


def collect_scan_floor_growth_crossover_fills(
    champion: list[int],
    *,
    max_fills: int = 16,
) -> list[list[int]]:
    """Merge ranked growth fills with recorded growth material for scan-floor breeding."""
    from evolution.champion_rollback import collect_growth_material_compress_parents, collect_growth_material_parents
    from evolution.terminal_stagnation import _scan_floor_probe_stall_mode, ranked_scan_floor_growth_fills

    anchor = scan_floor_anchor_program(list(champion))
    if not anchor:
        return []
    cap = max(1, int(max_fills))
    ranked = ranked_scan_floor_growth_fills(anchor)
    challengers = recent_scan_floor_growth_challenger_fills(anchor, max_fills=cap)
    material = collect_growth_material_parents(anchor, max_parents=cap)
    compress_material: list[list[int]] = []
    growth_challengers: list[list[int]] = []
    if _scan_floor_probe_stall_mode(len(anchor)):
        compress_material = collect_growth_material_compress_parents(anchor, max_parents=cap)
        from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

        try:
            anchor_prom = float(compact_l30_promotion_raw_fitness(list(anchor)))
        except (RuntimeError, TypeError, ValueError, OSError):
            anchor_prom = 0.0
        for prog in challengers:
            try:
                prog_prom = float(compact_l30_promotion_raw_fitness(list(prog)))
            except (RuntimeError, TypeError, ValueError, OSError):
                continue
            if prog_prom + 1e-6 < anchor_prom and len(prog) > len(anchor):
                growth_challengers.append(list(prog))
    champ_t = tuple(int(x) for x in anchor)
    seen: set[tuple[int, ...]] = {champ_t}
    merged: list[list[int]] = []
    if _scan_floor_probe_stall_mode(len(anchor)):
        from evolution.terminal_stagnation import get_scan_floor_gate

        gate = get_scan_floor_gate(list(anchor))
        for lift in collect_scan_floor_l10_lift_growth_candidates(list(anchor), gate=gate):
            key = tuple(int(x) for x in lift)
            if key in seen:
                continue
            seen.add(key)
            merged.append(list(lift))
    growth_keys = {tuple(int(x) for x in p) for p in growth_challengers}
    other_challengers = [p for p in challengers if tuple(int(x) for x in p) not in growth_keys]
    for prog in (
        list(growth_challengers) + list(compress_material) + list(ranked) + list(other_challengers) + list(material)
    ):
        if not prog:
            continue
        key = tuple(int(x) for x in prog)
        if key in seen:
            continue
        seen.add(key)
        merged.append(list(prog))
    anchor_n = len(anchor)
    if _scan_floor_probe_stall_mode(anchor_n):
        import os

        target_len = anchor_n + max(
            2,
            int(os.environ.get("EE_GROWTH_COMPRESS_TARGET_EXTRA_WORDS", "4")),
        )
        merged.sort(key=lambda p: (abs(len(p) - target_len), -len(p)))
    else:
        merged.sort(key=lambda p: (abs(len(p) - anchor_n), len(p)))
    return merged[:cap]


def scan_floor_anchor_program(current: list[int]) -> list[int]:
    """Use certified scan-floor bytes when the live champion shrank below the floor."""
    import os

    armed, floor_w = scan_floor_plateau_armed()
    if not armed or not current or len(current) == floor_w:
        return list(current)
    results = os.path.join(_evolution_project_root(), "multi_cycle_results.jsonl")
    from evolution.champion_rollback import find_best_certified_champion

    row = find_best_certified_champion(
        results,
        30,
        min_words=floor_w,
        max_words=floor_w,
    )
    if not row:
        return list(current)
    prog = row.get("champion_program")
    if not prog:
        return list(current)
    out = [int(x, 16) if isinstance(x, str) else int(x) for x in prog]
    return out if len(out) == floor_w else list(current)


def scan_floor_capability_growth_allowed(
    champ_words: int,
    project_root: str | None = None,
) -> bool:
    """True when scan floor is armed at this length or below (compression/growth hot)."""
    armed, floor_w = scan_floor_plateau_armed()
    if not armed:
        return False
    return int(champ_words) <= int(floor_w)


def scan_floor_breeding_active(
    champ_words: int,
    *,
    level: int,
    verify: float,
) -> bool:
    """True when scan-floor breeding should run (live champion may be shorter than floor)."""
    if int(level) < 30 or float(verify) < 0.99 - 1e-6:
        return False
    return scan_floor_capability_growth_allowed(int(champ_words))


def scan_floor_model_guided_slot_allowed(
    level: int,
    verify: float,
    champ_words: int,
    cycles_at_level: int,
    *,
    generation: int = 0,
    max_generations: int = 30,
    inject_generation: int | None = None,
    cycle_number: int = 0,
    project_root: str | None = None,
) -> bool:
    """Allow one capped model-guided slot at terminal scan floor after extended stall."""
    import os

    from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor

    explicit = os.environ.get("EE_SCAN_FLOOR_MODEL_GUIDED")
    if explicit == "0":
        return False
    root = project_root or _evolution_project_root()
    if explicit == "1":
        enabled = True
    else:
        from evolution.epi_momentum_governor import promotion_raw_capability_stalled, scan_floor_length_prom_stalled
        from evolution.terminal_stagnation import _read_progress_state

        enabled = promotion_raw_capability_stalled(root)
        if not enabled:
            state = _read_progress_state()
            floor_w = int(state.get("terminal_efficiency_scan_floor_words", 0))
            if floor_w > 0 and int(champ_words) == floor_w:
                enabled = scan_floor_length_prom_stalled(root, floor_w)
    if not enabled:
        return False
    if not terminal_efficiency_at_scan_floor(int(champ_words)):
        return False
    if int(level) < 29 or float(verify) < 0.99 - 1e-6:
        return False
    if int(cycles_at_level) < 6:
        return False
    default_cadence = "1" if enabled else "2"
    cadence = max(
        1,
        int(os.environ.get("EE_SCAN_FLOOR_MODEL_GUIDED_CYCLE_CADENCE", default_cadence)),
    )
    if cadence > 1 and int(cycle_number) % cadence != 0:
        return False
    return scan_floor_model_guided_injection_generation(
        int(generation),
        int(max_generations),
        inject_generation=inject_generation,
    )


def get_model_guided_ratio(config, champion_level: int) -> float:
    """Determine model-guided replacement ratio based on level and stall."""
    ratio = config.model_guided_ratio
    if ratio is not None and ratio > 0:
        return float(ratio)
    if champion_level >= 17:
        return max(config.model_guided_ratio_high, 0.35)
    if champion_level <= 5:
        return config.model_guided_ratio_low
    elif champion_level <= 7:
        return config.model_guided_ratio_mid
    else:
        return config.model_guided_ratio_high


def get_model_guided_prefix(
    champion_hex: list[str],
    ref_prefix_hex: list[str] | None,
    champion_level: int,
    stalled: int,
) -> tuple[list[str], str]:
    """Select prefix source using Adaptive Prefix Scheduler (APS)."""
    from evolution.register_repair import reference_hex_prefix

    composite_hex = None
    source_name = None
    if champion_level >= 17:
        ref_from_level = reference_hex_prefix(champion_level, max_words=28)
        if ref_from_level:
            composite_hex = ref_from_level
            source_name = f"register-repair-ref(L{champion_level}): {len(composite_hex)}w"
            return composite_hex, source_name
    if champion_level >= 8 and len(champion_hex) >= 4:
        if stalled >= 50:
            composite = list(champion_hex)
            if ref_prefix_hex and len(ref_prefix_hex) >= 2:
                composite.extend(ref_prefix_hex[: min(12, len(ref_prefix_hex))])
            composite_hex = composite
            source_name = (
                f"composite(L{champion_level}): "
                f"champ({len(composite)}w)+ref({len(ref_prefix_hex or [])}w) "
                f"[stall={stalled}]"
            )
        elif stalled >= 20:
            if ref_prefix_hex and len(ref_prefix_hex) >= 4:
                composite_hex = ref_prefix_hex[: min(12, len(ref_prefix_hex))]
                source_name = f"ref-only(L{champion_level}): {len(composite_hex)}w [stall={stalled}]"
            else:
                prefix_len = min(8, len(champion_hex))
                composite_hex = champion_hex[:prefix_len]
                source_name = f"champion-fallback(L{champion_level}): {prefix_len}w [stall={stalled}]"
        else:
            if ref_prefix_hex and len(ref_prefix_hex) >= 2:
                prefix_len = min(6, max(2, len(ref_prefix_hex) // 3))
                composite_hex = ref_prefix_hex[:prefix_len]
                source_name = f"ref-short(L{champion_level}): {len(composite_hex)}w [stall={stalled}]"
            else:
                prefix_len = min(4, len(champion_hex))
                composite_hex = champion_hex[:prefix_len]
                source_name = f"champion-short(L{champion_level}): {prefix_len}w [stall={stalled}]"
    if composite_hex is not None:
        return composite_hex, source_name or f"default (L{champion_level})"
    if ref_prefix_hex is not None:
        return ref_prefix_hex, source_name or f"reference (L{champion_level})"
    return champion_hex, source_name or f"champion body (L{champion_level})"


def compute_model_temperature(config, current_generation: int) -> float:
    """Compute model temperature with optional annealing."""
    gen = current_generation
    if config.use_temperature_annealing:
        anneal_progress = min(1.0, gen / max(1, config.temperature_schedule_generations))
        return config.temperature_start + (config.temperature_end - config.temperature_start) * anneal_progress
    if gen < 10:
        return 1.2
    elif gen < 30:
        return 0.9
    return 0.6


# ------------------------------------------------------------------
# Engine-dependent strategy helpers
# ------------------------------------------------------------------


def check_variance_override(engine):
    """Check variance-aware strategy override (priority 1)."""
    from mutation_proposer import MutationStrategy

    try:
        pf = engine._last_challenger_fitnesses
        if not pf or len(pf) < 3:
            return 0, None, ""
        n = len(pf)
        m = sum(pf) / n
        if abs(m) <= 1e-9:
            return 0, None, ""
        v = sum((f - m) ** 2 for f in pf) / n
        s = v**0.5
        cv = s / abs(m)
        sf = sorted(pf)
        iqr = sf[3 * n // 4] - sf[n // 4]
        fr = max(pf) - min(pf) + 1e-9
        skew = sum(((f - m) / max(s, 1e-9)) ** 3 for f in pf) / n
        vs = 0.5 * min(cv, 2.0) / 2.0 + 0.5 * iqr / fr
        if not hasattr(engine, "_variance_history"):
            engine._variance_history = []
        engine._variance_history.append(vs)
        if len(engine._variance_history) > 10:
            engine._variance_history = engine._variance_history[-10:]
        vt = 0.0
        if len(engine._variance_history) >= 4:
            h = len(engine._variance_history) // 2
            vt = sum(engine._variance_history[h:]) / max(1, len(engine._variance_history) - h) - sum(
                engine._variance_history[:h]
            ) / max(1, h)
        if vs > 0.5:
            ep = min(0.85, 0.4 + vs * 0.5 + (0.10 if vt > 0.05 else 0))
            if engine.rng.random() < ep:
                es = [s for s in MutationStrategy if s.value in ("RANDOM", "TASK_AWARE", "DIRECTED")]
                if es:
                    return (
                        1,
                        engine.rng.choice(es),
                        f"[Strategy] Variance exploit: vs={vs:.2f}",
                    )
        elif vs < 0.3:
            ep = min(0.85, 0.4 + (1.0 - vs) * 0.5 + (0.10 if vt < -0.05 else 0))
            if engine.rng.random() < ep:
                es = [
                    s for s in MutationStrategy if s.value in ("GROW", "PERTURBATION_CAPPED", "MODEL_GUIDED", "HYBRID")
                ]
                if es:
                    return (
                        1,
                        engine.rng.choice(es),
                        f"[Strategy] Variance explore: vs={vs:.2f}",
                    )
        elif skew < -0.5 and engine.rng.random() < 0.35:
            es = [s for s in MutationStrategy if s.value in ("GROW", "PERTURBATION_CAPPED", "MODEL_GUIDED", "HYBRID")]
            if es:
                return (
                    1,
                    engine.rng.choice(es),
                    f"[Strategy] Skew explore: skew={skew:+.2f}",
                )
        elif skew > 0.5 and engine.rng.random() < 0.35:
            es = [s for s in MutationStrategy if s.value in ("RANDOM", "TASK_AWARE", "DIRECTED")]
            if es:
                return (
                    1,
                    engine.rng.choice(es),
                    f"[Strategy] Skew exploit: skew={skew:+.2f}",
                )
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return 0, None, ""


def check_priority_overrides(engine, current_priority: int) -> tuple:
    """Check cross-cycle, mutation stats, and plateau overrides (priorities 2-4)."""
    from mutation_proposer import MutationStrategy

    override_priority = current_priority
    override_strategy = None
    override_log_msg = ""
    try:
        if hasattr(engine, "meta_controller") and engine.meta_controller is not None:
            _trends = engine.meta_controller.get_cross_cycle_trends()
            _fitness_trend = _trends.get("fitness", {})
            if _fitness_trend.get("direction") == "declining" and engine.rng.random() < 0.25 and override_priority < 2:
                _explore = [
                    s for s in MutationStrategy if s not in (MutationStrategy.GROW, MutationStrategy.MODEL_GUIDED)
                ]
                if _explore:
                    override_priority = 2
                    override_strategy = engine.rng.choice(_explore)
                    override_log_msg = (
                        f"[Strategy] Trend override: fitness declining, " f"slope={_fitness_trend.get('slope', 0):.4f}"
                    )
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    try:
        from mutation_proposer import DirectedMutator as _DM

        _mstats = _DM.get_mutation_stats()
        if (
            _mstats["total_attempts"] > 50
            and _mstats["recent_avg_delta"] < -0.5
            and engine.rng.random() < 0.3
            and override_priority < 3
        ):
            _exploration_strats = [
                s for s in MutationStrategy if s not in (MutationStrategy.GROW, MutationStrategy.MODEL_GUIDED)
            ]
            if _exploration_strats:
                override_priority = 3
                override_strategy = engine.rng.choice(_exploration_strats)
                override_log_msg = (
                    f"[Strategy] Mutation stats override: " f"recent_avg_delta={_mstats['recent_avg_delta']:.3f}"
                )
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    try:
        if hasattr(engine, "progression_tracker") and engine.progression_tracker is not None:
            _pa = engine.progression_tracker.get_plateau_analysis()
            if (
                _pa.get("status") == "plateau"
                and _pa.get("plateau_duration", 0) > 10
                and engine.rng.random() < 0.3
                and override_priority < 4
            ):
                _explore2 = [
                    s for s in MutationStrategy if s not in (MutationStrategy.GROW, MutationStrategy.MODEL_GUIDED)
                ]
                if _explore2:
                    override_priority = 4
                    override_strategy = engine.rng.choice(_explore2)
                    override_log_msg = f"[Strategy] Plateau override: " f"duration={_pa.get('plateau_duration', 0)}"
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return override_priority, override_strategy, override_log_msg


def apply_cycle_phase_bias(
    engine,
    chosen,
    override_priority: int,
):
    """Apply cycle-phase exploration-exploitation balance."""
    from mutation_proposer import MutationStrategy

    if engine.champion_level in engine._unsolved_levels_set:
        return chosen
    try:
        if engine.current_generation > 0 and override_priority < 3:
            _cycle_progress = engine.current_generation / max(engine.config.max_generations, 1)
            if _cycle_progress <= 0.3:
                _explore_pool = [
                    s for s in MutationStrategy if s.value in ("GROW", "PERTURBATION_CAPPED", "MODEL_GUIDED", "HYBRID")
                ]
                if _explore_pool and engine.rng.random() < 0.40:
                    chosen = engine.rng.choice(_explore_pool)
                    engine._log(
                        f"[Strategy] Cycle-phase early explore: "
                        f"progress={_cycle_progress:.2f}, chosen={chosen.value}"
                    )
            elif _cycle_progress >= 0.7:
                _exploit_pool = [s for s in MutationStrategy if s.value in ("RANDOM", "TASK_AWARE", "DIRECTED")]
                if _exploit_pool and engine.rng.random() < 0.40:
                    chosen = engine.rng.choice(_exploit_pool)
                    engine._log(
                        f"[Strategy] Cycle-phase late exploit: "
                        f"progress={_cycle_progress:.2f}, chosen={chosen.value}"
                    )
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return chosen


def apply_affinity_bias(engine, chosen):
    """Apply per-program mutation affinity bias."""
    try:
        _champ = getattr(engine, "champion", None)
        if _champ is None or len(_champ) == 0:
            return chosen
        from mutation_proposer import DirectedMutator as _AffDM
        from mutation_proposer import MutationStrategy

        _affinity = _AffDM.get_program_mutation_affinity(_champ, top_n=5)
        if not _affinity:
            return chosen
        _sv = {s.value for s in MutationStrategy}
        # Import the strategy enum cache
        try:
            from evolution_engine import _STRATEGY_ENUM_CACHE
        except ImportError:
            _STRATEGY_ENUM_CACHE = {s.value: s for s in MutationStrategy}
        _as = [
            (
                _STRATEGY_ENUM_CACHE.get(a, MutationStrategy.HYBRID) if a in _sv else MutationStrategy[a],
                r,
                at,
            )
            for a, r, at in _affinity
            if a in _sv or hasattr(MutationStrategy, a)
        ]
        if not _as:
            return chosen
        _ta = sum(a for _, _, a in _as)
        if _ta < 20:
            return chosen
        _tr, _rr = _as[0][1], _as[1][1] if len(_as) > 1 else 0.0
        if _tr - _rr >= 0.15 and engine.rng.random() < 0.6:
            return _as[0][0]
        if engine.rng.random() < 0.4:
            _w = [r for _, r, _ in _as]
            if sum(_w) > 0:
                _p = engine.rng.choices([s for s, _, _ in _as], weights=_w, k=1)
                if _p:
                    return _p[0]
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return chosen


def apply_level_aware_strategy_bias(engine, chosen):
    """Apply level-aware strategy weighting to bias selection by task difficulty.

    Different levels benefit from different mutation strategies:
      L1-L4:  Simple programs — small changes (RANDOM, TASK_AWARE, DIRECTED)
      L5-L6:  Structural mutations needed (GROW, PERTURBATION_CAPPED)
      L7-L11: Complex operations (GROW, PERTURBATION_CAPPED, HYBRID)
      L12-L16: Reference-guided repair (TASK_AWARE, DIRECTED, MODEL_GUIDED)
      L17+:   Register-targeted repair (TASK_AWARE, DIRECTED dominate)
    """
    from mutation_proposer import MutationStrategy

    try:
        lvl = engine.champion_level
        if lvl <= 4:
            tier_key = "L1_L4"
        elif lvl <= 6:
            tier_key = "L5_L6"
        elif lvl <= 11:
            tier_key = "L7_L11"
        elif lvl >= 17:
            tier_key = "L17_plus"
        else:
            tier_key = "L12_plus"

        _lvl_weights = engine._LEVEL_WEIGHT_PROFILES[tier_key]
        _lvl_prob = engine._LEVEL_OVERRIDE_PROBS[tier_key]

        if engine.rng.random() < _lvl_prob:
            _candidates = [s for s in MutationStrategy if s.value in _lvl_weights]
            if _candidates:
                _weighted: list = []
                for _s in _candidates:
                    _w = _lvl_weights.get(_s.value, 1.0)
                    _count = max(1, round(_w * 4))
                    _weighted.extend([_s] * _count)
                chosen = engine.rng.choice(_weighted)
                engine._log(f"[Strategy] Level-aware bias: L{lvl}, " f"prob={_lvl_prob:.2f}, chosen={chosen.value}")
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    try:
        if int(engine.champion_level) >= 29:
            from evolution.champion_bloat_guard import level_word_cap
            from mutation_proposer import MutationStrategy as _MS

            cap = level_word_cap(int(engine.champion_level))
            n = len(getattr(engine, "champion", None) or [])
            if cap and n <= max(12, int(cap) - 16):
                # Compact verify-saturated L30: search rewrites, not deletes.
                ta_p = 0.70 if n <= 20 else 0.40
                if engine.rng.random() < ta_p:
                    chosen = _MS.TASK_AWARE
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return chosen
