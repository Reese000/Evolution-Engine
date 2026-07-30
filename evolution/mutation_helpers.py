"""Mutation strategy helpers extracted from EvolutionEngine.

Functions for strategy selection, model-guided generation, and mutation
biasing. Each function that depended on ``self`` receives the engine
instance as the first parameter.
"""

from __future__ import annotations

from typing import Any


# ------------------------------------------------------------------
# Pure / near-pure helpers (no or minimal engine state)
# ------------------------------------------------------------------


def get_model_guided_ratio(config, champion_level: int) -> float:
    """Determine model-guided replacement ratio based on level and stall.

    Returns fraction of challengers to replace with model completions.
    When model_guided_ratio is None (or 0), uses curriculum-adaptive
    defaults (low/mid/high) based on champion level.
    """
    ratio = config.model_guided_ratio
    if ratio is not None and ratio > 0:
        return float(ratio)
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
    composite_hex = None
    source_name = None
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
        anneal_progress = min(
            1.0, gen / max(1, config.temperature_schedule_generations)
        )
        return config.temperature_start + (
            config.temperature_end - config.temperature_start
        ) * anneal_progress
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
            vt = (
                sum(engine._variance_history[h:])
                / max(1, len(engine._variance_history) - h)
                - sum(engine._variance_history[:h]) / max(1, h)
            )
        if vs > 0.5:
            ep = min(0.85, 0.4 + vs * 0.5 + (0.10 if vt > 0.05 else 0))
            if engine.rng.random() < ep:
                es = [
                    s
                    for s in MutationStrategy
                    if s.value in ("RANDOM", "TASK_AWARE", "DIRECTED")
                ]
                if es:
                    return (
                        1,
                        engine.rng.choice(es),
                        f"[Strategy] Variance exploit: vs={vs:.2f}",
                    )
        elif vs < 0.3:
            ep = min(
                0.85, 0.4 + (1.0 - vs) * 0.5 + (0.10 if vt < -0.05 else 0)
            )
            if engine.rng.random() < ep:
                es = [
                    s
                    for s in MutationStrategy
                    if s.value
                    in ("GROW", "PERTURBATION_CAPPED", "MODEL_GUIDED", "HYBRID")
                ]
                if es:
                    return (
                        1,
                        engine.rng.choice(es),
                        f"[Strategy] Variance explore: vs={vs:.2f}",
                    )
        elif skew < -0.5 and engine.rng.random() < 0.35:
            es = [
                s
                for s in MutationStrategy
                if s.value
                in ("GROW", "PERTURBATION_CAPPED", "MODEL_GUIDED", "HYBRID")
            ]
            if es:
                return (
                    1,
                    engine.rng.choice(es),
                    f"[Strategy] Skew explore: skew={skew:+.2f}",
                )
        elif skew > 0.5 and engine.rng.random() < 0.35:
            es = [
                s
                for s in MutationStrategy
                if s.value in ("RANDOM", "TASK_AWARE", "DIRECTED")
            ]
            if es:
                return (
                    1,
                    engine.rng.choice(es),
                    f"[Strategy] Skew exploit: skew={skew:+.2f}",
                )
    except Exception:
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
            if (
                _fitness_trend.get("direction") == "declining"
                and engine.rng.random() < 0.25
                and override_priority < 2
            ):
                _explore = [
                    s
                    for s in MutationStrategy
                    if s not in (MutationStrategy.GROW, MutationStrategy.MODEL_GUIDED)
                ]
                if _explore:
                    override_priority = 2
                    override_strategy = engine.rng.choice(_explore)
                    override_log_msg = (
                        f"[Strategy] Trend override: fitness declining, "
                        f"slope={_fitness_trend.get('slope', 0):.4f}"
                    )
    except Exception:
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
                s
                for s in MutationStrategy
                if s not in (MutationStrategy.GROW, MutationStrategy.MODEL_GUIDED)
            ]
            if _exploration_strats:
                override_priority = 3
                override_strategy = engine.rng.choice(_exploration_strats)
                override_log_msg = (
                    f"[Strategy] Mutation stats override: "
                    f"recent_avg_delta={_mstats['recent_avg_delta']:.3f}"
                )
    except Exception:
        pass
    try:
        if (
            hasattr(engine, "progression_tracker")
            and engine.progression_tracker is not None
        ):
            _pa = engine.progression_tracker.get_plateau_analysis()
            if (
                _pa.get("status") == "plateau"
                and _pa.get("plateau_duration", 0) > 10
                and engine.rng.random() < 0.3
                and override_priority < 4
            ):
                _explore2 = [
                    s
                    for s in MutationStrategy
                    if s not in (MutationStrategy.GROW, MutationStrategy.MODEL_GUIDED)
                ]
                if _explore2:
                    override_priority = 4
                    override_strategy = engine.rng.choice(_explore2)
                    override_log_msg = (
                        f"[Strategy] Plateau override: "
                        f"duration={_pa.get('plateau_duration', 0)}"
                    )
    except Exception:
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
            _cycle_progress = engine.current_generation / max(
                engine.config.max_generations, 1
            )
            if _cycle_progress <= 0.3:
                _explore_pool = [
                    s
                    for s in MutationStrategy
                    if s.value
                    in ("GROW", "PERTURBATION_CAPPED", "MODEL_GUIDED", "HYBRID")
                ]
                if _explore_pool and engine.rng.random() < 0.40:
                    chosen = engine.rng.choice(_explore_pool)
                    engine._log(
                        f"[Strategy] Cycle-phase early explore: "
                        f"progress={_cycle_progress:.2f}, chosen={chosen.value}"
                    )
            elif _cycle_progress >= 0.7:
                _exploit_pool = [
                    s
                    for s in MutationStrategy
                    if s.value in ("RANDOM", "TASK_AWARE", "DIRECTED")
                ]
                if _exploit_pool and engine.rng.random() < 0.40:
                    chosen = engine.rng.choice(_exploit_pool)
                    engine._log(
                        f"[Strategy] Cycle-phase late exploit: "
                        f"progress={_cycle_progress:.2f}, chosen={chosen.value}"
                    )
    except Exception:
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
                _STRATEGY_ENUM_CACHE.get(a, MutationStrategy.HYBRID)
                if a in _sv
                else MutationStrategy[a],
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
                _p = engine.rng.choices(
                    [s for s, _, _ in _as], weights=_w, k=1
                )
                if _p:
                    return _p[0]
    except Exception:
        pass
    return chosen


def apply_level_aware_strategy_bias(engine, chosen):
    """Apply level-aware strategy weighting to bias selection by task difficulty.

    Different levels benefit from different mutation strategies:
      L1-L4:  Simple programs — small changes (RANDOM, TASK_AWARE, DIRECTED)
      L5-L6:  Structural mutations needed (GROW, PERTURBATION_CAPPED)
      L7-L11: Complex operations (GROW, PERTURBATION_CAPPED, HYBRID)
      L12+:   Full program redesign (MODEL_GUIDED, GROW, HYBRID)
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
                    _count = max(1, int(round(_w * 4)))
                    _weighted.extend([_s] * _count)
                chosen = engine.rng.choice(_weighted)
                engine._log(
                    f"[Strategy] Level-aware bias: L{lvl}, "
                    f"prob={_lvl_prob:.2f}, chosen={chosen.value}"
                )
    except Exception:
        pass
    return chosen
