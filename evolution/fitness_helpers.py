"""Fitness computation helpers extracted from EvolutionEngine.

All functions that were previously methods on EvolutionEngine are now
standalone functions. Methods that depended on ``self`` receive the
engine instance as the first parameter (``engine``).

Typical usage from evolution_engine.py::

    from evolution.fitness_helpers import (
        score_register_match,
        compute_base_score,
        compute_time_credit,
        compute_efficiency_score,
        compute_structural_bonuses,
        compute_smc_bonus,
        compute_multi_task_score,
        log_bonus_diagnostics,
        extract_trace_info,
        compute_fitness,
        compute_all_bonuses,
        apply_bonuses_and_normalize,
        finalize_with_blending,
    )
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

# ------------------------------------------------------------------
# Pure scoring helpers (no engine state needed)
# ------------------------------------------------------------------


def score_register_match(
    expected_regs: dict,
    expected_total: int,
    final_regs: dict,
) -> float:
    """Score per-register match with distance-weighted partial credit."""
    if expected_total == 0:
        return 1.0
    total_score = 0.0
    get_actual = final_regs.get
    for reg_idx, expected_val in expected_regs.items():
        actual_val = get_actual(reg_idx, 0)
        if actual_val == expected_val:
            total_score += 1.0
            continue
        if expected_val == 0:
            total_score += 0.0
        else:
            error_ratio = abs(actual_val - expected_val) / max(1, abs(expected_val))
            total_score += 1.0 / (1.0 + error_ratio)
    return total_score / expected_total


def compute_base_score(
    reg_match: float,
    instr_count: int,
    status: str,
    budget: int,
) -> float:
    """Compute base fitness score from register match and execution metrics."""
    bloat_penalty = (instr_count - 50) * 0.05 if instr_count > 50 else 0.0
    exec_credit = (
        min(10.0, (instr_count / budget) * 10.0)
        if (status == "ok" and instr_count > 0 and reg_match > 0.0 and budget > 0)
        else 0.0
    )
    solved_bonus = 50.0 if reg_match >= 0.99 else 0.0
    return reg_match * 100.0 - bloat_penalty + exec_credit + solved_bonus


def compute_time_credit(
    status: str,
    reg_match: float,
    instr_count: int,
) -> float:
    """Compute time-to-solution credit (+20 points max)."""
    if status != "ok" or reg_match < 0.99 or instr_count <= 0:
        return 0.0
    if instr_count <= 5:
        return 20.0
    elif instr_count <= 15:
        return 20.0 * (1.0 - (instr_count - 5) / 10.0)
    else:
        return max(0.0, 10.0 - (instr_count - 15) * 0.2)


def compute_efficiency_score(
    program: list[int],
    instr_count: int,
    status: str,
    champion_level: int,
    eff_weight: float = 1.0,
) -> float:
    """Compute efficiency scoring (+15 points max)."""
    if eff_weight <= 0 or instr_count <= 0 or status != "ok":
        return 0.0
    nop_count = 0
    unique_vals = set()
    for word in program:
        unique_vals.add(word)
        if word == 0x00000013:
            nop_count += 1
    total_instr = len(program) if program else 1
    useful = max(1, total_instr - nop_count)
    unique_ratio = len(unique_vals) / total_instr if total_instr > 0 else 1.0
    if champion_level >= 8:
        if useful >= 8:
            instr_eff = 1.0
        elif useful >= 5:
            instr_eff = 0.7 + (useful - 5) * 0.1
        else:
            instr_eff = useful / 5.0
        opcode_bonus = unique_ratio * 0.5
    else:
        if useful <= 5:
            instr_eff = 1.0
        elif useful <= 10:
            instr_eff = 1.0 - (useful - 5) * 0.1
        else:
            instr_eff = max(0.0, 0.5 - (useful - 10) * 0.05)
        opcode_bonus = (1.0 - unique_ratio) * 0.5
    return (instr_eff + opcode_bonus) * 30.0 * eff_weight


def compute_smc_bonus(
    status: str,
    program: list[int],
    instr_count: int,
    champion_level: int,
) -> float:
    """Compute SMC pattern bonus (+25 points max) for L12+."""
    if status != "ok" or champion_level < 12 or instr_count <= 0:
        return 0.0
    smc_opcodes = {0x37, 0x13, 0x33, 0x23, 0x67, 0x0F}
    smc_count = sum(1 for w in program if (w & 0x7F) in smc_opcodes)
    return min(25.0, smc_count * 5.0) if smc_count >= 3 else 0.0


def compute_multi_task_score(
    secondary_results: list[tuple[dict, dict]] | None,
    mt_weight: float = 1.0,
) -> float:
    """Compute multi-task scoring (+15 points max)."""
    if mt_weight <= 0 or secondary_results is None or len(secondary_results) == 0:
        return 0.0
    task_count = 0
    task_total = 0.0
    for sec_task, sec_trace in secondary_results:
        sec_status = sec_trace.get("status", "")
        if sec_status in ("crash", "trap", "exception", "timeout"):
            continue
        task_count += 1
        sec_expected = sec_task.get("expected_output", {}).get("registers", {})
        sec_final = sec_trace.get("final_regs", {})
        sec_total = len(sec_expected)
        if sec_total == 0:
            task_total += 1.0
            continue
        sec_score = 0.0
        for reg_idx, exp_val in sec_expected.items():
            act_val = sec_final.get(reg_idx, 0)
            if act_val == exp_val:
                sec_score += 1.0
            else:
                diff = abs(act_val - exp_val)
                max_diff = max(abs(exp_val), 1)
                sec_score += 1.0 / (1.0 + diff / max_diff)
        task_total += sec_score / sec_total
    if task_count > 0:
        return (task_total / task_count) * 5.0 * min(task_count, 3) * mt_weight
    return 0.0


# ------------------------------------------------------------------
# Engine-dependent helpers (take ``engine`` as first parameter)
# ------------------------------------------------------------------


def compute_structural_bonuses(
    engine,
    status: str,
    instr_count: int,
    program: list[int],
    bonus_scale: float,
    opcode_cat_array: list[int],
) -> tuple[float, float, float]:
    """Compute loop, memory, and instruction diversity bonuses."""
    loop_bonus = 0.0
    if engine.champion_level >= 5 and status == "ok" and instr_count > 0:
        _gen = engine.current_generation
        if not hasattr(engine, "_backward_branch_cache") or engine._backward_branch_cache_gen != _gen:
            if not hasattr(engine, "_backward_branch_cache"):
                engine._backward_branch_cache = {}
                engine._backward_branch_cache_gen = -1
            engine._backward_branch_cache.clear()
            engine._backward_branch_cache_gen = _gen
        _prog_key = hash(tuple(program))
        n_loops = engine._backward_branch_cache.get(_prog_key)
        if n_loops is None:
            from evolution.diversity_helpers import count_backward_branches

            n_loops = count_backward_branches(program)
            engine._backward_branch_cache[_prog_key] = n_loops
        if n_loops > 0:
            loop_bonus = min(5.0, n_loops * 2.0)
    loop_bonus *= bonus_scale
    mem_bonus = 0.0
    if engine.champion_level >= 7 and status == "ok":
        mem_accesses = (
            getattr(engine, "_current_trace", {}).get("memory_accesses", [])
            if hasattr(engine, "_current_trace")
            else []
        )
        if mem_accesses:
            mem_bonus = min(3.0, len(mem_accesses) * 0.5)
    mem_bonus *= bonus_scale
    div_bonus = 0.0
    if engine.champion_level >= 5 and status == "ok" and instr_count > 0:
        seen_cats = set()
        for word in program:
            if isinstance(word, int):
                cat = word & 0x7F
                if 0 <= cat < 128:
                    seen_cats.add(opcode_cat_array[cat])
        n_cats = len(seen_cats - {-1})
        if n_cats >= 3:
            div_bonus = min(2.0, (n_cats - 2) * 0.5)
    div_bonus *= bonus_scale
    return loop_bonus, mem_bonus, div_bonus


def log_bonus_diagnostics(
    engine,
    base_score: float,
    bonuses: dict,
    max_single: float,
    norm_active: bool,
) -> None:
    """Log per-dimension bonus breakdown periodically."""
    _blog_int = getattr(engine.config, "bonus_log_interval", 50)
    if _blog_int <= 0 or engine.current_generation % _blog_int != 0:
        return
    _mbf = getattr(engine.config, "max_bonus_fraction", 1.0)
    _gen = engine.current_generation
    _names = list(bonuses.keys())
    _vals = list(bonuses.values())
    _raw_total = sum(_vals)
    _parts = [f"{_n}={_v:.2f}" for _n, _v in zip(_names, _vals)]
    if norm_active:
        _cap = _mbf * base_score
        _capped_names = [_n for _n, _v in zip(_names, _vals) if _v > _cap]
        _capped_str = ",".join(_capped_names) if _capped_names else "none"
        _norm_total = sum(min(_v, _cap) for _v in _vals)
        engine._log(
            f"[BonusDiag] gen={_gen} base={base_score:.2f} fraction={_mbf:.2f} "
            f"cap={_cap:.2f} capped=[{_capped_str}] raw_total={_raw_total:.2f} "
            f"norm_total={_norm_total:.2f} {' '.join(_parts)}"
        )
    else:
        engine._log(
            f"[BonusDiag] gen={_gen} base={base_score:.2f} norm=off " f"raw_total={_raw_total:.2f} {' '.join(_parts)}"
        )


def extract_trace_info(
    engine,
    trace: dict,
) -> tuple:
    """Extract and cache trace information for fitness computation."""
    task_inst = engine.task_instance
    if engine._cached_task_instance_id == engine.current_generation:
        expected_regs = engine._cached_expected_regs
        expected_total = engine._cached_expected_total
    else:
        expected = task_inst.get("expected_output", {})
        expected_regs = expected.get("registers", {})
        expected_total = len(expected_regs)
        engine._cached_expected_regs = expected_regs
        engine._cached_expected_total = expected_total
        engine._cached_task_instance_id = engine.current_generation
    return (
        expected_regs,
        expected_total,
        trace.get("status", ""),
        trace.get("final_regs", {}),
        trace.get("instructions_count", 0),
        trace.get("trap_cause", ""),
    )


def apply_novelty_qd_blending(
    engine,
    fitness,
    program,
    trace,
    base_score,
    _qd_fn,
    _prog_qd_desc,
):
    """Apply novelty-weighted fitness and QD archive bonus/penalty."""
    nw = engine._novelty_weight_cache
    if engine._current_diversity < 0.2 and nw > 0.0:
        nw = min(nw * (1.0 + (1.0 - engine._current_diversity / 0.2)), 0.8)
    ef = engine._extract_behavioral_descriptor
    if nw > 0.0 and engine.novelty_archive is not None:
        nv = 0.0
        if getattr(engine.novelty_archive, "size", 0):
            try:
                nv = engine.novelty_archive.decayed_novelty(program, engine.current_generation)
            except (KeyError, TypeError, ValueError):
                if ef is not None:
                    try:
                        d = ef(trace, program, budget=engine.config.eval_budget)
                        nv = engine.novelty_archive.novelty_score(d)
                    except (TypeError, ValueError, RuntimeError):
                        nv = 0.0
        fitness = fitness * (1.0 - nw) + nv * (base_score / 10.0 if base_score > 0 else 1.0) * nw
    if engine.config.qd_bonus_weight > 0.0 and engine.qd_archive is not None and _qd_fn is not None:
        try:
            if _prog_qd_desc is None:
                _prog_qd_desc = _qd_fn(program, trace)
            if _prog_qd_desc is not None:
                qb = engine.qd_archive.selection_bias(_prog_qd_desc)
                qw = engine.config.qd_bonus_weight
                st = engine.generations_stalled
                if st > 10:
                    qw *= min(2.5, 1.0 + 0.0375 * (st - 10))
                try:
                    ds = engine.qd_archive.get_illumination_diversity().get(
                        "overall_diversity_score", engine._qd_coverage
                    )
                except (KeyError, TypeError, ValueError):
                    ds = engine._qd_coverage
                if ds < 0.20:
                    qw *= 1.0 + 2.5 * (0.20 - ds)
                try:
                    fd = engine.qd_archive.get_cell_freshness(engine.current_generation)
                    fv = fd.get("avg_freshness", 0.5) if isinstance(fd, dict) else 0.5
                    qw *= 1.5 - fv
                except (AttributeError, TypeError):  # Intentional: QD freshness lookup may fail; fallback to default
                    pass
                fitness += qb * qw
                if engine.champion is not None and engine.champion_trace is not None:
                    cd = (
                        engine._cached_champion_qd_desc
                        if engine._champion_qd_valid
                        else _qd_fn(engine.champion, engine.champion_trace)
                    )
                    if cd is not None:
                        engine._cached_champion_qd_desc = cd
                        engine._champion_qd_valid = True
                        dist = abs(_prog_qd_desc[0] - cd[0]) + abs(_prog_qd_desc[1] - cd[1])
                        if dist < 2.0:
                            fitness *= 1.0 - 0.10 * (1.0 - dist / 2.0)
        except (TypeError, ValueError, RuntimeError):
            pass
    return fitness


def compute_all_bonuses(
    engine,
    status: str,
    instr_count: int,
    program: list[int],
    reg_match: float,
    base_score: float,
    opcode_cat_array: list[int],
) -> dict:
    """Compute all bonus components."""
    try:
        _bonus_scale = engine._get_structural_bonus_scale().get("scale_factor", 1.0)
    except (AttributeError, KeyError, TypeError):
        _bonus_scale = 1.0
    loop_bonus, mem_bonus, div_bonus = compute_structural_bonuses(
        engine, status, instr_count, program, _bonus_scale, opcode_cat_array
    )
    level_bonus = engine.champion_level * 25.0
    cached = engine._bonus_threshold_cache
    lvl = engine.champion_level
    from tasks_rv32i import LEVELS

    bonus_threshold = (
        cached[1] if cached and cached[0] == lvl else max(0.85, 0.95 - 0.10 * (lvl / max(1, len(LEVELS) - 1)))
    )
    graduated_bonus = (
        50.0 * (reg_match - bonus_threshold) / (1.0 - bonus_threshold) if reg_match >= bonus_threshold else 0.0
    )
    secondary_results = getattr(engine, "_secondary_task_results", None)
    mt_weight = getattr(engine.config, "secondary_task_weight", 1.0)
    eff_weight = getattr(engine.config, "efficiency_score_weight", 1.0)
    return {
        "loop": loop_bonus,
        "mem": mem_bonus,
        "div": div_bonus,
        "level": level_bonus,
        "efficiency": compute_efficiency_score(program, instr_count, status, engine.champion_level, eff_weight),
        "multi_task": compute_multi_task_score(secondary_results, mt_weight),
        "time_credit": compute_time_credit(status, reg_match, instr_count),
        "smc": compute_smc_bonus(status, program, instr_count, engine.champion_level),
        "graduated": graduated_bonus,
    }


def apply_bonuses_and_normalize(
    engine,
    base_score: float,
    bonuses: dict,
) -> float:
    """Apply bonuses with normalization and store raw score."""
    _base_for_norm = base_score if base_score > 0 else 1.0
    _max_single = getattr(engine.config, "max_bonus_fraction", 1.0) * _base_for_norm
    _norm_active = getattr(engine.config, "max_bonus_fraction", 1.0) < 1.0
    log_bonus_diagnostics(engine, base_score, bonuses, _max_single, _norm_active)
    base_score += sum(min(b, _max_single) for b in bonuses.values())
    engine._raw_task_score = base_score
    engine._task_fitness = base_score - min(bonuses["level"], _max_single)
    return base_score


@contextmanager
def terminal_certified_fitness_context(engine) -> Iterator[None]:
    """Disable archive blending that distorts terminal champion fitness comparisons.

    High terminal-diversity novelty/QD weights crush compute_fitness for certified
    champions, causing efficiency promote to pick bogus shorter programs and
    apply_l30_certified_metrics_if_degraded to revert real breakthroughs.
    """
    config = getattr(engine, "config", None)
    saved = {
        "novelty_weight": float(getattr(config, "novelty_weight", 0.0)) if config else 0.0,
        "novelty_weight_cache": getattr(engine, "_novelty_weight_cache", None),
        "qd_bonus_weight": float(getattr(config, "qd_bonus_weight", 0.0)) if config else 0.0,
        "cached_task_instance_id": getattr(engine, "_cached_task_instance_id", -1),
        "cached_expected_regs": getattr(engine, "_cached_expected_regs", {}),
        "cached_expected_total": getattr(engine, "_cached_expected_total", 0),
        "task_instance": getattr(engine, "task_instance", None),
    }
    if config is not None:
        engine.config.novelty_weight = 0.0
        engine.config.qd_bonus_weight = 0.0
    engine._novelty_weight_cache = 0.0
    engine._cached_task_instance_id = -1
    engine._cached_expected_regs = {}
    engine._cached_expected_total = 0
    try:
        yield
    finally:
        if config is not None:
            engine.config.novelty_weight = saved["novelty_weight"]
            engine.config.qd_bonus_weight = saved["qd_bonus_weight"]
        if saved["novelty_weight_cache"] is not None:
            engine._novelty_weight_cache = saved["novelty_weight_cache"]
        engine._cached_task_instance_id = saved["cached_task_instance_id"]
        engine._cached_expected_regs = saved["cached_expected_regs"]
        engine._cached_expected_total = saved["cached_expected_total"]
        engine.task_instance = saved["task_instance"]


def _terminal_length_efficiency_bonus(engine, program: list[int], base_fitness: float) -> float:
    """Reward shorter verified programs at certified L30 — unlocks fitness growth past plateau."""
    import os

    from evolution.terminal_stagnation import terminal_certified_plateau, terminal_verify_promote_active

    level = int(getattr(engine, "champion_level", 0))
    verify = float(getattr(engine, "_champion_verify_score", 0.0))
    fit = float(getattr(engine, "champion_fitness", 0.0))
    if not terminal_verify_promote_active() or not terminal_certified_plateau(level, verify, fit):
        return base_fitness
    baseline_words = int(os.environ.get("EE_TERMINAL_REF_WORDS", "73"))
    champ = getattr(engine, "champion", None)
    if champ:
        baseline_words = max(baseline_words, len(champ))
    delta_words = baseline_words - len(program)
    if delta_words <= 0:
        return base_fitness
    per_word = float(os.environ.get("EE_TERMINAL_LENGTH_BONUS", "0.05"))
    return base_fitness + delta_words * per_word


def selection_fitness(
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
        from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor

        # Reporting floor collapses equal-length scores at scan floor; rank on
        # promotion_raw so in-cycle challengers can advance on capability.
        if terminal_efficiency_at_scan_floor(len(program)):
            raw = promotion_raw_fitness(engine, program, trace, verify_score=verify)
            champ = getattr(engine, "champion", None)
            if champ and len(program) < len(champ):
                return _terminal_length_efficiency_bonus(engine, program, raw)
            return raw
        return reporting_champion_fitness(engine, program, trace, max(verify, champ_verify))
    return float(compute_fitness(engine, program, trace))


def promotion_raw_fitness(
    engine,
    program: list[int],
    trace: dict,
    *,
    verify_score: float | None = None,
) -> float:
    """Uncertified task fitness for saturated-verify promotion tie-breaks."""
    verify_fn = getattr(engine, "_verify_score_from_trace", None)
    verify = float(verify_score if verify_score is not None else (verify_fn(trace) if callable(verify_fn) else 0.0))
    with terminal_certified_fitness_context(engine):
        fitness = float(compute_fitness(engine, program, trace))
        if verify >= 0.99 - 1e-6:
            mult = partial_verify_multiplier(engine, trace)
            if mult < 1.0 - 1e-6:
                fitness /= mult
        return fitness


_CAPABILITY_SCORE_ENGINE = None
_PROMOTION_RAW_CACHE: dict[tuple[int, ...], float] = {}


def _isolate_capability_score_engine(engine) -> None:
    """Reset live-loop champion state so CPU scoring cannot inherit stale context."""
    engine.champion = None
    engine.champion_trace = None
    engine.champion_fitness = 0.0
    engine._champion_partial_verify = 1.0
    engine._champion_qd_valid = False
    engine._cached_champion_qd_desc = None
    engine._cached_task_instance_id = -1
    engine._cached_expected_regs = {}
    engine._cached_expected_total = 0


def decode_program_words(prog: Any) -> list[int] | None:
    """Decode hex/int program list from jsonl or carry-forward payloads."""
    if not isinstance(prog, list) or len(prog) < 12:
        return None
    words: list[int] = []
    for item in prog:
        if isinstance(item, str):
            words.append(int(item, 16))
        else:
            words.append(int(item))
    return words


def compact_l30_promotion_raw_fitness(program: list[int]) -> float:
    """CPU equal-footing promotion_raw_fitness for compact L30 capability ranking."""
    key = tuple(int(x) for x in program)
    cached = _PROMOTION_RAW_CACHE.get(key)
    if cached is not None:
        return cached
    disk_key = None
    try:
        from evolution.terminal_stagnation import _vm_score_disk_get, _vm_score_disk_put, _vm_score_program_key

        disk_key = _vm_score_program_key(key)
        disk_cached = _vm_score_disk_get("promotion_raw_l30", disk_key)
        if disk_cached is not None:
            _PROMOTION_RAW_CACHE[key] = disk_cached
            return disk_cached
    except ImportError:
        pass
    global _CAPABILITY_SCORE_ENGINE
    if _CAPABILITY_SCORE_ENGINE is None:
        from evolution_engine import EvolutionConfig, EvolutionEngine

        _CAPABILITY_SCORE_ENGINE = EvolutionEngine(EvolutionConfig(max_level=30, curriculum_level=30, use_gpu=False))
    eng = _CAPABILITY_SCORE_ENGINE
    from tasks_rv32i import get_task_level

    _isolate_capability_score_engine(eng)
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    eng.task_instance = get_task_level(30)(seed=30).generate_instance()
    with terminal_certified_fitness_context(eng):
        trace = eng._evaluate(list(program))
        verify = eng._verify_score_from_trace(trace)
        score = float(promotion_raw_fitness(eng, list(program), trace, verify_score=verify))
    if len(_PROMOTION_RAW_CACHE) > 512:
        _PROMOTION_RAW_CACHE.clear()
    _PROMOTION_RAW_CACHE[key] = score
    if disk_key is not None:
        try:
            from evolution.terminal_stagnation import _vm_score_disk_put

            _vm_score_disk_put("promotion_raw_l30", disk_key, score)
        except ImportError:
            pass
    return score


def row_capability_score(row: dict[str, Any]) -> float:
    """Primary capability score for jsonl/carry rows (promotion_raw, not noisy task_fitness)."""
    stored = row.get("promotion_raw_fitness")
    if stored is not None:
        try:
            stored_f = float(stored)
            # Ignore timeout/crash sentinels and GPU-inflated promotion_raw artifacts.
            if 0.0 < stored_f <= 330.0:
                return stored_f
        except (TypeError, ValueError):
            pass
    from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS

    words = int(row.get("champion_words", row.get("words", 0)))
    lvl = int(row.get("final_level", row.get("level", 0)))
    verify = float(row.get("champion_verify_score", row.get("verify_score", 0.0)))
    if lvl >= 30 and 12 <= words <= L30_COMPACT_MAX_WORDS and verify >= 0.99 - 1e-6:
        prog = decode_program_words(row.get("champion_program"))
        if prog is not None:
            return compact_l30_promotion_raw_fitness(prog)
    task = row.get("task_fitness")
    if task is not None:
        try:
            return float(task)
        except (TypeError, ValueError):
            pass
    return float(row.get("final_fitness", row.get("fitness", 0.0)))


def engine_promotion_raw_fitness(engine, program: list[int], trace: dict | None = None) -> float:
    """Promotion raw fitness for the live engine champion (GPU or CPU eval path)."""
    if trace is None:
        trace = engine._evaluate(program)
    verify = float(engine._verify_score_from_trace(trace))
    with terminal_certified_fitness_context(engine):
        return float(promotion_raw_fitness(engine, program, trace, verify_score=verify))


def ensure_champion_trace(engine, program: list[int] | None = None) -> dict:
    """Return an eval trace for champion metrics; re-evaluate if trace is missing or stale."""
    prog = list(program if program is not None else engine.champion or [])
    if not prog:
        raise ValueError("ensure_champion_trace: no champion program")
    trace = getattr(engine, "champion_trace", None)
    champ = getattr(engine, "champion", None)
    if trace is not None and champ is not None and tuple(int(x) for x in prog) == tuple(int(x) for x in champ):
        return trace
    trace = engine._evaluate(prog)
    if champ is not None and tuple(int(x) for x in prog) == tuple(int(x) for x in champ):
        engine.champion_trace = trace
    return trace


def align_scan_floor_persisted_metrics(
    engine: Any,
    metrics: dict[str, Any],
) -> None:
    """Persist selection-scale fitness at scan floor (matches promotion_raw)."""
    if not metrics.get("champion_verified"):
        return
    try:
        from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor
    except ImportError:
        return
    words = int(metrics.get("champion_words", 0))
    if not terminal_efficiency_at_scan_floor(words):
        return
    prom = metrics.get("promotion_raw_fitness")
    if prom is None or float(prom) <= 0.0:
        champ = getattr(engine, "champion", None) if engine is not None else None
        if champ is None:
            return
        prom = compact_l30_promotion_raw_fitness(list(champ))
    fit = float(prom)
    metrics["final_fitness"] = fit
    metrics["verified_fitness"] = fit
    if engine is not None:
        engine.champion_fitness = fit


def end_of_cycle_champion_fitness(
    engine,
    program: list[int],
    trace: dict,
    champion_verify: float,
) -> float:
    """Metrics/champion_fitness at cycle end — prom_raw at scan floor, reporting elsewhere."""
    from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor, terminal_verify_promote_active

    if (
        terminal_verify_promote_active()
        and terminal_efficiency_at_scan_floor(len(program))
        and champion_verify >= 0.99 - 1e-6
    ):
        return float(selection_fitness(engine, program, trace, verify_score=champion_verify))
    return float(reporting_champion_fitness(engine, program, trace, champion_verify))


def certified_reporting_floor(engine) -> float:
    """Best trusted fitness for saturated-verify reporting (survives noisy re-eval)."""
    candidates = [
        float(getattr(engine, "champion_fitness", 0.0)),
        float(getattr(engine, "_terminal_promoted_fitness", 0.0)),
        float(getattr(engine, "_certified_fitness_floor", 0.0)),
    ]
    return max(candidates)


def reporting_champion_fitness(
    engine,
    program: list[int],
    trace: dict,
    champion_verify: float,
) -> float:
    """Stable end-of-cycle fitness for metrics/EPI when champion fully verifies."""
    fitness = float(compute_fitness(engine, program, trace))
    if champion_verify < 1.0 - 1e-6:
        return fitness
    mult = partial_verify_multiplier(engine, trace)
    if mult < 1.0 - 1e-6:
        fitness /= mult
    if fitness < 200.0:
        from tasks_rv32i import get_task_level

        level = int(getattr(engine, "champion_level", 0))
        level_cls = get_task_level(level)
        saved_instance = engine.task_instance
        try:
            engine.task_instance = level_cls(seed=level).generate_instance()
            canon_trace = engine._evaluate(program)
            fitness = float(compute_fitness(engine, program, canon_trace))
            mult = partial_verify_multiplier(engine, canon_trace)
            if mult < 1.0 - 1e-6:
                fitness /= mult
        finally:
            engine.task_instance = saved_instance
    bonused = _terminal_length_efficiency_bonus(engine, program, fitness)
    floor = certified_reporting_floor(engine)
    if floor < 200.0:
        return bonused
    champ = getattr(engine, "champion", None)
    if (
        champ is not None
        and len(program) == len(champ)
        and tuple(int(x) for x in program) == tuple(int(x) for x in champ)
    ):
        # Same program re-reported end-of-cycle: floor already length-adjusted.
        return max(bonused, floor)
    floored = _terminal_length_efficiency_bonus(engine, program, floor)
    return max(bonused, floored)


def partial_verify_multiplier(engine, trace: dict) -> float:
    """Scale fitness by the program's own verify score (not the champion's stale partial)."""
    verify_fn = getattr(engine, "_verify_score_from_trace", None)
    if callable(verify_fn):
        partial = float(verify_fn(trace))
    else:
        partial = float(getattr(engine, "_champion_partial_verify", 1.0))
    partial = max(0.3, partial)
    level = int(getattr(engine, "champion_level", 0))
    if level >= 17:
        from evolution.progress_controller import ProgressController

        certify = ProgressController.certify_verify_threshold(level)
        if level >= int(__import__("os").environ.get("EE_MAX_LEVEL", "30")):
            from evolution.curriculum_advance import depth_carry_verify_floor

            certify = depth_carry_verify_floor(level)
        if partial < certify and certify > 0:
            partial = max(0.15, partial / certify)
    return partial


def finalize_with_blending(
    engine,
    base_score: float,
    program: list[int],
    trace: dict,
) -> float:
    """Apply novelty/QD blending and partial verify scaling."""
    fitness = base_score
    _qd_fn = engine._compute_qd_axes_fn
    _prog_qd_desc = None
    if engine.config.qd_bonus_weight > 0.0 and engine.qd_archive is not None and _qd_fn is not None:
        try:
            _prog_qd_desc = _qd_fn(program, trace)
        except (TypeError, ValueError, RuntimeError):
            pass
    fitness = apply_novelty_qd_blending(engine, fitness, program, trace, base_score, _qd_fn, _prog_qd_desc)
    fitness *= partial_verify_multiplier(engine, trace)
    return fitness


def challenger_rank_from_trace(engine, program: list[int], trace: dict, fitness: float) -> float:
    """Composite rank score for challenger selection at L12+."""
    from evolution.register_repair import champion_rank_score, score_target_register

    level = int(getattr(engine, "champion_level", 0))
    if level < 12:
        return fitness
    expected_regs, expected_total, _status, final_regs, _, _ = extract_trace_info(engine, trace)
    if expected_total <= 0:
        return fitness
    tr_score = score_target_register(level, expected_regs, final_regs)
    verify_fn = getattr(engine, "_verify_score_from_trace", None)
    if callable(verify_fn):
        verify = float(verify_fn(trace))
    else:
        verify = float(getattr(engine, "_champion_verify_score", 0.0))
    return champion_rank_score(level, fitness, tr_score, verify)


def compute_fitness(
    engine,
    program: list[int],
    trace: dict[str, Any],
    eval_budget: int | None = None,
    raw_only: bool = False,
) -> float:
    """Compute crash-penalized fitness."""
    if engine.task_instance is None:
        return 0.0
    budget = eval_budget if eval_budget is not None else engine._cached_eval_budget
    (
        expected_regs,
        expected_total,
        status,
        final_regs,
        instr_count,
        trap,
    ) = extract_trace_info(engine, trace)
    if status in ("crash", "trap", "exception"):
        engine._raw_task_score = 0.0
        return -100.0 if ("illegal" in trap or "invalid" in trap) else -50.0
    if status == "timeout" or instr_count >= budget:
        engine._raw_task_score = 0.0
        return -25.0
    level = int(getattr(engine, "champion_level", 0))
    if level >= 12:
        from evolution.register_repair import weighted_register_match

        reg_match = weighted_register_match(level, expected_regs, expected_total, final_regs)
    else:
        reg_match = score_register_match(expected_regs, expected_total, final_regs)
    base = compute_base_score(reg_match, instr_count, status, budget)
    opcode_cat_array = _get_opcode_cat_array()
    bonuses = compute_all_bonuses(engine, status, instr_count, program, reg_match, base, opcode_cat_array)
    base = apply_bonuses_and_normalize(engine, base, bonuses)
    if raw_only:
        return base
    return finalize_with_blending(engine, base, program, trace)


# ------------------------------------------------------------------
# Batch fitness helpers
# ------------------------------------------------------------------


def batch_classify_programs(traces: list, n: int, budget: int):
    """Classify programs by status (crash/timeout/ok) for batch fitness."""
    import numpy as np

    statuses = [t.get("status", "") for t in traces]
    instr_counts = np.array([t.get("instructions_count", 0) for t in traces], dtype=np.int64)
    trap_causes = [t.get("trap_cause", "") for t in traces]
    crash_mask = np.zeros(n, dtype=bool)
    illegal_mask = np.zeros(n, dtype=bool)
    timeout_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        s = statuses[i]
        if s in ("crash", "trap", "exception"):
            crash_mask[i] = True
            if "illegal" in trap_causes[i] or "invalid" in trap_causes[i]:
                illegal_mask[i] = True
        elif s == "timeout" or instr_counts[i] >= budget:
            timeout_mask[i] = True
    return instr_counts, crash_mask, illegal_mask, timeout_mask


def batch_score_registers(
    ok_traces: list,
    expected_keys: list,
    expected_vals,
    n_ok: int,
):
    """Vectorized register scoring for batch fitness."""
    import numpy as np

    expected_total = len(expected_keys)
    if expected_total == 0:
        return np.ones(n_ok, dtype=np.float64)
    actual_vals = np.zeros((n_ok, len(expected_keys)), dtype=np.float64)
    for j, trace in enumerate(ok_traces):
        fr = trace.get("final_regs", {})
        if isinstance(fr, dict):
            for k, key in enumerate(expected_keys):
                actual_vals[j, k] = fr.get(key, 0)
    exact_match = actual_vals == expected_vals
    diff = np.abs(actual_vals - expected_vals)
    max_diff = np.maximum(np.abs(expected_vals), 1.0)
    scores = np.where(exact_match, 1.0, 1.0 / (1.0 + diff / max_diff))
    return scores.mean(axis=1)


def batch_compute_efficiency(
    engine,
    ok_programs: list,
    n_ok: int,
    ok_instr,
    max_prog_len: int,
):
    """Vectorized efficiency scoring for batch fitness."""
    import numpy as np

    eff_weight = getattr(engine.config, "efficiency_score_weight", 1.0)
    if eff_weight <= 0 or max_prog_len <= 0:
        return np.zeros(n_ok, dtype=np.float64)
    padded = np.zeros((n_ok, max_prog_len), dtype=np.int64)
    prog_lengths = np.zeros(n_ok, dtype=np.int64)
    for j, prog in enumerate(ok_programs):
        pl = len(prog)
        prog_lengths[j] = pl
        padded[j, :pl] = prog
    nop_counts = np.sum(padded == 0x00000013, axis=1)
    useful = np.maximum(1, prog_lengths - nop_counts)
    unique_ratios = np.zeros(n_ok, dtype=np.float64)
    for j in range(n_ok):
        total_i = int(prog_lengths[j]) or 1
        unique_ratios[j] = len(set(ok_programs[j])) / total_i
    if engine.champion_level >= 8:
        instr_eff = np.where(
            useful >= 8,
            1.0,
            np.where(useful >= 5, 0.7 + (useful - 5) * 0.1, useful / 5.0),
        )
        opcode_bonus = unique_ratios * 0.5
    else:
        instr_eff = np.where(
            useful <= 5,
            1.0,
            np.where(
                useful <= 10,
                1.0 - (useful - 5) * 0.1,
                np.maximum(0.0, 0.5 - (useful - 10) * 0.05),
            ),
        )
        opcode_bonus = (1.0 - unique_ratios) * 0.5
    return (instr_eff + opcode_bonus) * 30.0 * eff_weight


def batch_compute_structural_bonuses(
    engine,
    ok_programs: list,
    ok_indices,
    traces: list,
    n_ok: int,
    ok_instr,
    _bonus_scale: float,
    _max_single,
    opcode_cat_array: list[int],
):
    """Vectorized structural bonuses (loop, mem, div, smc, graduated, level)."""
    import numpy as np

    lvl = engine.champion_level
    base = np.zeros(n_ok, dtype=np.float64)
    if lvl >= 5:
        _gen = engine.current_generation
        if not hasattr(engine, "_backward_branch_cache") or engine._backward_branch_cache_gen != _gen:
            if not hasattr(engine, "_backward_branch_cache"):
                engine._backward_branch_cache = {}
                engine._backward_branch_cache_gen = -1
            engine._backward_branch_cache.clear()
            engine._backward_branch_cache_gen = _gen
        _bb_cache = engine._backward_branch_cache
        loop_bonuses = np.zeros(n_ok, dtype=np.float64)
        for j, prog in enumerate(ok_programs):
            _prog_key = hash(tuple(prog))
            n_loops = _bb_cache.get(_prog_key)
            if n_loops is None:
                from evolution.diversity_helpers import count_backward_branches

                n_loops = count_backward_branches(prog)
                _bb_cache[_prog_key] = n_loops
            if n_loops > 0:
                loop_bonuses[j] = min(5.0, n_loops * 2.0)
        loop_bonuses *= _bonus_scale
        base += np.minimum(loop_bonuses, _max_single)
    if lvl >= 7:
        mem_bonuses = np.zeros(n_ok, dtype=np.float64)
        for j, idx in enumerate(ok_indices):
            mem_accesses = traces[idx].get("memory_accesses", [])
            if mem_accesses:
                mem_bonuses[j] = min(3.0, len(mem_accesses) * 0.5)
        mem_bonuses *= _bonus_scale
        base += np.minimum(mem_bonuses, _max_single)
    if lvl >= 5:
        div_bonuses = np.zeros(n_ok, dtype=np.float64)
        for j, prog in enumerate(ok_programs):
            seen_cats = set()
            for word in prog:
                if isinstance(word, int):
                    cat = word & 0x7F
                    if 0 <= cat < 128:
                        seen_cats.add(opcode_cat_array[cat])
            n_cats = len(seen_cats - {-1})
            if n_cats >= 3:
                div_bonuses[j] = min(2.0, (n_cats - 2) * 0.5)
        div_bonuses *= _bonus_scale
        base += np.minimum(div_bonuses, _max_single)
    base += np.minimum(lvl * 25.0, _max_single)
    return base


def batch_apply_crash_penalties(n, crash_mask, illegal_mask, timeout_mask):
    """Apply crash/timeout penalties to fitness array."""
    import numpy as np

    fitnesses = np.zeros(n, dtype=np.float64)
    raw_scores = np.zeros(n, dtype=np.float64)
    fitnesses[illegal_mask] = -100.0
    fitnesses[crash_mask & ~illegal_mask] = -50.0
    fitnesses[timeout_mask] = -25.0
    return fitnesses, raw_scores


def batch_get_expected(engine):
    """Extract expected output arrays for batch computation."""
    import numpy as np

    expected = engine.task_instance.get("expected_output", {})
    expected_regs = expected.get("registers", {})
    expected_keys = list(expected_regs.keys()) if expected_regs else []
    expected_vals = (
        np.array([expected_regs[k] for k in expected_keys], dtype=np.float64)
        if expected_keys
        else np.array([], dtype=np.float64)
    )
    return expected_regs, expected_keys, expected_vals


def batch_compute_base(
    engine,
    programs,
    traces,
    ok_indices,
    ok_instr,
    n_ok,
    expected_keys,
    expected_vals,
    budget,
    expected_regs,
):
    """Compute base fitness scores for batch."""
    import numpy as np

    [programs[i] for i in ok_indices]
    ok_traces = [traces[i] for i in ok_indices]
    reg_matches = batch_score_registers(ok_traces, expected_keys, expected_vals, n_ok)
    bloat = np.where(ok_instr > 50, (ok_instr - 50).astype(np.float64) * 0.05, 0.0)
    exec_credit = np.zeros(n_ok, dtype=np.float64)
    pos = reg_matches > 0.0
    if budget > 0:
        exec_credit[pos] = np.minimum(10.0, (ok_instr[pos].astype(np.float64) / budget) * 10.0)
    solved_bonus = np.where(reg_matches >= 0.99, 50.0, 0.0)
    return reg_matches * 100.0 - bloat + exec_credit + solved_bonus


def batch_apply_bonuses(
    engine,
    base,
    programs,
    traces,
    ok_indices,
    n_ok,
    expected_regs,
    ok_instr,
    opcode_cat_array: list[int],
):
    """Apply efficiency, structural, and graduated bonuses."""
    import numpy as np

    from tasks_rv32i import LEVELS

    _base_for_norm = np.maximum(base, 1.0)
    _max_single = getattr(engine.config, "max_bonus_fraction", 1.0) * _base_for_norm
    ok_programs = [programs[i] for i in ok_indices]
    max_prog_len = max((len(p) for p in ok_programs), default=0)
    base += np.minimum(
        batch_compute_efficiency(engine, ok_programs, n_ok, ok_instr, max_prog_len),
        _max_single,
    )
    base += batch_compute_structural_bonuses(
        engine,
        ok_programs,
        ok_indices,
        traces,
        n_ok,
        ok_instr,
        1.0,
        _max_single,
        opcode_cat_array,
    )
    lvl = engine.champion_level
    if lvl >= 12:
        smc_mask = np.zeros((n_ok, max_prog_len if max_prog_len > 0 else 1), dtype=bool)
        for j, prog in enumerate(ok_programs):
            for wi, word in enumerate(prog):
                if (word & 0x7F) in (0x37, 0x13, 0x33, 0x23, 0x67, 0x0F):
                    smc_mask[j, wi] = True
        smc_counts = smc_mask.sum(axis=1)
        smc_ok = smc_counts >= 3
        base[smc_ok] += np.minimum(
            np.minimum(25.0, smc_counts[smc_ok].astype(np.float64) * 5.0),
            _max_single[smc_ok],
        )
    # Time credit: +20 points for fast solutions (reg_match >= 0.99, instr_count <= 5)
    ok_traces = [traces[i] for i in ok_indices]
    exp_keys = list(expected_regs.keys())
    exp_vals = (
        np.array([expected_regs[k] for k in expected_regs], dtype=np.float64)
        if expected_regs
        else np.array([], dtype=np.float64)
    )
    tc_reg_matches = batch_score_registers(ok_traces, exp_keys, exp_vals, n_ok)
    tc_eligible = (tc_reg_matches >= 0.99) & (ok_instr > 0)
    tc_5_or_less = tc_eligible & (ok_instr <= 5)
    tc_6_to_15 = tc_eligible & (ok_instr > 5) & (ok_instr <= 15)
    tc_above_15 = tc_eligible & (ok_instr > 15)
    time_credit = np.zeros(n_ok, dtype=np.float64)
    time_credit[tc_5_or_less] = 20.0
    time_credit[tc_6_to_15] = 20.0 * (1.0 - (ok_instr[tc_6_to_15].astype(np.float64) - 5.0) / 10.0)
    time_credit[tc_above_15] = np.maximum(0.0, 10.0 - (ok_instr[tc_above_15].astype(np.float64) - 15.0) * 0.2)
    base += np.minimum(time_credit, _max_single)
    cached = engine._bonus_threshold_cache
    bonus_threshold = (
        cached[1] if cached and cached[0] == lvl else max(0.85, 0.95 - 0.10 * (lvl / max(1, len(LEVELS) - 1)))
    )
    grad_reg_matches = tc_reg_matches
    above = grad_reg_matches >= bonus_threshold
    denom = 1.0 - bonus_threshold
    grad_bonus = np.where(
        above,
        (50.0 * (grad_reg_matches - bonus_threshold) / denom if denom > 0 else 0.0),
        0.0,
    )
    base += np.minimum(grad_bonus, _max_single)
    return base


def batch_blend_novelty_qd(
    engine,
    fitnesses,
    programs,
    traces,
    ok_indices,
    n_ok,
    expected_regs,
    base,
    nw,
):
    """Apply novelty/QD blending to batch fitness values."""
    ef = engine._extract_behavioral_descriptor
    qdf = engine._compute_qd_axes_fn
    qd = engine.qd_archive
    qd_w = float(getattr(engine.config, "qd_bonus_weight", 0.0) or 0.0)
    illum_ds = engine._qd_coverage
    freshness_fv = 0.5
    if qd_w > 0.0 and qd is not None and qdf is not None:
        try:
            illum_ds = qd.get_illumination_diversity().get("overall_diversity_score", engine._qd_coverage)
        except (KeyError, TypeError, ValueError):
            illum_ds = engine._qd_coverage
        try:
            fd = qd.get_cell_freshness(engine.current_generation)
            freshness_fv = fd.get("avg_freshness", 0.5) if isinstance(fd, dict) else 0.5
        except (AttributeError, TypeError):
            freshness_fv = 0.5
    for j, idx in enumerate(ok_indices):
        fj = float(base[j])
        prog, trace = programs[idx], traces[idx]
        ms = float(len(expected_regs)) if expected_regs else 1.0
        if nw > 0.0 and engine.novelty_archive is not None:
            nv = 0.0
            if getattr(engine.novelty_archive, "size", 0):
                try:
                    nv = engine.novelty_archive.decayed_novelty(prog, engine.current_generation)
                except (KeyError, TypeError, ValueError):
                    if ef is not None:
                        try:
                            d = ef(trace, prog, budget=engine.config.eval_budget)
                            nv = engine.novelty_archive.novelty_score(d)
                        except (TypeError, ValueError, RuntimeError):
                            nv = 0.0
            fj = fj * (1.0 - nw) + nv * (ms / 10.0) * nw
        if qd_w > 0.0 and qd is not None and qdf is not None:
            try:
                pd = qdf(prog, trace)
                if pd is not None:
                    qb = qd.selection_bias(pd)
                    qw = qd_w
                    st = engine.generations_stalled
                    if st > 10:
                        qw *= min(2.5, 1.0 + 0.0375 * (st - 10))
                    ds = illum_ds
                    if ds < 0.20:
                        qw *= 1.0 + 2.5 * (0.20 - ds)
                    qw *= 1.5 - freshness_fv
                    fj += qb * qw
                    if engine.champion is not None and engine.champion_trace is not None:
                        cd = (
                            engine._cached_champion_qd_desc
                            if engine._champion_qd_valid
                            else qdf(engine.champion, engine.champion_trace)
                        )
                        if cd is not None:
                            engine._cached_champion_qd_desc = cd
                            engine._champion_qd_valid = True
                            dist = abs(pd[0] - cd[0]) + abs(pd[1] - cd[1])
                            if dist < 2.0:
                                fj *= 1.0 - 0.10 * (1.0 - dist / 2.0)
            except (TypeError, ValueError, RuntimeError):
                pass
        fitnesses[idx] = fj


# ------------------------------------------------------------------
# Batch fitness orchestrator
# ------------------------------------------------------------------


def batch_compute_fitness(
    engine,
    programs: list[list[int]],
    traces: list[dict[str, Any]],
    eval_budget: int | None = None,
    raw_only: bool = False,
) -> tuple[list[float], list[float]]:
    """Vectorized batch fitness computation using numpy arrays."""
    import numpy as np

    n = len(programs)
    if n == 0:
        return [], []
    if engine.task_instance is None:
        return [0.0] * n, [0.0] * n
    budget = eval_budget if eval_budget is not None else engine._cached_eval_budget
    expected_regs, expected_keys, expected_vals = batch_get_expected(engine)
    instr_counts, crash_mask, illegal_mask, timeout_mask = batch_classify_programs(traces, n, budget)
    fitnesses, raw_scores = batch_apply_crash_penalties(n, crash_mask, illegal_mask, timeout_mask)
    ok_mask = ~crash_mask & ~timeout_mask
    ok_indices = np.where(ok_mask)[0]
    n_ok = len(ok_indices)
    if n_ok == 0:
        return fitnesses.tolist(), raw_scores.tolist()
    base = batch_compute_base(
        engine,
        programs,
        traces,
        ok_indices,
        ok_instr := instr_counts[ok_indices],
        n_ok,
        expected_keys,
        expected_vals,
        budget,
        expected_regs,
    )
    opcode_cat_array = _get_opcode_cat_array()
    base = batch_apply_bonuses(
        engine, base, programs, traces, ok_indices, n_ok, expected_regs, ok_instr, opcode_cat_array
    )
    raw_scores[ok_indices] = base
    if raw_only:
        fitnesses[ok_indices] = base
        return fitnesses.tolist(), raw_scores.tolist()
    nw = engine._novelty_weight_cache
    if engine._current_diversity < 0.2 and nw > 0.0:
        nw = min(nw * (1.0 + (1.0 - engine._current_diversity / 0.2)), 0.8)
    batch_blend_novelty_qd(engine, fitnesses, programs, traces, ok_indices, n_ok, expected_regs, base, nw)
    for idx in ok_indices:
        fitnesses[idx] *= partial_verify_multiplier(engine, traces[int(idx)])
    return fitnesses.tolist(), raw_scores.tolist()


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _get_opcode_cat_array():
    """Get the module-level opcode category array.

    Lazily imported from the evolution_engine module to avoid circular imports.
    """
    try:
        from evolution_engine import _OPCODE_CAT_ARRAY

        return _OPCODE_CAT_ARRAY
    except ImportError:
        # Fallback: build a minimal one
        try:
            from mutation_proposer import OPCODE_CATEGORY_MAP

            arr = [-1] * 128
            for opcode_val, cat_val in OPCODE_CATEGORY_MAP.items():
                arr[opcode_val & 0x7F] = cat_val
            return arr
        except ImportError:
            return [-1] * 128
