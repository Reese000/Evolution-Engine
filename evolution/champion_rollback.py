"""Restore champion from last clean cycle when same-level collapse occurs."""

from __future__ import annotations

import json
import os
from typing import Any

from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS

_JSONL_INC: dict[tuple, dict[str, Any]] = {}


def _certified_rank_score(row: dict[str, Any]) -> float:
    """Capability score for certified selection (CPU promotion_raw, not noisy task_fitness)."""
    from evolution.fitness_helpers import row_capability_score

    return float(row_capability_score(row))


def compact_l30_certified_row_valid(row: dict[str, Any]) -> bool:
    """Reject saturated compact rows whose logged prom_raw masks side/held-out regression."""
    level = int(row.get("final_level", 0))
    verify = float(row.get("champion_verify_score", 0.0))
    words = int(row.get("champion_words", 0))
    if level < 30 or verify < 0.99 - 1e-6:
        return True
    if not (12 <= words <= L30_COMPACT_MAX_WORDS):
        return True
    from evolution.fitness_helpers import decode_program_words

    prog = decode_program_words(row.get("champion_program"))
    if not prog:
        return False
    from evolution.terminal_stagnation import curriculum_verify_mean, heldout_l30_verify_mean

    min_l10 = float(os.environ.get("EE_CERTIFIED_MIN_L10", "0.79"))
    min_h = float(os.environ.get("EE_CERTIFIED_MIN_HELDOUT", "0.989"))
    l10 = float(curriculum_verify_mean(prog, 10))
    heldout = float(heldout_l30_verify_mean(prog, 32))
    return l10 + 1e-6 >= min_l10 and heldout + 1e-6 >= min_h


def _jsonl_inc_begin(results_file: str, extra: tuple, empty_acc: dict[str, Any]) -> tuple:
    path = os.path.abspath(results_file)
    try:
        size = os.path.getsize(path)
    except OSError:
        return None, path, 0, 0, dict(empty_acc)
    key = (path, extra)
    st = _JSONL_INC.get(key)
    offset = 0
    acc = dict(empty_acc)
    if st is not None and int(st.get("size", 0)) <= size:
        offset = int(st.get("offset", 0))
        acc = dict(st.get("acc") or empty_acc)
    return key, path, size, offset, acc


def _jsonl_inc_commit(key: tuple | None, size: int, offset: int, acc: dict[str, Any]) -> None:
    if key is None:
        return
    if len(_JSONL_INC) > 24:
        _JSONL_INC.pop(next(iter(_JSONL_INC)))
    _JSONL_INC[key] = {"size": int(size), "offset": int(offset), "acc": acc}


def _jsonl_new_rows(path: str, offset: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    last_good = int(offset)
    with open(path, "rb") as f:
        if offset:
            f.seek(offset)
        while True:
            raw = f.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                break
            last_good = f.tell()
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows, last_good


def find_last_clean_champion(
    results_file: str,
    level: int,
    *,
    min_verify: float = 0.72,
    min_words: int = 20,
    max_words: int | None = None,
    before_cycle: int | None = None,
) -> dict[str, Any] | None:
    """Return the newest result row at *level* meeting quality gates."""
    if not os.path.isfile(results_file):
        return None
    extra = ("last_clean", int(level), float(min_verify), int(min_words), max_words, before_cycle)
    key, path, size, offset, acc = _jsonl_inc_begin(
        results_file, extra, {"best": None, "best_cycle": -1}
    )
    lvl = int(level)
    cap = max_words
    if cap is None and lvl >= 21:
        try:
            from evolution.champion_bloat_guard import level_word_cap

            cap = level_word_cap(lvl)
        except (ImportError, TypeError, ValueError):
            cap = 36 if lvl >= 21 else None
    best: dict[str, Any] | None = acc.get("best")
    best_cycle = int(acc.get("best_cycle", -1))
    new_rows, new_off = _jsonl_new_rows(path, offset)
    for row in new_rows:
        if "error" in row:
            continue
        cycle = int(row.get("cycle", -1))
        if cycle < 0:
            continue
        if before_cycle is not None and cycle >= before_cycle:
            continue
        if int(row.get("final_level", 0)) != lvl:
            continue
        verify = float(row.get("champion_verify_score", 0.0))
        words = int(row.get("champion_words", 0))
        fitness = float(row.get("final_fitness", 0.0))
        if verify < float(min_verify):
            continue
        word_floor = 8 if verify >= 0.99 - 1e-6 else int(min_words)
        if words < word_floor:
            continue
        if cap is not None and words > int(cap):
            continue
        if fitness <= -25.0:
            continue
        prog = row.get("champion_program")
        if not isinstance(prog, list) or not prog:
            continue
        if cycle > best_cycle:
            best_cycle = cycle
            best = row
    _jsonl_inc_commit(key, size, new_off, {"best": best, "best_cycle": best_cycle})
    return best


def find_best_certified_champion(
    results_file: str,
    level: int,
    *,
    min_verify: float = 0.99,
    min_fitness: float = 100.0,
    min_words: int = 8,
    max_words: int | None = None,
    before_cycle: int | None = None,
) -> dict[str, Any] | None:
    """Return the highest-fitness row at *level* with strong verify (not merely newest)."""
    if not os.path.isfile(results_file):
        return None
    extra = (
        "best_cert",
        int(level),
        float(min_verify),
        float(min_fitness),
        int(min_words),
        max_words,
        before_cycle,
    )
    key, path, size, offset, acc = _jsonl_inc_begin(
        results_file,
        extra,
        {"best": None, "best_fit": float("-inf"), "best_cycle": -1},
    )
    lvl = int(level)
    cap = max_words
    best: dict[str, Any] | None = acc.get("best")
    best_fit = float(acc.get("best_fit", float("-inf")))
    best_cycle = int(acc.get("best_cycle", -1))
    new_rows, new_off = _jsonl_new_rows(path, offset)
    for row in new_rows:
        if "error" in row:
            continue
        cycle = int(row.get("cycle", -1))
        if cycle < 0:
            continue
        if before_cycle is not None and cycle >= before_cycle:
            continue
        if int(row.get("final_level", 0)) != lvl:
            continue
        verify = float(row.get("champion_verify_score", 0.0))
        words = int(row.get("champion_words", 0))
        fitness = float(row.get("final_fitness", 0.0))
        stored_pr = row.get("promotion_raw_fitness")
        if stored_pr is not None:
            try:
                if float(stored_pr) <= 0.0:
                    continue
            except (TypeError, ValueError):
                pass
        rank_score = _certified_rank_score(row)
        word_floor = int(min_words)
        if lvl >= 21 and verify >= 0.99 - 1e-6:
            word_floor = max(word_floor, 12)
        if verify < float(min_verify) or words < word_floor:
            continue
        if fitness < float(min_fitness):
            continue
        if cap is not None and words > int(cap):
            continue
        if not compact_l30_certified_row_valid(row):
            continue
        prog = row.get("champion_program")
        if not isinstance(prog, list) or not prog:
            continue
        best_words = int(best.get("champion_words", 999)) if best else 999
        if rank_score > best_fit + 1e-6:
            best_fit = rank_score
            best_cycle = cycle
            best = row
        elif abs(rank_score - best_fit) <= 1e-6 and words < best_words:
            best_fit = rank_score
            best_cycle = cycle
            best = row
        elif (
            abs(rank_score - best_fit) <= 1e-6
            and words == best_words
            and cycle > best_cycle
        ):
            best_fit = rank_score
            best_cycle = cycle
            best = row
    _jsonl_inc_commit(
        key, size, new_off, {"best": best, "best_fit": best_fit, "best_cycle": best_cycle}
    )
    return best


def find_best_honest_compact_l30(
    results_file: str,
    level: int = 30,
    *,
    min_verify: float = 0.20,
    max_verify: float = 0.99,
    min_words: int = 12,
    max_words: int = 40,
) -> dict[str, Any] | None:
    """Highest logged verify among compact L30 programs, excluding trusted 1.0 rows.

    Compact verify=1.0 is often a GPU/instance lie; held-out CPU is much lower.
    Honest jsonl scores (e.g. 0.737) are the recoverable high-water mark.
    """
    if not os.path.isfile(results_file):
        return None
    extra = (
        "honest",
        int(level),
        float(min_verify),
        float(max_verify),
        int(min_words),
        int(max_words),
    )
    key, path, size, offset, acc = _jsonl_inc_begin(
        results_file,
        extra,
        {"best": None, "best_verify": float("-inf"), "best_cycle": -1},
    )
    lvl = int(level)
    best: dict[str, Any] | None = acc.get("best")
    best_verify = float(acc.get("best_verify", float("-inf")))
    best_cycle = int(acc.get("best_cycle", -1))
    new_rows, new_off = _jsonl_new_rows(path, offset)
    for row in new_rows:
        if "error" in row:
            continue
        if int(row.get("final_level", 0)) != lvl:
            continue
        words = int(row.get("champion_words", 0))
        if words < int(min_words) or words > int(max_words):
            continue
        verify = float(row.get("champion_verify_score", 0.0))
        if verify < float(min_verify) or verify >= float(max_verify) - 1e-6:
            continue
        prog = row.get("champion_program")
        if not isinstance(prog, list) or not prog:
            continue
        cycle = int(row.get("cycle", -1))
        if verify > best_verify + 1e-6 or (
            abs(verify - best_verify) <= 1e-6 and cycle > best_cycle
        ):
            best_verify = verify
            best_cycle = cycle
            best = row
    _jsonl_inc_commit(
        key,
        size,
        new_off,
        {"best": best, "best_verify": best_verify, "best_cycle": best_cycle},
    )
    return best


def flag_verify_regression_rollback(
    project_root: str,
    *,
    level: int,
    prev_verify: float,
    prev_program: list[Any],
    prev_cycle: int,
    current_verify: float,
    min_drop: float = 0.03,
) -> bool:
    """Queue champion rollback when verify drops sharply at high depth."""
    if int(level) < 21:
        return False
    drop = float(prev_verify) - float(current_verify)
    effective_drop = float(min_drop)
    if float(prev_verify) >= 0.99:
        effective_drop = min(effective_drop, 0.01)
    if drop < effective_drop or not prev_program:
        return False
    tick_path = os.path.join(project_root, ".agents", "tick_state.json")
    try:
        os.makedirs(os.path.dirname(tick_path), exist_ok=True)
        tick: dict[str, Any] = {}
        if os.path.isfile(tick_path):
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
        tick["champion_rollback_program"] = list(prev_program)
        tick["champion_rollback_cycle"] = int(prev_cycle)
        tick["apply_champion_rollback"] = True
        with open(tick_path, "w", encoding="utf-8") as f:
            json.dump(tick, f, indent=2)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return True


def apply_l30_certified_metrics_if_degraded(
    project_root: str,
    engine: Any,
    metrics: dict[str, Any],
) -> bool:
    """Sync engine + metrics to best certified champion when L30 regressed."""
    level = int(metrics.get("final_level", 0))
    if level < 30:
        return False

    from evolution.fitness_helpers import (
        end_of_cycle_champion_fitness,
        ensure_champion_trace,
        terminal_certified_fitness_context,
    )

    with terminal_certified_fitness_context(engine):
        if engine.champion is not None:
            verify = max(
                float(metrics.get("champion_verify_score", 0.0)),
                float(getattr(engine, "_champion_verify_score", 0.0)),
            )
            trace = ensure_champion_trace(engine)
            fit = float(
                end_of_cycle_champion_fitness(
                    engine,
                    engine.champion,
                    trace,
                    verify,
                )
            )
            words = len(engine.champion)
            metrics["final_fitness"] = fit
            metrics["champion_verify_score"] = verify
            metrics["champion_verified"] = verify >= 0.99 - 1e-6
            metrics["champion_words"] = words
            metrics["champion_program"] = [f"0x{w:08X}" for w in engine.champion]
            metrics["verified_fitness"] = fit if metrics["champion_verified"] else 0.0
            from evolution.fitness_helpers import align_scan_floor_persisted_metrics

            align_scan_floor_persisted_metrics(engine, metrics)
        else:
            fit = float(metrics.get("final_fitness", 0.0))
            verify = float(metrics.get("champion_verify_score", 0.0))
            words = int(metrics.get("champion_words", 0))

    results = os.path.join(project_root, "multi_cycle_results.jsonl")
    clean = find_best_certified_champion(
        results, level, min_verify=0.99, min_fitness=100.0
    )
    if not clean:
        return False

    from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS

    clean_words = int(clean.get("champion_words", words))
    clean_fit = float(clean.get("final_fitness", 0.0))
    clean_cap = _certified_rank_score(clean)
    if verify >= 0.99 - 1e-6:
        from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

        current_cap = float(metrics.get("promotion_raw_fitness", 0.0) or 0.0)
        if current_cap > 330.0 or current_cap <= 0.0:
            if engine.champion is not None:
                current_cap = float(compact_l30_promotion_raw_fitness(list(engine.champion)))
            else:
                current_cap = clean_cap
        if current_cap + 0.01 >= clean_cap:
            return False
    else:
        collapsed = words < int(clean_words) - 2
        if not collapsed:
            if 12 <= words <= L30_COMPACT_MAX_WORDS:
                return False
            if (
                fit >= 200.0
                and (words < clean_words and fit >= clean_fit - 0.5 or words <= clean_words)
            ):
                return False

    prog = [
        int(x, 16) if isinstance(x, str) else int(x)
        for x in clean["champion_program"]
    ]
    if len(prog) < 8:
        return False

    logged_verify = float(clean.get("champion_verify_score", 0.0))
    logged_fit = float(clean.get("final_fitness", 0.0))

    engine.champion = list(prog)
    engine.champion_level = level
    refresh = getattr(engine, "_refresh_eval_budget", None)
    if callable(refresh):
        refresh()
    evaluate = getattr(engine, "_evaluate", None)
    if not callable(evaluate):
        return False
    engine.champion_trace = evaluate(prog)

    from evolution.fitness_helpers import reporting_champion_fitness

    reported_fit = float(
        reporting_champion_fitness(engine, prog, engine.champion_trace, logged_verify)
    )
    if logged_fit > reported_fit:
        reported_fit = logged_fit

    engine.champion_fitness = reported_fit
    engine._champion_verify_score = logged_verify
    engine._champion_verified = logged_verify >= 0.99 - 1e-6

    metrics["final_fitness"] = reported_fit
    metrics["champion_verify_score"] = logged_verify
    metrics["champion_verified"] = logged_verify >= 0.99 - 1e-6
    metrics["verified_fitness"] = reported_fit if metrics["champion_verified"] else 0.0
    metrics["champion_words"] = len(prog)
    metrics["champion_program"] = [f"0x{w:08X}" for w in prog]
    clean_prom = clean.get("promotion_raw_fitness")
    if clean_prom is not None:
        try:
            metrics["promotion_raw_fitness"] = float(clean_prom)
        except (TypeError, ValueError):
            pass
    elif verify >= 0.99 - 1e-6:
        metrics["promotion_raw_fitness"] = clean_cap
    return True


def align_archive_parent_to_champion(
    parent: tuple[int, ...],
    champion: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Map a nearby-length archive program onto the champion word count."""
    ebreak = 0x00100073
    champ_n = len(champion)
    if not parent or champ_n < 4:
        return None
    if len(parent) == champ_n:
        return parent
    if len(parent) > champ_n:
        return tuple(parent[: champ_n - 1]) + (ebreak,)
    if len(parent) < 2:
        return None
    prefix_len = len(parent) - 1
    if parent[:prefix_len] != champion[:prefix_len]:
        return None
    return tuple(parent[:prefix_len]) + champion[prefix_len:]


def _archive_parent_seen_entry(value: object) -> tuple[int, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), float(value[1])
    if isinstance(value, (int, float)):
        return int(value), 0.0
    return 0, 0.0


def collect_jsonl_archive_parents(
    results_file: str,
    champion: list[int],
    level: int,
    *,
    max_parents: int = 12,
    word_tolerance: int = 0,
    rank_by_promotion_raw: bool = False,
) -> list[list[int]]:
    """Distinct champion programs from jsonl history (survives QD pruning)."""
    if not champion or not os.path.isfile(results_file):
        return []
    champ_t = tuple(int(w) for w in champion)
    champ_n = len(champ_t)
    tol = max(0, int(word_tolerance))
    extra = (
        "jsonl_archive",
        int(level),
        champ_n,
        tol,
        int(max_parents),
        champ_t,
        bool(rank_by_promotion_raw),
    )
    key, path, size, offset, acc = _jsonl_inc_begin(results_file, extra, {"seen": {}})
    seen: dict[tuple[int, ...], tuple[int, float]] = {}
    for prog_key, meta in (acc.get("seen") or {}).items():
        decoded_key = (
            tuple(int(w) for w in prog_key)
            if isinstance(prog_key, (list, tuple))
            else tuple(int(w) for w in (prog_key,))
        )
        seen[decoded_key] = _archive_parent_seen_entry(meta)
    lvl = int(level)
    new_rows, new_off = _jsonl_new_rows(path, offset)
    for row in new_rows:
        if "error" in row:
            continue
        cycle = int(row.get("cycle", -1))
        if cycle < 0:
            continue
        if int(row.get("final_level", 0)) != lvl:
            continue
        words = int(row.get("champion_words", 0))
        if abs(words - champ_n) > tol:
            continue
        prog = row.get("champion_program")
        if not isinstance(prog, list) or not prog:
            continue
        try:
            decoded = tuple(
                int(x, 16) if isinstance(x, str) else int(x) for x in prog
            )
        except (TypeError, ValueError):
            continue
        if tol > 0 and len(decoded) != champ_n:
            aligned = align_archive_parent_to_champion(decoded, champ_t)
            if aligned is None:
                continue
            decoded = aligned
        if len(decoded) != champ_n or decoded == champ_t:
            continue
        prom = float(row.get("promotion_raw_fitness") or 0.0)
        prev = seen.get(decoded)
        if prev is None:
            seen[decoded] = (cycle, prom)
        elif rank_by_promotion_raw:
            if prom > prev[1] + 1e-6 or (
                abs(prom - prev[1]) <= 1e-6 and cycle > prev[0]
            ):
                seen[decoded] = (cycle, prom)
        elif cycle > prev[0]:
            seen[decoded] = (cycle, prom)
    _jsonl_inc_commit(
        key,
        size,
        new_off,
        {"seen": {prog: [meta[0], meta[1]] for prog, meta in seen.items()}},
    )
    if rank_by_promotion_raw:
        ranked = sorted(seen.items(), key=lambda item: (item[1][1], item[1][0]), reverse=True)
        return [list(prog) for prog, _meta in ranked[: max(1, int(max_parents))]]
    ranked = sorted(seen.items(), key=lambda item: item[1][0], reverse=True)
    return [list(prog) for prog, _meta in ranked[: max(1, int(max_parents))]]


def collect_jsonl_long_archive_parents(
    results_file: str,
    champion: list[int],
    level: int,
    *,
    max_parents: int = 12,
    word_tolerance: int = 12,
    min_promotion_raw: float = 300.0,
) -> list[list[int]]:
    """Longer-than-champion archive programs ranked by promotion_raw (unaligned)."""
    if not champion or not os.path.isfile(results_file):
        return []
    champ_t = tuple(int(w) for w in champion)
    champ_n = len(champ_t)
    tol = max(0, int(word_tolerance))
    extra = (
        "jsonl_long_archive",
        int(level),
        champ_n,
        tol,
        int(max_parents),
        champ_t,
        float(min_promotion_raw),
    )
    key, path, size, offset, acc = _jsonl_inc_begin(results_file, extra, {"seen": {}})
    seen: dict[tuple[int, ...], tuple[int, float]] = {}
    for prog_key, meta in (acc.get("seen") or {}).items():
        decoded_key = (
            tuple(int(w) for w in prog_key)
            if isinstance(prog_key, (list, tuple))
            else tuple(int(w) for w in (prog_key,))
        )
        seen[decoded_key] = _archive_parent_seen_entry(meta)
    lvl = int(level)
    new_rows, new_off = _jsonl_new_rows(path, offset)
    for row in new_rows:
        if "error" in row:
            continue
        cycle = int(row.get("cycle", -1))
        if cycle < 0:
            continue
        if int(row.get("final_level", 0)) != lvl:
            continue
        words = int(row.get("champion_words", 0))
        if words <= champ_n or words > champ_n + tol:
            continue
        prog = row.get("champion_program")
        if not isinstance(prog, list) or not prog:
            continue
        try:
            decoded = tuple(
                int(x, 16) if isinstance(x, str) else int(x) for x in prog
            )
        except (TypeError, ValueError):
            continue
        if len(decoded) <= champ_n or decoded == champ_t:
            continue
        prom = float(row.get("promotion_raw_fitness") or 0.0)
        if prom < float(min_promotion_raw):
            continue
        prev = seen.get(decoded)
        if prev is None or prom > prev[1] + 1e-6 or (
            abs(prom - prev[1]) <= 1e-6 and cycle > prev[0]
        ):
            seen[decoded] = (cycle, prom)
    _jsonl_inc_commit(
        key,
        size,
        new_off,
        {"seen": {prog: [meta[0], meta[1]] for prog, meta in seen.items()}},
    )
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    try:
        champ_prom = float(compact_l30_promotion_raw_fitness(list(champion)))
    except (RuntimeError, TypeError, ValueError, OSError):
        champ_prom = float(min_promotion_raw)
    floor_prom = max(float(min_promotion_raw), champ_prom - 1.0)
    rescored: list[tuple[tuple[int, ...], float, int]] = []
    for prog, (cycle, stored) in seen.items():
        try:
            prom = float(compact_l30_promotion_raw_fitness(list(prog)))
        except (RuntimeError, TypeError, ValueError, OSError):
            prom = float(stored)
        if prom + 1e-6 < floor_prom:
            continue
        rescored.append((prog, prom, int(cycle)))
    ranked = sorted(rescored, key=lambda item: (item[1], item[2]), reverse=True)
    out = [list(prog) for prog, _prom, _cycle in ranked[: max(1, int(max_parents))]]
    material = collect_growth_material_parents(champion, max_parents=max_parents)
    if not material:
        return out
    seen_out = {champ_t, *(tuple(p) for p in out)}
    merged = list(out)
    for prog in material:
        key = tuple(prog)
        if key in seen_out:
            continue
        seen_out.add(key)
        merged.append(list(prog))
        if len(merged) >= max(1, int(max_parents)):
            break
    return merged


def _growth_material_path() -> str:
    from evolution.mutation_helpers import _evolution_project_root

    return os.path.join(_evolution_project_root(), ".agents", "scan_floor_growth_material.json")


def _growth_material_storage_anchor(champion: list[int]) -> list[int]:
    """Canonical champion bytes for growth-material bucketing (certified scan-floor length)."""
    from evolution.mutation_helpers import _evolution_project_root, scan_floor_plateau_armed

    if not champion:
        return []
    armed, floor_w = scan_floor_plateau_armed()
    if not armed or floor_w <= 0:
        return list(champion)
    results = os.path.join(_evolution_project_root(), "multi_cycle_results.jsonl")
    row = find_best_certified_champion(
        results,
        30,
        min_words=floor_w,
        max_words=floor_w,
    )
    if not row:
        return list(champion)
    prog = row.get("champion_program")
    if not isinstance(prog, list) or not prog:
        return list(champion)
    out = [int(x, 16) if isinstance(x, str) else int(x) for x in prog]
    return out if len(out) == floor_w else list(champion)


def _growth_material_storage_key(champion: list[int]) -> str:
    anchor = _growth_material_storage_anchor(champion)
    return str(tuple(int(x) for x in anchor))


def _load_growth_material_store() -> dict[str, Any]:
    path = _growth_material_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _save_growth_material_store(data: dict[str, Any]) -> None:
    path = _growth_material_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _normalize_growth_material_program(program: list[int]) -> list[int]:
    """Drop duplicate trailing EBREAK padding before persisting growth material."""
    ebreak = 0x0010_0073
    out = [int(x) for x in program]
    while len(out) > 2 and out[-1] == ebreak and out[-2] == ebreak:
        out.pop()
    return out


def _growth_material_entry_passes_prom_floor(
    anchor: list[int],
    prog: list[int],
    entry_prom: float,
    prom_floor: float,
) -> bool:
    if float(entry_prom) + 1e-6 >= float(prom_floor):
        return True
    return _growth_material_l10_lift_qualifies(anchor, prog)


def _growth_material_l10_lift_qualifies(
    anchor: list[int],
    variant: list[int],
    *,
    min_lift: float | None = None,
) -> bool:
    """True when variant materially lifts L10 for breeding-archive retention."""
    if not anchor or not variant or len(variant) <= len(anchor):
        return False
    from evolution.terminal_stagnation import curriculum_verify_mean

    anchor_l10 = float(curriculum_verify_mean(list(anchor), 10))
    if min_lift is not None:
        margin = float(min_lift)
    elif anchor_l10 < 0.85 - 1e-6:
        margin = float(
            os.environ.get("EE_GROWTH_MATERIAL_L10_LIFT_MARGIN_WEAK", "0.05")
        )
    else:
        margin = float(os.environ.get("EE_GROWTH_MATERIAL_L10_LIFT_MARGIN", "0.15"))
    variant_l10 = float(curriculum_verify_mean(list(variant), 10))
    return variant_l10 + 1e-6 >= anchor_l10 + margin


def _growth_material_capability_qualifies(
    anchor: list[int],
    variant: list[int],
) -> bool:
    """Held-out + curriculum side parity for persisted growth material."""
    if not anchor or not variant or len(variant) <= len(anchor):
        return False
    from evolution.mutation_helpers import (
        scan_floor_growth_completion_qualifies,
        scan_floor_growth_heldout_qualifies,
    )
    from evolution.terminal_stagnation import get_scan_floor_gate

    if not scan_floor_growth_heldout_qualifies(anchor, variant):
        return False
    gate = get_scan_floor_gate(list(anchor))
    return bool(scan_floor_growth_completion_qualifies(anchor, variant, gate=gate))


def record_scan_floor_growth_material(
    champion: list[int],
    variant: list[int],
    *,
    cycle: int = 0,
) -> bool:
    """Persist held-out-safe longer programs for cross-cycle compress/graft breeding."""
    anchor = _growth_material_storage_anchor(champion)
    if not anchor or not variant or len(variant) <= len(anchor):
        return False

    prog_key = _normalize_growth_material_program(variant)
    if len(prog_key) <= len(anchor):
        return False
    if not _growth_material_capability_qualifies(anchor, prog_key):
        return False
    champ_key = _growth_material_storage_key(champion)
    store = _load_growth_material_store()
    bucket = store.get(champ_key)
    if not isinstance(bucket, list):
        bucket = []
    cap = max(8, int(os.environ.get("EE_GROWTH_MATERIAL_CAP", "48")))
    if any(e.get("program") == prog_key for e in bucket if isinstance(e, dict)):
        return False
    from evolution.mutation_helpers import _compact_prom_raw_cached

    try:
        anchor_prom = float(_compact_prom_raw_cached(list(anchor)))
    except (RuntimeError, TypeError, ValueError, OSError):
        anchor_prom = 0.0
    try:
        prom = float(_compact_prom_raw_cached(list(variant)))
    except (RuntimeError, TypeError, ValueError, OSError):
        prom = 0.0
    prom_floor = max(
        100.0,
        anchor_prom - float(os.environ.get("EE_GROWTH_MATERIAL_PROM_MARGIN", "2.0")),
    )
    l10_lift = _growth_material_l10_lift_qualifies(anchor, prog_key)
    if prom + 1e-6 < prom_floor and not l10_lift:
        return False
    entry = {"program": prog_key, "cycle": int(cycle), "prom": prom}
    deduped: list[dict[str, Any]] = []
    for existing in bucket:
        if existing.get("program") == prog_key:
            continue
        existing_prog = existing.get("program")
        if not isinstance(existing_prog, list):
            continue
        existing_prom = float(existing.get("prom") or 0.0)
        if existing_prom + 1e-6 >= prom_floor or _growth_material_l10_lift_qualifies(
            anchor, list(existing_prog)
        ):
            deduped.append(existing)
    deduped.append(entry)

    def _growth_material_bucket_rank(e: dict[str, Any]) -> tuple[float, float, int]:
        prog = e.get("program")
        prog_list = list(prog) if isinstance(prog, list) else []
        l10 = 0.0
        if prog_list:
            from evolution.terminal_stagnation import curriculum_verify_mean

            l10 = float(curriculum_verify_mean(prog_list, 10))
        prom_val = float(e.get("prom") or 0.0)
        lift = (
            1.0
            if prog_list and _growth_material_l10_lift_qualifies(anchor, prog_list)
            else 0.0
        )
        return (lift, l10, prom_val, int(e.get("cycle") or 0))

    deduped.sort(key=_growth_material_bucket_rank, reverse=True)
    store = {k: v for k, v in store.items() if k == champ_key}
    store[champ_key] = deduped[:cap]
    _save_growth_material_store(store)
    return True


def _filter_growth_material_bucket(
    anchor: list[int],
    bucket: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop side-broken growth material so locus probes do not re-evaluate regressions."""
    filtered: list[dict[str, Any]] = []
    for entry in bucket:
        if not isinstance(entry, dict):
            continue
        prog = entry.get("program")
        if not isinstance(prog, list):
            continue
        if _growth_material_capability_qualifies(anchor, list(prog)):
            filtered.append(entry)
    return filtered


def _load_growth_material_entries(
    champion: list[int],
) -> tuple[list[int], list[dict[str, Any]]]:
    anchor = _growth_material_storage_anchor(champion)
    store = _load_growth_material_store()
    champ_key = _growth_material_storage_key(champion)
    entries: list[dict[str, Any]] = []
    dirty = False
    bucket = store.get(champ_key)
    if isinstance(bucket, list):
        entries = _filter_growth_material_bucket(anchor, bucket)
        if len(entries) < len(bucket):
            store[champ_key] = entries
            dirty = True
    if not entries:
        for fallback_key, fallback in store.items():
            if fallback_key == champ_key or not isinstance(fallback, list):
                continue
            entries.extend(_filter_growth_material_bucket(anchor, fallback))
    if dirty:
        _save_growth_material_store(store)
    return anchor, entries


def collect_growth_material_parents(
    champion: list[int],
    *,
    max_parents: int = 16,
) -> list[list[int]]:
    """Longer held-out-safe programs recorded from live evaluations."""
    if not champion:
        return []
    anchor, entries = _load_growth_material_entries(champion)
    if not entries:
        return []
    champ_n = len(anchor)
    champ_t = tuple(int(x) for x in anchor)
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    try:
        anchor_prom = float(compact_l30_promotion_raw_fitness(list(anchor)))
    except (RuntimeError, TypeError, ValueError, OSError):
        anchor_prom = 0.0
    prom_floor = max(
        100.0,
        anchor_prom - float(os.environ.get("EE_GROWTH_MATERIAL_PROM_MARGIN", "2.0")),
    )
    from evolution.mutation_helpers import scan_floor_plateau_armed

    armed, floor_w = scan_floor_plateau_armed()
    at_scan_floor = bool(armed and champ_n == int(floor_w))
    from evolution.terminal_stagnation import curriculum_verify_mean

    anchor_l10_weak = float(curriculum_verify_mean(list(anchor), 10)) < 0.85 - 1e-6

    def _parent_rank(entry: dict) -> tuple[float, ...]:
        prog = entry.get("program")
        if not isinstance(prog, list):
            return (0.0, 0.0, 0)
        if anchor_l10_weak:
            return (
                float(curriculum_verify_mean(list(prog), 10)),
                float(entry.get("prom") or 0.0),
                float(int(entry.get("cycle") or 0)),
            )
        return (
            float(entry.get("prom") or 0.0),
            float(int(entry.get("cycle") or 0)),
        )

    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = {champ_t}
    for entry in sorted(entries, key=_parent_rank, reverse=True):
        entry_prom = float(entry.get("prom") or 0.0)
        prog = entry.get("program")
        if not isinstance(prog, list) or len(prog) <= champ_n:
            continue
        if not _growth_material_entry_passes_prom_floor(
            list(anchor), list(prog), entry_prom, prom_floor
        ):
            continue
        if at_scan_floor and entry_prom + 1e-6 < anchor_prom:
            if not _growth_material_l10_lift_qualifies(list(anchor), list(prog)):
                continue
        if not _growth_material_capability_qualifies(list(anchor), list(prog)):
            continue
        decoded = tuple(int(x) for x in prog)
        if decoded in seen:
            continue
        seen.add(decoded)
        out.append(list(prog))
        if len(out) >= max(1, int(max_parents)):
            break
    return out


def collect_growth_material_compress_parents(
    champion: list[int],
    *,
    max_parents: int = 16,
) -> list[list[int]]:
    """Growth-phase longer programs for compress breeding (54w@325.03 -> 50w promote)."""
    if not champion:
        return []
    anchor, entries = _load_growth_material_entries(champion)
    if not entries:
        return []
    champ_n = len(anchor)
    champ_t = tuple(int(x) for x in anchor)
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    try:
        anchor_prom = float(compact_l30_promotion_raw_fitness(list(anchor)))
    except (RuntimeError, TypeError, ValueError, OSError):
        anchor_prom = 0.0
    prom_floor = max(
        100.0,
        anchor_prom - float(os.environ.get("EE_GROWTH_MATERIAL_PROM_MARGIN", "2.0")),
    )
    from evolution.terminal_stagnation import curriculum_verify_mean

    anchor_l10_weak = float(curriculum_verify_mean(list(anchor), 10)) < 0.85 - 1e-6
    extra_words = int(os.environ.get("EE_GROWTH_COMPRESS_TARGET_EXTRA_WORDS", "4"))
    if anchor_l10_weak:
        extra_words = int(
            os.environ.get("EE_GROWTH_COMPRESS_TARGET_EXTRA_WORDS_WEAK_L10", "2")
        )
    target_len = champ_n + max(2, extra_words)
    candidates: list[tuple[float, int, list[int]]] = []
    for entry in entries:
        entry_prom = float(entry.get("prom") or 0.0)
        prog = entry.get("program")
        if not isinstance(prog, list) or len(prog) <= champ_n:
            continue
        if not _growth_material_entry_passes_prom_floor(
            list(anchor), list(prog), entry_prom, prom_floor
        ):
            continue
        if not _growth_material_capability_qualifies(list(anchor), list(prog)):
            continue
        decoded = tuple(int(x) for x in prog)
        if decoded == champ_t:
            continue
        candidates.append((entry_prom, int(entry.get("cycle") or 0), list(prog)))
    if not candidates:
        return []

    def _compress_rank(item: tuple[float, int, list[int]]) -> tuple:
        entry_prom, cycle, prog = item
        growth_phase = entry_prom + 1e-6 < anchor_prom
        at_champ_prom = entry_prom + 1e-6 >= anchor_prom
        if at_champ_prom:
            tier = 0
        elif growth_phase:
            tier = 1
        else:
            tier = 2
        l10 = float(curriculum_verify_mean(list(prog), 10)) if anchor_l10_weak else 0.0
        if anchor_l10_weak:
            return (
                -l10,
                tier,
                abs(len(prog) - target_len),
                -entry_prom,
                -cycle,
            )
        return (
            tier,
            abs(len(prog) - target_len),
            -entry_prom,
            -cycle,
        )

    candidates.sort(key=_compress_rank)
    cap = max(1, int(max_parents))
    if len(candidates) <= cap or anchor_l10_weak:
        out: list[list[int]] = []
        seen: set[tuple[int, ...]] = {champ_t}
        for _, _, prog in candidates:
            key = tuple(int(x) for x in prog)
            if key in seen:
                continue
            seen.add(key)
            out.append(list(prog))
            if len(out) >= cap:
                break
        return out
    from evolution.mutation_helpers import select_diverse_growth_material_programs

    progs = [list(prog) for _, _, prog in candidates]
    return select_diverse_growth_material_programs(
        list(anchor), progs, max_select=cap
    )
