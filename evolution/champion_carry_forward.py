"""Persist and restore champion programs across evolution cycles."""

from __future__ import annotations

import json
import os
from typing import Any

from evolution.curriculum_advance import depth_carry_verify_floor
from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS


def carry_forward_path(project_root: str) -> str:
    return os.path.join(project_root, ".agents", "champion_carry_forward.json")


def _carry_rank(payload: dict[str, Any]) -> tuple:
    return (
        float(payload.get("verify_score", 0.0)),
        _carry_capability_score(payload),
        int(payload.get("cycle", 0)),
    )


def _carry_capability_score(payload: dict[str, Any]) -> float:
    """Primary capability score for carry-forward ties (promotion_raw, not reporting)."""
    from evolution.fitness_helpers import row_capability_score

    return float(row_capability_score(payload))


def _read_store(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    levels = data.get("levels")
    if isinstance(levels, dict):
        return {str(k): v for k, v in levels.items() if isinstance(v, dict)}
    if isinstance(data, dict) and data.get("champion_program"):
        lvl = str(int(data.get("level", 0)))
        return {lvl: data} if lvl != "0" else {}
    return {}


def _write_store(path: str, levels: dict[str, dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"levels": levels}, f, indent=2)


def persist_carry_forward_champion(project_root: str, evo_result: dict[str, Any]) -> None:
    """Save clean champion when cycle ends at nudge quality or better."""
    level = int(evo_result.get("final_level", 0))
    verify = float(evo_result.get("champion_verify_score", 0.0))
    words = int(evo_result.get("champion_words", 0))
    fitness = float(evo_result.get("final_fitness", 0.0))
    min_verify = depth_carry_verify_floor(level) if level >= 20 else 0.72
    if level < 20 or fitness <= -25.0:
        return
    from evolution.champion_bloat_guard import level_word_cap

    cap = level_word_cap(level) or (36 if level >= 21 else 40)
    if words > cap or words < 12:
        return
    prog = evo_result.get("champion_program")
    if not isinstance(prog, list) or not prog:
        return
    heldout = None
    decoded: list[int] | None = None
    if level >= 30 and words <= L30_COMPACT_MAX_WORDS:
        decoded = _decode_program(prog)
        if decoded is None:
            return
        from evolution.terminal_stagnation import (
            compact_curriculum_side_score,
            curriculum_verify_mean,
            heldout_l30_verify_mean,
        )

        heldout = float(heldout_l30_verify_mean(decoded))
        if verify >= 0.99 - 1e-6:
            if heldout + 1e-6 < 0.99:
                return
            verify = 0.989
        elif verify < min_verify:
            return
        payload_l10 = float(curriculum_verify_mean(decoded, 10))
        payload_side = float(compact_curriculum_side_score(decoded))
    elif verify < min_verify:
        return
    payload = {
        "level": level,
        "verify_score": verify,
        "fitness": fitness,
        "words": words,
        "cycle": int(evo_result.get("cycle", 0)),
        "champion_program": prog,
    }
    task_fit = evo_result.get("task_fitness")
    if task_fit is not None:
        try:
            payload["task_fitness"] = float(task_fit)
        except (TypeError, ValueError):
            pass
    prom_raw = evo_result.get("promotion_raw_fitness")
    if prom_raw is not None:
        try:
            payload["promotion_raw_fitness"] = float(prom_raw)
        except (TypeError, ValueError):
            pass
    elif decoded is not None and level >= 30 and words <= L30_COMPACT_MAX_WORDS and verify >= 0.99 - 1e-6:
        from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

        payload["promotion_raw_fitness"] = float(compact_l30_promotion_raw_fitness(decoded))
    if heldout is not None:
        payload["heldout_l30"] = heldout
        payload["heldout_l10"] = payload_l10
        payload["heldout_side"] = payload_side
    path = carry_forward_path(project_root)
    levels = _read_store(path)
    key = str(level)
    existing = levels.get(key)
    if existing is not None:
        existing = _enrich_payload_task_fitness(project_root, existing)
        if not _carry_payload_beats(existing, payload):
            return
    levels[key] = payload
    _write_store(path, levels)


def _carry_payload_beats(existing: dict[str, Any], payload: dict[str, Any]) -> bool:
    """True if payload should replace existing. Compact L30 ranks by held-out CPU."""
    if int(payload.get("level", 0)) >= 30 and int(payload.get("words", 0)) <= L30_COMPACT_MAX_WORDS:
        new_h = payload.get("heldout_l30")
        old_h = existing.get("heldout_l30")
        if new_h is not None:
            if old_h is None:
                old_prog = _decode_program(existing.get("champion_program", []))
                if old_prog is not None:
                    from evolution.terminal_stagnation import heldout_l30_verify_mean

                    old_h = float(heldout_l30_verify_mean(old_prog))
                else:
                    old_h = 0.0
            if float(new_h) > float(old_h) + 1e-6:
                return True
            if float(new_h) + 1e-6 < float(old_h):
                return False
            from evolution.terminal_stagnation import compact_curriculum_side_score

            new_prog = _decode_program(payload.get("champion_program", []))
            old_prog = _decode_program(existing.get("champion_program", []))
            new_side = float(compact_curriculum_side_score(new_prog)) if new_prog else 0.0
            old_side = float(compact_curriculum_side_score(old_prog)) if old_prog else 0.0
            if float(new_side) > float(old_side) + 0.01:
                return True
            if float(new_side) + 0.01 < float(old_side):
                return False
            new_words = int(payload.get("words", 0))
            old_words = int(existing.get("words", 0))
            if new_words < old_words:
                return True
            if new_words == old_words and _carry_capability_score(payload) > _carry_capability_score(
                existing
            ) + 1e-6:
                return True
            return False
    return _carry_rank(payload) > _carry_rank(existing)


def _decode_program(prog: list[Any]) -> list[int] | None:
    if not isinstance(prog, list) or len(prog) < 12:
        return None
    words: list[int] = []
    for item in prog:
        if isinstance(item, str):
            words.append(int(item, 16))
        else:
            words.append(int(item))
    return words


def _payload_for_level(
    levels: dict[str, dict[str, Any]], level: int
) -> dict[str, Any] | None:
    payload = levels.get(str(level))
    if not payload:
        return None
    stored_level = int(payload.get("level", level))
    verify = float(payload.get("verify_score", 0.0))
    fitness = float(payload.get("fitness", 0.0))
    min_verify = depth_carry_verify_floor(stored_level) if stored_level >= 20 else 0.72
    if verify < min_verify:
        return None
    if stored_level >= 30 and verify >= 0.99 - 1e-6 and fitness < 150.0:
        return None
    if (
        stored_level >= 30
        and int(payload.get("words", 0)) <= L30_COMPACT_MAX_WORDS
        and verify >= 0.989 - 1e-6
    ):
        return None
    if not _decode_program(payload.get("champion_program", [])):
        return None
    return payload


def load_carry_forward_start_level(project_root: str, sustained_cap: int = 0) -> int:
    """Best curriculum start level implied by persisted carry-forward champions."""
    from evolution.progress_controller import ProgressController

    levels = _read_store(carry_forward_path(project_root))
    best = 0
    for payload in levels.values():
        lvl = int(payload.get("level", 0))
        verify = float(payload.get("verify_score", 0.0))
        carried = ProgressController.carry_forward_start_level(lvl, verify, sustained_cap)
        best = max(best, carried)
    return best


def _payload_from_jsonl_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a multi_cycle_results row into carry-forward payload shape."""
    verify = float(row.get("champion_verify_score", 0.0))
    words = int(row.get("champion_words", 0))
    if verify >= 0.99 - 1e-6:
        verify = 0.989
    payload: dict[str, Any] = {
        "level": int(row.get("final_level", 0)),
        "verify_score": verify,
        "fitness": float(row.get("final_fitness", 0.0)),
        "words": words,
        "cycle": int(row.get("cycle", 0)),
        "champion_program": list(row.get("champion_program") or []),
    }
    task_fit = row.get("task_fitness")
    if task_fit is not None:
        try:
            payload["task_fitness"] = float(task_fit)
        except (TypeError, ValueError):
            pass
    prom_raw = row.get("promotion_raw_fitness")
    if prom_raw is not None:
        try:
            payload["promotion_raw_fitness"] = float(prom_raw)
        except (TypeError, ValueError):
            pass
    decoded = _decode_program(payload.get("champion_program", []))
    if decoded is not None and int(payload.get("level", 0)) >= 30 and words <= L30_COMPACT_MAX_WORDS:
        from evolution.terminal_stagnation import (
            compact_curriculum_side_score,
            curriculum_verify_mean,
            heldout_l30_verify_mean,
        )

        payload["heldout_l30"] = float(heldout_l30_verify_mean(decoded))
        payload["heldout_l10"] = float(curriculum_verify_mean(decoded, 10))
        payload["heldout_side"] = float(compact_curriculum_side_score(decoded))
    return payload


def _iter_compact_l30_task_rows(results_file: str, level: int = 30):
    if not os.path.isfile(results_file):
        return
    lvl = int(level)
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or "error" in row:
                continue
            if int(row.get("final_level", 0)) != lvl:
                continue
            words = int(row.get("champion_words", 0))
            if words < 12 or words > L30_COMPACT_MAX_WORDS:
                continue
            verify = float(row.get("champion_verify_score", 0.0))
            if verify < 0.99 - 1e-6:
                continue
            task = row.get("task_fitness")
            if task is None:
                continue
            try:
                float(task)
            except (TypeError, ValueError):
                continue
            prog = row.get("champion_program")
            if not isinstance(prog, list) or not prog:
                continue
            yield row


def _enrich_payload_task_fitness(
    project_root: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """Backfill task_fitness for legacy carry rows from the matching jsonl cycle."""
    if payload.get("task_fitness") is not None:
        return payload
    cycle = int(payload.get("cycle", 0))
    if cycle <= 0:
        return payload
    results = os.path.join(project_root, "multi_cycle_results.jsonl")
    if not os.path.isfile(results):
        return payload
    with open(results, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if int(row.get("cycle", -1)) != cycle:
                continue
            task = row.get("task_fitness")
            if task is None:
                return payload
            try:
                enriched = dict(payload)
                enriched["task_fitness"] = float(task)
                return enriched
            except (TypeError, ValueError):
                return payload
    return payload


def find_best_compact_l30_task_row(
    results_file: str,
    level: int = 30,
    *,
    existing: dict[str, Any] | None = None,
    min_cycle: int = 0,
) -> dict[str, Any] | None:
    """Best task_fitness compact L30 row newer than *min_cycle* that beats *existing*."""
    best_row: dict[str, Any] | None = None
    best_payload: dict[str, Any] | None = None
    floor_cycle = int(min_cycle)
    for row in _iter_compact_l30_task_rows(results_file, level):
        cycle = int(row.get("cycle", -1))
        if cycle <= floor_cycle:
            continue
        overlay = _payload_from_jsonl_row(row)
        if existing is not None and not _carry_payload_beats(existing, overlay):
            continue
        if best_payload is None or _carry_payload_beats(best_payload, overlay):
            best_row = row
            best_payload = overlay
    return best_row


def persist_carry_forward_from_engine(project_root: str, engine: Any) -> None:
    """Persist the live engine champion after a terminal promotion."""
    if engine.champion is None:
        return
    level = int(getattr(engine, "champion_level", 0))
    words = len(engine.champion)
    if level < 30 or words < 12 or words > L30_COMPACT_MAX_WORDS:
        return
    verify = float(getattr(engine, "_champion_verify_score", 0.0))
    if verify < 0.99 - 1e-6:
        return
    cycle = int(getattr(engine, "_cycle_number", 0))
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    prom_raw = float(compact_l30_promotion_raw_fitness(list(engine.champion)))
    evo_result = {
        "cycle": cycle,
        "final_level": level,
        "champion_verify_score": verify,
        "champion_words": words,
        "final_fitness": float(getattr(engine, "champion_fitness", 0.0)),
        "task_fitness": float(getattr(engine, "_task_fitness", 0.0)),
        "promotion_raw_fitness": prom_raw,
        "champion_program": [f"0x{w:08X}" for w in engine.champion],
    }
    persist_carry_forward_champion(project_root, evo_result)


def load_carry_forward_champion(
    project_root: str, start_level: int
) -> tuple[list[int], str] | None:
    """Load champion for start level, or prior level on depth-nudge (+1)."""
    path = carry_forward_path(project_root)
    levels = _read_store(path)
    if not levels:
        return None
    start = int(start_level)
    candidates = (start,) if start >= 30 else (start, start - 1)
    for candidate in candidates:
        if candidate < 20:
            continue
        if candidate == start - 1 and start < 21:
            continue
        payload = _payload_for_level(levels, candidate)
        if payload is None:
            continue
        payload = _enrich_payload_task_fitness(project_root, payload)
        words = _decode_program(payload.get("champion_program", []))
        if words is None:
            continue
        cycle = payload.get("cycle", "?")
        stored_level = int(payload.get("level", candidate))
        return words, f"carry_forward_cycle_{cycle}_L{stored_level}"
    if start >= 30:
        from evolution.champion_rollback import find_best_honest_compact_l30

        results = os.path.join(project_root, "multi_cycle_results.jsonl")
        honest = find_best_honest_compact_l30(results, start)
        if honest:
            words = _decode_program(honest.get("champion_program", []))
            if words is not None:
                return words, f"honest_compact_cycle_{honest['cycle']}"
    return None
