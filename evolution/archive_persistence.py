"""Sidecar persistence for evolution archives — keeps multi_cycle_results.jsonl lean."""

from __future__ import annotations

import json
import os
from typing import Any


def archive_dir(project_root: str) -> str:
    path = os.path.join(project_root, ".agents", "archive_state")
    os.makedirs(path, exist_ok=True)
    return path


def save_archive_sidecar(project_root: str, cycle: int, archive_state: dict[str, Any]) -> str:
    """Write archive blob to sidecar; return relative path for jsonl reference."""
    d = archive_dir(project_root)
    path = os.path.join(d, f"cycle_{cycle:05d}.json")
    latest = os.path.join(d, "latest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(archive_state, f)
    with open(latest, "w", encoding="utf-8") as f:
        json.dump({"cycle": cycle, "path": path, "records": _strategy_count(archive_state)}, f, indent=2)
    return path


def load_archive_sidecar(project_root: str, cycle: int | None = None) -> dict[str, Any] | None:
    d = archive_dir(project_root)
    if cycle is not None:
        path = os.path.join(d, f"cycle_{cycle:05d}.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    latest = os.path.join(d, "latest.json")
    if not os.path.isfile(latest):
        return None
    with open(latest, encoding="utf-8") as f:
        meta = json.load(f)
    path = meta.get("path")
    loaded: dict[str, Any] | None = None
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    # Trust a healthy live sidecar. The old 4000-record floor treated a
    # capped terminal archive (1500–3200) as "too thin" and restored a
    # 4890+ sidecar, undoing EE_TERMINAL_STRATEGY_ARCHIVE_CAP.
    min_ok = int(os.environ.get("EE_ARCHIVE_SIDECAR_MIN_RECORDS", "400"))
    if loaded is not None and _strategy_count(loaded) >= min_ok:
        return loaded
    fallback = _richest_real_sidecar(d, int(meta.get("cycle", 0) or 0))
    if fallback is not None and _strategy_count(fallback) > _strategy_count(loaded or {}):
        return fallback
    return loaded


def _richest_real_sidecar(archive_dir: str, ref_cycle: int) -> dict[str, Any] | None:
    """Prefer a nearby sidecar with the most strategy records (survives prune/fixtures)."""
    cap = int(os.environ.get("EE_ARCHIVE_SIDECAR_MAX_CYCLE", "7000"))
    best: dict[str, Any] | None = None
    best_n = -1
    try:
        names = os.listdir(archive_dir)
    except OSError:
        return None
    for name in names:
        if not name.startswith("cycle_") or not name.endswith(".json"):
            continue
        try:
            cycle = int(name[6:11])
        except ValueError:
            continue
        if cycle > cap:
            continue
        if ref_cycle > 0 and cycle > ref_cycle + 50:
            continue
        path = os.path.join(archive_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        n = _strategy_count(blob)
        if n > best_n:
            best_n = n
            best = blob
    return best


def load_archive_from_result(project_root: str, result: dict[str, Any]) -> dict[str, Any] | None:
    """Load archive from inline blob (legacy) or sidecar path."""
    if result.get("archive_state"):
        return result["archive_state"]
    ref = result.get("archive_state_path")
    if ref and os.path.isfile(ref):
        with open(ref, encoding="utf-8") as f:
            return json.load(f)
    cycle = result.get("cycle")
    if isinstance(cycle, int):
        return load_archive_sidecar(project_root, cycle)
    return load_archive_sidecar(project_root)


def slim_result_for_jsonl(result: dict[str, Any]) -> dict[str, Any]:
    """Remove embedded archive blob; keep metrics + sidecar path."""
    out = dict(result)
    out.pop("archive_state", None)
    return out


def _strategy_count(archive_state: dict[str, Any]) -> int:
    strat = archive_state.get("strategy") or {}
    if isinstance(strat, dict):
        recs = strat.get("records") or strat.get("_records")
        if isinstance(recs, list):
            return len(recs)
    return 0


def cap_strategy_archive_state(
    archive_state: dict[str, Any] | None,
    cap: int,
) -> dict[str, Any] | None:
    """Keep only the newest strategy records — terminal fast path stays O(cap)."""
    if archive_state is None or cap <= 0:
        return archive_state
    strat = archive_state.get("strategy")
    if not isinstance(strat, dict):
        return archive_state
    records = strat.get("records")
    if not isinstance(records, list) or len(records) <= cap:
        return archive_state
    trimmed = dict(archive_state)
    trimmed["strategy"] = {**strat, "records": records[-cap:]}
    return trimmed
