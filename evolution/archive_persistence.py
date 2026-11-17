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


def _newest_sidecar_cycle(archive_dir: str) -> int:
    """Newest cycle on disk (0 when there is none) — the reference to use when
    latest.json is unusable, mirroring scripts/prune_agent_artifacts.py."""
    best = 0
    try:
        names = os.listdir(archive_dir)
    except OSError:
        return 0
    for name in names:
        cycle = _sidecar_cycle_from_name(name)
        if cycle is not None and cycle > best:
            best = cycle
    return best


def load_archive_sidecar(project_root: str, cycle: int | None = None) -> dict[str, Any] | None:
    d = archive_dir(project_root)
    if cycle is not None:
        blob = _read_sidecar_file(os.path.join(d, f"cycle_{cycle:05d}.json"))
        if blob is not None:
            return blob
    latest = os.path.join(d, "latest.json")
    # latest.json missing, corrupt or hand-edited must not crash resume and must
    # not silently restore nothing while sidecars sit on disk: drop the pointer
    # and let the fallback below work from the disk instead.
    try:
        with open(latest, encoding="utf-8") as f:
            meta = json.load(f)
        path = meta.get("path")
        ref_cycle = int(meta.get("cycle", 0) or 0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError):
        path, ref_cycle = None, 0
    # Non-dict target (a sidecar holding a JSON array) is unusable too: it must
    # not escape this function, or callers like cap_strategy_archive_state
    # crash on `.get`.
    loaded = _read_sidecar_file(path) if isinstance(path, str) else None
    # Trust a healthy live sidecar. The old 4000-record floor treated a
    # capped terminal archive (1500–3200) as "too thin" and restored a
    # 4890+ sidecar, undoing EE_TERMINAL_STRATEGY_ARCHIVE_CAP.
    min_ok = _min_records()
    if loaded is not None and _strategy_count(loaded) >= min_ok:
        return loaded
    fallback = _best_real_sidecar(d, ref_cycle if ref_cycle > 0 else _newest_sidecar_cycle(d))
    if fallback is not None and _strategy_count(fallback) > _strategy_count(loaded):
        return fallback
    return loaded


def _sidecar_cycle_from_name(name: str) -> int | None:
    """Parse `cycle_<n>.json` as a whole number (5-, 6-digit or unpadded)."""
    if not name.startswith("cycle_") or not name.endswith(".json"):
        return None
    digits = name[6:-5]
    if not digits.isascii() or not digits.isdigit():
        return None
    return int(digits)


def _env_int(name: str, default: int) -> int:
    """Env-var int with a safe default — a malformed value must not crash resume."""
    try:
        return int(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _read_sidecar_file(path: str | None) -> dict[str, Any] | None:
    """Sidecar blob as a dict, or None when it is missing, unreadable, not JSON,
    or not a JSON object. The single door every sidecar load goes through: a
    corrupt file degrades to "no archive from here", never to a crash."""
    if not isinstance(path, str) or not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return None
    return blob if isinstance(blob, dict) else None


def _min_records() -> int:
    """Record floor for a sidecar to be restorable (env-tunable, default 400)."""
    return _env_int("EE_ARCHIVE_SIDECAR_MIN_RECORDS", 400)


def _best_real_sidecar(archive_dir: str, ref_cycle: int) -> dict[str, Any] | None:
    """Pick the fallback sidecar when latest.json's own file is unusable.

    Ranking, in order:
      1. The NEWEST sidecar (highest cycle) with at least
         EE_ARCHIVE_SIDECAR_MIN_RECORDS strategy records. Recency, not record
         count, says which archive the live run last wrote: a terminal-capped
         live sidecar (1500 records) must beat an older uncapped one (4890).
         The floor only decides which candidates are ELIGIBLE — a `loaded`
         blob under the floor is still replaced by an eligible candidate (see
         load_archive_sidecar), which is the same stance
         test_load_prefers_richer_nearby_sidecar encodes.
      2. The RICHEST sidecar, only when nothing on disk clears the floor.

    Richest-wins alone rewound resume by two weeks in the 2026-09-30 incident:
    latest.json dangled at the deleted cycle_07602.json while only the Sep-14
    sidecars (cycle_06991..07000, up to 4890 records) survived, all inside the
    `ref + 50` window, so the 16-day-old archive was restored instead of a
    healthy live one.

    "Nearby" still means at most 50 cycles above the reference — the margin
    scripts/prune_agent_artifacts.py calls SIDECAR_FIXTURE_MARGIN (keep the two
    in sync) — which is what keeps fixture sidecars (far ahead of the run) out
    of the ranking. A reference read from latest.json is the live run's own
    cycle; when latest.json is unreadable the caller passes the newest cycle on
    disk instead, so in that case a fixture planted far ahead would be both the
    window and the winner. That needs a corrupt latest.json AND a fixture on
    disk at once. scripts/prune_agent_artifacts.py::_reference_sidecar_cycle
    derives its reference the same way (latest.json, else newest on disk) and
    carries an equally accepted residual risk for it, so the two stay
    consistent; the historical static EE_ARCHIVE_SIDECAR_MAX_CYCLE cap only
    applies when the directory holds no sidecar at all, where the window cannot
    admit anything anyway.
    """
    min_ok = _min_records()
    cap = _env_int("EE_ARCHIVE_SIDECAR_MAX_CYCLE", 7000)
    limit = (ref_cycle + 50) if ref_cycle > 0 else cap
    newest_ok: tuple[int, dict[str, Any]] | None = None  # (cycle, blob) over the floor
    richest: tuple[int, dict[str, Any]] | None = None  # (records, blob) regardless
    try:
        names = os.listdir(archive_dir)
    except OSError:
        return None
    for name in names:
        cycle = _sidecar_cycle_from_name(name)
        if cycle is None:
            continue
        if cycle > limit:
            continue
        blob = _read_sidecar_file(os.path.join(archive_dir, name))
        if blob is None:
            continue
        n = _strategy_count(blob)
        if richest is None or n > richest[0]:
            richest = (n, blob)
        if n >= min_ok and (newest_ok is None or cycle > newest_ok[0]):
            newest_ok = (cycle, blob)
    if newest_ok is not None:
        return newest_ok[1]
    if richest is not None:
        return richest[1]
    return None


def load_archive_from_result(project_root: str, result: dict[str, Any]) -> dict[str, Any] | None:
    """Load archive from inline blob (legacy) or sidecar path."""
    inline = result.get("archive_state")
    if isinstance(inline, dict) and inline:
        return inline
    blob = _read_sidecar_file(result.get("archive_state_path"))
    if blob is not None:
        return blob
    cycle = result.get("cycle")
    if isinstance(cycle, int):
        return load_archive_sidecar(project_root, cycle)
    return load_archive_sidecar(project_root)


def slim_result_for_jsonl(result: dict[str, Any]) -> dict[str, Any]:
    """Remove embedded archive blob; keep metrics + sidecar path."""
    out = dict(result)
    out.pop("archive_state", None)
    return out


def _strategy_count(archive_state: Any) -> int:
    """Strategy-record count; anything that is not a dict counts as 0 records.

    File-backed blobs are already filtered by `_read_sidecar_file`, but legacy
    inline `archive_state` values come straight out of the jsonl, so this stays
    total rather than assuming the declared dict type."""
    if not isinstance(archive_state, dict):
        return 0
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
