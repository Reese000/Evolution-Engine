#!/usr/bin/env python3
"""Prune stale daemon batch logs and other .agents artifacts to avoid EMFILE."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(PROJECT_ROOT, ".agents")

# A sidecar more than this many cycles AHEAD of the live run cannot belong to
# the live run: save_archive_sidecar() writes the sidecar and latest.json
# together, so the reference never trails the newest file by this much.
SIDECAR_FIXTURE_MARGIN = 50


def prune_batch_logs(agents_dir: str, keep: int = 30) -> int:
    """Delete oldest batch_*.log files, keeping the newest `keep` entries."""
    if keep < 1:
        return 0
    if not os.path.isdir(agents_dir):
        return 0
    paths: list[tuple[float, str]] = []
    for name in os.listdir(agents_dir):
        if not name.startswith("batch_") or not name.endswith(".log"):
            continue
        path = os.path.join(agents_dir, name)
        try:
            paths.append((os.path.getmtime(path), path))
        except OSError:
            continue
    if len(paths) <= keep:
        return 0
    paths.sort(key=lambda item: item[0])
    removed = 0
    for _, path in paths[: len(paths) - keep]:
        try:
            os.remove(path)
            removed += 1
        except OSError:
            continue
    return removed


def _sidecar_cycle_from_name(name: str) -> int | None:
    """Parse `cycle_<n>.json` as a whole number (5-, 6-digit or unpadded)."""
    if not name.startswith("cycle_") or not name.endswith(".json"):
        return None
    digits = name[6:-5]
    if not digits.isascii() or not digits.isdigit():
        return None
    return int(digits)


def _reference_sidecar_cycle(archive_dir: str) -> int | None:
    """Live run cycle, used to tell live sidecars from fixture sidecars.

    `latest.json` is rewritten alongside every sidecar by
    evolution.archive_persistence.save_archive_sidecar(), so its cycle is the
    authoritative live cycle; when it is missing or unreadable fall back to the
    newest sidecar on disk (never guess downward — that is what deleted live
    history).

    The former EE_ARCHIVE_SIDECAR_MAX_CYCLE (7000) cap must NOT bound this
    value: the live run crossed 7000 on 2026-09-14, and capping the reference
    there made every newer — i.e. every live — sidecar look like a test fixture.
    """
    latest = os.path.join(archive_dir, "latest.json")
    if os.path.isfile(latest):
        try:
            with open(latest, encoding="utf-8") as f:
                meta = json.load(f)
            cycle = int(meta.get("cycle", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError):
            cycle = 0
        # Trust any positive cycle, even one far ahead of what is on disk: after
        # a bad prune latest.json legitimately points past every surviving
        # sidecar (observed: cycle 7602 vs newest file 7000), and bounding this
        # with a static cap is exactly what deleted live history. Accepted
        # residual risk: an absurd latest.json cycle would let fixture sidecars
        # compete for the newest-N slots — still bounded at `keep` survivors,
        # and locally indistinguishable from the incident case above.
        if cycle > 0:
            return cycle
    found: list[int] = []
    try:
        for name in os.listdir(archive_dir):
            cycle = _sidecar_cycle_from_name(name)
            if cycle is not None:
                found.append(cycle)
    except OSError:
        return None
    return max(found) if found else None


def _fixture_sidecar_paths(archive_dir: str, ref_cycle: int | None) -> list[str]:
    """Sidecars further than SIDECAR_FIXTURE_MARGIN cycles ahead of the live run.

    With no reference (empty/corrupt directory) nothing is claimed to be a
    fixture: deleting on a guess is how live history was lost. This boundary is
    the same one prune_archive_sidecars() uses to skip candidates, so no sidecar
    is ever selected by both paths: parseable sidecars at or below the boundary
    are decided by the newest-N cut (or kept when latest.json points at them),
    while unparseable `cycle_*.json` names are skipped by both paths, as before
    — save_archive_sidecar() never writes such names.
    """
    if ref_cycle is None:
        return []
    limit = ref_cycle + SIDECAR_FIXTURE_MARGIN
    try:
        names = sorted(os.listdir(archive_dir))
    except OSError:
        return []
    paths: list[str] = []
    for name in names:
        cycle = _sidecar_cycle_from_name(name)
        if cycle is None or cycle <= limit:
            continue
        paths.append(os.path.join(archive_dir, name))
    return paths


def _latest_protected_paths(archive_dir: str) -> set[str]:
    """Norm-cased path of the sidecar latest.json points at.

    Empty when latest.json is absent, unreadable, or points at a missing file —
    that is already the damaged state this prune must not make worse.
    """
    latest = os.path.join(archive_dir, "latest.json")
    if not os.path.isfile(latest):
        return set()
    try:
        with open(latest, encoding="utf-8") as f:
            meta = json.load(f)
        path = meta.get("path")
        if isinstance(path, str) and os.path.isfile(path):
            return {os.path.normcase(os.path.abspath(path))}
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    return set()


def purge_fixture_archive_sidecars(agents_dir: str) -> int:
    """Delete sidecars far ahead of the live run (test fixtures); count them.

    Deliberately kept as a standalone entry point for a fixtures-only purge —
    `prune_archive_sidecars` runs the same selection through
    `plan_archive_sidecar_deletions` so a single pass plans both kinds of
    deletion and counts them together.
    """
    archive_dir = os.path.join(agents_dir, "archive_state")
    if not os.path.isdir(archive_dir):
        return 0
    ref_cycle = _reference_sidecar_cycle(archive_dir)
    protected = _latest_protected_paths(archive_dir)
    removed = 0
    for path in _fixture_sidecar_paths(archive_dir, ref_cycle):
        if os.path.normcase(os.path.abspath(path)) in protected:
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError:
            continue
    return removed


def plan_archive_sidecar_deletions(agents_dir: str, keep: int = 10) -> list[str]:
    """Every sidecar `prune_archive_sidecars` would delete, in deletion order.

    The newest `keep` cycles are chosen by cycle NUMBER (name order is
    irrelevant: `cycle_9999` sorts after `cycle_10000` as text), the sidecar
    latest.json points at is never selected — neither by the newest-N cut nor by
    the fixture purge — and fixture matches are exactly the set the newest-N cut
    skips, so a parseable sidecar is either deleted here or survives (newest-N
    or protection). `--dry-run` shares this planner so the printed count and the
    executed count cannot drift apart.
    """
    if keep < 1:
        return []
    archive_dir = os.path.join(agents_dir, "archive_state")
    if not os.path.isdir(archive_dir):
        return []
    ref_cycle = _reference_sidecar_cycle(archive_dir)
    protected = _latest_protected_paths(archive_dir)

    # Fixtures go through `protected` as well: whatever latest.json points at
    # stays on disk even if its `path` disagrees with its `cycle` number.
    deletions = [
        path
        for path in _fixture_sidecar_paths(archive_dir, ref_cycle)
        if os.path.normcase(os.path.abspath(path)) not in protected
    ]

    limit = None if ref_cycle is None else ref_cycle + SIDECAR_FIXTURE_MARGIN
    try:
        names = os.listdir(archive_dir)
    except OSError:
        return deletions
    paths: list[tuple[int, str]] = []
    for name in names:
        cycle = _sidecar_cycle_from_name(name)
        if cycle is None:
            continue
        if limit is not None and cycle > limit:
            continue  # selected as a fixture above
        paths.append((cycle, os.path.join(archive_dir, name)))

    if len(paths) > keep:
        paths.sort(key=lambda item: item[0])  # ascending cycle = oldest first
        for _, path in paths[: len(paths) - keep]:
            if os.path.normcase(os.path.abspath(path)) in protected:
                continue
            deletions.append(path)
    return deletions


def prune_archive_sidecars(agents_dir: str, keep: int = 10) -> int:
    """Delete the oldest archive_state/cycle_*.json sidecars, keeping the
    newest `keep` cycles by cycle number.

    Returns the number of files actually deleted, fixture purge included — that
    count used to be discarded, so a restart printed "0 archive sidecars
    removed" while a batch of live sidecars vanished.
    """
    removed = 0
    for path in plan_archive_sidecar_deletions(agents_dir, keep):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            continue
    return removed


def prune_all(
    agents_dir: str,
    *,
    batch_keep: int = 30,
    archive_keep: int = 10,
) -> dict[str, int]:
    """Prune batch logs and archive sidecars; return counts removed."""
    return {
        "batch_logs": prune_batch_logs(agents_dir, keep=batch_keep),
        "archive_sidecars": prune_archive_sidecars(agents_dir, keep=archive_keep),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune evolution daemon artifacts")
    parser.add_argument("--keep", type=int, default=int(os.environ.get("EE_BATCH_LOG_KEEP", "30")))
    parser.add_argument(
        "--archive-keep",
        type=int,
        default=int(os.environ.get("EE_ARCHIVE_SIDECAR_KEEP", "10")),
        help="Keep newest N archive_state/cycle_*.json sidecars (latest.json always kept)",
    )
    parser.add_argument("--agents", default=AGENTS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        batch_count = 0
        if args.keep >= 1 and os.path.isdir(args.agents):
            names = [n for n in os.listdir(args.agents) if n.startswith("batch_") and n.endswith(".log")]
            # prune_batch_logs() refuses keep < 1, so it deletes nothing there.
            batch_count = max(0, len(names) - args.keep)
        # Same planner the real prune executes, so the prediction is exact.
        archive_count = len(plan_archive_sidecar_deletions(args.agents, args.archive_keep))
        print(
            f"Would remove {batch_count} batch logs (keep {args.keep}) "
            f"and {archive_count} archive sidecars (keep {args.archive_keep})"
        )
        return 0
    counts = prune_all(args.agents, batch_keep=args.keep, archive_keep=args.archive_keep)
    print(
        f"Removed {counts['batch_logs']} batch logs (kept newest {args.keep}) "
        f"and {counts['archive_sidecars']} archive sidecars (kept newest {args.archive_keep})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
