#!/usr/bin/env python3
"""data_rotation.py — Production data rotation for checkpoints, recurrent data, archives, and reports.

Manages disk usage by rotating out old training artifacts while preserving
sacred files (best checkpoint, current checkpoint, live report). Every
deletion is wrapped in try/except so a single failure never halts the
rotation pass.

Usage:
    from data_rotation import run_all_rotations, rotate_checkpoints
    summary = run_all_rotations()
    # {'checkpoints': 3, 'recurrent': 1, 'archives': 2, 'reports': 5}

    # Or rotate individually with custom retention
    deleted = rotate_checkpoints(keep_n=10)
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from paths import M1_CHECKPOINTS, M1_RECURRENT, PROJECT_ROOT

# ── Constants ─────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT_KEEP: int = 5
DEFAULT_RECURRENT_MAX_MB: int = 100
DEFAULT_ARCHIVE_KEEP: int = 20
DEFAULT_REPORT_KEEP: int = 50
DEFAULT_RECURRENT_BACKUPS: int = 3

# Files that must never be deleted regardless of retention policy
_SACRED_CHECKPOINT_FILES: frozenset[str] = frozenset(
    {
        "checkpoint_best.pt",
        "checkpoint_current.pt",
    }
)

# Live report file that must never be deleted
_LIVE_REPORT_FILE: str = "progress_report.json"

# Timestamp format used for rotated recurrent data backups
_TIMESTAMP_FMT: str = "%Y%m%d_%H%M%S"

logger = logging.getLogger("ee.data_rotation")


# ── Checkpoint Rotation ──────────────────────────────────────────────


def rotate_checkpoints(keep_n: int = DEFAULT_CHECKPOINT_KEEP) -> list[str]:
    """Remove old checkpoint files, preserving sacred and most-recent files.

    Scans ``M1_CHECKPOINTS`` for ``.pt`` files, excludes sacred files
    (``checkpoint_best.pt``, ``checkpoint_current.pt``), and deletes the
    oldest files beyond ``keep_n``.  ``.regressed`` files are prioritized
    for deletion — they are removed first among the candidates.

    Args:
        keep_n: Number of non-sacred checkpoint files to retain.

    Returns:
        List of absolute paths that were deleted.
    """
    deleted: list[str] = []
    ckpt_dir = Path(M1_CHECKPOINTS)

    if not ckpt_dir.is_dir():
        logger.warning("Checkpoint directory does not exist: %s", ckpt_dir)
        return deleted

    pt_files: list[Path] = []
    for entry in ckpt_dir.iterdir():
        if not entry.is_file():
            continue
        # Match .pt files and .pt.regressed files (regression backups)
        if not (entry.name.endswith(".pt") or entry.name.endswith(".pt.regressed")):
            continue
        if entry.name in _SACRED_CHECKPOINT_FILES:
            continue
        pt_files.append(entry)

    if len(pt_files) <= keep_n:
        logger.info("Checkpoint count (%d) <= keep_n (%d), nothing to rotate.", len(pt_files), keep_n)
        return deleted

    # Sort newest-first by mtime; .regressed files sort to the end so
    # they are the first candidates for deletion.
    pt_files.sort(key=lambda p: (p.suffix == ".regressed", -p.stat().st_mtime))

    # Everything beyond keep_n is a deletion candidate
    candidates = pt_files[keep_n:]

    for path in candidates:
        try:
            path.unlink()
            deleted.append(str(path))
            logger.info("Deleted checkpoint: %s", path.name)
        except OSError as exc:
            logger.warning("Failed to delete checkpoint %s: %s", path.name, exc)

    return deleted


# ── Recurrent Data Rotation ──────────────────────────────────────────


def rotate_recurrent_data(
    max_mb: int = DEFAULT_RECURRENT_MAX_MB,
) -> dict:
    """Rotate ``recurrent_data.jsonl`` when it exceeds ``max_mb``.

    If the file does not exist or is under the size threshold, no action
    is taken.  Otherwise the current file is renamed to a timestamped
    backup, a fresh empty ``recurrent_data.jsonl`` is created, and older
    backups beyond ``DEFAULT_RECURRENT_BACKUPS`` are pruned.

    Args:
        max_mb: Maximum size in megabytes before rotation is triggered.

    Returns:
        Dict with keys:
          - ``rotated`` (bool): whether a rotation occurred.
          - ``backup_path`` (str | None): path of the new backup, or None.
          - ``deleted`` (list[str]): paths of pruned old backups.
    """
    result: dict = {"rotated": False, "backup_path": None, "deleted": []}

    recurrent_dir = Path(M1_RECURRENT)
    data_file = recurrent_dir / "recurrent_data.jsonl"

    if not data_file.is_file():
        logger.info("No recurrent data file found at %s, skipping.", data_file)
        return result

    size_bytes = data_file.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if size_mb < max_mb:
        logger.info("Recurrent data is %.2f MB (under %d MB limit), skipping.", size_mb, max_mb)
        return result

    # ── Rotate current file to timestamped backup ─────────────────────

    timestamp = datetime.now().strftime(_TIMESTAMP_FMT)
    backup_name = f"recurrent_data.jsonl.{timestamp}"
    backup_path = recurrent_dir / backup_name

    try:
        data_file.rename(backup_path)
        result["rotated"] = True
        result["backup_path"] = str(backup_path)
        logger.info("Rotated recurrent data (%.2f MB) -> %s", size_mb, backup_name)
    except OSError as exc:
        logger.error("Failed to rotate recurrent data file: %s", exc)
        return result

    # Create a fresh empty file
    try:
        data_file.touch()
        logger.info("Created fresh recurrent_data.jsonl")
    except OSError as exc:
        logger.error("Failed to create fresh recurrent_data.jsonl: %s", exc)

    # ── Prune old backups ─────────────────────────────────────────────

    backup_pattern = re.compile(r"^recurrent_data\.jsonl\.\d{8}_\d{6}$")
    backups: list[Path] = []
    for entry in recurrent_dir.iterdir():
        if not entry.is_file():
            continue
        if backup_pattern.match(entry.name):
            backups.append(entry)

    # Sort newest-first
    backups.sort(key=lambda p: -p.stat().st_mtime)

    if len(backups) > DEFAULT_RECURRENT_BACKUPS:
        for old_backup in backups[DEFAULT_RECURRENT_BACKUPS:]:
            try:
                old_backup.unlink()
                result["deleted"].append(str(old_backup))
                logger.info("Pruned old recurrent backup: %s", old_backup.name)
            except OSError as exc:
                logger.warning("Failed to prune backup %s: %s", old_backup.name, exc)

    return result


# ── Archive Rotation ──────────────────────────────────────────────────


def rotate_archives(keep_n: int = DEFAULT_ARCHIVE_KEEP) -> list[str]:
    """Remove old cycle and archive directories beyond ``keep_n``.

    Scans two locations:
      1. ``PROJECT_ROOT`` for ``cycle_*`` directories (sorted by cycle number).
      2. ``M1_CHECKPOINTS`` for ``archive_*`` subdirectories (sorted by name).

    Both sets are independently pruned to ``keep_n`` most-recent entries.

    Args:
        keep_n: Number of directories to retain per scan location.

    Returns:
        List of absolute paths that were deleted.
    """
    deleted: list[str] = []
    proj_root = Path(PROJECT_ROOT)
    ckpt_dir = Path(M1_CHECKPOINTS)

    # ── Cycle directories in PROJECT_ROOT ─────────────────────────────

    cycle_pattern = re.compile(r"^cycle_(\d+)$")
    cycle_dirs: list[tuple[int, Path]] = []

    if proj_root.is_dir():
        for entry in proj_root.iterdir():
            if not entry.is_dir():
                continue
            match = cycle_pattern.match(entry.name)
            if match:
                cycle_dirs.append((int(match.group(1)), entry))

    # Sort by cycle number ascending; delete oldest first
    cycle_dirs.sort(key=lambda t: t[0])

    if len(cycle_dirs) > keep_n:
        for _num, path in cycle_dirs[: len(cycle_dirs) - keep_n]:
            try:
                shutil.rmtree(path)
                deleted.append(str(path))
                logger.info("Deleted old cycle directory: %s", path.name)
            except OSError as exc:
                logger.warning("Failed to delete cycle directory %s: %s", path.name, exc)

    # ── Archive subdirectories in M1_CHECKPOINTS ──────────────────────

    archive_pattern = re.compile(r"^archive_(\d+)$")
    archive_dirs: list[tuple[int, Path]] = []

    if ckpt_dir.is_dir():
        for entry in ckpt_dir.iterdir():
            if not entry.is_dir():
                continue
            match = archive_pattern.match(entry.name)
            if match:
                archive_dirs.append((int(match.group(1)), entry))

    archive_dirs.sort(key=lambda t: t[0])

    if len(archive_dirs) > keep_n:
        for _num, path in archive_dirs[: len(archive_dirs) - keep_n]:
            try:
                shutil.rmtree(path)
                deleted.append(str(path))
                logger.info("Deleted old archive directory: %s", path.name)
            except OSError as exc:
                logger.warning("Failed to delete archive directory %s: %s", path.name, exc)

    return deleted


# ── Report Rotation ───────────────────────────────────────────────────


def rotate_reports(keep_n: int = DEFAULT_REPORT_KEEP) -> list[str]:
    """Remove old progress report backups while preserving the live report.

    Scans ``PROJECT_ROOT`` for ``progress_report*.json`` files, excluding
    the live ``progress_report.json`` itself.  The most recent ``keep_n``
    files by mtime are retained; older ones are deleted.

    Args:
        keep_n: Number of report backup files to retain.

    Returns:
        List of absolute paths that were deleted.
    """
    deleted: list[str] = []
    proj_root = Path(PROJECT_ROOT)

    if not proj_root.is_dir():
        logger.warning("Project root does not exist: %s", proj_root)
        return deleted

    report_files: list[Path] = []
    for entry in proj_root.iterdir():
        if not entry.is_file():
            continue
        if not entry.name.startswith("progress_report"):
            continue
        if not entry.name.endswith(".json"):
            continue
        # Preserve the live report
        if entry.name == _LIVE_REPORT_FILE:
            continue
        report_files.append(entry)

    if len(report_files) <= keep_n:
        logger.info("Report backup count (%d) <= keep_n (%d), nothing to rotate.", len(report_files), keep_n)
        return deleted

    # Sort newest-first
    report_files.sort(key=lambda p: -p.stat().st_mtime)

    candidates = report_files[keep_n:]

    for path in candidates:
        try:
            path.unlink()
            deleted.append(str(path))
            logger.info("Deleted old report: %s", path.name)
        except OSError as exc:
            logger.warning("Failed to delete report %s: %s", path.name, exc)

    return deleted


# ── Convenience Runner ────────────────────────────────────────────────


def run_all_rotations() -> dict[str, int]:
    """Execute all rotation passes and return a summary.

    Calls ``rotate_checkpoints``, ``rotate_recurrent_data``,
    ``rotate_archives``, and ``rotate_reports`` in sequence, then logs
    a single summary line.

    Returns:
        Dict mapping category name to number of items deleted:
          - ``checkpoints``: files deleted
          - ``recurrent``: backups pruned (1 if rotated, 0 otherwise)
          - ``archives``: directories deleted
          - ``reports``: files deleted
    """
    cp_deleted = rotate_checkpoints()
    recurrent_result = rotate_recurrent_data()
    arch_deleted = rotate_archives()
    rep_deleted = rotate_reports()

    summary: dict[str, int] = {
        "checkpoints": len(cp_deleted),
        "recurrent": len(recurrent_result["deleted"]),
        "archives": len(arch_deleted),
        "reports": len(rep_deleted),
    }

    total = sum(summary.values())
    logger.info(
        "Data rotation complete: checkpoints=%d, recurrent=%d, " "archives=%d, reports=%d (total=%d)",
        summary["checkpoints"],
        summary["recurrent"],
        summary["archives"],
        summary["reports"],
        total,
    )

    return summary
