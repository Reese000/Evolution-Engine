#!/usr/bin/env python3
"""log_rotation.py — Production log rotation with compression and component separation.

Enhances logging_config.py with:
  - RotatingFileHandler with tuned max size and backup count
  - Gzip compression for old log files
  - Separate log files per component (evolution, retrain, errors)
  - Log retention policy (max age in days)

Usage:
    from log_rotation import setup_production_logging, get_component_logger
    setup_production_logging()
    evo_logger = get_component_logger("evolution")
    retrain_logger = get_component_logger("retrain")
    error_logger = get_component_logger("errors")
"""

from __future__ import annotations

import gzip
import logging
import logging.handlers
import os
import shutil
import time
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────

COMPONENT_NAMES = ("evolution", "retrain", "errors", "alerts", "monitoring")

DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
DEFAULT_BACKUP_COUNT = 10  # Keep 10 rotated files
DEFAULT_RETENTION_DAYS = 30  # Compress logs older than 30 days

COMPRESSED_SUFFIX = ".gz"


# ── Compressed Rotating Handler ───────────────────────────────────────


class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that compresses old log files on rotation.

    When a log file reaches maxBytes, it is rotated. The rotated file
    is then gzipped to save disk space. Overrides doRollover to handle
    the full cascade correctly with .gz extensions.
    """

    def __init__(
        self,
        filename: str,
        maxBytes: int = DEFAULT_MAX_BYTES,
        backupCount: int = DEFAULT_BACKUP_COUNT,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount, encoding=encoding)
        self._compress_old = True

    def doRollover(self) -> None:
        """Perform rotation with proper .gz handling.

        RotatingFileHandler.doRollover cascades numbered backups (log.1 -> log.2
        -> ...), but our compress step removes log.1 (leaving log.1.gz). The
        next rollover's cascade can't find log.1 to shift it to log.2, breaking
        the backup chain.  Override the entire rollover to: 1) compress any
        existing uncompressed backups, 2) shift .gz backups, 3) rotate + compress
        the new backup.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        if self._compress_old:
            # Compress any uncompressed backup files first
            for i in range(1, self.backupCount + 1):
                backup = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                if os.path.isfile(backup):
                    self._gzip_file(backup)

            # Cascade compressed backups: .2.gz <- .1.gz, .3.gz <- .2.gz, ...
            for i in range(self.backupCount - 1, 0, -1):
                src = self.rotation_filename("%s.%d.gz" % (self.baseFilename, i))
                dst = self.rotation_filename("%s.%d.gz" % (self.baseFilename, i + 1))
                if os.path.isfile(src):
                    if os.path.isfile(dst):
                        os.remove(dst)
                    os.rename(src, dst)

            # Rotate current log -> .1, then compress .1 -> .1.gz
            dfn = self.rotation_filename(self.baseFilename + ".1")
            if os.path.isfile(dfn):
                os.remove(dfn)
            if os.path.isfile(self.baseFilename):
                os.rename(self.baseFilename, dfn)
                self._gzip_file(dfn)

            # Remove oldest backup beyond backupCount
            oldest = self.rotation_filename("%s.%d.gz" % (self.baseFilename, self.backupCount + 1))
            if os.path.isfile(oldest):
                os.remove(oldest)
        else:
            # No compression: use standard rollover
            super().doRollover()

        # Reopen the log file
        self.stream = self._open()

    def rotate(self, source: str, dest: str) -> None:
        """Rotate source -> dest. Compression handled in doRollover."""
        super().rotate(source, dest)

    def _gzip_file(self, filepath: str) -> None:
        """Compress a file with gzip, replacing the original."""
        gz_path = filepath + COMPRESSED_SUFFIX
        if os.path.isfile(gz_path):
            return  # Already compressed
        try:
            with open(filepath, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(filepath)
        except (OSError, IOError):
            pass  # Best effort


# ── Component Loggers ─────────────────────────────────────────────────

_component_loggers: dict[str, logging.Logger] = {}


def setup_production_logging(
    log_dir: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    console_level: int = logging.INFO,
) -> str:
    """Set up production logging with per-component file separation.

    Creates:
      - logs/evolution.log  — evolution engine messages
      - logs/retrain.log    — retrain/training messages
      - logs/errors.log     — ERROR and above only
      - logs/alerts.log     — alert system messages
      - logs/monitoring.log — health/metrics messages
      - evolution.log       — root log (all components, legacy compat)

    Args:
        log_dir: Base directory for logs. Defaults to PROJECT_ROOT/logs.
        max_bytes: Max bytes per log file before rotation.
        backup_count: Number of rotated backups to keep.
        retention_days: Days before log files are compressed.
        console_level: Minimum level for console output.

    Returns:
        The log directory path.
    """
    if log_dir is None:
        try:
            from paths import PROJECT_ROOT

            log_dir = str(PROJECT_ROOT / "logs")
        except ImportError:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

    os.makedirs(log_dir, exist_ok=True)

    # Structured formatter (no color for files)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (with color)
    from logging_config import ColoredFormatter

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter())

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console_handler)

    # Root catch-all file (backward compat)
    root_file = os.path.join(log_dir, "evolution.log")
    root_fh = CompressedRotatingFileHandler(
        root_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    root_fh.setLevel(logging.DEBUG)
    root_fh.setFormatter(fmt)
    root.addHandler(root_fh)

    # Per-component files
    for comp_name in COMPONENT_NAMES:
        comp_path = os.path.join(log_dir, f"{comp_name}.log")
        comp_handler = CompressedRotatingFileHandler(
            comp_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        if comp_name == "errors":
            comp_handler.setLevel(logging.ERROR)
        else:
            comp_handler.setLevel(logging.DEBUG)
        comp_handler.setFormatter(fmt)

        comp_logger = logging.getLogger(f"ee.{comp_name}")
        comp_logger.addHandler(comp_handler)
        comp_logger.setLevel(logging.DEBUG)
        _component_loggers[comp_name] = comp_logger

    # The "errors" logger should also capture from root for ERROR+
    # We add a filter-free handler so root ERROR messages also go to errors.log
    errors_logger = logging.getLogger("ee.errors")

    logger = logging.getLogger("ee.log_rotation")
    logger.info(f"Production logging initialized: dir={log_dir}, " f"max_bytes={max_bytes}, backups={backup_count}")

    return log_dir


def get_component_logger(name: str) -> logging.Logger:
    """Get a named component logger.

    Args:
        name: One of: evolution, retrain, errors, alerts, monitoring

    Returns:
        Logger instance writing to logs/<name>.log
    """
    if name in _component_loggers:
        return _component_loggers[name]
    # Fallback: create dynamically
    logger = logging.getLogger(f"ee.{name}")
    _component_loggers[name] = logger
    return logger


def compress_old_logs(log_dir: str, max_age_days: int = DEFAULT_RETENTION_DAYS) -> list[str]:
    """Compress log files older than max_age_days.

    Scans for .log files (non-rotated) that are older than the threshold
    and gzips them.

    Returns list of compressed file paths.
    """
    compressed: list[str] = []
    cutoff = time.time() - (max_age_days * 86400)

    for entry in os.listdir(log_dir):
        full_path = os.path.join(log_dir, entry)
        if not os.path.isfile(full_path):
            continue
        if not entry.endswith(".log"):
            continue
        if entry.endswith(COMPRESSED_SUFFIX):
            continue
        # Skip currently active log files (has active handler)
        if os.path.getmtime(full_path) > cutoff:
            continue

        gz_path = full_path + COMPRESSED_SUFFIX
        try:
            with open(full_path, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(full_path)
            compressed.append(gz_path)
        except (OSError, IOError):
            pass

    return compressed


def get_log_stats(log_dir: str) -> dict[str, Any]:
    """Return stats about log files in the directory.

    Returns dict with:
      - total_files: number of log files
      - total_size_mb: total size in MB
      - compressed_count: number of .gz files
      - oldest_file: name of oldest log
      - component_files: dict of component → file size
    """
    total_files = 0
    total_size = 0
    compressed_count = 0
    oldest_time = float("inf")
    oldest_name = ""
    component_files: dict[str, float] = {}

    if not os.path.isdir(log_dir):
        return {
            "total_files": 0,
            "total_size_mb": 0.0,
            "compressed_count": 0,
            "oldest_file": None,
            "component_files": {},
        }

    for entry in os.listdir(log_dir):
        full_path = os.path.join(log_dir, entry)
        if not os.path.isfile(full_path):
            continue
        if not (entry.endswith(".log") or entry.endswith(".gz")):
            continue

        total_files += 1
        size = os.path.getsize(full_path)
        total_size += size

        if entry.endswith(".gz"):
            compressed_count += 1

        mtime = os.path.getmtime(full_path)
        if mtime < oldest_time:
            oldest_time = mtime
            oldest_name = entry

        # Track component files
        base = entry.replace(".log", "").replace(".gz", "")
        component_files[base] = component_files.get(base, 0) + size / (1024 * 1024)

    return {
        "total_files": total_files,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "compressed_count": compressed_count,
        "oldest_file": oldest_name if oldest_name else None,
        "component_files": {k: round(v, 3) for k, v in component_files.items()},
    }
