#!/usr/bin/env python3
"""logging_config.py — Structured logging configuration for Evolution Engine.

Sets up Python logging with:
  - Console handler (INFO level, colored output)
  - File handler (evolution.log, DEBUG level, rotating)
  - Structured format: [timestamp] [level] [component] message

Usage:
    from logging_config import setup_logging, get_logger
    setup_logging()
    logger = get_logger("EvolutionEngine")
    logger.info("Starting evolution run")
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys

# ── Theme-driven color codes for console output ──────────────────────────────
# Uses the Design System token palette via theme.layer instead of hardcoded
# ANSI codes. Falls back to standard ANSI if theme import fails.

try:
    from theme.manager import ThemeManager as _ThemeManager
    _theme = _ThemeManager.instance()
    COLORS = {
        "DEBUG":    _theme.color_for_level("DEBUG"),
        "INFO":     _theme.color_for_level("INFO"),
        "WARNING":  _theme.color_for_level("WARNING"),
        "ERROR":    _theme.color_for_level("ERROR"),
        "CRITICAL": _theme.color_for_level("CRITICAL"),
        "RESET":    "\033[0m",
    }
except ImportError:
    # Fallback: standard ANSI if theme module unavailable
    COLORS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Green
        "WARNING":  "\033[33m",   # Yellow
        "ERROR":    "\033[31m",   # Red
        "CRITICAL": "\033[35m",   # Magenta
        "RESET":    "\033[0m",    # Reset
    }


class ColoredFormatter(logging.Formatter):
    """Formatter that adds color codes to console output."""

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color = COLORS.get(levelname, COLORS["RESET"])
        reset = COLORS["RESET"]

        # Format: [timestamp] [LEVEL] [component] message
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        component = record.name
        msg = record.getMessage()

        return f"{color}[{timestamp}] [{levelname}] [{component}]{reset} {msg}"


class StructuredFormatter(logging.Formatter):
    """Formatter for file output (no color codes)."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        component = record.name
        msg = record.getMessage()

        return f"[{timestamp}] [{record.levelname}] [{component}] {msg}"


# ── Setup ─────────────────────────────────────────────────────────────

_initialized = False


def setup_logging(
    log_dir: str | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """Initialize structured logging for the Evolution Engine.

    Args:
        log_dir: Directory for log files. If None, uses PROJECT_ROOT.
        console_level: Minimum level for console output.
        file_level: Minimum level for file output.
        max_bytes: Max size per log file before rotation.
        backup_count: Number of rotated log files to keep.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Resolve log directory
    if log_dir is None:
        try:
            from paths import PROJECT_ROOT
            log_dir = str(PROJECT_ROOT)
        except ImportError:
            log_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "evolution.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to prevent duplicates
    root_logger.handlers.clear()

    # ── Console handler ───────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)

    # ── File handler (rotating) ───────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)

    # Log initialization
    logger = get_logger("LoggingConfig")
    logger.info(f"Logging initialized: console={logging.getLevelName(console_level)}, "
                f"file={logging.getLevelName(file_level)} ({log_path})")


def get_logger(name: str) -> logging.Logger:
    """Get a named logger component.

    Args:
        name: Component name (e.g., "EvolutionEngine", "MetaController", "Runner").

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(name)
