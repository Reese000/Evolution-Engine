"""Theme manager — dark/light toggle for terminal output.

Manages the active theme state and provides convenience accessors
for semantic colors mapped to the Design System token palette.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .ansi import ansi_bg, ansi_color, colorize
from .tokens import LIGHT_OVERRIDES, TOKENS, get_token

logger = logging.getLogger(__name__)

# Theme storage path
_CONFIG_DIR = Path.home() / ".evolution-engine"
_CONFIG_FILE = _CONFIG_DIR / "theme.json"

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING LEVEL → SEMANTIC COLOR MAPPING
# Maps Python log levels to Design System semantic tokens.
# ══════════════════════════════════════════════════════════════════════════════

# Default (dark theme) mapping
_LEVEL_TOKENS_DARK = {
    "DEBUG":    "code_info",       # Cyan — debug/reference
    "INFO":     "code_success",    # Green — success/running
    "WARNING":  "code_warning",    # Amber — warning/caution
    "ERROR":    "code_danger",     # Red — error/destructive
    "CRITICAL": "code_accent",     # Purple — critical/accent
}

# Light theme mapping (same tokens, different base values)
_LEVEL_TOKENS_LIGHT = dict(_LEVEL_TOKENS_DARK)


class ThemeManager:
    """Manages dark/light theme switching for terminal output.

    Singleton pattern — use ThemeManager.instance() to get the global manager.
    """

    _instance: ThemeManager | None = None

    def __init__(self) -> None:
        self._dark = True
        self._load_preference()

    @classmethod
    def instance(cls) -> ThemeManager:
        """Get or create the singleton ThemeManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_dark(self) -> bool:
        """Whether the current theme is dark."""
        return self._dark

    def set_dark(self, dark: bool) -> None:
        """Set theme explicitly and persist preference."""
        if self._dark != dark:
            self._dark = dark
            self._save_preference()

    def toggle(self) -> bool:
        """Toggle between dark and light themes. Returns new state."""
        self._dark = not self._dark
        self._save_preference()
        return self._dark

    # ── Token access ─────────────────────────────────────────────────────
    def get(self, name: str, fallback: str = "") -> str:
        """Get a token value for the current theme."""
        if not self._dark:
            return LIGHT_OVERRIDES.get(name, TOKENS.get(name, fallback))
        return TOKENS.get(name, fallback)

    def color_for_level(self, level: str) -> str:
        """Get the ANSI color escape for a logging level."""
        tokens = _LEVEL_TOKENS_DARK if self._dark else _LEVEL_TOKENS_LIGHT
        token_name = tokens.get(level, "ink_secondary")
        hex_val = self.get(token_name, "#888888")
        return ansi_color(hex_val)

    def colorize(self, text: str, fg_token: str, **kwargs) -> str:
        """Colorize text using a token name as the foreground color."""
        hex_val = self.get(fg_token, "#888888")
        return colorize(text, fg=hex_val, **kwargs)

    # ── Persistence ──────────────────────────────────────────────────────
    def _save_preference(self) -> None:
        """Save theme preference to disk."""
        try:
            _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            data = {"theme": "dark" if self._dark else "light"}
            _CONFIG_FILE.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.debug("Failed to save theme preference: %s", exc)

    def _load_preference(self) -> None:
        """Load theme preference from disk."""
        try:
            if _CONFIG_FILE.exists():
                data = json.loads(_CONFIG_FILE.read_text())
                self._dark = data.get("theme", "dark") == "dark"
        except Exception as exc:
            logger.debug("Failed to load theme preference: %s", exc)
            self._dark = True  # Default to dark
