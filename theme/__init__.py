"""Design System theme layer for Evolution Engine.

Translates the Reese Design System tokens (tokens.css) into terminal-ready
ANSI color codes and styling utilities. Single source of truth for all
colors used in CLI output.
"""
from .tokens import TOKENS, LIGHT_OVERRIDES, get_token, get_all_tokens
from .ansi import ansi_color, ansi_style, ansi_reset, colorize
from .manager import ThemeManager

__all__ = [
    "TOKENS",
    "LIGHT_OVERRIDES",
    "get_token",
    "get_all_tokens",
    "ansi_color",
    "ansi_style",
    "ansi_reset",
    "colorize",
    "ThemeManager",
]
