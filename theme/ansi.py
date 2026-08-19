"""ANSI color utilities for terminal output.

Converts Design System hex tokens to ANSI 24-bit (truecolor) escape
sequences for rich terminal styling.
"""
from __future__ import annotations

import re
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# ANSI ESCAPE CODES
# ══════════════════════════════════════════════════════════════════════════════

ESC = "\033"
RESET = f"{ESC}[0m"

# ── SGR codes ────────────────────────────────────────────────────────────────
BOLD = f"{ESC}[1m"
DIM = f"{ESC}[2m"
ITALIC = f"{ESC}[3m"
UNDERLINE = f"{ESC}[4m"
INVERSE = f"{ESC}[7m"
STRIKETHROUGH = f"{ESC}[9m"

# ── 24-bit foreground: \033[38;2;R;G;Bm ─────────────────────────────────────
_FG_TEMPLATE = f"{ESC}[38;2;{{r}};{{g}};{{b}}m"

# ── 24-bit background: \033[48;2;R;G;Bm ─────────────────────────────────────
_BG_TEMPLATE = f"{ESC}[48;2;{{r}};{{g}};{{b}}m"

# ── Combined fg+bg ──────────────────────────────────────────────────────────
_FG_BG_TEMPLATE = f"{ESC}[38;2;{{r}};{{g}};{{b}};48;2;{{br}};{{bg}};{{bb}}m"


# ══════════════════════════════════════════════════════════════════════════════
# HEX → RGB CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$")


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to (R, G, B) integers.

    Accepts: #RRGGBB, #RRGGBBAA, RRGGBB, RRGGBBAA.
    The alpha component is ignored.
    """
    m = _HEX_RE.match(hex_color)
    if not m:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    hex6 = m.group(1)
    return (
        int(hex6[0:2], 16),
        int(hex6[2:4], 16),
        int(hex6[4:6], 16),
    )


def _parse_rgba(color: str) -> tuple[int, int, int]:
    """Parse a color that might be #hex, rgb(), or rgba()."""
    if color.startswith("#"):
        return hex_to_rgb(color)
    # rgba(r,g,b,a) or rgb(r,g,b)
    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", color)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    raise ValueError(f"Cannot parse color: {color!r}")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def ansi_color(hex_color: str) -> str:
    """Return ANSI escape sequence for a foreground color from hex."""
    r, g, b = _parse_rgba(hex_color)
    return _FG_TEMPLATE.format(r=r, g=g, b=b)


def ansi_bg(hex_color: str) -> str:
    """Return ANSI escape sequence for a background color from hex."""
    r, g, b = _parse_rgba(hex_color)
    return _BG_TEMPLATE.format(r=r, g=g, b=b)


def ansi_fg_bg(fg_hex: str, bg_hex: str) -> str:
    """Return ANSI escape for combined foreground + background."""
    r, g, b = _parse_rgba(fg_hex)
    br, bg, bb = _parse_rgba(bg_hex)
    return _FG_BG_TEMPLATE.format(r=r, g=g, b=b, br=br, bg=bg, bb=bb)


def ansi_reset() -> str:
    """Return the ANSI reset escape sequence."""
    return RESET


def ansi_style(
    bold: bool = False,
    dim: bool = False,
    italic: bool = False,
    underline: bool = False,
    inverse: bool = False,
    strikethrough: bool = False,
) -> str:
    """Build an ANSI style escape sequence from flags."""
    parts = []
    if bold:
        parts.append(BOLD)
    if dim:
        parts.append(DIM)
    if italic:
        parts.append(ITALIC)
    if underline:
        parts.append(UNDERLINE)
    if inverse:
        parts.append(INVERSE)
    if strikethrough:
        parts.append(STRIKETHROUGH)
    return "".join(parts)


def colorize(
    text: str,
    fg: Optional[str] = None,
    bg: Optional[str] = None,
    bold: bool = False,
    dim: bool = False,
    italic: bool = False,
    underline: bool = False,
) -> str:
    """Apply ANSI styling to text.

    Args:
        text: The text to style.
        fg: Foreground hex color (e.g. "#6c8aff").
        bg: Background hex color.
        bold: Bold text.
        dim: Dimmed text.
        italic: Italic text.
        underline: Underlined text.

    Returns:
        Styled text with ANSI escapes and trailing reset.
    """
    parts = []
    if fg:
        parts.append(ansi_color(fg))
    if bg:
        parts.append(ansi_bg(bg))
    if bold or dim or italic or underline:
        parts.append(ansi_style(
            bold=bold, dim=dim, italic=italic, underline=underline,
        ))
    if not parts:
        return text
    return f"{''.join(parts)}{text}{RESET}"
