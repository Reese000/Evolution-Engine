"""Design System tokens — translated from tokens.css for CLI/ANSI usage.

Single source of truth for all colors, spacing, typography, and component
styles. Derived from the Reese Design System v1.3.1.

For terminal output, we map semantic colors to ANSI 24-bit (truecolor)
escape sequences. Canvas/ink tokens map to background/foreground pairs.
"""
from __future__ import annotations

from typing import Dict

# ══════════════════════════════════════════════════════════════════════════════
# TOKEN DICTIONARY — CSS hex values from tokens.css
# ══════════════════════════════════════════════════════════════════════════════

TOKENS: Dict[str, str] = {
    # ── CANVAS (backgrounds) ──────────────────────────────────────────────────
    "canvas_void":     "#0e1013",  # Deepest background
    "canvas_base":     "#14161b",  # App background
    "canvas_surface":  "#1b1d24",  # Panel backgrounds
    "canvas_raised":   "#23252d",  # Card/elevated surfaces
    "canvas_elevated": "#2c2e38",  # Hover states, inputs
    "canvas_bright":   "#363840",  # Scrollbars, skeletons

    # ── INK (text hierarchy) ──────────────────────────────────────────────────
    "ink_primary":   "#e8eaef",  # Main text
    "ink_secondary": "#a8adb8",  # Body/supporting
    "ink_tertiary":  "#7c8294",  # Labels
    "ink_muted":     "#6b7088",  # Disabled/hints
    "ink_ghost":     "#555a70",  # Lowest contrast

    # ── SEMANTIC COLORS (13 categories × 3 variants) ──────────────────────────
    # Action (blue) — primary actions, buttons, links
    "code_action":     "#6c8aff",
    "code_action_dim": "#6c8aff22",
    "code_action_mid": "#6c8aff55",
    # Danger (red) — destructive / error
    "code_danger":     "#ff6b6b",
    "code_danger_dim": "#ff6b6b22",
    "code_danger_mid": "#ff6b6b55",
    # Success (green) — success / complete
    "code_success":     "#4ecb71",
    "code_success_dim": "#4ecb7122",
    "code_success_mid": "#4ecb7155",
    # Warning (amber) — warning / caution
    "code_warning":     "#ffba42",
    "code_warning_dim": "#ffba4222",
    "code_warning_mid": "#ffba4255",
    # Info (cyan) — info / reference
    "code_info":     "#7ce0ff",
    "code_info_dim": "#7ce0ff22",
    "code_info_mid": "#7ce0ff55",
    # Accent (purple) — accent / brand
    "code_accent":     "#c084fc",
    "code_accent_dim": "#c084fc22",
    "code_accent_mid": "#c084fc55",
    # Tool (orange) — tool/operation
    "code_tool":     "#ff9a5c",
    "code_tool_dim": "#ff9a5c22",
    "code_tool_mid": "#ff9a5c55",
    # Material (teal) — material/supply
    "code_material":     "#5ce1e6",
    "code_material_dim": "#5ce1e622",
    "code_material_mid": "#5ce1e655",
    # Data (pink) — metrics
    "code_data":     "#ff6baa",
    "code_data_dim": "#ff6baa22",
    "code_data_mid": "#ff6baa55",
    # View (lime) — viewport
    "code_view":     "#a3e635",
    "code_view_dim": "#a3e63522",
    "code_view_mid": "#a3e63555",
    # Config (gold) — configuration
    "code_config":     "#fbbf24",
    "code_config_dim": "#fbbf2422",
    "code_config_mid": "#fbbf2455",
    # Process (indigo) — workflow
    "code_process":     "#818cf8",
    "code_process_dim": "#818cf822",
    "code_process_mid": "#818cf855",
    # AI (magenta) — AI/ML
    "code_ai":     "#f472b6",
    "code_ai_dim": "#f472b622",
    "code_ai_mid": "#f472b655",

    # ── BORDERS ───────────────────────────────────────────────────────────────
    "border_subtle":  "rgba(255,255,255,0.06)",
    "border_default": "rgba(255,255,255,0.09)",
    "border_strong":  "rgba(255,255,255,0.14)",
    "border_vivid":   "rgba(255,255,255,0.20)",

    # ── TYPOGRAPHY ────────────────────────────────────────────────────────────
    "font_ui":   "Segoe UI, Inter, system-ui, sans-serif",
    "font_mono": "JetBrains Mono, Consolas, monospace",
    "text_xs":   "11px",
    "text_sm":   "12px",
    "text_base": "13px",
    "text_md":   "14px",
    "text_lg":   "16px",
    "text_xl":   "20px",
    "text_2xl":  "24px",

    # ── SPACING (4px grid) ────────────────────────────────────────────────────
    "sp_1":  "4px",
    "sp_2":  "8px",
    "sp_3":  "12px",
    "sp_4":  "16px",
    "sp_5":  "24px",
    "sp_6":  "32px",
    "sp_8":  "48px",
    "sp_10": "64px",
    "sp_12": "80px",

    # ── RADII ─────────────────────────────────────────────────────────────────
    "r_xs":   "2px",
    "r_sm":   "3px",
    "r_md":   "4px",
    "r_lg":   "6px",
    "r_xl":   "8px",
    "r_xxl":  "12px",
    "r_full": "9999px",

    # ── CONTROL HEIGHTS ───────────────────────────────────────────────────────
    "h_xs":  "20px",
    "h_sm":  "24px",
    "h_md":  "28px",
    "h_lg":  "36px",
    "h_xl":  "44px",
    "h_2xl": "52px",
}

# Light mode overrides (for future light terminal themes)
LIGHT_OVERRIDES: Dict[str, str] = {
    "canvas_void":     "#f0f1f3",
    "canvas_base":     "#e8e9ec",
    "canvas_surface":  "#e0e1e5",
    "canvas_raised":   "#d8d9dd",
    "canvas_elevated": "#d0d2d6",
    "canvas_bright":   "#c8c9ce",
    "ink_primary":     "#1a1c20",
    "ink_secondary":   "#3a3d48",
    "ink_tertiary":    "#5a5d66",
    "ink_muted":       "#6b7088",
    "ink_ghost":       "#8a8e98",
    "border_subtle":   "rgba(0,0,0,0.06)",
    "border_default":  "rgba(0,0,0,0.09)",
    "border_strong":   "rgba(0,0,0,0.14)",
    "border_vivid":    "rgba(0,0,0,0.20)",
}


def get_token(name: str, fallback: str = "") -> str:
    """Get a design token value by name."""
    return TOKENS.get(name, fallback)


def get_all_tokens(dark: bool = True) -> Dict[str, str]:
    """Get all tokens, optionally with light mode overrides."""
    tokens = dict(TOKENS)
    if not dark:
        tokens.update(LIGHT_OVERRIDES)
    return tokens
