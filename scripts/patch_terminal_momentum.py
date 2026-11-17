#!/usr/bin/env python3
"""Patch progress_trajectory.py with terminal sustained momentum floor."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PT = ROOT / "scripts" / "progress_trajectory.py"
TEST = ROOT / "tests" / "unit" / "test_progress_trajectory.py"

CONST_BLOCK = """TERMINAL_MOMENTUM_FLOOR = float(os.environ.get("EE_TERMINAL_MOMENTUM_FLOOR", "0.55"))
TERMINAL_MOMENTUM_EPI_MIN = float(os.environ.get("EE_TERMINAL_MOMENTUM_EPI_MIN", "90.0"))
"""

HELPERS = '''

def _terminal_growth_momentum(points: list[dict[str, Any]], index: int) -> float:
    """Credit sustained max-depth excellence when classic EPI delta flattens."""
    p = points[index]
    level = int(p.get("level", 0))
    if level < int(MAX_LEVEL):
        return 0.0
    verify = float(p.get("verify_score", 0.0))
    if verify < 0.99 - 1e-6:
        return 0.0
    epi = float(p.get("epi", 0.0))
    if epi < TERMINAL_MOMENTUM_EPI_MIN:
        return 0.0
    bonus = TERMINAL_MOMENTUM_FLOOR
    if index <= 0:
        return bonus
    prev = points[index - 1]
    prev_fit = float(prev.get("fitness", 0.0))
    fit = float(p.get("fitness", 0.0))
    if fit > prev_fit + 1e-3:
        rel = (fit - prev_fit) / max(abs(prev_fit), 1.0)
        bonus += min(3.0, rel * 50.0)
    prev_epi = float(prev.get("epi", 0.0))
    if epi > prev_epi + 0.01:
        bonus += min(2.0, (epi - prev_epi) * 2.0)
    comps = p.get("components") or {}
    prev_comps = prev.get("components") or {}
    arch = float(comps.get("archive_learning", 0.0))
    prev_arch = float(prev_comps.get("archive_learning", 0.0))
    if arch > prev_arch + 0.05:
        bonus += min(1.5, (arch - prev_arch) * 0.05)
    return bonus


def _momentum_for_point(
    points: list[dict[str, Any]],
    index: int,
    epi_series: list[float],
    rolling: list[float],
) -> float:
    classic = round(epi_series[index] - rolling[index], 2)
    terminal = round(_terminal_growth_momentum(points, index), 2)
    if terminal > 0.0:
        return round(max(classic, terminal), 2)
    return classic
'''

TEST_FN = """

def test_terminal_sustained_momentum_floor():
    from scripts.progress_trajectory import _apply_epi_momentum

    points = []
    for i in range(60):
        points.append(
            {
                "cycle": 3400 + i,
                "epi": 93.5,
                "level": 30,
                "fitness": 264.0,
                "verify_score": 1.0,
                "components": {"archive_learning": 25.0},
            }
        )
    _apply_epi_momentum(points)
    tail = [p["epi_momentum"] for p in points[-10:]]
    assert all(m >= 0.55 for m in tail)


"""


def main() -> int:
    t = PT.read_text(encoding="utf-8")
    if "TERMINAL_MOMENTUM_FLOOR" not in t:
        t = t.replace(
            'WATCH_INTERVAL = float(os.environ.get("EE_TRAJECTORY_WATCH_SEC", "12"))\n',
            'WATCH_INTERVAL = float(os.environ.get("EE_TRAJECTORY_WATCH_SEC", "12"))\n' + CONST_BLOCK,
        )
    if "_terminal_growth_momentum" not in t:
        t = t.replace(
            "def _rolling_mean(values: list[float], window: int) -> list[float]:",
            HELPERS + "\ndef _rolling_mean(values: list[float], window: int) -> list[float]:",
        )
    t = t.replace(
        'p["epi_momentum"] = round(p["epi"] - rolling[i], 2)',
        'p["epi_momentum"] = _momentum_for_point(points, i, epi_series, rolling)',
    )
    t = t.replace(
        'p["epi_momentum"] = round(epi_series[i] - rolling[i], 2)',
        'p["epi_momentum"] = _momentum_for_point(points, i, epi_series, rolling)',
    )
    PT.write_text(t, encoding="utf-8", newline="\n")

    tt = TEST.read_text(encoding="utf-8")
    if "test_terminal_sustained_momentum_floor" not in tt:
        tt = tt.replace("def test_epi_increases_with_level():", TEST_FN + "def test_epi_increases_with_level():")
        TEST.write_text(tt, encoding="utf-8", newline="\n")

    print("patched progress_trajectory terminal momentum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
