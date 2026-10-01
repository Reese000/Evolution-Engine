"""Tests for growth gate lookahead warnings."""

from __future__ import annotations

import json
import os

from evolution.epi_momentum_governor import assess_growth_gate_lookahead


def _write_trajectory(tmp: str, fitness_start: float, fitness_step: float, count: int) -> None:
    agents = os.path.join(tmp, ".agents")
    os.makedirs(agents)
    path = os.path.join(agents, "progress_trajectory.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            fit = fitness_start + i * fitness_step
            f.write(
                json.dumps(
                    {
                        "cycle": i,
                        "fitness": fit,
                        "epi": 80.0 + i * 0.05,
                        "epi_momentum": 0.6,
                    }
                )
                + "\n"
            )


def test_lookahead_not_at_risk_with_strong_growth(tmp_path):
    _write_trajectory(str(tmp_path), 200.0, 0.5, 3000)
    report = assess_growth_gate_lookahead(str(tmp_path), window=5000)
    assert not report.at_risk
    assert report.projected_fitness_gain > 0.1


def test_lookahead_at_risk_when_fitness_flat(tmp_path):
    _write_trajectory(str(tmp_path), 264.04, 0.0, 3000)
    report = assess_growth_gate_lookahead(str(tmp_path), window=5000)
    assert report.at_risk
    assert report.reason in (
        "projected_fitness_gain_below_threshold",
        "full_window_fitness_stalling",
    )
