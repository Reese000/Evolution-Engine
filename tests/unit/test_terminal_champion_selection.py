"""Champion selection at certified L30 terminal plateau."""

import pytest

from evolution_engine import EvolutionConfig, EvolutionEngine


def _l30_engine(monkeypatch, *, promote: bool = True) -> EvolutionEngine:
    if promote:
        monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    else:
        monkeypatch.delenv("EE_TERMINAL_VERIFY_PROMOTE", raising=False)
    monkeypatch.setenv("EE_MAX_LEVEL", "30")
    cfg = EvolutionConfig(max_level=30)
    eng = EvolutionEngine(config=cfg)
    eng.champion_level = 30
    eng.champion_fitness = 264.04
    eng._champion_verify_score = 1.0
    eng.champion = [0] * 73
    eng.champion_trace = {}
    eng._oscillation_positive_count = 10
    eng._oscillation_threshold = 3
    eng.generations_stalled = 0
    return eng


def test_l30_accepts_small_fitness_gain_despite_hysteresis(monkeypatch):
    """Sub-0.5 fitness improvements must be accepted at saturated verify L30."""
    eng = _l30_engine(monkeypatch, promote=True)
    base = float(eng.champion_fitness)
    assert eng._should_replace_champion(base + 0.01) is True


def test_l30_hysteresis_still_applies_below_certify_bar(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    cfg = EvolutionConfig(max_level=30)
    eng = EvolutionEngine(config=cfg)
    eng.champion_level = 30
    eng.champion_fitness = 200.0
    eng._champion_verify_score = 0.95
    eng.champion = [0] * 40
    eng.champion_trace = {}
    eng._oscillation_positive_count = 10
    eng._oscillation_threshold = 3
    assert eng._should_replace_champion(200.05) is False
