"""Timeout champion replacement when fitness stuck at -25."""

from unittest.mock import MagicMock

from evolution_engine import EvolutionConfig, EvolutionEngine


def _engine() -> EvolutionEngine:
    eng = EvolutionEngine.__new__(EvolutionEngine)
    eng.config = EvolutionConfig()
    eng.champion = [0x00100093]
    eng.champion_trace = {"status": "ok", "instructions_count": 5000, "final_regs": {}}
    eng.champion_fitness = -25.0
    eng.champion_level = 19
    eng.current_generation = 1
    eng.generations_stalled = 5
    eng._champion_raw_task_score = 0.0
    eng._hall_of_fame = MagicMock()
    eng._hall_of_fame.get_best.return_value = None
    eng._log = MagicMock()
    return eng


def test_should_replace_when_champion_timeout():
    eng = _engine()
    assert eng._should_replace_champion(-24.0) is True
    assert eng._should_replace_champion(-25.0) is False


def test_champion_is_timeout_penalized():
    eng = _engine()
    assert eng._champion_is_timeout_penalized() is True
    eng.champion_fitness = 50.0
    assert eng._champion_is_timeout_penalized() is False
