"""Curriculum advance uses level-aware partial verify thresholds."""

from __future__ import annotations

from unittest.mock import MagicMock

from evolution_engine import EvolutionConfig, EvolutionEngine


def test_l13_threshold_allows_875():
    engine = EvolutionEngine(config=EvolutionConfig(curriculum_stall_threshold=3))
    engine.champion_level = 13
    engine.generations_stalled = 3
    assert engine._curriculum_verify_threshold() == 0.875


def test_l13_extended_stall_allows_75():
    engine = EvolutionEngine(config=EvolutionConfig(curriculum_stall_threshold=3))
    engine.champion_level = 13
    engine.generations_stalled = 6
    assert engine._curriculum_verify_threshold() == 0.75


def test_try_curriculum_advance_at_partial_875():
    engine = EvolutionEngine(config=EvolutionConfig(curriculum_stall_threshold=3, auto_curriculum=True))
    engine.champion_level = 13
    engine.generations_stalled = 3
    engine._champion_raw_task_score = 130.0
    engine._curriculum_fitness_gate_passes = lambda: True
    engine._verify_champion = MagicMock(return_value=(False, 0.875))
    engine._advance_curriculum = MagicMock()

    advanced = engine._try_curriculum_advance()

    assert advanced is True
    engine._advance_curriculum.assert_called_once()


def test_end_of_run_curriculum_push():
    engine = EvolutionEngine(config=EvolutionConfig(curriculum_stall_threshold=3, auto_curriculum=True))
    engine.champion_level = 17
    engine.generations_stalled = 0
    engine.champion_fitness = 130.0
    engine._champion_raw_task_score = 130.0
    engine._champion_verify_score = 0.875
    engine._verify_champion = MagicMock(return_value=(False, 0.875))
    engine._try_curriculum_advance = MagicMock(side_effect=[True, False])
    engine._try_forced_curriculum_escape = MagicMock(return_value=False)

    assert engine._try_end_of_run_curriculum_push() is True
    assert engine._try_curriculum_advance.call_count == 2
