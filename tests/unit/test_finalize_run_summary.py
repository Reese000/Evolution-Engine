"""Regression: _finalize_run_summary must not reference undefined effective_verify."""

import os
from unittest.mock import MagicMock

os.environ.setdefault("EE_USE_GPU", "0")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_finalize_run_summary_with_progress_controller():
    from evolution.progress_controller import ProgressController
    from evolution_engine import EvolutionConfig, EvolutionEngine

    engine = EvolutionEngine(EvolutionConfig(curriculum_level=30, auto_curriculum=True))
    engine.champion_level = 30
    engine.champion = [0x00000013, 0x00100073]
    engine.champion_fitness = 100.0
    engine.champion_trace = {"status": "ok", "final_regs": {}, "final_memory": {}, "instructions_count": 2}
    engine._champion_raw_task_score = 50.0
    engine._last_max_pop_verify = 1.0
    engine.progress_controller = ProgressController(PROJECT_ROOT)
    engine._verify_champion = MagicMock(return_value=(True, 1.0))
    engine._evaluate = MagicMock(return_value=engine.champion_trace)

    summary = {"generations_completed": 0}
    out = engine._finalize_run_summary(summary, ng=10, start_time=0.0)

    assert out["champion_verify_score"] == 1.0
    assert out["champion_verified"] is True
    assert "final_fitness" in out
