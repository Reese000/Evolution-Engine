"""MetaController behavior at certified L30 terminal plateau."""

import os
import tempfile

import pytest

from meta_controller import MetaController, TelemetrySnapshot


def _snapshot(
    cycle: int,
    fitness: float = 264.04,
    level: int = 30,
    champion_words: int = 73,
    val_loss: float | None = None,
) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        cycle=cycle,
        fitness=fitness,
        level=level,
        champion_words=champion_words,
        val_loss=val_loss,
        diversity=0.5,
        novelty_size=0,
        qd_occupied=0,
        qd_total=100,
        traces_count=0,
        retrained=False,
        generations_completed=50,
    )


def test_shorter_words_count_as_fitness_improvement_at_l30(monkeypatch):
    monkeypatch.setenv("EE_MAX_LEVEL", "30")
    with tempfile.TemporaryDirectory() as tmp:
        mc = MetaController(checkpoint_dir=tmp)
        mc.update_cycle(_snapshot(1, fitness=264.04, champion_words=73))
        assert mc.cycles_since_fitness_improvement == 0
        mc.update_cycle(_snapshot(2, fitness=264.04, champion_words=73))
        assert mc.cycles_since_fitness_improvement == 1
        mc.update_cycle(_snapshot(3, fitness=264.04, champion_words=71))
        assert mc.cycles_since_fitness_improvement == 0
        assert mc._best_words_at_best_fitness == 71


def test_scan_floor_skips_level_stall_for_compound_stall(monkeypatch):
    """At scan floor, prom refinement should not compound-stall on level stall alone."""
    monkeypatch.setenv("EE_MAX_LEVEL", "30")
    with tempfile.TemporaryDirectory() as tmp:
        mc = MetaController(checkpoint_dir=tmp, level_stall_cycles=3)
        monkeypatch.setattr(
            "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
            lambda words: int(words) == 27,
        )
        for i in range(6):
            mc.update_cycle(
                _snapshot(
                    i + 1,
                    fitness=325.8257,
                    level=30,
                    champion_words=27,
                    val_loss=0.05,
                )
            )
        assert mc.cycles_since_level_up >= mc.level_stall_cycles
        assert mc.cycles_since_fitness_improvement >= mc.fitness_plateau_cycles
        state = mc.detect_stagnation()
        assert state.value != "compound_stall"
        assert state.value == "fitness_plateau"


def test_should_rollback_suppressed_on_terminal_plateau(monkeypatch):
    monkeypatch.setenv("EE_MAX_LEVEL", "30")
    with tempfile.TemporaryDirectory() as tmp:
        mc = MetaController(checkpoint_dir=tmp)
        for i in range(5):
            mc.update_cycle(_snapshot(i + 1, fitness=264.04, champion_words=73, val_loss=0.05))
        mc.cycles_since_fitness_improvement = 5
        mc._compound_stall_counter = 5
        assert mc._terminal_plateau_active()
        assert mc.should_rollback() is False
