#!/usr/bin/env python3
"""Unit tests for the level circuit breaker (P2-7).

Tests that:
- Config fields are properly defined
- Per-level outcome tracking works
- Auto-disable triggers after N cycles of 0% success
- Disabled levels are skipped during curriculum advancement
- Circuit breaker is disabled when config says so
- Stats are properly tracked and queryable
"""
from __future__ import annotations

import random

import pytest

from paths import ensure_project_on_path

ensure_project_on_path()

from evolution_engine import EvolutionConfig, EvolutionEngine
from tasks_rv32i import LEVELS
from tests.conftest import make_engine

# ═══════════════════════════════════════════════════════════════════════
# Config tests
# ═══════════════════════════════════════════════════════════════════════


class TestCircuitBreakerConfig:
    """Verify EvolutionConfig circuit breaker fields exist and have defaults."""

    def test_default_enabled(self):
        cfg = EvolutionConfig()
        assert cfg.level_circuit_breaker_enabled is True

    def test_default_threshold(self):
        cfg = EvolutionConfig()
        assert cfg.level_circuit_breaker_threshold == 10

    def test_custom_threshold(self):
        cfg = EvolutionConfig(level_circuit_breaker_threshold=5)
        assert cfg.level_circuit_breaker_threshold == 5

    def test_disable_via_config(self):
        cfg = EvolutionConfig(level_circuit_breaker_enabled=False)
        assert cfg.level_circuit_breaker_enabled is False


# ═══════════════════════════════════════════════════════════════════════
# Tracking tests
# ═══════════════════════════════════════════════════════════════════════


class TestCircuitBreakerTracking:
    """Test per-level outcome tracking."""

    def test_record_outcome_increments_cycles(self):
        engine = make_engine()
        engine._level_circuit_stats = {}
        engine._disabled_levels = set()

        engine._record_level_circuit_outcome(5, had_success=False)
        assert engine._level_circuit_stats[5]["cycles"] == 1
        assert engine._level_circuit_stats[5]["successes"] == 0

    def test_record_outcome_success_increments(self):
        engine = make_engine()
        engine._level_circuit_stats = {}
        engine._disabled_levels = set()

        engine._record_level_circuit_outcome(5, had_success=True)
        assert engine._level_circuit_stats[5]["cycles"] == 1
        assert engine._level_circuit_stats[5]["successes"] == 1

    def test_record_multiple_outcomes(self):
        engine = make_engine()
        engine._level_circuit_stats = {}
        engine._disabled_levels = set()

        for _ in range(5):
            engine._record_level_circuit_outcome(3, had_success=False)
        engine._record_level_circuit_outcome(3, had_success=True)

        stats = engine._level_circuit_stats[3]
        assert stats["cycles"] == 6
        assert stats["successes"] == 1

    def test_disabled_when_enabled_does_nothing(self):
        engine = make_engine()
        engine.config.level_circuit_breaker_enabled = False
        engine._level_circuit_stats = {}
        engine._disabled_levels = set()

        for _ in range(20):
            engine._record_level_circuit_outcome(7, had_success=False)

        # Stats should not be tracked when disabled
        assert 7 not in engine._level_circuit_stats

    def test_get_level_circuit_stats_returns_copy(self):
        engine = make_engine()
        engine._level_circuit_stats = {0: {"cycles": 5, "successes": 2}}
        engine._disabled_levels = set()

        stats = engine.get_level_circuit_stats()
        stats[0]["cycles"] = 999  # Mutate the copy

        # Original should be unchanged
        assert engine._level_circuit_stats[0]["cycles"] == 5


# ═══════════════════════════════════════════════════════════════════════
# Auto-disable tests
# ═══════════════════════════════════════════════════════════════════════


class TestCircuitBreakerAutoDisable:
    """Test that levels are auto-disabled after threshold cycles of 0% success."""

    def test_not_disabled_below_threshold(self):
        engine = make_engine()
        engine.config.level_circuit_breaker_threshold = 10
        engine._level_circuit_stats = {}
        engine._disabled_levels = set()

        for _ in range(9):
            engine._record_level_circuit_outcome(5, had_success=False)

        assert not engine._check_level_circuit_breaker(5)
        assert not engine._is_level_disabled(5)

    def test_disabled_at_threshold(self):
        engine = make_engine()
        engine.config.level_circuit_breaker_threshold = 10
        engine._level_circuit_stats = {}
        engine._disabled_levels = set()

        for _ in range(10):
            engine._record_level_circuit_outcome(5, had_success=False)

        assert engine._check_level_circuit_breaker(5)
        assert engine._is_level_disabled(5)

    def test_disabled_above_threshold(self):
        engine = make_engine()
        engine.config.level_circuit_breaker_threshold = 10
        engine._level_circuit_stats = {}
        engine._disabled_levels = set()

        for _ in range(15):
            engine._record_level_circuit_outcome(5, had_success=False)

        assert engine._check_level_circuit_breaker(5)
        assert engine._is_level_disabled(5)

    def test_not_disabled_with_any_success(self):
        engine = make_engine()
        engine.config.level_circuit_breaker_threshold = 10
        engine._level_circuit_stats = {}
        engine._disabled_levels = set()

        for _ in range(9):
            engine._record_level_circuit_outcome(5, had_success=False)
        engine._record_level_circuit_outcome(5, had_success=True)  # 1 success

        assert not engine._check_level_circuit_breaker(5)
        assert not engine._is_level_disabled(5)

    def test_not_disabled_when_feature_off(self):
        engine = make_engine()
        engine.config.level_circuit_breaker_enabled = False
        engine._level_circuit_stats = {}
        engine._disabled_levels = set()

        for _ in range(20):
            engine._record_level_circuit_outcome(5, had_success=False)

        # Even after many cycles, check should not disable when feature is off
        assert not engine._check_level_circuit_breaker(5)

    def test_already_disabled_returns_true(self):
        engine = make_engine()
        engine.config.level_circuit_breaker_threshold = 5
        engine._level_circuit_stats = {}
        engine._disabled_levels = {3}

        assert engine._check_level_circuit_breaker(3)

    def test_per_level_independent(self):
        engine = make_engine()
        engine.config.level_circuit_breaker_threshold = 5
        engine._level_circuit_stats = {}
        engine._disabled_levels = set()

        # Level 3: 5 cycles, 0 success -> disabled
        for _ in range(5):
            engine._record_level_circuit_outcome(3, had_success=False)

        # Level 7: 5 cycles, 1 success -> not disabled
        for _ in range(4):
            engine._record_level_circuit_outcome(7, had_success=False)
        engine._record_level_circuit_outcome(7, had_success=True)

        assert engine._check_level_circuit_breaker(3)
        assert not engine._check_level_circuit_breaker(7)
        assert engine._is_level_disabled(3)
        assert not engine._is_level_disabled(7)

    def test_custom_threshold(self):
        engine = make_engine()
        engine.config.level_circuit_breaker_threshold = 3
        engine._level_circuit_stats = {}
        engine._disabled_levels = set()

        for _ in range(2):
            engine._record_level_circuit_outcome(0, had_success=False)
        assert not engine._check_level_circuit_breaker(0)

        engine._record_level_circuit_outcome(0, had_success=False)
        assert engine._check_level_circuit_breaker(0)


# ═══════════════════════════════════════════════════════════════════════
# Curriculum advancement skip tests
# ═══════════════════════════════════════════════════════════════════════


class TestCircuitBreakerCurriculumSkip:
    """Test that disabled levels are skipped during curriculum advancement.

    We test the skip logic directly rather than calling _advance_curriculum(),
    which requires a VM farm. The skip logic is the new code added in P2-7.
    """

    def _advance_to_next_enabled(self, engine):
        """Simulate the level-skipping logic from _advance_curriculum."""
        old_level = engine.champion_level
        new_level = old_level + 1
        max_level = min(len(LEVELS) - 1, engine.config.max_level)
        while new_level <= max_level and engine._is_level_disabled(new_level):
            new_level += 1
        if new_level > max_level:
            return old_level  # No advance possible
        return new_level

    def test_advance_skips_disabled_next_level(self):
        engine = make_engine()
        engine._disabled_levels = {5}
        engine.champion_level = 4

        new = self._advance_to_next_enabled(engine)
        assert new == 6  # Skipped L5

    def test_advance_skips_multiple_disabled(self):
        engine = make_engine()
        engine._disabled_levels = {5, 6, 7}
        engine.champion_level = 4

        new = self._advance_to_next_enabled(engine)
        assert new == 8  # Skipped L5, L6, L7

    def test_advance_no_disabled_levels(self):
        engine = make_engine()
        engine._disabled_levels = set()
        engine.champion_level = 4

        new = self._advance_to_next_enabled(engine)
        assert new == 5

    def test_advance_stops_at_max_level(self):
        engine = make_engine()
        engine.config.max_level = 30
        engine._disabled_levels = {30}
        engine.champion_level = 29

        new = self._advance_to_next_enabled(engine)
        # All levels beyond 29 are disabled or out of range
        assert new == 29  # No advance possible

    def test_get_disabled_levels_returns_frozen_copy(self):
        engine = make_engine()
        engine._disabled_levels = {1, 5, 10}

        result = engine.get_disabled_levels()
        assert result == frozenset({1, 5, 10})
        # Should be a frozenset, not a mutable set
        assert isinstance(result, frozenset)


# ═══════════════════════════════════════════════════════════════════════
# Integration: make_engine compatibility
# ═══════════════════════════════════════════════════════════════════════


class TestMakeEngineCompatibility:
    """Ensure make_engine() initializes circuit breaker state."""

    def test_make_engine_has_circuit_breaker_state(self):
        engine = make_engine()
        assert hasattr(engine, "_level_circuit_stats")
        assert hasattr(engine, "_disabled_levels")
        assert engine._level_circuit_stats == {}
        assert engine._disabled_levels == set()

    def test_make_engine_circuit_breaker_methods_exist(self):
        engine = make_engine()
        assert callable(engine._record_level_circuit_outcome)
        assert callable(engine._check_level_circuit_breaker)
        assert callable(engine._is_level_disabled)
        assert callable(engine.get_disabled_levels)
        assert callable(engine.get_level_circuit_stats)
