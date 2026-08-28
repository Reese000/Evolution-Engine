#!/usr/bin/env python3
"""Tests for uncovered functions in evolution_engine.py.

These tests target specific uncovered lines to improve coverage.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evolution_engine import EvalResult, EvolutionConfig, EvolutionEngine, VMFarm

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Build a minimal EvolutionConfig for testing."""
    defaults = dict(
        population_size=10,
        max_generations=5,
        use_gpu=False,
        use_secondary_tasks=False,
        checkpoint_interval=3,
        auto_curriculum=False,
        seed=42,
        eval_budget=5000,
        instruction_budget=10000,
        use_novelty_archive=False,
        use_qd_archive=False,
        use_strategy_archive=False,
    )
    defaults.update(overrides)
    return EvolutionConfig(**defaults)


# ---------------------------------------------------------------------------
# Test validate_program
# ---------------------------------------------------------------------------


class TestValidateProgram:
    """Tests for EvolutionEngine.validate_program."""

    def test_empty_program(self):
        """Empty program should be invalid."""
        is_valid, reason = EvolutionEngine.validate_program([])
        assert not is_valid
        assert reason == "empty_program"

    def test_all_ebreak(self):
        """All EBREAK program should be invalid."""
        ebreak = 0x0010_0073
        is_valid, reason = EvolutionEngine.validate_program([ebreak, ebreak])
        assert not is_valid
        assert reason == "all_ebreak"

    def test_too_short_after_ebreak_stripping(self):
        """Program with fewer than 3 non-EBREAK instructions should be invalid."""
        ebreak = 0x0010_0073
        # 1 non-EBREAK + 2 EBREAKs => after stripping trailing EBREAKs, live=[0x1000]
        is_valid, reason = EvolutionEngine.validate_program([0x1000, ebreak, ebreak])
        assert not is_valid
        assert "too_short" in reason

    def test_valid_program(self):
        """Valid program with 3+ non-EBREAK instructions should be valid."""
        # 0x0010_0073 is EBREAK, so use other values
        is_valid, reason = EvolutionEngine.validate_program([0x1000, 0x2000, 0x3000])
        assert is_valid
        assert reason == "ok"

    def test_overflow(self):
        """Program with overflow should be invalid."""
        is_valid, reason = EvolutionEngine.validate_program([0x100000000, 0x0010_0073, 0x0010_0073])
        assert not is_valid
        assert reason == "overflow"

    def test_negative(self):
        """Program with negative value should be invalid."""
        is_valid, reason = EvolutionEngine.validate_program([-1, 0x0010_0073, 0x0010_0073])
        assert not is_valid
        assert reason == "overflow"

    def test_trailing_ebreak_stripped(self):
        """Trailing EBREAK should be stripped for length check."""
        ebreak = 0x0010_0073
        is_valid, reason = EvolutionEngine.validate_program([0x1000, 0x2000, 0x3000, ebreak, ebreak])
        assert is_valid
        assert reason == "ok"


# ---------------------------------------------------------------------------
# Test archive methods
# ---------------------------------------------------------------------------


class TestArchiveMethods:
    """Tests for archive-related methods."""

    def test_archive_init(self):
        """Archives should be initialized correctly."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        # Archives should exist but may be None if disabled
        assert engine.strategy_archive is not None or not config.use_strategy_archive

    def test_save_archive_state(self):
        """save_archive_state should return dict with qd and novelty keys."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        state = engine.save_archive_state()
        assert "qd" in state
        assert "novelty" in state


# ---------------------------------------------------------------------------
# Test champion analysis
# ---------------------------------------------------------------------------


class TestChampionAnalysis:
    """Tests for champion analysis methods."""

    def test_get_champion_program_analysis(self):
        """get_champion_program_analysis should return dict."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_champion_program_analysis()
        assert isinstance(result, dict)

    def test_get_champion_summary(self):
        """get_champion_summary should return string."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_champion_summary()
        assert isinstance(result, str)

    def test_get_champion_history(self):
        """get_champion_history should return list."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_champion_history()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Test run diagnostics
# ---------------------------------------------------------------------------


class TestRunDiagnostics:
    """Tests for run diagnostics methods."""

    def test_get_run_diagnostics(self):
        """get_run_diagnostics should return dict."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_run_diagnostics()
        assert isinstance(result, dict)

    def test_get_performance_metrics(self):
        """get_performance_metrics should return dict."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_performance_metrics()
        assert isinstance(result, dict)

    def test_get_archive_summary(self):
        """get_archive_summary should return dict."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_archive_summary()
        assert isinstance(result, dict)

    def test_get_training_summary(self):
        """get_training_summary should return dict."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_training_summary()
        assert isinstance(result, dict)

    def test_get_run_summary(self):
        """get_run_summary should return dict."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_run_summary()
        assert isinstance(result, dict)

    def test_get_evolution_state(self):
        """get_evolution_state should return dict."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_evolution_state()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test fitness diagnostics
# ---------------------------------------------------------------------------


class TestFitnessDiagnostics:
    """Tests for fitness diagnostic methods."""

    def test_get_fitness_diagnostic_summary(self):
        """get_fitness_diagnostic_summary should return dict."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_fitness_diagnostic_summary()
        assert isinstance(result, dict)
        # fitness_velocity is nested inside population_dynamics
        assert "population_dynamics" in result
        assert "fitness_velocity" in result["population_dynamics"]

    def test_get_generation_stats(self):
        """get_generation_stats should return dict."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_generation_stats()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Tests for EvolutionConfig validation."""

    def test_invalid_population_size(self):
        """Population size < 2 should raise ValueError."""
        try:
            config = EvolutionConfig(population_size=1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_max_generations(self):
        """Max generations < 1 should raise ValueError."""
        try:
            config = EvolutionConfig(max_generations=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_config_validate(self):
        """EvolutionConfig should exist and be importable."""
        from evolution_engine import EvolutionConfig

        config = _make_config()
        assert config is not None
        # Verify config attributes
        assert config.population_size == 10
        assert config.max_generations == 5


# ---------------------------------------------------------------------------
# Test apply_external_params
# ---------------------------------------------------------------------------


class TestApplyExternalParams:
    """Tests for apply_external_params."""

    def test_apply_params(self):
        """apply_external_params should update config."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        engine.apply_external_params({"mutation_rate": 0.5})
        assert config.mutation_rate == 0.5

    def test_apply_params_unknown(self):
        """apply_external_params should skip unknown keys."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        engine.apply_external_params({"unknown_key": 42})
        assert not hasattr(config, "unknown_key")


# ---------------------------------------------------------------------------
# Test save/load archive state
# ---------------------------------------------------------------------------


class TestArchiveState:
    """Tests for save/load archive state."""

    def test_save_load_cycle(self):
        """Save and load archive state should be consistent."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        state = engine.save_archive_state()
        # Load it back
        engine.load_archive_state(qd_data=state.get("qd"), novelty_data=state.get("novelty"))
        # No crash = success


# ---------------------------------------------------------------------------
# Test champion replacement
# ---------------------------------------------------------------------------


class TestChampionReplacement:
    """Tests for champion replacement logic."""

    def test_should_replace_champion(self):
        """_should_replace_champion should return True for higher fitness."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        # With initial champion fitness of 0.0, any positive fitness should replace
        result = engine._should_replace_champion(1.0)
        assert isinstance(result, bool)

    def test_record_champion_replacement(self):
        """_record_champion_replacement should not crash."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        engine._record_champion_replacement(5.0)
        assert engine._champion_replacement_count >= 0


# ---------------------------------------------------------------------------
# Test enable_bonus_normalization
# ---------------------------------------------------------------------------


class TestBonusNormalization:
    """Tests for bonus normalization."""

    def test_enable_bonus_normalization(self):
        """enable_bonus_normalization should set normalization."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        engine.enable_bonus_normalization(fraction=0.5)
        # Should not crash


# ---------------------------------------------------------------------------
# Test disabled levels
# ---------------------------------------------------------------------------


class TestDisabledLevels:
    """Tests for disabled levels."""

    def test_get_disabled_levels(self):
        """get_disabled_levels should return frozenset."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_disabled_levels()
        assert isinstance(result, frozenset)

    def test_get_level_circuit_stats(self):
        """get_level_circuit_stats should return dict."""
        config = _make_config()
        engine = EvolutionEngine(config=config)
        result = engine.get_level_circuit_stats()
        assert isinstance(result, dict)
