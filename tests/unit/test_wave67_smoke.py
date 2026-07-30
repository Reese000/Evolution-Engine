#!/usr/bin/env python3
"""Wave 67 final smoke tests.

Fast sanity checks that the engine and config classes can be instantiated
and that their public API methods return the expected types.  No actual
evolution is run.
"""

from __future__ import annotations

import pytest

from evolution_engine import EvolutionEngine, EvolutionConfig
from config import EvolutionSection


# ══════════════════════════════════════════════════════════════════════════════
# TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify EvolutionEngine instantiation and public API shapes."""

    def test_engine_creation_default_config(self):
        """EvolutionEngine can be created with default config."""
        from tests.conftest import make_engine

        engine = make_engine()
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_expected_attributes(self):
        """Engine carries all attributes expected by the hot path."""
        from tests.conftest import make_engine

        engine = make_engine()

        expected_attrs = [
            "config",
            "rng",
            "champion_level",
            "current_generation",
            "generations_stalled",
            "champion",
            "champion_trace",
            "champion_fitness",
            "task_instance",
            "novelty_archive",
            "qd_archive",
            "_hall_of_fame",
            "best_fitness_history",
            "vm_farm",
            "curriculum",
            "use_gpu",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine):
        """engine.run is a callable (we don't invoke it here)."""
        assert callable(getattr(engine, "run", None))

    def test_engine_get_run_diagnostics_returns_dict(self, engine: EvolutionEngine):
        """get_run_diagnostics returns a dict with expected keys."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        expected_keys = [
            "champion_level",
            "champion_fitness",
            "champion_words",
            "current_generation",
            "generations_stalled",
            "mutation_rate",
            "population_size",
            "elite_ratio",
        ]
        for key in expected_keys:
            assert key in diag, f"Missing key in diagnostics: {key}"

    def test_engine_get_performance_metrics_returns_dict(
        self, engine: EvolutionEngine
    ):
        """get_performance_metrics returns a dict with expected keys."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        expected_keys = [
            "current_generation",
            "population_size",
            "elite_ratio",
            "eval_budget",
            "mutation_rate",
            "champion_words",
            "novelty_weight",
            "qd_bonus_weight",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key in metrics: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify EvolutionSection (config) instantiation and introspection methods."""

    def test_config_creation_defaults(self):
        """EvolutionConfig (evolution_engine) can be created with all defaults."""
        cfg = EvolutionConfig()
        assert cfg is not None
        assert isinstance(cfg, EvolutionConfig)

    def test_config_summary_returns_expected_keys(self):
        """EvolutionSection.get_config_summary returns a dict with expected keys."""
        cfg = EvolutionSection()
        summary = cfg.get_config_summary()
        assert isinstance(summary, dict)

        expected_keys = [
            "seed",
            "use_gpu",
            "auto_curriculum",
            "archives_enabled",
            "novelty_weight",
            "qd_bonus_weight",
            "mutation_rate",
            "population_size",
            "eval_budget",
            "max_consecutive_same_strategy",
            "total_config_fields",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key in config summary: {key}"

    def test_validate_config_returns_list(self):
        """EvolutionSection.validate_config returns a list (possibly empty)."""
        cfg = EvolutionSection()
        result = cfg.validate_config()
        assert isinstance(result, list)

    def test_default_config_no_warnings(self):
        """Default EvolutionSection config should produce no validation warnings."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert warnings == [], f"Unexpected warnings on default config: {warnings}"
