#!/usr/bin/env python3
"""Wave 66 — Final smoke tests for engine and config health.

Fast, no-evolution tests that verify core objects can be instantiated
and their diagnostic/reporting APIs return the expected shapes.
"""
from __future__ import annotations

from config import EvolutionSection
from evolution_engine import EvolutionEngine

# ══════════════════════════════════════════════════════════════════════════════
# TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify the EvolutionEngine wiring via the shared _make_engine fixture."""

    def test_engine_creation(self, engine: EvolutionEngine) -> None:
        """Engine can be created with default config (via conftest fixture)."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine) -> None:
        """Engine exposes all critical attributes after construction."""
        expected_attrs = [
            "config",
            "champion",
            "champion_trace",
            "champion_fitness",
            "champion_level",
            "current_generation",
            "generations_stalled",
            "novelty_archive",
            "qd_archive",
            "strategy_archive",
            "_hall_of_fame",
            "best_fitness_history",
            "task_instance",
            "vm_farm",
            "curriculum",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine) -> None:
        """engine.run is a callable method (without actually invoking it)."""
        assert callable(getattr(engine, "run", None)), "engine.run is not callable"

    def test_get_run_diagnostics_returns_dict(self, engine: EvolutionEngine) -> None:
        """get_run_diagnostics returns a dict with expected keys."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        required_keys = [
            "champion_level",
            "champion_fitness",
            "champion_words",
            "generations_stalled",
            "current_generation",
            "novelty_archive_size",
            "qd_archive_size",
            "strategy_archive_records",
            "hall_of_fame_size",
            "fitness_velocity",
            "mutation_rate",
            "population_size",
            "elite_ratio",
        ]
        for key in required_keys:
            assert key in diag, f"Missing diagnostic key: {key}"

    def test_get_performance_metrics_returns_dict(self, engine: EvolutionEngine) -> None:
        """get_performance_metrics returns a dict with expected keys."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        required_keys = [
            "current_generation",
            "population_size",
            "elite_ratio",
            "eval_budget",
            "model_guided_ratio",
            "mutation_rate",
            "champion_words",
            "novelty_weight",
            "qd_bonus_weight",
            "total_recurrent_examples",
            "hof_size",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing metrics key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify EvolutionSection (config.py) config health APIs."""

    def test_config_creation_defaults(self) -> None:
        """EvolutionSection can be created with defaults."""
        cfg = EvolutionSection()
        assert cfg is not None
        assert isinstance(cfg, EvolutionSection)

    def test_get_config_summary_keys(self) -> None:
        """get_config_summary returns dict with expected keys."""
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
            assert key in summary, f"Missing summary key: {key}"

    def test_validate_config_returns_list(self) -> None:
        """validate_config returns a list (possibly empty)."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert isinstance(warnings, list)

    def test_no_warnings_on_default_config(self) -> None:
        """Default EvolutionSection produces no validation warnings."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert warnings == [], f"Unexpected warnings on default config: {warnings}"
