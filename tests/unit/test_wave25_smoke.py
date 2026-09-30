#!/usr/bin/env python3
"""Wave 25 final smoke tests — engine and config health checks.

Lightweight tests that verify core construction and API surfaces are
wired correctly.  No actual evolution is run.  Keeps tests fast.
"""
from __future__ import annotations

from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine

# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify EvolutionEngine basic construction and interface via conftest fixture."""

    def test_engine_creation_with_default_config(self, engine: EvolutionEngine) -> None:
        """Engine can be created with default config via engine fixture."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)
        assert isinstance(engine.config, EvolutionConfig)

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine) -> None:
        """Engine exposes all critical attributes expected by the system."""
        expected_attrs = [
            "config",
            "champion",
            "champion_trace",
            "champion_fitness",
            "champion_level",
            "current_generation",
            "generations_stalled",
            "task_instance",
            "novelty_archive",
            "qd_archive",
            "_hall_of_fame",
            "vm_farm",
            "curriculum",
            "strategy_archive",
            "best_fitness_history",
            "circuit_breaker",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"engine missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine) -> None:
        """engine.run is callable (we do NOT invoke it — too slow)."""
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self, engine: EvolutionEngine) -> None:
        """get_run_diagnostics returns a dict with expected top-level keys."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        for key in (
            "champion_level",
            "current_generation",
            "champion_fitness",
            "generations_stalled",
            "champion_words",
        ):
            assert key in diag, f"diagnostics missing key: {key}"

    def test_engine_get_performance_metrics_returns_dict(self, engine: EvolutionEngine) -> None:
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
            "hof_size",
        ]
        for key in expected_keys:
            assert key in metrics, f"performance metrics missing key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify EvolutionSection construction, summary, and validation."""

    def test_config_creation_with_defaults(self) -> None:
        """EvolutionSection can be created with all defaults."""
        cfg = EvolutionSection()
        assert cfg is not None
        assert isinstance(cfg, EvolutionSection)

    def test_config_get_config_summary_returns_dict_with_expected_keys(self) -> None:
        """get_config_summary returns a dict containing all expected keys."""
        cfg = EvolutionSection()
        summary = cfg.get_config_summary()
        assert isinstance(summary, dict)
        for key in (
            "seed",
            "population_size",
            "mutation_rate",
            "novelty_weight",
            "qd_bonus_weight",
            "eval_budget",
            "archives_enabled",
        ):
            assert key in summary, f"config summary missing key: {key}"

    def test_config_validate_config_returns_list(self) -> None:
        """validate_config returns a list (empty on defaults)."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert isinstance(warnings, list)

    def test_config_no_warnings_on_default(self) -> None:
        """Default config passes validation without warnings."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert len(warnings) == 0, f"Unexpected warnings on default config: {warnings}"
