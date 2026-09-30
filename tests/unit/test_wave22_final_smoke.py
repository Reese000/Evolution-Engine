#!/usr/bin/env python3
"""Wave 22: Final smoke tests — engine instantiation and config health.

Fast, non-evolving smoke tests that verify core construction and API
surfaces are wired correctly. No actual evolution is run.
"""
from __future__ import annotations

# Ensure project root is on sys.path
from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine

# ══════════════════════════════════════════════════════════════════════
# 1. TestEngineInstantiation
# ══════════════════════════════════════════════════════════════════════


class TestEngineInstantiation:
    """Verify EvolutionEngine can be created with default config and
    exposes all expected attributes and methods."""

    def test_engine_created_with_default_config(self, engine: EvolutionEngine):
        """Engine created via make_engine fixture has a default config."""
        assert isinstance(engine.config, EvolutionConfig)

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine):
        """Engine exposes the full set of attributes needed for evolution."""
        expected_attrs = [
            "config",
            "champion",
            "champion_trace",
            "champion_fitness",
            "champion_level",
            "current_generation",
            "generations_stalled",
            "rng",
            "task_instance",
            "novelty_archive",
            "qd_archive",
            "strategy_archive",
            "best_fitness_history",
            "circuit_breaker",
            "_hall_of_fame",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine):
        """engine.run is a callable method."""
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self, engine: EvolutionEngine):
        """get_run_diagnostics returns a dict with at least some keys."""
        diagnostics = engine.get_run_diagnostics()
        assert isinstance(diagnostics, dict)
        # The diagnostics dict should be non-empty for a freshly built engine
        assert len(diagnostics) > 0


# ══════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ══════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify EvolutionSection (config.py) can be created, summarized,
    and validated with sensible defaults."""

    def test_config_created_with_defaults(self):
        """EvolutionSection() constructs without errors."""
        cfg = EvolutionSection()
        assert isinstance(cfg, EvolutionSection)
        assert cfg.population_size > 0
        assert cfg.eval_budget > 0

    def test_get_config_summary_returns_dict_with_expected_keys(self):
        """get_config_summary returns a dict containing all expected keys."""
        cfg = EvolutionSection()
        summary = cfg.get_config_summary()
        assert isinstance(summary, dict)
        expected_keys = {
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
        }
        missing = expected_keys - summary.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_validate_config_returns_list(self):
        """validate_config returns a list (possibly empty on defaults)."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert isinstance(warnings, list)

    def test_no_warnings_on_default_config(self):
        """Default config values produce no validation warnings."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert warnings == [], f"Unexpected warnings on default config: {warnings}"
