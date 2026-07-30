#!/usr/bin/env python3
"""Wave 27 — Final smoke tests.

Two fast, read-only test classes:

1. TestEngineHealth — verify EvolutionEngine can be created with default
   config, has all expected attributes, and exposes callable API methods.

2. TestConfigHealth — verify EvolutionSection (config layer) can be
   created with defaults, returns valid summaries, validates cleanly,
   and produces no warnings on default config.

Uses conftest fixtures (_make_engine). All tests are fast — no actual
evolution is run. No existing files are modified.
"""
from __future__ import annotations

import pytest

from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth — engine creation, attributes, callable methods
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineHealth:
    """Verify EvolutionEngine can be created and has all expected attributes."""

    def test_engine_creation_with_default_config(self, engine: EvolutionEngine) -> None:
        """Engine created via conftest fixture should be a valid instance."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)
        assert isinstance(engine.config, EvolutionConfig)

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine) -> None:
        """Engine exposes all critical attributes expected by the system."""
        expected_attrs = [
            "config",
            "rng",
            "champion_level",
            "current_generation",
            "generations_stalled",
            "champion",
            "champion_trace",
            "champion_fitness",
            "novelty_archive",
            "qd_archive",
            "strategy_archive",
            "_hall_of_fame",
            "_fitness_velocity",
            "_champion_replacement_count",
            "best_fitness_history",
            "circuit_breaker",
            "vm_farm",
            "curriculum",
            "task_instance",
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
            "hall_of_fame_size",
        ):
            assert key in diag, f"diagnostics missing key: {key}"

    def test_engine_get_performance_metrics_returns_dict(self, engine: EvolutionEngine) -> None:
        """get_performance_metrics returns a dict with expected keys."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        for key in (
            "current_generation",
            "population_size",
            "elite_ratio",
            "eval_budget",
            "mutation_rate",
        ):
            assert key in metrics, f"performance metrics missing key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth — config creation, summary, validation
# ══════════════════════════════════════════════════════════════════════════════

_EXPECTED_SUMMARY_KEYS: list[str] = [
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


class TestConfigHealth:
    """Verify EvolutionSection can be created, summarized, and validated."""

    @pytest.fixture
    def default_section(self) -> EvolutionSection:
        """Create an EvolutionSection with all defaults."""
        return EvolutionSection()

    def test_config_creation_with_defaults(self, default_section: EvolutionSection) -> None:
        """EvolutionSection() must succeed with default arguments."""
        assert default_section is not None
        assert isinstance(default_section, EvolutionSection)

    def test_get_config_summary_returns_dict(self, default_section: EvolutionSection) -> None:
        """get_config_summary must return a dict."""
        summary = default_section.get_config_summary()
        assert isinstance(summary, dict)

    def test_get_config_summary_has_expected_keys(self, default_section: EvolutionSection) -> None:
        """get_config_summary must include all expected keys."""
        summary = default_section.get_config_summary()
        actual_keys = set(summary.keys())
        missing = set(_EXPECTED_SUMMARY_KEYS) - actual_keys
        assert not missing, f"get_config_summary missing keys: {missing}"

    def test_validate_config_returns_list(self, default_section: EvolutionSection) -> None:
        """validate_config must return a list."""
        warnings = default_section.validate_config()
        assert isinstance(warnings, list)

    def test_no_warnings_on_default_config(self, default_section: EvolutionSection) -> None:
        """Default config should produce no warnings."""
        warnings = default_section.validate_config()
        assert warnings == [], f"Default config produced unexpected warnings: {warnings}"
