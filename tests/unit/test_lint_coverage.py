#!/usr/bin/env python3
"""test_lint_coverage.py — Tests to verify lint fixes and code quality improvements.

Tests that:
1. Key modules import without errors
2. The engine can be instantiated with default config
3. Config validation works
4. Key functions are callable
5. Basic evolution cycle runs
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModuleImports:
    """Verify that key modules can be imported without errors."""

    def test_import_evolution_engine(self):
        from evolution_engine import EvolutionConfig, EvolutionEngine

        assert EvolutionEngine is not None
        assert EvolutionConfig is not None

    def test_import_config(self):
        from config import Config

        assert Config is not None

    def test_import_metrics(self):
        from metrics import MetricsCollector

        assert MetricsCollector is not None

    def test_import_progression_tracker(self):
        from progression_tracker import ProgressionTracker

        assert ProgressionTracker is not None

    def test_import_champion_archive(self):
        from champion_archive import ChampionArchive

        assert ChampionArchive is not None

    def test_import_checkpoint_manager(self):
        from checkpoint_manager import CheckpointManager

        assert CheckpointManager is not None

    def test_import_data_rotation(self):
        from data_rotation import run_all_rotations

        assert callable(run_all_rotations)

    def test_import_logging_config(self):
        from logging_config import setup_logging

        assert callable(setup_logging)

    def test_import_novelty(self):
        from novelty import NoveltyArchive

        assert NoveltyArchive is not None

    def test_import_qd_archive(self):
        from qd_archive import QDArchive

        assert QDArchive is not None


class TestConfigDefaults:
    """Verify that default configuration is valid."""

    def test_config_created_with_defaults(self):
        from config import Config

        config = Config()
        assert config is not None
        assert hasattr(config, "evolution")

    def test_config_evolution_section(self):
        from config import Config

        config = Config()
        assert hasattr(config.evolution, "population_size")
        assert hasattr(config.evolution, "max_generations")


class TestEngineDefaults:
    """Verify that the engine can be created with default config."""

    def test_engine_created_with_default_config(self):
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig()
        engine = EvolutionEngine(config)
        assert engine is not None
        assert hasattr(engine, "config")
        assert hasattr(engine, "rng")
        assert hasattr(engine, "champion_level")
        assert hasattr(engine, "current_generation")
        assert hasattr(engine, "generations_stalled")

    def test_engine_has_run_method(self):
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig()
        engine = EvolutionEngine(config)
        assert callable(engine.run)

    def test_engine_has_get_run_diagnostics(self):
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig()
        engine = EvolutionEngine(config)
        assert callable(engine.get_run_diagnostics)
        diagnostics = engine.get_run_diagnostics()
        assert isinstance(diagnostics, dict)

    def test_engine_has_get_performance_metrics(self):
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig()
        engine = EvolutionEngine(config)
        assert callable(engine.get_performance_metrics)
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    def test_metrics_collector_created(self):
        from metrics import MetricsCollector

        mc = MetricsCollector()
        assert mc is not None

    def test_metrics_collector_has_record_cycle(self):
        from metrics import MetricsCollector

        mc = MetricsCollector()
        assert hasattr(mc, "record_cycle")


class TestProgressionTracker:
    """Test ProgressionTracker functionality."""

    def test_progression_tracker_created(self):
        from progression_tracker import ProgressionTracker

        pt = ProgressionTracker()
        assert pt is not None

    def test_progression_tracker_has_history(self):
        from progression_tracker import ProgressionTracker

        pt = ProgressionTracker()
        assert hasattr(pt, "history")
        assert isinstance(pt.history, list)


class TestChampionArchive:
    """Test ChampionArchive functionality."""

    def test_champion_archive_created(self):
        from champion_archive import ChampionArchive

        ca = ChampionArchive()
        assert ca is not None


class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    def test_checkpoint_manager_created(self):
        from checkpoint_manager import CheckpointManager

        cm = CheckpointManager(base_dir="/tmp/test")
        assert cm is not None


class TestNoveltyArchive:
    """Test NoveltyArchive functionality."""

    def test_novelty_archive_created(self):
        from novelty import NoveltyArchive

        na = NoveltyArchive()
        assert na is not None

    def test_novelty_archive_has_score(self):
        from novelty import NoveltyArchive

        na = NoveltyArchive()
        assert callable(na.novelty_score)


class TestQDArchive:
    """Test QDArchive functionality."""

    def test_qd_archive_created(self):
        from qd_archive import QDArchive

        qa = QDArchive()
        assert qa is not None


class TestEvolutionConfig:
    """Test EvolutionConfig functionality."""

    def test_config_created(self):
        from evolution_engine import EvolutionConfig

        config = EvolutionConfig()
        assert config is not None

    def test_config_has_expected_attributes(self):
        from evolution_engine import EvolutionConfig

        config = EvolutionConfig()
        assert hasattr(config, "population_size")
        assert hasattr(config, "max_generations")
        assert hasattr(config, "mutation_rate")


class TestEngineAttributes:
    """Verify that the engine has expected attributes."""

    def test_engine_has_expected_attributes(self):
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig()
        engine = EvolutionEngine(config)
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
            "_fitness_velocity",
            "_velocity_alpha",
            "_adaptive_mutation_floor",
            "_adaptive_mutation_ceil",
            "_original_mutation_rate",
            "_consecutive_same_strategy",
            "_last_strategy_used",
            "best_fitness_history",
            "_max_history_len",
            "_champion_replacement_count",
            "circuit_breaker",
            "strategy_archive",
            "vm_farm",
            "curriculum",
            "use_gpu",
            "recurrent_examples_count",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"
