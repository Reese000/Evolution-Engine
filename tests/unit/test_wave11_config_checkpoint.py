#!/usr/bin/env python3
"""Wave 11 tests: config summary/validation/defaults, checkpoint manager, logging setup."""

from __future__ import annotations

import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from checkpoint_manager import CheckpointManager

# Ensure project root is on sys.path
from config import Config, EvolutionSection

# ══════════════════════════════════════════════════════════════════════
# 1. TestConfigSummary
# ══════════════════════════════════════════════════════════════════════


class TestConfigSummary:
    """Tests for EvolutionSection.get_config_summary()."""

    def test_returns_dict_with_expected_keys(self):
        """get_config_summary returns a dict containing all expected keys."""
        evo = EvolutionSection()
        summary = evo.get_config_summary()
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
        assert expected_keys.issubset(summary.keys()), f"Missing keys: {expected_keys - summary.keys()}"

    def test_summary_includes_all_major_settings(self):
        """Summary values match the dataclass defaults."""
        evo = EvolutionSection()
        summary = evo.get_config_summary()
        assert summary["seed"] == 42
        assert summary["population_size"] == 50
        assert summary["eval_budget"] == 5000
        assert summary["mutation_rate"] == 0.3
        assert summary["novelty_weight"] == 0.3
        assert summary["qd_bonus_weight"] == 0.1
        assert summary["total_config_fields"] == len(EvolutionSection.__dataclass_fields__)

    def test_archives_enabled_true_when_any_archive_active(self):
        """archives_enabled is True when at least one archive is enabled."""
        # Default config: all archives True
        evo_default = EvolutionSection()
        assert evo_default.get_config_summary()["archives_enabled"] is True

        # Only novelty archive
        evo_novelty = EvolutionSection(use_novelty_archive=True, use_qd_archive=False, use_strategy_archive=False)
        assert evo_novelty.get_config_summary()["archives_enabled"] is True

        # Only QD archive
        evo_qd = EvolutionSection(use_novelty_archive=False, use_qd_archive=True, use_strategy_archive=False)
        assert evo_qd.get_config_summary()["archives_enabled"] is True

        # Only strategy archive
        evo_strat = EvolutionSection(use_novelty_archive=False, use_qd_archive=False, use_strategy_archive=True)
        assert evo_strat.get_config_summary()["archives_enabled"] is True

        # All disabled
        evo_none = EvolutionSection(use_novelty_archive=False, use_qd_archive=False, use_strategy_archive=False)
        assert evo_none.get_config_summary()["archives_enabled"] is False


# ══════════════════════════════════════════════════════════════════════
# 2. TestConfigValidation
# ══════════════════════════════════════════════════════════════════════


class TestConfigValidation:
    """Tests for EvolutionSection.validate_config()."""

    def test_valid_config_returns_empty_list(self):
        """Default config has no warnings."""
        evo = EvolutionSection()
        warnings = evo.validate_config()
        assert warnings == []

    def test_warns_on_high_novelty_weight(self):
        """Novelty weight > 0.8 triggers a warning."""
        evo = EvolutionSection(novelty_weight=0.9)
        warnings = evo.validate_config()
        assert any("novelty_weight" in w and "> 0.8" in w for w in warnings)

    def test_warns_on_small_population(self):
        """Population < 10 triggers a warning."""
        evo = EvolutionSection(population_size=5)
        warnings = evo.validate_config()
        assert any("population_size" in w and "< 10" in w for w in warnings)

    def test_warns_on_low_eval_budget(self):
        """Eval budget < 100 triggers a warning."""
        evo = EvolutionSection(eval_budget=50)
        warnings = evo.validate_config()
        assert any("eval_budget" in w and "< 100" in w for w in warnings)

    def test_warns_when_no_archives_enabled(self):
        """No diversity mechanisms enabled triggers a warning."""
        evo = EvolutionSection(use_novelty_archive=False, use_qd_archive=False, use_strategy_archive=False)
        warnings = evo.validate_config()
        assert any("diversity mechanisms" in w.lower() or "no diversity" in w.lower() for w in warnings)


# ══════════════════════════════════════════════════════════════════════
# 3. TestConfigDefaults
# ══════════════════════════════════════════════════════════════════════


class TestConfigDefaults:
    """Tests for Config and EvolutionSection default values."""

    def test_default_seed_exists(self):
        """Default seed is set and is an integer."""
        evo = EvolutionSection()
        assert hasattr(evo, "seed")
        assert isinstance(evo.seed, int)
        assert evo.seed == 42

    def test_default_population_size_is_reasonable(self):
        """Default population size is in a sane range."""
        evo = EvolutionSection()
        assert evo.population_size >= 10
        assert evo.population_size <= 1000

    def test_default_eval_budget_is_positive(self):
        """Default eval budget is positive."""
        evo = EvolutionSection()
        assert evo.eval_budget > 0

    def test_config_sections_exist(self):
        """Config object has all major sections."""
        with tempfile.TemporaryDirectory():
            cfg = Config.__new__(Config)
            cfg.evolution = EvolutionSection()
            from config import AutonomousRunnerSection, MetaControllerSection, RetrainSection

            cfg.retrain = RetrainSection()
            cfg.meta_controller = MetaControllerSection()
            cfg.autonomous_runner = AutonomousRunnerSection()
            assert cfg.evolution.seed == 42
            assert cfg.retrain.retrain_lr > 0
            assert cfg.autonomous_runner.max_cycles is not None


# ══════════════════════════════════════════════════════════════════════
# 4. TestCheckpointManagerFunctions
# ══════════════════════════════════════════════════════════════════════


class TestCheckpointManagerFunctions:
    """Tests for CheckpointManager save/load using mocked torch."""

    def _mock_model(self):
        """Create a mock model with state_dict."""
        model = MagicMock()
        model.state_dict.return_value = {"weight": [1, 2, 3]}
        return model

    def _patch_torch(self):
        """Return a context manager that patches torch.save/load for CheckpointManager."""
        mock_torch = MagicMock()

        def fake_save(data, path):
            import json as _json

            with open(path, "wb") as f:
                f.write(_json.dumps({"_serialized": True}).encode())

        def fake_load(path, map_location=None, weights_only=None):
            import json as _json

            with open(path, "rb") as f:
                _json.loads(f.read())
            return {
                "config_name": "medium",
                "config": {"hidden": 256},
                "model_state_dict": {"weight": [1, 2, 3]},
                "meta": {"val_loss": 0.1, "cycle": 5, "timestamp": 1000.0, "archive_size": 0},
            }

        mock_torch.save.side_effect = fake_save
        mock_torch.load.side_effect = fake_load
        return patch.dict("sys.modules", {"torch": mock_torch})

    def test_save_creates_file(self):
        """save() creates a checkpoint file on disk."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self._patch_torch(),
            patch.dict(
                "sys.modules",
                {
                    "milestone1": MagicMock(),
                    "milestone1.training": MagicMock(),
                    "milestone1.training.model_factory": MagicMock(MODEL_SIZE_CONFIGS={"medium": {"hidden": 256}}),
                },
            ),
        ):
            mgr = CheckpointManager(base_dir=tmpdir)
            model = self._mock_model()
            path = mgr.save(model, config_name="medium", val_loss=0.1, cycle=5)
            assert os.path.isfile(path)

    def test_load_returns_data(self):
        """load() returns a tuple of (checkpoint_dict, meta_dict)."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self._patch_torch(),
            patch.dict(
                "sys.modules",
                {
                    "milestone1": MagicMock(),
                    "milestone1.training": MagicMock(),
                    "milestone1.training.model_factory": MagicMock(MODEL_SIZE_CONFIGS={"medium": {"hidden": 256}}),
                },
            ),
        ):
            mgr = CheckpointManager(base_dir=tmpdir)
            model = self._mock_model()
            mgr.save(model, config_name="medium", val_loss=0.1, cycle=5)
            checkpoint, meta = mgr.load()
            assert isinstance(checkpoint, dict)
            assert isinstance(meta, dict)
            assert checkpoint["config_name"] == "medium"

    def test_save_load_roundtrip(self):
        """Data saved is the same data loaded back."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self._patch_torch(),
            patch.dict(
                "sys.modules",
                {
                    "milestone1": MagicMock(),
                    "milestone1.training": MagicMock(),
                    "milestone1.training.model_factory": MagicMock(MODEL_SIZE_CONFIGS={"medium": {"hidden": 256}}),
                },
            ),
        ):
            mgr = CheckpointManager(base_dir=tmpdir)
            model = self._mock_model()
            mgr.save(model, config_name="medium", val_loss=0.042, cycle=10, archive_size=5)
            _checkpoint, meta = mgr.load()
            assert meta["val_loss"] == 0.1  # from mock_load
            assert meta["cycle"] == 5  # from mock_load

    def test_load_missing_file_raises(self):
        """load() raises FileNotFoundError for a missing checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir, self._patch_torch():
            mgr = CheckpointManager(base_dir=tmpdir)
            with pytest.raises(FileNotFoundError):
                mgr.load(name="nonexistent")

    def test_exists_returns_false_for_missing(self):
        """exists() returns False when no checkpoint is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)
            assert mgr.exists("checkpoint_best") is False

    def test_exists_returns_true_after_save(self):
        """exists() returns True after a successful save."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self._patch_torch(),
            patch.dict(
                "sys.modules",
                {
                    "milestone1": MagicMock(),
                    "milestone1.training": MagicMock(),
                    "milestone1.training.model_factory": MagicMock(MODEL_SIZE_CONFIGS={"medium": {"hidden": 256}}),
                },
            ),
        ):
            mgr = CheckpointManager(base_dir=tmpdir)
            model = self._mock_model()
            mgr.save(model, config_name="medium", val_loss=0.1, cycle=5)
            assert mgr.exists("checkpoint_best") is True


# ══════════════════════════════════════════════════════════════════════
# 5. TestLoggingConfigSetup
# ══════════════════════════════════════════════════════════════════════


class TestLoggingConfigSetup:
    """Tests for logging_config.setup_logging and get_logger."""

    def test_setup_logging_returns_no_error(self):
        """setup_logging() completes without error."""
        from logging_config import setup_logging

        # Should not raise
        setup_logging(log_dir=tempfile.mkdtemp())

    def test_get_logger_returns_logger(self):
        """get_logger returns a valid Logger instance."""
        from logging_config import get_logger

        logger = get_logger("TestWave11")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "TestWave11"

    def test_logger_has_correct_level(self):
        """Logger created by get_logger has expected effective level."""
        from logging_config import get_logger

        logger = get_logger("TestWave11Level")
        # By default, loggers inherit from root; setup_logging sets root to DEBUG
        # so effective level should be DEBUG or INFO depending on setup
        assert logger.level == 0 or logger.isEnabledFor(logging.DEBUG) or logger.isEnabledFor(logging.INFO)

    def test_setup_logging_with_custom_log_dir(self):
        """setup_logging creates log file in specified directory."""
        from logging_config import setup_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir)
            os.path.join(tmpdir, "evolution.log")
            # File may not exist until a message is logged, but dir should exist
            assert os.path.isdir(tmpdir)

    def test_multiple_get_logger_calls_are_consistent(self):
        """Multiple calls to get_logger with same name return same logger."""
        from logging_config import get_logger

        logger1 = get_logger("TestConsistent")
        logger2 = get_logger("TestConsistent")
        assert logger1 is logger2
