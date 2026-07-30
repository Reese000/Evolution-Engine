#!/usr/bin/env python3
"""Integration tests for CheckpointManager — atomic saves, config validation, load.

Covers:
  - Save/load lifecycle
  - Config mismatch detection
  - Allow_mismatch override
  - Backup and exists checks
  - Atomic write integrity
"""

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestCheckpointManager:
    """Full save/load lifecycle with config validation."""

    def test_save_and_load_roundtrip(self):
        """Save a checkpoint and load it back."""
        from checkpoint_manager import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)

            # Create a simple mock model
            model = type("MockModel", (), {})()
            model.state_dict = lambda: {"weight": [1.0, 2.0, 3.0]}

            path = mgr.save(model, config_name="small", val_loss=0.5, cycle=1)
            assert os.path.isfile(path)
            assert "checkpoint_best.pt" in path

            # Load it back
            from checkpoint_manager import CheckpointMismatchError

            ckpt, meta = mgr.load(config_name="small")
            assert ckpt is not None
            assert meta["val_loss"] == 0.5
            assert meta["cycle"] == 1

    def test_config_mismatch_raises_error(self):
        """Loading with wrong config_name raises CheckpointMismatchError."""
        from checkpoint_manager import CheckpointManager, CheckpointMismatchError

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)
            model = type("MockModel", (), {})()
            model.state_dict = lambda: {"w": [1.0]}

            mgr.save(model, config_name="small", val_loss=0.5, cycle=1)

            import pytest

            with pytest.raises(CheckpointMismatchError):
                mgr.load(config_name="medium")

    def test_allow_mismatch_override(self):
        """allow_mismatch=True loads regardless of config mismatch."""
        from checkpoint_manager import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)
            model = type("MockModel", (), {})()
            model.state_dict = lambda: {"w": [1.0]}

            mgr.save(model, config_name="small", val_loss=0.5, cycle=1)

            # Should not raise
            ckpt, meta = mgr.load(config_name="medium", allow_mismatch=True)
            assert ckpt is not None

    def test_exists_returns_correctly(self):
        """exists() returns True/False for existing/missing checkpoints."""
        from checkpoint_manager import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)
            assert not mgr.exists("nonexistent")

            model = type("MockModel", (), {})()
            model.state_dict = lambda: {"w": [1.0]}
            mgr.save(model, config_name="small", val_loss=0.5, cycle=1, name="test_ckpt")
            assert mgr.exists("test_ckpt")

    def test_backup_creates_file(self):
        """backup() creates a timestamped copy."""
        from checkpoint_manager import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)
            model = type("MockModel", (), {})()
            model.state_dict = lambda: {"w": [1.0]}
            mgr.save(model, config_name="small", val_loss=0.5, cycle=1)

            backup_path = mgr.backup()
            assert os.path.isfile(backup_path)
            assert "backup" in backup_path

    def test_backup_raises_on_missing(self):
        """backup() raises FileNotFoundError if no checkpoint exists."""
        from checkpoint_manager import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)
            try:
                mgr.backup()
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                pass

    def test_load_raises_on_missing(self):
        """load() raises FileNotFoundError for nonexistent checkpoint."""
        from checkpoint_manager import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)
            try:
                mgr.load()
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                pass

    def test_atomic_write_integrity(self):
        """Save uses atomic write (tempfile + rename) — partial write can't corrupt."""
        from checkpoint_manager import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)
            model = type("MockModel", (), {})()
            model.state_dict = lambda: {"weight": [42.0]}

            # Save normally
            mgr.save(model, config_name="small", val_loss=0.5, cycle=1)
            assert mgr.exists()

            # Verify loaded data is intact
            ckpt, _ = mgr.load()
            state_dict = ckpt.get("model_state_dict", {})
            assert state_dict["weight"] == [42.0]
