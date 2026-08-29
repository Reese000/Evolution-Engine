#!/usr/bin/env python3
"""CheckpointManager — atomic, config-tagged checkpoints for safe model switching.

Prevents the corruption bug where switching between model sizes
(8L/256d ↔ 14L/512d) destroys trained weights because load_or_create_model()
creates a fresh model when sizes don't match.

Usage:
    from checkpoint_manager import CheckpointManager
    mgr = CheckpointManager(base_dir="milestone1/checkpoints")
    mgr.save(model, config_name="medium", val_loss=0.0986, cycle=41)
    model, meta = mgr.load(config_name="medium")  # validates config match
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CheckpointMeta:
    """Metadata stored alongside model weights."""

    config_name: str  # "small", "medium", "large", "xlarge"
    config: dict  # model architecture params
    val_loss: float
    cycle: int
    timestamp: float
    archive_size: int = 0


class CheckpointManager:
    """Manages checkpoints with config validation and atomic writes.

    Key features:
    - Config tagging: prevents loading wrong architecture
    - Atomic writes: tempfile + os.replace prevents corruption from interrupted saves
    - Migration path: explicit error when switching architectures
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save(
        self,
        model: Any,  # torch.nn.Module
        config_name: str,
        val_loss: float,
        cycle: int,
        archive_size: int = 0,
        name: str = "checkpoint_best",
    ) -> str:
        """Save checkpoint with config tag and atomic write.

        Returns the path to the saved checkpoint.
        """
        from milestone1.training.model_factory import MODEL_SIZE_CONFIGS

        config = MODEL_SIZE_CONFIGS.get(config_name, {})
        meta = CheckpointMeta(
            config_name=config_name,
            config=config,
            val_loss=val_loss,
            cycle=cycle,
            timestamp=time.time(),
            archive_size=archive_size,
        )

        checkpoint = {
            "config_name": config_name,
            "config": config,
            "model_state_dict": model.state_dict(),
            "meta": {
                "val_loss": val_loss,
                "cycle": cycle,
                "timestamp": meta.timestamp,
                "archive_size": archive_size,
            },
        }

        path = os.path.join(self.base_dir, f"{name}.pt")

        # Atomic write: save to temp file, then rename
        dir_name = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(dir=dir_name, suffix=".tmp", delete=False) as f:
            tmp_path = f.name

        try:
            import torch

            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return path

    def load(
        self,
        name: str = "checkpoint_best",
        config_name: str | None = None,
        allow_mismatch: bool = False,
    ) -> tuple[Any, dict]:
        """Load checkpoint with optional config validation.

        Args:
            name: Checkpoint file name (without .pt extension)
            config_name: Expected config. If provided and mismatch, raises ValueError
                        unless allow_mismatch is True.
            allow_mismatch: If True, loads regardless of config mismatch (creates fresh model).

        Returns:
            (model_state_dict_or_config, meta_dict)
        """
        import torch

        path = os.path.join(self.base_dir, f"{name}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Validate config if requested
        ckpt_config = checkpoint.get("config_name", None)
        if config_name and ckpt_config and ckpt_config != config_name and not allow_mismatch:
            raise CheckpointMismatchError(
                f"Checkpoint is '{ckpt_config}' but requested '{config_name}'. "
                f"Use allow_mismatch=True to force load, or "
                f"start fresh with a new checkpoint."
            )

        meta = checkpoint.get("meta", {})
        return checkpoint, meta

    def exists(self, name: str = "checkpoint_best") -> bool:
        """Check if a checkpoint exists."""
        return os.path.isfile(os.path.join(self.base_dir, f"{name}.pt"))

    def backup(self, name: str = "checkpoint_best") -> str:
        """Create a backup of the current checkpoint."""
        src = os.path.join(self.base_dir, f"{name}.pt")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"No checkpoint to backup: {src}")

        import torch

        checkpoint = torch.load(src, map_location="cpu", weights_only=False)
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{name}_backup_{ts}.pt"
        backup_path = os.path.join(self.base_dir, backup_name)
        torch.save(checkpoint, backup_path)
        return backup_path


class CheckpointMismatchError(Exception):
    """Raised when checkpoint architecture doesn't match requested config."""

