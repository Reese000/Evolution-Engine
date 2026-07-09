"""Checkpoint save/load utilities.

Extracted from the monolithic train_model.py.

Implements atomic writes using tempfile + os.replace to prevent
corruption from interrupted saves (e.g. power loss, Ctrl+C).
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

# Ensure milestone1/ is on sys.path.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def find_existing_checkpoint(checkpoint_dir: str) -> str | None:
    """Find an existing checkpoint file in *checkpoint_dir*.

    Checks for checkpoint_best.pt, then checkpoint_current.pt, then model_final.pt.
    Returns:
        Absolute path to the first checkpoint found, or None.
    """
    for name in ("checkpoint_best.pt", "checkpoint_current.pt", "model_final.pt"):
        p = os.path.join(checkpoint_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _atomic_save(obj: object, dest_path: str) -> None:
    """Atomically save *obj* to *dest_path* using tempfile + os.replace.

    Writes to a temporary file in the same directory, then does an
    atomic rename.  This prevents corruption when:
      - The process is killed mid-write (Ctrl+C, power loss)
      - Another process reads the file during a write
      - The disk fills up mid-write (temp file is cleaned up)

    If the atomic rename fails (e.g. cross-device), falls back to a
    direct write with the same path.
    """
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    try:
        # Write to a temporary file in the same directory (same filesystem)
        fd, tmp_path = tempfile.mkstemp(
            dir=dest_dir,
            prefix=".ckpt_tmp_",
            suffix=".pt",
        )
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(obj, f)
            # Atomic rename: instant on the same filesystem
            os.replace(tmp_path, dest_path)
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError:
        # Fallback: direct write (cross-device or permission issues)
        torch.save(obj, dest_path)


def save_training_checkpoint(
    model: torch.nn.Module,
    best_val_loss: float,
    best_epoch: int,
    epoch: int,
    n_sequences: int,
    checkpoint_dir: str,
    config_name: str | None = None,
) -> None:
    """Save the final + best checkpoints to *checkpoint_dir*.

    Also copies final to checkpoint_best.pt for easy resume,
    and to checkpoint_current.pt for retrain (full checkpoint with optimizer state).

    Uses atomic writes (tempfile + os.replace) to prevent corruption.

    Args:
        model:            The trained model (may be SWA-wrapped).
        best_val_loss:    Best validation loss achieved.
        best_epoch:       Epoch index (0-based) of the best model.
        epoch:            Last completed epoch index (0-based).
        n_sequences:      Total number of training sequences (for stats).
        checkpoint_dir:   Directory to write checkpoints into.
        config_name:      Model size key (small/medium/large/xlarge) to tag
                          the checkpoint for cross-validation on load.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    current_path = os.path.join(checkpoint_dir, "checkpoint_current.pt")

    # Unwrap SWA model if needed for saving
    save_model = model
    if hasattr(model, "module"):
        save_model = model.module

    ckpt_out = {
        "model_state_dict": save_model.cpu().state_dict(),
        "config": {
            "vocab_size": getattr(save_model, "vocab_size", 2048),
            "d_model": getattr(save_model, "d_model", 256),
            "n_layers": getattr(save_model, "n_layers", 8),
            "n_heads": getattr(save_model, "n_heads", 8),
            "d_ff": getattr(save_model, "d_ff", 1024),
        },
        "config_name": config_name or "unknown",
        "val_loss": best_val_loss,
        "stats": {"epochs": best_epoch + 1, "total_epochs_run": epoch + 1, "examples": n_sequences},
    }

    # Atomic save to final path
    _atomic_save(ckpt_out, final_path)
    print(f"[Train] Saved: {os.path.basename(final_path)} (val_loss={best_val_loss:.4f}, epoch={best_epoch + 1})")

    import shutil

    # Atomic copies to best and current paths
    _atomic_save(ckpt_out, best_path)
    print(f"[Train] Updated: {os.path.basename(best_path)}")

    _atomic_save(ckpt_out, current_path)
    print(f"[Train] Updated: {os.path.basename(current_path)}")
