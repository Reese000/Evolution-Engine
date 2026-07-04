"""Checkpoint save/load utilities.

Extracted from the monolithic train_model.py.
"""

from __future__ import annotations

import os
import sys

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


def save_training_checkpoint(
    model: torch.nn.Module,
    best_val_loss: float,
    best_epoch: int,
    epoch: int,
    n_sequences: int,
    checkpoint_dir: str,
) -> None:
    """Save the final + best checkpoints to *checkpoint_dir*.

    Also copies final to checkpoint_best.pt for easy resume,
    and to checkpoint_current.pt for retrain (full checkpoint with optimizer state).

    Args:
        model:            The trained model (may be SWA-wrapped).
        best_val_loss:    Best validation loss achieved.
        best_epoch:       Epoch index (0-based) of the best model.
        epoch:            Last completed epoch index (0-based).
        n_sequences:      Total number of training sequences (for stats).
        checkpoint_dir:   Directory to write checkpoints into.
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
        "val_loss": best_val_loss,
        "stats": {"epochs": best_epoch + 1, "total_epochs_run": epoch + 1, "examples": n_sequences},
    }
    torch.save(ckpt_out, final_path)
    print(f"[Train] Saved: {os.path.basename(final_path)} (val_loss={best_val_loss:.4f}, epoch={best_epoch + 1})")

    import shutil

    shutil.copy2(final_path, best_path)
    print(f"[Train] Updated: {os.path.basename(best_path)}")

    shutil.copy2(final_path, current_path)
    print(f"[Train] Updated: {os.path.basename(current_path)}")
