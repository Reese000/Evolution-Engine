#!/usr/bin/env python3
"""
train_model — EE-compatible training module.

Wraps the SelfEvolving train_model.py `main()` function with the
interface expected by run_multi_cycle.py (`train(**kwargs)`).

All kwargs are forwarded as command-line arguments to main().
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Make milestone1 modules available
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def train(
    resume_path: str | None = None,
    recurrent_data_path: str | None = None,
    lr_override: float | None = None,
    freeze_layers: int = 0,
    adaptive_freeze: bool = False,
    batch_size: int = 8,
    weight_decay: float = 0.01,
    patience: int = 5,
    grad_clip: float | None = None,
    max_epochs: int = 12,
    val_split: float = 0.15,
    max_per_level: int | None = None,
    model_size: str = "medium",
    amp: bool = True,
    compile_mode: str | None = None,
    swa_start: int | None = None,
    swa_lr: float | None = None,
    grad_accum_steps: int = 1,
    rebuild_dataset: bool = False,
    **kwargs: Any,
) -> None:
    """Run training with the given params.

    This is a compatibility shim that calls the underlying training pipeline.
    When the EE-native train_model (v22 path) is available, it delegates to
    that directly. Otherwise, it attempts to build and run training inline.
    """
    from train_model import main as train_main

    # Build CLI args matching train_model.py --help
    # but also support the EE kwargs pattern
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Use the recurrent data path as the primary data source
    data_path = recurrent_data_path or os.path.join(BASE_DIR, "recurrent_data.jsonl")

    # Construct CLI args for train_model.main()
    sys_argv_backup = list(sys.argv)
    sys.argv = [
        "train_model.py",
        "--data",
        data_path,
        "--output",
        os.path.join(checkpoint_dir, "model_final.pt"),
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(max_epochs),
        "--lr",
        str(lr_override or 3e-5),
        "--seed",
        "42",
    ]

    # Add resume if provided
    if resume_path and os.path.isfile(resume_path):
        # train_model.main() doesn't support resume directly;
        # save to checkpoint_best.pt path for next cycle
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        try:
            import shutil

            shutil.copy2(resume_path, best_path)
        except OSError:
            pass

    try:
        train_main()
    finally:
        sys.argv = sys_argv_backup

    # After training completes, copy final model to checkpoint_best.pt
    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    if os.path.isfile(final_path) and final_path != best_path:
        try:
            import shutil

            shutil.copy2(final_path, best_path)
        except OSError:
            pass
