#!/usr/bin/env python3
"""init_checkpoint.py -- Bootstrap an initial model checkpoint.

The multi-cycle / autonomous entry points (``run_multi_cycle.py`` and
``autonomous_runner.py``) require a seed model checkpoint at
``milestone1/checkpoints/checkpoint_best.pt`` (and ``checkpoint_current.pt``)
to run the model-guided evolution strategy and the retrain loop. Those files
are gitignored and absent in a fresh clone, so those entry points abort before
doing any work.

This script creates a freshly-initialised (untrained) RV32I transformer and
writes it in the checkpoint format expected by both
``milestone1/training/model_factory.load`` (retrain) and
``prefix_generator.PrefixGenerator`` (model-guided evolution), so the full
evolve -> retrain -> evolve loop is runnable from a clean checkout.

Usage:
    python init_checkpoint.py                 # medium model (matches config default)
    python init_checkpoint.py --model-size small
    python init_checkpoint.py --force         # overwrite existing checkpoints
"""
from __future__ import annotations

import argparse
import os
import sys

from paths import M1_CHECKPOINTS, ensure_project_on_path

# Model size presets mirror milestone1/training/model_factory.MODEL_SIZE_CONFIGS.
MODEL_SIZE_CONFIGS: dict[str, dict] = {
    "small": dict(d_model=256, n_layers=6, n_heads=8, d_ff=683),
    "medium": dict(d_model=256, n_layers=8, n_heads=8, d_ff=1024),
    "large": dict(d_model=512, n_layers=14, n_heads=8, d_ff=1365),
    "xlarge": dict(d_model=768, n_layers=16, n_heads=12, d_ff=2048),
}


def build_checkpoint(model_size: str) -> dict:
    """Instantiate a fresh RV32ITransformer and wrap it in a checkpoint dict."""
    ensure_project_on_path()
    import torch
    from rv32i_model import RV32ITransformer, RV32IVocab

    if model_size not in MODEL_SIZE_CONFIGS:
        raise SystemExit(f"Unknown --model-size {model_size!r}. Choose from {list(MODEL_SIZE_CONFIGS)}.")

    arch = MODEL_SIZE_CONFIGS[model_size]
    vocab_size = RV32IVocab.VOCAB_SIZE
    max_seq_len = 2048

    torch.manual_seed(42)
    model = RV32ITransformer(
        vocab_size=vocab_size,
        d_model=arch["d_model"],
        n_layers=arch["n_layers"],
        n_heads=arch["n_heads"],
        d_ff=arch["d_ff"],
        max_seq_len=max_seq_len,
        use_checkpointing=False,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model_state_dict": model.cpu().state_dict(),
        # `config` is read by model_factory.load; config_name is intentionally
        # omitted so that size validation is skipped and the loader matches on
        # architecture instead (robust to differing --retrain-model-size).
        "config": {
            "vocab_size": vocab_size,
            "d_model": arch["d_model"],
            "n_layers": arch["n_layers"],
            "n_heads": arch["n_heads"],
            "d_ff": arch["d_ff"],
            "max_seq_len": max_seq_len,
        },
        # No training has happened yet -- val_loss is unknown. The retrain gate
        # (MetaController.accept_checkpoint) treats <= 0 / None as "accept
        # cautiously", so leaving this None lets the first real retrain set it.
        "val_loss": None,
        "stats": {"epochs": 0, "total_epochs_run": 0, "examples": 0, "bootstrap": True},
        "_n_params": n_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap an initial model checkpoint.")
    parser.add_argument(
        "--model-size",
        default="medium",
        help="Model size preset (default: medium, matches config/engine_config.json retrain_model_size).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing checkpoint files.")
    args = parser.parse_args()

    import torch

    ckpt_dir = str(M1_CHECKPOINTS)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "checkpoint_best.pt")
    current_path = os.path.join(ckpt_dir, "checkpoint_current.pt")

    existing = [p for p in (best_path, current_path) if os.path.isfile(p)]
    if existing and not args.force:
        print("Checkpoint(s) already exist; not overwriting (use --force):")
        for p in existing:
            print(f"  {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
        return

    ckpt = build_checkpoint(args.model_size)
    n_params = ckpt.pop("_n_params")

    torch.save(ckpt, best_path)
    torch.save(ckpt, current_path)

    print(f"Bootstrapped {args.model_size} checkpoint ({n_params / 1e6:.2f}M params):")
    for p in (best_path, current_path):
        print(f"  wrote {p} ({os.path.getsize(p) / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
