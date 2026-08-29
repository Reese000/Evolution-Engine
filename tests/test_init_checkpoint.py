#!/usr/bin/env python3
"""Tests for init_checkpoint.py bootstrap.

init_checkpoint.py lets the multi-cycle / autonomous entry points run from a
clean clone (they abort without a seed checkpoint). These tests guard the
bootstrap against model-API drift and confirm the produced checkpoint is
loadable in the format model_factory.load / PrefixGenerator expect.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from init_checkpoint import MODEL_SIZE_CONFIGS, build_checkpoint


def test_build_checkpoint_structure_medium():
    ckpt = build_checkpoint("medium")
    arch = MODEL_SIZE_CONFIGS["medium"]
    assert set(("model_state_dict", "config", "val_loss", "stats")).issubset(ckpt)
    cfg = ckpt["config"]
    assert cfg["d_model"] == arch["d_model"]
    assert cfg["n_layers"] == arch["n_layers"]
    assert cfg["n_heads"] == arch["n_heads"]
    assert cfg["d_ff"] == arch["d_ff"]
    # Untrained -> val_loss None so the first real retrain sets it.
    assert ckpt["val_loss"] is None
    assert ckpt["stats"].get("bootstrap") is True
    assert ckpt["model_state_dict"], "state dict should be non-empty"


def test_unknown_model_size_raises():
    with pytest.raises(SystemExit):
        build_checkpoint("gigantic")


def test_checkpoint_state_dict_loads_into_model():
    """The bootstrapped weights must load cleanly into a matching model."""
    from paths import ensure_project_on_path

    ensure_project_on_path()
    from rv32i_model import RV32ITransformer

    ckpt = build_checkpoint("small")
    cfg = ckpt["config"]
    model = RV32ITransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        use_checkpointing=False,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not missing, f"unexpected missing keys: {missing[:5]}"
    assert not unexpected, f"unexpected extra keys: {unexpected[:5]}"
