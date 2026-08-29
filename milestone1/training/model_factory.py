"""Model creation and checkpoint loading.

Handles the logic of creating a fresh RV32ITransformer or loading from a
checkpoint (including architecture inference from stripped weights and
GELU->SwiGLU migration).  Extracted from the monolithic train_model.py.
"""

from __future__ import annotations

import os
import sys

import torch

# Ensure milestone1/ and parent are on sys.path so sibling modules resolve.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
_LOGIT_MASK_DIR = os.path.join(BASE_DIR, "..", "milestone0", "logit-masking")
if os.path.isdir(_LOGIT_MASK_DIR) and _LOGIT_MASK_DIR not in sys.path:
    sys.path.insert(0, _LOGIT_MASK_DIR)


class CheckpointMismatchError(Exception):
    """Raised when a checkpoint's config_name does not match the target model.

    This prevents accidentally loading weights from one architecture
    (e.g. 8L/256d) into another (e.g. 14L/512d), which would silently
    produce garbage output or crash with dimension errors.
    """



# ── Model size configs ──────────────────────────────────────────────────────
MODEL_SIZE_CONFIGS: dict[str, dict] = {
    "small": {"d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 683},
    "medium": {"d_model": 256, "n_layers": 8, "n_heads": 8, "d_ff": 1024},
    "large": {"d_model": 512, "n_layers": 14, "n_heads": 8, "d_ff": 1365},
    "xlarge": {"d_model": 768, "n_layers": 16, "n_heads": 12, "d_ff": 2048},
}


def from_config(config: dict, device: torch.device | None = None) -> torch.nn.Module:
    """Create an RV32ITransformer from a config dict.

    The config dict may contain:
        - Any key from RV32ITransformer.__init__ (d_model, n_layers, etc.)
        - 'size' key: lookup in MODEL_SIZE_CONFIGS to fill missing fields
        - 'vocab_size': defaults to RV32IVocab.VOCAB_SIZE (2048)
        - 'max_seq_len': defaults to 2048
        - 'dropout': defaults to 0.1
        - 'tie_weights': defaults to True
        - 'use_rope': defaults to True
        - 'use_checkpointing': defaults to False

    Args:
        config: configuration dictionary
        device: target device (defaults to cuda if available)

    Returns:
        Instantiated RV32ITransformer on the target device
    """
    from rv32i_model import RV32IFieldTokenizer, RV32ITransformer, RV32IVocab

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start from size preset if provided, then overlay explicit config
    cfg = dict(config)
    size_key = cfg.pop("size", None)
    if size_key and size_key in MODEL_SIZE_CONFIGS:
        base = dict(MODEL_SIZE_CONFIGS[size_key])
        base.update({k: v for k, v in cfg.items() if v is not None})
        cfg = base

    # Filter to only RV32ITransformer constructor kwargs
    valid_kwargs = {
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "max_seq_len",
        "dropout",
        "tie_weights",
        "use_rope",
        "use_checkpointing",
    }
    filtered = {k: v for k, v in cfg.items() if k in valid_kwargs}

    # Defaults
    filtered.setdefault("vocab_size", RV32IVocab.VOCAB_SIZE)
    filtered.setdefault("d_model", 256)
    filtered.setdefault("n_layers", 8)
    filtered.setdefault("n_heads", 8)
    filtered.setdefault("d_ff", 1024)
    filtered.setdefault("max_seq_len", 2048)
    filtered.setdefault("dropout", 0.1)
    filtered.setdefault("tie_weights", True)
    filtered.setdefault("use_rope", True)
    filtered.setdefault("use_checkpointing", False)

    model = RV32ITransformer(**filtered)
    return model.to(device)


def load_or_create_model(
    resume_path: str | None,
    model_size: str = "large",
    device: torch.device | None = None,
    allow_mismatch: bool = False,
) -> tuple[torch.nn.Module, dict]:
    """Load a model from checkpoint or create a fresh one.

    Handles:
      - Architecture inference from stripped checkpoints (empty config).
      - GELU -> SwiGLU weight migration for old checkpoints.
      - Size mismatch: creates a fresh model if checkpoint dimensions differ.
      - Config_name validation: prevents loading checkpoints saved with a
        different model_size (e.g. 'small' checkpoint into 'large' target).

    Args:
        resume_path: Path to a .pt checkpoint, or None for a fresh model.
        model_size:  Target model size key (small/medium/large/xlarge).
        device:      Target device (defaults to cuda if available).
        allow_mismatch: If True, skip config_name validation and proceed
            even when the checkpoint's config_name differs from model_size.
            Used by --reset-checkpoint / --migrate flags.

    Returns:
        (model, ckpt_config) where ckpt_config is the config dict from the
        checkpoint (may be empty for stripped checkpoints).

    Raises:
        CheckpointMismatchError: If the checkpoint's config_name does not
            match model_size and allow_mismatch is False.
    """
    from rv32i_model import RV32ITransformer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_cfg = MODEL_SIZE_CONFIGS.get(model_size, MODEL_SIZE_CONFIGS["large"])
    ckpt_path = resume_path

    model = None
    ckpt_config: dict = {}

    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[Train] Loading checkpoint: {os.path.basename(ckpt_path)}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_cfg = ckpt.get("config", {})
        ckpt_sd = ckpt.get("model_state_dict", ckpt)

        # ── Config-name validation (prevents wrong-model loading) ──
        ckpt_config_name = ckpt_cfg.get("config_name", "")
        if ckpt_config_name and not allow_mismatch:
            if ckpt_config_name != model_size:
                raise CheckpointMismatchError(
                    f"Checkpoint was saved with config_name={ckpt_config_name!r} "
                    f"but target model_size={model_size!r}. "
                    f"Use --reset-checkpoint to discard the old checkpoint "
                    f"or --migrate to load with architecture mismatch."
                )
            print(f"[Train] Config-name validated: {ckpt_config_name!r}")

        # When config is empty (stripped checkpoint), infer size from actual weights
        if not ckpt_cfg:
            if ckpt_sd and isinstance(ckpt_sd, dict) and any(k.startswith("blocks.") for k in ckpt_sd):
                n_blk = len([k for k in ckpt_sd if k.startswith("blocks.") and "ln1.weight" in k])
                qkv_key = next(k for k in ckpt_sd if k.startswith("blocks.0.attn.qkv.weight"))
                ckpt_d = ckpt_sd[qkv_key].shape[1]
                ckpt_l = n_blk
                print(f"[Train] Empty config -- inferred from weights: {ckpt_l}L, d={ckpt_d}")
            else:
                ckpt_d, ckpt_l = 256, 8
        else:
            ckpt_d = ckpt_cfg.get("d_model", 256)
            ckpt_l = ckpt_cfg.get("n_layers", 8)

        size_compatible = ckpt_d == target_cfg["d_model"] and ckpt_l == target_cfg["n_layers"]

        if size_compatible:
            print(f"[Train] Checkpoint matches target: {ckpt_l}L/d={ckpt_d}")
            model = RV32ITransformer(
                vocab_size=ckpt_cfg.get("vocab_size", 2048),
                d_model=ckpt_d,
                n_layers=ckpt_l,
                n_heads=ckpt_cfg.get("n_heads", 8),
                d_ff=ckpt_cfg.get("d_ff", 1024),
                max_seq_len=ckpt_cfg.get("max_seq_len", 2048),
                use_checkpointing=False,
            )
        else:
            print(
                f"[Train] Checkpoint ({ckpt_l}L/d={ckpt_d}) != target ({target_cfg['n_layers']}L/d={target_cfg['d_model']})"
            )
            print(f"[Train] Creating fresh model ({model_size}): {target_cfg['n_layers']}L, d={target_cfg['d_model']}")
            model = RV32ITransformer(
                vocab_size=2048,
                d_model=target_cfg["d_model"],
                n_layers=target_cfg["n_layers"],
                n_heads=target_cfg["n_heads"],
                d_ff=target_cfg["d_ff"],
                max_seq_len=ckpt_cfg.get("max_seq_len", 2048),
                use_checkpointing=False,
            )

        sd = ckpt_sd if size_compatible else None
        if isinstance(sd, dict) and any(k.startswith("blocks.") for k in sd):
            # Handle old GELU checkpoint -> new SwiGLU architecture
            has_old_ffn = any("ffn.0.weight" in k for k in sd)
            has_new_ffn = any("ffn.w1.weight" in k for k in sd)
            if has_old_ffn and not has_new_ffn:
                print("[Train] Adapting old GELU weights -> SwiGLU architecture")
                new_sd = {}
                for k, v in sd.items():
                    if "pos_embedding" in k:
                        continue
                    if "ffn.0.weight" in k:
                        new_k = k.replace("ffn.0.weight", "ffn.w1.weight")
                        new_sd[new_k] = v
                        w3_k = k.replace("ffn.0.weight", "ffn.w3.weight")
                        new_sd[w3_k] = v + 0.01 * torch.randn_like(v)
                        continue
                    elif "ffn.2.weight" in k:
                        new_k = k.replace("ffn.2.weight", "ffn.w2.weight")
                        new_sd[new_k] = v
                        continue
                    new_sd[k] = v
                sd = new_sd
            missing, _unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"[Train] Missing keys (randomly initialized): {len(missing)}")
        print(f"[Train] Loaded model: {target_cfg['n_layers']}L, d={target_cfg['d_model']}")

        ckpt_config = ckpt_cfg
    else:
        # No checkpoint -- create a fresh model.
        print(f"[Train] Creating fresh model ({model_size}): {target_cfg['n_layers']}L, d={target_cfg['d_model']}")
        model = RV32ITransformer(
            vocab_size=2048,
            d_model=target_cfg["d_model"],
            n_layers=target_cfg["n_layers"],
            n_heads=target_cfg["n_heads"],
            d_ff=target_cfg["d_ff"],
            max_seq_len=2048,
            use_checkpointing=False,
        )

    model.to(device)
    model.train()

    return model, ckpt_config
