"""Training loop — the refactored train() function.

Orchestrates calls to data loading, augmentation, packing, model creation,
scheduling, and checkpoint saving.  Same signature as the original monolithic
train() in train_model.py.
"""

from __future__ import annotations

import gc
import os
import sys
from typing import Any

import torch

# Ensure milestone1/ and parent are on sys.path so sibling modules resolve.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
_LOGIT_MASK_DIR = os.path.join(BASE_DIR, "..", "milestone0", "logit-masking")
if os.path.isdir(_LOGIT_MASK_DIR) and _LOGIT_MASK_DIR not in sys.path:
    sys.path.insert(0, _LOGIT_MASK_DIR)


def train(
    resume_path: str | None = None,
    recurrent_data_path: str | None = None,
    lr_override: float | None = None,
    freeze_layers: int = 0,
    adaptive_freeze: bool = False,
    batch_size: int = 8,
    weight_decay: float = 0.01,
    patience: int = 3,
    grad_clip: float | None = None,
    max_epochs: int = 12,
    val_split: float = 0.15,
    max_per_level: int | None = None,
    model_size: str = "large",
    amp: bool = True,
    compile_mode: str | None = None,
    swa_start: int | None = None,
    swa_lr: float | None = None,
    grad_accum_steps: int = 1,
    augmentation_k: int = 8,
    rebuild_dataset: bool = False,
    reset_checkpoint: bool = False,
    migrate: bool = False,
    use_grad_checkpointing: bool = False,
    output_path: str | None = None,
    **kwargs: Any,
) -> None:
    """Train the RV32ITransformer on champion program traces."""
    from data.augmentation import inject_reference_solutions, inject_synthetic_data, register_rename_augment
    from data.loading import apply_level_cap, load_champion_traces
    from data.packing import pad_or_pack_sequences
    from rv32i_mask import decode as rv32i_decode
    from rv32i_model import RV32IFieldTokenizer
    from training.checkpoint import find_existing_checkpoint, save_training_checkpoint
    from training.model_factory import load_or_create_model
    from training.schedule import adjust_dropout, build_optimizer_and_schedulers

    # ── Checkpoint directory ───────────────────────────────────────
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Load checkpoint path ───────────────────────────────────────
    ckpt_path = resume_path
    if not ckpt_path or not os.path.isfile(ckpt_path):
        found = find_existing_checkpoint(checkpoint_dir)
        if found:
            ckpt_path = found

    # ── Handle reset/migrate flags ─────────────────────────────────
    if reset_checkpoint and ckpt_path and os.path.isfile(ckpt_path):
        from training.checkpoint import save_training_checkpoint  # noqa: F811

        print(f"[Train] --reset-checkpoint: deleting {os.path.basename(ckpt_path)}")
        os.remove(ckpt_path)
        # Also clear any sibling checkpoints
        for name in ("checkpoint_best.pt", "checkpoint_current.pt", "model_final.pt"):
            p = os.path.join(checkpoint_dir, name)
            if os.path.isfile(p) and p != ckpt_path:
                os.remove(p)
                print(f"[Train] --reset-checkpoint: deleted {name}")
        ckpt_path = None

    # ── Device ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load or create model ───────────────────────────────────────
    model, _ckpt_config = load_or_create_model(
        ckpt_path,
        model_size=model_size,
        device=device,
        allow_mismatch=migrate,
    )

    # ── Gradient checkpointing for larger models ──────────────────
    if use_grad_checkpointing:
        model.use_checkpointing = True
        print(
            f"[Train] Gradient checkpointing enabled (saves ~{sum(p.numel() for p in model.parameters()) // 1000}K params worth of activation memory)"
        )

    # ── torch.compile for faster training ──────────────────────────
    if device.type == "cuda" and compile_mode:
        try:
            import triton  # noqa: F401

            model = torch.compile(model, mode=compile_mode)
            print(f"[Train] torch.compile enabled (mode={compile_mode})")
        except ImportError:
            print("[Train] Triton not available — skipping torch.compile")
        except Exception as e:
            print(f"[Train] torch.compile failed (falling back to eager): {e}")

    # ── Load champion traces ───────────────────────────────────────
    data_path = recurrent_data_path or os.path.join(BASE_DIR, "recurrent_data.jsonl")
    tokenizer = RV32IFieldTokenizer()
    sequences, sample_weights, raw_dicts_for_augment, entry_levels = load_champion_traces(
        data_path,
        tokenizer,
        rv32i_decode=rv32i_decode,
    )

    if not sequences:
        print("[Train] No usable sequences -- skipping retrain")
        return

    _cap_before = len(sequences)
    apply_level_cap(sequences, sample_weights, raw_dicts_for_augment, entry_levels)
    print(f"[Train] Per-level cap: {_cap_before} -> {len(sequences)} sequences")

    # ── Rebalance level weights (upsample underrepresented levels) ──
    from data.loading import rebalance_level_weights

    rebalance_level_weights(sample_weights, entry_levels)

    print(f"[Train] Loaded {len(sequences)} sequences from {os.path.basename(data_path)}")

    # ── Register renaming augmentation ─────────────────────────────
    aug_count = 0
    augmented_seqs: list = []
    augmented_weights: list[float] = []
    for _aug_i, instr_dicts in enumerate(raw_dicts_for_augment):
        variants = register_rename_augment(instr_dicts, k=augmentation_k)
        parent_weight = sample_weights[_aug_i]
        for variant in variants:
            aug_tokens = tokenizer.encode_trace(variant)
            if aug_tokens and len(aug_tokens) >= 2:
                augmented_seqs.append(torch.tensor(aug_tokens, dtype=torch.long))
                augmented_weights.append(parent_weight)
                aug_count += 1
    sequences.extend(augmented_seqs)
    sample_weights.extend(augmented_weights)
    if aug_count > 0:
        print(f"[Train] Register renaming augmentation: +{aug_count} sequences")
    del raw_dicts_for_augment, augmented_seqs, augmented_weights

    # ── Synthetic data generation ──────────────────────────────────
    inject_synthetic_data(sequences, sample_weights, seed=42)

    # ── Reference solution injection ───────────────────────────────
    inject_reference_solutions(tokenizer, sequences, sample_weights, rv32i_decode=rv32i_decode)

    # ── Pack or pad sequences ──────────────────────────────────────
    split = pad_or_pack_sequences(
        sequences,
        sample_weights,
        max_seq_len=1024,
        val_split=val_split,
        device=device,
    )
    train_data = split["train_data"]
    val_data = split["val_data"]
    train_loss_mask = split["train_loss_mask"]
    val_loss_mask = split["val_loss_mask"]
    train_weights = split["train_weights"]
    val_weights = split["val_weights"]

    # ── Training setup ─────────────────────────────────────────────
    actual_epochs = max_epochs
    lr = lr_override or 3e-5
    n_train_batches = max(1, len(train_data) // batch_size)

    sched = build_optimizer_and_schedulers(
        model,
        lr,
        weight_decay,
        actual_epochs,
        n_train_batches,
        swa_start=swa_start,
        swa_lr=swa_lr,
    )
    optimizer = sched["optimizer"]
    scheduler = sched["scheduler"]
    use_swa = sched["use_swa"]
    swa_model = sched["swa_model"]
    swa_scheduler = sched["swa_scheduler"]

    # AMP (mixed precision)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Enable TF32 for faster matmul on Ampere+ GPUs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from rv32i_model import compute_loss as weighted_compute_loss
    from rv32i_model import compute_loss_with_scheduled_sampling

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    step_count = 0

    for epoch in range(actual_epochs):
        current_drop = adjust_dropout(model, epoch, actual_epochs)

        # Phase 3: Scheduled sampling
        teacher_prob = max(0.3, 1.0 - (epoch / actual_epochs) * 0.7)
        # Phase 4: Embedding noise
        noise_std = 0.02 * max(0.0, 1.0 - epoch / actual_epochs)
        # Label smoothing (constant)
        label_smooth = 0.05

        model.train()
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            batch_mask = train_loss_mask[i : i + batch_size] if train_loss_mask is not None else None
            batch_weights = train_weights[i : i + batch_size] if train_weights is not None else None

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = compute_loss_with_scheduled_sampling(
                    model,
                    batch,
                    teacher_prob=teacher_prob,
                    noise_std=noise_std,
                    boundary_mask=batch_mask,
                    sample_weights=batch_weights,
                    label_smoothing=label_smooth,
                )
                loss = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step_count + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            total_loss += loss.item() * grad_accum_steps
            n_batches += 1
            step_count += 1

        # Update SWA model after swa_start epoch
        if use_swa and swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        avg_loss = total_loss / max(1, n_batches)

        # Validation
        val_loss = avg_loss
        if val_data is not None and len(val_data) > 0:
            model.eval()
            vloss = 0.0
            vn_batches = 0
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i : i + batch_size]
                    vbatch_mask = val_loss_mask[i : i + batch_size] if val_loss_mask is not None else None
                    val_batch_weights = val_weights[i : i + batch_size] if val_weights is not None else None
                    vbatch_loss = weighted_compute_loss(
                        model, batch, boundary_mask=vbatch_mask, sample_weights=val_batch_weights
                    )
                    vloss += vbatch_loss.item()
                    vn_batches += 1
            val_loss = vloss / max(1, vn_batches)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"[Train] Epoch {epoch + 1}/{actual_epochs}: train={avg_loss:.4f} val={val_loss:.4f} (best={best_val_loss:.4f} @ ep{best_epoch + 1})"
        )

        # Early stopping
        if no_improve >= patience:
            print(f"[Train] Early stopping at epoch {epoch + 1} -- no improvement for {patience} epochs")
            break

    # Finalize SWA
    if use_swa and swa_model is not None:
        torch.optim.swa_utils.update_bn(train_data, swa_model, device=device)
        model = swa_model.module

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Train] Restored best model from epoch {best_epoch + 1} (val_loss={best_val_loss:.4f})")

    # ── Save checkpoint ────────────────────────────────────────────
    save_training_checkpoint(
        model,
        best_val_loss,
        best_epoch,
        epoch,
        len(sequences),
        checkpoint_dir,
        config_name=model_size,
        output_path=output_path,
    )

    # Cleanup
    del model, optimizer
    if swa_model is not None:
        del swa_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
