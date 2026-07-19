#!/usr/bin/env python3
"""
Per-step micro-timer: instruments a single training epoch to measure
where time goes within each step (data fetch, forward, backward, optim).
"""
import gc
import os
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
M1_DIR = os.path.join(SCRIPT_DIR, "milestone1")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if M1_DIR not in sys.path:
    sys.path.insert(0, M1_DIR)

import json as _json

from data.loading import apply_level_cap, load_champion_traces, rebalance_level_weights
from data.packing import pad_or_pack_sequences
from rv32i_mask import decode as rv32i_decode
from rv32i_model import RV32IFieldTokenizer, compute_loss
from training.checkpoint import find_existing_checkpoint
from training.model_factory import load_or_create_model
from training.schedule import build_optimizer_and_schedulers


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    tokenizer = RV32IFieldTokenizer()
    data_path = os.path.join(M1_DIR, "recurrent_data.jsonl")
    sequences, sample_weights, raw_dicts, entry_levels = load_champion_traces(
        data_path, tokenizer, rv32i_decode=rv32i_decode
    )
    apply_level_cap(sequences, sample_weights, raw_dicts, entry_levels)

    # Augmentation
    from data.augmentation import register_rename_augment

    aug_seqs, aug_wts = [], []
    for d in raw_dicts:
        for v in register_rename_augment(d, k=8):
            t = tokenizer.encode_trace(v)
            if t and len(t) >= 2:
                aug_seqs.append(torch.tensor(t, dtype=torch.long))
                aug_wts.append(sample_weights[0] if sample_weights else 1.0)
    sequences.extend(aug_seqs)
    sample_weights.extend(aug_wts)
    del raw_dicts, aug_seqs, aug_wts

    full_entry_levels = entry_levels + [0] * (len(sequences) - len(entry_levels))
    rebalance_level_weights(sample_weights, full_entry_levels)

    split = pad_or_pack_sequences(
        sequences,
        sample_weights,
        max_seq_len=1024,
        val_split=0.15,
        device=device,
    )
    train_data = split["train_data"]
    train_mask = split["train_loss_mask"]
    train_weights = split["train_weights"]

    # Create model
    model, _ = load_or_create_model(None, model_size="xlarge", device=device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler("cuda")

    batch_size = 8
    grad_accum = 2

    # Warmup
    model.train()
    for i in range(0, min(batch_size * 3, len(train_data)), batch_size):
        batch = train_data[i : i + batch_size]
        b_mask = train_mask[i : i + batch_size] if train_mask is not None else None
        b_wts = train_weights[i : i + batch_size] if train_weights is not None else None
        with torch.amp.autocast("cuda", enabled=True):
            loss = compute_loss(model, batch, boundary_mask=b_mask, sample_weights=b_wts)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed single epoch
    n_batches = max(1, len(train_data) // batch_size)
    times = {"fetch": [], "forward": [], "backward": [], "optim": [], "sync": []}

    epoch_start = time.perf_counter()
    for step in range(n_batches):
        i = step * batch_size

        # Data fetch
        t0 = time.perf_counter()
        batch = train_data[i : i + batch_size]
        b_mask = train_mask[i : i + batch_size] if train_mask is not None else None
        b_wts = train_weights[i : i + batch_size] if train_weights is not None else None
        t1 = time.perf_counter()
        times["fetch"].append(t1 - t0)

        # Forward
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=True):
            loss = compute_loss(model, batch, boundary_mask=b_mask, sample_weights=b_wts)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        times["forward"].append(t3 - t2)

        # Backward
        t4 = time.perf_counter()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        times["backward"].append(t5 - t4)

        # Optimizer step
        t6 = time.perf_counter()
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t7 = time.perf_counter()
        times["optim"].append(t7 - t6)

        times["sync"].append((t1 - t0) + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) + (t6 - t5) + (t7 - t6))

    epoch_time = time.perf_counter() - epoch_start
    total_tok = train_data[: batch_size * n_batches].shape[0] * train_data[: batch_size * n_batches].shape[1]
    tok_per_sec = total_tok / epoch_time

    print(f"\n{'Phase':>12s} {'Total ms':>10s} {'Avg ms':>10s} {'%':>6s} {'tok/s (GPU)':>12s}")
    print("-" * 55)
    for phase in ["fetch", "forward", "backward", "optim"]:
        total_ms = sum(times[phase]) * 1000
        avg_ms = (sum(times[phase]) / len(times[phase])) * 1000
        pct = total_ms / (epoch_time * 1000) * 100
        if phase in ("forward", "backward"):
            tok_s = total_tok / (sum(times[phase]))
        else:
            tok_s = 0
        print(f"{phase:>12s} {total_ms:>10.1f} {avg_ms:>10.2f} {pct:>5.1f}% {tok_s:>12.0f}")
    print("-" * 55)
    print(f"{'TOTAL':>12s} {epoch_time*1000:>10.1f} {'':>10s} {'100%':>6s} {tok_per_sec:>12.0f}")
    print(f"\nTokens: {total_tok:,}, Steps: {n_batches}, Batch: {batch_size}, Effective: {batch_size * grad_accum}")
    print(f"Seqs: {len(sequences)}, Packed: {len(train_data)}")

    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"GPU peak: {peak_mem:.0f} MB")


if __name__ == "__main__":
    main()
