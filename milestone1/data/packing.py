"""Sequence packing and padding utilities.

Bin-packs short sequences to reduce PAD waste, with a fallback to
simple longest-padding.  Extracted from the monolithic train_model.py.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch

# Ensure milestone1/ is on sys.path.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def pack_sequences(
    sequences: list[torch.Tensor],
    max_seq_len: int = 1024,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Sort sequences by length and pack short ones into fixed-width packs.

    Each pack is concatenated up to *max_seq_len*, reducing wasted compute on
    PAD tokens from 40-80% down to near zero.

    Args:
        sequences:   list of 1-D LongTensors (each a tokenized trace).
        max_seq_len: maximum total tokens per pack.

    Returns:
        packed:          (N_packs, max_seq_len) LongTensor of packed sequences
                         (padded with 0 only in the final pack if needed).
        pack_lengths:    list of lists -- per-pack sub-sequence lengths.
                         pack_lengths[p][j] is the token count of the j-th
                         sub-sequence within pack p. Sub-sequences are
                         concatenated in list order within each pack.
    """
    if not sequences:
        return torch.empty(0, 0, dtype=torch.long), []

    # Sort longest-first so greedy bin-packing packs tightly.
    sorted_seqs = sorted(sequences, key=len, reverse=True)

    packs: list[list[torch.Tensor]] = []
    pack_remaining: list[int] = []
    pack_lengths: list[list[int]] = []

    for seq in sorted_seqs:
        seq_len = len(seq)

        # Single sequence exceeds max_seq_len -- truncate and make its own pack.
        if seq_len > max_seq_len:
            packs.append([seq[:max_seq_len]])
            pack_remaining.append(0)
            pack_lengths.append([max_seq_len])
            continue

        # Find first pack with enough room.
        placed = False
        for i in range(len(packs)):
            if pack_remaining[i] >= seq_len:
                packs[i].append(seq)
                pack_lengths[i].append(seq_len)
                pack_remaining[i] -= seq_len
                placed = True
                break

        if not placed:
            packs.append([seq])
            pack_remaining.append(max_seq_len - seq_len)
            pack_lengths.append([seq_len])

    # Concatenate each pack and pad the last one if it is short.
    packed_list: list[torch.Tensor] = []
    for pack in packs:
        t = torch.cat(pack)
        if len(t) < max_seq_len:
            t = torch.cat([
                t,
                torch.zeros(max_seq_len - len(t), dtype=torch.long),
            ])
        packed_list.append(t)

    packed = torch.stack(packed_list)
    return packed, pack_lengths


def pad_or_pack_sequences(
    sequences: list[torch.Tensor],
    sample_weights: list[float],
    *,
    max_seq_len: int = 1024,
    val_split: float = 0.15,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Try sequence packing first; fall back to simple padding.

    Returns a dict with keys:
        packed (bool):             Whether packing succeeded.
        padded (Tensor):           (N, max_seq_len) LongTensor.
        loss_mask (Tensor):        (N, max_seq_len) bool tensor.
        sample_weights (Tensor):   (N,) float tensor.
        train_data, val_data:      Split tensors on *device*.
        train_loss_mask, val_loss_mask: Corresponding masks on *device*.
        train_weights, val_weights:     Corresponding weight tensors.
        n_packs, waste_pct:        Stats for logging.
        max_len:                   Max sequence length (padding fallback only).
    """
    packed = False
    loss_mask = None

    try:
        packed_tensors, pack_lengths = pack_sequences(sequences, max_seq_len=max_seq_len)
        n_packs = len(packed_tensors)
        n_total_tokens = int(packed_tensors.numel())
        n_effective = sum(sum(pl) for pl in pack_lengths)
        waste_pct = 100.0 * (1.0 - n_effective / max(1, n_total_tokens))
        print(f"[Train] Sequence packing: {n_packs} packs from {len(sequences)} sequences, "
              f"padding waste={waste_pct:.1f}%")
        packed = True
        padded = packed_tensors

        # Build loss mask for packed sequences.
        loss_mask = torch.zeros(n_packs, padded.shape[1], dtype=torch.bool)
        for p in range(n_packs):
            lengths = pack_lengths[p]
            offset = 0
            for j, slen in enumerate(lengths):
                if j == 0:
                    loss_mask[p, offset:offset + slen] = True
                else:
                    loss_mask[p, offset + 1:offset + slen] = True
                offset += slen

        result = {"packed": True, "padded": padded, "loss_mask": loss_mask,
                      "n_packs": n_packs, "waste_pct": waste_pct, "max_len": None}

    except Exception as exc:
        print(f"[Train] Sequence packing failed ({exc}), falling back to padding")
        result = {"packed": False, "padded": None, "loss_mask": None,
                      "n_packs": 0, "waste_pct": 0.0, "max_len": None}

    if not result["packed"]:
        # Fallback: simple longest-padding.
        max_len = max(len(s) for s in sequences)
        padded = torch.stack([
            torch.cat([s, torch.zeros(max_len - len(s), dtype=torch.long)])
            for s in sequences
        ])
        loss_mask = padded != 0
        waste_pct = 100.0 * (1.0 - sum(len(s) for s in sequences) / padded.numel())
        print(f"[Train] Padding fallback: {len(sequences)} sequences, max_len={max_len}, "
              f"padding waste={waste_pct:.1f}%")
        result["padded"] = padded
        result["loss_mask"] = loss_mask
        result["waste_pct"] = waste_pct
        result["max_len"] = max_len

    # ── Train/val split ────────────────────────────────────────────
    padded = result["padded"]
    lmask = result["loss_mask"]
    import random as _rng
    _idx = list(range(len(padded)))
    _rng.seed(42)
    _rng.shuffle(_idx)
    _n_val = max(1, int(len(padded) * val_split))
    _val_idx = set(_idx[:_n_val])
    _train_idx = [i for i in range(len(padded)) if i not in _val_idx]

    _dev = device or torch.device("cpu")
    result["train_data"] = padded[_train_idx].to(_dev)
    result["val_data"] = padded[list(_val_idx)].to(_dev) if _n_val > 0 else None
    result["train_loss_mask"] = lmask[_train_idx].to(_dev) if lmask is not None else None
    result["val_loss_mask"] = lmask[list(_val_idx)].to(_dev) if lmask is not None and _n_val > 0 else None

    # Phase 5: split sample_weights alongside train/val
    result["train_weights"] = torch.tensor(
        [sample_weights[i] for i in _train_idx], dtype=torch.float, device=_dev,
    )
    result["val_weights"] = (
        torch.tensor([sample_weights[i] for i in _val_idx], dtype=torch.float, device=_dev)
        if _n_val > 0 else None
    )

    return result
