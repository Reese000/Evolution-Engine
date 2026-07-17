"""Extracts champion trace loading, tokenization, and per-level capping"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any

import torch


def load_champion_traces(
    data_path: str,
    tokenizer: Any,
    rv32i_decode: Any = None,
) -> tuple[list[torch.Tensor], list[float], list[list[dict]], list[int]]:
    """Load champion traces from a JSONL file and tokenize them.

    Returns:
        (sequences, sample_weights, raw_dicts_for_augment, entry_levels)
        - sequences: list of tokenized tensors
        - sample_weights: per-sample weights (1.0 default)
        - raw_dicts_for_augment: raw instruction dicts for augmentation
        - entry_levels: per-sample curriculum level
    """
    sequences: list[torch.Tensor] = []
    sample_weights: list[float] = []
    raw_dicts_for_augment: list[list[dict]] = []
    entry_levels: list[int] = []

    if not os.path.isfile(data_path):
        return sequences, sample_weights, raw_dicts_for_augment, entry_levels

    seen_sha256: set[str] = set()
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Dedup by champion sha256
            sha = entry.get("champion_sha256", "")
            if sha and sha in seen_sha256:
                continue
            if sha:
                seen_sha256.add(sha)

            program = entry.get("champion_program", [])
            if not program:
                continue

            # Convert hex strings to ints if needed
            if program and isinstance(program[0], str):
                try:
                    program = [int(x, 16) if x.startswith("0x") else int(x) for x in program]
                except (ValueError, TypeError):
                    continue

            # Tokenize
            tokens = tokenizer.encode_program(program)
            if not tokens or len(tokens) < 2:
                continue

            sequences.append(torch.tensor(tokens, dtype=torch.long))
            sample_weights.append(1.0)
            entry_levels.append(entry.get("level", 0))

            # Store raw dicts for augmentation (if decode available)
            if rv32i_decode is not None:
                try:
                    instr_dicts = []
                    for word in program:
                        d = rv32i_decode(word)
                        if d:
                            instr_dicts.append(d)
                    raw_dicts_for_augment.append(instr_dicts)
                except Exception:
                    raw_dicts_for_augment.append([])
            else:
                raw_dicts_for_augment.append([])

    return sequences, sample_weights, raw_dicts_for_augment, entry_levels


def apply_level_cap(
    sequences: list[torch.Tensor],
    sample_weights: list[float],
    raw_dicts: list[list[dict]],
    entry_levels: list[int],
    cap_low: int = 500,
    cap_high: int = 2000,
) -> tuple[list[torch.Tensor], list[float], list[list[dict]]]:
    """Cap per-level counts to prevent data imbalance.

    L0/L7 dominate the data (~50%/35%) and drown out higher-level patterns.
    Each level is capped independently: levels < 9 use *cap_low*, levels >= 9
    use *cap_high*.

    Mutates / replaces the three parallel lists in-place (re-assigns each list
    to the filtered version).

    Returns:
        (sequences, sample_weights, raw_dicts) — the filtered versions.
    """
    _level_counts: Counter = Counter()
    _level_indices: dict[int, list[int]] = {}
    for i, entry_level in enumerate(entry_levels):
        _level_counts[entry_level] += 1
        _level_indices.setdefault(entry_level, []).append(i)

    keep_indices: list[int] = []
    for lvl, indices in _level_indices.items():
        cap = cap_high if lvl >= 9 else cap_low
        if len(indices) > cap:
            # Keep the last `cap` entries (most recent / highest fitness)
            keep_indices.extend(indices[-cap:])
            print(f"  [Cap] L{lvl}: {len(indices)} -> {cap}")
        else:
            keep_indices.extend(indices)

    keep_set = set(keep_indices)
    sequences[:] = [sequences[i] for i in range(len(sequences)) if i in keep_set]
    sample_weights[:] = [sample_weights[i] for i in range(len(sample_weights)) if i in keep_set]
    raw_dicts[:] = [raw_dicts[i] for i in range(len(raw_dicts)) if i in keep_set]

    return sequences, sample_weights, raw_dicts


def rebalance_level_weights(
    sample_weights: list[float],
    entry_levels: list[int],
    target_ratio: float = 5.0,
) -> None:
    """Upsample underrepresented levels via sample_weights.

    After apply_level_cap, high-level traces (L12-L17) can dominate 94% of
    the data while low levels (L0-L8) have <6%. This causes the model to
    predict poorly at low levels, leading to regression on curriculum resets.

    Fix: increase sample_weights for underrepresented levels so the loss
    function pays proportionally more attention to them. The maximum weight
    boost is capped at target_ratio (default 5x) to prevent outlier levels
    from destabilizing training.

    Mutates sample_weights in-place. entry_levels must be the same length.

    Args:
        sample_weights: Per-sample weights (mutated in-place).
        entry_levels: Per-sample level labels (read-only).
        target_ratio: Max weight multiplier for the rarest level vs the most
                      common. E.g. 5.0 means the rarest level gets up to 5x
                      the weight of the most common.
    """
    if not sample_weights or not entry_levels or len(sample_weights) != len(entry_levels):
        return

    level_counts: Counter = Counter(entry_levels)
    if not level_counts:
        return

    max_count = max(level_counts.values())
    if max_count == 0:
        return

    # Compute per-level weight multiplier: rare levels get boosted.
    # mult = min(target_ratio, max_count / level_count)
    level_mult: dict[int, float] = {}
    for lvl, count in level_counts.items():
        if count > 0:
            mult = max_count / count
            level_mult[lvl] = min(target_ratio, mult)
        else:
            level_mult[lvl] = 1.0

    # Apply multipliers to sample_weights
    boosted = 0
    for i, lvl in enumerate(entry_levels):
        mult = level_mult.get(lvl, 1.0)
        if mult > 1.01:
            sample_weights[i] *= mult
            boosted += 1

    if boosted > 0:
        levels_str = ", ".join(
            f"L{l}:{level_counts[l]}x{level_mult[l]:.1f}" for l in sorted(level_mult) if level_mult[l] > 1.01
        )
        print(
            f"  [Rebalance] Boosted {boosted} samples across {len([m for m in level_mult.values() if m > 1.01])} levels: {levels_str}"
        )
