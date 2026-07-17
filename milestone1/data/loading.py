"""Data loading and level-capping utilities.

Extracts champion trace loading, tokenization, and per-level capping
from the monolithic train_model.py.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import torch

# Ensure milestone1/ and parent are on sys.path so sibling modules resolve.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# milestone0/logit-masking (for rv32i_mask)
_LOGIT_MASK_DIR = os.path.join(BASE_DIR, "..", "milestone0", "logit-masking")
if os.path.isdir(_LOGIT_MASK_DIR) and _LOGIT_MASK_DIR not in sys.path:
    sys.path.insert(0, _LOGIT_MASK_DIR)


def load_champion_traces(
    data_path: str,
    tokenizer,
    *,
    rv32i_decode,
) -> tuple[list[torch.Tensor], list[float], list[list[dict]], list[int]]:
    """Load champion traces from a JSONL file and tokenize them.

    Args:
        data_path:  Path to recurrent_data.jsonl.
        tokenizer:  An RV32IFieldTokenizer instance.
        rv32i_decode: The rv32i_mask.decode function.

    Returns:
        (sequences, sample_weights, raw_dicts_for_augment, entry_levels)
    """
    from rv32i_mask import decode as _default_decode

    if rv32i_decode is None:
        rv32i_decode = _default_decode

    sequences: list[torch.Tensor] = []
    sample_weights: list[float] = []
    raw_dicts_for_augment: list[list[dict]] = []
    entry_levels: list[int] = []
    seen_hashes: set[int] = set()

    if not os.path.isfile(data_path):
        return sequences, sample_weights, raw_dicts_for_augment, entry_levels

    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Filter low-quality traces
            fitness = entry.get("fitness", 0)
            task_score = entry.get("task_score", 0)
            is_reference = entry.get("save_reason") == "reference_solution"
            is_model_generated = not is_reference and entry.get("generation", -1) >= 0
            if fitness <= 0 or (task_score > 0 and task_score < 10.0):
                continue
            # Accept model-generated programs only if very high-fitness
            # (model learns best from top-quality examples, not average ones)
            if is_model_generated and task_score <= 0 and fitness < 100:
                continue
            # Skip Level 10 entries — nearly all have fitness < 100 (0.02% pass rate)
            # These drag down training quality and cause model regression
            entry_level = entry.get("level", 0)
            if entry_level == 10 and not is_reference:
                continue
            # Skip short programs (< 10 instructions) — model should learn from
            # longer programs that solve tasks better (28-30 instr have highest fitness)
            # R43: Level-aware threshold — low levels (L0-L8) legitimately produce
            # short programs (3-9 instr). Only apply the short-program filter to
            # high levels where longer programs are expected.
            instr_count = len(entry.get("champion_program", []))
            min_instr = 10 if entry_level >= 9 else 3
            if is_model_generated and instr_count < min_instr:
                continue

            prog = entry.get("champion_program", [])
            if not prog:
                continue

            entry_level = entry.get("level", 0)

            # Convert hex strings -> ints -> instruction dicts -> tokens
            instr_dicts = []
            for hw in prog:
                if isinstance(hw, str) and hw.startswith("0x"):
                    word = int(hw, 16)
                elif isinstance(hw, int):
                    word = hw
                else:
                    continue
                try:
                    decoded = rv32i_decode(word)
                    if decoded:
                        instr_dicts.append(decoded)
                except (ValueError, KeyError):
                    continue

            if not instr_dicts:
                continue

            # Tokenize
            tokens = tokenizer.encode_trace(instr_dicts)
            if tokens and len(tokens) >= 2:
                # Dedup
                seq_hash = hash(tuple(tokens))
                if seq_hash in seen_hashes:
                    continue
                seen_hashes.add(seq_hash)
                sequences.append(torch.tensor(tokens, dtype=torch.long))
                raw_dicts_for_augment.append(instr_dicts)
                entry_levels.append(entry_level)
                # Phase 5: compute reward weight for this sample
                ts = entry.get("task_score", None)
                if ts is not None and ts > 0:
                    sw = max(0.1, min(1.0, ts / 120.0))
                else:
                    sw = max(0.1, min(1.0, entry.get("fitness", 1.0) / 150.0))
                sample_weights.append(sw)

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

    After apply_level_cap, high-level traces can dominate the data while
    low levels are underrepresented. This causes the model to predict
    poorly at low levels, leading to regression on curriculum resets.

    Increases sample_weights for underrepresented levels so the loss
    function pays proportionally more attention to them. The maximum
    weight boost is capped at target_ratio to prevent outlier levels
    from destabilizing training.

    Mutates sample_weights in-place. entry_levels must be the same length.
    """
    from collections import Counter

    if not sample_weights or not entry_levels or len(sample_weights) != len(entry_levels):
        return

    level_counts: Counter = Counter(entry_levels)
    if not level_counts:
        return

    max_count = max(level_counts.values())
    if max_count == 0:
        return

    # mult = min(target_ratio, max_count / level_count)
    level_mult: dict[int, float] = {}
    for lvl, count in level_counts.items():
        level_mult[lvl] = min(target_ratio, max_count / count) if count > 0 else 1.0

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
            f"  [Rebalance] Boosted {boosted} samples across "
            f"{sum(1 for m in level_mult.values() if m > 1.01)} levels: {levels_str}"
        )
