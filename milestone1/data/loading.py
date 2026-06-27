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
            # Reject model-generated programs that don't solve any task
            if is_model_generated and task_score <= 0:
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
