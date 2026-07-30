"""Data pipeline modules for the training pipeline.

Re-exports all public symbols so callers can do:
    from data import load_champion_traces, pack_sequences, ...
"""

from data.loading import load_champion_traces, apply_level_cap
from data.packing import pack_sequences, pad_or_pack_sequences
from data.augmentation import (
    register_rename_augment,
    inject_synthetic_data,
    inject_reference_solutions,
)

__all__ = [
    "load_champion_traces",
    "apply_level_cap",
    "pack_sequences",
    "pad_or_pack_sequences",
    "register_rename_augment",
    "inject_synthetic_data",
    "inject_reference_solutions",
]
