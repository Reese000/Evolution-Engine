"""Data pipeline modules for the training pipeline.

Re-exports all public symbols so callers can do:
    from data import load_champion_traces, pack_sequences, ...
"""

from data.augmentation import (
    inject_reference_solutions,
    inject_synthetic_data,
    register_rename_augment,
)
from data.loading import apply_level_cap, load_champion_traces
from data.packing import pack_sequences, pad_or_pack_sequences

__all__ = [
    "apply_level_cap",
    "inject_reference_solutions",
    "inject_synthetic_data",
    "load_champion_traces",
    "pack_sequences",
    "pad_or_pack_sequences",
    "register_rename_augment",
]
