#!/usr/bin/env python3
"""train_model -- backward-compatible facade for the modular training pipeline.

All public symbols are re-exported here so that:
  - ``from train_model import train`` works (called by run_multi_cycle.py)
  - ``from train_model import register_rename_augment`` works
  - ``from train_model import pack_sequences`` works
"""

from __future__ import annotations

import os
import sys

# Ensure milestone1/ is on sys.path so the subpackages resolve.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
_LOGIT_MASK_DIR = os.path.join(BASE_DIR, "..", "milestone0", "logit-masking")
if os.path.isdir(_LOGIT_MASK_DIR) and _LOGIT_MASK_DIR not in sys.path:
    sys.path.insert(0, _LOGIT_MASK_DIR)

from training.loop import train
from data.augmentation import register_rename_augment
from data.packing import pack_sequences

__all__ = ["train", "register_rename_augment", "pack_sequences"]
