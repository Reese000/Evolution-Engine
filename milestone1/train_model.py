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

# Ensure project paths are on sys.path so subpackages resolve.
from pathlib import Path as _Path

_root = str(_Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
from paths import ensure_project_on_path

ensure_project_on_path()

from training.loop import train
from data.augmentation import register_rename_augment
from data.packing import pack_sequences

__all__ = ["train", "register_rename_augment", "pack_sequences"]
