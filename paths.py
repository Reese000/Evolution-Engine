#!/usr/bin/env python3
"""paths.py — Centralized project path resolution.

All modules should import paths from here instead of using
relative sys.path.insert hacks. This eliminates fragile CWD-dependent
imports and provides a single source of truth for project layout.

Usage:
    from paths import PROJECT_ROOT, MILESTONE0_DIR, MILESTONE1_DIR
    ensure_project_on_path()
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Root ───────────────────────────────────────────────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ── Milestones ─────────────────────────────────────────────────────────

MILESTONE0_DIR: Path = PROJECT_ROOT / "milestone0"
MILESTONE1_DIR: Path = PROJECT_ROOT / "milestone1"
MILESTONE3_DIR: Path = PROJECT_ROOT / "milestone3"

# ── Sub-directories used by imports ────────────────────────────────────

MILESTONE0_LOGIT_MASKING: Path = MILESTONE0_DIR / "logit-masking"
MILESTONE0_VM_FARM: Path = MILESTONE0_DIR / "vm-farm"
MILESTONE0_GPU_VM: Path = MILESTONE0_DIR / "gpu-vm"

# ── Checkpoints & data ────────────────────────────────────────────────

M1_CHECKPOINTS: Path = MILESTONE1_DIR / "checkpoints"
M1_RECURRENT: Path = MILESTONE1_DIR / "recurrent_data"
M1_TRAIN_MODEL: Path = MILESTONE1_DIR / "train_model.py"

# ── Config & results ──────────────────────────────────────────────────

CONFIG_DIR: Path = PROJECT_ROOT / "config"
ENGINE_CONFIG: Path = CONFIG_DIR / "engine_config.json"
TOOL_LIBRARY: Path = CONFIG_DIR / "tool_library.json"

RESULTS_FILE: Path = PROJECT_ROOT / "multi_cycle_results.jsonl"
META_CONTROLLER_STATE: Path = PROJECT_ROOT / "meta_controller_state.json"
BARRIER_TUNER_STATE: Path = PROJECT_ROOT / "barrier_tuner_state.json"

# ── Utility: add all project paths to sys.path ────────────────────────


def ensure_project_on_path() -> None:
    """Add all project-critical directories to sys.path (idempotent).

    Call once at startup of any entry-point script. Import-only modules
    should NOT call this -- they should import specific modules directly.
    """
    import sys

    dirs = [
        PROJECT_ROOT,
        MILESTONE0_LOGIT_MASKING,
        MILESTONE0_VM_FARM,
        MILESTONE0_GPU_VM,
        MILESTONE3_DIR,
        MILESTONE1_DIR,
    ]
    for d in dirs:
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))


# ── All importable paths as a flat list (backward-compat) ──────────────

ALL_IMPORT_PATHS: list[Path] = [
    PROJECT_ROOT,
    MILESTONE0_LOGIT_MASKING,
    MILESTONE0_VM_FARM,
    MILESTONE0_GPU_VM,
    MILESTONE3_DIR,
    MILESTONE1_DIR,
]
