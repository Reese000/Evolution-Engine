#!/usr/bin/env python3
"""cycle_state.py — Complete cycle state persistence with integrity checksums.

Defines the canonical cycle_state.json schema and provides load/save
with SHA-256 checksum verification to detect corruption.

Schema (version 2):
{
  "version": 2,
  "checksum": "<sha256-hex>",
  "current_cycle": 11,
  "best_fitness": -0.085,
  "best_level": 2,
  "best_checkpoint": "milestone1/checkpoints/checkpoint_best.pt",
  "champion_program": [...],
  "strategy_archive": {...},
  "diversity_score": 0.15,
  "consecutive_stalls": 0,
  "consecutive_errors": 0,
  "total_errors": 0,
  "last_cycle_timestamp": "...",
  "start_timestamp": "...",
  "cycles_completed": 11,
  "stop_reason": null,
  "population_size": 50,
  "retrain_config": {...},
  "metrics_summary": {
    "fitness_trend": "improving",
    "val_loss_last": 0.0123
  }
}

Usage:
    from cycle_state import CycleState
    cs = CycleState("cycle_state.json")
    cs.load()   # Returns True on success, False on corruption
    cs.save()
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 2

# Fields that are included in checksum (excludes "checksum" itself)
_CHECKSUMMED_KEYS = sorted(
    [
        "version",
        "current_cycle",
        "best_fitness",
        "best_level",
        "best_checkpoint",
        "champion_program",
        "strategy_archive",
        "diversity_score",
        "consecutive_stalls",
        "consecutive_errors",
        "total_errors",
        "last_cycle_timestamp",
        "start_timestamp",
        "cycles_completed",
        "stop_reason",
        "population_size",
        "retrain_config",
        "metrics_summary",
    ]
)


def _compute_checksum(data: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 checksum over checksummed fields."""
    payload = {}
    for key in _CHECKSUMMED_KEYS:
        if key in data:
            payload[key] = data[key]
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def default_cycle_state() -> dict[str, Any]:
    """Return a fresh cycle_state dict with all fields."""
    return {
        "version": SCHEMA_VERSION,
        "current_cycle": 0,
        "best_fitness": 0.0,
        "best_level": 0,
        "best_checkpoint": "",
        "champion_program": [],
        "strategy_archive": {},
        "diversity_score": 0.0,
        "consecutive_stalls": 0,
        "consecutive_errors": 0,
        "total_errors": 0,
        "last_cycle_timestamp": "",
        "start_timestamp": datetime.now(timezone.utc).isoformat(),
        "cycles_completed": 0,
        "stop_reason": None,
        "population_size": 50,
        "retrain_config": {},
        "metrics_summary": {},
    }


class CycleState:
    """Persist and validate the complete cycle state with integrity checks.

    Args:
        path: Filesystem path for cycle_state.json.
    """

    def __init__(self, path: str = "cycle_state.json") -> None:
        self.path = path
        self.data: dict[str, Any] = default_cycle_state()

    # ── Save ──────────────────────────────────────────────────────────

    def save(self) -> bool:
        """Write state to disk with a fresh checksum using atomic write.

        Uses temp file + rename to prevent corruption if the process
        crashes mid-write.  Returns True on success.
        """
        self.data["last_cycle_timestamp"] = datetime.now(timezone.utc).isoformat()
        self.data["checksum"] = _compute_checksum(self.data)
        try:
            dir_name = os.path.dirname(self.path) or "."
            os.makedirs(dir_name, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=2)
                os.replace(tmp_path, self.path)
            except (OSError, PermissionError):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            return True
        except (OSError, PermissionError):
            return False

    # ── Load ──────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load state from disk and verify integrity.

        Returns True if loaded + checksum OK.
        Returns False if file missing, corrupt, or checksum mismatch.
        On failure, data is reset to defaults.
        """
        if not os.path.isfile(self.path):
            self.data = default_cycle_state()
            return False

        try:
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            self.data = default_cycle_state()
            return False

        # Version check
        if raw.get("version", 0) < 1:
            self.data = default_cycle_state()
            return False

        # Checksum verification
        # Use dict copy to avoid mutating the caller's dict via .pop()
        stored = raw.get("checksum", None)
        raw_for_checksum = {k: v for k, v in raw.items() if k != "checksum"}
        computed = _compute_checksum(raw_for_checksum)
        if stored is not None and stored != computed:
            self.data = default_cycle_state()
            return False

        # Merge into our data (keeps any extra fields from file)
        result = default_cycle_state()
        result.update(raw_for_checksum)
        self.data = result
        return True

    # ── Convenience ───────────────────────────────────────────────────

    def update(self, **kwargs: Any) -> None:
        """Update multiple state fields at once."""
        self.data.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def reset(self) -> None:
        """Reset to fresh defaults (preserves start_timestamp)."""
        start = self.data.get("start_timestamp", datetime.now(timezone.utc).isoformat())
        self.data = default_cycle_state()
        self.data["start_timestamp"] = start

    def file_exists(self) -> bool:
        return os.path.isfile(self.path)
