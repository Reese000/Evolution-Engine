#!/usr/bin/env python3
"""ChampionAnchor — Level-based champion persistence for the Evolution Engine.

Stores champions as individual files: champion_l<N>.<ext> in a champions/ directory.
Supports loading/saving by level with SHA-256 integrity verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any


class IntegrityError(Exception):
    """Raised when a champion file's SHA-256 checksum doesn't match."""
    pass


def sha256_program(program: list[int]) -> str:
    """Compute SHA-256 hash of a program (list of ints)."""
    data = json.dumps([hex(w) for w in program], sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


class ChampionAnchor:
    """Level-based champion persistence with integrity verification.

    Stores champions as individual files in a champions/ subdirectory.
    Each file is named champion_l<N>.json and contains:
        {"sha256": "...", "program": [...], "level": N}
    """

    def __init__(self, champions_dir: str | None = None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.champions_dir = champions_dir or os.path.join(base, "champions")
        os.makedirs(self.champions_dir, exist_ok=True)

    def save(self, program: list[int], base_path: str) -> None:
        """Save a champion program to a file.

        Args:
            program: List of 32-bit instruction words.
            base_path: File path to save (e.g., champions/champion_l5).
                The .json extension is added automatically.
        """
        checksum = sha256_program(program)
        record = {
            "sha256": checksum,
            "program": program,
        }
        path = f"{base_path}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f)

    def load(self, base_path: str) -> list[int]:
        """Load a champion program from a file with integrity verification.

        Args:
            base_path: File path to load (e.g., champions/champion_l5).
                The .json extension is added automatically.

        Returns:
            List of 32-bit instruction words.

        Raises:
            IntegrityError: If the SHA-256 checksum doesn't match.
            FileNotFoundError: If the file doesn't exist.
        """
        path = f"{base_path}.json"
        with open(path) as f:
            record = json.load(f)
        program = record.get("program", [])
        checksum = record.get("sha256", "")
        if checksum and sha256_program(program) != checksum:
            raise IntegrityError(
                f"SHA-256 mismatch for {path}: "
                f"expected {checksum}, got {sha256_program(program)}"
            )
        return program
