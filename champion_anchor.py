#!/usr/bin/env python3
"""ChampionAnchor — Level-based champion persistence for the Evolution Engine.

Stores champions as individual files: champion_l<N>.<ext> in a champions/ directory.
Supports loading/saving by level with SHA-256 integrity verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct as _struct

# R22: Binary champion format constants
_NULL_4: bytes = _struct.pack("<4B", 0, 0, 0, 0)  # 4-byte reserved field


class IntegrityError(Exception):
    """Raised when a champion file's SHA-256 checksum doesn't match."""

    pass


def sha256_program(program: list[int]) -> str:
    """Compute SHA-256 hash of a program (list of ints).

    Uses struct.pack for binary serialization instead of json.dumps with
    hex conversion -- ~10-20x faster for typical program lengths (5-50 words).
    Called thousands of times per evolution run (once per recurrent feedback
    candidate, once per champion save).
    """
    data = _struct.pack(f"<{len(program)}I", *program)
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
        """Save a champion program to a file using atomic writes.

        Uses temp file + os.replace to prevent corruption if the process
        crashes mid-write.  This is critical for long-running evolution
        cycles where a crash during champion save would destroy the best
        program found so far.

        R22: Uses binary format (.bin) with struct.pack for program data
        (5-10x faster than json for integer arrays). Falls back to .json
        for backward compatibility on load.

        Args:
            program: List of 32-bit instruction words.
            base_path: File path to save (e.g., champions/champion_l5).
                The .bin extension is added automatically.
        """
        import tempfile

        checksum = sha256_program(program)
        path = f"{base_path}.bin"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Atomic write: write to temp, then rename (POSIX-guaranteed atomic)
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                # Header: 8-byte magic + 64-byte sha256 hex + 4-byte count
                # Followed by raw uint32 program data
                prog_len = len(program)
                f.write(b"CHMP")  # magic
                f.write(_NULL_4)  # reserved
                f.write(checksum.encode("ascii"))  # 32 bytes
                f.write(_struct.pack("<I", prog_len))  # 4 bytes
                if prog_len > 0:
                    f.write(_struct.pack(f"<{prog_len}I", *program))
            os.replace(tmp_path, path)
        except (OSError, PermissionError):
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self, base_path: str) -> list[int]:
        """Load a champion program from a file with integrity verification.

        R22: Tries binary format (.bin) first, falls back to JSON (.json)
        for backward compatibility. Binary load is 5-10x faster for integer
        arrays.

        Args:
            base_path: File path to load (e.g., champions/champion_l5).
                Extension is added automatically.

        Returns:
            List of 32-bit instruction words.

        Raises:
            IntegrityError: If the SHA-256 checksum doesn't match.
            FileNotFoundError: If the file doesn't exist.
        """
        # Try binary format first (R22)
        bin_path = f"{base_path}.bin"
        if os.path.isfile(bin_path):
            return self._load_binary(bin_path)
        # Fallback to JSON format
        json_path = f"{base_path}.json"
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"No champion file found: {base_path}.*")
        with open(json_path) as f:
            record = json.load(f)
        program = record.get("program", [])
        checksum = record.get("sha256", "")
        if checksum:
            actual = sha256_program(program)
            if actual != checksum:
                raise IntegrityError(f"SHA-256 mismatch for {json_path}: " f"expected {checksum}, got {actual}")
        return program

    def _load_binary(self, path: str) -> list[int]:
        """Load champion from binary format (R22).

        Format: CHMP magic + 4 reserved bytes + 64-byte sha256 hex +
                4-byte little-endian count + count * 4 bytes of uint32 data.
        """
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"CHMP":
                raise IntegrityError(f"Invalid binary magic in {path}")
            _reserved = f.read(4)  # skip reserved bytes
            checksum_bytes = f.read(64)  # SHA-256 hex digest = 64 chars
            checksum = checksum_bytes.decode("ascii")
            count_data = f.read(4)
            if len(count_data) < 4:
                raise IntegrityError(f"Truncated binary header in {path}")
            prog_len = _struct.unpack("<I", count_data)[0]
            if prog_len > 0:
                raw = f.read(prog_len * 4)
                if len(raw) < prog_len * 4:
                    raise IntegrityError(f"Truncated program data in {path}")
                program = list(_struct.unpack(f"<{prog_len}I", raw))
            else:
                program = []
        if checksum:
            actual = sha256_program(program)
            if actual != checksum:
                raise IntegrityError(f"SHA-256 mismatch for {path}: " f"expected {checksum}, got {actual}")
        return program
