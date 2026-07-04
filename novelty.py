"""novelty.py — Novelty Search archive with k-nearest-neighbor scoring.

Stores behavioural descriptors of programs and scores novelty as the
average distance to the k-nearest neighbours in descriptor space.

Behavioural descriptors are dictionaries with a fixed set of numeric
features derived from program execution traces:

  - ``inst_diversity``:   fraction of distinct opcode categories used
  - ``reg_pressure``:     fraction of active (non-zero) general-purpose regs
  - ``program_length``:   normalised instruction count
  - ``branch_fraction``:  fraction of branch-type instructions
  - ``alu_fraction``:     fraction of ALU-type instructions
  - ``load_store_ratio``: ratio of memory ops to total ops (0 if none)
  - ``unique_regs_used``: number of distinct registers written to / 31

Public API
----------
extract_behavioral_descriptor(trace, program, budget) -> dict
SparseArchive(k_neighbors, novelty_threshold)
    .add_to_archive(descriptor, metadata, check_threshold) -> (bool, float)
    .novelty_score(descriptor) -> float
    .size -> int
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import KDTree

# Default k for k-nearest-neighbour novelty scoring.
_DEFAULT_K: int = 5

# Default novelty threshold below which descriptors are rejected.
_DEFAULT_THRESHOLD: float = 0.15

# Opcode categories matching evolution_engine.py's OPCODE_CATEGORY_MAP.
# RV32I base instructions grouped by functional unit.
_OPCODE_CATEGORIES: dict[str, list[str]] = {
    "ALU": [
        "ADD",
        "SUB",
        "SLL",
        "SLT",
        "SLTU",
        "XOR",
        "SRL",
        "SRA",
        "OR",
        "AND",
        "ADDI",
        "SLTI",
        "SLTIU",
        "XORI",
        "ORI",
        "ANDI",
        "SLLI",
        "SRLI",
        "SRAI",
    ],
    "BRANCH": ["BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"],
    "JUMP": ["JAL", "JALR"],
    "LOAD": ["LB", "LH", "LW", "LBU", "LHU"],
    "STORE": ["SB", "SH", "SW"],
    "LUI": ["LUI"],
    "AUIPC": ["AUIPC"],
    "SYSTEM": ["ECALL", "EBREAK", "FENCE"],
}

# Mapping from opcode name to category index.
_OPCODE_TO_CAT: dict[str, int] = {}
for _cat_idx, (_cat_name, _opcodes) in enumerate(_OPCODE_CATEGORIES.items()):
    for _op in _opcodes:
        _OPCODE_TO_CAT[_op] = _cat_idx

_NUM_CATEGORIES: int = len(_OPCODE_CATEGORIES)


def _opcode_category_bits(program: list[int]) -> int:
    """Return a bitset of which opcode categories appear in *program*."""
    seen = 0
    for word in program:
        if not isinstance(word, int):
            continue
        # Extract opcode name from the 7-bit opcode field (bits 0-6)
        # For simplicity, use the low 7 bits to index a prebuilt map.
        # We approximate by mapping common encoding ranges.
        low7 = word & 0x7F
        # RV32I opcode map (simplified):
        # 0x33/0x13 = ALU, 0x63 = BRANCH, 0x6F = JAL, 0x67 = JALR
        # 0x03 = LOAD, 0x23 = STORE, 0x37 = LUI, 0x17 = AUIPC, 0x73 = SYSTEM
        if low7 in (0x33, 0x13):
            cat = 0  # ALU
        elif low7 == 0x63:
            cat = 1  # BRANCH
        elif low7 in (0x6F, 0x67):
            cat = 2  # JUMP
        elif low7 == 0x03:
            cat = 3  # LOAD
        elif low7 == 0x23:
            cat = 4  # STORE
        elif low7 == 0x37:
            cat = 5  # LUI
        elif low7 == 0x17:
            cat = 6  # AUIPC
        elif low7 == 0x73:
            cat = 7  # SYSTEM
        else:
            continue
        seen |= 1 << cat
    return seen


def extract_behavioral_descriptor(
    trace: dict[str, Any],
    program: list[int],
    budget: int = 200,
) -> dict[str, float]:
    """Extract a behavioural descriptor from an execution trace and program.

    Returns a dictionary of normalised numeric features suitable for
    novelty comparison.
    """
    n_words = len(program) if program else 0
    n_budget = max(1, budget)

    # Instruction diversity: fraction of distinct opcode categories
    if program and isinstance(program, list) and n_words > 0:
        cat_bits = _opcode_category_bits(program)
        inst_diversity = cat_bits.bit_count() / max(1, _NUM_CATEGORIES)
    else:
        inst_diversity = 0.0

    # Register pressure: fraction of active registers in final state
    final_regs = trace.get("final_regs", {}) if isinstance(trace, dict) else {}
    if isinstance(final_regs, dict):
        active_regs = sum(1 for idx, val in final_regs.items() if isinstance(idx, int) and 1 <= idx <= 31 and val != 0)
        reg_pressure = active_regs / 31.0
    else:
        reg_pressure = 0.0

    # Program length (normalised by budget)
    program_length = min(n_words / n_budget, 1.0)

    # Branch fraction
    branch_count = 0
    load_count = 0
    store_count = 0
    total_ops = 0
    regs_written: set[int] = set()
    if program:
        for word in program:
            if not isinstance(word, int):
                continue
            low7 = word & 0x7F
            total_ops += 1
            if low7 == 0x63:
                branch_count += 1
            elif low7 == 0x03:
                load_count += 1
            elif low7 == 0x23:
                store_count += 1
            # Extract rd (bits 7-11) for unique regs written
            rd = (word >> 7) & 0x1F
            if 1 <= rd <= 31:
                regs_written.add(rd)

    branch_fraction = branch_count / max(1, total_ops)
    alu_count = total_ops - branch_count - load_count - store_count
    # Remove system/jump from alu count for a rough fraction
    alu_fraction = max(0, alu_count) / max(1, total_ops)
    load_store_ratio = (load_count + store_count) / max(1, total_ops)
    unique_regs_used = len(regs_written) / 31.0

    return {
        "inst_diversity": inst_diversity,
        "reg_pressure": reg_pressure,
        "program_length": program_length,
        "branch_fraction": branch_fraction,
        "alu_fraction": alu_fraction,
        "load_store_ratio": load_store_ratio,
        "unique_regs_used": unique_regs_used,
    }


def _flatten_descriptor(desc: dict[str, float]) -> list[float]:
    """Convert a descriptor dict to a sorted-key list for distance computation."""
    return [desc[k] for k in sorted(desc.keys())]


def _euclidean(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two equal-length vectors."""
    total = 0.0
    for x, y in zip(a, b):
        d = x - y
        total += d * d
    return math.sqrt(total)


class SparseArchive:
    """Novelty archive that stores behavioural descriptors of novel individuals.

    Novelty is measured as the average Euclidean distance to the k-nearest
    neighbours in the archive.  Descriptors whose novelty score falls below
    a configurable threshold are rejected.
    """

    def __init__(
        self,
        k_neighbors: int = _DEFAULT_K,
        novelty_threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self.k_neighbors: int = k_neighbors
        self.novelty_threshold: float = novelty_threshold

        # Storage: list of (flattened_descriptor, original_descriptor, metadata)
        self._entries: list[tuple[list[float], dict[str, float], dict[str, Any]]] = []
        self.size: int = 0

        # KD-tree for fast k-NN queries
        self._tree: KDTree | None = None
        self._tree_dirty: bool = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_tree(self) -> None:
        """Rebuild the KD-tree from current entries."""
        if not self._entries:
            self._tree = None
            return
        points = [entry[0] for entry in self._entries]
        self._tree = KDTree(points)
        self._tree_dirty = False

    def _k_nn_distance(self, flat: list[float]) -> float:
        """Compute average distance to the k-nearest neighbours.

        If the archive has fewer than k entries, uses all entries.
        Returns 0.0 for an empty archive.
        """
        if self.size == 0:
            return 0.0
        if self._tree_dirty:
            self._rebuild_tree()
        if self._tree is None:
            return 0.0

        k = min(self.k_neighbors, self.size)
        dists, _ = self._tree.query(np.array(flat), k=k)
        if k == 1:
            return float(dists)
        return float(np.mean(dists))

    def adaptive_threshold(self) -> float:
        """Compute adaptive novelty threshold based on archive size.

        Starts at 0.3 (high for early exploration) and decreases as the
        archive grows, with a floor of 0.05.
        """
        base = 0.3
        min_threshold = 0.05
        return max(min_threshold, base / max(1.0, self.size**0.5))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def novelty_score(self, descriptor: dict[str, float]) -> float:
        """Compute the novelty score of a descriptor against the archive.

        Novelty is the average Euclidean distance to the k-nearest
        neighbours.  Returns 0.0 if the archive is empty.
        """
        flat = _flatten_descriptor(descriptor)
        return self._k_nn_distance(flat)

    def add_to_archive(
        self,
        descriptor: dict[str, float],
        metadata: dict[str, Any] | None = None,
        check_threshold: bool = True,
    ) -> tuple[bool, float]:
        """Add a descriptor to the archive if it is sufficiently novel.

        Returns ``(was_added, novelty_score)``.  The descriptor is added
        if its novelty score exceeds the adaptive threshold (when
        ``check_threshold`` is True) or unconditionally when the archive
        is empty.
        """
        if metadata is None:
            metadata = {}

        flat = _flatten_descriptor(descriptor)
        score = self._k_nn_distance(flat)

        # Always add to an empty archive (first entry establishes baseline).
        if self.size == 0:
            self._entries.append((flat, descriptor, metadata))
            self.size = 1
            self._tree_dirty = True
            return True, score

        threshold = self.adaptive_threshold()
        if check_threshold and score < threshold:
            return False, score

        self._entries.append((flat, descriptor, metadata))
        self.size = len(self._entries)
        self._tree_dirty = True
        return True, score

    def clear(self) -> None:
        """Reset the archive."""
        self._entries.clear()
        self.size = 0
        self._tree_dirty = True

    def recent_descriptors(self, n: int = 10) -> list[dict[str, float]]:
        """Return the last *n* descriptor dicts added to the archive."""
        return [entry[1] for entry in self._entries[-n:]]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        """Serialize archive to JSON-compatible dict."""
        return {
            "entries": [{"descriptor": entry[1], "program": entry[2]} for entry in self._entries],
            "threshold": self.novelty_threshold,
        }

    @classmethod
    def deserialize(cls, data: dict, k_neighbors: int = 10) -> "SparseArchive":
        """Reconstruct from serialized data."""
        archive = cls(k_neighbors=k_neighbors, novelty_threshold=data.get("threshold", _DEFAULT_THRESHOLD))
        for entry in data.get("entries", []):
            archive.add_to_archive(entry["descriptor"], entry.get("program", {}))
        return archive


# Public alias -- evolution_engine and tests may import by either name.
NoveltyArchive = SparseArchive
