"""qd_archive.py — MAP-Elites Quality-Diversity archive.

Grid-based archive that stores the highest-fitness program for each
behavioural cell in a discretised behaviour space.  Descriptors are
2-D vectors in [-1, 1] with the following axes:

  0: instruction diversity (fraction of distinct opcode categories used)
  1: register pressure     (fraction of non-zero general-purpose registers)

Public API
----------
MAPElitesArchive()
    .add_solution(descriptor, fitness, program, trace) -> bool
    .selection_bias(descriptor) -> float
    .coverage() -> float
    .grid_solutions() -> list[dict]

The archive is intentionally stateless w.r.t. I/O — the caller
(``evolution_engine.py``) owns persistence.
"""

from __future__ import annotations

import math
from typing import Any

# Default grid resolution per axis.  20 x 20 = 400 cells gives finer
# granularity for behaviour-space coverage while still remaining tractable.
_DEFAULT_BINS: int = 20

# Behavioural axis bounds.  Descriptors outside these bounds are clamped.
_LOWER: float = -1.0
_UPPER: float = 1.0


class MAPElitesArchive:
    """MAP-Elites grid archive for Quality-Diversity optimisation.

    The archive discretises a 2-D behaviour space into ``bins x bins``
    cells and keeps the best (highest-fitness) program in each cell.
    """

    def __init__(self, bins: int = _DEFAULT_BINS, lower: float = _LOWER, upper: float = _UPPER) -> None:
        self.bins: int = bins
        self._per_dim_lower: list[float] = [lower, lower]
        self._per_dim_upper: list[float] = [upper, upper]

        # Grid storage: flat list of length bins*bins.  Each cell is either
        # None (empty) or a dict with keys: fitness, program, trace, descriptor.
        self._grid: list[dict[str, Any] | None] = [None] * (bins * bins)

        # Alias expected by run_multi_cycle.py
        self.num_bins: int = bins

        # Running min/max of descriptors actually seen (for diagnostics).
        self._descriptor_min: list[float] = [math.inf, math.inf]
        self._descriptor_max: list[float] = [-math.inf, -math.inf]

        # Counter for rebin operations (diagnostics).
        self._rebins: int = 0

        # Occupied cell count (maintained incrementally).
        self._occupied: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp(self, val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    def _cell_index(self, descriptor: list[float]) -> int | None:
        """Map a continuous descriptor to a flat grid index.

        Returns ``None`` if the descriptor is None or has wrong length.
        """
        if descriptor is None or len(descriptor) != 2:
            return None
        lo, hi = self._per_dim_lower[0], self._per_dim_upper[0]
        span = hi - lo
        if span <= 0:
            return None
        indices: list[int] = []
        for i in range(2):
            val = self._clamp(descriptor[i], lo + 1e-12, hi - 1e-12)
            idx = int(((val - lo) / span) * self.bins)
            idx = max(0, min(self.bins - 1, idx))
            indices.append(idx)
        return indices[0] * self.bins + indices[1]

    def _update_bounds(self, descriptor: list[float]) -> None:
        """Update running descriptor min/max (for diagnostics only)."""
        for i in range(2):
            if descriptor[i] < self._descriptor_min[i]:
                self._descriptor_min[i] = descriptor[i]
            if descriptor[i] > self._descriptor_max[i]:
                self._descriptor_max[i] = descriptor[i]

    def _distance(self, a: list[float], b: list[float]) -> float:
        """Manhattan distance between two 2-D descriptors."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_solution(
        self,
        descriptor: list[float],
        fitness: float,
        program: list[int],
        trace: dict[str, Any] | None = None,
    ) -> bool:
        """Attempt to insert a solution into the archive.

        The solution replaces the incumbent of its cell if and only if
        *fitness* is strictly higher.  Returns ``True`` when a cell was
        added or replaced (i.e. the archive changed).
        """
        idx = self._cell_index(descriptor)
        if idx is None:
            return False

        self._update_bounds(descriptor)

        existing = self._grid[idx]
        if existing is None:
            # Empty cell — always insert.
            self._grid[idx] = {
                "fitness": fitness,
                "program": list(program),
                "trace": trace,
                "descriptor": list(descriptor),
            }
            self._occupied += 1
            return True

        # Cell occupied — replace only if strictly better.
        if fitness > existing["fitness"]:
            self._grid[idx] = {
                "fitness": fitness,
                "program": list(program),
                "trace": trace,
                "descriptor": list(descriptor),
            }
            self._rebins += 1
            return True

        return False

    def selection_bias(self, descriptor: list[float]) -> float:
        """Return a bonus score for a given behavioural descriptor.

        The bonus is inversely proportional to how close the descriptor's
        cell is to an already-occupied cell.  Empty neighbours receive a
        high bonus; crowded regions receive a low bonus.

        Returns a value in approximately [0.0, 1.0].
        """
        idx = self._cell_index(descriptor)
        if idx is None:
            return 0.5  # neutral fallback

        if self._grid[idx] is None:
            # The cell itself is empty — strong exploration bonus.
            return 1.0

        # Count occupied neighbours (8-connected) and compute average
        # distance to the nearest occupied cell within the same radius.
        row, col = divmod(idx, self.bins)
        occupied_neighbours = 0
        total_neighbours = 0
        min_dist = float("inf")

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.bins and 0 <= nc < self.bins:
                    total_neighbours += 1
                    n_idx = nr * self.bins + nc
                    if self._grid[n_idx] is not None:
                        occupied_neighbours += 1

        # Base bias from occupancy: empty neighbourhood -> 1.0, fully
        # occupied -> 0.3 (floor so even crowded cells get a small nudge).
        if total_neighbours > 0:
            occupancy_ratio = occupied_neighbours / total_neighbours
        else:
            occupancy_ratio = 0.0
        base_bias = 1.0 - 0.7 * occupancy_ratio

        # Also factor in grid coverage: when coverage is very low,
        # boost the bonus to encourage wider exploration.
        cov = self.coverage()
        if cov < 0.10:
            base_bias *= 1.0 + 0.5 * (0.10 - cov) / 0.10

        return max(0.1, min(1.0, base_bias))

    def coverage(self) -> float:
        """Return the fraction of grid cells that are occupied [0.0, 1.0]."""
        total = self.bins * self.bins
        if total == 0:
            return 0.0
        return self._occupied / total

    def grid_solutions(self) -> list[dict[str, Any]]:
        """Return a list of all occupied cells' solution dicts.

        Each dict contains: fitness, program, trace, descriptor.
        This is used for QD crossover and champion-reset operations.
        """
        return [cell for cell in self._grid if cell is not None]

    @property
    def size(self) -> int:
        """Number of occupied cells."""
        return self._occupied

    def clear(self) -> None:
        """Reset the archive to empty."""
        for i in range(len(self._grid)):
            self._grid[i] = None
        self._occupied = 0
        self._descriptor_min = [math.inf, math.inf]
        self._descriptor_max = [-math.inf, -math.inf]
        self._rebins = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        """Serialize the archive to a JSON-compatible dict.

        Captures grid cells (with fitness, program, trace, descriptor),
        bin count, and diagnostic bounds so the archive can be
        reconstructed exactly.
        """
        entries: list[dict[str, Any]] = []
        for idx, cell in enumerate(self._grid):
            if cell is not None:
                entries.append(
                    {
                        "idx": idx,
                        "fitness": cell["fitness"],
                        "program": cell["program"],
                        "trace": cell["trace"],
                        "descriptor": cell["descriptor"],
                    }
                )
        return {
            "bins": self.bins,
            "entries": entries,
            "descriptor_min": list(self._descriptor_min),
            "descriptor_max": list(self._descriptor_max),
            "rebins": self._rebins,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "MAPElitesArchive":
        """Reconstruct an archive from serialized data."""
        archive = cls(bins=data.get("bins", _DEFAULT_BINS))
        for entry in data.get("entries", []):
            idx = entry["idx"]
            archive._grid[idx] = {
                "fitness": entry["fitness"],
                "program": list(entry["program"]),
                "trace": entry["trace"],
                "descriptor": list(entry["descriptor"]),
            }
        archive._occupied = len(data.get("entries", []))
        archive._descriptor_min = data.get("descriptor_min", list(archive._descriptor_min))
        archive._descriptor_max = data.get("descriptor_max", list(archive._descriptor_max))
        archive._rebins = data.get("rebins", 0)
        return archive


# Public alias -- evolution_engine and tests may import by either name.
QDArchive = MAPElitesArchive
