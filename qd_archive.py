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

        # Addition counter for velocity tracking. Increments each time a
        # cell is filled or replaced in add_solution.
        self._addition_counter: int = 0
        # Per-cell counter: records the value of _addition_counter when
        # each cell was last updated.
        self._cell_addition_counts: list[int] = [0] * (bins * bins)
        # Per-cell freshness timestamps (generation when last updated)
        self._cell_last_updated: list[int] = [0] * (bins * bins)

        # Per-cell program history: maps cell index -> list of program hashes.
        # Tracks every program that has occupied each cell for diversity analysis.
        self._cell_program_history: dict[int, list[int]] = {}

        # Illumination diversity history: list of snapshots taken at each
        # record_illumination_snapshot() call.  Each entry is a dict with
        # "generation" (value of _addition_counter at recording time) and
        # "snapshot" (the full get_illumination_diversity() result).
        # Capped to _MAX_ILLUMINATION_HISTORY entries to bound memory.
        self._illumination_history: list[dict[str, Any]] = []
        self._MAX_ILLUMINATION_HISTORY: int = 200

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

        Uses a composite fitness-and-freshness replacement strategy:

        * **Coverage-adaptive staleness decay** — the freshness discount
          scales with archive coverage.  A nearly-full archive (coverage
          > 90 %) aggressively recycles stale entries to promote turnover;
          a sparse archive (< 30 %) is conservative so that early
          discoveries aren't prematurely evicted.

        * **Minimum improvement margin** — fresh incumbents require a
          meaningful improvement to be displaced; stale incumbents can
          be replaced by marginal gains.  This prevents "churn" in
          recently-populated cells while still allowing progress in
          frozen regions.

        * **Diversity-aware protection** — cells whose program history
          contains many unique programs are slightly easier to replace,
          since the archive has already captured their diversity
          contribution.

        Returns ``True`` when a cell was added or replaced (i.e. the
        archive changed).
        """
        idx = self._cell_index(descriptor)
        if idx is None:
            return False

        self._update_bounds(descriptor)

        # Track program hash for diversity measurement (uses hash index
        # pattern from novelty.py).
        prog_hash = hash(tuple(program))

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
            self._addition_counter += 1
            self._cell_addition_counts[idx] = self._addition_counter
            self._cell_last_updated[idx] = self._addition_counter
            self._cell_program_history.setdefault(idx, []).append(prog_hash)
            return True

        # ------------------------------------------------------------------
        # Cell occupied — composite fitness + freshness replacement strategy
        # ------------------------------------------------------------------

        # (1) Staleness of the incumbent cell.
        #     Exponential decay: ~63 % stale after 300 archive-wide updates,
        #     ~95 % stale after 1 000.  Gentler than the old 0.005 rate so
        #     that moderately-old cells still retain some protection.
        cell_age = max(0, self._addition_counter - self._cell_last_updated[idx])
        staleness_score = 1.0 - math.exp(-cell_age / 300.0)

        # (2) Coverage-adaptive staleness weight.
        #     Low coverage  → keep what we have (max 15 % discount).
        #     High coverage → prioritise freshness (max 45 % discount).
        coverage = self._occupied / max(1, self.bins * self.bins)
        staleness_weight = 0.15 + 0.30 * coverage  # range: [0.15, 0.45]

        # (3) Effective incumbent fitness: stale cells have a reduced
        #     effective fitness, making them easier to replace.
        freshness_factor = 1.0 - staleness_weight * staleness_score

        # (4) Diversity-aware adjustment: cells with many unique programs
        #     have already contributed to the archive's diversity, so they
        #     carry a slight replaceability bonus (up to 5 %).
        history = self._cell_program_history.get(idx, [])
        diversity_ratio = 0.0
        if len(history) > 1:
            unique_count = len(set(history))
            diversity_ratio = unique_count / len(history)
        diversity_discount = 0.05 * diversity_ratio  # max 5 %

        effective_incumbent = existing["fitness"] * freshness_factor * (1.0 - diversity_discount)

        # (5) Minimum improvement margin: fresh cells demand a meaningful
        #     improvement to displace the incumbent; stale cells can be
        #     replaced by marginal gains.  This prevents rapid churn in
        #     recently-populated cells.
        min_margin = 0.03 * (1.0 - staleness_score)  # 3 % for fresh → 0 % for stale

        # (6) Replacement decision.
        if fitness > effective_incumbent * (1.0 + min_margin):
            self._grid[idx] = {
                "fitness": fitness,
                "program": list(program),
                "trace": trace,
                "descriptor": list(descriptor),
            }
            self._rebins += 1
            self._addition_counter += 1
            self._cell_addition_counts[idx] = self._addition_counter
            self._cell_last_updated[idx] = self._addition_counter
            self._cell_program_history.setdefault(idx, []).append(prog_hash)
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

        # Freshness modulation: stale cells get a bonus, fresh cells get reduced
        try:
            if hasattr(self, '_cell_last_updated') and hasattr(self, '_addition_counter'):
                cell_age = max(0, self._addition_counter - self._cell_last_updated[idx]) if self._cell_last_updated[idx] > 0 else 100
                # Stale cells (>50 updates ago) get up to 20% boost
                # Fresh cells (last 10 updates) get up to 10% reduction
                if cell_age > 50:
                    base_bias *= 1.0 + min(0.2, cell_age / 500.0)
                elif cell_age < 10:
                    base_bias *= max(0.8, 1.0 - (10 - cell_age) * 0.02)
        except Exception:
            pass
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

    # ------------------------------------------------------------------
    # Diagnostics / introspection
    # ------------------------------------------------------------------

    def get_cell_stats(self) -> dict[str, Any]:
        """Return per-cell statistics for the grid.

        Returns a dict with:
          - total_cells: total grid cells
          - occupied: number of occupied cells
          - empty: number of empty cells
          - avg_fitness: average fitness across occupied cells (0.0 if none)
          - min_fitness / max_fitness: fitness range
          - avg_descriptor: average descriptor across occupied cells
        """
        occupied = [c for c in self._grid if c is not None]
        total = self.bins * self.bins
        if not occupied:
            return {
                "total_cells": total,
                "occupied": 0,
                "empty": total,
                "avg_fitness": 0.0,
                "min_fitness": 0.0,
                "max_fitness": 0.0,
                "avg_descriptor": [0.0, 0.0],
            }
        fitnesses = [c["fitness"] for c in occupied]
        descs = [c["descriptor"] for c in occupied]
        return {
            "total_cells": total,
            "occupied": len(occupied),
            "empty": total - len(occupied),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "min_fitness": min(fitnesses),
            "max_fitness": max(fitnesses),
            "avg_descriptor": [
                sum(d[0] for d in descs) / len(descs),
                sum(d[1] for d in descs) / len(descs),
            ],
        }

    def get_neighborhood_density(self, descriptor: list[float], radius: int = 1) -> float:
        """Return the fraction of cells within *radius* that are occupied.

        The density is computed in grid-cell space around the cell that
        *descriptor* maps to.  A value of 1.0 means every cell in the
        neighbourhood is filled; 0.0 means none are.
        """
        idx = self._cell_index(descriptor)
        if idx is None:
            return 0.0
        row, col = divmod(idx, self.bins)
        occupied = 0
        total = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.bins and 0 <= nc < self.bins:
                    total += 1
                    if self._grid[nr * self.bins + nc] is not None:
                        occupied += 1
        return occupied / total if total > 0 else 0.0

    def get_freshest_cell(self) -> dict[str, Any] | None:
        """Return the cell with the most-recently added program.

        Each cell entry does not carry a timestamp, so we return the
        occupied cell with the highest fitness as a proxy for the
        "freshest" (most recently updated) cell.  Returns None if the
        archive is empty.
        """
        best: dict[str, Any] | None = None
        for cell in self._grid:
            if cell is not None:
                if best is None or cell["fitness"] > best["fitness"]:
                    best = cell
        return best

    def get_fitness_distribution(self) -> dict[str, Any]:
        """Return a binned fitness distribution across occupied cells.

        Returns:
          - bins: list of {min, max, count} dicts (10 equal-width bins)
          - mean, median, stddev: summary statistics
        """
        fitnesses = [c["fitness"] for c in self._grid if c is not None]
        if not fitnesses:
            return {
                "bins": [],
                "mean": 0.0,
                "median": 0.0,
                "stddev": 0.0,
            }
        f_min = min(fitnesses)
        f_max = max(fitnesses)
        n_bins = 10
        if f_max - f_min < 1e-12:
            # All identical — single bin
            bins = [{"min": f_min, "max": f_max, "count": len(fitnesses)}]
        else:
            width = (f_max - f_min) / n_bins
            bins = [{"min": f_min + i * width, "max": f_min + (i + 1) * width, "count": 0} for i in range(n_bins)]
            for f in fitnesses:
                b_idx = int((f - f_min) / width)
                b_idx = min(b_idx, n_bins - 1)
                bins[b_idx]["count"] += 1
        mean = sum(fitnesses) / len(fitnesses)
        sorted_f = sorted(fitnesses)
        mid = len(sorted_f) // 2
        median = sorted_f[mid] if len(sorted_f) % 2 == 1 else (sorted_f[mid - 1] + sorted_f[mid]) / 2
        variance = sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)
        stddev = variance**0.5
        return {
            "bins": bins,
            "mean": mean,
            "median": median,
            "stddev": stddev,
        }

    def get_archive_summary(self) -> dict:
        """Return archive summary for diagnostics."""
        stats = self.get_cell_stats()
        fd = self.get_fitness_distribution()
        return {
            "cells": stats,
            "fitness": fd,
            "rebins": self._rebins,
        }

    def get_cell_freshness(self, current_generation: int = 0) -> dict[str, Any]:
        """Return freshness scores for all cells.

        Freshness = 1.0 for recently updated cells, decaying to 0.0 for old cells.
        Uses the same exponential decay as novelty decayed_novelty.
        """
        freshness_grid = [0.0] * (self.bins * self.bins)
        occupied = 0
        total_freshness = 0.0
        for idx in range(len(self._grid)):
            if self._grid[idx] is not None:
                occupied += 1
                last_update = self._cell_last_updated[idx]
                age = max(0, current_generation - last_update) if current_generation > 0 else 0
                freshness = math.exp(-0.01 * age)  # gentle decay
                freshness_grid[idx] = freshness
                total_freshness += freshness
        avg_freshness = total_freshness / max(occupied, 1)
        return {
            'freshness_grid': freshness_grid,
            'avg_freshness': avg_freshness,
            'occupied_cells': occupied,
        }

    def get_cell_program_diversity(self, cell_idx: int) -> dict[str, Any]:
        """Return the program diversity score for a specific cell.

        Uses the program hash index (``hash(tuple(program))``, matching
        the approach in ``novelty.py``) to identify unique programs that
        have occupied the cell over its history.  Diversity is computed as
        Shannon entropy of the unique-program distribution, normalised to
        [0.0, 1.0].

        A cell with only one distinct program has diversity 0.0; a cell
        where every program has been unique has diversity 1.0.

        Parameters
        ----------
        cell_idx:
            Flat grid index (``row * bins + col``).

        Returns
        -------
        dict
            ``diversity``:          normalised Shannon entropy in [0.0, 1.0]
            ``unique_programs``:    number of distinct program hashes seen
            ``total_programs``:     total number of programs added to cell
            ``unique_ratio``:       ``unique_programs / total_programs``
        """
        total = self.bins * self.bins
        if cell_idx < 0 or cell_idx >= total:
            return {
                "diversity": 0.0,
                "unique_programs": 0,
                "total_programs": 0,
                "unique_ratio": 0.0,
            }

        history: list[int] = self._cell_program_history.get(cell_idx, [])
        if not history:
            return {
                "diversity": 0.0,
                "unique_programs": 0,
                "total_programs": 0,
                "unique_ratio": 0.0,
            }

        total_programs = len(history)

        # Count occurrences of each unique program hash.
        freq: dict[int, int] = {}
        for h in history:
            freq[h] = freq.get(h, 0) + 1

        unique_programs = len(freq)
        unique_ratio = unique_programs / total_programs

        # Shannon entropy over the frequency distribution, normalised
        # by log2(unique_count) so the result is in [0.0, 1.0].
        if unique_programs <= 1:
            entropy = 0.0
        else:
            entropy = 0.0
            for count in freq.values():
                p = count / total_programs
                if p > 0:
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(unique_programs)
            entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "diversity": entropy,
            "unique_programs": unique_programs,
            "total_programs": total_programs,
            "unique_ratio": unique_ratio,
        }

    def get_improvement_velocity(self) -> dict[str, Any]:
        """Track how fast fitness improves per cell using a heuristic.

        Since cells don't have timestamps, uses a counter-based heuristic:
        velocity = (fitness - avg_fitness) / max(addition_count, 1)
        where addition_count is the archive-wide counter at the time of
        the cell's last update.

        Returns a dict with:
          - per_cell_velocity: list of {row, col, velocity} for each occupied cell
          - avg_velocity: mean velocity across occupied cells
          - max_velocity: maximum velocity
          - velocity_stddev: standard deviation of velocities
        """
        occupied_cells: list[tuple[int, int, int, dict[str, Any]]] = []
        fitnesses: list[float] = []
        for idx, cell in enumerate(self._grid):
            if cell is not None:
                row, col = divmod(idx, self.bins)
                occupied_cells.append((idx, row, col, cell))
                fitnesses.append(cell["fitness"])

        if not occupied_cells:
            return {
                "per_cell_velocity": [],
                "avg_velocity": 0.0,
                "max_velocity": 0.0,
                "velocity_stddev": 0.0,
            }

        avg_fitness = sum(fitnesses) / len(fitnesses)

        per_cell_velocity: list[dict[str, Any]] = []
        velocities: list[float] = []
        for idx, row, col, cell in occupied_cells:
            addition_count = self._cell_addition_counts[idx]
            velocity = (cell["fitness"] - avg_fitness) / max(addition_count, 1)
            per_cell_velocity.append({"row": row, "col": col, "velocity": velocity})
            velocities.append(velocity)

        avg_velocity = sum(velocities) / len(velocities)
        max_velocity = max(velocities)
        variance = sum((v - avg_velocity) ** 2 for v in velocities) / len(velocities)
        velocity_stddev = variance**0.5

        return {
            "per_cell_velocity": per_cell_velocity,
            "avg_velocity": avg_velocity,
            "max_velocity": max_velocity,
            "velocity_stddev": velocity_stddev,
        }

    def get_illumination_diversity(self) -> dict[str, Any]:
        """Measure behavioral space coverage quality beyond simple occupancy.

        Computes:
          - coverage_ratio: fraction of occupied cells (occupied / total)
          - centroid_distance: avg Euclidean distance of occupied cells from
            the grid centroid (measures spread)
          - uniformity: 1 - (stddev of neighborhood densities /
            mean neighborhood density) (measures evenness)
          - overall_diversity_score: coverage_ratio * uniformity
          - quality_weighted_uniformity: fitness-weighted neighborhood
            density uniformity — each bin's contribution is weighted by
            the fitness of its occupant, so high-quality regions count
            more toward evenness (measures quality-aware spread)
          - quality_weighted_diversity: coverage_ratio *
            quality_weighted_uniformity — the overall diversity score
            that accounts for both spatial coverage and the quality of
            programs filling that space
          - quality_weighted_shannon_entropy: Shannon entropy over a
            fitness-weighted probability distribution across occupied
            bins.  Each bin's weight equals the fitness of its occupant,
            so high-quality bins contribute more to the entropy.  Normalised
            to [0.0, 1.0] where 1.0 = perfectly even quality-weighted
            spread across all occupied bins.
          - quality_weighted_illumination_diversity: coverage_ratio *
            quality_weighted_shannon_entropy — combines spatial coverage
            with the quality-aware information-theoretic spread metric.

        Returns a dict with coverage_ratio, centroid_distance, uniformity,
        overall_diversity_score, quality_weighted_uniformity,
        quality_weighted_diversity, quality_weighted_shannon_entropy, and
        quality_weighted_illumination_diversity.
        """
        total = self.bins * self.bins
        if total == 0:
            return {
                "coverage_ratio": 0.0,
                "centroid_distance": 0.0,
                "uniformity": 0.0,
                "overall_diversity_score": 0.0,
                "quality_weighted_uniformity": 0.0,
                "quality_weighted_diversity": 0.0,
                "quality_weighted_shannon_entropy": 0.0,
                "quality_weighted_illumination_diversity": 0.0,
            }

        occupied_count = self._occupied
        coverage_ratio = occupied_count / total

        # (b) Centroid distance: average Euclidean distance from grid center.
        centroid_distance = 0.0
        if occupied_count > 0:
            centroid_row = (self.bins - 1) / 2.0
            centroid_col = (self.bins - 1) / 2.0
            total_dist = 0.0
            for idx, cell in enumerate(self._grid):
                if cell is not None:
                    row, col = divmod(idx, self.bins)
                    dist = math.sqrt((row - centroid_row) ** 2 + (col - centroid_col) ** 2)
                    total_dist += dist
            centroid_distance = total_dist / occupied_count

        # (c) Uniformity: coefficient of variation of neighborhood densities.
        #     Also compute quality-weighted density for each cell — the local
        #     density is scaled by the fitness of the cell's occupant so that
        #     high-quality bins contribute more strongly to uniformity.
        densities: list[float] = []
        quality_weighted_densities: list[float] = []

        # Precompute per-cell fitness lookup for quality weighting.
        fitness_lookup: list[float] = [0.0] * total
        for idx, cell in enumerate(self._grid):
            if cell is not None:
                fitness_lookup[idx] = cell["fitness"]

        for idx, cell in enumerate(self._grid):
            if cell is not None:
                row, col = divmod(idx, self.bins)
                occ = 0
                tot = 0
                weighted_sum = 0.0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.bins and 0 <= nc < self.bins:
                            n_idx = nr * self.bins + nc
                            tot += 1
                            if self._grid[n_idx] is not None:
                                occ += 1
                                weighted_sum += fitness_lookup[n_idx]
                density = occ / tot if tot > 0 else 0.0
                densities.append(density)
                # Quality-weighted density: weight each occupied neighbour's
                # contribution by its fitness.  We normalise by the total
                # fitness of all neighbours (occupied or not) so the result
                # is comparable across cells with different fitness scales.
                if occ > 0:
                    qw_density = weighted_sum / tot
                else:
                    qw_density = 0.0
                quality_weighted_densities.append(qw_density)

        if densities:
            mean_density = sum(densities) / len(densities)
            if mean_density > 0:
                density_variance = sum((d - mean_density) ** 2 for d in densities) / len(densities)
                density_stddev = density_variance**0.5
                uniformity = 1.0 - (density_stddev / mean_density)
            else:
                uniformity = 0.0
        else:
            uniformity = 0.0

        uniformity = max(0.0, min(1.0, uniformity))
        overall_diversity_score = coverage_ratio * uniformity

        # (d) Quality-weighted uniformity: coefficient of variation of the
        #     fitness-weighted neighborhood densities.  Bins occupied by
        #     high-fitness programs produce higher weighted-density values,
        #     shifting the uniformity measurement to favour quality-aware
        #     spatial distribution.  The coefficient of variation is
        #     normalised similarly to the unweighted uniformity so that a
        #     perfectly even quality-weighted distribution yields 1.0.
        quality_weighted_uniformity = 0.0
        if quality_weighted_densities:
            mean_qw = sum(quality_weighted_densities) / len(quality_weighted_densities)
            if mean_qw > 0:
                qw_variance = sum(
                    (d - mean_qw) ** 2 for d in quality_weighted_densities
                ) / len(quality_weighted_densities)
                qw_stddev = qw_variance**0.5
                quality_weighted_uniformity = 1.0 - (qw_stddev / mean_qw)
            else:
                quality_weighted_uniformity = 0.0
        quality_weighted_uniformity = max(0.0, min(1.0, quality_weighted_uniformity))

        # (e) Quality-weighted overall diversity: combines coverage with the
        #     quality-aware uniformity metric so that high-fitness regions
        #     of behaviour space contribute more to the overall score.
        quality_weighted_diversity = coverage_ratio * quality_weighted_uniformity

        # (f) Quality-weighted Shannon entropy: weight each occupied bin's
        #     contribution to the probability distribution by its fitness,
        #     so high-quality bins dominate the entropy calculation.
        #     P(cell_i) = fitness_i / sum(fitness_j) for all occupied j.
        #     Then H = -sum(P * log2(P)), normalised by log2(N_occupied)
        #     so the result is in [0.0, 1.0].
        quality_weighted_shannon_entropy = 0.0
        occupied_fitnesses: list[float] = []
        for idx, cell in enumerate(self._grid):
            if cell is not None:
                occupied_fitnesses.append(cell["fitness"])

        if occupied_fitnesses:
            total_fitness = sum(occupied_fitnesses)
            if total_fitness > 0:
                qw_probs = [f / total_fitness for f in occupied_fitnesses]
                qw_entropy = 0.0
                for p in qw_probs:
                    if p > 0:
                        qw_entropy -= p * math.log2(p)
                # Normalise by maximum possible entropy (uniform distribution)
                n_occupied = len(occupied_fitnesses)
                max_qw_entropy = math.log2(n_occupied) if n_occupied > 1 else 0.0
                quality_weighted_shannon_entropy = (
                    qw_entropy / max_qw_entropy if max_qw_entropy > 0 else 0.0
                )

        quality_weighted_illumination_diversity = (
            coverage_ratio * quality_weighted_shannon_entropy
        )

        return {
            "coverage_ratio": coverage_ratio,
            "centroid_distance": centroid_distance,
            "uniformity": uniformity,
            "overall_diversity_score": overall_diversity_score,
            "quality_weighted_uniformity": quality_weighted_uniformity,
            "quality_weighted_diversity": quality_weighted_diversity,
            "quality_weighted_shannon_entropy": quality_weighted_shannon_entropy,
            "quality_weighted_illumination_diversity": quality_weighted_illumination_diversity,
        }

    def record_illumination_snapshot(self) -> dict[str, Any]:
        """Record the current illumination diversity as a history snapshot.

        Call this at the end of each generation or evaluation epoch to
        build up a longitudinal record of archive quality.  Each call
        appends a snapshot containing:

          - generation: the archive-wide ``_addition_counter`` value at
            the time of recording (serves as a logical timestamp).
          - snapshot: the full dict returned by ``get_illumination_diversity()``
            at the moment of recording.

        The history is capped at ``_MAX_ILLUMINATION_HISTORY`` entries
        (oldest entries are dropped) to bound memory usage.

        Returns the snapshot that was just recorded.
        """
        generation = self._addition_counter
        snapshot = self.get_illumination_diversity()
        entry: dict[str, Any] = {
            "generation": generation,
            "snapshot": snapshot,
        }
        self._illumination_history.append(entry)
        # Enforce history cap — drop oldest entries first.
        if len(self._illumination_history) > self._MAX_ILLUMINATION_HISTORY:
            self._illumination_history = self._illumination_history[
                -self._MAX_ILLUMINATION_HISTORY:
            ]
        return entry

    def get_illumination_history(self) -> list[dict[str, Any]]:
        """Return the longitudinal history of illumination diversity snapshots.

        Each element is a dict with:

          - ``generation``: the ``_addition_counter`` value when the
            snapshot was recorded (logical timestamp).
          - ``snapshot``: the full illumination diversity dict produced
            by ``get_illumination_diversity()`` at that time, containing
            keys such as ``coverage_ratio``, ``centroid_distance``,
            ``uniformity``, ``overall_diversity_score``,
            ``quality_weighted_uniformity``, ``quality_weighted_diversity``,
            ``quality_weighted_shannon_entropy``, and
            ``quality_weighted_illumination_diversity``.

        The list is ordered chronologically (oldest first).  At most
        ``_MAX_ILLUMINATION_HISTORY`` entries are retained.

        Use this to analyse longitudinal trends:

        - **Divergence**: compare early vs late snapshots to see if
          the archive is becoming more or less diverse over time.
        - **Plateau detection**: spot generations where diversity
          metrics stop improving.
        - **Regression alerts**: detect sudden drops in coverage or
          quality-weighted diversity after replacement events.
        """
        return list(self._illumination_history)

    def clear(self) -> None:
        """Reset the archive to empty."""
        for i in range(len(self._grid)):
            self._grid[i] = None
        self._occupied = 0
        self._descriptor_min = [math.inf, math.inf]
        self._descriptor_max = [-math.inf, -math.inf]
        self._rebins = 0
        self._addition_counter = 0
        self._cell_addition_counts = [0] * (self.bins * self.bins)
        self._cell_program_history.clear()
        self._illumination_history.clear()

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
            "addition_counter": self._addition_counter,
            "cell_addition_counts": list(self._cell_addition_counts),
            "cell_last_updated": self._cell_last_updated,
            "cell_program_history": {str(k): v for k, v in self._cell_program_history.items()},
            "illumination_history": self._illumination_history,
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
        archive._addition_counter = data.get("addition_counter", 0)
        saved_counts = data.get("cell_addition_counts")
        if saved_counts is not None and len(saved_counts) == archive.bins * archive.bins:
            archive._cell_addition_counts = list(saved_counts)
        else:
            archive._cell_addition_counts = [0] * (archive.bins * archive.bins)
        if 'cell_last_updated' in data:
            archive._cell_last_updated = data['cell_last_updated']
        else:
            archive._cell_last_updated = [0] * (archive.bins * archive.bins)
        # Restore per-cell program history (stored as str keys for JSON compat).
        saved_history = data.get("cell_program_history", {})
        archive._cell_program_history = {int(k): list(v) for k, v in saved_history.items()}
        archive._illumination_history = list(data.get("illumination_history", []))
        return archive


# Public alias -- evolution_engine and tests may import by either name.
QDArchive = MAPElitesArchive
