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
    .get_diversity_score() -> float
    .recent_novelty_scores(n) -> list[float]
    .trim_archive(max_entries) -> int
    .get_archive_summary() -> dict
    .batch_lookup(programs, current_generation, half_life) -> dict[int, float]
    .get_novelty_stats() -> dict[str, float]
    .get_novelty_trajectory(program_key) -> list[dict]
    .get_novelty_distribution_stats() -> dict[str, Any]
    .record_distribution_snapshot(generation) -> dict[str, Any]
    .get_novelty_distribution_history(max_snapshots) -> list[dict[str, Any]]
    .track_distribution_evolution(window) -> dict[str, Any]
    .get_novelty_impact_correlation() -> dict[str, Any]
    .size -> int
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import KDTree
from scipy.stats import pearsonr, spearmanr

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

        # Program hash index for O(1) lookup in decayed_novelty
        self._prog_hash_index: dict[int, list[int]] = {}  # hash -> list of entry indices

        # Longitudinal distribution history: list of per-generation snapshots
        self._distribution_history: list[dict[str, Any]] = []

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

    def _rebuild_prog_hash_index(self) -> None:
        """Rebuild the program hash index from current entries."""
        self._prog_hash_index.clear()
        for i, entry in enumerate(self._entries):
            meta = entry[2]
            if isinstance(meta, dict) and meta.get("program") is not None:
                _ph = hash(tuple(meta["program"]))
                self._prog_hash_index.setdefault(_ph, []).append(i)

    def _relative_novelty_multiplier(self, program_novelty: float) -> float:
        """Compute a decay multiplier based on a program's novelty relative to the archive.

        Programs that are more novel than the archive average get a multiplier > 1
        (slower decay, more valuable). Programs less novel than average get a
        multiplier < 1 (faster decay, less distinctive).

        Returns a multiplier in [0.3, 2.5] range:
          - 0.3: very similar to archive entries (fast decay)
          - 1.0: average novelty (baseline)
          - 2.5: highly novel relative to archive (slow decay)
        """
        if self.size == 0 or self._tree is None:
            return 1.0

        # Compute average k-NN distance across the archive to establish baseline.
        k = min(self.k_neighbors, self.size)
        total_dist = 0.0
        count = 0
        for entry in self._entries:
            flat = entry[0]
            try:
                dists, _ = self._tree.query(np.array(flat), k=k)
                if k == 1:
                    total_dist += float(dists)
                else:
                    total_dist += float(np.mean(dists))
                count += 1
            except Exception:
                continue

        if count == 0:
            return 1.0

        archive_avg_novelty = total_dist / count
        if archive_avg_novelty <= 0:
            return 1.0

        # Relative novelty ratio: how this program compares to archive average.
        # > 1 means more novel than average; < 1 means less novel.
        relative_novelty = program_novelty / archive_avg_novelty

        # Map to multiplier with bounded scaling:
        #   Floor at 0.3 prevents premature elimination of borderline programs.
        #   Ceiling at 2.5 prevents immortal programs even at extreme novelty.
        return max(0.3, min(2.5, relative_novelty))

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
            # Update program hash index
            if isinstance(metadata, dict) and metadata.get("program") is not None:
                _ph = hash(tuple(metadata["program"]))
                self._prog_hash_index.setdefault(_ph, []).append(len(self._entries) - 1)
            return True, score

        threshold = self.adaptive_threshold()
        if check_threshold and score < threshold:
            return False, score

        self._entries.append((flat, descriptor, metadata))
        self.size = len(self._entries)
        self._tree_dirty = True
        # Update program hash index
        if isinstance(metadata, dict) and metadata.get("program") is not None:
            _ph = hash(tuple(metadata["program"]))
            self._prog_hash_index.setdefault(_ph, []).append(len(self._entries) - 1)
        return True, score

    def clear(self) -> None:
        """Reset the archive."""
        self._entries.clear()
        self.size = 0
        self._tree_dirty = True
        self._prog_hash_index.clear()
        self._distribution_history.clear()

    def recent_descriptors(self, n: int = 10) -> list[dict[str, float]]:
        """Return the last *n* descriptor dicts added to the archive."""
        return [entry[1] for entry in self._entries[-n:]]

    def recent_novelty_scores(self, n: int = 10) -> list[float]:
        """Return the novelty scores of the last *n* entries added to the archive."""
        if self.size == 0:
            return []
        if self._tree_dirty:
            self._rebuild_tree()
        if self._tree is None:
            return [0.0] * min(n, self.size)
        n = min(n, self.size)
        scores = []
        for entry in self._entries[-n:]:
            flat = entry[0]
            k = min(self.k_neighbors, self.size)
            dists, _ = self._tree.query(np.array(flat), k=k)
            if k == 1:
                scores.append(float(dists))
            else:
                scores.append(float(np.mean(dists)))
        return scores

    def trim_archive(self, max_entries: int = 500) -> int:
        """FIFO-evict oldest entries when the archive exceeds *max_entries*.

        Returns the number of entries removed.
        """
        if self.size <= max_entries:
            return 0
        excess = self.size - max_entries
        self._entries = self._entries[excess:]
        self.size = len(self._entries)
        self._tree_dirty = True
        self._rebuild_prog_hash_index()
        return excess

    def get_diversity_score(self) -> float:
        """Shannon entropy of the distribution of descriptor values, normalised to [0, 1].

        Quantifies how spread-out the archive is in descriptor space.
        Returns 0.0 for an empty or single-entry archive.
        """
        if self.size <= 1:
            return 0.0
        # Build a coarse histogram across all descriptor dimensions.
        # Each descriptor value is binned into 10 buckets over [0, 1].
        n_bins = 10
        bin_counts: dict[int, int] = {}
        total = 0
        for entry in self._entries:
            for v in entry[1].values():
                b = int(min(v, 0.9999) * n_bins)
                bin_counts[b] = bin_counts.get(b, 0) + 1
                total += 1
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in bin_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        # Normalise by the maximum possible entropy (uniform distribution)
        max_entropy = math.log2(n_bins) if n_bins > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_archive_summary(self) -> dict:
        """Return archive summary for diagnostics."""
        recent = self.recent_novelty_scores(10)
        return {
            "size": self.size,
            "k_neighbors": self.k_neighbors,
            "novelty_threshold": self.novelty_threshold,
            "diversity_score": self.get_diversity_score(),
            "avg_novelty": sum(recent) / len(recent) if recent else 0.0,
        }

    def batch_lookup(
        self,
        programs: list[list[int]],
        current_generation: int | None = None,
        half_life: int = 500,
    ) -> dict[int, float]:
        """Look up novelty scores for multiple programs in one call.

        Uses the program hash index for O(1) per lookup instead of
        scanning the full archive.  For programs found in the archive,
        returns their novelty score (optionally with exponential time-decay
        if *current_generation* is provided).  Programs not found get 0.0.

        Parameters
        ----------
        programs:
            List of programs (each a ``list[int]`` of instruction words).
        current_generation:
            If given, apply exponential time-decay to scores using
            ``half_life``.  If ``None``, return raw novelty scores.
        half_life:
            Decay half-life in generations (default 500).

        Returns
        -------
        dict[int, float]
            Mapping from ``id(program)`` to novelty score for every
            program in the input list.
        """
        results: dict[int, float] = {}
        for program in programs:
            prog_id = id(program)
            _prog_hash = hash(tuple(program))
            indices = self._prog_hash_index.get(_prog_hash, [])
            found = False
            for idx in indices:
                if idx < len(self._entries):
                    entry = self._entries[idx]
                    meta = entry[2]
                    stored_prog = meta.get("program") if isinstance(meta, dict) else None
                    if stored_prog is not None and stored_prog == program:
                        flat_descriptor = entry[0]
                        base_novelty = self._k_nn_distance(flat_descriptor)
                        if current_generation is not None:
                            program_generation = meta.get("generation") if isinstance(meta, dict) else None
                            if program_generation is not None:
                                age = current_generation - int(program_generation)
                                decay = math.exp(-0.693 * age / half_life)
                                base_novelty *= decay
                        results[prog_id] = base_novelty
                        found = True
                        break
            if not found:
                results[prog_id] = 0.0
        return results

    def get_novelty_stats(self) -> dict[str, float]:
        """Return descriptive statistics of novelty scores across the archive.

        Useful for diagnostics: detects whether novelty pressure is too
        high (mean/median near 0) or too low (scores diverging / no
        selection pressure).  Returns an empty dict if the archive is
        empty.

        Returns
        -------
        dict[str, float]
            Keys: ``mean``, ``median``, ``std``, ``min``, ``max``,
            ``q25`` (25th percentile), ``q75`` (75th percentile).
        """
        if self.size == 0:
            return {}
        if self._tree_dirty:
            self._rebuild_tree()
        if self._tree is None:
            return {}

        k = min(self.k_neighbors, self.size)
        scores: list[float] = []
        for entry in self._entries:
            flat = entry[0]
            dists, _ = self._tree.query(np.array(flat), k=k)
            if k == 1:
                scores.append(float(dists))
            else:
                scores.append(float(np.mean(dists)))

        arr = np.array(scores)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }

    # ------------------------------------------------------------------
    # Longitudinal distribution history
    # ------------------------------------------------------------------

    def record_distribution_snapshot(self, generation: int | None = None) -> dict[str, Any]:
        """Capture a snapshot of the current distribution and append to history.

        Call this once per generation (typically at the end of a generation
        cycle) to build up a longitudinal record of how novelty pressure
        evolves over time.

        Parameters
        ----------
        generation:
            The current generation number.  If ``None``, uses the number
            of existing snapshots as a fallback counter.

        Returns
        -------
        dict[str, Any]
            The snapshot that was recorded (same shape as
            :meth:`get_novelty_distribution_stats` output plus
            ``generation`` and ``archive_size``).
        """
        stats = self.get_novelty_distribution_stats()
        if not stats:
            # Degenerate archive — still record a minimal snapshot so the
            # timeline isn't interrupted.
            snap: dict[str, Any] = {
                "generation": generation if generation is not None else len(self._distribution_history),
                "archive_size": self.size,
                "populated": False,
            }
        else:
            snap = {
                "generation": generation if generation is not None else len(self._distribution_history),
                "archive_size": self.size,
                "populated": True,
                **stats,
            }

        # Deduplicate: if the last snapshot has the same generation, replace it
        # (allows callers to re-snapshot a generation in-place).
        if self._distribution_history and self._distribution_history[-1]["generation"] == snap["generation"]:
            self._distribution_history[-1] = snap
        else:
            self._distribution_history.append(snap)

        return snap

    def get_novelty_distribution_history(
        self,
        max_snapshots: int = 50,
    ) -> list[dict[str, Any]]:
        """Return longitudinal distribution snapshots over recent generations.

        Each snapshot is a dict containing:
          - ``generation``: the generation number
          - ``archive_size``: how many entries were in the archive at that point
          - ``populated``: ``True`` if full stats were available, ``False`` otherwise
          - ``mean``, ``median``, ``std``, ``min``, ``max``: basic moments
          - ``q10``, ``q25``, ``q75``, ``q90``: percentiles
          - ``iqr``, ``skewness``, ``cv``, ``novelty_pressure``: advanced diagnostics

        Returns an empty list if no snapshots have been recorded yet.

        Parameters
        ----------
        max_snapshots:
            Maximum number of recent snapshots to return (most recent first
            in the output list).  Set to ``0`` or negative for no limit.
        """
        if not self._distribution_history:
            return []

        populated = [s for s in self._distribution_history if s.get("populated", False)]
        if not populated:
            return []

        if max_snapshots > 0:
            return populated[-max_snapshots:]
        return list(populated)

    # ------------------------------------------------------------------
    # Distribution evolution tracking
    # ------------------------------------------------------------------

    # Metrics contributing to composite score: (key, weight, direction).
    # direction=+1 means upward slope is "improving"; -1 means upward
    # slope is "declining".
    _EVOLUTION_METRICS: list[tuple[str, float, int]] = [
        ("mean",     0.30, +1),
        ("median",   0.20, +1),
        ("std",      0.15, +1),
        ("cv",       0.15, -1),
        ("iqr",      0.10, +1),
        ("skewness", 0.10, -1),
    ]

    @staticmethod
    def _metric_slope(
        values: list[float], gens: np.ndarray,
    ) -> tuple[float, str]:
        """Normalised linear slope of *values* over *gens*.

        Returns ``(norm_slope, direction)`` where direction is one of
        ``"increasing"``, ``"decreasing"``, or ``"flat"``.
        """
        arr = np.array(values)
        valid = ~np.isnan(arr)
        if valid.sum() < 2:
            return 0.0, "flat"
        g, y = gens[valid], arr[valid]
        if g[-1] == g[0]:
            return 0.0, "flat"
        g_norm = (g - g[0]) / (g[-1] - g[0])
        raw_slope = float(np.polyfit(g_norm, y, 1)[0])
        y_range = float(np.max(y) - np.min(y))
        ns = raw_slope / y_range if y_range > 0 else 0.0
        d = "increasing" if ns > 0.05 else ("decreasing" if ns < -0.05 else "flat")
        return ns, d

    def track_distribution_evolution(
        self,
        window: int = 0,
    ) -> dict[str, Any]:
        """Analyze how the novelty distribution evolves and classify the trend.

        Computes linear-trend slopes for key distribution metrics across
        recorded snapshots and produces a composite score classifying
        evolution as **improving** (>+0.15), **declining** (<-0.15),
        or **stable**.

        Parameters
        ----------
        window:
            If > 0, only the last *window* snapshots are used.
            ``0`` (default) uses all available data.

        Returns
        -------
        dict[str, Any]
            Keys: ``trend``, ``composite_score``, ``confidence``,
            ``n_snapshots``, ``n_generations``, ``metrics``,
            ``interpretation``, ``recommendation``.

        Returns an empty dict when fewer than 2 populated snapshots exist.
        """
        populated = [s for s in self._distribution_history if s.get("populated")]
        if window > 0:
            populated = populated[-window:]
        if len(populated) < 2:
            return {}

        gens = np.array([s["generation"] for s in populated], dtype=float)
        gen_span = int(gens[-1] - gens[0])

        # Per-metric slope analysis
        metric_trends: dict[str, dict[str, Any]] = {}
        weighted_slopes: list[float] = []
        total_weight = 0.0
        for key, weight, direction in self._EVOLUTION_METRICS:
            vals = [float(s.get(key, float("nan"))) for s in populated]
            ns, d = self._metric_slope(vals, gens)
            adj = ns * direction
            metric_trends[key] = {
                "slope": round(ns, 6), "direction": d,
                "adjusted_slope": round(adj, 6),
            }
            weighted_slopes.append(weight * adj)
            total_weight += weight

        composite = sum(weighted_slopes) / total_weight if total_weight else 0.0
        composite = max(-1.0, min(1.0, composite))
        trend = "improving" if composite > 0.15 else ("declining" if composite < -0.15 else "stable")

        n = len(populated)
        confidence = "high" if n >= 10 and gen_span >= 10 else ("medium" if n >= 5 else "low")

        improving = [k for k, v in metric_trends.items() if v["adjusted_slope"] > 0.05]
        declining = [k for k, v in metric_trends.items() if v["adjusted_slope"] < -0.05]

        interp = (
            f"Novelty distribution trend is {trend} (composite score: "
            f"{composite:+.3f}) across {n} snapshots spanning {gen_span} generations."
        )
        if improving:
            interp += f" Improving metrics: {', '.join(improving)}."
        if declining:
            interp += f" Declining metrics: {', '.join(declining)}."

        recs = {
            "improving": (
                "Novelty pressure is effective -- continue current strategy. "
                "Monitor for diminishing returns as the archive matures."),
            "declining": (
                "Novelty scores are falling. Consider increasing mutation "
                "rate, widening the behavioural descriptor space, or "
                "adjusting the novelty threshold to maintain exploration."),
            "stable": (
                "Novelty pressure is stable. No immediate action needed. "
                "If diversity stagnates, consider periodic resets or "
                "adaptive threshold adjustments."),
        }

        return {
            "trend": trend, "composite_score": round(composite, 4),
            "confidence": confidence, "n_snapshots": n,
            "n_generations": gen_span, "metrics": metric_trends,
            "interpretation": interp, "recommendation": recs[trend],
        }

    # ------------------------------------------------------------------
    # Capacity management & decayed novelty
    # ------------------------------------------------------------------

    def manage_capacity(self, max_size: int | None = None) -> int:
        """Trim the archive to *max_size* by removing lowest-novelty entries first.

        *max_size* defaults to ``self.max_archive_size`` if that attribute
        exists, otherwise ``1000``.

        Returns the number of entries removed.
        """
        effective = max_size if max_size is not None else getattr(self, "max_archive_size", 1000)
        if self.size <= effective:
            return 0

        if self._tree_dirty:
            self._rebuild_tree()

        # Compute novelty score for every entry against the *current* archive.
        scored: list[tuple[float, int]] = []
        k = min(self.k_neighbors, self.size)
        for idx, entry in enumerate(self._entries):
            flat = entry[0]
            if self._tree is not None and self.size > 0:
                dists, _ = self._tree.query(np.array(flat), k=k)
                score = float(np.mean(dists)) if k > 1 else float(dists)
            else:
                score = 0.0
            scored.append((score, idx))

        # Sort ascending by novelty – lowest-novelty entries go first.
        scored.sort(key=lambda t: t[0])

        n_remove = self.size - effective
        remove_indices = {idx for _, idx in scored[:n_remove]}
        self._entries = [e for i, e in enumerate(self._entries) if i not in remove_indices]
        self.size = len(self._entries)
        self._tree_dirty = True
        self._rebuild_prog_hash_index()
        return n_remove

    def decayed_novelty(
        self,
        program: list[int],
        current_generation: int,
        half_life: int = 500,
    ) -> float:
        """Compute novelty with exponential time-decay, fitness-based adaptive decay,
        archive-diversity weighting, and relative novelty weighting.

        Uses hash index for O(1) program lookup instead of O(N) linear scan.

        The decay rate is adaptive based on three factors:

        1. **Fitness-based decay**: High-fitness programs decay slower
           (worth preserving novel), low-fitness programs decay faster
           (less important).

        2. **Archive-diversity weighting**: Diverse archives (high Shannon
           entropy) lower novelty weight because novelty is less critical
           when many unique programs exist. Homogeneous archives increase
           novelty weight to push exploration into under-represented
           regions of behaviour space.

        3. **Relative novelty weighting**: Programs more novel than the
           archive average decay slower (they explore new regions and are
           more valuable). Programs similar to many archive entries decay
           faster (they are less distinctive and more replaceable).
           The multiplier is bounded in [0.3, 2.5] to prevent premature
           elimination or immortal programs.
        """
        _prog_hash = hash(tuple(program))
        indices = self._prog_hash_index.get(_prog_hash, [])
        for idx in indices:
            if idx < len(self._entries):
                entry = self._entries[idx]
                meta = entry[2]
                stored_prog = meta.get("program") if isinstance(meta, dict) else None
                if stored_prog is not None and stored_prog == program:
                    program_generation = meta.get("generation") if isinstance(meta, dict) else None
                    flat_descriptor = entry[0]
                    if program_generation is not None and flat_descriptor is not None:
                        base_novelty = self._k_nn_distance(flat_descriptor)
                        age = current_generation - int(program_generation)

                        # Adaptive decay: fitness modulates the effective half-life.
                        # fitness=0 -> multiplier=0.25  (fast decay)
                        # fitness=1 -> multiplier=2.0   (slow decay)
                        fitness = meta.get("fitness") if isinstance(meta, dict) else None
                        if fitness is not None and isinstance(fitness, (int, float)):
                            clamped = max(0.0, min(1.0, float(fitness)))
                            fitness_multiplier = 0.25 + 1.75 * clamped
                        else:
                            fitness_multiplier = 1.0  # no fitness data -> unchanged decay

                        effective_half_life = half_life * fitness_multiplier

                        # Relative novelty weighting: compare this program's novelty
                        # to the archive average to modulate decay rate.
                        # Programs more novel than average decay slower (more valuable
                        # as they explore new behaviour regions). Programs less novel
                        # decay faster (they are less distinctive).
                        novelty_multiplier = self._relative_novelty_multiplier(base_novelty)
                        effective_half_life *= novelty_multiplier

                        decay = math.exp(-0.693 * age / effective_half_life)

                        # Archive-diversity weighting.
                        # get_diversity_score() returns Shannon entropy in [0, 1].
                        # diversity_weight maps entropy → novelty importance:
                        #   entropy=0 (homogeneous) → weight=1.0 (max pressure)
                        #   entropy=1 (diverse)    → weight=0.1 (floor, never zero)
                        diversity = self.get_diversity_score()
                        diversity_weight = max(0.1, 1.0 - diversity)

                        return base_novelty * decay * diversity_weight
        return 0.0

    # ------------------------------------------------------------------
    # Trajectory & distribution diagnostics
    # ------------------------------------------------------------------

    def get_novelty_trajectory(self, program_key: list[int]) -> list[dict[str, Any]]:
        """Return the novelty trajectory for a specific program across generations.

        Scans all archive entries that match *program_key* (by content equality)
        and returns a list of dicts sorted by generation, each containing:

        - ``generation``: the generation at which the entry was recorded
        - ``novelty_score``: the k-NN novelty score at the time of that entry
        - ``metadata``: the full metadata dict for that entry

        This lets callers track whether a program's novelty is increasing or
        decreasing over time.  An empty list means the program was never
        recorded in the archive.

        Parameters
        ----------
        program_key:
            The program as a ``list[int]`` of instruction words.
        """
        if self.size == 0:
            return []

        _prog_hash = hash(tuple(program_key))
        indices = self._prog_hash_index.get(_prog_hash, [])
        if not indices:
            return []

        if self._tree_dirty:
            self._rebuild_tree()
        if self._tree is None:
            return []

        trajectory: list[dict[str, Any]] = []
        k = min(self.k_neighbors, self.size)
        for idx in indices:
            if idx >= len(self._entries):
                continue
            entry = self._entries[idx]
            meta = entry[2]
            stored_prog = meta.get("program") if isinstance(meta, dict) else None
            if stored_prog is None or stored_prog != program_key:
                continue

            flat = entry[0]
            dists, _ = self._tree.query(np.array(flat), k=k)
            score = float(dists) if k == 1 else float(np.mean(dists))

            generation = meta.get("generation") if isinstance(meta, dict) else None
            trajectory.append({
                "generation": generation,
                "novelty_score": score,
                "metadata": meta,
            })

        # Sort by generation (entries without generation go last).
        trajectory.sort(key=lambda d: (d["generation"] is None, d["generation"] or 0))
        return trajectory

    def get_novelty_distribution_stats(self) -> dict[str, Any]:
        """Return comprehensive distribution statistics of archive novelty scores.

        Unlike :meth:`get_novelty_stats` (which focuses on basic percentiles),
        this method returns additional diagnostics useful for tuning novelty
        pressure:

        - ``mean``, ``median``, ``std``, ``min``, ``max``: basic moments
        - ``q10``, ``q25``, ``q75``, ``q90``: decile/percentile spread
        - ``iqr``: inter-quartile range (Q75 - Q25), measures central spread
        - ``skewness``: Fisher skewness — positive means long right tail
        - ``cv``: coefficient of variation (std / mean) — dispersion relative
          to the mean
        - ``novelty_pressure``: heuristic string ("high", "moderate", "low")
          based on the coefficient of variation

        Returns an empty dict if the archive has fewer than 2 entries.

        Returns
        -------
        dict[str, Any]
            Distribution statistics or ``{}`` for degenerate archives.
        """
        if self.size < 2:
            return {}

        if self._tree_dirty:
            self._rebuild_tree()
        if self._tree is None:
            return {}

        k = min(self.k_neighbors, self.size)
        scores: list[float] = []
        for entry in self._entries:
            flat = entry[0]
            dists, _ = self._tree.query(np.array(flat), k=k)
            score = float(dists) if k == 1 else float(np.mean(dists))
            scores.append(score)

        arr = np.array(scores)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))  # sample std for n>=2
        median_val = float(np.median(arr))

        # Percentiles
        q10 = float(np.percentile(arr, 10))
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        q90 = float(np.percentile(arr, 90))

        # IQR
        iqr = q75 - q25

        # Skewness (Fisher): E[((X-mean)/std)^3]
        if std_val > 0:
            skew = float(np.mean(((arr - mean_val) / std_val) ** 3))
        else:
            skew = 0.0

        # Coefficient of variation
        cv = std_val / mean_val if mean_val > 0 else 0.0

        # Heuristic novelty pressure label
        if cv < 0.3:
            pressure = "high"     # Low variance → scores cluster → strong pressure
        elif cv > 0.8:
            pressure = "low"      # High variance → scores scattered → weak pressure
        else:
            pressure = "moderate"

        return {
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q10": q10,
            "q25": q25,
            "q75": q75,
            "q90": q90,
            "iqr": iqr,
            "skewness": skew,
            "cv": cv,
            "novelty_pressure": pressure,
        }

    # ------------------------------------------------------------------
    # Novelty-impact correlation analysis
    # ------------------------------------------------------------------

    def get_novelty_impact_correlation(self) -> dict[str, Any]:
        """Analyze whether novelty pressure is increasing program diversity.

        Computes correlations between novelty scores and program diversity
        metrics across the archive, and (when distribution history exists)
        between longitudinal novelty pressure and per-generation diversity.

        Diversity metrics extracted from each entry's descriptor:

        - ``inst_diversity``:    fraction of distinct opcode categories used
        - ``reg_pressure``:      fraction of active general-purpose registers
        - ``unique_regs_used``:  number of distinct registers written / 31
        - ``branch_fraction``:   fraction of branch-type instructions
        - ``alu_fraction``:      fraction of ALU-type instructions
        - ``load_store_ratio``:  ratio of memory ops to total ops
        - ``program_length``:    normalised instruction count

        Returns
        -------
        dict[str, Any]
            ``cross_sectional`` (dict):
                Per-metric Pearson and Spearman correlations between each
                entry's novelty score and its descriptor values.

                Keys are metric names (e.g. ``inst_diversity_pearson``).
                Values are floats in [-1, 1].  A ``None`` value means the
                metric was constant across all entries (undefined correlation).

            ``longitudinal`` (dict | None):
                Generation-level correlation between ``novelty_pressure``
                (from distribution snapshots) and ``diversity_score``
                (Shannon entropy of the archive at each snapshot).

                Contains ``pearson_r``, ``pearson_p``, ``spearman_r``,
                ``spearman_p``, and ``n_generations``.  ``None`` when
                fewer than 3 populated snapshots are available.

            ``summary`` (dict):
                High-level interpretation:
                - ``cross_sectional_effect``: ``"positive"`` / ``"negative"``
                  / ``"neutral"`` based on median Pearson r across metrics.
                - ``longitudinal_effect``: ``"positive"`` / ``"negative"``
                  / ``"neutral"`` / ``"insufficient_data"``
                - ``overall_interpretation``: human-readable one-liner.

        Returns an empty dict if the archive has fewer than 3 entries
        (too few for meaningful correlation).
        """
        if self.size < 3:
            return {}

        # --- Cross-sectional analysis: novelty score × descriptor values ---
        if self._tree_dirty:
            self._rebuild_tree()
        if self._tree is None:
            return {}

        k = min(self.k_neighbors, self.size)

        novelty_scores: list[float] = []
        descriptor_values: dict[str, list[float]] = {}

        for entry in self._entries:
            flat = entry[0]
            desc = entry[1]  # original descriptor dict
            dists, _ = self._tree.query(np.array(flat), k=k)
            score = float(dists) if k == 1 else float(np.mean(dists))
            novelty_scores.append(score)

            for metric, val in desc.items():
                if isinstance(val, (int, float)):
                    descriptor_values.setdefault(metric, []).append(float(val))

        novelty_arr = np.array(novelty_scores)
        cross_sectional: dict[str, Any] = {}

        for metric, vals in sorted(descriptor_values.items()):
            metric_arr = np.array(vals)
            # Skip constant metrics (std == 0 → undefined correlation)
            if np.std(metric_arr) == 0.0 or np.std(novelty_arr) == 0.0:
                cross_sectional[f"{metric}_pearson"] = None
                cross_sectional[f"{metric}_spearman"] = None
                continue

            pr, pp = pearsonr(novelty_arr, metric_arr)
            sr, sp = spearmanr(novelty_arr, metric_arr)
            cross_sectional[f"{metric}_pearson"] = float(pr)
            cross_sectional[f"{metric}_pearson_p"] = float(pp)
            cross_sectional[f"{metric}_spearman"] = float(sr)
            cross_sectional[f"{metric}_spearman_p"] = float(sp)

        # --- Longitudinal analysis: novelty pressure × diversity over time ---
        longitudinal: dict[str, Any] | None = None
        populated = [s for s in self._distribution_history if s.get("populated", False)]

        if len(populated) >= 3:
            pressures: list[float] = []
            diversities: list[float] = []
            for snap in populated:
                cv = snap.get("cv")
                # Recompute diversity_score from archive size at snapshot time
                # We approximate: use the entropy-like metric from the snapshot
                # if available, otherwise use the current diversity as a proxy.
                # For a more accurate longitudinal diversity, we'd need to store
                # diversity_score in each snapshot (which record_distribution_snapshot
                # does not currently do).  So we use cv as the diversity proxy:
                # high cv → scattered scores → more diversity.
                # Instead, let's compute a diversity proxy from the snapshot stats.
                if cv is not None and cv > 0:
                    # Normalise cv to [0, 1] as a diversity proxy
                    div_proxy = min(cv / 2.0, 1.0)
                else:
                    div_proxy = 0.0

                # For pressure, lower cv means stronger novelty pressure
                # (scores cluster tightly).  Map cv to pressure: cv→0 = high.
                if cv is not None:
                    pressure_val = max(0.0, 1.0 - min(cv, 2.0) / 2.0)
                else:
                    pressure_val = 0.0

                pressures.append(pressure_val)
                diversities.append(div_proxy)

            if len(pressures) >= 3:
                p_arr = np.array(pressures)
                d_arr = np.array(diversities)
                # Guard: if either array is constant, correlation is undefined.
                if np.std(p_arr) > 0 and np.std(d_arr) > 0:
                    pr, pp = pearsonr(p_arr, d_arr)
                    sr, sp = spearmanr(p_arr, d_arr)
                else:
                    pr, pp, sr, sp = 0.0, 1.0, 0.0, 1.0
                longitudinal = {
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "spearman_r": float(sr),
                    "spearman_p": float(sp),
                    "n_generations": len(populated),
                    "mean_pressure": float(np.mean(p_arr)),
                    "mean_diversity_proxy": float(np.mean(d_arr)),
                }

        # --- Summary / interpretation ---
        pearson_rs = [
            v for k, v in cross_sectional.items()
            if k.endswith("_pearson") and v is not None
        ]
        if pearson_rs:
            median_r = float(np.median(pearson_rs))
            if median_r > 0.1:
                cs_effect = "positive"
            elif median_r < -0.1:
                cs_effect = "negative"
            else:
                cs_effect = "neutral"
        else:
            median_r = 0.0
            cs_effect = "neutral"

        if longitudinal is not None:
            lr = longitudinal.get("pearson_r", 0.0)
            if lr > 0.1:
                lg_effect = "positive"
            elif lr < -0.1:
                lg_effect = "negative"
            else:
                lg_effect = "neutral"
        else:
            lg_effect = "insufficient_data"

        # Build interpretation
        if cs_effect == "positive" and lg_effect in ("positive", "neutral"):
            interp = (
                f"Novelty pressure shows positive cross-sectional correlation "
                f"(median r={median_r:.3f}) with diversity metrics, "
                f"suggesting it is increasing program diversity."
            )
        elif cs_effect == "negative":
            interp = (
                f"Novelty pressure shows negative cross-sectional correlation "
                f"(median r={median_r:.3f}) with diversity metrics — "
                f"higher novelty scores may concentrate rather than diversify."
            )
        else:
            interp = (
                f"Cross-sectional correlation is weak (median r={median_r:.3f}). "
                f"Novelty pressure has neutral impact on diversity."
            )

        if longitudinal:
            lr = longitudinal.get("pearson_r", 0.0)
            lp = longitudinal.get("pearson_p", 1.0)
            sig = " (p<0.05)" if lp < 0.05 else ""
            interp += (
                f" Longitudinal r={lr:.3f}{sig} across "
                f"{longitudinal['n_generations']} generations."
            )
        elif lg_effect == "insufficient_data":
            interp += " Insufficient longitudinal data (<3 generations) for temporal analysis."

        summary = {
            "cross_sectional_effect": cs_effect,
            "longitudinal_effect": lg_effect,
            "overall_interpretation": interp,
            "median_pearson_r": median_r,
            "n_metrics_correlated": len(pearson_rs),
        }

        return {
            "cross_sectional": cross_sectional,
            "longitudinal": longitudinal,
            "summary": summary,
        }

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
