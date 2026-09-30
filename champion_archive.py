#!/usr/bin/env python3
"""ChampionArchive — persistent training data across evolution cycles.

Stores champion programs with metadata and provides weighted sampling
for the training pipeline. This replaces the ad-hoc champion trace
collection that only kept 5 traces across all cycles.

Design goals:
- Persistent: survives process restarts (JSONL on disk)
- Deduplicated: same instruction sequence stored once
- Quality-aware: evicts lowest-fitness when over capacity
- Weighted sampling: recent + high-fitness champions weighted higher
- Fast: O(1) append, O(N) sample with precomputed weights
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Tier constants for preventing catastrophic forgetting / mode collapse
# ──────────────────────────────────────────────────────────────────────
TIER_RECENT_SIZE = 20  # last N entries added
TIER_ELITE_SIZE = 15  # top N by fitness
TIER_DIVERSE_SIZE = 15  # top N by instruction pattern uniqueness
FOUNDATIONAL_LEVELS: set[int] = set(range(7))  # L0-L6 never evicted

TIER_QUOTAS = {
    "recent": 0.40,
    "elite": 0.25,
    "diverse": 0.20,
    "foundational": 0.15,
}


@dataclass
class ChampionEntry:
    """A single champion program with metadata."""

    program: list[int]  # instruction words
    program_hex: list[str]  # hex strings for training pipeline
    level: int  # curriculum level achieved
    fitness: float  # fitness score
    cycle: int  # which evolution cycle produced this
    generation: int  # which generation within the cycle
    instructions_count: int  # number of instructions
    sha256: str  # dedup key
    trace_status: str = "ok"  # execution status
    task_score: float = 0.0  # raw task score
    timestamp: float = 0.0  # time.time() when saved

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ChampionEntry:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _hash_program(program: list[int]) -> str:
    """Fast hash of instruction sequence for deduplication.

    Handles 32-bit instruction words by converting to 4-byte big-endian chunks.
    """
    data = b"".join(w.to_bytes(4, "big") for w in program)
    return hashlib.sha256(data).hexdigest()[:16]


class ChampionArchive:
    """Persistent archive of champion programs across evolution cycles.

    Storage format: JSONL (one JSON object per line) for fast append
    and easy inspection. Memory-mapped for reads during training.

    Usage:
        archive = ChampionArchive(path="champions/archive.jsonl")
        archive.add(program, level=13, fitness=224.14, cycle=1)
        archive.add(program, level=13, fitness=208.67, cycle=2)

        # For training pipeline
        examples = archive.sample_training_batch(batch_size=512)
        archive.save_recurrent_data("milestone1/recurrent_data.jsonl")
    """

    def __init__(
        self,
        path: str = "champions/archive.jsonl",
        max_size: int = 500,
        recent_weight: float = 0.5,
        high_fitness_weight: float = 0.3,
        stale_threshold_generations: int = 100,
    ):
        self.path = path
        self.max_size = max_size
        self.recent_weight = recent_weight
        self.high_fitness_weight = high_fitness_weight
        self.stale_threshold_generations = stale_threshold_generations
        self.entries: list[ChampionEntry] = []
        self._seen_hashes: set = set()
        self.tiers: dict[str, list[int]] = {
            "recent": [],
            "elite": [],
            "diverse": [],
            "foundational": [],
        }
        self._load()
        self._update_tiers()

    def _load(self) -> None:
        """Load existing archive from disk."""
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        entry = ChampionEntry.from_dict(d)
                        if entry.sha256 not in self._seen_hashes:
                            self.entries.append(entry)
                            self._seen_hashes.add(entry.sha256)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:  # Intentional: file may not exist yet or be corrupted; skip gracefully
            pass

    def add(
        self,
        program: list[int],
        level: int,
        fitness: float,
        cycle: int,
        generation: int = 0,
        trace_status: str = "ok",
        task_score: float = 0.0,
    ) -> bool:
        """Add a champion to the archive. Returns True if actually added (not duplicate).

        Deduplicates by instruction sequence hash. If archive is full,
        evicts the lowest-fitness entry from the oldest cycle.
        """
        sha256 = _hash_program(program)
        if sha256 in self._seen_hashes:
            return False

        import time

        entry = ChampionEntry(
            program=program,
            program_hex=[f"0x{w:08x}" for w in program],
            level=level,
            fitness=fitness,
            cycle=cycle,
            generation=generation,
            instructions_count=len(program),
            sha256=sha256,
            trace_status=trace_status,
            task_score=task_score,
            timestamp=time.time(),
        )

        self.entries.append(entry)
        self._seen_hashes.add(sha256)

        # Evict if over capacity: prefer evicting lower-level entries
        if len(self.entries) > self.max_size:
            self._evict(new_level=level)

        self._update_tiers()
        return True

    def _evict(self, new_level: int = 0) -> None:
        """Remove an entry to make room, preferring lower-level entries.

        R47: When a higher-level entry is added, evict lower-level entries
        first (they're redundant once the model has progressed). Falls back
        to oldest + lowest-fitness among same-level entries.

        Foundational entries (L0-L6) are never evicted.
        """
        if not self.entries:
            return
        # Filter out foundational entries — they are never evicted
        evictable = [i for i in range(len(self.entries)) if self.entries[i].level not in FOUNDATIONAL_LEVELS]
        if not evictable:
            evictable = list(range(len(self.entries)))

        # Prefer evicting entries from levels BELOW the new entry's level
        lower_level = [i for i in evictable if self.entries[i].level < new_level]
        if lower_level:
            # Among lower-level entries, evict the oldest
            worst_idx = min(lower_level, key=lambda i: self.entries[i].cycle)
        else:
            # Fallback: oldest + lowest fitness (same or higher level)
            worst_idx = min(
                evictable,
                key=lambda i: (self.entries[i].cycle, self.entries[i].fitness),
            )
        removed = self.entries.pop(worst_idx)
        self._seen_hashes.discard(removed.sha256)

    def sample_training_batch(
        self,
        batch_size: int = 512,
        rng: random.Random | None = None,
    ) -> list[ChampionEntry]:
        """Weighted sample of champions for training.

        Weights combine:
        - Recency: recent cycles weighted higher
        - Fitness: higher fitness weighted higher
        - Level: higher levels weighted higher
        """
        if not self.entries:
            return []
        if rng is None:
            rng = random.Random()

        max_cycle = max(e.cycle for e in self.entries) if self.entries else 1
        max_fitness = max(e.fitness for e in self.entries) if self.entries else 1
        max_level = max(e.level for e in self.entries) if self.entries else 1

        weights = []
        for e in self.entries:
            recency = (e.cycle / max(max_cycle, 1)) * self.recent_weight
            quality = (e.fitness / max(max_fitness, 1)) * self.high_fitness_weight
            level_bonus = (e.level / max(max_level, 1)) * 0.2
            w = recency + quality + level_bonus + 0.1  # base weight
            weights.append(max(w, 0.01))

        total = sum(weights)
        if total <= 0:
            return list(self.entries)[:batch_size]

        probs = [w / total for w in weights]
        n = min(batch_size, len(self.entries))
        indices = rng.choices(range(len(self.entries)), weights=probs, k=n)
        return [self.entries[i] for i in indices]

    # ──────────────────────────────────────────────────────────────
    # Tiered sampling — prevents catastrophic forgetting / mode collapse
    # ──────────────────────────────────────────────────────────────

    def _update_tiers(self) -> None:
        """Recompute tier assignments from current entries.

        Tiers:
          recent      — last TIER_RECENT_SIZE entries (by insertion order)
          elite       — top TIER_ELITE_SIZE by fitness
          diverse     — top TIER_DIVERSE_SIZE by instruction-pattern uniqueness
          foundational— entries at L0-L6, never evicted
        """
        self.tiers = {
            "recent": [],
            "elite": [],
            "diverse": [],
            "foundational": [],
        }
        if not self.entries:
            return

        n = len(self.entries)

        # Foundational: L0-L6, never evicted
        self.tiers["foundational"] = [i for i in range(n) if self.entries[i].level in FOUNDATIONAL_LEVELS]

        # Recent: last TIER_RECENT_SIZE entries by insertion order
        start = max(0, n - TIER_RECENT_SIZE)
        self.tiers["recent"] = list(range(start, n))

        # Elite: top TIER_ELITE_SIZE by fitness
        sorted_by_fit = sorted(range(n), key=lambda i: self.entries[i].fitness, reverse=True)
        self.tiers["elite"] = sorted_by_fit[:TIER_ELITE_SIZE]

        # Diverse: top TIER_DIVERSE_SIZE by Hamming distance from nearest elite
        elite_set = set(self.tiers["elite"])
        candidates = [(i, self._pattern_diversity_score(i)) for i in range(n) if i not in elite_set]
        candidates.sort(key=lambda x: x[1], reverse=True)
        self.tiers["diverse"] = [i for i, _ in candidates[:TIER_DIVERSE_SIZE]]

    def _pattern_diversity_score(self, idx: int) -> float:
        """Measure Hamming distance from nearest elite pattern.

        Higher score = more unique instruction patterns = higher diversity.
        """
        if not self.tiers["elite"]:
            return 0.0
        entry = self.entries[idx]
        min_dist = float("inf")
        for elite_idx in self.tiers["elite"]:
            dist = self._hamming_distance(entry.program, self.entries[elite_idx].program)
            min_dist = min(min_dist, dist)
        return min_dist if min_dist != float("inf") else 0.0

    @staticmethod
    def _hamming_distance(prog1: list[int], prog2: list[int]) -> int:
        """Count total differing bits between two instruction sequences."""
        total = 0
        max_len = max(len(prog1), len(prog2))
        for i in range(max_len):
            w1 = prog1[i] if i < len(prog1) else 0
            w2 = prog2[i] if i < len(prog2) else 0
            total += (w1 ^ w2).bit_count()
        return total

    def _compute_diversity_scores(self, candidates: list[ChampionEntry]) -> list[float]:
        """Compute diversity scores for a list of candidates.

        Combines two signals:
        - Program length deviation: how far each program's length is from the mean
        - Instruction uniqueness: count of distinct instruction opcodes

        Returns scores normalized to [0, 1] where higher = more diverse.
        """
        if len(candidates) <= 1:
            return [1.0] * len(candidates)

        # Program length diversity: absolute deviation from mean length
        lengths = [len(e.program) for e in candidates]
        mean_len = sum(lengths) / len(lengths)
        max_dev = max(abs(length - mean_len) for length in lengths) or 1.0

        # Instruction diversity: unique instruction count per program
        unique_counts = [len(set(e.program)) for e in candidates]
        max_unique = max(unique_counts) or 1

        scores = []
        for i in range(len(candidates)):
            length_div = abs(lengths[i] - mean_len) / max_dev
            instr_div = unique_counts[i] / max_unique
            # Equal-weight combination of the two diversity signals
            div_score = 0.5 * length_div + 0.5 * instr_div
            scores.append(div_score)

        return scores

    def _compute_combined_scores(self, candidates: list, diversity_weight: float) -> list[tuple[float, object]]:
        """Compute combined fitness+diversity scores for candidates."""
        fitnesses = [e.fitness for e in candidates]
        min_fit = min(fitnesses)
        max_fit = max(fitnesses)
        fit_range = max_fit - min_fit
        diversity_scores = self._compute_diversity_scores(candidates)
        combined: list[tuple[float, object]] = []
        for i, e in enumerate(candidates):
            norm_fit = (e.fitness - min_fit) / max(fit_range, 1e-10)
            score = (1 - diversity_weight) * norm_fit + diversity_weight * diversity_scores[i]
            combined.append((score, e))
        return combined

    def _tournament_select(self, combined: list[tuple[float, object]], n: int) -> list[dict]:
        """Select n champions via tournament selection from combined scores."""
        pool_size = min(2 * n, len(combined))
        combined.sort(key=lambda x: x[0], reverse=True)
        pool = list(combined[:pool_size])
        tournament_size = max(2, pool_size // n) if n > 0 else 2
        selected: list[dict] = []
        available = list(pool)
        for _ in range(n):
            if not available:
                break
            t_size = min(tournament_size, len(available))
            contestants = random.sample(available, t_size)
            winner = max(contestants, key=lambda x: x[0])
            selected.append(winner[1].to_dict())
            available.remove(winner)
        return selected

    def diversity_aware_select(
        self,
        level: int,
        n: int = 1,
        diversity_weight: float = 0.3,
    ) -> list[dict]:
        """Select n champions at *level* while balancing quality and diversity.

        Combined score per candidate:
            score = (1 - diversity_weight) * normalized_fitness
                  + diversity_weight * diversity_score

        Selection uses tournament picks among the top-scoring pool.

        Parameters
        ----------
        level : int
            Curriculum level to select from.
        n : int
            Number of champions to return.
        diversity_weight : float
            Weight given to diversity vs. fitness in [0, 1].

        Returns
        -------
        List[Dict]
            List of champion dicts (ChampionEntry.to_dict() format).
        """
        candidates = [e for e in self.entries if e.level == level]
        if not candidates:
            return []
        n = min(n, len(candidates))
        if n == 0:
            return []
        combined = self._compute_combined_scores(candidates, diversity_weight)
        return self._tournament_select(combined, n)

    def sample_tiered_batch(
        self,
        batch_size: int = 512,
        rng: random.Random | None = None,
    ) -> list[ChampionEntry]:
        """Tiered sampling with quotas to prevent catastrophic forgetting.

        Quotas (default):
          recent:      40%  — keeps training aligned with latest evolution
          elite:       25%  — high-fitness exemplars anchor quality
          diverse:     20%  — unique instruction patterns combat mode collapse
          foundational:15%  — L0-L6 fundamentals never forgotten

        Falls back to sample_training_batch() when the archive is too small
        for meaningful tier separation.
        """
        if not self.entries:
            return []
        if rng is None:
            rng = random.Random()

        # If archive is small, tiered sampling adds no value — fall back
        if len(self.entries) <= TIER_RECENT_SIZE:
            return self.sample_training_batch(batch_size, rng)

        # Calculate per-tier quotas
        quotas = {k: int(batch_size * v) for k, v in TIER_QUOTAS.items()}
        # Distribute rounding remainder to 'recent'
        remainder = batch_size - sum(quotas.values())
        quotas["recent"] += remainder

        result: list[ChampionEntry] = []
        used: set = set()

        for tier_name, count in quotas.items():
            tier_indices = self.tiers.get(tier_name, [])
            if not tier_indices:
                continue
            available = [i for i in tier_indices if i not in used]
            if not available:
                continue
            n = min(count, len(available))
            sampled = rng.sample(available, n)
            for i in sampled:
                result.append(self.entries[i])
                used.add(i)

        return result

    def save(self) -> None:
        """Persist archive to disk (JSONL)."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            f.writelines(json.dumps(entry.to_dict()) + "\n" for entry in self.entries)

    def _select_recurrent_entries(self, eligible: list[ChampionEntry], max_entries: int) -> list[ChampionEntry]:
        """Select entries using weighted mix: 50% top, 30% recent, 20% diverse."""
        if len(eligible) <= max_entries:
            return eligible
        top_count = int(max_entries * 0.50)
        recent_count = int(max_entries * 0.30)
        diverse_count = max_entries - top_count - recent_count
        sorted_by_fitness = sorted(eligible, key=lambda e: e.fitness, reverse=True)
        top_set = {id(e) for e in sorted_by_fitness[:top_count]}
        top_selected = sorted_by_fitness[:top_count]
        entry_order = {id(e): idx for idx, e in enumerate(self.entries)}
        sorted_by_recency = sorted(eligible, key=lambda e: entry_order.get(id(e), 0), reverse=True)
        recent_selected = []
        for e in sorted_by_recency:
            if len(recent_selected) >= recent_count:
                break
            if id(e) not in top_set:
                recent_selected.append(e)
        reference_progs = [e.program for e in top_selected + recent_selected]

        def _min_hamming(prog: list[int]) -> int:
            if not reference_progs:
                return 0
            return min(self._hamming_distance(prog, ref) for ref in reference_progs)

        remaining = [e for e in eligible if id(e) not in top_set and id(e) not in {id(r) for r in recent_selected}]
        remaining_with_diversity = [(e, _min_hamming(e.program)) for e in remaining]
        remaining_with_diversity.sort(key=lambda x: x[1], reverse=True)
        diverse_selected = [e for e, _ in remaining_with_diversity[:diverse_count]]
        selected = top_selected + recent_selected + diverse_selected
        seen = set()
        deduped = []
        for e in selected:
            if e.sha256 not in seen:
                seen.add(e.sha256)
                deduped.append(e)
        return deduped

    def _write_recurrent_entries(self, output_path: str, selected: list[ChampionEntry]) -> int:
        """Write selected entries as JSONL."""
        count = 0
        with open(output_path, "w") as f:
            for entry in selected:
                example = {
                    "level": entry.level,
                    "generation": entry.generation,
                    "fitness": entry.fitness,
                    "champion_sha256": entry.sha256,
                    "champion_words": entry.instructions_count,
                    "champion_program": entry.program_hex,
                    "trace_status": entry.trace_status,
                    "trace_instr_count": entry.instructions_count,
                    "task_score": entry.task_score,
                    "save_reason": "champion_archive",
                    "cycle": entry.cycle,
                }
                f.write(json.dumps(example) + "\n")
                count += 1
        return count

    def save_recurrent_data(
        self,
        output_path: str,
        max_entries: int = 100,
        fitness_threshold: float = 150.0,
    ) -> int:
        """Export archive as recurrent_data.jsonl for training pipeline.

        Parameters
        ----------
        output_path : str
            Path to write the JSONL output.
        max_entries : int, optional
            Maximum number of entries to write. When exceeded, entries are
            selected using a weighted mix: 50% top performers (by fitness),
            30% recent (last entries added), 20% diverse (highest Hamming
            distance from nearest elite). Default 100.
        fitness_threshold : float, optional
            Minimum fitness for an entry to be included. Entries with
            fitness below this threshold are filtered out. Default 150.0.
            Set to 0.0 to disable filtering.

        Returns
        -------
        int
            Number of examples written.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        eligible = [entry for entry in self.entries if entry.fitness >= fitness_threshold]
        selected = self._select_recurrent_entries(eligible, max_entries)
        return self._write_recurrent_entries(output_path, selected)

    @property
    def current_max_cycle(self) -> int:
        """Return the highest cycle number across all entries."""
        if not self.entries:
            return 0
        return max(e.cycle for e in self.entries)

    def get_quality_distribution(self) -> dict[str, Any]:
        """Return a distribution of fitness values across entries.

        Returns a dict with:
          - fitness_buckets: list of {min, max, count} dicts (10 bins)
          - mean, median, stddev: summary statistics
        """
        if not self.entries:
            return {"fitness_buckets": [], "mean": 0.0, "median": 0.0, "stddev": 0.0}
        fitnesses = [e.fitness for e in self.entries]
        f_min, f_max = min(fitnesses), max(fitnesses)
        n_bins = 10
        if f_max - f_min < 1e-12:
            buckets = [{"min": f_min, "max": f_max, "count": len(fitnesses)}]
        else:
            width = (f_max - f_min) / n_bins
            buckets = [{"min": f_min + i * width, "max": f_min + (i + 1) * width, "count": 0} for i in range(n_bins)]
            for f in fitnesses:
                b_idx = min(int((f - f_min) / width), n_bins - 1)
                buckets[b_idx]["count"] += 1
        mean = sum(fitnesses) / len(fitnesses)
        sorted_f = sorted(fitnesses)
        mid = len(sorted_f) // 2
        median = sorted_f[mid] if len(sorted_f) % 2 == 1 else (sorted_f[mid - 1] + sorted_f[mid]) / 2
        variance = sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)
        return {"fitness_buckets": buckets, "mean": mean, "median": median, "stddev": variance**0.5}

    def get_recency_distribution(self) -> dict[str, Any]:
        """Return a distribution of cycle numbers across entries.

        Returns a dict with:
          - cycle_buckets: list of {cycle, count} dicts
          - max_cycle, min_cycle: cycle range
          - avg_age_cycles: average distance from max_cycle
        """
        if not self.entries:
            return {"cycle_buckets": [], "max_cycle": 0, "min_cycle": 0, "avg_age_cycles": 0.0}
        cycles = [e.cycle for e in self.entries]
        max_c = max(cycles)
        min_c = min(cycles)
        cycle_counts: dict[int, int] = {}
        for c in cycles:
            cycle_counts[c] = cycle_counts.get(c, 0) + 1
        buckets = [{"cycle": c, "count": cycle_counts[c]} for c in sorted(cycle_counts)]
        avg_age = sum(max_c - c for c in cycles) / len(cycles)
        return {"cycle_buckets": buckets, "max_cycle": max_c, "min_cycle": min_c, "avg_age_cycles": avg_age}

    def prune_stale(self, max_cycles_old: int = 20) -> int:
        """Remove entries older than stale_threshold_generations from the current max cycle.

        Foundational entries (L0-L6) are never pruned.
        Uses self.stale_threshold_generations as the default threshold.
        Returns the number of entries removed.
        """
        if not self.entries:
            return 0
        current_max = max(e.cycle for e in self.entries)
        cutoff = current_max - self.stale_threshold_generations
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.cycle >= cutoff or e.level in FOUNDATIONAL_LEVELS]
        # Rebuild seen hashes
        self._seen_hashes = {e.sha256 for e in self.entries}
        self._update_tiers()
        return before - len(self.entries)

    def get_coverage_stats(self) -> dict[str, Any]:
        """Return coverage statistics across cycles and levels.

        Returns:
          - total_entries: total count
          - unique_cycles: number of distinct cycles
          - unique_levels: number of distinct levels
          - cycle_range: [min_cycle, max_cycle]
          - level_range: [min_level, max_level]
          - foundational_count: entries at L0-L6
        """
        if not self.entries:
            return {
                "total_entries": 0,
                "unique_cycles": 0,
                "unique_levels": 0,
                "cycle_range": [0, 0],
                "level_range": [0, 0],
                "foundational_count": 0,
            }
        cycles = sorted({e.cycle for e in self.entries})
        levels = sorted({e.level for e in self.entries})
        foundational = sum(1 for e in self.entries if e.level in FOUNDATIONAL_LEVELS)
        return {
            "total_entries": len(self.entries),
            "unique_cycles": len(cycles),
            "unique_levels": len(levels),
            "cycle_range": [cycles[0], cycles[-1]],
            "level_range": [levels[0], levels[-1]],
            "foundational_count": foundational,
        }

    def compute_sample_weights(self, entries: list[ChampionEntry] | None = None) -> list[float]:
        """Compute normalized sample weights for the given entries (or all).

        Weights combine recency, fitness, and level — same formula as
        sample_training_batch but returned as a flat list of floats.
        """
        if entries is None:
            entries = self.entries
        if not entries:
            return []
        max_cycle = max(e.cycle for e in entries) if entries else 1
        max_fitness = max(e.fitness for e in entries) if entries else 1
        max_level = max(e.level for e in entries) if entries else 1
        weights = []
        for e in entries:
            recency = (e.cycle / max(max_cycle, 1)) * self.recent_weight
            quality = (e.fitness / max(max_fitness, 1)) * self.high_fitness_weight
            level_bonus = (e.level / max(max_level, 1)) * 0.2
            w = recency + quality + level_bonus + 0.1
            weights.append(max(w, 0.01))
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        return weights

    @property
    def size(self) -> int:
        return len(self.entries)

    def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        if not self.entries:
            return {"size": 0}
        cycles = {e.cycle for e in self.entries}
        levels = {e.level for e in self.entries}
        return {
            "size": len(self.entries),
            "cycles": sorted(cycles),
            "levels": sorted(levels),
            "max_fitness": max(e.fitness for e in self.entries),
            "avg_fitness": sum(e.fitness for e in self.entries) / len(self.entries),
            "avg_words": sum(e.instructions_count for e in self.entries) / len(self.entries),
        }

    def get_sample_weights_summary(self) -> dict:
        """Return statistics about the tiered sampling weights."""
        return {
            "tier_quotas": TIER_QUOTAS.copy(),
            "tier_sizes": {k: len(v) for k, v in self.tiers.items()},
            "total_entries": self.size,
            "foundational_levels": sorted(FOUNDATIONAL_LEVELS),
        }

    def quality_weighted_prune(self) -> int:
        """Prune champions using quality-weighted scoring instead of simple staleness.

        Score = fitness_score * recency_score where:
          fitness_score = fitness / max_fitness
          recency_score = exp(-0.693 * age_generations / stale_threshold_generations)

        Foundational entries (L0-L6) are never pruned.
        When over capacity, the bottom 20% by quality score are removed.
        Returns the count of entries removed.
        """
        if not self.entries:
            return 0

        if len(self.entries) <= self.max_size:
            return 0

        current_max_cycle = max(e.cycle for e in self.entries)
        max_fitness = max(e.fitness for e in self.entries) if self.entries else 1.0
        max_fitness = max(max_fitness, 1e-10)

        # Compute quality scores for all entries
        scored = []
        for i, e in enumerate(self.entries):
            fitness_score = e.fitness / max_fitness
            age = current_max_cycle - e.cycle
            recency_score = math.exp(-0.693 * age / max(self.stale_threshold_generations, 1))
            quality_score = fitness_score * recency_score
            scored.append((quality_score, e))

        # Separate foundational entries (never pruned)
        non_foundational = [(s, e) for s, e in scored if e.level not in FOUNDATIONAL_LEVELS]
        [(s, e) for s, e in scored if e.level in FOUNDATIONAL_LEVELS]

        # Sort non-foundational by quality score (ascending = worst first)
        non_foundational.sort(key=lambda x: x[0])

        # How many to remove: bottom 20% or enough to get to max_size
        over_by = len(self.entries) - self.max_size
        remove_count = max(over_by, int(len(non_foundational) * 0.2))
        remove_count = min(remove_count, len(non_foundational))

        # Remove the worst entries
        removed_set = set()
        for score, entry in non_foundational[:remove_count]:
            removed_set.add(entry.sha256)

        self.entries = [e for e in self.entries if e.sha256 not in removed_set]
        self._seen_hashes -= removed_set
        self._update_tiers()

        return remove_count

    def get_diversity_report(self) -> dict[str, Any]:
        """Analyze champion population diversity.

        Returns a dict with:
          - total_champions (int)
          - unique_levels (list of int)
          - avg_fitness_by_level (dict level -> float)
          - program_length_variance (float)
          - instruction_diversity (float, fraction of unique opcodes across all champions)
          - fitness_correlation_with_length (float, Pearson correlation between program length and fitness)
        """
        if not self.entries:
            return {
                "total_champions": 0,
                "unique_levels": [],
                "avg_fitness_by_level": {},
                "program_length_variance": 0.0,
                "instruction_diversity": 0.0,
                "fitness_correlation_with_length": 0.0,
            }

        total = len(self.entries)

        # Unique levels
        unique_levels = sorted({e.level for e in self.entries})

        # Average fitness by level
        level_fitness: dict[int, list[float]] = {}
        for e in self.entries:
            level_fitness.setdefault(e.level, []).append(e.fitness)
        avg_fitness_by_level = {lvl: sum(fits) / len(fits) for lvl, fits in level_fitness.items()}

        # Program length variance
        lengths = [e.instructions_count for e in self.entries]
        mean_len = sum(lengths) / total
        program_length_variance = sum((length - mean_len) ** 2 for length in lengths) / total

        # Instruction diversity: fraction of unique opcodes across all champions
        all_opcodes: set[int] = set()
        for e in self.entries:
            all_opcodes.update(e.program)
        total_unique_opcodes = len(all_opcodes)
        # Use total program length as denominator for fraction
        total_instructions = sum(e.instructions_count for e in self.entries)
        instruction_diversity = total_unique_opcodes / total_instructions if total_instructions > 0 else 0.0

        # Pearson correlation between program length and fitness
        fitnesses = [e.fitness for e in self.entries]
        mean_fit = sum(fitnesses) / total
        numerator = sum((lengths[i] - mean_len) * (fitnesses[i] - mean_fit) for i in range(total))
        denom_len = math.sqrt(sum((length - mean_len) ** 2 for length in lengths))
        denom_fit = math.sqrt(sum((f - mean_fit) ** 2 for f in fitnesses))
        if denom_len > 0 and denom_fit > 0:
            fitness_correlation_with_length = numerator / (denom_len * denom_fit)
        else:
            fitness_correlation_with_length = 0.0

        return {
            "total_champions": total,
            "unique_levels": unique_levels,
            "avg_fitness_by_level": avg_fitness_by_level,
            "program_length_variance": program_length_variance,
            "instruction_diversity": instruction_diversity,
            "fitness_correlation_with_length": fitness_correlation_with_length,
        }
