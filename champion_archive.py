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
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ──────────────────────────────────────────────────────────────────────
# Tier constants for preventing catastrophic forgetting / mode collapse
# ──────────────────────────────────────────────────────────────────────
TIER_RECENT_SIZE = 20  # last N entries added
TIER_ELITE_SIZE = 15  # top N by fitness
TIER_DIVERSE_SIZE = 15  # top N by instruction pattern uniqueness
FOUNDATIONAL_LEVELS: Set[int] = set(range(7))  # L0-L6 never evicted

TIER_QUOTAS = {
    "recent": 0.40,
    "elite": 0.25,
    "diverse": 0.20,
    "foundational": 0.15,
}


@dataclass
class ChampionEntry:
    """A single champion program with metadata."""

    program: List[int]  # instruction words
    program_hex: List[str]  # hex strings for training pipeline
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
    def from_dict(cls, d: dict) -> "ChampionEntry":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _hash_program(program: List[int]) -> str:
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
    ):
        self.path = path
        self.max_size = max_size
        self.recent_weight = recent_weight
        self.high_fitness_weight = high_fitness_weight
        self.entries: List[ChampionEntry] = []
        self._seen_hashes: set = set()
        self.tiers: Dict[str, List[int]] = {
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
        except Exception:
            pass

    def add(
        self,
        program: List[int],
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
        rng: Optional[random.Random] = None,
    ) -> List[ChampionEntry]:
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
    def _hamming_distance(prog1: List[int], prog2: List[int]) -> int:
        """Count total differing bits between two instruction sequences."""
        total = 0
        max_len = max(len(prog1), len(prog2))
        for i in range(max_len):
            w1 = prog1[i] if i < len(prog1) else 0
            w2 = prog2[i] if i < len(prog2) else 0
            total += bin(w1 ^ w2).count("1")
        return total

    def sample_tiered_batch(
        self,
        batch_size: int = 512,
        rng: Optional[random.Random] = None,
    ) -> List[ChampionEntry]:
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

        result: List[ChampionEntry] = []
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
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict()) + "\n")

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

        # 1) Filter by fitness threshold
        eligible = [entry for entry in self.entries if entry.fitness >= fitness_threshold]

        # 2) If under the cap, write all eligible
        if len(eligible) <= max_entries:
            selected = eligible
        else:
            # 3) Weighted selection: 50% top, 30% recent, 20% diverse
            top_count = int(max_entries * 0.50)
            recent_count = int(max_entries * 0.30)
            diverse_count = max_entries - top_count - recent_count  # remainder

            # --- Top performers (by fitness) ---
            sorted_by_fitness = sorted(eligible, key=lambda e: e.fitness, reverse=True)
            top_set = set(id(e) for e in sorted_by_fitness[:top_count])
            top_selected = sorted_by_fitness[:top_count]

            # --- Recent entries (by insertion order = position in self.entries) ---
            entry_order = {id(e): idx for idx, e in enumerate(self.entries)}
            sorted_by_recency = sorted(eligible, key=lambda e: entry_order.get(id(e), 0), reverse=True)
            # Avoid duplicates with top set
            recent_selected = []
            for e in sorted_by_recency:
                if len(recent_selected) >= recent_count:
                    break
                if id(e) not in top_set:
                    recent_selected.append(e)

            # --- Diverse entries (highest Hamming distance from nearest elite) ---
            # Build a reference set from top + recent already selected
            reference_progs = [e.program for e in top_selected + recent_selected]

            def _min_hamming(prog: List[int]) -> int:
                """Minimum Hamming distance from any reference program."""
                if not reference_progs:
                    return 0
                return min(self._hamming_distance(prog, ref) for ref in reference_progs)

            remaining = [
                e for e in eligible if id(e) not in top_set and id(e) not in set(id(r) for r in recent_selected)
            ]
            remaining_with_diversity = [(e, _min_hamming(e.program)) for e in remaining]
            remaining_with_diversity.sort(key=lambda x: x[1], reverse=True)
            diverse_selected = [e for e, _ in remaining_with_diversity[:diverse_count]]

            selected = top_selected + recent_selected + diverse_selected
            # Deduplicate (shouldn't happen, but safety)
            seen = set()
            deduped = []
            for e in selected:
                if e.sha256 not in seen:
                    seen.add(e.sha256)
                    deduped.append(e)
            selected = deduped

        # 4) Write selected entries
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

    @property
    def size(self) -> int:
        return len(self.entries)

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        if not self.entries:
            return {"size": 0}
        cycles = set(e.cycle for e in self.entries)
        levels = set(e.level for e in self.entries)
        return {
            "size": len(self.entries),
            "cycles": sorted(cycles),
            "levels": sorted(levels),
            "max_fitness": max(e.fitness for e in self.entries),
            "avg_fitness": sum(e.fitness for e in self.entries) / len(self.entries),
            "avg_words": sum(e.instructions_count for e in self.entries) / len(self.entries),
        }
