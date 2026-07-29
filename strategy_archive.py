#!/usr/bin/env python3
"""StrategyArchive — Record and recommend mutation strategies by level.

Pandas-free implementation using raw Python lists and dicts.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple


class StrategyArchive:
    """Records strategy outcomes and provides data-driven recommendations.

    Internal storage: list of dicts {strategy, level, success, gen, timestamp}.
    Recency-weighted success rates give more weight to recent results.

    Optimization: maintains level+strategy index counts for fast
    UCB exploration_bonus computation, avoiding O(N) scans on every
    recommend_strategy call (called once per challenger per generation).

    Also maintains per-strategy record indices for O(1) filtered access
    in success_rate(), avoiding full-record-list scans.
    """

    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self._records: List[Dict[str, Any]] = []
        self._loaded_from: Optional[str] = None
        # Level+strategy indexed counts for O(1) UCB bonus queries.
        self._level_strategy_counts: Dict[Tuple[int, str], int] = {}
        self._level_total: Dict[int, int] = {}
        self._global_total: int = 0
        # Per-strategy record index: strategy -> set of record positions.
        # Enables O(1) filtered access in success_rate instead of O(N) scan.
        self._strategy_record_idx: Dict[str, set] = {}
        # R13: Level-to-strategies index: level -> set of strategy names.
        # Avoids O(K) scan of _level_strategy_counts keys in best_for_level.
        self._level_strategies: Dict[int, set] = {}
        # Per-strategy total count across all levels (for O(1) global UCB bonus).
        # Avoids the O(N) sum over _level_strategy_counts keys in exploration_bonus
        # when level is None.
        self._strategy_total: Dict[str, int] = {}
        # Lazily imported math.sqrt for exploration_bonus
        self._sqrt = None
        # Cached timestamp for record() calls (refreshed per batch via
        # refresh_timestamp(), avoids calling time.time() ~50x per generation)
        self._gen_timestamp: float = time.time()
        # R22: Cached set of all strategy names seen across all levels.
        # Avoids rebuilding from _level_strategies on every recommend_strategy
        # call (pop_size times per generation). Invalidated on record/trim.
        self._global_strategy_set: set = set()
        # Delta tracking: champion quality deltas per strategy for tiebreaking.
        self._champion_deltas: Dict[str, List[float]] = {}

    def refresh_timestamp(self) -> None:
        """Refresh the cached timestamp (call once per generation)."""
        self._gen_timestamp = time.time()

    # ── Record keeping ────────────────────────────────────────────────

    def record(
        self,
        strategy: str,
        level: int,
        success: bool,
        generation: int = 0,
    ) -> None:
        """Record the outcome of a strategy application.

        Args:
            strategy: Strategy name (e.g., 'RANDOM', 'TASK_AWARE', etc.).
            level:    Curriculum level number.
            success:  Whether the mutation led to an improved champion.
            generation: Generation number in the evolution run.
        """
        # Use pre-allocated dict template with setdefault to avoid
        # creating a new dict literal on every call (~50 per generation).
        self._records.append(
            {
                "strategy": strategy,
                "level": level,
                "success": success,
                "gen": generation,
                "timestamp": self._gen_timestamp,
            }
        )
        new_idx = len(self._records) - 1
        # Update level+strategy index
        key = (level, strategy)
        self._level_strategy_counts[key] = self._level_strategy_counts.get(key, 0) + 1
        self._level_total[level] = self._level_total.get(level, 0) + 1
        self._global_total += 1
        # Update per-strategy total count
        self._strategy_total[strategy] = self._strategy_total.get(strategy, 0) + 1
        # Update per-strategy record index
        strat_set = self._strategy_record_idx.get(strategy)
        if strat_set is None:
            self._strategy_record_idx[strategy] = {new_idx}
        else:
            strat_set.add(new_idx)
        # Update level-to-strategies index
        ls = self._level_strategies.get(level)
        if ls is None:
            self._level_strategies[level] = {strategy}
        else:
            ls.add(strategy)
        # R22: Maintain global strategy set cache
        self._global_strategy_set.add(strategy)
        # Trim old records -- slice FIRST, THEN rebuild indices
        # so indices are consistent with the final record set.
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]
            self._rebuild_indices()

    def record_batch(self, records: List[Dict[str, Any]]) -> None:
        """Bulk-record multiple outcomes.

        Uses list.extend for bulk append (single allocation) instead of
        per-record append. Updates indices incrementally and trims once.
        Uses cached _gen_timestamp (refreshed per generation) instead of
        calling time.time() per batch — avoids unnecessary syscall overhead.

        Optimization: when records already contain all required keys
        (strategy, level, success, gen), skips the dict recreation
        overhead — just stamps them in-place and extends directly.
        """
        now = self._gen_timestamp
        _needs_rebuild = False
        _lc = self._level_strategy_counts
        _lt = self._level_total
        _st = self._strategy_total
        _sri = self._strategy_record_idx
        _ls = self._level_strategies

        base_idx = len(self._records)

        for i, r in enumerate(records):
            # Fast path: record already has all keys (from _record_strategy_outcomes)
            if "strategy" in r and "level" in r and "success" in r and "gen" in r:
                if "timestamp" not in r:
                    r["timestamp"] = now
                strat = r["strategy"]
                lev = r["level"]
            else:
                # Slow path: reconstruct from partial dict
                strat = r.get("strategy", "UNKNOWN")
                lev = r.get("level", -1)
                r["strategy"] = strat
                r["level"] = lev
                r["success"] = bool(r.get("success", False))
                r["gen"] = r.get("gen", 0)
                r["timestamp"] = r.get("timestamp", now)

            self._records.append(r)
            new_idx = base_idx + i
            key = (lev, strat)
            _lc[key] = _lc.get(key, 0) + 1
            _lt[lev] = _lt.get(lev, 0) + 1
            self._global_total += 1
            _st[strat] = _st.get(strat, 0) + 1
            # R25: Update per-strategy record index for O(1) filtered access
            strat_set = _sri.get(strat)
            if strat_set is None:
                _sri[strat] = {new_idx}
            else:
                strat_set.add(new_idx)
            # R25: Update level-to-strategies index
            ls = _ls.get(lev)
            if ls is None:
                _ls[lev] = {strat}
            else:
                ls.add(strat)

        # R22: Rebuild global strategy set from level_strategies after bulk append
        self._global_strategy_set = set()
        for strats in self._level_strategies.values():
            self._global_strategy_set.update(strats)
        # R25: Single trim -- slice records FIRST, THEN rebuild indices
        # so indices are consistent with the final record set.
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]
            self._rebuild_indices()

    # ── Query ─────────────────────────────────────────────────────────

    def success_rate(
        self,
        strategy: str,
        window: int = 50,
        level: Optional[int] = None,
        use_laplace: bool = True,
    ) -> float:
        """Compute recency-weighted success rate for a strategy.

        Uses linear recency weighting: the most recent record gets weight
        *window*, the oldest in the window gets weight 1.

        v11: Laplace smoothing (add 1 to numerator and denominator) prevents
        low-count strategies from getting extreme rates (e.g., PERTURBATION_CAPPED with
        20/20 -> 0.954 instead of 1.0). This avoids archive always preferring
        PERTURBATION_CAPPED just because it's been tried fewer times.

        R21: Avoids list comprehension filtering when level is None.
        When level is None, iterates sorted indices directly without
        creating an intermediate list. When level is specified, uses a
        generator expression instead of list comprehension to avoid
        allocating a full filtered list just to take the last N elements.

        Args:
            strategy: Strategy name.
            window:   Number of most recent records to consider.
            level:    Optional level filter.
            use_laplace: Apply Laplace smoothing (default: True).

        Returns:
            Float in [0.0, 1.0] representing weighted success rate,
            or 0.0 if no records in window.
        """
        # Filter using per-strategy index for O(K) where K = records of
        # this strategy (much smaller than full N).
        strat_indices = self._strategy_record_idx.get(strategy)
        if strat_indices is None or not strat_indices:
            return 0.0

        records = self._records
        n_records = len(records)

        # R21: For level=None, iterate indices directly without list alloc.
        # For level!=None, collect+score in one pass (R24: eliminated the
        # previous two-pass count-then-skip pattern).
        if level is not None:
            # R24: Single pass -- collect all matching records, trim to
            # window, then score. Eliminates the O(N) count pass that
            # preceded the scoring loop in the old implementation.
            idx_list = sorted(strat_indices)
            matches = []
            append = matches.append
            for i in idx_list:
                if i >= n_records:
                    continue
                r = records[i]
                if r["level"] == level:
                    append(r)
            if not matches:
                return 0.0
            # Keep only the last `window` entries (most recent)
            start = max(0, len(matches) - window)
            matches = matches[start:]

            # Score in a single pass over the (up to window-sized) list
            count = 0
            total_weight = 0.0
            weighted_success = 0.0
            for r in matches:
                count += 1
                w = count  # linear weight: 1..n
                total_weight += w
                if r["success"]:
                    weighted_success += w
            n = count
        else:
            # R21: For no level filter, iterate indices sorted (which
            # preserves recency order since indices are appended in order).
            sorted_indices = sorted(strat_indices)
            # Take only the last *window* indices (most recent)
            start = max(0, len(sorted_indices) - window)
            n = 0
            total_weight = 0.0
            weighted_success = 0.0
            for i in sorted_indices[start:]:
                if i >= n_records:
                    continue
                n += 1
                w = n
                total_weight += w
                if records[i]["success"]:
                    weighted_success += w

        if total_weight <= 0:
            return 0.0

        raw_rate = weighted_success / total_weight

        if use_laplace and n > 0:
            # Laplace smoothing: (successes + 1) / (attempts + 2)
            # Pulls low-count strategies toward 0.5, preventing extreme bias
            smoothed = (weighted_success + 1.0) / (total_weight + 2.0)
            return smoothed

        return raw_rate

    def time_weighted_success_rate(self, strategy: str, decay_halflife: int = 100) -> float:
        """Compute success rate with exponential time-decay.

        Recent records count more than old ones using exponential decay.
        Formula: weight = exp(-0.693 * age_records / decay_halflife)
        where age_records is distance from end of records list (0 = most recent).

        Args:
            strategy: Strategy name.
            decay_halflife: Number of records after which weight halves
                (default 100). Smaller values emphasize recency more.

        Returns:
            Float in [0.0, 1.0] representing time-weighted success rate,
            or 0.0 if no records exist for this strategy.
        """
        import math

        strat_indices = self._strategy_record_idx.get(strategy)
        if not strat_indices:
            return 0.0

        records = self._records
        n_records = len(records)
        if n_records == 0:
            return 0.0

        total_weight = 0.0
        weighted_success = 0.0
        for i in sorted(strat_indices):
            if i >= n_records:
                continue
            age = n_records - 1 - i
            weight = math.exp(-0.693 * age / decay_halflife)
            total_weight += weight
            if records[i]["success"]:
                weighted_success += weight

        if total_weight <= 0:
            return 0.0

        return weighted_success / total_weight

    def exploration_bonus(
        self,
        strategy: str,
        level: Optional[int] = None,
    ) -> float:
        """Upper Confidence Bound exploration bonus for a strategy.

        UCB = sqrt(2 * ln(N) / n_i)
        where N = total attempts of all strategies at this level
        and n_i = attempts of this strategy at this level.

        High for rarely-tried strategies, encouraging exploration.
        Added to the success rate during best_for_level selection.

        Uses pre-built index counts for O(1) queries instead of O(N) scans.
        """
        # O(1) count via indexed totals
        if level is not None:
            N = self._level_total.get(level, 0)
        else:
            N = self._global_total
        if N == 0:
            return 0.0

        # O(1) count via indexed totals
        if level is not None:
            n_i = self._level_strategy_counts.get((level, strategy), 0)
        else:
            # O(1) lookup via per-strategy total count
            n_i = self._strategy_total.get(strategy, 0)
        if n_i == 0:
            return 1.0  # Maximum bonus for untried strategies

        if self._sqrt is None:
            import math

            self._sqrt = math.sqrt
            self._log_fn = math.log
        bonus = self._sqrt(2.0 * self._log_fn(N) / n_i)
        return min(bonus, 1.0)  # Cap at 1.0 to avoid dominating the rate

    def best_for_level(self, level: int, skip_strategy: Optional[str] = None) -> Optional[Tuple[str, float]]:
        """Return (strategy_name, success_rate) with highest success rate
        for the given level.

        Uses pre-built _level_strategies index for O(K) enumeration where K
        is the number of distinct strategies seen at this level, instead
        of O(N) over all records.

        R21: Avoids set subtraction (strategies_used - {skip_strategy}) on
        every call by using inline skip check during iteration instead.

        Args:
            level: Curriculum level number.
            skip_strategy: If provided, this strategy is excluded from
                consideration (to prevent monoculture).

        Returns None if no records exist for this level.
        """
        # O(K) enumeration via _level_strategies index (direct lookup)
        strategies_used = self._level_strategies.get(level)
        if not strategies_used:
            return None

        # R21: Two-pass: count valid strategies first to set exploration_weight,
        # then single scoring pass. Avoids set subtraction overhead.
        n_valid = sum(1 for s in strategies_used if s != skip_strategy)
        if n_valid == 0:
            return None
        exploration_weight = 0.15 if n_valid >= 3 else 0.25

        best_strat = None
        best_score = -float("inf")
        second_strat = None
        second_score = -float("inf")
        for s in strategies_used:
            if s == skip_strategy:
                continue
            rate = self.success_rate(s, window=50, level=level)
            bonus = self.exploration_bonus(s, level=level)
            score = rate + (bonus * exploration_weight)
            if score > best_score:
                second_strat = best_strat
                second_score = best_score
                best_score = score
                best_strat = s
            elif score > second_score:
                second_strat = s
                second_score = score

        if best_strat is None:
            return None

        # Delta tiebreaker: when top 2 are within 5%, prefer higher avg_delta
        if (
            second_strat is not None
            and best_score - second_score < 0.05
            and best_strat is not None
            and second_strat is not None
        ):
            d_best = self.avg_delta(best_strat)
            d_second = self.avg_delta(second_strat)
            if d_second > d_best:
                return (second_strat, second_score)

        return (best_strat, best_score)

    def recommend_strategy(self, level: int, skip_strategy: Optional[str] = None) -> str:
        """Return the best strategy name for this level.

        Falls back to:
          1. Best strategy for this level (if records exist)
          2. Best strategy across all levels (global)
          3. 'RANDOM' as the default

        R13: Uses _level_strategies for O(K) global enumeration instead of
        iterating all _level_strategy_counts keys.
        R22: Uses cached _global_strategy_set to avoid rebuilding union on
        every call (called pop_size times per generation).
        """
        # Try level-specific
        best = self.best_for_level(level, skip_strategy=skip_strategy)
        if best is not None and best[1] > 0.0:
            return best[0]

        # Try global best using cached union set (R22)
        # Skip building set each time — just iterate cached set
        candidates = self._global_strategy_set
        if not candidates:
            return "RANDOM"

        # For skip_strategy, iterate inline (avoids set.discard/add overhead
        # and avoids mutating the shared cache)
        best_global = None
        best_rate = -1.0
        second_global = None
        second_rate = -1.0
        for s in candidates:
            if s == skip_strategy:
                continue
            rate = self.success_rate(s, window=100)
            if rate > best_rate:
                second_global = best_global
                second_rate = best_rate
                best_rate = rate
                best_global = s
            elif rate > second_rate:
                second_global = s
                second_rate = rate

        if best_global is not None and best_rate > 0.0:
            # Delta tiebreaker: when top 2 are within 5%, prefer higher avg_delta
            if (
                second_global is not None
                and best_rate - second_rate < 0.05
                and best_global is not None
                and second_global is not None
            ):
                d_best = self.avg_delta(best_global)
                d_second = self.avg_delta(second_global)
                if d_second > d_best:
                    return second_global
            return best_global

        return "RANDOM"

    def total_records(self) -> int:
        return len(self._records)

    def records_by_level(self, level: int) -> List[Dict[str, Any]]:
        """Get all records for a specific level.

        Uses the _level_strategies index to enumerate strategies at this level,
        then uses per-strategy indices for filtered access.
        """
        result: List[Dict[str, Any]] = []
        records = self._records
        strategies = self._level_strategies.get(level)
        if not strategies:
            return result
        for strat in strategies:
            indices = self._strategy_record_idx.get(strat)
            if indices:
                for i in indices:
                    if i < len(records) and records[i]["level"] == level:
                        result.append(records[i])
        return result

    def records_by_strategy(self, strategy: str) -> List[Dict[str, Any]]:
        """Get all records for a specific strategy.

        Uses the per-strategy index for O(K) access instead of O(N) scan.
        """
        indices = self._strategy_record_idx.get(strategy)
        if not indices:
            return []
        records = self._records
        return [records[i] for i in indices if i < len(records)]

    def get_strategy_history(self, strategy: str, level: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return recent records for a specific strategy."""
        result: List[Dict[str, Any]] = []
        for r in reversed(self._records):
            if r["strategy"] != strategy:
                continue
            if level is not None and r["level"] != level:
                continue
            result.append(r)
            if len(result) >= 20:
                break
        return result

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of all recorded strategies.

        Uses _strategy_total for counts (O(K) vs O(N) full scan) but still
        needs a single pass for successes and levels (these aren't indexed).
        """
        # Use _strategy_total for counts (avoids counting from records)
        strategies: Dict[str, Dict[str, Any]] = {}
        for strat, count in self._strategy_total.items():
            strategies[strat] = {"count": count, "successes": 0, "levels": set()}

        # Single pass for successes and levels
        for r in self._records:
            s = r["strategy"]
            entry = strategies.get(s)
            if entry is None:
                entry = {"count": 0, "successes": 0, "levels": set()}
                strategies[s] = entry
            if r["success"]:
                entry["successes"] += 1
            entry["levels"].add(r["level"])

        return {
            "total_records": len(self._records),
            "strategies": {
                name: {
                    "count": info["count"],
                    "successes": info["successes"],
                    "rate": info["successes"] / info["count"] if info["count"] > 0 else 0.0,
                    "levels": sorted(info["levels"]),
                    "time_weighted_success_rate": self.time_weighted_success_rate(name),
                }
                for name, info in strategies.items()
            },
            "dominance": self.get_dominance_report(),
        }

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist to JSON file using atomic write (temp + rename).

        Prevents corruption if the process crashes mid-write during a long
        evolution run.  Uses the same tempfile + os.replace pattern as
        ChampionAnchor.save() and MetaController.save_state().
        """
        import tempfile

        data = {
            "version": 1,
            "max_records": self.max_records,
            "records": self._records,
        }
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except (OSError, PermissionError):
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self, path: str) -> bool:
        """Load from JSON file.  Returns True if loaded successfully.

        Rebuilds level+strategy indices after loading for O(1) UCB queries.
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._records = data.get("records", [])
            self.max_records = data.get("max_records", self.max_records)
            self._loaded_from = path
            self._rebuild_indices()
            return True
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False

    def _rebuild_indices(self) -> None:
        """Rebuild all indices from scratch.

        Called after bulk operations (load, trim) to ensure index consistency.
        O(N) over all records, amortized over many queries.
        """
        self._level_strategy_counts.clear()
        self._level_total.clear()
        self._strategy_record_idx.clear()
        self._level_strategies.clear()
        self._strategy_total.clear()
        self._global_total = len(self._records)
        for i, r in enumerate(self._records):
            lev = r["level"]
            strat = r["strategy"]
            key = (lev, strat)
            self._level_strategy_counts[key] = self._level_strategy_counts.get(key, 0) + 1
            self._level_total[lev] = self._level_total.get(lev, 0) + 1
            self._strategy_total[strat] = self._strategy_total.get(strat, 0) + 1
            strat_set = self._strategy_record_idx.get(strat)
            if strat_set is None:
                self._strategy_record_idx[strat] = {i}
            else:
                strat_set.add(i)
            # Rebuild level-to-strategies index
            ls = self._level_strategies.get(lev)
            if ls is None:
                self._level_strategies[lev] = {strat}
            else:
                ls.add(strat)
        # R22: Rebuild global strategy set cache
        self._global_strategy_set = set()
        for strats in self._level_strategies.values():
            self._global_strategy_set.update(strats)

    def decay_level_records(
        self, level: int, factor: float = 0.5, rng: Optional[random.Random] = None, level_dependent: bool = False
    ) -> int:
        """Decay records at a given level by flipping `success` toward False.

        Called on curriculum level advance (D3 fix) to prevent lower-level
        strategy biases from polluting higher-level decisions.

        For each record at `level`: success is set to False with probability
        (1 - factor). This reduces the effective success count without
        deleting records (preserving recency information for UCB exploration).

        R16: Uses per-strategy record indices to iterate only records for
        strategies that appear at this level, avoiding a full N scan.

        Args:
            level: Curriculum level whose records to decay.
            factor: Retention probability -- 0.5 means ~50% of successes survive.
                    Ignored when level_dependent=True.
            rng: Optional seeded RNG for reproducibility. If None, uses
                the global random module (non-deterministic).
            level_dependent: If True, use level-dependent decay rates:
                L0-L4: 0.3, L5-L8: 0.5, L9+: 0.7.

        Returns:
            Number of records modified.
        """
        modified = 0
        rng_local = rng if rng is not None else random
        # Level-dependent decay overrides the factor parameter
        if level_dependent:
            if level <= 4:
                factor = 0.3
            elif level <= 8:
                factor = 0.5
            else:
                factor = 0.7
        records = self._records
        # Use _level_strategies index to iterate only strategies present at this level
        strategies_at_level = self._level_strategies.get(level)
        if not strategies_at_level:
            return 0
        for strat in strategies_at_level:
            indices = self._strategy_record_idx.get(strat)
            if not indices:
                continue
            for i in indices:
                if i < len(records) and records[i]["level"] == level and records[i]["success"]:
                    if rng_local.random() > factor:
                        records[i]["success"] = False
                        modified += 1
        return modified

    def clear(self) -> None:
        """Clear all records and reset indices."""
        self._records.clear()
        self._level_strategy_counts.clear()
        self._level_total.clear()
        self._strategy_record_idx.clear()
        self._level_strategies.clear()
        self._strategy_total.clear()
        self._global_total = 0
        self._global_strategy_set.clear()
        self._champion_deltas.clear()

    # ── Delta tracking ──────────────────────────────────────────────

    def record_delta(self, strategy: str, delta: float) -> None:
        """Record a champion quality delta for a strategy.

        Args:
            strategy: Strategy name.
            delta: Quality improvement delta (success_value - baseline).
        """
        deltas = self._champion_deltas.get(strategy)
        if deltas is None:
            self._champion_deltas[strategy] = [delta]
        else:
            deltas.append(delta)

    def avg_delta(self, strategy: str) -> float:
        """Return average champion quality delta for a strategy.

        Returns 0.0 if no deltas have been recorded for this strategy.
        """
        deltas = self._champion_deltas.get(strategy)
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    def per_level_effectiveness(self, strategy: str) -> Dict[int, float]:
        """Return effectiveness score per level for a given strategy.

        Effectiveness = successful_delta_count / total_records_at_level,
        where successful means avg_delta > 0 (i.e. the record succeeded).

        Args:
            strategy: Strategy name to evaluate.

        Returns:
            Dict mapping level -> effectiveness score in [0.0, 1.0].
            Empty dict if strategy has no records.
        """
        indices = self._strategy_record_idx.get(strategy)
        if not indices:
            return {}

        level_counts: Dict[int, int] = {}
        level_successes: Dict[int, int] = {}
        for idx in indices:
            rec = self._records[idx]
            lev = rec["level"]
            level_counts[lev] = level_counts.get(lev, 0) + 1
            if rec.get("success", False):
                level_successes[lev] = level_successes.get(lev, 0) + 1

        result: Dict[int, float] = {}
        for lev, total in level_counts.items():
            successful = level_successes.get(lev, 0)
            result[lev] = successful / total if total > 0 else 0.0
        return result

    def adaptive_exploration_bonus(
        self,
        strategy: str,
        level: int,
        base_bonus: float = 0.1,
    ) -> float:
        """Return an exploration bonus that increases for underperforming strategies.

        Uses per_level_effectiveness to gauge how well a strategy performs at
        the given level:
          - effectiveness < 0.3  ->  bonus = base_bonus * 2  (underperforming)
          - effectiveness > 0.7  ->  bonus = base_bonus * 0.5 (well-performing)
          - otherwise             ->  bonus = base_bonus

        Args:
            strategy: Strategy name.
            level: Curriculum level.
            base_bonus: Base exploration bonus value (default 0.1).

        Returns:
            Scaled exploration bonus for the strategy at the given level.
        """
        effectiveness = self.per_level_effectiveness(strategy).get(level, 0.0)
        if effectiveness < 0.3:
            return base_bonus * 2.0
        elif effectiveness > 0.7:
            return base_bonus * 0.5
        return base_bonus

    # ── Diversity ────────────────────────────────────────────────────

    def get_strategy_diversity(self, level: Optional[int] = None) -> float:
        """Compute Shannon entropy of strategy distribution, normalized to [0, 1].

        1.0 = perfectly uniform distribution (maximum diversity).
        0.0 = all records are one strategy (no diversity).

        Args:
            level: Optional level filter. If None, uses all records.
        """
        # Count records per strategy
        counts: Dict[str, int] = {}
        if level is not None:
            strategies = self._level_strategies.get(level)
            if not strategies:
                return 0.0
            for strat in strategies:
                idx = self._level_strategy_counts.get((level, strat), 0)
                if idx > 0:
                    counts[strat] = idx
        else:
            for strat, cnt in self._strategy_total.items():
                if cnt > 0:
                    counts[strat] = cnt

        total = sum(counts.values())
        if total == 0:
            return 0.0

        import math

        entropy = 0.0
        for c in counts.values():
            if c > 0:
                p = c / total
                entropy -= p * math.log(p)

        max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_dominance_report(self) -> Dict[str, Any]:
        """Detect monoculture in strategy usage.

        Returns dict with:
            dominant_strategy: strategy with highest success_rate
            dominance_ratio: its success_rate / total of all success_rates
            is_monoculture: True if dominance_ratio > 0.6
            consecutive_usage: int from self._consecutive_same_strategy
                if available, else 0
            recommendation: 'explore' if monoculture else 'balanced'
        """
        strategies = self._global_strategy_set
        if not strategies:
            return {
                "dominant_strategy": None,
                "dominance_ratio": 0.0,
                "is_monoculture": False,
                "consecutive_usage": 0,
                "recommendation": "balanced",
                "all_strategy_rates": {},
            }

        strat_rates: Dict[str, float] = {}
        for strat in strategies:
            strat_rates[strat] = self.success_rate(strat, window=100)

        dominant = max(strat_rates, key=strat_rates.get)
        total_rate = sum(strat_rates.values())
        dominance_ratio = strat_rates[dominant] / total_rate if total_rate > 0 else 0.0

        consecutive = getattr(self, "_consecutive_same_strategy", 0)

        is_monoculture = dominance_ratio > 0.6

        return {
            "dominant_strategy": dominant,
            "dominance_ratio": dominance_ratio,
            "is_monoculture": is_monoculture,
            "consecutive_usage": consecutive,
            "recommendation": "explore" if is_monoculture else "balanced",
            "all_strategy_rates": strat_rates,
        }

    # ── Archive summary ─────────────────────────────────────────────

    def get_archive_summary(self) -> Dict[str, Any]:
        """Return a comprehensive archive summary.

        Returns dict with: total_records, unique_strategies, levels_covered,
        global_diversity, per_level_diversity, top_strategies.
        """
        global_div = self.get_strategy_diversity()

        per_level_div: Dict[int, float] = {}
        for lev in self._level_strategies:
            per_level_div[lev] = self.get_strategy_diversity(level=lev)

        # Top strategies: best-per-strategy global success rate, up to 5
        strat_rates: List[Tuple[str, float]] = []
        for strat in self._global_strategy_set:
            rate = self.success_rate(strat, window=100)
            strat_rates.append((strat, rate))
        strat_rates.sort(key=lambda x: x[1], reverse=True)
        top_strategies = strat_rates[:5]

        return {
            "total_records": len(self._records),
            "unique_strategies": len(self._global_strategy_set),
            "levels_covered": sorted(self._level_strategies.keys()),
            "global_diversity": global_div,
            "per_level_diversity": per_level_div,
            "top_strategies": top_strategies,
        }
