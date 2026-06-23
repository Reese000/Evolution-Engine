#!/usr/bin/env python3
"""StrategyArchive — Record and recommend mutation strategies by level.

Pandas-free implementation using raw Python lists and dicts.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple


class StrategyArchive:
    """Records strategy outcomes and provides data-driven recommendations.

    Internal storage: list of dicts {strategy, level, success, gen, timestamp}.
    Recency-weighted success rates give more weight to recent results.
    """

    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self._records: List[Dict[str, Any]] = []
        self._loaded_from: Optional[str] = None

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
        self._records.append({
            "strategy": strategy,
            "level": level,
            "success": success,
            "gen": generation,
            "timestamp": time.time(),
        })
        # Trim old records
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

    def record_batch(self, records: List[Dict[str, Any]]) -> None:
        """Bulk-record multiple outcomes."""
        for r in records:
            self._records.append({
                "strategy": r.get("strategy", "UNKNOWN"),
                "level": r.get("level", -1),
                "success": bool(r.get("success", False)),
                "gen": r.get("gen", 0),
                "timestamp": r.get("timestamp", time.time()),
            })
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

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
        low-count strategies from getting extreme rates (e.g., CROSSOVER with
        20/20 → 0.954 instead of 1.0). This avoids archive always preferring
        CROSSOVER just because it's been tried fewer times.

        Args:
            strategy: Strategy name.
            window:   Number of most recent records to consider.
            level:    Optional level filter.
            use_laplace: Apply Laplace smoothing (default: True).

        Returns:
            Float in [0.0, 1.0] representing weighted success rate,
            or 0.0 if no records in window.
        """
        # Filter
        filtered = [
            r for r in self._records
            if r["strategy"] == strategy
            and (level is None or r["level"] == level)
        ]
        if not filtered:
            return 0.0

        # Take the most recent *window* records
        recent = filtered[-window:]

        # Recency weighting
        n = len(recent)
        total_weight = 0.0
        weighted_success = 0.0
        for i, r in enumerate(recent):
            w = i + 1.0  # linear weight: 1..n
            total_weight += w
            if r["success"]:
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
        """
        # Count total attempts of all strategies at this level
        all_at_level = [r for r in self._records if level is None or r["level"] == level]
        N = len(all_at_level)
        if N == 0:
            return 0.0

        # Count attempts of this strategy at this level
        strat_at_level = [
            r for r in all_at_level if r["strategy"] == strategy
        ]
        n_i = len(strat_at_level)
        if n_i == 0:
            return 1.0  # Maximum bonus for untried strategies

        import math
        bonus = math.sqrt(2.0 * math.log(N) / n_i)
        return min(bonus, 1.0)  # Cap at 1.0 to avoid dominating the rate

    def best_for_level(self, level: int, skip_strategy: Optional[str] = None) -> Optional[Tuple[str, float]]:
        """Return (strategy_name, success_rate) with highest success rate
        for the given level.

        Args:
            level: Curriculum level number.
            skip_strategy: If provided, this strategy is excluded from
                consideration (to prevent monoculture).

        Returns None if no records exist for this level.
        """
        strategies_used: set = set()
        for r in self._records:
            if r["level"] == level:
                strategies_used.add(r["strategy"])

        # Remove skipped strategy
        if skip_strategy and skip_strategy in strategies_used:
            strategies_used.discard(skip_strategy)

        if not strategies_used:
            return None

        best_strat = None
        best_score = -float("inf")
        for s in strategies_used:
            rate = self.success_rate(s, window=50, level=level)
            bonus = self.exploration_bonus(s, level=level)
            exploration_weight = 0.1  # 10% exploration bonus
            score = rate + (bonus * exploration_weight)
            if score > best_score:
                best_score = score
                best_strat = s

        if best_strat is None:
            return None
        return (best_strat, best_score)

    def recommend_strategy(self, level: int, skip_strategy: Optional[str] = None) -> str:
        """Return the best strategy name for this level.

        Falls back to:
          1. Best strategy for this level (if records exist)
          2. Best strategy across all levels (global)
          3. 'RANDOM' as the default

        Args:
            level: Curriculum level.
            skip_strategy: If provided, this strategy will be skipped
                (return next best) to prevent strategy monoculture.
        """
        # Try level-specific
        best = self.best_for_level(level, skip_strategy=skip_strategy)
        if best is not None and best[1] > 0.0:
            return best[0]

        # Try global best
        all_strategies: set = set()
        for r in self._records:
            all_strategies.add(r["strategy"])

        # Remove the skipped strategy if provided
        candidates = all_strategies - ({skip_strategy} if skip_strategy else set())

        if not candidates:
            return "RANDOM"

        best_global = None
        best_rate = -1.0
        for s in candidates:
            rate = self.success_rate(s, window=100)
            if rate > best_rate:
                best_rate = rate
                best_global = s

        if best_global is not None and best_rate > 0.0:
            return best_global

        return "RANDOM"

    def total_records(self) -> int:
        return len(self._records)

    def records_by_level(self, level: int) -> List[Dict[str, Any]]:
        return [r for r in self._records if r["level"] == level]

    def records_by_strategy(self, strategy: str) -> List[Dict[str, Any]]:
        return [r for r in self._records if r["strategy"] == strategy]

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of all recorded strategies."""
        strategies: Dict[str, Dict[str, Any]] = {}
        for r in self._records:
            s = r["strategy"]
            if s not in strategies:
                strategies[s] = {"count": 0, "successes": 0, "levels": set()}
            strategies[s]["count"] += 1
            if r["success"]:
                strategies[s]["successes"] += 1
            strategies[s]["levels"].add(r["level"])

        return {
            "total_records": len(self._records),
            "strategies": {
                name: {
                    "count": info["count"],
                    "successes": info["successes"],
                    "rate": info["successes"] / info["count"] if info["count"] > 0 else 0.0,
                    "levels": sorted(info["levels"]),
                }
                for name, info in strategies.items()
            }
        }

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist to JSON file."""
        data = {
            "version": 1,
            "max_records": self.max_records,
            "records": self._records,
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> bool:
        """Load from JSON file.  Returns True if loaded successfully."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._records = data.get("records", [])
            self.max_records = data.get("max_records", self.max_records)
            self._loaded_from = path
            return True
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()
