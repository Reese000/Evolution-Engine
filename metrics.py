#!/usr/bin/env python3
"""metrics.py — Cycle metrics collection and trend analysis.

Writes cycle metrics to JSONL for analysis and provides query
methods for history retrieval and trend computation.

Usage:
    from metrics import MetricsCollector

    mc = MetricsCollector("metrics/cycles.jsonl")
    mc.record_cycle({"cycle": 1, "fitness": 0.5, "level": 2})
    history = mc.get_history(last_n=10)
    trend = mc.get_trend("fitness", window=5)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


class MetricsCollector:
    """Writes cycle metrics to JSONL and provides query/trend methods.

    Each cycle's metrics are stored as one JSON line, making the file
    append-friendly and easy to tail for live monitoring.

    Caches the last get_history result to avoid re-reading the entire file
    on repeated trend queries within the same cycle.
    """

    def __init__(self, path: str = "metrics/cycles.jsonl") -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Cache for get_history to avoid re-parsing JSONL within same cycle
        self._cached_records: list[dict[str, Any]] | None = None
        self._cached_file_mtime: float = 0.0

    def record_cycle(self, cycle_data: dict[str, Any]) -> None:
        """Append one cycle's metrics as a JSON line.

        Automatically adds a timestamp if not present. Invalidates history cache.
        """
        record = dict(cycle_data)
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        # Invalidate cache after write
        self._cached_records = None

    def get_history(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """Read back metrics history.

        Caches results based on file mtime to avoid re-reading the entire
        file on repeated calls (e.g., trend queries within same cycle).

        Args:
            last_n: If set, return only the last N records.

        Returns:
            List of metric dicts, oldest first.
        """
        if not os.path.isfile(self.path):
            return []

        file_mtime = os.path.getmtime(self.path)

        # Use cache if file hasn't changed
        if self._cached_records is not None and file_mtime == self._cached_file_mtime:
            records = self._cached_records
        else:
            records = []
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    try:
                        records.append(json.loads(line_stripped))
                    except json.JSONDecodeError:
                        continue
            self._cached_records = records
            self._cached_file_mtime = file_mtime

        if last_n is not None and last_n > 0:
            return records[-last_n:]
        return records

    def get_trend(self, metric: str, window: int = 10) -> dict[str, Any]:
        """Compute rolling average trend for a metric.

        R24: Merged value extraction + stats into a single pass, and
        computes the half-sums during the same scan (avoids a separate
        sum(values[mid:]) call).

        Args:
            metric: Key name in the cycle data dict (e.g., "fitness", "level").
            window: Number of recent records to include in the average.

        Returns:
            Dict with:
              - current: latest value
              - avg: rolling average over window
              - min: minimum in window
              - max: maximum in window
              - trend: "improving" / "declining" / "stable"
              - window_size: actual number of points used
        """
        records = self.get_history(last_n=window)

        n = 0
        total = 0.0
        mn = float("inf")
        mx = float("-inf")
        first_val = None
        last_val = None
        # R24: Compute first-half and second-half sums during the same pass
        # that computes min/max/total. This eliminates the separate
        # sum(values[mid:]) call.
        first_half_sum = 0.0
        second_half_sum = 0.0

        # We need to know n first to compute mid, so collect values in
        # a small buffer, then process. But since window is typically
        # small (10), a two-pass approach with a tiny list is fine.
        values: list[float] = []
        append = values.append
        for r in records:
            v = r.get(metric)
            if isinstance(v, (int, float)):
                append(float(v))

        n = len(values)
        if n == 0:
            return {
                "current": None,
                "avg": None,
                "min": None,
                "max": None,
                "trend": "unknown",
                "window_size": 0,
            }

        current = values[-1]
        # R24: Single pass for total, min, max, and half-sums
        mid = n // 2
        for i in range(n):
            v = values[i]
            total += v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
            if i < mid:
                first_half_sum += v
            else:
                second_half_sum += v
        avg = total / n

        # Determine trend: compare first half average to second half average
        if n >= 4:
            first_half_avg = first_half_sum / mid
            second_half_avg = second_half_sum / (n - mid)
            delta = second_half_avg - first_half_avg
            threshold = max(abs(first_half_avg) * 0.01, 0.001)
            if delta > threshold:
                trend = "improving"
            elif delta < -threshold:
                trend = "declining"
            else:
                trend = "stable"
        elif n >= 2:
            delta = values[-1] - values[0]
            if abs(delta) < max(abs(values[0]) * 0.01, 0.001):
                trend = "stable"
            elif delta > 0:
                trend = "improving"
            else:
                trend = "declining"
        else:
            trend = "stable"

        return {
            "current": current,
            "avg": round(avg, 6),
            "min": mn,
            "max": mx,
            "trend": trend,
            "window_size": n,
        }

    def count_records(self) -> int:
        """Return total number of metric records.

        R16: Uses sum(1 for line in f) to avoid loading the entire file
        into memory. For large JSONL files (100K+ lines) this avoids
        allocating a potentially large string.
        """
        if not os.path.isfile(self.path):
            return 0
        count = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def clear(self) -> None:
        """Remove the metrics file (for testing or reset)."""
        if os.path.isfile(self.path):
            os.remove(self.path)
        self._cached_records = None
