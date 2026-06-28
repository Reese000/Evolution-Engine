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
    """

    def __init__(self, path: str = "metrics/cycles.jsonl") -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def record_cycle(self, cycle_data: dict[str, Any]) -> None:
        """Append one cycle's metrics as a JSON line.

        Automatically adds a timestamp if not present.
        """
        record = dict(cycle_data)
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_history(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """Read back metrics history.

        Args:
            last_n: If set, return only the last N records.

        Returns:
            List of metric dicts, oldest first.
        """
        records: list[dict[str, Any]] = []
        if not os.path.isfile(self.path):
            return records

        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if last_n is not None and last_n > 0:
            return records[-last_n:]
        return records

    def get_trend(self, metric: str, window: int = 10) -> dict[str, Any]:
        """Compute rolling average trend for a metric.

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
        values: list[float] = []
        for r in records:
            v = r.get(metric)
            if isinstance(v, (int, float)):
                values.append(float(v))

        if not values:
            return {
                "current": None,
                "avg": None,
                "min": None,
                "max": None,
                "trend": "unknown",
                "window_size": 0,
            }

        current = values[-1]
        avg = sum(values) / len(values)
        mn = min(values)
        mx = max(values)

        # Determine trend: compare first half average to second half average
        if len(values) >= 4:
            mid = len(values) // 2
            first_half_avg = sum(values[:mid]) / mid
            second_half_avg = sum(values[mid:]) / (len(values) - mid)
            delta = second_half_avg - first_half_avg
            threshold = max(abs(first_half_avg) * 0.01, 0.001)
            if delta > threshold:
                trend = "improving"
            elif delta < -threshold:
                trend = "declining"
            else:
                trend = "stable"
        elif len(values) >= 2:
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
            "window_size": len(values),
        }

    def count_records(self) -> int:
        """Return total number of metric records."""
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
