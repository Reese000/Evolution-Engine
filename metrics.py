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
from datetime import datetime, timezone
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
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

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

    def _extract_metric_values(self, records: list[dict[str, Any]], metric: str) -> list[float]:
        """Extract numeric values for a metric from records."""
        values: list[float] = []
        for r in records:
            v = r.get(metric)
            if isinstance(v, (int, float)):
                values.append(float(v))
        return values

    def _compute_trend(self, values: list[float]) -> str:
        """Determine trend direction from values."""
        n = len(values)
        if n >= 4:
            mid = n // 2
            first_half_sum = sum(values[:mid])
            second_half_sum = sum(values[mid:])
            first_half_avg = first_half_sum / mid
            second_half_avg = second_half_sum / (n - mid)
            delta = second_half_avg - first_half_avg
            threshold = max(abs(first_half_avg) * 0.01, 0.001)
            if delta > threshold:
                return "improving"
            elif delta < -threshold:
                return "declining"
            return "stable"
        elif n >= 2:
            delta = values[-1] - values[0]
            if abs(delta) < max(abs(values[0]) * 0.01, 0.001):
                return "stable"
            elif delta > 0:
                return "improving"
            return "declining"
        return "stable"

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
        values = self._extract_metric_values(records, metric)
        n = len(values)
        if n == 0:
            return {"current": None, "avg": None, "min": None, "max": None, "trend": "unknown", "window_size": 0}
        current = values[-1]
        total = 0.0
        mn = float("inf")
        mx = float("-inf")
        for v in values:
            total += v
            mn = min(mn, v)
            mx = max(mx, v)
        avg = total / n
        trend = self._compute_trend(values)
        return {"current": current, "avg": round(avg, 6), "min": mn, "max": mx, "trend": trend, "window_size": n}

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

    def detect_anomalies(self, metric: str, window: int = 20, threshold: float = 2.0) -> list[dict]:
        """Detect anomalous values in a metric using z-score analysis.

        Uses a rolling window to compute mean and standard deviation, then
        flags any value whose absolute z-score exceeds the threshold.

        Args:
            metric: Key name in the cycle data dict (e.g., "fitness", "level").
            window: Number of records to use for the rolling statistics.
            threshold: Z-score threshold for anomaly detection (default 2.0).

        Returns:
            List of anomaly dicts, each containing:
              - index: position in the full metric series
              - value: the anomalous value
              - z_score: computed z-score (absolute > threshold)
              - direction: 'high' if z_score > 0, 'low' if z_score < 0
              - timestamp: ISO timestamp of the record, or None
        """
        records = self.get_history()

        # Extract metric values and timestamps in order
        values: list[float] = []
        timestamps: list[str | None] = []
        for r in records:
            v = r.get(metric)
            if isinstance(v, (int, float)):
                values.append(float(v))
                timestamps.append(r.get("timestamp"))

        anomalies: list[dict] = []
        n = len(values)

        if n == 0:
            return anomalies

        # Compute rolling z-scores over the full series
        for i in range(n):
            # Determine the window slice: up to `window` most recent values ending at i
            start = max(0, i - window + 1)
            window_slice = values[start : i + 1]
            w = len(window_slice)

            if w < 2:
                # Need at least 2 points for a meaningful stddev
                continue

            mean = sum(window_slice) / w
            variance = sum((v - mean) ** 2 for v in window_slice) / w
            stddev = variance**0.5

            if stddev > 0:
                z_score = (values[i] - mean) / stddev
            else:
                z_score = 0.0

            if abs(z_score) > threshold:
                anomalies.append(
                    {
                        "index": i,
                        "value": values[i],
                        "z_score": round(z_score, 6),
                        "direction": "high" if z_score > 0 else "low",
                        "timestamp": timestamps[i],
                    }
                )

        return anomalies

    def _extract_paired_values(
        self, records: list[dict[str, Any]], metric_a: str, metric_b: str
    ) -> tuple[list[float], list[float]]:
        """Extract paired values for two metrics from records."""
        a_vals: list[float] = []
        b_vals: list[float] = []
        for r in records:
            va = r.get(metric_a)
            vb = r.get(metric_b)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                a_vals.append(float(va))
                b_vals.append(float(vb))
        return a_vals, b_vals

    @staticmethod
    def _pearson_correlation(a_vals: list[float], b_vals: list[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(a_vals)
        if n < 2:
            return 0.0
        mean_a = sum(a_vals) / n
        mean_b = sum(b_vals) / n
        sum_ab = 0.0
        sum_a2 = 0.0
        sum_b2 = 0.0
        for i in range(n):
            da = a_vals[i] - mean_a
            db = b_vals[i] - mean_b
            sum_ab += da * db
            sum_a2 += da * da
            sum_b2 += db * db
        denom = (sum_a2**0.5) * (sum_b2**0.5)
        r = sum_ab / denom if denom > 0 else 0.0
        return max(-1.0, min(1.0, r))

    def correlate_metrics(self, metric_a: str, metric_b: str, window: int = 20) -> dict[str, Any]:
        """Compute Pearson correlation between two metrics over a window.

        Extracts paired values for both metrics from the most recent records,
        then computes the Pearson correlation coefficient (r).

        Args:
            metric_a: First metric key name.
            metric_b: Second metric key name.
            window: Number of recent records to use (default 20).

        Returns:
            Dict with:
              - metric_a: name of first metric
              - metric_b: name of second metric
              - correlation: Pearson r (-1.0 to 1.0)
              - strength: 'strong' (|r| >= 0.7), 'moderate' (0.4 <= |r| < 0.7),
                          'weak' (0.2 <= |r| < 0.4), 'none' (|r| < 0.2)
              - direction: 'positive' (r > 0), 'negative' (r < 0), 'none' (r == 0)
              - sample_size: number of paired observations used
        """
        records = self.get_history(last_n=window)
        a_vals, b_vals = self._extract_paired_values(records, metric_a, metric_b)
        n = len(a_vals)
        if n < 2:
            return {
                "metric_a": metric_a,
                "metric_b": metric_b,
                "correlation": 0.0,
                "strength": "none",
                "direction": "none",
                "sample_size": n,
            }
        r = self._pearson_correlation(a_vals, b_vals)
        abs_r = abs(r)
        if abs_r >= 0.7:
            strength = "strong"
        elif abs_r >= 0.4:
            strength = "moderate"
        elif abs_r >= 0.2:
            strength = "weak"
        else:
            strength = "none"
        if r > 0:
            direction = "positive"
        elif r < 0:
            direction = "negative"
        else:
            direction = "none"
        return {
            "metric_a": metric_a,
            "metric_b": metric_b,
            "correlation": round(r, 6),
            "strength": strength,
            "direction": direction,
            "sample_size": n,
        }
