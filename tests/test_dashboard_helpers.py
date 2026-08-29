#!/usr/bin/env python3
"""Tests for pure helpers in v11_dashboard (audit HIGH-4: dashboard had no tests).

Covers the display/analysis helpers that don't require curses or a live engine:
fmt_time, compute_metrics_trend, format_val_loss_trajectory, and the JSON readers.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v11_dashboard import (
    compute_metrics_trend,
    fmt_time,
    format_val_loss_trajectory,
    read_health_json,
    read_metrics_jsonl,
)


class TestFmtTime:
    def test_sub_hour(self):
        assert fmt_time(0) == "00m00s"
        assert fmt_time(65) == "01m05s"
        assert fmt_time(599) == "09m59s"

    def test_with_hours(self):
        assert fmt_time(3661) == "1h01m01s"
        assert fmt_time(7325) == "2h02m05s"


class TestComputeMetricsTrend:
    def test_empty_records(self):
        out = compute_metrics_trend([], metric="fitness")
        assert out["trend"] == "unknown"
        assert out["current"] is None
        assert out["window_size"] == 0

    def test_improving(self):
        recs = [{"fitness": v} for v in (1, 2, 3, 4, 5, 6, 7, 8)]
        out = compute_metrics_trend(recs, metric="fitness", window=8)
        assert out["trend"] == "improving"
        assert out["current"] == 8.0
        assert out["max"] == 8.0
        assert out["min"] == 1.0

    def test_declining(self):
        recs = [{"fitness": v} for v in (8, 7, 6, 5, 4, 3, 2, 1)]
        out = compute_metrics_trend(recs, metric="fitness", window=8)
        assert out["trend"] == "declining"

    def test_stable(self):
        recs = [{"fitness": 5.0} for _ in range(8)]
        out = compute_metrics_trend(recs, metric="fitness", window=8)
        assert out["trend"] == "stable"

    def test_ignores_non_numeric_and_missing(self):
        recs = [{"fitness": "x"}, {"other": 1}, {"fitness": 2.0}, {"fitness": 3.0}]
        out = compute_metrics_trend(recs, metric="fitness")
        assert out["current"] == 3.0
        assert out["window_size"] == 2


class TestFormatValLossTrajectory:
    def test_insufficient_data(self):
        assert format_val_loss_trajectory([{"val_loss": 1.0}]) == "(insufficient data)"
        assert format_val_loss_trajectory([{"val_loss": None}, {"val_loss": None}]) == "(insufficient data)"

    def test_flat(self):
        out = format_val_loss_trajectory([{"val_loss": 2.0}, {"val_loss": 2.0}, {"val_loss": 2.0}])
        assert out == "flat at 2.0000"

    def test_trajectory_renders_bounds(self):
        entries = [{"val_loss": v} for v in (4.3, 4.0, 3.8, 3.6)]
        out = format_val_loss_trajectory(entries)
        assert out.startswith("3.6000 ")
        assert out.endswith(" 4.3000")


class TestJsonReaders:
    def test_read_health_json_roundtrip(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"status": "running", "cycles_completed": 3}, f)
            path = f.name
        try:
            out = read_health_json(path)
            assert out["status"] == "running"
            assert out["cycles_completed"] == 3
        finally:
            os.unlink(path)

    def test_read_health_json_missing_returns_none(self):
        assert read_health_json("/nonexistent/health.json") is None

    def test_read_metrics_jsonl_last_n(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                f.write(json.dumps({"cycle": i, "fitness": float(i)}) + "\n")
            path = f.name
        try:
            all_rows = read_metrics_jsonl(path)
            assert len(all_rows) == 5
            last2 = read_metrics_jsonl(path, last_n=2)
            assert [r["cycle"] for r in last2] == [3, 4]
        finally:
            os.unlink(path)

    def test_read_metrics_jsonl_missing_returns_empty(self):
        assert read_metrics_jsonl("/nonexistent/metrics.jsonl") == []
