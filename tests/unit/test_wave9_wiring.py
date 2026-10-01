"""Wave 9 integration tests — pipeline wiring for progression tracker & metrics."""

from __future__ import annotations

from metrics import MetricsCollector
from progression_tracker import CycleMetrics, ProgressionTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cycle(cycle: int, fitness: float, **overrides) -> CycleMetrics:
    """Shorthand to build a CycleMetrics with sensible defaults."""
    defaults = {
        "cycle": cycle,
        "level": cycle,
        "fitness": fitness,
        "words": 100 + cycle * 10,
        "archive_size": cycle,
        "val_loss": None,
        "time_s": 1.0,
        "mg_count": cycle % 3,
        "total_challengers": 10,
        "champion_improvements": 0,
    }
    defaults.update(overrides)
    return CycleMetrics(**defaults)


# ===========================================================================
# 1. TestProgressionTrackerPlateau
# ===========================================================================


class TestProgressionTrackerPlateau:
    """Test plateau detection on ProgressionTracker.get_plateau_analysis."""

    def _tracker_with(self, path, cycles):
        t = ProgressionTracker(path)
        for cm in cycles:
            t.record(cm)
        return t

    # --- flat fitness → plateau ------------------------------------------
    def test_plateau_with_flat_fitness(self, tmp_path):
        path = str(tmp_path / "prog.jsonl")
        cycles = [_make_cycle(i, fitness=50.0) for i in range(1, 11)]
        tracker = self._tracker_with(path, cycles)
        result = tracker.get_plateau_analysis(window=10)
        assert result["status"] == "plateau"
        assert result["is_plateau"] is True
        assert result["plateau_duration"] >= 0

    # --- improving trend -------------------------------------------------
    def test_plateau_with_improving_trend(self, tmp_path):
        path = str(tmp_path / "prog.jsonl")
        cycles = [_make_cycle(i, fitness=float(i * 100)) for i in range(1, 11)]
        tracker = self._tracker_with(path, cycles)
        result = tracker.get_plateau_analysis(window=10)
        assert result["status"] == "improving"
        assert result["trend_slope"] > 0

    # --- insufficient data -----------------------------------------------
    def test_plateau_insufficient_data(self, tmp_path):
        path = str(tmp_path / "prog.jsonl")
        cycles = [_make_cycle(i, fitness=50.0) for i in range(1, 4)]
        tracker = self._tracker_with(path, cycles)
        result = tracker.get_plateau_analysis(window=10)
        assert result["status"] == "insufficient_data"
        assert result["cycles_available"] == 3


# ===========================================================================
# 2. TestProgressionTrackerWindows
# ===========================================================================


class TestProgressionTrackerWindows:
    """Test ProgressionTracker.compare_windows for metric comparison."""

    def _tracker_with(self, path, cycles):
        t = ProgressionTracker(path)
        for cm in cycles:
            t.record(cm)
        return t

    # --- improvement detected --------------------------------------------
    def test_compare_windows_shows_improvement(self, tmp_path):
        path = str(tmp_path / "prog.jsonl")
        old = [_make_cycle(i, fitness=10.0 + i * 0.5) for i in range(1, 6)]
        new = [_make_cycle(i + 5, fitness=30.0 + i * 0.5) for i in range(1, 6)]
        tracker = self._tracker_with(path, old + new)
        result = tracker.compare_windows("fitness", window_a=5, window_b=5)
        assert result["direction"] == "improving"
        assert result["improvement"] > 0

    # --- decline detected ------------------------------------------------
    def test_compare_windows_shows_decline(self, tmp_path):
        path = str(tmp_path / "prog.jsonl")
        old = [_make_cycle(i, fitness=50.0 + i * 0.5) for i in range(1, 6)]
        new = [_make_cycle(i + 5, fitness=30.0 + i * 0.5) for i in range(1, 6)]
        tracker = self._tracker_with(path, old + new)
        result = tracker.compare_windows("fitness", window_a=5, window_b=5)
        assert result["direction"] == "declining"
        assert result["improvement"] < 0

    # --- different metric (val_loss) -------------------------------------
    def test_compare_windows_different_metric(self, tmp_path):
        path = str(tmp_path / "prog.jsonl")
        old = [_make_cycle(i, fitness=10.0, val_loss=0.5) for i in range(1, 6)]
        new = [_make_cycle(i + 5, fitness=10.0, val_loss=0.2) for i in range(1, 6)]
        tracker = self._tracker_with(path, old + new)
        result = tracker.compare_windows("val_loss", window_a=5, window_b=5)
        assert result["metric"] == "val_loss"
        assert result["improvement"] < 0  # loss went down


# ===========================================================================
# 3. TestMetricsAnomalyDetection
# ===========================================================================


class TestMetricsAnomalyDetection:
    """Test MetricsCollector.detect_anomalies for outlier detection."""

    def _collector_with(self, path, records):
        mc = MetricsCollector(path)
        for r in records:
            mc.record_cycle(r)
        return mc

    # --- detects outliers -------------------------------------------------
    def test_detects_outliers(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        # Normal data around 10, then a huge spike
        normal = [{"cycle": i, "fitness": 10.0 + (i % 3) * 0.1} for i in range(20)]
        spike = [{"cycle": 20, "fitness": 1000.0}]
        mc = self._collector_with(path, normal + spike)
        anomalies = mc.detect_anomalies("fitness", window=20, threshold=2.0)
        # The spike at index 20 should be flagged
        assert len(anomalies) >= 1
        flagged_values = [a["value"] for a in anomalies]
        assert 1000.0 in flagged_values

    # --- no anomalies in normal data -------------------------------------
    def test_no_anomalies_in_normal_data(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        records = [{"cycle": i, "fitness": 10.0 + i * 0.1} for i in range(30)]
        mc = self._collector_with(path, records)
        anomalies = mc.detect_anomalies("fitness", window=20, threshold=2.0)
        assert anomalies == []

    # --- custom threshold -------------------------------------------------
    def test_custom_threshold(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        # Mild spike: 15.0 vs normal ~10.0
        normal = [{"cycle": i, "fitness": 10.0} for i in range(15)]
        mild_spike = [{"cycle": 15, "fitness": 15.0}]
        mc = self._collector_with(path, normal + mild_spike)
        # Very low threshold should catch mild spike
        anomalies_low = mc.detect_anomalies("fitness", window=20, threshold=0.5)
        assert len(anomalies_low) >= 1
        # Very high threshold should NOT catch it
        anomalies_high = mc.detect_anomalies("fitness", window=20, threshold=50.0)
        flagged_values = [a["value"] for a in anomalies_high]
        assert 15.0 not in flagged_values


# ===========================================================================
# 4. TestMetricsCorrelation
# ===========================================================================


class TestMetricsCorrelation:
    """Test MetricsCollector.correl_metrics for correlation analysis."""

    def _collector_with(self, path, records):
        mc = MetricsCollector(path)
        for r in records:
            mc.record_cycle(r)
        return mc

    # --- fitness vs generation (positively correlated) --------------------
    def test_fitness_vs_generation_correlation(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        records = [{"cycle": i, "fitness": float(i * 2), "generation": i} for i in range(1, 25)]
        mc = self._collector_with(path, records)
        result = mc.correlate_metrics("fitness", "generation", window=24)
        assert result["correlation"] > 0.95
        assert result["direction"] == "positive"
        assert result["strength"] == "strong"

    # --- uncorrelated metrics --------------------------------------------
    def test_uncorrelated_metrics(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        # Alternating fitness, monotonically increasing time_s
        records = [{"cycle": i, "fitness": 10.0 if i % 2 == 0 else 20.0, "time_s": float(i)} for i in range(1, 25)]
        mc = self._collector_with(path, records)
        result = mc.correlate_metrics("fitness", "time_s", window=24)
        # Correlation should be weak or none for alternating vs monotonic
        assert result["strength"] in ("none", "weak")

    # --- insufficient data -----------------------------------------------
    def test_insufficient_data(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        records = [{"cycle": 1, "fitness": 5.0, "generation": 1}]
        mc = self._collector_with(path, records)
        result = mc.correlate_metrics("fitness", "generation", window=20)
        assert result["sample_size"] < 2
        assert result["correlation"] == 0.0
        assert result["strength"] == "none"
