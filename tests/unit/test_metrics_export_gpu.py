"""GPU metrics helpers — graceful degradation without pynvml."""

from metrics_export import _get_gpu_utilization


def test_get_gpu_utilization_returns_float_without_pynvml():
    """Smoke path calls update_metrics; missing pynvml must not raise."""
    util = _get_gpu_utilization()
    assert isinstance(util, float)
    assert 0.0 <= util <= 100.0
