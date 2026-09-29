"""Shared isolation fixtures for unit tests."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_eq_refine_late_merged_disk_cache(tmp_path, monkeypatch):
    """Keep eq-refine late-merged cache disk I/O inside tmp_path.

    Production save_eq_refine_late_merged_cache() writes unconditionally after
    a late-inject merge, so any test exercising that path would otherwise write
    fake seeds keyed by the REAL champion anchor/plateau into
    .agents/scan_floor_eq_refine_late_merged_cache.json (observed 2026-09-24
    06:55Z) — state a future production load could hydrate. Tests that patch
    the path themselves (e.g. the disk roundtrip test) still win: their
    setattr runs after this fixture on the same function-scoped monkeypatch.
    """
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_eq_refine_late_merged_cache_path",
        lambda: str(tmp_path / "scan_floor_eq_refine_late_merged_cache.json"),
    )
