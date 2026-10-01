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


@pytest.fixture(autouse=True)
def _isolate_eq_refine_seeds_payload(tmp_path, monkeypatch):
    """Keep the production eq-refine seeds payload (.agents/scan_floor_eq_refine_seeds.json) untouched.

    Same leak class as the two fixtures below: record_scan_floor_eq_refine_seeds()
    writes unconditionally, and scan_floor_anchor_program(test_champ) resolves to
    the CERTIFIED 27-word scan-floor champion, so a test exercising inject/bootstrap
    rewrites the real payload keyed by the real anchor — observed clobbering the
    C7518 warm payload's cycle field (7518 -> 1, and 7518 -> 0 with plateau_key
    325.8257 -> 100.0 when fitness was stubbed) on 2026-09-29. Tests that patch
    the path themselves (e.g. the disk roundtrip test) still win: their setattr
    runs after this fixture on the same function-scoped monkeypatch.
    """
    monkeypatch.setattr(
        "evolution.mutation_helpers.scan_floor_eq_refine_seeds_path",
        lambda: str(tmp_path / "scan_floor_eq_refine_seeds.json"),
    )


@pytest.fixture(autouse=True)
def _isolate_coldstart_disk_caches(tmp_path, monkeypatch):
    """Keep l10-lift bootstrap / projection-seed disk caches inside tmp_path.

    Same production-state-leak class as the eq-refine late-merged fixture:
    the new cold-start save hooks persist anchor-keyed entries after a build,
    which a test run would otherwise write into .agents/ for a future
    production hydration (added with the t_5610b3e5 warm-up fix).
    """
    monkeypatch.setattr(
        "evolution.mutation_helpers.l10_lift_eq_bootstrap_cache_path",
        lambda: str(tmp_path / "scan_floor_l10_lift_eq_bootstrap_cache.json"),
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers.projection_seeds_cache_path",
        lambda: str(tmp_path / "scan_floor_projection_seeds_cache.json"),
    )
