"""QD illumination/freshness must not be recomputed per program."""

from qd_archive import MAPElitesArchive


def test_illumination_diversity_cached_until_add():
    archive = MAPElitesArchive(bins=8)
    archive.add_solution([0.0, 0.0], 1.0, [1, 2], {"status": "ok"})
    first = archive.get_illumination_diversity()
    second = archive.get_illumination_diversity()
    assert first is second
    archive.add_solution([0.9, 0.9], 2.0, [3, 4], {"status": "ok"})
    third = archive.get_illumination_diversity()
    assert third is not first
    assert third["coverage_ratio"] >= first["coverage_ratio"]


def test_cell_freshness_cached_for_same_generation():
    archive = MAPElitesArchive(bins=8)
    archive.add_solution([0.1, 0.1], 1.0, [1], {"status": "ok"})
    a = archive.get_cell_freshness(10)
    b = archive.get_cell_freshness(10)
    assert a is b
    c = archive.get_cell_freshness(11)
    assert c is not a
