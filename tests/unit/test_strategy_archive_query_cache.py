"""Strategy success_rate stays correct with ordered indices and query cache."""

from strategy_archive import StrategyArchive


def test_success_rate_matches_after_batch_and_repeat_queries():
    archive = StrategyArchive(max_records=500)
    archive.record_batch(
        [{"strategy": "RANDOM", "level": 30, "success": True, "gen": i} for i in range(40)]
        + [{"strategy": "RANDOM", "level": 30, "success": False, "gen": 40 + i} for i in range(10)]
    )
    first = archive.success_rate("RANDOM", window=20, level=30)
    second = archive.success_rate("RANDOM", window=20, level=30)
    assert first == second
    assert 0.0 < first < 1.0
    archive.record("RANDOM", 30, True, generation=99)
    after = archive.success_rate("RANDOM", window=20, level=30)
    assert after != first


def test_recommend_after_trim_uses_lists():
    archive = StrategyArchive(max_records=80)
    for i in range(200):
        archive.record("TERMINAL_VERIFY_PROBE" if i % 3 else "RANDOM", 30, i % 2 == 0, i)
    rec = archive.recommend_strategy(30)
    assert rec in {"TERMINAL_VERIFY_PROBE", "RANDOM"}
    assert archive.total_records() == 80
