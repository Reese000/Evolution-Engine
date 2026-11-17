"""Live strategy archive must stay capped during terminal cycles."""

from strategy_archive import StrategyArchive


def test_default_max_records_uses_terminal_cap(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_STRATEGY_ARCHIVE_CAP", "1500")
    monkeypatch.delenv("EE_STRATEGY_ARCHIVE_MAX_RECORDS", raising=False)
    archive = StrategyArchive()
    assert archive.max_records == 1500


def test_record_trims_to_max_records():
    archive = StrategyArchive(max_records=100)
    for i in range(250):
        archive.record("RANDOM", level=30, success=i % 2 == 0, generation=i)
    assert archive.total_records() == 100


def test_deserialize_clamps_oversized_sidecar(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_STRATEGY_ARCHIVE_CAP", "3200")
    monkeypatch.delenv("EE_STRATEGY_ARCHIVE_MAX_RECORDS", raising=False)
    blob = {
        "max_records": 10000,
        "records": [{"strategy": "RANDOM", "level": 30, "success": True, "gen": i} for i in range(5000)],
    }
    restored = StrategyArchive.deserialize(blob)
    assert restored.max_records == 3200
    assert restored.total_records() == 3200
    assert restored._records[0]["gen"] == 1800
