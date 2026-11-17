"""Tests for archive sidecar persistence."""

import json
import os

import pytest

from evolution.archive_persistence import (
    cap_strategy_archive_state,
    load_archive_from_result,
    load_archive_sidecar,
    save_archive_sidecar,
    slim_result_for_jsonl,
)


@pytest.fixture(autouse=True)
def _pinned_sidecar_env(monkeypatch):
    """Pin the ranking knobs to their documented defaults.

    The assertions below encode EE_ARCHIVE_SIDECAR_MIN_RECORDS (400) and
    EE_ARCHIVE_SIDECAR_MAX_CYCLE (7000) behaviour, so an ambient value in the
    shell must not be able to flip a result.
    """
    monkeypatch.setenv("EE_ARCHIVE_SIDECAR_MIN_RECORDS", "400")
    monkeypatch.setenv("EE_ARCHIVE_SIDECAR_MAX_CYCLE", "7000")


def test_sidecar_roundtrip(tmp_path):
    root = str(tmp_path)
    blob = {"strategy": {"records": [{"a": 1}]}, "qd": {}, "novelty": {}}
    path = save_archive_sidecar(root, 42, blob)
    loaded = load_archive_sidecar(root, 42)
    assert loaded == blob
    assert path.endswith("cycle_00042.json")


def test_slim_result_strips_inline_archive():
    big = {"cycle": 1, "final_level": 17, "archive_state": {"strategy": {"x": 1}}}
    slim = slim_result_for_jsonl(big)
    assert "archive_state" not in slim
    assert slim["final_level"] == 17


def test_load_from_sidecar_path(tmp_path):
    root = str(tmp_path)
    blob = {"strategy": {}}
    path = save_archive_sidecar(root, 5, blob)
    loaded = load_archive_from_result(root, {"archive_state_path": path, "cycle": 5})
    assert loaded == blob


def test_cap_strategy_archive_state_keeps_newest():
    blob = {"strategy": {"records": [{"i": i} for i in range(5000)]}, "qd": {}}
    capped = cap_strategy_archive_state(blob, 3200)
    assert capped is not None
    assert len(capped["strategy"]["records"]) == 3200
    assert capped["strategy"]["records"][0]["i"] == 1800
    assert capped["strategy"]["records"][-1]["i"] == 4999


def test_load_prefers_richer_nearby_sidecar(tmp_path):
    root = str(tmp_path)
    thin = {"strategy": {"records": [{"i": i} for i in range(100)]}}
    rich = {"strategy": {"records": [{"i": i} for i in range(8150)]}}
    save_archive_sidecar(root, 10, rich)
    save_archive_sidecar(root, 11, thin)
    loaded = load_archive_sidecar(root)
    assert loaded is not None
    assert len(loaded["strategy"]["records"]) == 8150


def test_load_keeps_capped_latest_instead_of_richer(tmp_path):
    root = str(tmp_path)
    rich = {"strategy": {"records": [{"i": i} for i in range(8150)]}}
    capped = {"strategy": {"records": [{"i": i} for i in range(1500)]}}
    save_archive_sidecar(root, 10, rich)
    save_archive_sidecar(root, 11, capped)
    loaded = load_archive_sidecar(root)
    assert loaded is not None
    assert len(loaded["strategy"]["records"]) == 1500


def _write_raw_sidecar(root: str, cycle: int, records: int, tag: str | None = None) -> str:
    directory = os.path.join(root, ".agents", "archive_state")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"cycle_{cycle:05d}.json")
    recs = [{"i": i, "tag": tag} if tag is not None else {"i": i} for i in range(records)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"strategy": {"records": recs}}, f)
    return path


def test_dangling_latest_restores_live_sidecar_not_the_richest_old_one(tmp_path):
    """Live incident (2026-09-30): dangling latest.json + a 16-day-old uncapped
    sidecar (4890 records) + a healthy live capped one (1500 records).

    Both sit inside the `ref + 50` window, so richest-wins restored the Sep-14
    archive on resume. Recency must outrank raw record count: the newest sidecar
    at or above EE_ARCHIVE_SIDECAR_MIN_RECORDS wins, richest is only a last
    resort when nothing on disk clears the floor.
    """
    root = str(tmp_path)
    directory = os.path.join(root, ".agents", "archive_state")
    _write_raw_sidecar(root, 6999, 4890, tag="old")  # Sep-14 era, uncapped
    _write_raw_sidecar(root, 7604, 1500, tag="live")  # live, terminal-capped
    deleted = os.path.join(directory, "cycle_07602.json")  # latest.json's target
    with open(os.path.join(directory, "latest.json"), "w", encoding="utf-8") as f:
        json.dump({"cycle": 7602, "path": deleted, "records": 1500}, f)

    loaded = load_archive_sidecar(root)

    assert loaded is not None
    assert len(loaded["strategy"]["records"]) == 1500
    assert loaded["strategy"]["records"][0]["tag"] == "live"
    # the incident's essential condition: latest.json's target is really gone
    # and these two are the only sidecars on disk
    assert sorted(name for name in os.listdir(directory) if name.startswith("cycle_")) == [
        "cycle_06999.json",
        "cycle_07604.json",
    ]


def test_fallback_uses_richest_only_when_no_sidecar_clears_the_floor(tmp_path):
    """Richest-wins survives as the last resort: every candidate is under
    EE_ARCHIVE_SIDECAR_MIN_RECORDS (400), so "newest usable" has no answer and
    the richest of a thin field is still the best available resume state.
    """
    root = str(tmp_path)
    directory = os.path.join(root, ".agents", "archive_state")
    _write_raw_sidecar(root, 10, 300, tag="thin-rich")
    _write_raw_sidecar(root, 11, 100, tag="thinner")
    deleted = os.path.join(directory, "cycle_00012.json")
    with open(os.path.join(directory, "latest.json"), "w", encoding="utf-8") as f:
        json.dump({"cycle": 12, "path": deleted, "records": 100}, f)

    loaded = load_archive_sidecar(root)

    assert loaded is not None
    assert len(loaded["strategy"]["records"]) == 300
    assert loaded["strategy"]["records"][0]["tag"] == "thin-rich"


def test_fallback_skips_sidecars_that_are_not_json_objects(tmp_path):
    """A sidecar whose JSON parses to a list must not crash resume.

    The fallback ranks every sidecar file on disk, so one odd-but-valid file
    used to raise AttributeError out of `_strategy_count` and take the whole
    resume path down with it.
    """
    root = str(tmp_path)
    directory = os.path.join(root, ".agents", "archive_state")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "cycle_07603.json"), "w", encoding="utf-8") as f:
        json.dump([1, 2, 3], f)
    _write_raw_sidecar(root, 7604, 1500, tag="live")
    with open(os.path.join(directory, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "cycle": 7602,
                "path": os.path.join(directory, "cycle_07602.json"),
                "records": 1500,
            },
            f,
        )

    loaded = load_archive_sidecar(root)

    assert loaded is not None
    assert len(loaded["strategy"]["records"]) == 1500
    assert loaded["strategy"]["records"][0]["tag"] == "live"


def test_fallback_recovers_live_sidecar_above_static_cap(tmp_path):
    """Recovery must see cycles above EE_ARCHIVE_SIDECAR_MAX_CYCLE (7000).

    The live run crossed 7000 on 2026-09-14 and the daemon's restart prune has
    been deleting live sidecars since (observed 2026-09-30), so latest.json can
    dangle while live cycles sit above that stale cap. The old bare
    `cycle > cap` skip hid every one of them, and the 6-digit fixture (parsed
    with a fixed 5-char slice as cycle 10000) used to be the only candidate.
    """
    root = str(tmp_path)
    directory = os.path.join(root, ".agents", "archive_state")
    deleted = _write_raw_sidecar(root, 49998, 3000)  # latest.json's target
    live = _write_raw_sidecar(root, 49999, 1500)  # live, above the stale cap
    fixture = _write_raw_sidecar(root, 100000, 8150)  # far ahead of the live run
    os.remove(deleted)  # what the buggy prune did to latest.json's target
    with open(os.path.join(directory, "latest.json"), "w", encoding="utf-8") as f:
        json.dump({"cycle": 49998, "path": deleted}, f)

    loaded = load_archive_sidecar(root)

    assert loaded is not None
    assert len(loaded["strategy"]["records"]) == 1500
    assert os.path.basename(live).endswith("49999.json")
    assert os.path.basename(fixture).endswith("100000.json")


def test_load_survives_corrupt_latest_json(tmp_path):
    """latest.json that parses to garbage must not crash resume."""
    root = str(tmp_path)
    directory = os.path.join(root, ".agents", "archive_state")
    _write_raw_sidecar(root, 49999, 1500)
    latest = os.path.join(directory, "latest.json")
    for broken in ("not json at all", "[1, 2, 3]", '{"cycle": "abc", "path": 5}'):
        with open(latest, "w", encoding="utf-8") as f:
            f.write(broken)
        loaded = load_archive_sidecar(root)
        assert loaded is not None, broken
        assert len(loaded["strategy"]["records"]) == 1500, broken


def test_corrupt_latest_json_still_prefers_live_over_rich_old(tmp_path):
    """Unreadable latest.json makes the reference the newest cycle on disk; the
    16-day-old rich sidecar must still lose to the live one on that path."""
    root = str(tmp_path)
    directory = os.path.join(root, ".agents", "archive_state")
    _write_raw_sidecar(root, 6999, 4890, tag="old")
    _write_raw_sidecar(root, 7604, 1500, tag="live")
    with open(os.path.join(directory, "latest.json"), "w", encoding="utf-8") as f:
        f.write("not json at all")

    loaded = load_archive_sidecar(root)

    assert isinstance(loaded, dict)
    assert len(loaded["strategy"]["records"]) == 1500
    assert loaded["strategy"]["records"][0]["tag"] == "live"


def test_missing_latest_json_still_restores_the_newest_sidecar(tmp_path):
    """latest.json gone entirely must not restore NOTHING while sidecars are on
    disk: resume starts from the newest live archive, not from empty."""
    root = str(tmp_path)
    _write_raw_sidecar(root, 7603, 1500, tag="older")
    _write_raw_sidecar(root, 7604, 1500, tag="live")
    # no latest.json written at all

    loaded = load_archive_sidecar(root)

    assert isinstance(loaded, dict)
    assert len(loaded["strategy"]["records"]) == 1500
    assert loaded["strategy"]["records"][0]["tag"] == "live"


def test_sub_floor_latest_loses_to_a_sidecar_that_clears_the_floor(tmp_path):
    """A live sidecar under EE_ARCHIVE_SIDECAR_MIN_RECORDS is not restorable on
    its own, so an eligible candidate wins even when it is older and richer.

    Same stance as test_load_prefers_richer_nearby_sidecar (thin latest vs 8150
    older), pinned here with live-shaped cycles so the floor's intent — never
    resume from an unusably thin archive — stays explicit rather than implied.
    """
    root = str(tmp_path)
    directory = os.path.join(root, ".agents", "archive_state")
    _write_raw_sidecar(root, 6999, 4890, tag="old")
    thin = _write_raw_sidecar(root, 7604, 300, tag="live-thin")
    with open(os.path.join(directory, "latest.json"), "w", encoding="utf-8") as f:
        json.dump({"cycle": 7604, "path": thin, "records": 300}, f)

    loaded = load_archive_sidecar(root)

    assert isinstance(loaded, dict)
    assert len(loaded["strategy"]["records"]) == 4890
    assert loaded["strategy"]["records"][0]["tag"] == "old"


def test_non_dict_latest_target_is_replaced_not_returned(tmp_path):
    """latest.json's own target holding a JSON array must neither crash resume
    nor escape as a list — callers such as cap_strategy_archive_state call
    `.get` on whatever this returns."""
    root = str(tmp_path)
    directory = os.path.join(root, ".agents", "archive_state")
    os.makedirs(directory, exist_ok=True)
    bad = os.path.join(directory, "cycle_07603.json")
    with open(bad, "w", encoding="utf-8") as f:
        json.dump([1, 2, 3], f)
    _write_raw_sidecar(root, 7604, 1500, tag="live")
    with open(os.path.join(directory, "latest.json"), "w", encoding="utf-8") as f:
        json.dump({"cycle": 7603, "path": bad, "records": 1500}, f)

    loaded = load_archive_sidecar(root)

    assert isinstance(loaded, dict)
    assert loaded["strategy"]["records"][0]["tag"] == "live"


def test_explicit_cycle_non_dict_sidecar_falls_through(tmp_path):
    """load_archive_sidecar(root, N) on a corrupt cycle_N file must not hand a
    JSON array back: the pointer is dropped and the lookup keeps going."""
    root = str(tmp_path)
    directory = os.path.join(root, ".agents", "archive_state")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "cycle_00042.json"), "w", encoding="utf-8") as f:
        json.dump([1, 2, 3], f)
    _write_raw_sidecar(root, 7604, 1500, tag="live")

    loaded = load_archive_sidecar(root, 42)

    assert isinstance(loaded, dict)
    assert loaded["strategy"]["records"][0]["tag"] == "live"


def test_malformed_env_values_do_not_crash_resume(monkeypatch, tmp_path):
    """A hand-edited EE_ARCHIVE_SIDECAR_* must fall back to its default instead
    of raising ValueError out of the resume path.

    The dangling pointer below forces the fallback, so both knobs are read on
    this path: the record floor and the no-reference cycle cap.
    """
    root = str(tmp_path)
    directory = os.path.join(root, ".agents", "archive_state")
    _write_raw_sidecar(root, 7604, 1500, tag="live")
    with open(os.path.join(directory, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "cycle": 7603,
                "path": os.path.join(directory, "cycle_07603.json"),
                "records": 1500,
            },
            f,
        )
    monkeypatch.setenv("EE_ARCHIVE_SIDECAR_MIN_RECORDS", "not-a-number")
    monkeypatch.setenv("EE_ARCHIVE_SIDECAR_MAX_CYCLE", "7000.5")

    loaded = load_archive_sidecar(root)

    assert isinstance(loaded, dict)
    assert loaded["strategy"]["records"][0]["tag"] == "live"


def test_load_from_result_rejects_non_dict_blobs(tmp_path):
    """Legacy inline `archive_state` / `archive_state_path` blobs that are not
    JSON objects fall through to a usable sidecar instead of escaping as-is."""
    root = str(tmp_path)
    directory = os.path.join(root, ".agents", "archive_state")
    os.makedirs(directory, exist_ok=True)
    bad = os.path.join(directory, "cycle_00042.json")
    with open(bad, "w", encoding="utf-8") as f:
        json.dump([1, 2, 3], f)
    _write_raw_sidecar(root, 7604, 1500, tag="live")

    from_path = load_archive_from_result(root, {"archive_state_path": bad, "cycle": 42})
    from_inline = load_archive_from_result(root, {"archive_state": [1, 2, 3], "cycle": 42})

    assert isinstance(from_path, dict)
    assert from_path["strategy"]["records"][0]["tag"] == "live"
    assert isinstance(from_inline, dict)
    assert from_inline["strategy"]["records"][0]["tag"] == "live"
