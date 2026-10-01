"""Tests for nudge-triggered reference champion reseed flags."""

import json
import os

from run_multi_cycle_loop import _flag_nudge_reference_reseed


def test_flag_nudge_reference_reseed_writes_tick_state(tmp_path, monkeypatch):
    agents_dir = tmp_path / ".agents"
    agents_dir.mkdir()
    tick_path = agents_dir / "tick_state.json"
    tick_path.write_text('{"cycle": 2741}', encoding="utf-8")

    monkeypatch.delenv("EE_FORCE_REFERENCE_CHAMPION", raising=False)

    _flag_nudge_reference_reseed(str(tmp_path), 20)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert tick["force_champion_reseed"] is True
    assert tick["force_reference_champion"] is True
    assert tick["reseed_level"] == 20
    assert tick["cycle"] == 2741
    assert os.environ.get("EE_FORCE_REFERENCE_CHAMPION") == "1"
