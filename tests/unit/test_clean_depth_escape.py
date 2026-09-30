"""Tests for clean L21 depth escape nudge."""

import json

from run_multi_cycle_loop import _maybe_flag_clean_depth_escape


def test_clean_l21_triggers_l22_nudge(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text("{}", encoding="utf-8")
    state_path = agents / "progress_controller_state.json"
    state_path.write_text(
        json.dumps({"level": 21, "cycles_at_level": 8, "verify_score": 0.75}),
        encoding="utf-8",
    )

    evo_result = {
        "final_level": 21,
        "champion_verify_score": 0.75,
        "champion_words": 36,
        "final_fitness": 96.0,
    }
    escape_level = _maybe_flag_clean_depth_escape(str(tmp_path), evo_result, cycle=2829)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert tick.get("depth_nudge_level") == 22
    assert not tick.get("force_reference_champion")
    assert escape_level == 22


def test_depth_escape_requires_min_cycles(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text("{}", encoding="utf-8")
    state_path = agents / "progress_controller_state.json"
    state_path.write_text(
        json.dumps({"level": 21, "cycles_at_level": 7, "verify_score": 0.75}),
        encoding="utf-8",
    )

    evo_result = {
        "final_level": 21,
        "champion_verify_score": 0.75,
        "champion_words": 36,
        "final_fitness": 96.0,
    }
    assert _maybe_flag_clean_depth_escape(str(tmp_path), evo_result, cycle=1) is None


def test_clean_l23_triggers_l24_nudge_at_lower_verify(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text("{}", encoding="utf-8")
    state_path = agents / "progress_controller_state.json"
    state_path.write_text(
        json.dumps({"level": 23, "cycles_at_level": 10, "verify_score": 0.643}),
        encoding="utf-8",
    )

    evo_result = {
        "final_level": 23,
        "champion_verify_score": 0.643,
        "champion_words": 36,
        "final_fitness": 74.0,
    }
    escape_level = _maybe_flag_clean_depth_escape(str(tmp_path), evo_result, cycle=3005)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert tick.get("depth_nudge_level") == 24
    assert escape_level == 24


def test_max_level_skips_depth_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("EE_MAX_LEVEL", "30")
    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "tick_state.json").write_text("{}", encoding="utf-8")
    (agents / "progress_controller_state.json").write_text(
        json.dumps({"level": 30, "cycles_at_level": 300, "verify_score": 1.0}),
        encoding="utf-8",
    )
    evo_result = {
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 73,
        "final_fitness": 264.04,
    }
    assert _maybe_flag_clean_depth_escape(str(tmp_path), evo_result, cycle=3720) is None


def test_corrupt_l21_skips_escape_nudge(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text("{}", encoding="utf-8")

    evo_result = {
        "final_level": 21,
        "champion_verify_score": 0.58,
        "champion_words": 37,
        "final_fitness": 70.0,
    }
    _maybe_flag_clean_depth_escape(str(tmp_path), evo_result, cycle=2830)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert not tick.get("force_champion_reseed")
