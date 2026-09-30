"""Tests for verify-collapse recovery (no broken reference graft at L22+)."""

import json

from run_multi_cycle_loop import _maybe_flag_collapse_reseed, _prune_stale_tick_reseed_state


def test_collapse_reseed_on_low_verify_at_l21(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text(json.dumps({"force_register_repair": True}), encoding="utf-8")

    evo_result = {
        "final_level": 21,
        "champion_verify_score": 0.17,
        "final_fitness": 47.5,
    }
    _maybe_flag_collapse_reseed(str(tmp_path), evo_result, cycle=2761)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert tick.get("reseed_level") == 21
    assert tick.get("depth_nudge_level") == 21
    assert tick.get("collapse_reseed") is True
    assert "force_reference_champion" not in tick
    assert "force_champion_reseed" not in tick
    assert "force_register_repair" not in tick


def test_no_collapse_reseed_on_l22_first_depth_cycle(tmp_path):
    """L22@0.69 on first depth cycle must not false-trigger L21 graft band."""
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text("{}", encoding="utf-8")

    evo_result = {
        "final_level": 22,
        "champion_verify_score": 0.692,
        "final_fitness": 69.0,
        "champion_words": 36,
    }
    _maybe_flag_collapse_reseed(str(tmp_path), evo_result, cycle=2969)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert not tick.get("collapse_reseed")


def test_no_collapse_reseed_when_l21_nudge_quality(tmp_path):
    """L21@0.75 is nudge-stable but below certify — must not trigger collapse."""
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text("{}", encoding="utf-8")

    evo_result = {
        "final_level": 21,
        "champion_verify_score": 0.75,
        "final_fitness": 80.0,
        "champion_words": 36,
    }
    state_path = agents / "progress_controller_state.json"
    state_path.write_text(json.dumps({"cycles_at_level": 15}), encoding="utf-8")
    _maybe_flag_collapse_reseed(str(tmp_path), evo_result, cycle=2925)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert not tick.get("collapse_reseed")


def test_no_collapse_reseed_when_verify_ok(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text("{}", encoding="utf-8")

    evo_result = {
        "final_level": 21,
        "champion_verify_score": 0.75,
        "final_fitness": 96.0,
        "champion_words": 36,
    }
    state_path = agents / "progress_controller_state.json"
    state_path.write_text(json.dumps({"cycles_at_level": 8}), encoding="utf-8")
    _maybe_flag_collapse_reseed(str(tmp_path), evo_result, cycle=2752)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert not tick.get("collapse_reseed")


def test_collapse_reseed_on_graft_corruption_band(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text("{}", encoding="utf-8")

    evo_result = {
        "final_level": 21,
        "champion_verify_score": 0.667,
        "final_fitness": 77.5,
        "champion_words": 37,
    }
    _maybe_flag_collapse_reseed(str(tmp_path), evo_result, cycle=2769)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert tick.get("reseed_level") == 21
    assert tick.get("depth_nudge_level") == 21


def test_collapse_reseed_on_timeout_bloat(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text("{}", encoding="utf-8")

    evo_result = {
        "final_level": 21,
        "champion_verify_score": 0.69,
        "final_fitness": -60.4,
        "champion_words": 54,
    }
    _maybe_flag_collapse_reseed(str(tmp_path), evo_result, cycle=2765)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert tick.get("reseed_level") == 21


def test_collapse_reseed_on_l21_bloat_at_39_words(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text("{}", encoding="utf-8")

    evo_result = {
        "final_level": 21,
        "champion_verify_score": 0.72,
        "final_fitness": 90.0,
        "champion_words": 39,
    }
    _maybe_flag_collapse_reseed(str(tmp_path), evo_result, cycle=2799)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert tick.get("reseed_level") == 21


def test_collapse_reseed_hollow_l23_drops_to_l21(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text(
        json.dumps({"reseed_level": 23, "force_reference_champion": True}),
        encoding="utf-8",
    )

    evo_result = {
        "final_level": 23,
        "champion_verify_score": 0.07,
        "final_fitness": 50.0,
        "champion_words": 40,
    }
    _maybe_flag_collapse_reseed(str(tmp_path), evo_result, cycle=2897)

    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert tick.get("reseed_level") == 21
    assert tick.get("depth_nudge_level") == 21
    assert tick.get("collapse_reseed") is True
    assert "force_reference_champion" not in tick


def test_apply_collapse_recovery_drops_level(tmp_path):
    from run_multi_cycle_loop import _maybe_apply_collapse_recovery

    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text(
        json.dumps({"collapse_reseed": True, "reseed_level": 21, "depth_nudge_level": 21}),
        encoding="utf-8",
    )
    state_path = agents / "progress_controller_state.json"
    state_path.write_text(json.dumps({"level": 23, "cycles_at_level": 20}), encoding="utf-8")

    new_level = _maybe_apply_collapse_recovery(str(tmp_path), cycle=2910, last_champion_level=23)
    assert new_level == 21
    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert "collapse_reseed" not in tick
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["level"] == 21
    assert state["cycles_at_level"] == 0


def test_apply_collapse_recovery_clears_when_already_at_level(tmp_path):
    from run_multi_cycle_loop import _maybe_apply_collapse_recovery

    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text(json.dumps({"collapse_reseed": True, "reseed_level": 21}), encoding="utf-8")
    new_level = _maybe_apply_collapse_recovery(str(tmp_path), cycle=2925, last_champion_level=21)
    assert new_level == 21
    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert "collapse_reseed" not in tick


def test_apply_collapse_recovery_queues_same_level_rollback(tmp_path):
    from run_multi_cycle_loop import _maybe_apply_collapse_recovery

    agents = tmp_path / ".agents"
    agents.mkdir()
    results = tmp_path / "multi_cycle_results.jsonl"
    good = {
        "cycle": 10,
        "final_level": 21,
        "champion_verify_score": 0.75,
        "champion_words": 36,
        "final_fitness": 80.0,
        "champion_program": [f"0x{i:08X}" for i in range(36)] + ["0x00100073"],
    }
    results.write_text(json.dumps(good) + "\n", encoding="utf-8")
    tick_path = agents / "tick_state.json"
    tick_path.write_text(
        json.dumps(
            {
                "collapse_reseed": True,
                "reseed_level": 21,
                "champion_rollback_program": good["champion_program"],
                "champion_rollback_cycle": 10,
            }
        ),
        encoding="utf-8",
    )
    state_path = agents / "progress_controller_state.json"
    state_path.write_text(json.dumps({"level": 21, "cycles_at_level": 9}), encoding="utf-8")

    new_level = _maybe_apply_collapse_recovery(str(tmp_path), cycle=11, last_champion_level=21)
    assert new_level == 21
    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert tick.get("apply_champion_rollback") is True
    assert "collapse_reseed" not in tick


def test_clear_stale_collapse_tick_flags_on_stable_l22(tmp_path):
    from run_multi_cycle_loop import _clear_stale_collapse_tick_flags

    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text(
        json.dumps(
            {
                "reseed_level": 21,
                "champion_rollback_program": ["0x00100073"],
                "champion_rollback_cycle": 10,
                "collapse_reseed": True,
            }
        ),
        encoding="utf-8",
    )
    evo = {"final_level": 22, "champion_verify_score": 0.769}
    _clear_stale_collapse_tick_flags(str(tmp_path), evo)
    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert "champion_rollback_program" not in tick
    assert "collapse_reseed" not in tick
    assert "reseed_level" not in tick


def test_prune_stale_tick_reseed_state(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text(
        json.dumps(
            {
                "reseed_level": 21,
                "champion_rollback_program": ["0x00100073"],
                "champion_rollback_cycle": 2983,
                "force_reference_champion": True,
            }
        ),
        encoding="utf-8",
    )
    _prune_stale_tick_reseed_state(str(tmp_path), active_level=24)
    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert "champion_rollback_program" not in tick
    assert "reseed_level" not in tick
    assert "force_reference_champion" not in tick


def test_auto_depth_escape_start_level_at_l24(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    state_path = agents / "progress_controller_state.json"
    state_path.write_text(
        json.dumps(
            {
                "level": 24,
                "cycles_at_level": 10,
                "verify_score": 0.6,
                "raw_fitness": 57.3,
            }
        ),
        encoding="utf-8",
    )
    from run_multi_cycle_loop import _auto_depth_escape_start_level

    assert _auto_depth_escape_start_level(str(tmp_path), cycle=3031, last_champion_level=24) == 25


def test_clear_stale_collapse_tick_flags_on_stable_l24(tmp_path):
    from run_multi_cycle_loop import _clear_stale_collapse_tick_flags

    agents = tmp_path / ".agents"
    agents.mkdir()
    tick_path = agents / "tick_state.json"
    tick_path.write_text(
        json.dumps(
            {
                "reseed_level": 21,
                "champion_rollback_program": ["0x00100073"],
                "champion_rollback_cycle": 2983,
            }
        ),
        encoding="utf-8",
    )
    evo = {"final_level": 24, "champion_verify_score": 0.6}
    _clear_stale_collapse_tick_flags(str(tmp_path), evo)
    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    assert "champion_rollback_program" not in tick
    assert "reseed_level" not in tick
