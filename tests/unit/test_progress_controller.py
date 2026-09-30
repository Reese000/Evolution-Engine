"""Tests for unified ProgressController."""

from evolution.progress_controller import ProgressAction, ProgressController, StallTier


def test_stall_tier_escalation(tmp_path):
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.generations_stalled = 1
    assert pc.stall_tier() == StallTier.NORMAL
    pc.state.generations_stalled = 2
    assert pc.stall_tier() == StallTier.SOFT
    pc.state.generations_stalled = 8
    assert pc.stall_tier() == StallTier.ESCAPE


def test_l17_can_advance_at_partial_verify(tmp_path):
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.generations_stalled = 3
    assert pc.can_advance_level(0.875, 128.0, 17) is True


def test_l19_nudge_at_observed_fitness(tmp_path):
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.level = 19
    for _ in range(2):
        pc.on_cycle_end(19, 0.9, 83.3)
    plan = pc.on_cycle_end(19, 0.9, 83.3)
    assert ProgressAction.NUDGE_START_LEVEL in plan.actions
    assert plan.nudge_start_level == 20


def test_on_cycle_end_nudge(tmp_path):
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.level = 17
    for _ in range(2):
        pc.on_cycle_end(17, 0.875, 130.0)
    plan = pc.on_cycle_end(17, 0.875, 130.0)
    assert ProgressAction.NUDGE_START_LEVEL in plan.actions
    assert plan.nudge_start_level == 18


def test_l21_nudge_at_partial_verify_075(tmp_path):
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.level = 21
    plan = None
    for _ in range(7):
        pc.on_cycle_end(21, 0.75, 96.4)
    plan = pc.on_cycle_end(21, 0.75, 96.4)
    assert ProgressAction.NUDGE_START_LEVEL in plan.actions
    assert plan.nudge_start_level == 22


def test_l21_nudge_at_observed_fitness_68(tmp_path):
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.level = 21
    plan = None
    for _ in range(7):
        pc.on_cycle_end(21, 0.75, 68.0)
    plan = pc.on_cycle_end(21, 0.75, 68.0)
    assert ProgressAction.NUDGE_START_LEVEL in plan.actions
    assert plan.nudge_start_level == 22


def test_l21_no_nudge_before_depth_escape_cycles(tmp_path):
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.level = 21
    for _ in range(7):
        plan = pc.on_cycle_end(21, 0.75, 96.4)
        assert ProgressAction.NUDGE_START_LEVEL not in plan.actions
    assert pc.state.cycles_at_level == 7


def test_save_preserves_extra_progress_keys(tmp_path):
    agents = tmp_path / ".agents"
    agents.mkdir()
    state_path = agents / "progress_controller_state.json"
    state_path.write_text(
        '{"level": 30, "terminal_efficiency_promote_cycle": 5553}',
        encoding="utf-8",
    )
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.level = 30
    pc.state.cycles_at_level = 2250
    pc.state.verify_score = 1.0
    pc.state.raw_fitness = 350.0
    pc.save()
    import json

    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["terminal_efficiency_promote_cycle"] == 5553
    assert data["cycles_at_level"] == 2250


def test_reconcile_from_cycle_result_fixes_stale_level(tmp_path):
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.level = 21
    pc.state.cycles_at_level = 0
    pc.save()
    pc.reconcile_from_cycle_result(22, 0.769, 106.0)
    pc._load()
    assert pc.state.level == 22
    assert pc.state.cycles_at_level == 1
    pc.reconcile_from_cycle_result(22, 0.769, 99.0)
    pc._load()
    assert pc.state.level == 22
    assert pc.state.cycles_at_level == 1


def test_sustained_resume_level():
    results = [
        {"final_level": 22, "final_fitness": -25, "champion_verify_score": 0.5},
        {"final_level": 17, "final_fitness": 128, "champion_verify_score": 0.875},
        {"final_level": 21, "final_fitness": 90, "champion_verify_score": 0.83},
    ]
    assert ProgressController.sustained_resume_level(results) == 17


def test_sustained_resume_level_accepts_l22_nudge_quality():
    results = [
        {"final_level": 22, "final_fitness": 84.87, "champion_verify_score": 0.769},
        {"final_level": 21, "final_fitness": 96.0, "champion_verify_score": 0.75},
    ]
    assert ProgressController.sustained_resume_level(results) == 22


def test_carry_forward_clamps_hollow_spike():
    carried = ProgressController.carry_forward_start_level(23, 0.5714, sustained_cap=17)
    assert carried == 17


def test_carry_forward_keeps_quality_level():
    carried = ProgressController.carry_forward_start_level(17, 0.875, sustained_cap=17)
    assert carried == 17


def test_carry_forward_keeps_l22_nudge_quality():
    carried = ProgressController.carry_forward_start_level(22, 0.7692, sustained_cap=21)
    assert carried == 22


def test_carry_forward_keeps_l23_nudge_quality():
    carried = ProgressController.carry_forward_start_level(23, 0.6429, sustained_cap=22)
    assert carried == 23


def test_carry_forward_keeps_l24_nudge_quality():
    carried = ProgressController.carry_forward_start_level(24, 0.60, sustained_cap=23)
    assert carried == 24


def test_carry_forward_keeps_l25_nudge_quality():
    carried = ProgressController.carry_forward_start_level(25, 0.5625, sustained_cap=24)
    assert carried == 25


def test_sustained_resume_level_accepts_l25_nudge_quality():
    results = [
        {"final_level": 25, "final_fitness": 65.6, "champion_verify_score": 0.5625},
        {"final_level": 24, "final_fitness": 57.3, "champion_verify_score": 0.60},
    ]
    assert ProgressController.sustained_resume_level(results) == 25


def test_carry_forward_keeps_l25_at_verify_057():
    carried = ProgressController.carry_forward_start_level(25, 0.57, sustained_cap=24)
    assert carried == 25


def test_carry_forward_keeps_l28_nudge_quality():
    carried = ProgressController.carry_forward_start_level(28, 0.5263, sustained_cap=27)
    assert carried == 28


def test_carry_forward_keeps_l29_nudge_quality():
    carried = ProgressController.carry_forward_start_level(29, 0.4737, sustained_cap=28)
    assert carried == 29


def test_sustained_resume_level_accepts_l24_nudge_quality():
    results = [
        {"final_level": 23, "final_fitness": 54.88, "champion_verify_score": 0.643},
        {"final_level": 22, "final_fitness": 96.0, "champion_verify_score": 0.75},
    ]
    assert ProgressController.sustained_resume_level(results) == 23


def test_passes_certify_verify_l23_bar():
    assert ProgressController.passes_certify_verify(23, 0.57) is False
    assert ProgressController.passes_certify_verify(17, 0.875) is True


def test_strict_advance_blocks_hollow_l22():
    pc = ProgressController("/tmp", stall_threshold=2)
    pc.state.generations_stalled = 10
    assert pc.passes_certify_verify(22, 0.5385) is False
    assert pc.can_advance_level(0.5385, 130.0, 22) is False


def test_forced_escape_disabled_in_strict_mode(tmp_path):
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.generations_stalled = 20
    assert pc.can_forced_escape(0.6, 130.0, 17) is False


def test_flag_nudge_reference_reseed_writes_tick_state(tmp_path, monkeypatch):
    import json
    import os

    from run_multi_cycle_loop import _flag_nudge_reference_reseed

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
