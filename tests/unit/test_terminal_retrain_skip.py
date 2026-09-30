"""Tests for skipping retrain on certified terminal plateau."""

from types import SimpleNamespace

from run_multi_cycle_loop import _evaluate_retrain_skip


def _args(**kwargs):
    defaults = {"skip_retrain_if_no_new_data": False, "force_retrain_interval": 0}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_terminal_plateau_skips_retrain_despite_traces(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_RETRAIN_INTERVAL", "40")
    evo = {
        "final_fitness": 264.04,
        "final_level": 30,
        "champion_words": 73,
        "champion_verify_score": 1.0,
        "champion_traces": 257,
    }
    should, result, _ = _evaluate_retrain_skip(
        cycle=10,
        args=_args(),
        evo_result=evo,
        meta_controller=None,
        checkpoint="ckpt.pt",
        last_champion_fitness=264.04,
        last_champion_level=30,
        last_champion_words=73,
        cycles_since_last_retrain=0,
        last_retrain_epochs=0,
        retrain_meta_params={},
    )
    assert should is False
    assert result["reason"] == "terminal_plateau_unchanged"


def test_terminal_plateau_skips_when_last_level_nudged_past_max(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_RETRAIN_INTERVAL", "40")
    monkeypatch.setenv("EE_MAX_LEVEL", "30")
    evo = {
        "final_fitness": 264.04,
        "final_level": 30,
        "champion_words": 73,
        "champion_verify_score": 1.0,
        "champion_traces": 257,
    }
    should, result, _ = _evaluate_retrain_skip(
        cycle=10,
        args=_args(),
        evo_result=evo,
        meta_controller=None,
        checkpoint="ckpt.pt",
        last_champion_fitness=264.04,
        last_champion_level=31,
        last_champion_words=73,
        cycles_since_last_retrain=0,
        last_retrain_epochs=0,
        retrain_meta_params={},
    )
    assert should is False
    assert result["reason"] == "terminal_plateau_unchanged"


def test_terminal_plateau_suppresses_meta_force_retrain(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_RETRAIN_INTERVAL", "40")
    meta = type("MC", (), {"force_retrain": True, "_consecutive_none_val_loss": 4})()
    evo = {
        "final_fitness": 264.04,
        "final_level": 30,
        "champion_words": 73,
        "champion_verify_score": 1.0,
        "champion_traces": 257,
    }
    should, result, _ = _evaluate_retrain_skip(
        cycle=10,
        args=_args(),
        evo_result=evo,
        meta_controller=meta,
        checkpoint="ckpt.pt",
        last_champion_fitness=264.04,
        last_champion_level=30,
        last_champion_words=73,
        cycles_since_last_retrain=0,
        last_retrain_epochs=0,
        retrain_meta_params={},
    )
    assert should is False
    assert result["reason"] == "terminal_plateau_unchanged"
    assert meta.force_retrain is False
    assert meta._consecutive_none_val_loss == 0


def test_terminal_plateau_ignores_default_force_interval(monkeypatch):
    """Production default force_retrain_interval=3 must not break L30 skip."""
    monkeypatch.setenv("EE_TERMINAL_RETRAIN_INTERVAL", "40")
    evo = {
        "final_fitness": 264.04,
        "final_level": 30,
        "champion_words": 73,
        "champion_verify_score": 1.0,
        "champion_traces": 257,
    }
    should, result, csr = _evaluate_retrain_skip(
        cycle=3733,
        args=_args(force_retrain_interval=3),
        evo_result=evo,
        meta_controller=None,
        checkpoint="ckpt.pt",
        last_champion_fitness=264.04,
        last_champion_level=30,
        last_champion_words=73,
        cycles_since_last_retrain=6,
        last_retrain_epochs=0,
        retrain_meta_params={},
    )
    assert should is False
    assert result["reason"] == "terminal_plateau_unchanged"
    assert csr == 7


def test_terminal_plateau_periodic_retrain(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_RETRAIN_INTERVAL", "40")
    evo = {
        "final_fitness": 264.04,
        "final_level": 30,
        "champion_words": 73,
        "champion_verify_score": 1.0,
        "champion_traces": 257,
    }
    should, _, _ = _evaluate_retrain_skip(
        cycle=50,
        args=_args(),
        evo_result=evo,
        meta_controller=None,
        checkpoint="ckpt.pt",
        last_champion_fitness=264.04,
        last_champion_level=30,
        last_champion_words=73,
        cycles_since_last_retrain=39,
        last_retrain_epochs=0,
        retrain_meta_params={},
    )
    assert should is True


def test_terminal_periodic_retrain_at_tier7_interval(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_RETRAIN_INTERVAL", "40")
    evo = {
        "final_fitness": 264.04,
        "final_level": 30,
        "champion_words": 73,
        "champion_verify_score": 1.0,
        "champion_traces": 257,
    }
    should, result, _ = _evaluate_retrain_skip(
        cycle=100,
        args=_args(),
        evo_result=evo,
        meta_controller=None,
        checkpoint="ckpt.pt",
        last_champion_fitness=264.04,
        last_champion_level=30,
        last_champion_words=73,
        cycles_since_last_retrain=32,
        last_retrain_epochs=0,
        retrain_meta_params={},
    )
    assert should is True
    assert result.get("reason") != "terminal_plateau_unchanged"


def test_cap_terminal_archive_state_trims_strategy_records(monkeypatch):
    from run_multi_cycle_loop import _cap_terminal_archive_state

    monkeypatch.setenv("EE_TERMINAL_STRATEGY_ARCHIVE_CAP", "100")
    big = {"strategy": {"records": [{"i": i} for i in range(500)]}}
    evo = {
        "final_level": 30,
        "champion_verify_score": 1.0,
        "final_fitness": 445.0,
    }
    capped = _cap_terminal_archive_state(big, evo)
    assert capped is not None
    assert len(capped["strategy"]["records"]) == 100


def test_l30_scan_floor_verify_regression_skips_force_retrain(monkeypatch):
    """Transient compression verify drop must not trigger RV retrain at scan floor."""
    monkeypatch.setenv("EE_TERMINAL_RETRAIN_INTERVAL", "40")
    evo = {
        "final_fitness": 199.78,
        "final_level": 30,
        "champion_words": 26,
        "champion_verify_score": 0.8947368421052632,
        "champion_traces": 152,
    }
    should, result, _ = _evaluate_retrain_skip(
        cycle=7575,
        args=_args(),
        evo_result=evo,
        meta_controller=None,
        checkpoint="ckpt.pt",
        last_champion_fitness=325.82,
        last_champion_level=30,
        last_champion_words=27,
        cycles_since_last_retrain=2,
        last_retrain_epochs=0,
        retrain_meta_params={},
    )
    assert should is False
    assert result["reason"] == "l30_verify_regression_skip_retrain"


def test_terminal_plateau_skips_when_fitness_ticks_up(monkeypatch):
    """Small L30 fitness gains must not trigger a full retrain every cycle."""
    monkeypatch.setenv("EE_TERMINAL_RETRAIN_INTERVAL", "40")
    evo = {
        "final_fitness": 443.88,
        "final_level": 30,
        "champion_words": 71,
        "champion_verify_score": 1.0,
        "champion_traces": 200,
    }
    should, result, _ = _evaluate_retrain_skip(
        cycle=4282,
        args=_args(),
        evo_result=evo,
        meta_controller=None,
        checkpoint="ckpt.pt",
        last_champion_fitness=443.58,
        last_champion_level=30,
        last_champion_words=71,
        cycles_since_last_retrain=5,
        last_retrain_epochs=0,
        retrain_meta_params={},
    )
    assert should is False
    assert result["reason"] == "terminal_plateau_unchanged"
