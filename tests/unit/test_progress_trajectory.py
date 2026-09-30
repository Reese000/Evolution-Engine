"""Unit tests for Evolution Progress Index."""

import json

from scripts.progress_trajectory import _compute_epi, _fitness_vs_level_best, _verify_effective


def test_verify_effective_raw_not_capped_at_partial():
    assert _verify_effective(13, 0.875) == 0.875
    assert _verify_effective(13, 1.0) == 1.0


def test_fitness_vs_level_best():
    assert _fitness_vs_level_best(100, 100) == 1.0
    assert _fitness_vs_level_best(90, 100) == 0.9


def test_smooth_rolling_mean():
    from scripts.progress_trajectory import _smooth

    raw = [10.0, 20.0, 30.0, 40.0]
    sm = _smooth(raw)
    assert sm[-1] == round(sum(raw) / len(raw), 2)


def test_epi_terminal_depth_allows_fitness_growth():
    """L30 partial verify should not cap fitness_at_level when above carry floor."""
    row = {
        "final_level": 30,
        "champion_verify_score": 0.8421,
        "final_fitness": 50.8,
        "generations_completed": 50,
        "strategy_archive_records": 2500,
        "vm_gpu_batches": 100,
        "vm_cpu_fallbacks": 0,
    }
    capped = _compute_epi(row, {30: 40.0}, {30: 50.8})
    growing = _compute_epi(row, {30: 40.0}, {30: 50.8}, prev_level_best=47.5)
    assert capped["components"]["fitness_at_level"] > 35.0
    assert growing["epi"] >= capped["epi"]


def test_epi_terminal_verify_growth_bonus():
    row = {
        "final_level": 30,
        "champion_verify_score": 0.8421,
        "max_pop_verify": 0.8421,
        "final_fitness": 50.8,
        "generations_completed": 50,
        "strategy_archive_records": 2500,
        "vm_gpu_batches": 100,
        "vm_cpu_fallbacks": 0,
    }
    flat = _compute_epi(row, {30: 40.0}, {30: 50.8}, prev_level_verify=0.8421)
    rising = _compute_epi(row, {30: 40.0}, {30: 50.8}, prev_level_verify=0.6842)
    assert rising["epi"] > flat["epi"]


def test_epi_uses_max_pop_verify_at_terminal():
    row = {
        "final_level": 30,
        "champion_verify_score": 0.8421,
        "max_pop_verify": 0.8947,
        "final_fitness": 50.8,
        "generations_completed": 50,
        "strategy_archive_records": 2500,
        "vm_gpu_batches": 100,
        "vm_cpu_fallbacks": 0,
    }
    champ_only = _compute_epi({**row, "max_pop_verify": 0.8421}, {30: 40.0}, {30: 50.8})
    with_pop = _compute_epi(row, {30: 40.0}, {30: 50.8})
    assert with_pop["epi"] >= champ_only["epi"]


def test_terminal_gens_cap_gets_full_throughput_credit():
    from scripts.progress_trajectory import _gens_ratio

    assert _gens_ratio(30, level=30) == 1.0
    assert _gens_ratio(20, level=10) < 1.0


def test_carry_terminal_archive_learning_holds_peak():
    from scripts.progress_trajectory import _carry_terminal_archive_learning

    dropped = {
        "epi": 94.81,
        "level": 30,
        "components": {"archive_learning": 15.0},
    }
    held = _carry_terminal_archive_learning(dropped, 81.5)
    assert held["components"]["archive_learning"] == 81.5
    assert held["epi"] > 94.81


def test_fitness_for_epi_unclamps_reporting_floor():
    from scripts.progress_trajectory import _fitness_for_epi

    frozen = {
        "final_level": 30,
        "final_fitness": 445.38,
        "fitness_end": 389.08,
    }
    live = {
        "final_level": 30,
        "final_fitness": 445.38,
        "fitness_end": 446.58,
    }
    collapse = {
        "final_level": 30,
        "final_fitness": 445.38,
        "fitness_end": 45.14,
    }
    assert abs(_fitness_for_epi(frozen) - 389.08) < 1e-6
    assert abs(_fitness_for_epi(live) - 446.58) < 1e-6
    assert abs(_fitness_for_epi(collapse) - 445.38) < 1e-6


def test_unclamped_loop_fitness_moves_epi_over_lookback():
    """Frozen final_fitness + climbing fitness_end must produce EPI gain >= 0.5 / 200."""
    base = {
        "final_level": 30,
        "champion_verify_score": 1.0,
        "max_pop_verify": 1.0,
        "generations_completed": 30,
        "strategy_archive_records": 8150,
        "vm_gpu_batches": 100,
        "vm_cpu_fallbacks": 0,
        "champion_words": 71,
        "final_fitness": 445.38,
    }
    start = _compute_epi({**base, "fitness_end": 389.08}, {30: 400.0}, {30: 446.0})
    end = _compute_epi({**base, "fitness_end": 446.58}, {30: 400.0}, {30: 446.0})
    assert end["fitness"] > start["fitness"]
    assert end["epi"] - start["epi"] >= 0.5


def test_certified_terminal_epi_rises_with_fitness_when_saturated():
    """Saturated L30 verify=1.0: EPI must track fitness for growth-gate honesty."""
    base_row = {
        "final_level": 30,
        "champion_verify_score": 1.0,
        "max_pop_verify": 1.0,
        "generations_completed": 50,
        "strategy_archive_records": 8150,
        "vm_gpu_batches": 100,
        "vm_cpu_fallbacks": 0,
        "champion_words": 73,
    }
    low = _compute_epi({**base_row, "final_fitness": 394.0}, {30: 390.0}, {30: 433.0})
    high = _compute_epi({**base_row, "final_fitness": 433.0}, {30: 390.0}, {30: 433.0})
    assert "terminal_fitness_bump" in high["components"]
    assert high["epi"] > low["epi"]
    assert high["epi"] - low["epi"] >= 2.0


def test_terminal_floor_suppressed_after_fitness_stagnation():
    from scripts.progress_trajectory import _apply_epi_momentum

    points = []
    for i in range(60):
        points.append(
            {
                "cycle": 3400 + i,
                "epi": 93.5,
                "level": 30,
                "fitness": 264.0,
                "verify_score": 1.0,
                "components": {"archive_learning": 25.0},
            }
        )
    _apply_epi_momentum(points)
    mid = [p["epi_momentum"] for p in points[20:25]]
    tail = [p["epi_momentum"] for p in points[-10:]]
    assert any(m >= 0.55 for m in mid)
    assert all(m < 0.55 for m in tail)


def test_terminal_momentum_rewards_fitness_breakthrough():
    from scripts.progress_trajectory import _apply_epi_momentum

    points = []
    for i in range(40):
        points.append(
            {
                "cycle": 3500 + i,
                "epi": 87.5 + i * 0.05,
                "level": 30,
                "fitness": 264.0 + i * 0.05,
                "verify_score": 1.0,
                "components": {"archive_learning": 25.0},
            }
        )
    _apply_epi_momentum(points)
    assert points[-1]["epi_momentum"] > 0.55


def test_terminal_momentum_survives_epi_dip_at_plateau():
    """Certified L30 verify=1.0 must keep terminal floor when EPI dips on median tie."""
    from scripts.progress_trajectory import _apply_epi_momentum, _momentum_for_point, _rolling_mean

    points = [
        {
            "cycle": 3588,
            "epi": 93.5,
            "level": 30,
            "fitness": 264.0,
            "verify_score": 1.0,
            "components": {"archive_learning": 25.0, "fitness_momentum": 100.0},
        },
        {
            "cycle": 3589,
            "epi": 87.5,
            "level": 30,
            "fitness": 264.0,
            "verify_score": 1.0,
            "components": {"archive_learning": 25.0, "fitness_momentum": 50.0},
        },
    ]
    epi_series = [p["epi"] for p in points]
    rolling = _rolling_mean(epi_series, min(50, len(epi_series)))
    assert _momentum_for_point(points, 1, epi_series, rolling) >= 0.55
    _apply_epi_momentum(points)
    assert points[1]["epi_momentum"] >= 0.55


def test_terminal_streak_bonus_starts_after_threshold():
    from scripts.progress_trajectory import _streak_growth_bonus

    points = [
        {
            "cycle": 3400 + i,
            "epi": 93.5,
            "level": 30,
            "fitness": 264.0,
            "verify_score": 1.0,
        }
        for i in range(25)
    ]
    assert _streak_growth_bonus(points, 18) == 0.0
    assert _streak_growth_bonus(points, 19) == 0.02
    assert _streak_growth_bonus(points, 24) == 0.12


def test_append_trajectory_lines_rewrites_entire_file(tmp_path, monkeypatch):
    from scripts import progress_trajectory as pt

    monkeypatch.setattr(pt, "AGENTS", str(tmp_path))
    monkeypatch.setattr(pt, "TRAJECTORY", str(tmp_path / "progress_trajectory.jsonl"))
    points = [
        {"cycle": 1, "epi_momentum": 1.1},
        {"cycle": 2, "epi_momentum": 2.2},
    ]
    pt._append_trajectory_lines(points, 1)
    loaded = pt._load_trajectory_points()
    assert len(loaded) == 2
    assert loaded[0]["epi_momentum"] == 1.1
    assert loaded[1]["epi_momentum"] == 2.2


def test_certified_terminal_with_ee_max_level_clamp(monkeypatch):
    """L30 must count as terminal when EE_MAX_LEVEL recovery clamp is 31."""
    monkeypatch.setenv("EE_MAX_LEVEL", "31")
    from scripts.progress_trajectory import _apply_epi_momentum, _is_certified_terminal_point, _terminal_growth_momentum

    point = {
        "cycle": 3916,
        "epi": 88.37,
        "level": 30,
        "fitness": 333.58,
        "verify_score": 1.0,
        "components": {"archive_learning": 25.0},
    }
    prev = {
        "cycle": 3915,
        "epi": 82.96,
        "level": 30,
        "fitness": 234.29,
        "verify_score": 1.0,
        "components": {"archive_learning": 25.0},
    }
    assert _is_certified_terminal_point(point) is True
    growth = _terminal_growth_momentum([prev, point], 1)
    assert growth >= 3.0
    points = [prev, point]
    _apply_epi_momentum(points)
    assert points[1]["epi_momentum"] >= 3.0


def test_length_efficiency_higher_for_shorter_programs(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_REF_WORDS", "73")
    from scripts.progress_trajectory import _length_efficiency_score

    long_score = _length_efficiency_score(30, 80, prev_best_words=80.0)
    short_score = _length_efficiency_score(30, 60, prev_best_words=80.0)
    neutral = _length_efficiency_score(20, 60)
    assert short_score > long_score
    assert neutral == 0.5


def test_epi_length_efficiency_component_at_terminal():
    row = {
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 65,
        "final_fitness": 264.0,
        "generations_completed": 50,
        "strategy_archive_records": 2500,
        "vm_gpu_batches": 100,
        "vm_cpu_fallbacks": 0,
    }
    shorter = _compute_epi(row, {30: 200.0}, {30: 264.0}, prev_best_words=73.0)
    longer = _compute_epi({**row, "champion_words": 80}, {30: 200.0}, {30: 264.0}, prev_best_words=73.0)
    assert shorter["components"]["length_efficiency"] > longer["components"]["length_efficiency"]
    assert shorter["champion_words"] == 65


def test_terminal_momentum_rewards_word_shrink():
    from scripts.progress_trajectory import _apply_epi_momentum, _terminal_growth_momentum

    prev = {
        "cycle": 4000,
        "epi": 93.5,
        "level": 30,
        "fitness": 264.0,
        "verify_score": 1.0,
        "champion_words": 73,
        "components": {"archive_learning": 25.0, "length_efficiency": 50.0},
    }
    point = {
        "cycle": 4001,
        "epi": 93.5,
        "level": 30,
        "fitness": 264.0,
        "verify_score": 1.0,
        "champion_words": 68,
        "components": {"archive_learning": 25.0, "length_efficiency": 62.0},
    }
    growth = _terminal_growth_momentum([prev, point], 1)
    assert growth > 0.55
    _apply_epi_momentum([prev, point])
    assert point["epi_momentum"] > 0.55


def test_write_canvas_single_status_panel(tmp_path, monkeypatch):
    from scripts import progress_trajectory as pt

    canvas_path = tmp_path / "evolution-progress-trajectory.canvas.tsx"
    monkeypatch.setattr(pt, "CANVAS_DIR", str(tmp_path))
    monkeypatch.setattr(pt, "CANVAS_PATH", str(canvas_path))
    monkeypatch.setattr(pt, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(pt, "PROGRESS_CONTROLLER_STATE", str(tmp_path / "missing-ctrl.json"))
    monkeypatch.setattr(pt, "AGENT_STATUS", str(tmp_path / "missing-status.json"))
    monkeypatch.setattr(pt, "_stop_stale_watcher", lambda: False)
    points = [
        {
            "cycle": 1,
            "epi": 50.0,
            "level": 10,
            "fitness": 100.0,
            "components": {"fitness_at_level": 80.0},
            "epi_momentum": 0.0,
        }
    ]
    summary = {
        "updated_at": "2026-01-01T00:00:00Z",
        "count": 1,
        "latest": points[0],
        "peak_level": 10,
        "fitness_at_level_end": 80.0,
    }
    pt._write_canvas(points, summary)
    text = canvas_path.read_text(encoding="utf-8")
    assert text.count("Evolution Progress</H2>") == 1
    assert "<H1>" not in text
    assert f"canvas-template:{pt.CANVAS_TEMPLATE_VERSION}" in text
    assert text.count('variant="borderless"') == 1
    assert text.count("<Grid columns") >= 1


def test_robust_bounds_ignore_fitness_spike():
    from scripts.progress_trajectory import _robust_bounds

    vals = [324.0] * 40 + [6741.0] + [348.0] * 10
    lo, hi = _robust_bounds(vals, include_latest=True)
    assert hi < 800
    assert lo > 200


def test_chart_bounds_allow_negative_momentum():
    from scripts.progress_trajectory import _chart_bounds

    lo, hi = _chart_bounds([-12.0, 0.0, 4.0], pad=1.0, cap_max=None, allow_negative=True)
    assert lo < 0
    assert hi > 0


def test_write_canvas_terminal_shows_words_not_flat_level(tmp_path, monkeypatch):
    from scripts import progress_trajectory as pt

    canvas_path = tmp_path / "evolution-progress-trajectory.canvas.tsx"
    monkeypatch.setattr(pt, "CANVAS_DIR", str(tmp_path))
    monkeypatch.setattr(pt, "CANVAS_PATH", str(canvas_path))
    monkeypatch.setattr(pt, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(pt, "PROGRESS_CONTROLLER_STATE", str(tmp_path / "ctrl.json"))
    monkeypatch.setattr(pt, "AGENT_STATUS", str(tmp_path / "status.json"))
    monkeypatch.setattr(pt, "_stop_stale_watcher", lambda: False)
    (tmp_path / "ctrl.json").write_text(
        json.dumps(
            {
                "level": 30,
                "cycles_at_level": 100,
                "raw_fitness": 325.428,
                "terminal_efficiency_scan_floor_words": 50,
                "terminal_efficiency_scan_floor_heldout": 0.9901,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "status.json").write_text(
        json.dumps({"champion_promotion_raw": 325.428, "current_task": "50w@325.428"}),
        encoding="utf-8",
    )
    points = []
    for i in range(60):
        words = 54 if i < 50 else 50
        points.append(
            {
                "cycle": 7000 + i,
                "epi": 71.0,
                "level": 30,
                "fitness": 350.0 + (i % 5),
                "verify_score": 1.0,
                "champion_words": words,
                "cumulative_hours": 100 + i * 0.01,
                "components": {
                    "fitness_at_level": 5.2,
                    "length_efficiency": 61.5,
                },
                "epi_momentum": 1.5,
            }
        )
    summary = {
        "updated_at": "2026-01-01T00:00:00Z",
        "count": len(points),
        "latest": points[-1],
        "peak_level": 30,
        "fitness_at_level_end": 5.2,
    }
    pt._write_canvas(points, summary, force=True)
    text = canvas_path.read_text(encoding="utf-8")
    assert "Champion length (words, unsmoothed)" in text
    assert "Curriculum level by cycle" not in text
    assert "vs L-best" not in text
    assert "Held-out L30" in text
    assert "Promotion raw" in text
    assert "RECENT_WORDS" in text
    assert "yMax={6741" not in text
    assert "Latest cycles" in text
    assert "50w scan floor" in text


def test_stop_stale_watcher_ignores_current_template(monkeypatch, tmp_path):
    from scripts import progress_trajectory as pt

    monkeypatch.setattr(pt, "WATCHER_PID_FILE", str(tmp_path / "watcher.pid"))
    monkeypatch.setattr(pt, "HEARTBEAT", str(tmp_path / "heartbeat.json"))
    (tmp_path / "watcher.pid").write_text("99999", encoding="utf-8")
    (tmp_path / "heartbeat.json").write_text(
        json.dumps(
            {
                "pid": 99999,
                "status": "idle",
                "canvas_template_version": pt.CANVAS_TEMPLATE_VERSION,
                "epi_formula_version": pt.EPI_FORMULA_VERSION,
            }
        ),
        encoding="utf-8",
    )

    called = False

    def fake_terminate(pid):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr(pt, "_terminate_pid", fake_terminate)
    monkeypatch.setattr(pt, "_pid_alive", lambda _pid: True)
    assert pt._stop_stale_watcher() is False
    assert called is False


def test_stop_stale_watcher_kills_old_epi_formula(monkeypatch, tmp_path):
    from scripts import progress_trajectory as pt

    monkeypatch.setattr(pt, "WATCHER_PID_FILE", str(tmp_path / "watcher.pid"))
    monkeypatch.setattr(pt, "HEARTBEAT", str(tmp_path / "heartbeat.json"))
    (tmp_path / "watcher.pid").write_text("4242", encoding="utf-8")
    (tmp_path / "heartbeat.json").write_text(
        json.dumps(
            {
                "pid": 4242,
                "status": "idle",
                "canvas_template_version": pt.CANVAS_TEMPLATE_VERSION,
                "epi_formula_version": "legacy-epi",
            }
        ),
        encoding="utf-8",
    )
    terminated: list[int] = []
    monkeypatch.setattr(pt, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(pt, "_terminate_pid", lambda pid: terminated.append(pid) or True)
    assert pt._stop_stale_watcher() is True
    assert terminated == [4242]


def test_trajectory_needs_epi_rebuild_when_terminal_bump_missing():
    from scripts.progress_trajectory import _trajectory_needs_epi_rebuild

    points = [
        {
            "cycle": 4282,
            "level": 30,
            "verify_score": 1.0,
            "fitness": 443.88,
            "epi": 91.17,
            "components": {},
        }
    ]
    assert _trajectory_needs_epi_rebuild(points) is True
    points[0]["components"] = {"terminal_fitness_bump": 10.66}
    points[0]["epi"] = 101.83
    assert _trajectory_needs_epi_rebuild(points) is False


def test_stop_stale_watcher_kills_legacy_heartbeat(monkeypatch, tmp_path):
    import json as json_mod

    from scripts import progress_trajectory as pt

    monkeypatch.setattr(pt, "WATCHER_PID_FILE", str(tmp_path / "watcher.pid"))
    monkeypatch.setattr(pt, "HEARTBEAT", str(tmp_path / "heartbeat.json"))
    (tmp_path / "watcher.pid").write_text("4242", encoding="utf-8")
    (tmp_path / "heartbeat.json").write_text(
        json_mod.dumps({"pid": 4242, "status": "idle", "smooth_window": 25}),
        encoding="utf-8",
    )
    terminated: list[int] = []
    monkeypatch.setattr(pt, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(pt, "_terminate_pid", lambda pid: terminated.append(pid) or True)
    assert pt._stop_stale_watcher() is True
    assert terminated == [4242]


def test_canvas_write_skipped_when_fingerprint_unchanged(tmp_path, monkeypatch):
    from scripts import progress_trajectory as pt

    canvas_path = tmp_path / "evolution-progress-trajectory.canvas.tsx"
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(pt, "CANVAS_DIR", str(tmp_path))
    monkeypatch.setattr(pt, "CANVAS_PATH", str(canvas_path))
    monkeypatch.setattr(pt, "STATE_FILE", str(state_path))
    monkeypatch.setattr(pt, "PROGRESS_CONTROLLER_STATE", str(tmp_path / "missing-ctrl.json"))
    monkeypatch.setattr(pt, "AGENT_STATUS", str(tmp_path / "missing-status.json"))
    monkeypatch.setattr(pt, "_stop_stale_watcher", lambda: False)
    points = [
        {
            "cycle": 1,
            "epi": 50.0,
            "level": 10,
            "fitness": 100.0,
            "components": {"fitness_at_level": 80.0},
            "epi_momentum": 0.0,
        }
    ]
    summary = {
        "updated_at": "2026-01-01T00:00:00Z",
        "count": 1,
        "latest": points[0],
        "peak_level": 10,
        "fitness_at_level_end": 80.0,
    }
    pt._write_canvas(points, summary)
    first = canvas_path.read_text(encoding="utf-8")
    pt._write_canvas(points, summary)
    second = canvas_path.read_text(encoding="utf-8")
    assert first == second


def test_epi_increases_with_level():
    low = _compute_epi(
        {"final_level": 5, "champion_verify_score": 1.0, "final_fitness": 100, "generations_completed": 50},
        {},
        {5: 100},
    )
    high = _compute_epi(
        {
            "final_level": 17,
            "champion_verify_score": 0.875,
            "final_fitness": 120,
            "generations_completed": 50,
            "strategy_archive_records": 10000,
            "vm_gpu_batches": 10,
            "vm_cpu_fallbacks": 0,
        },
        {},
        {17: 120},
    )
    assert high["epi"] > low["epi"]
