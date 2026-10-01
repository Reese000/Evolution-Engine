"""Tests for terminal diversity boost param application."""

import json
import os
import tempfile

from run_multi_cycle_loop import _apply_terminal_diversity_boost


def test_terminal_diversity_boost_raises_scheduler_and_ceil():
    with tempfile.TemporaryDirectory() as tmp:
        agents = os.path.join(tmp, ".agents")
        os.makedirs(agents)
        with open(os.path.join(agents, "tick_state.json"), "w", encoding="utf-8") as f:
            json.dump({"terminal_diversity_boost": True}, f)
        with open(os.path.join(agents, "progress_controller_state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "level": 30,
                    "cycles_at_level": 339,
                    "verify_score": 1.0,
                    "raw_fitness": 264.04,
                },
                f,
            )
        evo: dict = {"mutation_rate": 0.52, "mutation_rate_start": 0.5, "mutation_rate_end": 0.1}
        _apply_terminal_diversity_boost(1, evo, tmp)
        assert evo["mutation_rate"] >= 0.88
        assert evo["mutation_rate_start"] >= 0.88
        assert evo["mutation_rate_end"] >= 0.80
        assert evo["adaptive_mutation_ceil"] >= 0.95
        assert evo["diversity_inject_ratio"] >= 0.70
        assert evo["novelty_weight"] >= 0.85
        assert evo["crossover_rate"] >= 0.45
        assert evo["champion_crossover_prob"] >= 0.35


def test_terminal_diversity_boost_suppressed_during_sustained_goal_recovery():
    with tempfile.TemporaryDirectory() as tmp:
        agents = os.path.join(tmp, ".agents")
        os.makedirs(agents)
        tick_path = os.path.join(agents, "tick_state.json")
        with open(tick_path, "w", encoding="utf-8") as f:
            json.dump({"terminal_diversity_boost": True}, f)
        with open(os.path.join(agents, "progress_controller_state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "level": 30,
                    "cycles_at_level": 100,
                    "verify_score": 1.0,
                    "raw_fitness": 264.04,
                },
                f,
            )
        traj = os.path.join(agents, "progress_trajectory.jsonl")
        with open(traj, "w", encoding="utf-8") as f:
            for i in range(200):
                mom = -0.5 if i < 120 else 1.5
                f.write(
                    json.dumps(
                        {
                            "cycle": i,
                            "epi_momentum": mom,
                            "fitness": 400.0 + i * 0.5,
                            "epi": 100.0 + i * 0.05,
                            "level": 30,
                        }
                    )
                    + "\n"
                )
        evo: dict = {"mutation_rate": 0.52}
        assert _apply_terminal_diversity_boost(2, evo, tmp) is False
        with open(tick_path, encoding="utf-8") as f:
            tick = json.load(f)
        assert "terminal_diversity_boost" not in tick
        assert evo["mutation_rate"] == 0.52


def test_scan_floor_caps_diversity_boost_tier(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        agents = os.path.join(tmp, ".agents")
        os.makedirs(agents)
        with open(os.path.join(agents, "tick_state.json"), "w", encoding="utf-8") as f:
            json.dump({"terminal_diversity_boost": True}, f)
        progress = {
            "level": 30,
            "cycles_at_level": 2300,
            "verify_score": 1.0,
            "raw_fitness": 370.0,
            "terminal_efficiency_stable_words": 60,
            "terminal_efficiency_scan_floor_words": 60,
            "terminal_efficiency_scan_floor_heldout": 1.0,
        }
        with open(os.path.join(agents, "progress_controller_state.json"), "w", encoding="utf-8") as f:
            json.dump(progress, f)
        monkeypatch.setattr(
            "evolution.terminal_stagnation._project_agents_dir",
            lambda: agents,
        )
        evo: dict = {"mutation_rate": 0.52}
        evo["mutation_rate"] = 0.912
        evo["adaptive_mutation_ceil"] = 0.95
        _apply_terminal_diversity_boost(1, evo, tmp)
        from run_multi_cycle_loop import cap_evo_meta_params_at_scan_floor

        cap_evo_meta_params_at_scan_floor(evo, tmp)
        assert evo["mutation_rate"] <= 0.78
        assert evo["mutation_rate"] >= 0.78
        assert evo["adaptive_mutation_ceil"] <= 0.78
        assert evo["diversity_inject_ratio"] <= 0.55


def test_terminal_diversity_boost_applies_when_promotion_raw_stalled_despite_epi():
    with tempfile.TemporaryDirectory() as tmp:
        agents = os.path.join(tmp, ".agents")
        os.makedirs(agents)
        tick_path = os.path.join(agents, "tick_state.json")
        with open(tick_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        with open(os.path.join(agents, "progress_controller_state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "level": 30,
                    "cycles_at_level": 100,
                    "verify_score": 1.0,
                    "raw_fitness": 264.04,
                },
                f,
            )
        traj = os.path.join(agents, "progress_trajectory.jsonl")
        with open(traj, "w", encoding="utf-8") as f:
            for i in range(200):
                mom = -0.5 if i < 120 else 1.5
                f.write(
                    json.dumps(
                        {
                            "cycle": i,
                            "epi_momentum": mom,
                            "fitness": 400.0 + i * 0.5,
                            "epi": 100.0 + i * 0.05,
                            "level": 30,
                        }
                    )
                    + "\n"
                )
        results = os.path.join(tmp, "multi_cycle_results.jsonl")
        with open(results, "w", encoding="utf-8") as f:
            for i in range(25):
                f.write(
                    json.dumps(
                        {
                            "cycle": 6000 + i,
                            "final_level": 30,
                            "promotion_raw_fitness": 321.85,
                            "champion_words": 61,
                        }
                    )
                    + "\n"
                )
        evo: dict = {"mutation_rate": 0.52}
        assert _apply_terminal_diversity_boost(2, evo, tmp) is True
        with open(tick_path, encoding="utf-8") as f:
            tick = json.load(f)
        assert tick.get("terminal_diversity_boost") is True
        assert evo["mutation_rate"] >= 0.82


def test_terminal_diversity_boost_auto_enables_on_certified_plateau():
    with tempfile.TemporaryDirectory() as tmp:
        agents = os.path.join(tmp, ".agents")
        os.makedirs(agents)
        tick_path = os.path.join(agents, "tick_state.json")
        with open(tick_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        with open(os.path.join(agents, "progress_controller_state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "level": 30,
                    "cycles_at_level": 100,
                    "verify_score": 1.0,
                    "raw_fitness": 264.04,
                },
                f,
            )
        evo: dict = {}
        assert _apply_terminal_diversity_boost(2, evo, tmp) is True
        with open(tick_path, encoding="utf-8") as f:
            tick = json.load(f)
        assert tick.get("terminal_diversity_boost") is True
        assert evo["mutation_rate"] >= 0.82
