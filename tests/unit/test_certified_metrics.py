"""L30 certified metrics gate."""

import json
import os

os.environ.setdefault("EE_USE_GPU", "0")


def test_apply_l30_certified_metrics_restores_degraded(tmp_path, monkeypatch):
    from evolution.champion_rollback import apply_l30_certified_metrics_if_degraded

    results = tmp_path / "multi_cycle_results.jsonl"
    prog = ["0x00000013"] * 73
    row = {
        "cycle": 3511,
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 73,
        "final_fitness": 264.0,
        "champion_program": prog,
    }
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")

    class Eng:
        champion = [1] * 76
        champion_level = 30
        champion_trace = {}
        champion_fitness = 28.0
        _champion_verify_score = 0.842
        _champion_verified = False

        def _refresh_eval_budget(self):
            pass

        def _evaluate(self, p):
            return {"status": "ok", "final_regs": {}, "final_memory": {}, "instructions_count": len(p)}

    eng = Eng()
    metrics = {
        "final_level": 30,
        "final_fitness": 28.0,
        "champion_verify_score": 0.842,
        "champion_words": 76,
    }

    monkeypatch.setattr(
        "evolution.champion_rollback.find_best_certified_champion",
        lambda path, level, **kw: row,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.reporting_champion_fitness",
        lambda *a, **k: 264.0,
    )

    assert apply_l30_certified_metrics_if_degraded(str(tmp_path), eng, metrics) is True
    assert metrics["champion_words"] == 73
    assert metrics["final_fitness"] == 264.0
    assert metrics["champion_verify_score"] == 1.0


def test_apply_l30_certified_metrics_keeps_compact_fill(tmp_path, monkeypatch):
    from evolution.champion_rollback import apply_l30_certified_metrics_if_degraded

    class Eng:
        champion = [1] * 21
        champion_level = 30
        champion_trace = {"status": "ok"}
        champion_fitness = 100.0
        _champion_verify_score = 0.63
        _champion_verified = False

        def _refresh_eval_budget(self):
            pass

        def _evaluate(self, p):
            return {"status": "ok"}

    metrics = {
        "final_level": 30,
        "final_fitness": 100.0,
        "champion_verify_score": 0.63,
        "champion_words": 21,
    }
    monkeypatch.setattr(
        "evolution.champion_rollback.find_best_certified_champion",
        lambda path, level, **kw: {
            "cycle": 1,
            "champion_words": 13,
            "final_fitness": 2000.0,
            "champion_verify_score": 1.0,
            "champion_program": ["0x13"] * 13,
        },
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.reporting_champion_fitness",
        lambda *a, **k: 100.0,
    )
    assert apply_l30_certified_metrics_if_degraded(str(tmp_path), Eng(), metrics) is False
    assert metrics["champion_words"] == 21


def test_apply_l30_certified_metrics_restores_compact_capability_regression(tmp_path, monkeypatch):
    from evolution.champion_rollback import apply_l30_certified_metrics_if_degraded

    results = tmp_path / "multi_cycle_results.jsonl"
    good_prog = ["0x00008193"] * 56 + ["0x00100073"]
    row = {
        "cycle": 6061,
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 57,
        "final_fitness": 371.98,
        "promotion_raw_fitness": 321.645,
        "champion_program": good_prog,
    }
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")

    class Eng:
        champion = [1] * 48
        champion_level = 30
        champion_trace = {"status": "ok"}
        champion_fitness = 351.0
        _champion_verify_score = 1.0
        _champion_verified = True

        def _refresh_eval_budget(self):
            pass

        def _evaluate(self, p):
            return {
                "status": "ok",
                "final_regs": {},
                "final_memory": {},
                "instructions_count": len(p),
            }

    eng = Eng()
    metrics = {
        "final_level": 30,
        "final_fitness": 351.0,
        "champion_verify_score": 1.0,
        "champion_words": 48,
        "promotion_raw_fitness": 310.09,
        "champion_program": ["0x13"] * 48,
    }

    monkeypatch.setattr(
        "evolution.champion_rollback.find_best_certified_champion",
        lambda path, level, **kw: row,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.reporting_champion_fitness",
        lambda *a, **k: 371.98,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.end_of_cycle_champion_fitness",
        lambda *a, **k: 351.0,
    )

    assert apply_l30_certified_metrics_if_degraded(str(tmp_path), eng, metrics) is True
    assert metrics["champion_words"] == 57
    assert metrics["promotion_raw_fitness"] == 321.645


def test_apply_l30_certified_metrics_keeps_prom_raw_scale_when_not_degraded(tmp_path, monkeypatch):
    from evolution.champion_rollback import apply_l30_certified_metrics_if_degraded

    row = {
        "cycle": 6543,
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 48,
        "final_fitness": 881.0,
        "promotion_raw_fitness": 324.236,
        "champion_program": ["0x00008193"] * 47 + ["0x00100073"],
    }

    class Eng:
        champion = [0x13] * 47 + [0x00100073]
        champion_level = 30
        champion_trace = {"status": "ok"}
        champion_fitness = 881.0
        _champion_verify_score = 1.0

        def _refresh_eval_budget(self):
            pass

        def _evaluate(self, p):
            return {"status": "ok", "instructions_count": len(p)}

    metrics = {
        "final_level": 30,
        "final_fitness": 324.31,
        "champion_verify_score": 1.0,
        "champion_words": 48,
        "promotion_raw_fitness": 324.236,
        "champion_program": ["0x13"] * 48,
    }
    monkeypatch.setattr(
        "evolution.champion_rollback.find_best_certified_champion",
        lambda path, level, **kw: row,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.end_of_cycle_champion_fitness",
        lambda *a, **k: 324.31,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.reporting_champion_fitness",
        lambda *a, **k: 881.0,
    )
    assert apply_l30_certified_metrics_if_degraded(str(tmp_path), Eng(), metrics) is False
    assert metrics["final_fitness"] == 324.236


def test_apply_l30_certified_metrics_without_champion_trace(tmp_path, monkeypatch):
    """Missing champion_trace must not leave reporting-scale final_fitness in metrics."""
    from evolution.champion_rollback import apply_l30_certified_metrics_if_degraded

    row = {
        "cycle": 6543,
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 48,
        "final_fitness": 881.0,
        "promotion_raw_fitness": 324.236,
        "champion_program": ["0x00008193"] * 47 + ["0x00100073"],
    }

    class Eng:
        champion = [0x13] * 47 + [0x00100073]
        champion_level = 30
        champion_trace = None
        champion_fitness = 346.7
        _champion_verify_score = 1.0

        def _refresh_eval_budget(self):
            pass

        def _evaluate(self, p):
            return {"status": "ok", "instructions_count": len(p)}

    metrics = {
        "final_level": 30,
        "final_fitness": 346.735,
        "champion_verify_score": 1.0,
        "champion_words": 48,
        "promotion_raw_fitness": 324.236,
        "champion_program": ["0x13"] * 48,
    }
    monkeypatch.setattr(
        "evolution.champion_rollback.find_best_certified_champion",
        lambda path, level, **kw: row,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.end_of_cycle_champion_fitness",
        lambda *a, **k: 324.31,
    )
    eng = Eng()
    assert apply_l30_certified_metrics_if_degraded(str(tmp_path), eng, metrics) is False
    assert metrics["final_fitness"] == 324.236
    assert eng.champion_trace is not None


def test_apply_l30_certified_metrics_restores_short_degraded_verify(tmp_path, monkeypatch):
    from evolution.champion_rollback import apply_l30_certified_metrics_if_degraded

    row = {
        "cycle": 6470,
        "final_level": 30,
        "champion_verify_score": 1.0,
        "champion_words": 61,
        "final_fitness": 417.4,
        "promotion_raw_fitness": 323.639,
        "champion_program": ["0x00008193"] * 60 + ["0x00100073"],
    }

    class Eng:
        champion = [0x13] * 22 + [0x00100073]
        champion_level = 30
        champion_trace = {"status": "ok"}
        champion_fitness = 157.0
        _champion_verify_score = 0.736

        def _refresh_eval_budget(self):
            pass

        def _evaluate(self, p):
            return {"status": "ok", "instructions_count": len(p)}

    metrics = {
        "final_level": 30,
        "final_fitness": 157.0,
        "champion_verify_score": 0.736,
        "champion_words": 23,
    }
    monkeypatch.setattr(
        "evolution.champion_rollback.find_best_certified_champion",
        lambda path, level, **kw: row,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.reporting_champion_fitness",
        lambda *a, **k: 417.4,
    )
    assert apply_l30_certified_metrics_if_degraded(str(tmp_path), Eng(), metrics) is True
    assert metrics["champion_words"] == 61
    assert metrics["promotion_raw_fitness"] == 323.639
