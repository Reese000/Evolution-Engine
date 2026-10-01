#!/usr/bin/env python3
"""Tests for enhanced progress report generation (schema v1.0).

Validates that progress_report.json is generated with all expected keys,
matches the JSON schema, and handles edge cases like empty metrics history.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

# Path setup


# ══════════════════════════════════════════════════════════════════════
# Schema validation
# ══════════════════════════════════════════════════════════════════════


class TestProgressReportSchema:
    """Validate the JSON schema file itself."""

    def test_schema_file_exists(self):
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "progress_report_schema.json",
        )
        assert os.path.isfile(schema_path), f"Schema file not found: {schema_path}"

    def test_schema_is_valid_json(self):
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "progress_report_schema.json",
        )
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        assert isinstance(schema, dict)

    def test_schema_has_required_top_level_keys(self):
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "progress_report_schema.json",
        )
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        assert schema.get("title") == "Evolution Engine Progress Report"
        assert schema.get("schema_version") == "1.0" or "1.0" in str(
            schema.get("properties", {}).get("schema_version", {})
        )
        required = schema.get("required", [])
        expected_required = [
            "generated_at",
            "cycle",
            "schema_version",
            "summary",
            "fitness",
            "level",
            "retrain",
            "diversity",
            "time_breakdown",
            "errors",
            "trends",
            "latest_cycle",
            "resource_usage",
        ]
        for key in expected_required:
            assert key in required, f"Missing required key: {key}"

    def test_schema_trend_stats_def_exists(self):
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "progress_report_schema.json",
        )
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        defs = schema.get("$defs", {})
        assert "trend_stats" in defs, "Missing $defs/trend_stats"
        trend_props = defs["trend_stats"].get("properties", {})
        for field in ["current", "avg", "min", "max", "trend", "window_size"]:
            assert field in trend_props, f"Missing trend_stats field: {field}"


# ══════════════════════════════════════════════════════════════════════
# Helpers: build a minimal runner mock
# ══════════════════════════════════════════════════════════════════════


def _make_mock_runner(
    state: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    trend_result: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a mock AutonomousRunner with the attributes needed by
    _generate_progress_report().
    """
    runner = MagicMock()
    runner._state = state or {
        "best_fitness": 0.0,
        "best_level": 0,
        "cycles_completed": 0,
        "total_errors": 0,
        "consecutive_errors": 0,
        "error_types": {},
        "start_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    runner._config = config or {
        "evolution": {"population_size": 50},
    }
    runner._health = MagicMock()
    runner._health._start_time = datetime.now(timezone.utc).timestamp()

    # MetricsCollector mock
    runner._metrics = MagicMock()
    runner._metrics.get_history.return_value = history or []

    if trend_result is None:
        trend_result = {
            "current": None,
            "avg": None,
            "min": None,
            "max": None,
            "trend": "unknown",
            "window_size": 0,
        }
    runner._metrics.get_trend.return_value = trend_result

    return runner


def _build_report_via_real_method(
    runner_mock: MagicMock,
    cycle: int,
    cycle_metrics: dict[str, Any],
    tmpdir: str,
) -> dict[str, Any]:
    """Call _generate_progress_report on a real AutonomousRunner instance
    (using the mock's attributes) and return the generated report dict.

    Imports AutonomousRunner and patches its PROJECT_ROOT so the report
    writes to tmpdir.
    """
    from autonomous_runner import AutonomousRunner

    # Create a minimal real runner bypassing __init__
    real_runner = AutonomousRunner.__new__(AutonomousRunner)
    real_runner._state = runner_mock._state
    real_runner._config = runner_mock._config
    real_runner._health = runner_mock._health
    real_runner._metrics = runner_mock._metrics

    # Patch PROJECT_ROOT inside the method so it writes to tmpdir
    with patch("autonomous_runner.PROJECT_ROOT", tmpdir):
        real_runner._generate_progress_report(cycle, cycle_metrics)

    report_path = os.path.join(tmpdir, "progress_report.json")
    assert os.path.isfile(report_path), "progress_report.json was not created"
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
# Report structure tests
# ══════════════════════════════════════════════════════════════════════


class TestProgressReportStructure:
    """Verify the report has all expected top-level and nested keys."""

    def test_all_top_level_keys_present(self, tmp_path):
        runner_mock = _make_mock_runner()
        cycle_metrics = {
            "cycle": 1,
            "fitness": 0.5,
            "level": 1,
            "best_fitness": 0.5,
            "best_level": 1,
            "val_loss": None,
            "retrained": False,
            "population_size": 50,
            "cycle_time_seconds": 1.23,
            "errors": 0,
            "evolution_seconds": 0.8,
            "retrain_seconds": 0.3,
        }
        report = _build_report_via_real_method(runner_mock, 1, cycle_metrics, str(tmp_path))

        expected_keys = [
            "generated_at",
            "cycle",
            "schema_version",
            "summary",
            "fitness",
            "level",
            "retrain",
            "diversity",
            "time_breakdown",
            "errors",
            "trends",
            "latest_cycle",
            "resource_usage",
        ]
        for key in expected_keys:
            assert key in report, f"Missing top-level key: {key}"

    def test_schema_version_is_1_0(self, tmp_path):
        runner_mock = _make_mock_runner()
        report = _build_report_via_real_method(
            runner_mock,
            1,
            {
                "cycle": 1,
                "fitness": 0.0,
                "level": 0,
                "best_fitness": 0.0,
                "best_level": 0,
                "val_loss": None,
                "retrained": False,
                "population_size": 50,
                "cycle_time_seconds": 0.0,
                "errors": 0,
                "evolution_seconds": 0.0,
                "retrain_seconds": 0.0,
            },
            str(tmp_path),
        )
        assert report["schema_version"] == "1.0"

    def test_summary_keys(self, tmp_path):
        runner_mock = _make_mock_runner(
            state={
                "best_fitness": 1.5,
                "best_level": 3,
                "cycles_completed": 10,
                "total_errors": 2,
                "start_timestamp": "2026-01-01T00:00:00",
            }
        )
        report = _build_report_via_real_method(
            runner_mock,
            10,
            {
                "cycle": 10,
                "fitness": 1.5,
                "level": 3,
                "best_fitness": 1.5,
                "best_level": 3,
                "val_loss": 0.42,
                "retrained": True,
                "population_size": 50,
                "cycle_time_seconds": 2.1,
                "errors": 2,
                "evolution_seconds": 1.5,
                "retrain_seconds": 0.5,
            },
            str(tmp_path),
        )
        summary = report["summary"]
        assert summary["best_fitness"] == 1.5
        assert summary["best_level"] == 3
        assert summary["cycles_completed"] == 10
        assert summary["total_errors"] == 2
        assert "uptime_seconds" in summary
        assert summary["start_timestamp"] == "2026-01-01T00:00:00"

    def test_fitness_section_keys(self, tmp_path):
        runner_mock = _make_mock_runner(
            history=[
                {"fitness": 0.1, "level": 0},
                {"fitness": 0.3, "level": 1},
                {"fitness": 0.5, "level": 1},
            ]
        )
        report = _build_report_via_real_method(
            runner_mock,
            3,
            {
                "cycle": 3,
                "fitness": 0.5,
                "level": 1,
                "best_fitness": 0.5,
                "best_level": 1,
                "val_loss": None,
                "retrained": False,
                "population_size": 50,
                "cycle_time_seconds": 1.0,
                "errors": 0,
                "evolution_seconds": 0.8,
                "retrain_seconds": 0.2,
            },
            str(tmp_path),
        )
        fitness = report["fitness"]
        assert "trajectory" in fitness
        assert "trend" in fitness
        assert "best_ever" in fitness
        assert fitness["trajectory"] == [0.1, 0.3, 0.5]
        assert fitness["best_ever"] == 0.5

    def test_level_section_keys(self, tmp_path):
        runner_mock = _make_mock_runner(
            history=[
                {"fitness": 0.1, "level": 0},
                {"fitness": 0.3, "level": 1},
            ]
        )
        report = _build_report_via_real_method(
            runner_mock,
            2,
            {
                "cycle": 2,
                "fitness": 0.3,
                "level": 1,
                "best_fitness": 0.3,
                "best_level": 1,
                "val_loss": None,
                "retrained": False,
                "population_size": 50,
                "cycle_time_seconds": 1.0,
                "errors": 0,
                "evolution_seconds": 0.5,
                "retrain_seconds": 0.3,
            },
            str(tmp_path),
        )
        level = report["level"]
        assert "progression_history" in level
        assert "trend" in level
        assert level["progression_history"] == [0, 1]

    def test_retrain_section_keys(self, tmp_path):
        runner_mock = _make_mock_runner(
            history=[
                {"val_loss": 0.9, "retrained": True},
                {"val_loss": None, "retrained": False},
                {"val_loss": 0.7, "retrained": True},
            ]
        )
        report = _build_report_via_real_method(
            runner_mock,
            3,
            {
                "cycle": 3,
                "fitness": 0.5,
                "level": 1,
                "best_fitness": 0.5,
                "best_level": 1,
                "val_loss": 0.7,
                "retrained": True,
                "population_size": 50,
                "cycle_time_seconds": 1.0,
                "errors": 0,
                "evolution_seconds": 0.5,
                "retrain_seconds": 0.3,
            },
            str(tmp_path),
        )
        retrain = report["retrain"]
        assert retrain["val_loss_history"] == [0.9, None, 0.7]
        assert retrain["last_val_loss"] == 0.7
        assert retrain["retrain_count"] == 2

    def test_diversity_section_all_none(self, tmp_path):
        runner_mock = _make_mock_runner()
        report = _build_report_via_real_method(
            runner_mock,
            1,
            {
                "cycle": 1,
                "fitness": 0.0,
                "level": 0,
                "best_fitness": 0.0,
                "best_level": 0,
                "val_loss": None,
                "retrained": False,
                "population_size": 50,
                "cycle_time_seconds": 0.0,
                "errors": 0,
                "evolution_seconds": 0.0,
                "retrain_seconds": 0.0,
            },
            str(tmp_path),
        )
        div = report["diversity"]
        assert div["unique_individuals"] is None
        assert div["qd_occupied"] is None
        assert div["qd_total"] is None
        assert div["diversity_score"] is None
        assert div["latest_diversity"] is None

    def test_time_breakdown_from_cycle_metrics(self, tmp_path):
        runner_mock = _make_mock_runner(
            history=[
                {"cycle_time_seconds": 1.0},
                {"cycle_time_seconds": 2.0},
                {"cycle_time_seconds": 3.0},
            ]
        )
        report = _build_report_via_real_method(
            runner_mock,
            3,
            {
                "cycle": 3,
                "fitness": 0.5,
                "level": 1,
                "best_fitness": 0.5,
                "best_level": 1,
                "val_loss": None,
                "retrained": False,
                "population_size": 50,
                "cycle_time_seconds": 3.0,
                "errors": 0,
                "evolution_seconds": 2.0,
                "retrain_seconds": 0.8,
            },
            str(tmp_path),
        )
        tb = report["time_breakdown"]
        assert tb["evolution_seconds"] == 2.0
        assert tb["retrain_seconds"] == 0.8
        assert tb["total_cycle_seconds"] == 3.0
        assert tb["avg_cycle_seconds"] == 2.0

    def test_errors_section_with_types(self, tmp_path):
        runner_mock = _make_mock_runner(
            state={
                "best_fitness": 0.5,
                "best_level": 1,
                "cycles_completed": 5,
                "total_errors": 3,
                "consecutive_errors": 1,
                "error_types": {"RuntimeError": 2, "MemoryError": 1},
                "start_timestamp": "2026-01-01T00:00:00",
            }
        )
        report = _build_report_via_real_method(
            runner_mock,
            5,
            {
                "cycle": 5,
                "fitness": 0.5,
                "level": 1,
                "best_fitness": 0.5,
                "best_level": 1,
                "val_loss": None,
                "retrained": False,
                "population_size": 50,
                "cycle_time_seconds": 1.0,
                "errors": 3,
                "evolution_seconds": 0.5,
                "retrain_seconds": 0.3,
            },
            str(tmp_path),
        )
        errors = report["errors"]
        assert errors["total_count"] == 3
        assert errors["consecutive_errors"] == 1
        assert errors["types"]["RuntimeError"] == 2
        assert errors["types"]["MemoryError"] == 1

    def test_trends_top_level_backward_compatible(self, tmp_path):
        trend = {
            "current": 0.5,
            "avg": 0.4,
            "min": 0.1,
            "max": 0.5,
            "trend": "improving",
            "window_size": 3,
        }
        runner_mock = _make_mock_runner(trend_result=trend)
        report = _build_report_via_real_method(
            runner_mock,
            3,
            {
                "cycle": 3,
                "fitness": 0.5,
                "level": 1,
                "best_fitness": 0.5,
                "best_level": 1,
                "val_loss": None,
                "retrained": False,
                "population_size": 50,
                "cycle_time_seconds": 1.0,
                "errors": 0,
                "evolution_seconds": 0.5,
                "retrain_seconds": 0.3,
            },
            str(tmp_path),
        )
        assert "trends" in report
        assert "fitness" in report["trends"]
        assert "level" in report["trends"]
        assert report["trends"]["fitness"]["trend"] == "improving"

    def test_latest_cycle_preserved(self, tmp_path):
        cycle_metrics = {
            "cycle": 7,
            "fitness": 2.1,
            "level": 4,
            "best_fitness": 2.1,
            "best_level": 4,
            "val_loss": 0.3,
            "retrained": True,
            "population_size": 50,
            "cycle_time_seconds": 1.5,
            "errors": 0,
            "evolution_seconds": 1.0,
            "retrain_seconds": 0.4,
        }
        runner_mock = _make_mock_runner()
        report = _build_report_via_real_method(
            runner_mock,
            7,
            cycle_metrics,
            str(tmp_path),
        )
        assert report["latest_cycle"] == cycle_metrics


# ══════════════════════════════════════════════════════════════════════
# Edge case tests
# ══════════════════════════════════════════════════════════════════════


class TestProgressReportEdgeCases:
    """Test behavior with empty or minimal data."""

    def test_empty_history_produces_empty_trajectories(self, tmp_path):
        runner_mock = _make_mock_runner(history=[])
        report = _build_report_via_real_method(
            runner_mock,
            1,
            {
                "cycle": 1,
                "fitness": 0.0,
                "level": 0,
                "best_fitness": 0.0,
                "best_level": 0,
                "val_loss": None,
                "retrained": False,
                "population_size": 50,
                "cycle_time_seconds": 0.0,
                "errors": 0,
                "evolution_seconds": 0.0,
                "retrain_seconds": 0.0,
            },
            str(tmp_path),
        )
        assert report["fitness"]["trajectory"] == []
        assert report["level"]["progression_history"] == []
        assert report["retrain"]["val_loss_history"] == []
        assert report["retrain"]["retrain_count"] == 0
        assert report["fitness"]["best_ever"] == 0.0
        assert report["time_breakdown"]["avg_cycle_seconds"] is None

    def test_history_records_missing_keys_handled(self, tmp_path):
        # Records with no fitness/level keys should be skipped in trajectory
        runner_mock = _make_mock_runner(
            history=[
                {"cycle": 1},  # missing fitness and level
                {"fitness": 0.5},  # missing level
                {"level": 2},  # missing fitness
                {"fitness": 0.8, "level": 3},
            ]
        )
        report = _build_report_via_real_method(
            runner_mock,
            4,
            {
                "cycle": 4,
                "fitness": 0.8,
                "level": 3,
                "best_fitness": 0.8,
                "best_level": 3,
                "val_loss": None,
                "retrained": False,
                "population_size": 50,
                "cycle_time_seconds": 1.0,
                "errors": 0,
                "evolution_seconds": 0.5,
                "retrain_seconds": 0.3,
            },
            str(tmp_path),
        )
        assert report["fitness"]["trajectory"] == [0.5, 0.8]
        assert report["level"]["progression_history"] == [2, 3]

    def test_missing_error_types_key_handled(self, tmp_path):
        state = {
            "best_fitness": 1.0,
            "best_level": 1,
            "cycles_completed": 1,
            "total_errors": 0,
            "consecutive_errors": 0,
            "start_timestamp": "2026-01-01T00:00:00",
            # Note: no error_types key
        }
        runner_mock = _make_mock_runner(state=state)
        report = _build_report_via_real_method(
            runner_mock,
            1,
            {
                "cycle": 1,
                "fitness": 1.0,
                "level": 1,
                "best_fitness": 1.0,
                "best_level": 1,
                "val_loss": None,
                "retrained": False,
                "population_size": 50,
                "cycle_time_seconds": 1.0,
                "errors": 0,
                "evolution_seconds": 0.5,
                "retrain_seconds": 0.3,
            },
            str(tmp_path),
        )
        assert report["errors"]["types"] == {}

    def test_null_timing_fields_handled(self, tmp_path):
        # When evolution_seconds / retrain_seconds are not present in cycle_metrics
        runner_mock = _make_mock_runner()
        report = _build_report_via_real_method(
            runner_mock,
            1,
            {
                "cycle": 1,
                "fitness": 0.0,
                "level": 0,
                "best_fitness": 0.0,
                "best_level": 0,
                "val_loss": None,
                "retrained": False,
                "population_size": 50,
                "cycle_time_seconds": 1.0,
                "errors": 0,
            },
            str(tmp_path),
        )
        assert report["time_breakdown"]["evolution_seconds"] is None
        assert report["time_breakdown"]["retrain_seconds"] is None
        assert report["time_breakdown"]["total_cycle_seconds"] == 1.0


# ══════════════════════════════════════════════════════════════════════
# Error type tracking tests
# ══════════════════════════════════════════════════════════════════════


class TestLogErrorTracking:
    """Verify log_error() tracks error types correctly."""

    def test_log_error_increments_error_types(self):
        from autonomous_runner import AutonomousRunner

        runner = AutonomousRunner.__new__(AutonomousRunner)
        runner._state = {
            "error_count": 0,
            "total_errors": 0,
            "consecutive_errors": 0,
        }
        runner._health = MagicMock()

        # Mock the log function
        with patch("autonomous_runner.log"):
            runner.log_error(RuntimeError("test error"), 1)
            runner.log_error(RuntimeError("test error 2"), 2)
            runner.log_error(MemoryError("oom"), 3)

        assert runner._state["error_count"] == 3
        assert runner._state["total_errors"] == 3
        assert runner._state["consecutive_errors"] == 3
        assert runner._state["error_types"]["RuntimeError"] == 2
        assert runner._state["error_types"]["MemoryError"] == 1

    def test_default_state_includes_error_types(self):
        from autonomous_runner import AutonomousRunner

        runner = AutonomousRunner.__new__(AutonomousRunner)
        runner._engine_config = MagicMock()
        runner._engine_config.retrain = MagicMock()
        runner._engine_config.retrain.retrain_lr = 0.001
        runner._engine_config.retrain.retrain_freeze_layers = 5
        runner._engine_config.retrain.retrain_max_per_level = 3
        runner._engine_config.retrain.retrain_model_size = "small"
        runner._engine_config.retrain.retrain_timeout = 300
        runner._engine_config.retrain.retrain_patience = 5
        runner._engine_config.retrain.retrain_max_epochs = 20
        runner._engine_config.retrain.retrain_grad_clip = 1.0
        runner._engine_config.retrain.retrain_batch_size = 32
        runner._engine_config.retrain.retrain_weight_decay = 0.0
        runner._engine_config.retrain.retrain_min_examples = 10
        runner._engine_config.retrain.retrain_val_split = 0.2
        runner._config = {"evolution": {"population_size": 50}}

        with patch("autonomous_runner.M1_CHECKPOINTS", "/tmp/test"):
            state = runner._default_state()

        assert "error_types" in state
        assert state["error_types"] == {}
