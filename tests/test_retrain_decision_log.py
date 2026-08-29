#!/usr/bin/env python3
"""Tests for the shared retrain-decision log helper (run_multi_cycle).

build_retrain_decision / write_retrain_decision are shared by run_multi_cycle
and autonomous_runner so both emit the same retrain_decisions.jsonl schema
(observability parity). Budget/gate fields come from the retrain_result.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_multi_cycle import build_retrain_decision, write_retrain_decision

_EXPECTED_KEYS = {
    "cycle",
    "decision",
    "reason",
    "retrained",
    "fitness",
    "level",
    "words",
    "cycles_since_last_retrain",
    "champion_traces",
    "requested_max_epochs",
    "effective_max_epochs",
    "retrain_lr",
    "retrain_freeze_layers",
    "val_loss",
    "gate_accepted",
    "timestamp",
}


def test_attempted_decision_carries_budget_and_gate():
    rr = {
        "retrained": True,
        "reason": "ok",
        "requested_max_epochs": 1,
        "effective_max_epochs": 1,
        "retrain_lr": 3e-5,
        "retrain_freeze_layers": 0,
        "val_loss": 3.14,
        "gate_accepted": True,
    }
    rec = build_retrain_decision(rr_cycle := 5, rr, fitness=412.2, level=10, words=6, champion_traces=42)
    assert set(rec.keys()) == _EXPECTED_KEYS
    assert rec["cycle"] == 5
    assert rec["decision"] == "attempted"
    assert rec["retrained"] is True
    assert rec["fitness"] == 412.2
    assert rec["level"] == 10
    assert rec["words"] == 6
    assert rec["champion_traces"] == 42
    assert rec["effective_max_epochs"] == 1
    assert rec["val_loss"] == 3.14
    assert rec["gate_accepted"] is True


def test_skipped_decision():
    rr = {"retrained": False, "reason": "champion_unchanged"}
    rec = build_retrain_decision(2, rr)
    assert rec["decision"] == "skipped"
    assert rec["reason"] == "champion_unchanged"
    # Missing budget/gate fields default to None (not KeyError).
    assert rec["effective_max_epochs"] is None
    assert rec["gate_accepted"] is None


def test_write_appends_jsonl(tmp_path):
    path = str(tmp_path / "retrain_decisions.jsonl")
    write_retrain_decision(1, {"retrained": True, "val_loss": 2.0}, path=path, fitness=1.0, level=1)
    write_retrain_decision(2, {"retrained": False, "reason": "no_data"}, path=path)
    lines = [json.loads(x) for x in open(path)]
    assert len(lines) == 2
    assert lines[0]["cycle"] == 1 and lines[0]["decision"] == "attempted"
    assert lines[1]["cycle"] == 2 and lines[1]["decision"] == "skipped"
