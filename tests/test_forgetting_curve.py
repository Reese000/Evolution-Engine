#!/usr/bin/env python3
"""Tests for run_multi_cycle.compute_forgetting_curve.

The forgetting curve re-scores the champion on all previously-solved levels to
detect catastrophic forgetting after retrain. It was unwired/untested; these
cover its scoring branches with a mock VM and a patched task definition.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tasks_rv32i
from run_multi_cycle import compute_forgetting_curve


class _Result:
    def __init__(self, success=True, registers=None, cycles=0, error=None):
        self.success = success
        self.registers = registers or {}
        self.cycles = cycles
        self.error = error


class _VM:
    def __init__(self, result):
        self._result = result

    def evaluate(self, champion, inputs, budget=5000):
        return self._result


class _FakeTask:
    """Task whose instance expects register 3 == 10."""

    def __init__(self, seed=0):
        pass

    def generate_instance(self):
        return {"inputs": {}, "expected_output": {"registers": {"3": 10}}}


def _patch_task(monkeypatch):
    monkeypatch.setattr(tasks_rv32i, "get_task_level", lambda level: _FakeTask)


def test_perfect_match_scores_as_perfect(monkeypatch):
    _patch_task(monkeypatch)
    vm = _VM(_Result(success=True, registers={"3": 10}, cycles=0))
    scores = compute_forgetting_curve([0x0], champion_level=0, vm_farm=vm)
    # reg_match=1.0 -> 100 + 50 perfect bonus = 150 (>=140 => "perfect")
    assert scores == {0: 150.0}


def test_partial_match_scores_partial(monkeypatch):
    _patch_task(monkeypatch)
    vm = _VM(_Result(success=True, registers={"3": 0}, cycles=0))
    scores = compute_forgetting_curve([0x0], champion_level=0, vm_farm=vm)
    # diff=10, max_diff=10 -> 1/(1+1)=0.5 -> 50.0, no perfect bonus
    assert scores == {0: 50.0}


def test_crash_gets_penalty(monkeypatch):
    _patch_task(monkeypatch)
    vm = _VM(_Result(success=False))
    scores = compute_forgetting_curve([0x0], champion_level=0, vm_farm=vm)
    assert scores == {0: -50.0}


def test_evaluates_all_levels_up_to_champion(monkeypatch):
    _patch_task(monkeypatch)
    vm = _VM(_Result(success=True, registers={"3": 10}, cycles=0))
    scores = compute_forgetting_curve([0x0], champion_level=3, vm_farm=vm)
    assert set(scores.keys()) == {0, 1, 2, 3}
    assert all(v == 150.0 for v in scores.values())
