#!/usr/bin/env python3
"""Tests for run_multi_cycle.compute_weight_delta.

compute_weight_delta reports per-parameter L2 norms of the weight change between
two checkpoints (a retrain-analysis utility). It was unwired and untested;
these lock down its contract and confirm no latent bug (unwired code is exactly
where the earlier fitness_benchmark constructor bug hid).
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_multi_cycle import compute_weight_delta


def _save(path, state_dict):
    torch.save({"model_state_dict": state_dict}, path)


def test_missing_files_return_empty(tmp_path):
    a = str(tmp_path / "a.pt")
    b = str(tmp_path / "b.pt")
    assert compute_weight_delta(a, b) == {}
    _save(a, {"w": torch.zeros(2)})
    # after_path still missing
    assert compute_weight_delta(a, b) == {}


def test_identical_checkpoints_zero_delta(tmp_path):
    p = str(tmp_path / "c.pt")
    _save(p, {"w": torch.tensor([3.0, 4.0])})
    assert compute_weight_delta(p, p) == {"w": 0.0}


def test_known_l2_delta(tmp_path):
    pb = str(tmp_path / "before.pt")
    pa = str(tmp_path / "after.pt")
    _save(pb, {"w": torch.tensor([0.0, 0.0])})
    _save(pa, {"w": torch.tensor([3.0, 4.0])})  # L2 of delta = 5.0
    assert compute_weight_delta(pb, pa) == {"w": 5.0}


def test_shape_mismatch_is_skipped(tmp_path):
    pb = str(tmp_path / "before.pt")
    pa = str(tmp_path / "after.pt")
    _save(pb, {"w": torch.zeros(2), "x": torch.zeros(2)})
    _save(pa, {"w": torch.zeros(3), "x": torch.ones(2)})  # w: shape differs -> skip; x: L2 = sqrt(2)
    d = compute_weight_delta(pb, pa)
    assert "w" not in d
    assert d["x"] == round(math.sqrt(2), 4)


def test_bare_state_dict_without_wrapper(tmp_path):
    # Falls back to treating the loaded object as the state dict itself.
    pb = str(tmp_path / "before.pt")
    pa = str(tmp_path / "after.pt")
    torch.save({"w": torch.zeros(2)}, pb)
    torch.save({"w": torch.tensor([6.0, 8.0])}, pa)  # L2 = 10.0
    assert compute_weight_delta(pb, pa) == {"w": 10.0}
