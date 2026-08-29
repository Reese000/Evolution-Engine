#!/usr/bin/env python3
"""Regression test for the default checkpoint directory in EvolutionEngine.

Bug: _checkpoint() defaulted cp_dir to os.path.join(SCRIPT_DIR, "..",
"milestone1", "checkpoints"). With evolution_engine.py at the repo root,
SCRIPT_DIR is the repo root, so ".." escaped to <repo-parent>/milestone1
(e.g. /milestone1) and raised PermissionError whenever the engine ran without
an explicit log_dir (e.g. benchmarks/verify_v13_features.py's full cycle).
The default must resolve to <repo>/milestone1/checkpoints instead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import PROJECT_ROOT
from evolution_engine import EvolutionConfig, EvolutionEngine


def test_default_checkpoint_dir_stays_in_repo(tmp_path):
    # log_dir="" forces the default checkpoint-dir path (the buggy branch);
    # checkpoint_interval=1 ensures _checkpoint() actually runs.
    cfg = EvolutionConfig(
        population_size=6,
        max_generations=2,
        use_gpu=False,
        log_dir="",
        champion_dir=str(tmp_path / "champions"),
        checkpoint_interval=1,
        use_novelty_archive=False,
        use_qd_archive=False,
        use_strategy_archive=False,
        use_secondary_tasks=False,
        seed=42,
    )
    engine = EvolutionEngine(config=cfg)
    # Previously raised PermissionError: '/milestone1/checkpoints'.
    engine.run()

    cp = engine._cached_checkpoint_dir
    assert cp is not None, "checkpoint dir should be resolved after a run"
    expected = os.path.join(str(PROJECT_ROOT), "milestone1", "checkpoints")
    assert os.path.normpath(cp) == os.path.normpath(expected)
    # And it must stay inside the repo (no '..' escape).
    assert os.path.normpath(cp).startswith(os.path.normpath(str(PROJECT_ROOT)) + os.sep)
