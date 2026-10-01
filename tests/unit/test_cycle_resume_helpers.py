"""Resume / cycle numbering helpers."""

import time

from run_multi_cycle import _is_valid_cycle_result, _max_cycle_in_results
from run_multi_cycle_loop import _build_combined_cycle_results


def test_valid_cycle_accepts_phase_total_seconds(tmp_path):
    row = {"elapsed_seconds": 5.0, "phase_total_seconds": 120.0}
    assert _is_valid_cycle_result(row) is True


def test_build_combined_sets_elapsed_to_phase_total():
    evo = {"cycle": 1, "elapsed_seconds": 12.0, "final_level": 13}
    retrain = {"retrained": False}
    retrain_start = time.time()
    combined = _build_combined_cycle_results(1, evo, retrain, 12.0, retrain_start)
    assert combined["elapsed_seconds"] == combined["phase_total_seconds"]
    assert combined["elapsed_seconds"] >= 12.0


def test_max_cycle_in_results(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(
        '{"cycle": 5}\n{"cycle": 12}\n{"cycle": 3}\n',
        encoding="utf-8",
    )
    assert _max_cycle_in_results(str(path)) == 12
