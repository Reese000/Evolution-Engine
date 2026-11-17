"""QD population fill skips inject/probe slots (illumination only)."""

from types import SimpleNamespace

from paths import ensure_project_on_path

ensure_project_on_path()

from evolution.qd_archive_mixin import QDArchiveMixin
from evolution_engine import compute_qd_axes
from qd_archive import MAPElitesArchive

_ADDI_X1 = 0x00100093
_ADDI_X2 = 0x00200113
_EBREAK = 0x00100073


class _Engine(QDArchiveMixin):
    def __init__(self):
        self.qd_archive = MAPElitesArchive(bins=8)
        self.config = SimpleNamespace(use_qd_archive=True, qd_bonus_weight=0.1)
        self._qd_coverage = 0.0
        self.logs: list[str] = []

    def _log(self, msg: str) -> None:
        self.logs.append(msg)


def _trace(active_regs: int) -> dict:
    regs = {i: (1 if i <= active_regs else 0) for i in range(1, 32)}
    return {"status": "ok", "final_regs": regs}


def test_qd_fill_skips_inject_and_probe_strategies():
    eng = _Engine()
    varied = [
        [_ADDI_X1, _EBREAK],
        [_ADDI_X1, _ADDI_X2, _EBREAK],
    ]
    traces = [_trace(2), _trace(8)]
    strategies = ["DIVERSITY_INJECT", "RANDOM"]
    added_skip = eng._fill_empty_qd_cells_from_evaluated(varied, traces, strategies)
    eng2 = _Engine()
    added_all = eng2._fill_empty_qd_cells_from_evaluated(varied, traces, None)
    assert added_skip < added_all
    assert added_all >= 1


def test_probe_strategy_skipped():
    eng = _Engine()
    prog = [_ADDI_X2, _EBREAK]
    trace = _trace(6)
    desc = compute_qd_axes(prog, trace)
    assert desc is not None
    assert eng.qd_archive.cell_is_empty(desc)
    added = eng._fill_empty_qd_cells_from_evaluated([prog], [trace], ["TERMINAL_VERIFY_PROBE"])
    assert added == 0
    assert eng.qd_archive.cell_is_empty(desc)
