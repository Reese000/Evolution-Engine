"""Population ingest fills empty MAP-Elites cells without replacing occupied ones."""

from types import SimpleNamespace

from paths import ensure_project_on_path

ensure_project_on_path()

from evolution.qd_archive_mixin import QDArchiveMixin
from evolution_engine import compute_qd_axes
from qd_archive import MAPElitesArchive

# ADDI x1, x0, 1  and  EBREAK-like ebreak 0x00100073
_ADDI_X1 = 0x00100093
_ADDI_X2 = 0x00200113
_LUI_X3 = 0x000031B7
_JAL_X0 = 0x0000006F
_EBREAK = 0x00100073


class _Engine(QDArchiveMixin):
    def __init__(self):
        self.qd_archive = MAPElitesArchive(bins=8)
        self.config = SimpleNamespace(use_qd_archive=True, qd_bonus_weight=0.1)
        self._qd_coverage = 0.0
        self.logs: list[str] = []
        self.generations_stalled = 0
        self.champion = None
        self.champion_level = 0

    def _log(self, msg: str) -> None:
        self.logs.append(msg)

    def _compute_fitness(self, _program, _trace) -> float:
        return 12.0


def _trace(active_regs: int) -> dict:
    regs = {i: (1 if i <= active_regs else 0) for i in range(1, 32)}
    return {"status": "ok", "final_regs": regs}


def test_cell_is_empty_then_occupied():
    archive = MAPElitesArchive(bins=8)
    desc = compute_qd_axes([_ADDI_X1, _EBREAK], _trace(2))
    assert desc is not None
    assert archive.cell_is_empty(desc)
    assert archive.add_solution(desc, 10.0, [_ADDI_X1, _EBREAK], _trace(2))
    assert not archive.cell_is_empty(desc)
    assert archive.size == 1


def test_population_fills_distinct_empty_cells():
    eng = _Engine()
    champ = [_ADDI_X1, _EBREAK]
    champ_trace = _trace(2)
    champ_desc = compute_qd_axes(champ, champ_trace)
    assert champ_desc is not None
    eng.qd_archive.add_solution(champ_desc, 100.0, champ, champ_trace)

    varied = [
        champ,
        [_ADDI_X1, _ADDI_X2, _LUI_X3, _JAL_X0, _EBREAK],
        [_JAL_X0, _LUI_X3, _ADDI_X2, _EBREAK],
    ]
    traces = [_trace(2), _trace(12), _trace(28)]
    added = eng._fill_empty_qd_cells_from_evaluated(varied, traces)
    assert added >= 1
    assert eng.qd_archive.size >= 2
    # Champion cell still holds the high-fitness incumbent.
    cell = eng.qd_archive._grid[eng.qd_archive._cell_index(champ_desc)]
    assert cell["fitness"] == 100.0
    assert cell["program"] == champ


def test_fill_exception_does_not_raise():
    eng = _Engine()

    def boom(*_a, **_k):
        raise RuntimeError("descriptor failed")

    eng.qd_archive.cell_is_empty = boom  # type: ignore[method-assign]
    added = eng._fill_empty_qd_cells_from_evaluated(
        [[_ADDI_X1, _EBREAK]],
        [_trace(2)],
    )
    assert added == 0

    eng = _Engine()
    added = eng._fill_empty_qd_cells_from_evaluated(
        [[_ADDI_X1, _EBREAK]],
        [{"status": "invalid", "final_regs": {1: 1}}],
    )
    assert added == 0
    assert eng.qd_archive.size == 0


def test_qd_hybrids_injected_into_population_tail():
    eng = _Engine()
    eng.generations_stalled = 8
    eng.champion_level = 30
    eng.champion = [_ADDI_X1, _EBREAK]
    eng.rng = __import__("random").Random(0)
    eng.config = SimpleNamespace(
        use_qd_archive=True,
        qd_bonus_weight=0.1,
        champion_crossover_prob=1.0,
    )
    a = [_ADDI_X1, _EBREAK]
    b = [_ADDI_X1, _ADDI_X2, _LUI_X3, _JAL_X0, _EBREAK]
    eng.qd_archive.add_solution(compute_qd_axes(a, _trace(2)), 100.0, a, _trace(2))
    eng.qd_archive.add_solution(compute_qd_axes(b, _trace(12)), 50.0, b, _trace(12))
    pop = [[_ADDI_X1, _EBREAK] for _ in range(16)]
    eng._last_challenger_strategies = ["RANDOM"] * 16
    n = eng._inject_qd_archive_hybrids(pop)
    assert n >= 1
    labels = set(eng._last_challenger_strategies)
    assert "QD_CELL" in labels or "QD_CROSSOVER" in labels
    assert any(p != [_ADDI_X1, _EBREAK] for p in pop)


def test_empty_cell_fill_uses_proxy_not_champion_fitness():
    eng = _Engine()
    champ = [_ADDI_X1, _EBREAK]
    champ_trace = _trace(2)
    champ_desc = compute_qd_axes(champ, champ_trace)
    eng.qd_archive.add_solution(champ_desc, 572.0, champ, champ_trace)
    varied = [champ, [_ADDI_X1, _ADDI_X2, _LUI_X3, _JAL_X0, _EBREAK]]
    traces = [_trace(2), _trace(12)]
    eng._fill_empty_qd_cells_from_evaluated(varied, traces)
    cell = eng.qd_archive._grid[eng.qd_archive._cell_index(champ_desc)]
    assert cell["fitness"] == 572.0


def test_pick_mutation_parent_uses_shorter_archive_cell():
    eng = _Engine()
    eng.generations_stalled = 5
    eng.champion = [_ADDI_X1, _EBREAK] * 20 + [_EBREAK]
    eng.champion_trace = _trace(2)
    eng.rng = __import__("random").Random(1)
    short = [_ADDI_X1, _ADDI_X2, _EBREAK]
    longp = eng.champion
    eng.qd_archive.add_solution(compute_qd_axes(short, _trace(12)), 1.0, short, _trace(12))
    eng.qd_archive.add_solution(compute_qd_axes(longp, _trace(2)), 572.0, longp, _trace(2))

    # Force archive branch
    class _Always:
        def random(self):
            return 0.0

        def randrange(self, n):
            return 0

    eng.rng = _Always()
    parent, traces = eng._pick_mutation_parent()
    assert parent == short
    assert traces and traces[0]["final_regs"]


def test_pick_mutation_parent_uses_l10_lift_growth_material(monkeypatch):
    eng = _Engine()
    eng.generations_stalled = 5
    eng.champion = [_ADDI_X1, _EBREAK] * 13 + [_EBREAK]
    eng.champion_trace = _trace(2)
    l10_parent = eng.champion + [0x11111111, 0x22222222, 0x33333333, 0x44444444]
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _n: True,
    )
    monkeypatch.setattr(
        "evolution.mutation_helpers._curriculum_l10_cached",
        lambda prog: 1.0 if len(prog) > len(eng.champion) else 0.8,
    )
    monkeypatch.setattr(
        "evolution.champion_rollback.collect_growth_material_compress_parents",
        lambda _c, max_parents=8: [list(l10_parent)],
    )

    class _LiftPick:
        def random(self):
            return 0.0

        def randrange(self, n):
            return 0

    eng.rng = _LiftPick()
    parent, _ = eng._pick_mutation_parent()
    assert parent == l10_parent


def test_scan_floor_growth_crossover_prefers_equal_length(monkeypatch):
    import json
    import os

    if not os.path.isfile("multi_cycle_results.jsonl"):
        import pytest

        pytest.skip("needs live results jsonl")
    row = None
    for line in open("multi_cycle_results.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("champion_words") == 48 and candidate.get("final_level") == 30:
            row = candidate
    if row is None:
        import pytest

        pytest.skip("needs live 48w champion")
    from evolution_engine import EvolutionConfig, EvolutionEngine

    prog = [int(x, 16) for x in row["champion_program"]]
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = prog
    eng.champion_level = 30
    eng._champion_verify_score = 1.0
    eng._cycle_number = 6885
    eng.current_generation = 5
    eng.rng = __import__("random").Random(0)
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_capability_stalled",
        lambda: True,
    )
    hybrid = eng._try_scan_floor_growth_crossover()
    if hybrid is None:
        import pytest

        pytest.skip("no gate-qualified equal-length growth hybrids in pool")
    assert len(hybrid) == len(prog)
    assert tuple(hybrid) != tuple(prog)
