"""Promotion gates must not regress compact L30 promotion_raw capability."""

import os

os.environ.setdefault("EE_USE_GPU", "0")


def test_verify_promotion_rejects_shorter_raw_regression(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    from evolution.terminal_stagnation import verify_promotion_beats_champion

    class _Eng:
        _champion_verify_score = 1.0

        @staticmethod
        def _verify_score_from_trace(trace):
            return float(trace.get("verify", 1.0))

    eng = _Eng()
    champ = [1] * 57
    cand = [1] * 48
    trace = {"verify": 1.0}

    def _raw(_eng, prog, _trace, *, verify_score=None):
        return 310.0 if len(prog) < 57 else 321.645

    monkeypatch.setattr("evolution.fitness_helpers.promotion_raw_fitness", _raw)

    assert not verify_promotion_beats_champion(
        eng,
        champ,
        trace,
        cand,
        trace,
        351.0,
        360.0,
        champion_verify=1.0,
        candidate_verify=1.0,
    )


def test_capability_growth_bypasses_saturated_bloat_pin(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    monkeypatch.setenv("EE_USE_GPU", "0")
    monkeypatch.setenv("EE_MAX_LEVEL", "30")
    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = [0x00008193] * 60 + [0x00100073]
    grown = list(champ) + [0x00000013]
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, curriculum_level=30))
    eng.champion_level = 30
    eng.champion = list(champ)
    eng.champion_trace = {"status": "ok", "final_regs": {}}
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 370.0
    eng._verify_champion = lambda: (True, 1.0)  # type: ignore[method-assign]

    monkeypatch.setattr(
        "evolution.terminal_stagnation.heldout_l30_verify_mean",
        lambda _p, *_a, **_k: 1.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.verify_promotion_beats_champion",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "evolution.fitness_helpers.promotion_raw_fitness",
        lambda *_a, **_k: 322.5,
    )

    trace = {"status": "ok", "final_regs": {}}
    assert eng._verify_and_replace_champion(grown, trace, 1.0, 375.0) is True
    assert len(eng.champion) == 61


def test_verify_promotion_allows_shorter_same_raw(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    from evolution.terminal_stagnation import verify_promotion_beats_champion

    class _Eng:
        _champion_verify_score = 1.0

        @staticmethod
        def _verify_score_from_trace(trace):
            return float(trace.get("verify", 1.0))

    eng = _Eng()
    champ = [1] * 57
    cand = [1] * 56
    trace = {"verify": 1.0}

    def _raw(_eng, prog, _trace, *, verify_score=None):
        return 321.645

    monkeypatch.setattr("evolution.fitness_helpers.promotion_raw_fitness", _raw)

    assert verify_promotion_beats_champion(
        eng,
        champ,
        trace,
        cand,
        trace,
        371.0,
        371.0,
        champion_verify=1.0,
        candidate_verify=1.0,
    )


def test_verify_promotion_rejects_scan_floor_growth_with_side_regression(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor, verify_promotion_beats_champion

    class _Eng:
        _champion_verify_score = 1.0

        @staticmethod
        def _verify_score_from_trace(trace):
            return float(trace.get("verify", 1.0))

    eng = _Eng()
    champ = [1] * 30 + [0x00100073]
    grown = list(champ) + [2, 3, 3, 4]
    trace = {"verify": 1.0}
    record_terminal_efficiency_scan_floor(7448, 30, 0.99)

    monkeypatch.setattr(
        "evolution.fitness_helpers.promotion_raw_fitness",
        lambda _e, prog, _t, **kw: 326.0 if len(prog) > 31 else 325.8,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.heldout_l30_verify_mean",
        lambda prog, *_a, **_k: 0.98 if len(prog) > 31 else 0.99,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.compact_curriculum_side_score",
        lambda prog: 10.0 if len(prog) > 31 else 12.0,
    )

    assert not verify_promotion_beats_champion(
        eng,
        champ,
        trace,
        grown,
        trace,
        325.8,
        326.0,
        champion_verify=1.0,
        candidate_verify=1.0,
    )


def test_verify_promotion_rejects_fitness_led_longer_bloat(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    from evolution.terminal_stagnation import record_terminal_efficiency_scan_floor, verify_promotion_beats_champion

    class _Eng:
        _champion_verify_score = 1.0

        @staticmethod
        def _verify_score_from_trace(trace):
            return float(trace.get("verify", 1.0))

    eng = _Eng()
    champ = [1] * 30 + [0x00100073]
    grown = list(champ) + [2, 3, 3, 4, 5]
    trace = {"verify": 1.0}
    record_terminal_efficiency_scan_floor(7448, 30, 0.99)

    monkeypatch.setattr(
        "evolution.fitness_helpers.promotion_raw_fitness",
        lambda _e, prog, _t, **kw: 325.826,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.scan_floor_promotion_capability_qualifies",
        lambda *_a, **_k: True,
    )

    assert not verify_promotion_beats_champion(
        eng,
        champ,
        trace,
        grown,
        trace,
        325.0,
        330.0,
        champion_verify=1.0,
        candidate_verify=1.0,
    )
