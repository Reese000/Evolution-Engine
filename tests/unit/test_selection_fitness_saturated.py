"""Verify saturated-verify promotion uses fitness tie-break."""

import os

import pytest

os.environ.setdefault("EE_USE_GPU", "0")


def test_candidate_beats_champion_at_saturated_verify(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    from evolution.terminal_stagnation import candidate_beats_champion_on_verify

    assert candidate_beats_champion_on_verify(1.0, 1.0, 264.0, 264.2)
    assert not candidate_beats_champion_on_verify(1.0, 1.0, 264.0, 264.0)
    assert not candidate_beats_champion_on_verify(1.0, 0.99, 264.0, 300.0)


def test_candidate_beats_champion_raw_tie_break_at_floor(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    from evolution.terminal_stagnation import candidate_beats_champion_on_verify

    assert not candidate_beats_champion_on_verify(1.0, 1.0, 348.0, 348.0)
    assert candidate_beats_champion_on_verify(
        1.0,
        1.0,
        348.0,
        348.0,
        champion_raw_fitness=311.0,
        candidate_raw_fitness=312.5,
        champion_words=60,
        candidate_words=60,
    )
    assert candidate_beats_champion_on_verify(
        1.0,
        1.0,
        348.0,
        348.0,
        champion_words=60,
        candidate_words=59,
    )
    assert not candidate_beats_champion_on_verify(
        1.0,
        1.0,
        348.0,
        348.0,
        champion_raw_fitness=312.0,
        candidate_raw_fitness=311.0,
        champion_words=60,
        candidate_words=60,
    )


def test_update_verify_best_prefers_higher_raw_on_fitness_tie(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    from evolution.terminal_stagnation import update_verify_best_candidate

    class _Eng:
        _champion_verify_score = 1.0

        @staticmethod
        def _verify_score_from_trace(trace):
            return float(trace.get("verify", 0.0))

    eng = _Eng()
    champ = [1, 2, 3]
    champ_trace = {"verify": 0.68}
    low = ([1, 2, 4], {"verify": 0.68})
    high = ([1, 2, 5], {"verify": 0.68})

    def _raw(_eng, prog, trace, *, verify_score=None):
        return 130.0 if prog[-1] == 5 else 120.0

    monkeypatch.setattr("evolution.fitness_helpers.promotion_raw_fitness", _raw)
    best = update_verify_best_candidate(
        eng,
        None,
        0.68,
        344.0,
        low[0],
        low[1],
        0.0,
        champion_verify=0.68,
        champion_program=champ,
        champion_trace=champ_trace,
        champion_fitness=344.0,
    )
    best = update_verify_best_candidate(
        eng,
        best,
        0.68,
        344.0,
        high[0],
        high[1],
        0.0,
        champion_verify=0.68,
        champion_program=champ,
        champion_trace=champ_trace,
        champion_fitness=344.0,
    )
    assert best is not None
    assert best[2][-1] == 5


def test_verify_led_candidate_eligible_certified_plateau_partial_trace():
    from evolution.terminal_stagnation import verify_led_candidate_eligible

    assert verify_led_candidate_eligible(0.68, 0.68, champion_certified_verify=1.0)
    assert not verify_led_candidate_eligible(0.68, 0.60, champion_certified_verify=1.0)


def test_verify_promotion_beats_on_certified_plateau_partial_trace(monkeypatch):
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    from evolution.terminal_stagnation import verify_promotion_beats_champion

    class _Eng:
        _champion_verify_score = 1.0

        @staticmethod
        def _verify_score_from_trace(trace):
            return float(trace.get("verify", 0.0))

    eng = _Eng()
    champ = [1, 2, 3]
    cand = [1, 2, 4]
    champ_trace = {"verify": 0.68}
    cand_trace = {"verify": 0.68}

    def _raw(_eng, prog, trace, *, verify_score=None):
        return 130.0 if prog[-1] == 4 else 120.0

    monkeypatch.setattr(
        "evolution.fitness_helpers.promotion_raw_fitness",
        _raw,
    )
    assert verify_promotion_beats_champion(
        eng,
        champ,
        champ_trace,
        cand,
        cand_trace,
        344.0,
        344.0,
        champion_verify=0.68,
        candidate_verify=0.68,
    )
    assert not verify_promotion_beats_champion(
        eng,
        champ,
        champ_trace,
        cand,
        {"verify": 0.60},
        344.0,
        344.0,
        champion_verify=0.68,
        candidate_verify=0.60,
    )
