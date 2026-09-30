"""Tests for evolution.champion_bloat_guard."""

from evolution.champion_bloat_guard import champion_bloat_blocks


def test_allows_clean_l21_replacement() -> None:
    assert champion_bloat_blocks(21, 36, 36, 0.76, 0.75) is None


def test_blocks_l21_hard_cap() -> None:
    assert champion_bloat_blocks(21, 36, 37, 0.70, 0.75) is not None


def test_blocks_word_growth_without_verify_gain() -> None:
    reason = champion_bloat_blocks(21, 36, 37, 0.69, 0.75)
    assert reason is not None
    assert "words=37>36" in reason


def test_ignores_low_levels() -> None:
    assert champion_bloat_blocks(19, 30, 50, 0.5, 0.5) is None


def test_blocks_l21_stub_from_compact_champion() -> None:
    reason = champion_bloat_blocks(30, 13, 7, 1.0, 1.0)
    assert reason is not None
    assert "stub=7w" in reason


def test_allows_l30_compress_to_13w() -> None:
    assert champion_bloat_blocks(30, 22, 13, 1.0, 1.0) is None


def test_allows_l30_equal_verify_growth_under_cap() -> None:
    assert champion_bloat_blocks(30, 26, 28, 1.0, 1.0) is None


def test_blocks_l30_scan_floor_growth_without_verify_gain() -> None:
    reason = champion_bloat_blocks(30, 60, 67, 1.0, 1.0)
    assert reason is not None
    assert "growth=7w" in reason
    assert champion_bloat_blocks(30, 60, 60, 1.0, 1.0) is None
    assert champion_bloat_blocks(30, 60, 59, 1.0, 1.0) is None


def test_l30_scan_floor_rejects_longer_despite_side_curriculum(monkeypatch):
    monkeypatch.setenv("EE_USE_GPU", "0")
    monkeypatch.setenv("EE_MAX_LEVEL", "30")
    from evolution_engine import EvolutionConfig, EvolutionEngine

    champ = [0x00000013] * 59 + [0x00100073]
    longer = [0x00000013] * 66 + [0x00100073]
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, curriculum_level=30))
    eng.champion_level = 30
    eng.champion = list(champ)
    eng.champion_trace = {"status": "ok", "final_regs": {}}
    eng._champion_verify_score = 1.0
    eng.champion_fitness = 300.0
    eng._verify_champion = lambda: (True, 1.0)  # type: ignore[method-assign]
    monkeypatch.setattr(
        "evolution.terminal_stagnation.heldout_l30_verify_mean",
        lambda _p, *_a, **_k: 1.0,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.compact_curriculum_beats",
        lambda _a, _b: True,
    )
    monkeypatch.setattr(
        "evolution.terminal_stagnation.verify_promotion_beats_champion",
        lambda *a, **k: False,
    )
    dummy = {"status": "ok", "final_regs": {}}
    replaced = eng._verify_and_replace_champion(longer, dummy, 1.0, 310.0)
    assert replaced is False
    assert eng.champion == champ


def test_certified_champion_pinned_at_l21() -> None:
    from evolution.champion_bloat_guard import certified_champion_pinned

    assert certified_champion_pinned(21, 36, 0.75)
    assert not certified_champion_pinned(21, 37, 0.75)


def test_certified_champion_pinned_at_l22() -> None:
    from evolution.champion_bloat_guard import certified_champion_pinned

    assert certified_champion_pinned(22, 45, 0.769)
    assert certified_champion_pinned(23, 44, 0.643)
    assert not certified_champion_pinned(22, 30, 0.769)


def test_reuse_accepted_trace():
    from evolution.selection_mixin import reuse_accepted_trace

    prog = [1, 2, 3]
    assert reuse_accepted_trace(prog, prog, {"status": "ok"}) is True
    assert reuse_accepted_trace(prog, [1, 2], {"status": "ok"}) is False
    assert reuse_accepted_trace(prog, prog, None) is False


def test_verify_cache_key_identity():
    prog = [1, 2, 3]
    task = {"a": 1}
    k1 = (tuple(prog), 30, id(task))
    k2 = (tuple(prog), 30, id(task))
    k3 = (tuple(prog), 30, id({"a": 1}))
    assert k1 == k2
    assert k1 != k3
