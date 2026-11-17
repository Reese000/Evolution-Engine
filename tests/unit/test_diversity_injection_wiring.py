"""Category-diversity injection uses inject_diversity, not raw opcode count."""

from evolution_engine import EvolutionConfig, EvolutionEngine


def test_diversity_injection_uses_category_threshold(monkeypatch):
    eng = EvolutionEngine(config=EvolutionConfig(diversity_threshold=0.9, diversity_inject_ratio=0.2))
    eng.rng.seed(0)
    pop = [[0x00000013, 0x00100073] for _ in range(10)]
    calls = {"n": 0}

    def fake_inject(challengers, n_inject):
        calls["n"] = int(n_inject)

    monkeypatch.setattr(eng, "_inject_diversity", fake_inject)
    eng._generate_diversity_injection(pop, 10)
    assert calls["n"] > 0


def test_scan_floor_topup_raises_diversity(monkeypatch):
    eng = EvolutionEngine(config=EvolutionConfig(max_level=30, diversity_threshold=0.55, diversity_inject_ratio=0.55))
    eng.champion_level = 30
    eng.champion = [0x00000013] * 59 + [0x00100073]
    eng.rng.seed(0)
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _n: True,
    )
    pop = [list(eng.champion) for _ in range(20)]
    eng._last_challenger_strategies = ["DIRECTED"] * 20
    eng._maybe_topup_scan_floor_diversity(pop)
    assert eng._compute_population_diversity(pop) >= 0.99


def test_scan_floor_raises_diversity_threshold(monkeypatch):
    eng = EvolutionEngine(config=EvolutionConfig(diversity_threshold=0.55, max_level=30))
    eng.champion_level = 30
    eng.champion = [0x00000013] * 59 + [0x00100073]
    monkeypatch.setattr(
        "evolution.terminal_stagnation.terminal_efficiency_at_scan_floor",
        lambda _n: True,
    )
    assert eng._effective_diversity_threshold() >= 0.99


def test_diversity_injection_skips_when_above_threshold(monkeypatch):
    eng = EvolutionEngine(config=EvolutionConfig(diversity_threshold=0.3))
    pop = [[0x00000013, 0x00100073] for _ in range(10)]
    monkeypatch.setattr(eng, "_compute_population_diversity", lambda _c: 0.95)
    calls = {"n": 0}
    monkeypatch.setattr(eng, "_inject_diversity", lambda _c, n: calls.update(n=n))
    eng._generate_diversity_injection(pop, 10)
    assert calls.get("n", 0) == 0
