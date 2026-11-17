"""High terminal novelty weight must not distort certified champion fitness."""

import os

os.environ.setdefault("EE_USE_GPU", "0")


def test_terminal_certified_fitness_context_restores_novelty_weight():
    from evolution.fitness_helpers import terminal_certified_fitness_context
    from evolution_engine import EvolutionConfig, EvolutionEngine

    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.config.novelty_weight = 0.85
    eng._novelty_weight_cache = 0.85
    eng.config.qd_bonus_weight = 0.1

    with terminal_certified_fitness_context(eng):
        assert eng.config.novelty_weight == 0.0
        assert eng._novelty_weight_cache == 0.0
        assert eng.config.qd_bonus_weight == 0.0

    assert eng.config.novelty_weight == 0.85
    assert eng._novelty_weight_cache == 0.85
    assert eng.config.qd_bonus_weight == 0.1


def test_diversity_boost_promote_persists_shorter_champion(monkeypatch):
    """Regression: tier-7 diversity boost must not block 68w breakthrough."""
    monkeypatch.setenv("EE_TERMINAL_VERIFY_PROMOTE", "1")
    from run_multi_cycle import PROJECT_ROOT, run_evolution
    from run_multi_cycle_loop import _apply_terminal_diversity_boost

    evo: dict = {}
    _apply_terminal_diversity_boost(1, evo, PROJECT_ROOT)
    assert evo.get("mutation_rate", 0) >= 0.8

    result = run_evolution(
        9994,
        50,
        "milestone1/checkpoints/checkpoint_best.pt",
        meta_evolution_params=evo,
        start_level=30,
    )
    assert result["champion_words"] < 73
    assert result["final_fitness"] > 264.04
    assert result["champion_verify_score"] >= 0.99
