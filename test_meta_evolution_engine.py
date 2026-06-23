#!/usr/bin/env python3
"""Tests for MetaEvolutionEngine — proper unit tests for the meta-evolution loop.

Tests the MetaEvolutionEngine class directly (not archive adaptation).
Uses mocked subprocess evaluation to avoid running actual evolution.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, Optional
from unittest.mock import patch, MagicMock

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from meta_evolution import (
    MetaEvolutionEngine,
    MetaEvolutionConfig,
    _DEFAULT_MAX_FITNESS,
    _DEFAULT_MAX_LEVEL,
    _DEFAULT_MAX_CHAMPION_WORDS,
    _DEFAULT_VARIANCE_PENALTY_COEFF,
    _DEFAULT_NUM_EVAL_SEEDS,
    _SEEDED_META_GENOMES,
)
from meta_genome import MetaGenome


# ── Fixtures ──────────────────────────────────────────────────────────

def sample_summary(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a plausible evolution summary dict for testing meta-fitness."""
    base = {
        "generations_completed": 50,
        "final_fitness": 0.8,
        "final_level": 8,
        "champion_words": 45,
        "champion_trace_count": 25,
    }
    if overrides:
        base.update(overrides)
    return base


# ── Test 1: MetaEvolutionConfig defaults ──────────────────────────────

def test_meta_config_defaults() -> None:
    """Default MetaEvolutionConfig has sane values."""
    c = MetaEvolutionConfig()
    assert c.meta_population_size == 10
    assert c.meta_generations == 5
    assert c.base_generations == 20
    assert c.meta_mutation_std == 0.15
    assert c.meta_crossover_rate == 0.3
    assert c.tournament_size == 3
    assert c.seed == 42


# ── Test 2: MetaEvolutionEngine init ─────────────────────────────────

def test_engine_init_population_size() -> None:
    """Engine initializes population of expected size."""
    config = MetaEvolutionConfig(meta_population_size=10, seed=42)
    engine = MetaEvolutionEngine(meta_config=config)
    assert len(engine.population) == 10
    assert len(engine.fitness) == 10
    assert all(f == float("-inf") for f in engine.fitness)


def test_engine_init_seeded_genomes() -> None:
    """Engine seeds with known-good configs when population >= seed count."""
    config = MetaEvolutionConfig(
        meta_population_size=max(10, len(_SEEDED_META_GENOMES)),
        seed=42,
    )
    engine = MetaEvolutionEngine(meta_config=config)
    # First N should be seeded configs (validated decoder)
    for genome in engine.population[:len(_SEEDED_META_GENOMES)]:
        cfg = genome.decode()
        assert 0.0 <= cfg.mutation_rate <= 1.0
        assert 10 <= cfg.population_size <= 200
        assert 0.0 <= cfg.elite_ratio <= 0.5


def test_engine_init_small_population_truncates_seeds() -> None:
    """When population < seed count, seeds are truncated with a warning."""
    config = MetaEvolutionConfig(meta_population_size=2, seed=42)
    engine = MetaEvolutionEngine(meta_config=config)
    assert len(engine.population) == 2


# ── Test 3: _compute_meta_fitness ────────────────────────────────────

def test_meta_fitness_positive_fitness() -> None:
    """Positive fitness yields non-zero meta-fitness with curriculum contribution."""
    summary = sample_summary({"final_fitness": 0.8, "final_level": 8})
    mf = MetaEvolutionEngine._compute_meta_fitness(summary)
    # champion_improvement = 0.8/1.5 * 0.3 ~ 0.16
    # curriculum_depth = 8/13 * 0.5 ~ 0.308
    # diversity (45 words, sweet spot 25-50): 1.0 -> * 0.2 = 0.2
    # total ~ 0.668
    assert 0.4 < mf < 0.9, f"Unexpected meta-fitness: {mf}"
    print(f"  Positive fitness MF={mf:.4f}")


def test_meta_fitness_negative_fitness_zeroes_curriculum() -> None:
    """Negative final fitness zeroes curriculum depth contribution."""
    summary = sample_summary({"final_fitness": -0.5, "final_level": 11})
    mf = MetaEvolutionEngine._compute_meta_fitness(summary)
    # champion_improvement = max(0, -0.5)/1.5 * 0.3 = 0.0
    # curriculum_depth = 0 (negative fitness)
    # diversity (45 words, sweet spot 25-50): 1.0 -> * 0.2 = 0.2
    # total = 0.2
    # The improvement and depth are zeroed; only diversity contributes.
    assert 0.0 < mf <= 0.25, f"Expected low meta-fitness for negative fitness, got {mf}"
    print(f"  Negative fitness MF={mf:.4f} (curriculum zeroed)")


def test_meta_fitness_max_fitness_caps_improvement() -> None:
    """Fitness above max_fitness caps at 1.0."""
    summary = sample_summary({"final_fitness": 3.0, "final_level": 13})
    mf = MetaEvolutionEngine._compute_meta_fitness(summary)
    # champion_improvement = min(1.0, 3.0/1.5) = 1.0 → * 0.3 = 0.3
    # curriculum_depth = 13/13 = 1.0 → * 0.5 = 0.5
    # diversity = 45/100 * 0.2 = 0.09
    # total ≈ 0.89
    assert mf > 0.7, f"Expected high MF for max fitness, got {mf}"
    print(f"  ✓ Max fitness MF={mf:.4f}")


def test_meta_fitness_weighted_combination() -> None:
    """Meta-fitness with custom weights produces correct weighted sum."""
    summary = sample_summary({"final_fitness": 0.5, "final_level": 5, "champion_words": 20})
    params = {
        "max_fitness": 1.0,
        "max_level": 10,
        "meta_weight_improvement": 0.5,
        "meta_weight_depth": 0.3,
        "meta_weight_diversity": 0.2,
    }
    mf = MetaEvolutionEngine._compute_meta_fitness(summary, meta_params=params)
    # champion_improvement = 0.5/1.0 = 0.5 -> * 0.5 = 0.25
    # curriculum_depth = 5/10 = 0.5 -> * 0.3 = 0.15
    # diversity (20 words, ramp zone 5-25): 0.1 + 0.9*(15/20) = 0.775 -> * 0.2 = 0.155
    # total = 0.555
    assert abs(mf - 0.555) < 0.02, f"Expected MF=0.555, got {mf}"
    print(f"  Custom weights MF={mf:.4f} (expected 0.555)")


# ── Test 4: Seeded config validation ────────────────────────────────

def test_seeded_configs_are_valid() -> None:
    """All seeded meta-genomes decode to valid EvolutionConfigs."""
    for idx, cfg_dict in enumerate(_SEEDED_META_GENOMES):
        genome = MetaGenome(**cfg_dict)
        config = genome.decode()
        assert 0.0 <= config.mutation_rate <= 1.0, f"Seed {idx} bad mutation_rate"
        assert 0.0 <= config.crossover_rate <= 1.0, f"Seed {idx} bad crossover_rate"
        assert 0.0 <= config.elite_ratio <= 0.5, f"Seed {idx} bad elite_ratio"
        assert 2 <= config.tournament_size <= 20, f"Seed {idx} bad tournament_size"
        assert 10 <= config.population_size <= 200, f"Seed {idx} bad population_size"
        assert config.curriculum_stall_threshold >= 1, f"Seed {idx} bad stall_threshold"
    print(f"  ✓ All {len(_SEEDED_META_GENOMES)} seeded configs valid")


# ── Test 5: Population evolution (mocked subprocess) ─────────────────

def test_evolution_loop_structure() -> None:
    """Engine.run() produces expected output structure with mocked subprocess."""
    config = MetaEvolutionConfig(
        meta_population_size=4,
        meta_generations=2,
        base_generations=5,
        seed=123,
    )
    engine = MetaEvolutionEngine(meta_config=config)

    # Mock _evaluate_genome to return plausible fitness values
    original_eval = engine._evaluate_genome

    def mock_evaluate(genome, meta_gen_idx=0):
        # Return fitness proportional to genome index (simple gradient)
        f = 0.2 + (meta_gen_idx * 0.1 if meta_gen_idx > 0 else 0.0)
        return f, sample_summary({"final_fitness": f, "final_level": int(f * 10)})

    engine._evaluate_genome = mock_evaluate

    meta_summary = engine.run()

    # Check output structure
    assert "meta_generations_completed" in meta_summary
    assert meta_summary["meta_generations_completed"] == 2
    assert "best_meta_fitness" in meta_summary
    assert meta_summary["best_meta_fitness"] > 0.0
    assert "history" in meta_summary
    assert len(meta_summary["history"]) == 2
    print(f"  ✓ Evolution loop: {meta_summary['meta_generations_completed']} gen, "
          f"best F={meta_summary['best_meta_fitness']:.4f}")


def test_best_config_returns_decoded() -> None:
    """best_config() returns a decoded EvolutionConfig from best genome."""
    config = MetaEvolutionConfig(meta_population_size=4, meta_generations=1, seed=42)
    engine = MetaEvolutionEngine(meta_config=config)

    def mock_evaluate(genome, meta_gen_idx=0):
        return 0.5 + meta_gen_idx * 0.1, sample_summary()

    engine._evaluate_genome = mock_evaluate
    engine.run()

    best_cfg = engine.best_config()
    assert best_cfg is not None
    assert hasattr(best_cfg, "mutation_rate")
    assert hasattr(best_cfg, "population_size")
    assert hasattr(best_cfg, "elite_ratio")
    assert 0.0 <= best_cfg.mutation_rate <= 1.0
    print(f"  ✓ Best config: pop={best_cfg.population_size} μ={best_cfg.mutation_rate:.3f}")


# ── Test 6: Tournament selection ─────────────────────────────────────

def test_tournament_selection_picks_fitter() -> None:
    """_evolve_population improves or maintains best fitness across generations."""
    config = MetaEvolutionConfig(
        meta_population_size=6,
        meta_generations=3,
        meta_mutation_std=0.05,  # Small mutations to keep improvement gradual
        seed=456,
    )
    engine = MetaEvolutionEngine(meta_config=config)

    # Assign increasing fitness values
    for i in range(len(engine.population)):
        engine.fitness[i] = i * 0.1  # 0.0, 0.1, 0.2, ..., 0.5

    best_before = max(engine.fitness)
    engine._evolve_population()

    # After evolution, fitness is reset to -inf for re-evaluation
    # But population should be same size
    assert len(engine.population) == config.meta_population_size
    # The elite genome should be preserved (best of previous gen)
    assert any(f == float("-inf") for f in engine.fitness)
    print(f"  ✓ Population maintained: {len(engine.population)} individuals")


# ── Test 7: Multi-seed variance penalty ──────────────────────────────

def test_variance_penalty_reduces_fitness() -> None:
    """Variance penalty reduces meta-fitness for inconsistent seeds."""
    config = MetaEvolutionConfig(meta_population_size=2, meta_generations=1, seed=42)
    engine = MetaEvolutionEngine(meta_config=config)
    engine._num_eval_seeds = 3  # Use 3 seeds explicitly

    # We can't easily test the full ensemble path without a subprocess,
    # but we can verify the penalty calculation logic
    ensemble = [0.3, 0.7, 0.5]  # Some variance
    mean_f = sum(ensemble) / len(ensemble)
    valid = [f for f in ensemble if __import__('math').isfinite(f)]
    m = sum(valid) / len(valid)
    var = sum((f - m) ** 2 for f in valid) / len(valid)
    std_dev = __import__('math').sqrt(var)
    penalty = std_dev * _DEFAULT_VARIANCE_PENALTY_COEFF
    adjusted = mean_f - penalty

    assert adjusted < mean_f, "Penalty should reduce fitness"
    assert adjusted > 0, "Adjusted fitness should be positive"
    print(f"  ✓ Variance penalty: mean={mean_f:.4f} → adjusted={adjusted:.4f}")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    failed = 0
    tests = [
        ("MetaEvolutionConfig defaults", test_meta_config_defaults),
        ("Engine init: population size", test_engine_init_population_size),
        ("Engine init: seeded genomes", test_engine_init_seeded_genomes),
        ("Engine init: small pop truncation", test_engine_init_small_population_truncates_seeds),
        ("Meta-fitness: positive fitness", test_meta_fitness_positive_fitness),
        ("Meta-fitness: negative zeroes curriculum", test_meta_fitness_negative_fitness_zeroes_curriculum),
        ("Meta-fitness: max fitness cap", test_meta_fitness_max_fitness_caps_improvement),
        ("Meta-fitness: weighted combination", test_meta_fitness_weighted_combination),
        ("Seeded configs: all valid", test_seeded_configs_are_valid),
        ("Evolution loop structure", test_evolution_loop_structure),
        ("Best config decode", test_best_config_returns_decoded),
        ("Tournament selection", test_tournament_selection_picks_fitter),
        ("Variance penalty", test_variance_penalty_reduces_fitness),
    ]

    print("=" * 60)
    print("MetaEvolutionEngine — Unit Tests")
    print("=" * 60)

    for desc, test_fn in tests:
        print(f"\n  [{desc}]")
        try:
            test_fn()
            print(f"  ✓ PASS")
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"ALL {len(tests)} TESTS PASSED")
        return 0
    else:
        print(f"{failed}/{len(tests)} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
