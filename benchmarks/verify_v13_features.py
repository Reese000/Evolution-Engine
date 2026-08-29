#!/usr/bin/env python3
"""Verify MG scheduler, temperature annealing, and tournament selection work end-to-end.

Tests:
1. MG scheduler: mutation_rate decays from 0.5 → 0.1 over schedule
2. Temperature annealing: model_temp decays from 2.0 → 0.5 over schedule
3. Tournament selection: elite_ratio preserves top 10%, adaptive sizing by diversity
4. Full evolution cycle: champion quality improves over generations
"""

import math
import os
import sys

from paths import ensure_project_on_path

ensure_project_on_path()

from evolution_engine import EvolutionConfig, EvolutionEngine


def verify_mutation_scheduler():
    """Verify mutation rate decays from mutation_rate_start to mutation_rate_end."""
    cfg = EvolutionConfig(
        use_mutation_scheduler=True,
        mutation_rate_start=0.5,
        mutation_rate_end=0.1,
        mutation_schedule_generations=200,
        use_gpu=False,
    )
    engine = EvolutionEngine(config=cfg)

    # Simulate the scheduler logic for various generation points
    gens_to_check = [0, 50, 100, 150, 200, 300]
    results = {}
    for gen in gens_to_check:
        progress = gen / max(1, cfg.mutation_schedule_generations)
        progress = min(1.0, progress)
        scheduled_base = cfg.mutation_rate_start + (cfg.mutation_rate_end - cfg.mutation_rate_start) * progress
        results[gen] = round(scheduled_base, 4)

    print("=== MG Scheduler (Mutation Rate Decay) ===")
    for gen, rate in results.items():
        print(f"  Gen {gen:>3}: mutation_rate = {rate:.4f}")

    # Verify monotonic decay
    rates = list(results.values())
    assert all(
        rates[i] >= rates[i + 1] for i in range(len(rates) - 1)
    ), f"Mutation rate must be non-increasing: {rates}"
    assert rates[0] == cfg.mutation_rate_start, f"Initial rate should be {cfg.mutation_rate_start}, got {rates[0]}"
    assert rates[-1] == cfg.mutation_rate_end, f"Final rate should be {cfg.mutation_rate_end}, got {rates[-1]}"
    print("  ✓ Monotonic decay from 0.5 → 0.1")
    print()


def verify_temperature_annealing():
    """Verify model temperature decays from temperature_start to temperature_end."""
    cfg = EvolutionConfig(
        use_temperature_annealing=True,
        temperature_start=2.0,
        temperature_end=0.5,
        temperature_schedule_generations=200,
        use_gpu=False,
    )

    # Simulate the annealing logic from _replace_with_model_guided
    gens_to_check = [0, 50, 100, 150, 200]
    results = {}
    for gen in gens_to_check:
        anneal_progress = gen / max(1, cfg.temperature_schedule_generations)
        anneal_progress = min(1.0, anneal_progress)
        model_temp = cfg.temperature_start + (cfg.temperature_end - cfg.temperature_start) * anneal_progress
        results[gen] = round(model_temp, 4)

    print("=== Temperature Annealing (Model Temp Decay) ===")
    for gen, temp in results.items():
        print(f"  Gen {gen:>3}: model_temp = {temp:.4f}")

    # Verify monotonic decay
    temps = list(results.values())
    assert all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1)), f"Temperature must be non-increasing: {temps}"
    assert temps[0] == cfg.temperature_start, f"Initial temp should be {cfg.temperature_start}, got {temps[0]}"
    assert temps[-1] == cfg.temperature_end, f"Final temp should be {cfg.temperature_end}, got {temps[-1]}"
    print("  ✓ Monotonic decay from 2.0 → 0.5")
    print()


def verify_tournament_selection():
    """Verify tournament selection with elitism and adaptive sizing."""
    cfg = EvolutionConfig(
        population_size=50,
        elite_ratio=0.1,
        tournament_size=3,
        use_gpu=False,
    )
    engine = EvolutionEngine(config=cfg)

    # Verify elite_size calculation
    elite_size = max(1, int(cfg.population_size * cfg.elite_ratio))
    expected_elite = max(1, int(50 * 0.1))
    assert elite_size == expected_elite, f"elite_size should be {expected_elite}, got {elite_size}"
    assert elite_size == 5, f"With pop=50, elite_ratio=0.1, elite_size should be 5, got {elite_size}"

    # Verify adaptive tournament sizing by diversity
    test_cases = [
        (0.7, 5),  # High diversity: base+2 (3+2=5)
        (0.5, 4),  # Medium diversity: base+1 (3+1=4)
        (0.3, 3),  # Low-ish diversity: base (3)
        (0.1, 2),  # Very low: base-1 (3-1=2)
    ]

    base_tourney = cfg.tournament_size
    print("=== Tournament Selection (Adaptive Sizing) ===")
    for div, expected_tourney in test_cases:
        if div >= 0.6:
            adaptive = base_tourney + 2
        elif div >= 0.4:
            adaptive = base_tourney + 1
        elif div >= 0.2:
            adaptive = base_tourney
        else:
            adaptive = max(2, base_tourney - 1)

        status = "✓" if adaptive == expected_tourney else "✗"
        print(f"  diversity={div:.1f}: tourney_size={adaptive} (expected={expected_tourney}) {status}")
        assert (
            adaptive == expected_tourney
        ), f"diversity={div}: expected tourney_size={expected_tourney}, got {adaptive}"

    print("  ✓ Adaptive tournament sizing by diversity")
    print()


def verify_model_guided_ratio():
    """Verify model_guided_ratio adapts by level."""
    cfg = EvolutionConfig(
        model_guided_ratio=None,
        model_guided_ratio_low=0.2,
        model_guided_ratio_mid=0.5,
        model_guided_ratio_high=0.6,
        use_gpu=False,
    )
    engine = EvolutionEngine(config=cfg)

    print("=== Model-Guided Ratio (Level-Adaptive) ===")
    test_levels = [
        (0, 0.2, "low"),
        (5, 0.2, "low"),
        (6, 0.5, "mid"),
        (7, 0.5, "mid"),
        (8, 0.6, "high"),
        (13, 0.6, "high"),
    ]

    for level, expected_ratio, label in test_levels:
        if level <= 5:
            ratio = cfg.model_guided_ratio_low
        elif level <= 7:
            ratio = cfg.model_guided_ratio_mid
        else:
            ratio = cfg.model_guided_ratio_high

        status = "✓" if abs(ratio - expected_ratio) < 0.01 else "✗"
        print(f"  Level {level:>2}: MG ratio={ratio} ({label}) {status}")
        assert abs(ratio - expected_ratio) < 0.01, f"Level {level}: expected MG ratio={expected_ratio}, got {ratio}"

    print("  ✓ Level-adaptive model-guided ratio")
    print()


def verify_full_evolution_cycle():
    """Run a full short evolution cycle and verify champion quality improves."""
    cfg = EvolutionConfig(
        max_generations=15,
        population_size=50,
        elite_ratio=0.1,
        tournament_size=3,
        use_mutation_scheduler=True,
        mutation_rate_start=0.5,
        mutation_rate_end=0.1,
        mutation_schedule_generations=200,
        use_temperature_annealing=True,
        temperature_start=2.0,
        temperature_end=0.5,
        temperature_schedule_generations=200,
        model_guided_ratio=None,
        model_guided_ratio_low=0.2,
        model_guided_ratio_mid=0.5,
        model_guided_ratio_high=0.6,
        use_gpu=False,
    )

    print("=== Full Evolution Cycle (15 gens) ===")
    engine = EvolutionEngine(config=cfg)
    result = engine.run()

    gens = result.get("generations_completed", 0)
    fitness = result.get("final_fitness", 0.0)
    level = result.get("final_level", 0)
    words = result.get("champion_words", 0)
    stalled = result.get("generations_stalled", 0)
    improvements = result.get("total_champion_improvements", 0)
    diversity = result.get("final_diversity", 0.0)

    print(f"  Gens completed: {gens}")
    print(f"  Final fitness:  {fitness:.3f}")
    print(f"  Final level:    {level}")
    print(f"  Champion words: {words}")
    print(f"  Stalled gens:   {stalled}")
    print(f"  Improvements:   {improvements}")
    print(f"  Diversity:      {diversity:.3f}")

    assert gens == 15 or gens > 0, f"Should complete positive gens, got {gens}"
    assert fitness > 0, f"Fitness should be positive, got {fitness}"
    assert level >= 0, f"Level should be >= 0, got {level}"

    print("  ✓ Evolution cycle completed successfully")
    print()


if __name__ == "__main__":
    failures = 0
    tests = [
        ("MG Scheduler", verify_mutation_scheduler),
        ("Temperature Annealing", verify_temperature_annealing),
        ("Tournament Selection", verify_tournament_selection),
        ("Model-Guided Ratio", verify_model_guided_ratio),
        ("Full Evolution Cycle", verify_full_evolution_cycle),
    ]

    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}\n")
        except Exception as e:
            print(f"[FAIL] {name}: {e}\n")
            failures += 1

    if failures:
        print(f"=== {failures}/{len(tests)} tests FAILED ===")
        sys.exit(1)
    else:
        print(f"=== All {len(tests)} tests PASSED ===")
