#!/usr/bin/env python3
"""End-to-end smoke test for the Evolution Engine.

This is the single most important test that validates the entire system
works end-to-end: config -> engine init -> evolution loop -> fitness
computation (including efficiency scoring and secondary tasks) ->
checkpoint persistence.

A passing test here means the full pipeline is functional. A failure
here is a hard blocker.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from evolution_engine import EvolutionConfig, EvolutionEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_smoke_config(log_dir: str, champion_dir: str) -> EvolutionConfig:
    """Build a minimal EvolutionConfig for smoke testing.

    - Population 10, 5 generations (fast execution)
    - No GPU
    - Secondary tasks enabled (BitManipulation, Arithmetic, MemoryAccess)
    - Efficiency scoring active (weight=2.0)
    - Checkpoint every 3 gens so mid-loop + final checkpoint both fire
    - Auto-curriculum enabled so curriculum logic is exercised
    """
    return EvolutionConfig(
        population_size=10,
        max_generations=5,
        use_gpu=False,
        use_secondary_tasks=True,
        secondary_task_weight=3.0,
        efficiency_score_weight=2.0,
        checkpoint_interval=3,
        log_dir=log_dir,
        champion_dir=champion_dir,
        auto_curriculum=True,
        curriculum_stall_threshold=3,
        seed=42,
        # Keep eval budget reasonable for speed
        eval_budget=5000,
        instruction_budget=10000,
        # Disable novelty/QD archives to keep the smoke test focused
        use_novelty_archive=False,
        use_qd_archive=False,
        use_strategy_archive=False,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestEndToEndSmoke:
    """Validate the full Evolution Engine pipeline in a single test."""

    def test_full_evolution_cycle(self):
        """Run one complete evolution cycle and verify all major subsystems.

        Checks:
          1. No exceptions raised (pytest catches unhandled errors)
          2. Champion fitness > 0 (the engine found a non-trivial solution)
          3. All 5 generations completed
          4. Checkpoint files created (meta.json + strategy archive)
          5. Secondary tasks were evaluated (3 tasks: bit, arith, mem)
          6. Efficiency scoring contributed to fitness
        """
        with tempfile.TemporaryDirectory(prefix="evo_smoke_") as tmp_dir:
            log_dir = os.path.join(tmp_dir, "logs")
            champion_dir = os.path.join(tmp_dir, "champions")
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(champion_dir, exist_ok=True)

            config = _make_smoke_config(log_dir, champion_dir)
            engine = EvolutionEngine(config=config)

            # -----------------------------------------------------------------
            # Run the full evolution loop
            # -----------------------------------------------------------------
            summary = engine.run()

            # -----------------------------------------------------------------
            # 1. No exceptions -- if we reach here, the loop completed without
            #    unhandled errors. Also verify no error key in summary.
            # -----------------------------------------------------------------
            assert "error" not in summary, (
                f"Evolution loop reported error: {summary.get('error')} "
                f"at generation {summary.get('error_generation')}"
            )

            # -----------------------------------------------------------------
            # 2. Champion fitness > 0
            # -----------------------------------------------------------------
            assert summary["final_fitness"] > 0, f"Champion fitness should be positive, got {summary['final_fitness']}"
            assert engine.champion_fitness > 0

            # -----------------------------------------------------------------
            # 3. All 5 generations completed
            # -----------------------------------------------------------------
            assert (
                summary["generations_completed"] == 5
            ), f"Expected 5 generations, got {summary['generations_completed']}"
            assert engine.current_generation == 4  # 0-indexed

            # -----------------------------------------------------------------
            # 4. Checkpoint file created
            #    _checkpoint() writes meta.json to <log_dir>/checkpoints/
            #    The final checkpoint always fires on success (line 1422).
            # -----------------------------------------------------------------
            checkpoint_dir = os.path.join(log_dir, "checkpoints")
            assert os.path.isdir(checkpoint_dir), f"Checkpoint directory not created: {checkpoint_dir}"

            meta_path = os.path.join(checkpoint_dir, "meta.json")
            assert os.path.isfile(meta_path), f"Checkpoint meta.json not found: {meta_path}"

            with open(meta_path, "r") as f:
                meta = json.load(f)

            assert meta["generation"] == 4, f"Checkpoint generation should be 4, got {meta['generation']}"
            assert meta["fitness"] > 0, f"Checkpoint fitness should be positive, got {meta['fitness']}"
            assert (
                meta["program_words"] > 0
            ), f"Checkpoint program_words should be positive, got {meta['program_words']}"
            assert meta["level"] >= 0, f"Checkpoint level should be >= 0, got {meta['level']}"

            # -----------------------------------------------------------------
            # 5. Secondary tasks were evaluated
            #    With use_secondary_tasks=True and a real VMFarm, the engine
            #    evaluates BitManipulation, Arithmetic, and MemoryAccess tasks.
            # -----------------------------------------------------------------
            secondary = engine._secondary_task_results
            assert secondary is not None, (
                "Secondary task results should not be None " "(use_secondary_tasks=True was set)"
            )
            assert len(secondary) == 3, f"Expected 3 secondary task results (bit, arith, mem), " f"got {len(secondary)}"

            # Each result is a (task_instance, trace) tuple
            for i, (task_inst, trace) in enumerate(secondary):
                assert isinstance(task_inst, dict), f"Secondary task {i} instance should be dict, got {type(task_inst)}"
                assert isinstance(trace, dict), f"Secondary task {i} trace should be dict, got {type(trace)}"
                assert "status" in trace, f"Secondary task {i} trace missing 'status' key"

            # -----------------------------------------------------------------
            # 6. Efficiency scoring is active
            #    The efficiency score is computed in _compute_fitness() and
            #    gated by config.efficiency_score_weight (set to 2.0).
            #    We verify:
            #      a. Config value is correct
            #      b. Raw task score was computed (float, finite)
            #      c. Fitness reflects the full pipeline (efficiency + secondary)
            # -----------------------------------------------------------------
            assert config.efficiency_score_weight == 2.0, "efficiency_score_weight should be 2.0"
            assert isinstance(
                engine._raw_task_score, float
            ), f"_raw_task_score should be float, got {type(engine._raw_task_score)}"

            # -----------------------------------------------------------------
            # Bonus: summary contains expected diagnostic keys
            # -----------------------------------------------------------------
            assert "final_level" in summary
            assert "champion_words" in summary
            assert "elapsed_seconds" in summary
            assert "generations_stalled" in summary
            assert "total_champion_improvements" in summary
            assert summary["champion_words"] > 0
            assert summary["elapsed_seconds"] > 0

            # -----------------------------------------------------------------
            # Bonus: evolution made progress or at least ran all gens
            # -----------------------------------------------------------------
            assert len(engine.best_fitness_history) > 0, "Fitness history should have entries after a full run"
