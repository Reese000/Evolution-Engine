#!/usr/bin/env python3
"""
Meta-evolution loop for discovering optimal evolution configurations.

This module implements a population-based meta-evolution algorithm that
searches the space of evolution hyper-parameters (encoded as MetaGenome
instances). Each meta-genome is decoded into an EvolutionConfig, which is
used to run the base RV32I evolution engine as a **subprocess** (avoiding
nested imports from evolution_engine in this module). The resulting summary
dict is parsed to compute a meta-fitness score, and the meta-population
evolves via tournament selection, crossover, and mutation across
meta-generations.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from logging_config import get_logger

_logger = get_logger("MetaEvolution")

# ─── Variance penalty coefficient for ensemble evaluation ─────────────────
# Adjusted fitness = mean - std * _DEFAULT_VARIANCE_PENALTY_COEFF
# Higher values penalize unreliable configs more aggressively.
_DEFAULT_VARIANCE_PENALTY_COEFF: float = 0.2

# ─── Z-score normalization ─────────────────────────────────────────────────
# When enabled, meta-fitness values are z-score normalized across the
# population before selection. This prevents a single outlier from dominating
# and makes fitness comparisons robust across different evaluation conditions.
# Use rolling statistics (cumulative mean/std across generations) to avoid
# within-generation normalization bias in small populations.
# Set to True by default — the variance penalty already handles seed variance,
# and z-score normalization works at the population level.
_USE_ZSCORE_NORMALIZATION: bool = True

# Minimum population fitness standard deviation to avoid division by zero.
# When the population's fitness std is below this threshold, z-score
# normalization passes through raw fitness values unchanged.
_MIN_FITNESS_STD: float = 1e-6

# ─── Sibling path ───────────────────────────────────────────────────────────
# We need to be able to locate evolution_engine.py for the subprocess call.
# Use the same directory as this file.
_SCRIPT_DIR = Path(__file__).resolve().parent
_EVOLUTION_ENGINE = _SCRIPT_DIR / "evolution_engine.py"

# We import MetaGenome (which does import EvolutionConfig internally) but
# we do NOT import EvolutionEngine or EvolutionConfig directly at this level.
from meta_genome import MetaGenome  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Maximum curriculum level (0-indexed). The tasks_rv32i module defines 31
# levels (0…30). Used to normalise curriculum_speed in meta-fitness.
_DEFAULT_MAX_LEVEL: int = 30

# Upper bound for champion program length (words), used as a diversity
# normaliser.  Well beyond typical solutions to avoid clamping.
_DEFAULT_MAX_CHAMPION_WORDS: int = 500

# Maximum realistic raw fitness for a 50-generation base evolution run
# at curriculum level 0.  The raw task score can reach up to 150 for
# perfect solutions across all 14 levels, but for short runs (50-60 gen,
# starting at level 0) the typical range is 0.01–1.5.  1.5 provides
# headroom without compressing the signal into near-zero.
_DEFAULT_MAX_FITNESS: float = 1.5


# Number of independent seeds to average for meta-fitness evaluation.
# Higher values reduce evaluation noise at the cost of more compute.
# 3 seeds reduces variance by sqrt(3) ≈ 1.73×.
_DEFAULT_NUM_EVAL_SEEDS: int = 3

# ── Backward-compatible aliases for test imports ─────────────────────────
# These match the old names before the constants were renamed with _DEFAULT_*
# prefix during the June 2026 config parameterization.
_NUM_EVAL_SEEDS: int = _DEFAULT_NUM_EVAL_SEEDS
_MAX_FITNESS: float = _DEFAULT_MAX_FITNESS
_MAX_LEVEL: int = _DEFAULT_MAX_LEVEL
_MAX_CHAMPION_WORDS: int = _DEFAULT_MAX_CHAMPION_WORDS
_VARIANCE_PENALTY_COEFF: float = _DEFAULT_VARIANCE_PENALTY_COEFF


# ── Seeded meta-genomes for initial population ──────────────────────────
# These are known-good configurations from previous meta-evolution runs.
# The population is seeded with these before filling the rest randomly,
# giving meta-evolution a head start.
_SEEDED_META_GENOMES: List[Dict[str, Any]] = [
    # Config A: Default (pop=50, μ=0.3, elite=0.1) — stable baseline
    {
        "mutation_rate": 0.3,
        "crossover_rate": 0.3,
        "elite_ratio": 0.1,
        "tournament_size": 5,
        "population_size": 50,
        "stall_threshold": 3,
        "novelty_weight": 0.3,
        "strategy_type_weights": {"RANDOM": 0.25, "TASK_AWARE": 0.25, "DIRECTED": 0.25, "PERTURBATION_CAPPED": 0.25},
    },
    # Config B: V3 evolved (pop=163, μ=0.509, elite=0.049) — exploration-heavy
    {
        "mutation_rate": 0.509,
        "crossover_rate": 0.01,
        "elite_ratio": 0.049,
        "tournament_size": 19,
        "population_size": 163,
        "stall_threshold": 4,
        "novelty_weight": 0.0,
        "strategy_type_weights": {"RANDOM": 0.07, "TASK_AWARE": 0.90, "DIRECTED": 0.03, "PERTURBATION_CAPPED": 0.0},
    },
    # Config C: Sustained best (pop=116, μ=0.709, elite=0.395) — preservation-heavy
    {
        "mutation_rate": 0.709,
        "crossover_rate": 0.887,
        "elite_ratio": 0.395,
        "tournament_size": 16,
        "population_size": 116,
        "stall_threshold": 11,
        "novelty_weight": 0.0,
        "strategy_type_weights": {"RANDOM": 0.07, "TASK_AWARE": 0.90, "DIRECTED": 0.03, "PERTURBATION_CAPPED": 0.0},
    },
    # Config D: Evolved best (pop=49, μ=0.584, cx=0.0, elite=0.097, tourney=6, stall=5, novelty=0.17)
    # Achieved F=0.922 (2× improvement over default) in head-to-head validation (50 gen, 3 seeds).
    # 0% collapse at 200 gen. Lowest champion size (18 words) at max curriculum level.
    {
        "mutation_rate": 0.584,
        "crossover_rate": 0.0,
        "elite_ratio": 0.097,
        "tournament_size": 6,
        "population_size": 49,
        "stall_threshold": 5,
        "novelty_weight": 0.17,
        "strategy_type_weights": {
            "RANDOM": 0.023,
            "TASK_AWARE": 0.348,
            "DIRECTED": 0.083,
            "PERTURBATION_CAPPED": 0.546,
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# MetaEvolutionConfig
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MetaEvolutionConfig:
    """Configuration for the meta-evolution loop.

    Attributes
    ----------
    meta_population_size : int
        Number of meta-genomes in the population (default 10).
    meta_generations : int
        Number of meta-generations to run (default 5).
    base_generations : int
        Number of generations each base evolution run lasts (default 20).
    meta_mutation_std : float
        Standard deviation used when mutating meta-genome fields
        (default 0.15).  Passed as the *std* argument to
        ``MetaGenome.mutate()``.
    meta_crossover_rate : float
        Probability of applying crossover during meta-reproduction
        (default 0.3).
    tournament_size : int
        Number of individuals in each meta-tournament (default 3).
    seed : int
        Random seed for reproducibility of the meta-loop (default 42).
    """

    meta_population_size: int = 10
    meta_generations: int = 5
    base_generations: int = 20
    meta_mutation_std: float = 0.15
    meta_crossover_rate: float = 0.3
    tournament_size: int = 3
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# MetaEvolutionEngine
# ─────────────────────────────────────────────────────────────────────────────


class MetaEvolutionEngine:
    """Population-based meta-evolution over evolution hyper-parameters.

    The engine maintains a population of :class:`MetaGenome` instances and
    evaluates each by decoding it to an :class:`EvolutionConfig`, running
    the base RV32I evolution engine as a subprocess, and computing a
    weighted meta-fitness from the summary returned by the base run.  The
    meta-population then undergoes tournament selection, crossover, and
    mutation to produce the next generation of meta-genomes.

    Parameters
    ----------
    meta_config : MetaEvolutionConfig
        Parameters governing the meta-evolution loop itself.
    rng : Optional[random.Random]
        Optional seeded random number generator.  If ``None``, one is
        created from ``meta_config.seed``.
    """

    def __init__(
        self,
        meta_config: MetaEvolutionConfig,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.config = meta_config
        self._rng: random.Random = rng if rng is not None else random.Random(meta_config.seed)

        # ── Initialise the meta-population ──────────────────────────────
        # Seed with known-good configs first, then fill the rest randomly.
        # Validate seed configs have all required MetaGenome fields.
        _REQUIRED_SEED_KEYS = {
            "mutation_rate",
            "crossover_rate",
            "elite_ratio",
            "tournament_size",
            "population_size",
            "stall_threshold",
            "novelty_weight",
            "strategy_type_weights",
        }
        for idx, cfg in enumerate(_SEEDED_META_GENOMES):
            missing = _REQUIRED_SEED_KEYS - set(cfg.keys())
            if missing:
                raise ValueError(f"Seed config {idx} missing keys: {missing}")
            # Validate ranges
            if not (0.0 <= cfg["elite_ratio"] <= 0.5):
                raise ValueError(f"Seed {idx} elite_ratio={cfg['elite_ratio']} out of range [0, 0.5]")
            if not (0.0 <= cfg["mutation_rate"] <= 1.0):
                raise ValueError(f"Seed {idx} mutation_rate={cfg['mutation_rate']} out of range [0, 1]")
            if not (2 <= cfg["tournament_size"] <= 20):
                raise ValueError(f"Seed {idx} tournament_size={cfg['tournament_size']} out of range [2, 20]")
            if not (10 <= cfg["population_size"] <= 200):
                raise ValueError(f"Seed {idx} population_size={cfg['population_size']} out of range [10, 200]")

        # If population size is smaller than number of seeds, truncate seeds
        seed_list = list(_SEEDED_META_GENOMES)
        if meta_config.meta_population_size < len(seed_list):
            _logger.warning(
                "[Meta-pop] population size %d < %d seeds. Truncating to %d.",
                meta_config.meta_population_size,
                len(seed_list),
                meta_config.meta_population_size,
            )
            seed_list = seed_list[: meta_config.meta_population_size]

        seeded = [MetaGenome(**cfg) for cfg in seed_list]
        remaining = max(0, meta_config.meta_population_size - len(seeded))
        random_genomes = [MetaGenome.random(rng=self._rng) for _ in range(remaining)]
        self.population: List[MetaGenome] = seeded + random_genomes

        # Log seeded genomes for observability
        if seeded:
            _logger.info(
                "[Meta-pop] Seeded %d known-good config(s): %s",
                len(seeded),
                ", ".join(f"pop={g.population_size} μ={g.mutation_rate:.3f} elite={g.elite_ratio:.3f}" for g in seeded),
            )

        # Fitness array: same length as population, -inf means unevaluated
        self.fitness: List[float] = [float("-inf") for _ in range(meta_config.meta_population_size)]

        # Track the best across all generations
        self._best_genome: Optional[MetaGenome] = None
        self._best_fitness: float = float("-inf")
        self._best_summary: Optional[Dict[str, Any]] = None

        # Generation-level logging
        self.current_meta_gen: int = 0
        self.history: List[Dict[str, Any]] = []

        # Meta-params cache: populated from the first genome decoded during
        # evaluation.  Used to make meta-fitness computation configurable.
        self._meta_params: Optional[Dict[str, Any]] = None
        self._num_eval_seeds: int = _DEFAULT_NUM_EVAL_SEEDS

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the meta-evolution loop.

        For each meta-generation:
          1. Evaluate every meta-genome by running the base engine as a
             subprocess.
          2. Compute meta-fitness from the returned summary.
          3. Select parents via tournament selection.
          4. Reproduce via crossover (with probability) + mutation.
          5. Log and checkpoint.

        Returns
        -------
        Dict[str, Any]
            A summary of the entire meta-run containing the best config,
            fitness trajectory, and population statistics.
        """
        start_time = time.time()

        for meta_gen in range(self.config.meta_generations):
            self.current_meta_gen = meta_gen
            gen_start = time.time()

            # ── Evaluate all meta-genomes in this generation ────────────
            for i, genome in enumerate(self.population):
                fitness_val, summary = self._evaluate_genome(genome, meta_gen_idx=i)
                self.fitness[i] = fitness_val

                # Track global best
                if fitness_val > self._best_fitness:
                    self._best_fitness = fitness_val
                    self._best_genome = genome
                    self._best_summary = summary

            # ── Z-score normalize fitness across population ──────────────
            # Prevents a single outlier from dominating selection.
            # Uses within-generation statistics (robust for small pops).
            # Single-pass: compute stats and collect valid fitnesses together
            _neg_inf = float("-inf")
            raw_fitness: list[float] = []
            best_fit = _neg_inf
            worst_fit = float("inf")
            fitness_list = self.fitness
            for f_val in fitness_list:
                if f_val > _neg_inf:
                    raw_fitness.append(f_val)
                    if f_val > best_fit:
                        best_fit = f_val
                    if f_val < worst_fit:
                        worst_fit = f_val
            if best_fit == _neg_inf:
                best_fit = 0.0
                worst_fit = 0.0

            if _USE_ZSCORE_NORMALIZATION and len(raw_fitness) >= 2:
                m = sum(raw_fitness) / len(raw_fitness)
                var = sum((f - m) ** 2 for f in raw_fitness) / len(raw_fitness)
                std = math.sqrt(var) if var > 0 else 1.0
                if std > _MIN_FITNESS_STD:
                    inv_std = 1.0 / std
                    for i in range(len(fitness_list)):
                        if fitness_list[i] > _neg_inf:
                            fitness_list[i] = (fitness_list[i] - m) * inv_std
                    # R33: Recompute best/worst from normalized values so gen_info
                    # reports post-normalization statistics (consistent with the
                    # fitness values actually used for selection). Previously,
                    # reported pre-normalization best/worst while selection used
                    # normalized values -- misleading for diagnostics.
                    best_fit = _neg_inf
                    worst_fit = float("inf")
                    for f_val in fitness_list:
                        if f_val > _neg_inf:
                            if f_val > best_fit:
                                best_fit = f_val
                            if f_val < worst_fit:
                                worst_fit = f_val
                    if best_fit == _neg_inf:
                        best_fit = 0.0
                        worst_fit = 0.0

            # ── Statistics for this generation ───────────────────────────
            gen_elapsed = time.time() - gen_start
            # R34: Compute avg_fit from fitness_list (post-normalization) to
            # be consistent with best_fit/worst_fit. Previously computed from
            # pre-normalization raw_fitness, which diverged after z-score
            # normalization was applied.
            valid_normalized = [f for f in fitness_list if f > _neg_inf]
            avg_fit = sum(valid_normalized) / len(valid_normalized) if valid_normalized else 0.0

            gen_info: Dict[str, Any] = {
                "meta_gen": meta_gen,
                "avg_fitness": avg_fit,
                "best_fitness": best_fit,
                "worst_fitness": worst_fit,
                "elapsed_seconds": gen_elapsed,
                "best_genome_repr": repr(self._best_genome) if self._best_genome else repr(self.population[0]),
            }
            self.history.append(gen_info)

            # ── Log ──────────────────────────────────────────────────────
            self._log_generation(gen_info)

            # ── Selection + Reproduction (skip for last generation) ──────
            if meta_gen < self.config.meta_generations - 1:
                self._evolve_population()

        # ── Final summary ─────────────────────────────────────────────────
        total_elapsed = time.time() - start_time
        meta_summary: Dict[str, Any] = {
            "meta_generations_completed": self.config.meta_generations,
            "best_meta_fitness": self._best_fitness,
            "best_genome_repr": repr(self._best_genome) if self._best_genome else None,
            "total_elapsed_seconds": total_elapsed,
            "history": self.history,
        }
        return meta_summary

    def best_config(self) -> Any:
        """Return the ``EvolutionConfig`` with the highest meta-fitness.

        Returns
        -------
        EvolutionConfig
            The decoded configuration from the best meta-genome found
            during the run.
        """
        if self._best_genome is None:
            # Fall back to the best in the current population
            if not self.population:
                raise RuntimeError("No meta-genomes in population — cannot produce best config.")
            best_idx = max(
                range(len(self.fitness)),
                key=lambda i: self.fitness[i],
            )
            return self.population[best_idx].decode()
        return self._best_genome.decode()

    # ── Internals ──────────────────────────────────────────────────────────

    def _evaluate_genome(
        self,
        genome: MetaGenome,
        meta_gen_idx: int = 0,
    ) -> Tuple[float, Dict[str, Any]]:
        """Run one meta-genome through the base evolution engine.

        The genome is decoded to an ``EvolutionConfig``, serialised to a
        temporary JSON file, and the base engine is launched as a
        subprocess.  The output summary is parsed to compute meta-fitness.

        Parameters
        ----------
        genome : MetaGenome
            The meta-genome to evaluate.
        meta_gen_idx : int
            Index in the meta-population (used for logging only).

        Returns
        -------
        Tuple[float, Dict[str, Any]]
            (meta_fitness, summary_dict) where summary_dict contains the
            raw output from the base evolution run.
        """
        # Decode the meta-genome to an EvolutionConfig
        config = genome.decode()

        # ── Build meta-params from config for configurable meta-fitness ──
        meta_params = {
            "max_fitness": getattr(config, "max_fitness", _DEFAULT_MAX_FITNESS),
            "max_level": getattr(config, "max_level", _DEFAULT_MAX_LEVEL),
            "max_champion_words": getattr(config, "max_champion_words", _DEFAULT_MAX_CHAMPION_WORDS),
            "meta_weight_improvement": getattr(config, "meta_weight_improvement", 0.3),
            "meta_weight_depth": getattr(config, "meta_weight_depth", 0.5),
            "meta_weight_diversity": getattr(config, "meta_weight_diversity", 0.2),
        }
        # Cache on first decode
        if self._meta_params is None:
            self._meta_params = meta_params
            self._num_eval_seeds = getattr(config, "num_eval_seeds", _DEFAULT_NUM_EVAL_SEEDS)

        # Resolve the variance penalty coefficient from the config
        variance_penalty_coeff = getattr(config, "variance_penalty_coeff", _DEFAULT_VARIANCE_PENALTY_COEFF)

        # Build the base config dict (without seed — set per iteration)
        base_config_dict = {
            "population_size": config.population_size,
            "max_generations": self.config.base_generations,
            "mutation_rate": config.mutation_rate,
            "crossover_rate": config.crossover_rate,
            "elite_ratio": config.elite_ratio,
            "tournament_size": config.tournament_size,
            "curriculum_stall_threshold": config.curriculum_stall_threshold,
            "novelty_weight": config.novelty_weight,
            "curriculum_level": 0,
            "auto_curriculum": True,
            "instruction_budget": config.instruction_budget if hasattr(config, "instruction_budget") else 10000,
            "eval_budget": config.eval_budget if hasattr(config, "eval_budget") else 5000,
        }

        # ── Multi-seed ensemble evaluation ──────────────────────────────
        # Average meta-fitness across _NUM_EVAL_SEEDS independent seeds to
        # reduce evaluation variance.  Each seed runs a full base evolution
        # as a subprocess.
        ensemble_fitnesses: List[float] = []
        best_summary: Dict[str, Any] = {}
        best_meta_fitness: float = float("-inf")

        for seed_idx in range(self._num_eval_seeds):
            # Build config dict with a fresh random seed for this iteration
            config_dict = dict(base_config_dict)
            config_dict["seed"] = self._rng.randint(0, 2**31 - 1)

            # Serialise to a temporary JSON file
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            tmp_path = tmp.name
            try:
                json.dump(config_dict, tmp)
                tmp.close()

                # Build the subprocess command
                runner_script = (
                    "import sys, json\n"
                    f"sys.path.insert(0, {str(_SCRIPT_DIR)!r})\n"
                    f"with open({tmp_path!r}) as f:\n"
                    "    d = json.load(f)\n"
                    "from evolution_engine import EvolutionEngine, EvolutionConfig\n"
                    "cfg = EvolutionConfig(**d)\n"
                    "eng = EvolutionEngine(config=cfg)\n"
                    "summary = eng.run()\n"
                    "print('META_EVAL_RESULT:' + json.dumps(summary))"
                )

                result = subprocess.run(
                    [sys.executable, "-c", runner_script],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=str(_SCRIPT_DIR),
                )

                if result.returncode != 0:
                    _logger.warning(
                        "[Meta-eval seed %s] genome %s failed (rc=%s). stderr: %s",
                        seed_idx,
                        meta_gen_idx,
                        result.returncode,
                        result.stderr.strip()[:200],
                    )
                    # Skip this seed but continue with others (partial failure)
                    continue

                # Parse the summary from stdout
                summary: Dict[str, Any] = {}
                for line in result.stdout.splitlines():
                    if line.startswith("META_EVAL_RESULT:"):
                        try:
                            summary = json.loads(line[len("META_EVAL_RESULT:") :])
                        except json.JSONDecodeError:
                            pass
                        break

                if not summary:
                    _logger.warning(
                        "[Meta-eval seed %s] genome %s returned no parseable " "summary. stdout: %s",
                        seed_idx,
                        meta_gen_idx,
                        result.stdout.strip()[:200],
                    )
                    continue  # Skip this seed; do NOT discard previous successes

                # Compute meta-fitness for this seed
                meta_fitness = self._compute_meta_fitness(summary, meta_params=meta_params)
                ensemble_fitnesses.append(meta_fitness)

                # Log per-seed result for transparency
                final_f = summary.get("final_fitness", "?")
                final_l = summary.get("final_level", "?")
                final_w = summary.get("champion_words", "?")
                _logger.info(
                    "[ensemble seed %s/%s] F=%s L=%s W=%s MF=%.4f",
                    seed_idx,
                    self._num_eval_seeds,
                    final_f,
                    final_l,
                    final_w,
                    meta_fitness,
                )

                # Track the best summary (for diagnostics)
                if meta_fitness > best_meta_fitness:
                    best_meta_fitness = meta_fitness
                    best_summary = summary

            except subprocess.TimeoutExpired:
                _logger.warning(
                    "[Meta-eval seed %s] genome %s timed out.",
                    seed_idx,
                    meta_gen_idx,
                )
                continue  # skip failed seed, continue with others
            except Exception as exc:
                _logger.error(
                    "[Meta-eval seed %s] genome %s raised %s.",
                    seed_idx,
                    meta_gen_idx,
                    exc,
                )
                continue  # skip failed seed, continue with others
            finally:
                # Clean up the temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        # End seed loop

        # If all seeds failed, return -inf
        if not ensemble_fitnesses or all(f == float("-inf") for f in ensemble_fitnesses):
            return float("-inf"), {"error": "all seeds failed"}

        # Apply variance penalty: adjusted = mean - std * coeff.
        # Configs with consistent results across seeds score higher than
        # those that spike on one seed but fail on others.
        mean_fitness = sum(ensemble_fitnesses) / len(ensemble_fitnesses)
        if len(ensemble_fitnesses) >= 2:
            # Compute std dev, filtering any non-finite values
            valid = [f for f in ensemble_fitnesses if math.isfinite(f)]
            if len(valid) >= 2:
                m = sum(valid) / len(valid)
                var = sum((f - m) ** 2 for f in valid) / len(valid)
                std_dev = math.sqrt(var)
                penalty = std_dev * variance_penalty_coeff
                adjusted = mean_fitness - penalty
                # Only log if penalty is non-trivial
                if penalty > 1e-10:
                    _logger.info(
                        "[variance penalty] mean=%.4f std=%.4f penalty=%.4f " "adjusted=%.4f",
                        mean_fitness,
                        std_dev,
                        penalty,
                        adjusted,
                    )
                mean_fitness = adjusted
        return mean_fitness, best_summary

    @staticmethod
    def _compute_meta_fitness(summary: Dict[str, Any], meta_params: Optional[Dict[str, Any]] = None) -> float:
        """Compute a weighted meta-fitness score from a base evolution summary.

        The summary dict is expected to contain at least the keys:
          ``generations_completed``, ``final_fitness``, ``final_level``,
          ``champion_words``.

        Meta-fitness is a weighted combination:

            champion_improvement * meta_weight_improvement +
            curriculum_depth      * meta_weight_depth +
            solution_diversity    * meta_weight_diversity

        Each component is normalised to [0, 1] before weighting.

        *champion_improvement* is final_fitness / max_fitness (capped at 1.0).
        *curriculum_depth* is the absolute curriculum level reached
        (final_level / max_level), rewarded regardless of how many
        generations it took. If final fitness is negative, curriculum
        depth is zeroed.
        *solution_diversity* is champion program length capped at 100 words
        (to avoid rewarding NOP insertion via GROW spam).

        Parameters
        ----------
        summary : Dict[str, Any]
            Summary dict from ``EvolutionEngine.run()``.
        meta_params : Optional[Dict[str, Any]]
            Optional dict of meta-fitness parameters. If None, the default
            constants (``_DEFAULT_MAX_FITNESS``, ``_DEFAULT_MAX_LEVEL``, etc.)
            and hardcoded weights (0.3, 0.5, 0.2) are used.
            If provided, expected keys:
              ``max_fitness``, ``max_level``, ``max_champion_words``,
              ``meta_weight_improvement``, ``meta_weight_depth``,
              ``meta_weight_diversity``.

        Returns
        -------
        float
            Scalar meta-fitness score in [0, 1].
        """
        final_fitness: float = summary.get("final_fitness", 0.0) or 0.0
        final_level: int = summary.get("final_level", 0) or 0
        generations_completed: int = summary.get("generations_completed", 1) or 1
        champion_words: int = summary.get("champion_words", 0) or 0

        # ── Resolve parameters ──────────────────────────────────────────
        if meta_params is None:
            # Use defaults for backward compatibility
            max_fitness = _DEFAULT_MAX_FITNESS
            max_level = _DEFAULT_MAX_LEVEL
            w_improvement = 0.3
            w_depth = 0.5
            w_diversity = 0.2
        else:
            max_fitness = meta_params.get("max_fitness", _DEFAULT_MAX_FITNESS)
            max_level = meta_params.get("max_level", _DEFAULT_MAX_LEVEL)
            w_improvement = meta_params.get("meta_weight_improvement", 0.3)
            w_depth = meta_params.get("meta_weight_depth", 0.5)
            w_diversity = meta_params.get("meta_weight_diversity", 0.2)

        # champion_improvement: normalised final fitness (capped at max_fitness).
        champion_improvement = min(1.0, max(0.0, final_fitness) / max_fitness)

        # curriculum_depth: absolute curriculum level reached, normalised.
        curriculum_depth = min(1.0, final_level / max_level)

        # Fitness floor: if final fitness is negative, the champion cannot
        # solve the task at the declared level — that level is meaningless.
        # Zero out the curriculum contribution entirely.
        if final_fitness < 0:
            curriculum_depth = 0.0

        # solution_diversity: penalise both too-short and too-long programs.
        # Peak reward at ~25 words (typical correct RV32I solution length).
        # Programs < 5 words are likely trivial/stub, > 80 words suffer bloat.
        if champion_words < 5:
            diversity = 0.1  # stub penalty
        elif champion_words <= 25:
            diversity = 0.1 + 0.9 * (champion_words - 5) / 20.0  # ramp up
        elif champion_words <= 50:
            diversity = 1.0  # sweet spot
        elif champion_words <= 80:
            diversity = 1.0 - 0.4 * (champion_words - 50) / 30.0  # gentle decay
        else:
            diversity = max(0.2, 0.6 - 0.005 * (champion_words - 80))  # bloat penalty

        meta_fitness = champion_improvement * w_improvement + curriculum_depth * w_depth + diversity * w_diversity

        return meta_fitness

    def _evolve_population(self) -> None:
        """Apply tournament selection, crossover, and mutation to produce
        the next meta-generation.

        The top two parents (by tournament selection of size
        ``self.config.tournament_size``) are crossed over with probability
        ``self.config.meta_crossover_rate``.  Children are mutated with
        standard deviation ``self.config.meta_mutation_std``.

        Elitism: the single best meta-genome (by fitness) is preserved
        into the next generation.
        """
        pop_size = self.config.meta_population_size
        if pop_size < 2:
            return

        # ── Elitism: keep the best individual ────────────────────────────
        # R21: Manual loop avoids creating a range + lambda overhead
        best_idx = 0
        best_f = self.fitness[0]
        fit_list = self.fitness
        for i in range(1, pop_size):
            if fit_list[i] > best_f:
                best_f = fit_list[i]
                best_idx = i
        elite_genome = self.population[best_idx]

        # ── R35: Diversity-preserving crossover adjustment ──────────────
        # Compute population fitness variance for convergence detection.
        # Low variance = convergence detected = increase crossover to
        # remix genetic material and prevent premature convergence.
        valid_fits = [f for f in fit_list if f > float("-inf")]
        effective_cx_rate = self.config.meta_crossover_rate
        if len(valid_fits) >= 3:
            m = sum(valid_fits) / len(valid_fits)
            var = sum((f - m) ** 2 for f in valid_fits) / len(valid_fits)
            if var < 0.1:
                # Low variance: boost crossover to prevent convergence
                effective_cx_rate = min(0.8, self.config.meta_crossover_rate * 1.5)

        # ── Tournament selection ─────────────────────────────────────────
        def _tournament() -> MetaGenome:
            """Select one individual via tournament of size *k*."""
            k = self.config.tournament_size
            # R21: Manual best-of-k without creating a list + lambda
            winner_idx = self._rng.randint(0, pop_size - 1)
            winner_f = fit_list[winner_idx]
            for _ in range(k - 1):
                cand = self._rng.randint(0, pop_size - 1)
                if fit_list[cand] > winner_f:
                    winner_f = fit_list[cand]
                    winner_idx = cand
            return self.population[winner_idx]

        # ── Reproduction ─────────────────────────────────────────────────
        next_pop: List[MetaGenome] = [elite_genome]

        while len(next_pop) < pop_size:
            # R33: Select fresh parents per pair to maximize genetic
            # diversity across the meta-population. Previously, parents
            # were selected once for the entire population, meaning ALL
            # children came from the same two parents -- severely limiting
            # the effective search breadth of meta-evolution.
            parent1 = _tournament()
            parent2 = _tournament()

            # Optionally crossover the two selected parents
            # R35: Uses effective_cx_rate (diversity-adjusted) instead of
            # the static config rate, to break convergence in tight pops.
            if self._rng.random() < effective_cx_rate:
                child1, child2 = parent1.crossover(parent2, rng=self._rng)
            else:
                # R19: Use MetaGenome.__copy__ (30+ fields, avoids manual clone)
                child1 = copy.copy(parent1)
                child2 = copy.copy(parent2)

            # Mutate both children
            child1 = child1.mutate(std=self.config.meta_mutation_std, rng=self._rng)
            child2 = child2.mutate(std=self.config.meta_mutation_std, rng=self._rng)

            next_pop.append(child1)
            if len(next_pop) < pop_size:
                next_pop.append(child2)

        # Replace population and reset fitness
        self.population = next_pop[:pop_size]
        self.fitness = [float("-inf") for _ in range(pop_size)]

    def _log_generation(self, gen_info: Dict[str, Any]) -> None:
        """Log a formatted meta-generation summary via structured logging.

        Parameters
        ----------
        gen_info : Dict[str, Any]
            Generation statistics dict as produced by ``run()``.
        """
        mg = gen_info["meta_gen"]
        avg = gen_info["avg_fitness"]
        best = gen_info["best_fitness"]
        worst = gen_info["worst_fitness"]
        elapsed = gen_info["elapsed_seconds"]
        best_repr = gen_info["best_genome_repr"]

        sep = "─" * 60
        _logger.info(
            "\n%s\nMeta-Generation %d/%d\n%s\n"
            "  Best fitness : %.4f\n"
            "  Avg fitness  : %.4f\n"
            "  Worst fitness: %.4f\n"
            "  Elapsed      : %.2fs\n"
            "  Best genome  : %s\n%s",
            sep,
            mg + 1,
            self.config.meta_generations,
            sep,
            best,
            avg,
            worst,
            elapsed,
            best_repr,
            sep,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for the meta-evolution loop.

    Parses optional overrides for the :class:`MetaEvolutionConfig` fields
    and runs the meta-evolution, printing the best config at the end.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Meta-evolution: discover optimal RV32I evolution configs")
    parser.add_argument(
        "--meta-population",
        type=int,
        default=10,
        help="Number of meta-genomes in the population (default: 10)",
    )
    parser.add_argument(
        "--meta-generations",
        type=int,
        default=5,
        help="Number of meta-generations (default: 5)",
    )
    parser.add_argument(
        "--base-generations",
        type=int,
        default=20,
        help="Generations per base evolution run (default: 20)",
    )
    parser.add_argument(
        "--meta-mutation-std",
        type=float,
        default=0.15,
        help="Mutation std for meta-genome perturbation (default: 0.15)",
    )
    parser.add_argument(
        "--meta-crossover-rate",
        type=float,
        default=0.3,
        help="Crossover probability for meta-reproduction (default: 0.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    meta_config = MetaEvolutionConfig(
        meta_population_size=args.meta_population,
        meta_generations=args.meta_generations,
        base_generations=args.base_generations,
        meta_mutation_std=args.meta_mutation_std,
        meta_crossover_rate=args.meta_crossover_rate,
        seed=args.seed,
    )

    print("=" * 60)
    print("Meta-Evolution Engine")
    print("=" * 60)
    print(f"Meta config: {meta_config}")
    print()

    engine = MetaEvolutionEngine(meta_config=meta_config)
    meta_summary = engine.run()

    print("\n" + "=" * 60)
    print("Meta-Evolution Complete")
    print("=" * 60)
    print(f"  Meta-generations: {meta_summary['meta_generations_completed']}")
    print(f"  Best meta-fitness: {meta_summary['best_meta_fitness']:.4f}")
    print(f"  Best genome: {meta_summary['best_genome_repr']}")
    print(f"  Total elapsed: {meta_summary['total_elapsed_seconds']:.2f}s")
    print()

    best_evo_config = engine.best_config()
    print("Best EvolutionConfig:")
    print(f"  population_size: {best_evo_config.population_size}")
    print(f"  mutation_rate: {best_evo_config.mutation_rate:.4f}")
    print(f"  crossover_rate: {best_evo_config.crossover_rate:.4f}")
    print(f"  elite_ratio: {best_evo_config.elite_ratio:.4f}")
    print(f"  tournament_size: {best_evo_config.tournament_size}")
    print(f"  curriculum_stall_threshold: {best_evo_config.curriculum_stall_threshold}")
    print(f"  novelty_weight: {best_evo_config.novelty_weight:.4f}")


if __name__ == "__main__":
    main()
