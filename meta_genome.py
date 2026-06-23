#!/usr/bin/env python3
"""
Meta-genome representation for evolvable strategy parameters
in the RV32I evolution engine.

A MetaGenome encodes the hyper-parameters that govern the evolutionary
search itself — mutation rates, selection pressure, curriculum dynamics —
allowing these to be optimised by a meta-evolution loop (or hand-tuned
through serialisation round-trips).
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from evolution_engine import EvolutionConfig

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Known mutation strategy labels — must match MutationStrategy enum values
# in mutation_proposer.py.
KNOWN_STRATEGIES: Tuple[str, ...] = ("RANDOM", "TASK_AWARE", "DIRECTED", "PERTURBATION_CAPPED")

# Valid ranges for each scalar parameter
_BOUNDS: Dict[str, Tuple[float, float]] = {
    "mutation_rate": (0.0, 1.0),
    "crossover_rate": (0.0, 1.0),
    "elite_ratio": (0.01, 0.5),
    "tournament_size": (2, 20),
    "population_size": (10, 200),
    "stall_threshold": (1, 20),
    "novelty_weight": (0.0, 1.0),
    # New meta-weight fields
    "meta_weight_improvement": (0.0, 1.0),
    "meta_weight_depth": (0.0, 1.0),
    "meta_weight_diversity": (0.0, 1.0),
    "max_fitness": (0.5, 5.0),
    "variance_penalty_coeff": (0.0, 1.0),
    "bias_register_prob": (0.0, 1.0),
    "anchor_retention_floor": (0.1, 1.0),
    "adaptive_elite_low": (0.01, 0.5),
    "adaptive_elite_mid": (0.01, 0.5),
    "adaptive_elite_high": (0.01, 0.5),
    "adaptive_elite_max": (0.01, 0.5),
    "adaptive_cx_mid_add": (0.0, 0.5),
    "adaptive_cx_high_add": (0.0, 0.5),
    "archive_crossover_success_trigger": (0.3, 1.0),
    "archive_all_low_trigger": (0.1, 0.8),
    "archive_cx_adjust_factor": (1.01, 1.5),
    "archive_pop_adjust_factor": (1.01, 2.0),
    "archive_elite_reduce_factor": (0.5, 0.99),
    # New int fields
    "max_level": (5, 20),
    "adaptive_tourney_mid_add": (0, 10),
    "adaptive_tourney_high_add": (0, 10),
    "archive_consecutive_wins_trigger": (1, 10),
    # Retrain pipeline fields
    "retrain_lr": (1e-7, 1.0),
    "retrain_batch_size": (1, 256),
    "retrain_weight_decay": (0.0, 1.0),
    "retrain_patience": (1, 100),
    "retrain_grad_clip": (0.01, 100.0),
    "retrain_max_epochs": (1, 500),
    "retrain_val_split": (0.01, 0.5),
    "retrain_freeze_layers": (0, 12),
    "retrain_max_per_level": (1, 5000),
    "retrain_timeout": (60, 86400),
    "retrain_min_examples": (1, 1000),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the inclusive interval [lo, hi]."""
    return max(lo, min(hi, value))


def _clamp_int(value: float, lo: int, hi: int) -> int:
    """Clamp *value* to the inclusive integer interval [lo, hi]."""
    return max(lo, min(hi, round(value)))


# ──────────────────────────────────────────────────────────────────────────────
# MetaGenome
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MetaGenome:
    """Evolvable encoding of evolution-engine hyper-parameters.

    Attributes
    ----------
    mutation_rate : float
        Probability of mutating an individual (clamped to [0.0, 1.0]).
    crossover_rate : float
        Probability of crossing over two parents (clamped to [0.0, 1.0]).
    strategy_type_weights : Dict[str, float]
        Weight assigned to each mutation strategy (keys are the strategy
        labels, e.g. ``"RANDOM"``, ``"TASK_AWARE"``, ``"DIRECTED"``,
        ``"PERTURBATION_CAPPED"``).  Weights are normalised to sum to 1.0 at
        construction time.
    elite_ratio : float
        Fraction of the population preserved as elite (clamped to
        [0.01, 0.5]).
    tournament_size : int
        Number of individuals in each tournament (clamped to [2, 20]).
    population_size : int
        Number of individuals in the population (clamped to [10, 200]).
    stall_threshold : int
        Generations without improvement before curriculum advances
        (clamped to [1, 20]).
    novelty_weight : float
        How much to weight novelty vs. fitness in selection
        (clamped to [0.0, 1.0]).
    """

    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    strategy_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "RANDOM": 0.25,
        "TASK_AWARE": 0.25,
        "DIRECTED": 0.25,
        "PERTURBATION_CAPPED": 0.25,
    })
    elite_ratio: float = 0.1
    tournament_size: int = 3
    population_size: int = 50
    stall_threshold: int = 3
    novelty_weight: float = 0.3

    # ── New float fields ──────────────────────────────────────────────────
    meta_weight_improvement: float = 0.3
    meta_weight_depth: float = 0.5
    meta_weight_diversity: float = 0.2
    max_fitness: float = 1.5
    variance_penalty_coeff: float = 0.2
    bias_register_prob: float = 0.5
    anchor_retention_floor: float = 0.5
    adaptive_elite_low: float = 0.05
    adaptive_elite_mid: float = 0.10
    adaptive_elite_high: float = 0.15
    adaptive_elite_max: float = 0.20
    adaptive_cx_mid_add: float = 0.2
    adaptive_cx_high_add: float = 0.3
    archive_crossover_success_trigger: float = 0.80
    archive_all_low_trigger: float = 0.30
    archive_cx_adjust_factor: float = 1.1
    archive_pop_adjust_factor: float = 1.25
    archive_elite_reduce_factor: float = 0.80

    # ── New int fields ────────────────────────────────────────────────────
    max_level: int = 13
    adaptive_tourney_mid_add: int = 3
    adaptive_tourney_high_add: int = 5
    archive_consecutive_wins_trigger: int = 3

    # ── Retrain pipeline fields (for meta-evolution of retrain params) ──
    retrain_lr: float = -1.0  # -1 = use default
    retrain_batch_size: float = -1.0  # -1 = use default (float for GA compatibility)
    retrain_weight_decay: float = -1.0
    retrain_patience: float = -1.0
    retrain_grad_clip: float = -1.0
    retrain_max_epochs: float = -1.0
    retrain_val_split: float = -1.0
    retrain_freeze_layers: float = -1.0
    retrain_max_per_level: float = -1.0
    retrain_timeout: float = -1.0
    retrain_min_examples: float = -1.0

    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    # Life-cycle hooks
    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

    def __post_init__(self) -> None:
        """Clamp all fields to their valid ranges and normalise strategy
        weights."""
        self.mutation_rate = _clamp(self.mutation_rate, 0.0, 1.0)
        self.crossover_rate = _clamp(self.crossover_rate, 0.0, 1.0)
        self.elite_ratio = _clamp(self.elite_ratio, 0.01, 0.5)
        self.tournament_size = _clamp_int(self.tournament_size, 2, 20)
        self.population_size = _clamp_int(self.population_size, 10, 200)
        self.stall_threshold = _clamp_int(self.stall_threshold, 1, 20)
        self.novelty_weight = _clamp(self.novelty_weight, 0.0, 1.0)

        # ── Clamp new float fields ────────────────────────────────────
        lo, hi = _BOUNDS["meta_weight_improvement"]
        self.meta_weight_improvement = _clamp(self.meta_weight_improvement, lo, hi)
        lo, hi = _BOUNDS["meta_weight_depth"]
        self.meta_weight_depth = _clamp(self.meta_weight_depth, lo, hi)
        lo, hi = _BOUNDS["meta_weight_diversity"]
        self.meta_weight_diversity = _clamp(self.meta_weight_diversity, lo, hi)
        lo, hi = _BOUNDS["max_fitness"]
        self.max_fitness = _clamp(self.max_fitness, lo, hi)
        lo, hi = _BOUNDS["variance_penalty_coeff"]
        self.variance_penalty_coeff = _clamp(self.variance_penalty_coeff, lo, hi)
        lo, hi = _BOUNDS["bias_register_prob"]
        self.bias_register_prob = _clamp(self.bias_register_prob, lo, hi)
        lo, hi = _BOUNDS["anchor_retention_floor"]
        self.anchor_retention_floor = _clamp(self.anchor_retention_floor, lo, hi)
        lo, hi = _BOUNDS["adaptive_elite_low"]
        self.adaptive_elite_low = _clamp(self.adaptive_elite_low, lo, hi)
        lo, hi = _BOUNDS["adaptive_elite_mid"]
        self.adaptive_elite_mid = _clamp(self.adaptive_elite_mid, lo, hi)
        lo, hi = _BOUNDS["adaptive_elite_high"]
        self.adaptive_elite_high = _clamp(self.adaptive_elite_high, lo, hi)
        lo, hi = _BOUNDS["adaptive_elite_max"]
        self.adaptive_elite_max = _clamp(self.adaptive_elite_max, lo, hi)
        lo, hi = _BOUNDS["adaptive_cx_mid_add"]
        self.adaptive_cx_mid_add = _clamp(self.adaptive_cx_mid_add, lo, hi)
        lo, hi = _BOUNDS["adaptive_cx_high_add"]
        self.adaptive_cx_high_add = _clamp(self.adaptive_cx_high_add, lo, hi)
        lo, hi = _BOUNDS["archive_crossover_success_trigger"]
        self.archive_crossover_success_trigger = _clamp(self.archive_crossover_success_trigger, lo, hi)
        lo, hi = _BOUNDS["archive_all_low_trigger"]
        self.archive_all_low_trigger = _clamp(self.archive_all_low_trigger, lo, hi)
        lo, hi = _BOUNDS["archive_cx_adjust_factor"]
        self.archive_cx_adjust_factor = _clamp(self.archive_cx_adjust_factor, lo, hi)
        lo, hi = _BOUNDS["archive_pop_adjust_factor"]
        self.archive_pop_adjust_factor = _clamp(self.archive_pop_adjust_factor, lo, hi)
        lo, hi = _BOUNDS["archive_elite_reduce_factor"]
        self.archive_elite_reduce_factor = _clamp(self.archive_elite_reduce_factor, lo, hi)

        # ── Clamp new int fields ──────────────────────────────────────
        lo, hi = _BOUNDS["max_level"]
        self.max_level = _clamp_int(self.max_level, int(lo), int(hi))
        lo, hi = _BOUNDS["adaptive_tourney_mid_add"]
        self.adaptive_tourney_mid_add = _clamp_int(self.adaptive_tourney_mid_add, int(lo), int(hi))
        lo, hi = _BOUNDS["adaptive_tourney_high_add"]
        self.adaptive_tourney_high_add = _clamp_int(self.adaptive_tourney_high_add, int(lo), int(hi))
        lo, hi = _BOUNDS["archive_consecutive_wins_trigger"]
        self.archive_consecutive_wins_trigger = _clamp_int(self.archive_consecutive_wins_trigger, int(lo), int(hi))

        # Ensure all known strategies are present; fill missing with 0.0.
        for s in KNOWN_STRATEGIES:
            self.strategy_type_weights.setdefault(s, 0.0)

        # Remove any stale keys no longer in KNOWN_STRATEGIES.
        stale = [k for k in self.strategy_type_weights if k not in KNOWN_STRATEGIES]
        for k in stale:
            del self.strategy_type_weights[k]

        # Normalise weights to sum to 1.0.
        self._normalise_weights()

    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    # Internal helpers
    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

    def _normalise_weights(self) -> None:
        """Normalise ``strategy_type_weights`` so they sum to 1.0.

        If all weights are zero, reset to a uniform distribution.
        """
        total = sum(self.strategy_type_weights.values())
        if total <= 0.0:
            n = len(KNOWN_STRATEGIES)
            for s in KNOWN_STRATEGIES:
                self.strategy_type_weights[s] = 1.0 / n
        else:
            for s in list(self.strategy_type_weights):
                self.strategy_type_weights[s] /= total

    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    # Public API
    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

    def mutate(
        self,
        std: float = 0.1,
        reset_prob: float = 0.05,
        rng: Optional[random.Random] = None,
    ) -> MetaGenome:
        """Return a new ``MetaGenome`` obtained by perturbing this one.

        Each scalar-typed field is perturbed by adding Gaussian noise
        with mean 0 and standard deviation *std*, then clamped to its
        valid range.  Integer fields are rounded after perturbation.

        With probability *reset_prob*, the entire genome is replaced by
        a freshly-sampled random genome (a "reset" that prevents
        premature convergence of the meta-evolution loop).

        Parameters
        ----------
        std : float
            Standard deviation of the Gaussian perturbation applied to
            each normalised scalar field.  The perturbation is applied
            *after* the field has been converted to a [0, 1] scale for
            fields with natural bounds.
        reset_prob : float
            Probability (in [0, 1]) of resetting to a completely random
            genome instead of applying the perturbation.
        rng : Optional[random.Random]
            Optional seeded ``Random`` instance for reproducibility.
            If ``None``, a new local ``Random`` instance is created
            (seeded from system entropy).

        Returns
        -------
        MetaGenome
            A new, mutated meta-genome.  The original is not modified.
        """
        _rng: random.Random = rng if rng is not None else random.Random()

        # Occasional full reset.
        if _rng.random() < reset_prob:
            return MetaGenome.random(rng=_rng)

        child = copy.deepcopy(self)

        # ── Scalar perturbations ──────────────────────────────────────
        child.mutation_rate = _clamp(
            child.mutation_rate + _rng.gauss(0.0, std), 0.0, 1.0
        )
        child.crossover_rate = _clamp(
            child.crossover_rate + _rng.gauss(0.0, std), 0.0, 1.0
        )
        child.elite_ratio = _clamp(
            child.elite_ratio + _rng.gauss(0.0, std), 0.01, 0.5
        )
        child.tournament_size = _clamp_int(
            child.tournament_size + _rng.gauss(0.0, std * 5), 2, 20
        )
        child.population_size = _clamp_int(
            child.population_size + _rng.gauss(0.0, std * 10), 10, 200
        )
        child.stall_threshold = _clamp_int(
            child.stall_threshold + _rng.gauss(0.0, std * 3), 1, 20
        )
        child.novelty_weight = _clamp(
            child.novelty_weight + _rng.gauss(0.0, std), 0.0, 1.0
        )

        # ── New float perturbations ───────────────────────────────────
        lo, hi = _BOUNDS["meta_weight_improvement"]
        child.meta_weight_improvement = _clamp(child.meta_weight_improvement + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["meta_weight_depth"]
        child.meta_weight_depth = _clamp(child.meta_weight_depth + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["meta_weight_diversity"]
        child.meta_weight_diversity = _clamp(child.meta_weight_diversity + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["max_fitness"]
        child.max_fitness = _clamp(child.max_fitness + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["variance_penalty_coeff"]
        child.variance_penalty_coeff = _clamp(child.variance_penalty_coeff + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["bias_register_prob"]
        child.bias_register_prob = _clamp(child.bias_register_prob + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["anchor_retention_floor"]
        child.anchor_retention_floor = _clamp(child.anchor_retention_floor + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["adaptive_elite_low"]
        child.adaptive_elite_low = _clamp(child.adaptive_elite_low + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["adaptive_elite_mid"]
        child.adaptive_elite_mid = _clamp(child.adaptive_elite_mid + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["adaptive_elite_high"]
        child.adaptive_elite_high = _clamp(child.adaptive_elite_high + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["adaptive_elite_max"]
        child.adaptive_elite_max = _clamp(child.adaptive_elite_max + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["adaptive_cx_mid_add"]
        child.adaptive_cx_mid_add = _clamp(child.adaptive_cx_mid_add + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["adaptive_cx_high_add"]
        child.adaptive_cx_high_add = _clamp(child.adaptive_cx_high_add + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["archive_crossover_success_trigger"]
        child.archive_crossover_success_trigger = _clamp(child.archive_crossover_success_trigger + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["archive_all_low_trigger"]
        child.archive_all_low_trigger = _clamp(child.archive_all_low_trigger + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["archive_cx_adjust_factor"]
        child.archive_cx_adjust_factor = _clamp(child.archive_cx_adjust_factor + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["archive_pop_adjust_factor"]
        child.archive_pop_adjust_factor = _clamp(child.archive_pop_adjust_factor + _rng.gauss(0.0, std), lo, hi)
        lo, hi = _BOUNDS["archive_elite_reduce_factor"]
        child.archive_elite_reduce_factor = _clamp(child.archive_elite_reduce_factor + _rng.gauss(0.0, std), lo, hi)

        # ── New int perturbations ─────────────────────────────────────
        lo, hi = _BOUNDS["max_level"]
        child.max_level = _clamp_int(child.max_level + _rng.gauss(0.0, std * 2), int(lo), int(hi))
        lo, hi = _BOUNDS["adaptive_tourney_mid_add"]
        child.adaptive_tourney_mid_add = _clamp_int(child.adaptive_tourney_mid_add + _rng.gauss(0.0, std * 2), int(lo), int(hi))
        lo, hi = _BOUNDS["adaptive_tourney_high_add"]
        child.adaptive_tourney_high_add = _clamp_int(child.adaptive_tourney_high_add + _rng.gauss(0.0, std * 2), int(lo), int(hi))
        lo, hi = _BOUNDS["archive_consecutive_wins_trigger"]
        child.archive_consecutive_wins_trigger = _clamp_int(child.archive_consecutive_wins_trigger + _rng.gauss(0.0, std * 2), int(lo), int(hi))

        # ── Strategy weights: add noise and re-normalise ──────────────
        for s in KNOWN_STRATEGIES:
            child.strategy_type_weights[s] = max(
                0.0, child.strategy_type_weights[s] + _rng.gauss(0.0, std)
            )
        child._normalise_weights()

        return child

    def crossover(self, other: MetaGenome, rng: Optional[random.Random] = None) -> Tuple[MetaGenome, MetaGenome]:
        """Return two offspring via uniform crossover with *other*.

        For each scalar attribute the child inherits from one parent
        with equal probability.  Integer fields are inherited directly
        (no interpolation).  Strategy-type weights are crossed over
        per-strategy and re-normalised afterwards.

        Parameters
        ----------
        other : MetaGenome
            The other parent meta-genome.
        rng : Optional[random.Random]
            Optional seeded ``Random`` instance for reproducibility.

        Returns
        -------
        Tuple[MetaGenome, MetaGenome]
            Two offspring genomes.
        """
        _rng: random.Random = rng if rng is not None else random.Random()

        def _pick(a: float, b: float) -> float:
            return a if _rng.random() < 0.5 else b

        def _pick_int(a: int, b: int) -> int:
            return a if _rng.random() < 0.5 else b

        def _pick_weight(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
            child_w: Dict[str, float] = {}
            for s in KNOWN_STRATEGIES:
                child_w[s] = _pick(a.get(s, 0.0), b.get(s, 0.0))
            return child_w

        child1 = MetaGenome(
            mutation_rate=_pick(self.mutation_rate, other.mutation_rate),
            crossover_rate=_pick(self.crossover_rate, other.crossover_rate),
            strategy_type_weights=_pick_weight(
                self.strategy_type_weights, other.strategy_type_weights
            ),
            elite_ratio=_pick(self.elite_ratio, other.elite_ratio),
            tournament_size=_pick_int(self.tournament_size, other.tournament_size),
            population_size=_pick_int(self.population_size, other.population_size),
            stall_threshold=_pick_int(self.stall_threshold, other.stall_threshold),
            novelty_weight=_pick(self.novelty_weight, other.novelty_weight),
            # New float fields
            meta_weight_improvement=_pick(self.meta_weight_improvement, other.meta_weight_improvement),
            meta_weight_depth=_pick(self.meta_weight_depth, other.meta_weight_depth),
            meta_weight_diversity=_pick(self.meta_weight_diversity, other.meta_weight_diversity),
            max_fitness=_pick(self.max_fitness, other.max_fitness),
            variance_penalty_coeff=_pick(self.variance_penalty_coeff, other.variance_penalty_coeff),
            bias_register_prob=_pick(self.bias_register_prob, other.bias_register_prob),
            anchor_retention_floor=_pick(self.anchor_retention_floor, other.anchor_retention_floor),
            adaptive_elite_low=_pick(self.adaptive_elite_low, other.adaptive_elite_low),
            adaptive_elite_mid=_pick(self.adaptive_elite_mid, other.adaptive_elite_mid),
            adaptive_elite_high=_pick(self.adaptive_elite_high, other.adaptive_elite_high),
            adaptive_elite_max=_pick(self.adaptive_elite_max, other.adaptive_elite_max),
            adaptive_cx_mid_add=_pick(self.adaptive_cx_mid_add, other.adaptive_cx_mid_add),
            adaptive_cx_high_add=_pick(self.adaptive_cx_high_add, other.adaptive_cx_high_add),
            archive_crossover_success_trigger=_pick(self.archive_crossover_success_trigger, other.archive_crossover_success_trigger),
            archive_all_low_trigger=_pick(self.archive_all_low_trigger, other.archive_all_low_trigger),
            archive_cx_adjust_factor=_pick(self.archive_cx_adjust_factor, other.archive_cx_adjust_factor),
            archive_pop_adjust_factor=_pick(self.archive_pop_adjust_factor, other.archive_pop_adjust_factor),
            archive_elite_reduce_factor=_pick(self.archive_elite_reduce_factor, other.archive_elite_reduce_factor),
            # New int fields
            max_level=_pick_int(self.max_level, other.max_level),
            adaptive_tourney_mid_add=_pick_int(self.adaptive_tourney_mid_add, other.adaptive_tourney_mid_add),
            adaptive_tourney_high_add=_pick_int(self.adaptive_tourney_high_add, other.adaptive_tourney_high_add),
            archive_consecutive_wins_trigger=_pick_int(self.archive_consecutive_wins_trigger, other.archive_consecutive_wins_trigger),
        )

        # Second child gets the complement of each choice.
        def _complement(a: float, b: float, c1: float) -> float:
            return b if c1 == a else a

        def _complement_int(a: int, b: int, c1: int) -> int:
            return b if c1 == a else a

        def _complement_weights(
            a: Dict[str, float], b: Dict[str, float], c1: Dict[str, float]
        ) -> Dict[str, float]:
            child_w: Dict[str, float] = {}
            for s in KNOWN_STRATEGIES:
                av = a.get(s, 0.0)
                bv = b.get(s, 0.0)
                c1v = c1.get(s, 0.0)
                child_w[s] = bv if c1v == av else av
            return child_w

        child2 = MetaGenome(
            mutation_rate=_complement(self.mutation_rate, other.mutation_rate, child1.mutation_rate),
            crossover_rate=_complement(self.crossover_rate, other.crossover_rate, child1.crossover_rate),
            strategy_type_weights=_complement_weights(
                self.strategy_type_weights, other.strategy_type_weights,
                child1.strategy_type_weights,
            ),
            elite_ratio=_complement(self.elite_ratio, other.elite_ratio, child1.elite_ratio),
            tournament_size=_complement_int(self.tournament_size, other.tournament_size, child1.tournament_size),
            population_size=_complement_int(self.population_size, other.population_size, child1.population_size),
            stall_threshold=_complement_int(self.stall_threshold, other.stall_threshold, child1.stall_threshold),
            novelty_weight=_complement(self.novelty_weight, other.novelty_weight, child1.novelty_weight),
            # New float fields
            meta_weight_improvement=_complement(self.meta_weight_improvement, other.meta_weight_improvement, child1.meta_weight_improvement),
            meta_weight_depth=_complement(self.meta_weight_depth, other.meta_weight_depth, child1.meta_weight_depth),
            meta_weight_diversity=_complement(self.meta_weight_diversity, other.meta_weight_diversity, child1.meta_weight_diversity),
            max_fitness=_complement(self.max_fitness, other.max_fitness, child1.max_fitness),
            variance_penalty_coeff=_complement(self.variance_penalty_coeff, other.variance_penalty_coeff, child1.variance_penalty_coeff),
            bias_register_prob=_complement(self.bias_register_prob, other.bias_register_prob, child1.bias_register_prob),
            anchor_retention_floor=_complement(self.anchor_retention_floor, other.anchor_retention_floor, child1.anchor_retention_floor),
            adaptive_elite_low=_complement(self.adaptive_elite_low, other.adaptive_elite_low, child1.adaptive_elite_low),
            adaptive_elite_mid=_complement(self.adaptive_elite_mid, other.adaptive_elite_mid, child1.adaptive_elite_mid),
            adaptive_elite_high=_complement(self.adaptive_elite_high, other.adaptive_elite_high, child1.adaptive_elite_high),
            adaptive_elite_max=_complement(self.adaptive_elite_max, other.adaptive_elite_max, child1.adaptive_elite_max),
            adaptive_cx_mid_add=_complement(self.adaptive_cx_mid_add, other.adaptive_cx_mid_add, child1.adaptive_cx_mid_add),
            adaptive_cx_high_add=_complement(self.adaptive_cx_high_add, other.adaptive_cx_high_add, child1.adaptive_cx_high_add),
            archive_crossover_success_trigger=_complement(self.archive_crossover_success_trigger, other.archive_crossover_success_trigger, child1.archive_crossover_success_trigger),
            archive_all_low_trigger=_complement(self.archive_all_low_trigger, other.archive_all_low_trigger, child1.archive_all_low_trigger),
            archive_cx_adjust_factor=_complement(self.archive_cx_adjust_factor, other.archive_cx_adjust_factor, child1.archive_cx_adjust_factor),
            archive_pop_adjust_factor=_complement(self.archive_pop_adjust_factor, other.archive_pop_adjust_factor, child1.archive_pop_adjust_factor),
            archive_elite_reduce_factor=_complement(self.archive_elite_reduce_factor, other.archive_elite_reduce_factor, child1.archive_elite_reduce_factor),
            # New int fields
            max_level=_complement_int(self.max_level, other.max_level, child1.max_level),
            adaptive_tourney_mid_add=_complement_int(self.adaptive_tourney_mid_add, other.adaptive_tourney_mid_add, child1.adaptive_tourney_mid_add),
            adaptive_tourney_high_add=_complement_int(self.adaptive_tourney_high_add, other.adaptive_tourney_high_add, child1.adaptive_tourney_high_add),
            archive_consecutive_wins_trigger=_complement_int(self.archive_consecutive_wins_trigger, other.archive_consecutive_wins_trigger, child1.archive_consecutive_wins_trigger),
        )

        return child1, child2

    def decode(self) -> EvolutionConfig:
        """Decode this meta-genome into an ``EvolutionConfig``.

        Only the fields that the meta-genome encodes are set on the
        returned config; all other ``EvolutionConfig`` fields retain
        their current defaults.

        Returns
        -------
        EvolutionConfig
            A fully-populated evolution configuration reflecting the
            hyper-parameters encoded in this genome.
        """
        conf = EvolutionConfig(
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elite_ratio=self.elite_ratio,
            tournament_size=self.tournament_size,
            population_size=self.population_size,
            curriculum_stall_threshold=self.stall_threshold,
            novelty_weight=self.novelty_weight,
            # New float fields
            meta_weight_improvement=self.meta_weight_improvement,
            meta_weight_depth=self.meta_weight_depth,
            meta_weight_diversity=self.meta_weight_diversity,
            max_fitness=self.max_fitness,
            variance_penalty_coeff=self.variance_penalty_coeff,
            bias_register_prob=self.bias_register_prob,
            anchor_retention_floor=self.anchor_retention_floor,
            adaptive_elite_low=self.adaptive_elite_low,
            adaptive_elite_mid=self.adaptive_elite_mid,
            adaptive_elite_high=self.adaptive_elite_high,
            adaptive_elite_max=self.adaptive_elite_max,
            adaptive_cx_mid_add=self.adaptive_cx_mid_add,
            adaptive_cx_high_add=self.adaptive_cx_high_add,
            archive_crossover_success_trigger=self.archive_crossover_success_trigger,
            archive_all_low_trigger=self.archive_all_low_trigger,
            archive_cx_adjust_factor=self.archive_cx_adjust_factor,
            archive_pop_adjust_factor=self.archive_pop_adjust_factor,
            archive_elite_reduce_factor=self.archive_elite_reduce_factor,
            # New int fields
            max_level=self.max_level,
            adaptive_tourney_mid_add=self.adaptive_tourney_mid_add,
            adaptive_tourney_high_add=self.adaptive_tourney_high_add,
            archive_consecutive_wins_trigger=self.archive_consecutive_wins_trigger,
        )
        return conf

    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    # Class methods
    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

    @classmethod
    def random(cls, rng: Optional[random.Random] = None) -> MetaGenome:
        """Create a ``MetaGenome`` with all fields sampled uniformly at
        random from their valid ranges.

        Parameters
        ----------
        rng : Optional[random.Random]
            Optional seeded ``Random`` instance for reproducibility.
            If ``None``, a new local ``Random`` instance is created.

        Returns
        -------
        MetaGenome
            A randomly-initialised meta-genome.
        """
        _rng: random.Random = rng if rng is not None else random.Random()

        lo_s, hi_s = _BOUNDS["mutation_rate"]
        mutation_rate = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["crossover_rate"]
        crossover_rate = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["elite_ratio"]
        elite_ratio = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["tournament_size"]
        tournament_size = _rng.randint(int(lo_s), int(hi_s))

        lo_s, hi_s = _BOUNDS["population_size"]
        population_size = _rng.randint(int(lo_s), int(hi_s))

        lo_s, hi_s = _BOUNDS["stall_threshold"]
        stall_threshold = _rng.randint(int(lo_s), int(hi_s))

        lo_s, hi_s = _BOUNDS["novelty_weight"]
        novelty_weight = _rng.uniform(lo_s, hi_s)

        # ── Sample new float fields ───────────────────────────────────
        lo_s, hi_s = _BOUNDS["meta_weight_improvement"]
        meta_weight_improvement = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["meta_weight_depth"]
        meta_weight_depth = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["meta_weight_diversity"]
        meta_weight_diversity = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["max_fitness"]
        max_fitness = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["variance_penalty_coeff"]
        variance_penalty_coeff = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["bias_register_prob"]
        bias_register_prob = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["anchor_retention_floor"]
        anchor_retention_floor = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["adaptive_elite_low"]
        adaptive_elite_low = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["adaptive_elite_mid"]
        adaptive_elite_mid = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["adaptive_elite_high"]
        adaptive_elite_high = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["adaptive_elite_max"]
        adaptive_elite_max = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["adaptive_cx_mid_add"]
        adaptive_cx_mid_add = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["adaptive_cx_high_add"]
        adaptive_cx_high_add = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["archive_crossover_success_trigger"]
        archive_crossover_success_trigger = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["archive_all_low_trigger"]
        archive_all_low_trigger = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["archive_cx_adjust_factor"]
        archive_cx_adjust_factor = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["archive_pop_adjust_factor"]
        archive_pop_adjust_factor = _rng.uniform(lo_s, hi_s)

        lo_s, hi_s = _BOUNDS["archive_elite_reduce_factor"]
        archive_elite_reduce_factor = _rng.uniform(lo_s, hi_s)

        # ── Sample new int fields ─────────────────────────────────────
        lo_s, hi_s = _BOUNDS["max_level"]
        max_level = _rng.randint(int(lo_s), int(hi_s))

        lo_s, hi_s = _BOUNDS["adaptive_tourney_mid_add"]
        adaptive_tourney_mid_add = _rng.randint(int(lo_s), int(hi_s))

        lo_s, hi_s = _BOUNDS["adaptive_tourney_high_add"]
        adaptive_tourney_high_add = _rng.randint(int(lo_s), int(hi_s))

        lo_s, hi_s = _BOUNDS["archive_consecutive_wins_trigger"]
        archive_consecutive_wins_trigger = _rng.randint(int(lo_s), int(hi_s))

        # Sample strategy weights from Dirichlet-equivalent: sample n
        # uniform values, then normalise.
        raw_weights = [_rng.random() for _ in KNOWN_STRATEGIES]
        total = sum(raw_weights)
        strategy_type_weights = {
            s: w / total for s, w in zip(KNOWN_STRATEGIES, raw_weights)
        }

        return cls(
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            strategy_type_weights=strategy_type_weights,
            elite_ratio=elite_ratio,
            tournament_size=tournament_size,
            population_size=population_size,
            stall_threshold=stall_threshold,
            novelty_weight=novelty_weight,
            # New float fields
            meta_weight_improvement=meta_weight_improvement,
            meta_weight_depth=meta_weight_depth,
            meta_weight_diversity=meta_weight_diversity,
            max_fitness=max_fitness,
            variance_penalty_coeff=variance_penalty_coeff,
            bias_register_prob=bias_register_prob,
            anchor_retention_floor=anchor_retention_floor,
            adaptive_elite_low=adaptive_elite_low,
            adaptive_elite_mid=adaptive_elite_mid,
            adaptive_elite_high=adaptive_elite_high,
            adaptive_elite_max=adaptive_elite_max,
            adaptive_cx_mid_add=adaptive_cx_mid_add,
            adaptive_cx_high_add=adaptive_cx_high_add,
            archive_crossover_success_trigger=archive_crossover_success_trigger,
            archive_all_low_trigger=archive_all_low_trigger,
            archive_cx_adjust_factor=archive_cx_adjust_factor,
            archive_pop_adjust_factor=archive_pop_adjust_factor,
            archive_elite_reduce_factor=archive_elite_reduce_factor,
            # New int fields
            max_level=max_level,
            adaptive_tourney_mid_add=adaptive_tourney_mid_add,
            adaptive_tourney_high_add=adaptive_tourney_high_add,
            archive_consecutive_wins_trigger=archive_consecutive_wins_trigger,
        )

    @classmethod
    def from_config(cls, config: EvolutionConfig) -> MetaGenome:
        """Construct a ``MetaGenome`` from an existing ``EvolutionConfig``.

        Fields in *config* that are not encoded by the meta-genome
        (e.g. ``max_generations``, ``seed``, ``instruction_budget``)
        are simply ignored — the meta-genome carries only parameters
        that are safe to evolve.

        Parameters
        ----------
        config : EvolutionConfig
            An evolution configuration to convert.

        Returns
        -------
        MetaGenome
            A meta-genome whose fields mirror the relevant portions
            of *config*.

        Raises
        ------
        TypeError
            If *config* is not an ``EvolutionConfig`` instance.
        """
        if not isinstance(config, EvolutionConfig):
            raise TypeError(
                f"Expected EvolutionConfig, got {type(config).__name__}"
            )

        return cls(
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate,
            strategy_type_weights={
                s: 1.0 / len(KNOWN_STRATEGIES) for s in KNOWN_STRATEGIES
            },
            elite_ratio=config.elite_ratio,
            tournament_size=config.tournament_size,
            population_size=config.population_size,
            stall_threshold=config.curriculum_stall_threshold,
            novelty_weight=config.novelty_weight,
            # New float fields
            meta_weight_improvement=config.meta_weight_improvement,
            meta_weight_depth=config.meta_weight_depth,
            meta_weight_diversity=config.meta_weight_diversity,
            max_fitness=config.max_fitness,
            variance_penalty_coeff=config.variance_penalty_coeff,
            bias_register_prob=config.bias_register_prob,
            anchor_retention_floor=config.anchor_retention_floor,
            adaptive_elite_low=config.adaptive_elite_low,
            adaptive_elite_mid=config.adaptive_elite_mid,
            adaptive_elite_high=config.adaptive_elite_high,
            adaptive_elite_max=config.adaptive_elite_max,
            adaptive_cx_mid_add=config.adaptive_cx_mid_add,
            adaptive_cx_high_add=config.adaptive_cx_high_add,
            archive_crossover_success_trigger=config.archive_crossover_success_trigger,
            archive_all_low_trigger=config.archive_all_low_trigger,
            archive_cx_adjust_factor=config.archive_cx_adjust_factor,
            archive_pop_adjust_factor=config.archive_pop_adjust_factor,
            archive_elite_reduce_factor=config.archive_elite_reduce_factor,
            # New int fields
            max_level=config.max_level,
            adaptive_tourney_mid_add=config.adaptive_tourney_mid_add,
            adaptive_tourney_high_add=config.adaptive_tourney_high_add,
            archive_consecutive_wins_trigger=config.archive_consecutive_wins_trigger,
        )

    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    # Convenience
    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

    def __repr__(self) -> str:
        return (
            f"MetaGenome("
            f"μ={self.mutation_rate:.3f}, "
            f"cx={self.crossover_rate:.3f}, "
            f"elite={self.elite_ratio:.3f}, "
            f"tourney={self.tournament_size}, "
            f"pop={self.population_size}, "
            f"stall={self.stall_threshold}, "
            f"novelty={self.novelty_weight:.3f}, "
            f"weights={self.strategy_type_weights}, "
            f"meta_weight_improvement={self.meta_weight_improvement:.3f}, "
            f"meta_weight_depth={self.meta_weight_depth:.3f}, "
            f"meta_weight_diversity={self.meta_weight_diversity:.3f}, "
            f"max_fitness={self.max_fitness:.3f}, "
            f"max_level={self.max_level}, "
            f"variance_penalty_coeff={self.variance_penalty_coeff:.3f}, "
            f"bias_register_prob={self.bias_register_prob:.3f}, "
            f"anchor_retention_floor={self.anchor_retention_floor:.3f}, "
            f"adaptive_elite_low={self.adaptive_elite_low:.3f}, "
            f"adaptive_elite_mid={self.adaptive_elite_mid:.3f}, "
            f"adaptive_elite_high={self.adaptive_elite_high:.3f}, "
            f"adaptive_elite_max={self.adaptive_elite_max:.3f}, "
            f"adaptive_tourney_mid_add={self.adaptive_tourney_mid_add}, "
            f"adaptive_tourney_high_add={self.adaptive_tourney_high_add}, "
            f"adaptive_cx_mid_add={self.adaptive_cx_mid_add:.3f}, "
            f"adaptive_cx_high_add={self.adaptive_cx_high_add:.3f}, "
            f"archive_crossover_success_trigger={self.archive_crossover_success_trigger:.3f}, "
            f"archive_all_low_trigger={self.archive_all_low_trigger:.3f}, "
            f"archive_consecutive_wins_trigger={self.archive_consecutive_wins_trigger}, "
            f"archive_cx_adjust_factor={self.archive_cx_adjust_factor:.3f}, "
            f"archive_pop_adjust_factor={self.archive_pop_adjust_factor:.3f}, "
            f"archive_elite_reduce_factor={self.archive_elite_reduce_factor:.3f})"
        )
