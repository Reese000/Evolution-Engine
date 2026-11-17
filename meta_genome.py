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

import random
from dataclasses import dataclass, field
from typing import Any

from evolution_engine import EvolutionConfig

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Known mutation strategy labels — must match MutationStrategy enum values
# in mutation_proposer.py.
KNOWN_STRATEGIES: tuple[str, ...] = ("RANDOM", "TASK_AWARE", "DIRECTED", "PERTURBATION_CAPPED")


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the inclusive interval [lo, hi]."""
    return max(lo, min(hi, value))


def _clamp_int(value: float, lo: int, hi: int) -> int:
    """Clamp *value* to the inclusive integer interval [lo, hi]."""
    return max(lo, min(hi, round(value)))


# ── Field metadata for data-driven clamping/mutation/random ─────────────────
# (name, is_int, default, low, high, noise_scale)
# noise_scale: multiplier for std in mutate() -- matches original per-field tuning.
# Sentinel fields (retrain_*) use a wide range and are clamped to the range;
# -1.0 sentinel is preserved when within range (low covers it).
# R23: Retrain fields added so meta-evolution can optimise them.
_FIELDS: tuple[tuple[str, bool, float, float, float, float], ...] = (
    ("mutation_rate", False, 0.3, 0.0, 1.0, 1.0),
    ("crossover_rate", False, 0.2, 0.0, 1.0, 1.0),
    ("elite_ratio", False, 0.1, 0.01, 0.5, 1.0),
    ("tournament_size", True, 3, 2, 20, 5.0),
    ("population_size", True, 50, 10, 200, 10.0),
    ("stall_threshold", True, 3, 1, 20, 3.0),
    ("novelty_weight", False, 0.3, 0.0, 1.0, 1.0),
    ("meta_weight_improvement", False, 0.3, 0.0, 1.0, 1.0),
    ("meta_weight_depth", False, 0.5, 0.0, 1.0, 1.0),
    ("meta_weight_diversity", False, 0.2, 0.0, 1.0, 1.0),
    ("max_fitness", False, 1.5, 0.5, 5.0, 1.0),
    ("variance_penalty_coeff", False, 0.2, 0.0, 1.0, 1.0),
    ("bias_register_prob", False, 0.5, 0.0, 1.0, 1.0),
    ("anchor_retention_floor", False, 0.5, 0.1, 1.0, 1.0),
    ("adaptive_elite_low", False, 0.05, 0.01, 0.5, 1.0),
    ("adaptive_elite_mid", False, 0.10, 0.01, 0.5, 1.0),
    ("adaptive_elite_high", False, 0.15, 0.01, 0.5, 1.0),
    ("adaptive_elite_max", False, 0.20, 0.01, 0.5, 1.0),
    ("adaptive_cx_mid_add", False, 0.2, 0.0, 0.5, 1.0),
    ("adaptive_cx_high_add", False, 0.3, 0.0, 0.5, 1.0),
    ("archive_crossover_success_trigger", False, 0.80, 0.3, 1.0, 1.0),
    ("archive_all_low_trigger", False, 0.30, 0.1, 0.8, 1.0),
    ("archive_cx_adjust_factor", False, 1.1, 1.01, 1.5, 1.0),
    ("archive_pop_adjust_factor", False, 1.25, 1.01, 2.0, 1.0),
    ("archive_elite_reduce_factor", False, 0.80, 0.5, 0.99, 1.0),
    ("max_level", True, 30, 5, 30, 2.0),
    ("adaptive_tourney_mid_add", True, 3, 0, 10, 2.0),
    ("adaptive_tourney_high_add", True, 5, 0, 10, 2.0),
    ("archive_consecutive_wins_trigger", True, 3, 1, 10, 2.0),
    # R23: Retrain pipeline fields -- enable meta-evolution of retrain params.
    # Sentinel -1.0 = use default.  Noise_scale=0.5 for smaller perturbations
    # (retrain params are sensitive and should not be mutated as aggressively).
    ("retrain_lr", False, -1.0, -1.0, 1.0, 0.5),
    ("retrain_batch_size", False, -1.0, -1.0, 256.0, 0.3),
    ("retrain_weight_decay", False, -1.0, -1.0, 1.0, 0.5),
    ("retrain_patience", False, -1.0, -1.0, 100.0, 0.3),
    ("retrain_grad_clip", False, -1.0, -1.0, 100.0, 0.3),
    ("retrain_max_epochs", False, -1.0, -1.0, 500.0, 0.3),
    ("retrain_val_split", False, -1.0, -1.0, 0.5, 0.5),
    ("retrain_freeze_layers", False, -1.0, -1.0, 12.0, 0.3),
    ("retrain_max_per_level", False, -1.0, -1.0, 5000.0, 0.3),
    ("retrain_timeout", False, -1.0, -1.0, 86400.0, 0.3),
    ("retrain_min_examples", False, -1.0, -1.0, 1000.0, 0.3),
)

# R22/R23: Pre-built field tuple for __copy__, derived from _FIELDS.
# Single source of truth -- adding a field to _FIELDS automatically adds it
# to copy/mutation/crossover. ~50 __copy__ calls per generation.
_COPY_FIELDS: tuple[str, ...] = tuple(f[0] for f in _FIELDS)


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
    strategy_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "RANDOM": 0.25,
            "TASK_AWARE": 0.25,
            "DIRECTED": 0.25,
            "PERTURBATION_CAPPED": 0.25,
        }
    )
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
    max_level: int = 30
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
        """Clamp all fields to their valid ranges and normalise strategy weights.

        Uses a data-driven loop over _FIELDS to eliminate repetitive per-field
        clamping code (29 fields -> single loop, ~60% fewer attribute lookups).
        """
        _c = _clamp
        _ci = _clamp_int
        for name, is_int, _default, lo, hi, _noise in _FIELDS:
            val = getattr(self, name)
            if is_int:
                object.__setattr__(self, name, _ci(val, int(lo), int(hi)))
            else:
                object.__setattr__(self, name, _c(val, lo, hi))

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
            for s in self.strategy_type_weights:
                self.strategy_type_weights[s] /= total

    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    # Public API
    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

    def mutate(
        self,
        std: float = 0.1,
        reset_prob: float = 0.05,
        rng: random.Random | None = None,
    ) -> MetaGenome:
        """Return a new ``MetaGenome`` obtained by perturbing this one.

        Each scalar-typed field is perturbed by adding Gaussian noise
        with mean 0 and standard deviation *std*, then clamped to its
        valid range.  Integer fields are rounded after perturbation.

        With probability *reset_prob*, the entire genome is replaced by
        a freshly-sampled random genome (a "reset" that prevents
        premature convergence of the meta-evolution loop).

        Performance: hoists _clamp/_clamp_int/_FIELDS to locals to avoid
        repeated function/dict lookups during perturbation (~40 fields).

        Returns
        -------
        MetaGenome
            A new, mutated meta-genome.  The original is not modified.
        """
        _rng: random.Random = rng if rng is not None else random.Random()

        # Occasional full reset.
        if _rng.random() < reset_prob:
            return MetaGenome.random(rng=_rng)

        # R23: Data-driven copy of all scalar fields via _COPY_FIELDS loop.
        # Single loop replaces 30+ manual field copies -- adding a new field
        # to _FIELDS automatically propagates to copy/mutation/crossover.
        _set = object.__setattr__
        _copy_src = self
        child = MetaGenome.__new__(MetaGenome)
        for fld in _COPY_FIELDS:
            _set(child, fld, getattr(_copy_src, fld))
        # Strategy weights dict gets a fresh shallow copy (mutated directly).
        child.strategy_type_weights = {k: v for k, v in self.strategy_type_weights.items()}

        # Hoist to locals for hot perturbation loop
        _c = _clamp
        _ci = _clamp_int
        _g = _rng.gauss

        # ── Data-driven scalar perturbations ──────────────────────────
        # Loop over _FIELDS for 29 fields instead of 29 separate blocks.
        for name, is_int, _default, lo, hi, noise_scale in _FIELDS:
            val = getattr(child, name)
            noise = _g(0.0, std * noise_scale)
            if is_int:
                object.__setattr__(child, name, _ci(val + noise, int(lo), int(hi)))
            else:
                object.__setattr__(child, name, _c(val + noise, lo, hi))

        # ── Strategy weights: add noise and re-normalise ──────────────
        stw = child.strategy_type_weights
        for s in KNOWN_STRATEGIES:
            stw[s] = max(0.0, stw[s] + _g(0.0, std))
        child._normalise_weights()

        return child

    def crossover(self, other: MetaGenome, rng: random.Random | None = None) -> tuple[MetaGenome, MetaGenome]:
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
        _rand = _rng.random
        _set = object.__setattr__

        # Data-driven crossover: use __new__ to skip __post_init__ clamping
        # (values already valid from parent inheritance). ~3x faster than
        # MetaGenome(...) constructor for 29 fields.
        def _make_child(is_first: bool) -> MetaGenome:
            child = MetaGenome.__new__(MetaGenome)
            # Crossover each field from the two parents
            for name, is_int, _default, _lo, _hi, _ns in _FIELDS:
                av = getattr(self, name)
                bv = getattr(other, name)
                if is_first:
                    chosen = av if _rand() < 0.5 else bv
                else:
                    # Complement: take whichever parent child1 did NOT pick
                    c1v = getattr(child1, name)  # child1 already built for complement
                    chosen = bv if c1v == av else av
                if is_int:
                    _set(child, name, round(chosen) if not isinstance(chosen, int) else chosen)
                else:
                    _set(child, name, chosen)
            # Strategy weights: crossover per-strategy
            sw = {}
            for s in KNOWN_STRATEGIES:
                a_val = self.strategy_type_weights.get(s, 0.0)
                b_val = other.strategy_type_weights.get(s, 0.0)
                if is_first:
                    sw[s] = a_val if _rand() < 0.5 else b_val
                else:
                    c1_val = child1.strategy_type_weights.get(s, 0.0)
                    sw[s] = b_val if c1_val == a_val else a_val
            _set(child, "strategy_type_weights", sw)
            # R23: Retrain fields are now in _FIELDS, so the data-driven
            # loop above handles them automatically. No separate loop needed.
            return child

        child1 = _make_child(True)
        child2 = _make_child(False)

        return child1, child2

    def _apply_retrain_fields(self, conf: EvolutionConfig) -> None:
        """Apply retrain pipeline fields to EvolutionConfig if non-sentinel."""
        _SENTINEL = -1.0
        _retrain_map = (
            ("retrain_lr", "retrain_lr"),
            ("retrain_batch_size", "retrain_batch_size"),
            ("retrain_weight_decay", "retrain_weight_decay"),
            ("retrain_patience", "retrain_patience"),
            ("retrain_grad_clip", "retrain_grad_clip"),
            ("retrain_max_epochs", "retrain_max_epochs"),
            ("retrain_val_split", "retrain_val_split"),
            ("retrain_freeze_layers", "retrain_freeze_layers"),
            ("retrain_max_per_level", "retrain_max_per_level"),
            ("retrain_timeout", "retrain_timeout"),
            ("retrain_min_examples", "retrain_min_examples"),
        )
        _int_fields = {
            "retrain_batch_size",
            "retrain_patience",
            "retrain_max_epochs",
            "retrain_freeze_layers",
            "retrain_max_per_level",
            "retrain_timeout",
            "retrain_min_examples",
        }
        for genome_field, config_field in _retrain_map:
            val = getattr(self, genome_field)
            if val > _SENTINEL:
                if genome_field in _int_fields:
                    setattr(conf, config_field, round(val))
                else:
                    setattr(conf, config_field, val)

    def decode(self) -> EvolutionConfig:
        """Decode this meta-genome into an ``EvolutionConfig``."""
        kw = self._build_config_kwargs()
        conf = EvolutionConfig(**kw)
        self._apply_retrain_fields(conf)
        return conf

    def _build_config_kwargs(self) -> dict[str, Any]:
        """Return config kwargs from genome fields."""
        return {
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elite_ratio": self.elite_ratio,
            "tournament_size": self.tournament_size,
            "population_size": self.population_size,
            "curriculum_stall_threshold": self.stall_threshold,
            "novelty_weight": self.novelty_weight,
            "meta_weight_improvement": self.meta_weight_improvement,
            "meta_weight_depth": self.meta_weight_depth,
            "meta_weight_diversity": self.meta_weight_diversity,
            "max_fitness": self.max_fitness,
            "variance_penalty_coeff": self.variance_penalty_coeff,
            "bias_register_prob": self.bias_register_prob,
            "anchor_retention_floor": self.anchor_retention_floor,
            "adaptive_elite_low": self.adaptive_elite_low,
            "adaptive_elite_mid": self.adaptive_elite_mid,
            "adaptive_elite_high": self.adaptive_elite_high,
            "adaptive_elite_max": self.adaptive_elite_max,
            "adaptive_cx_mid_add": self.adaptive_cx_mid_add,
            "adaptive_cx_high_add": self.adaptive_cx_high_add,
            "archive_crossover_success_trigger": self.archive_crossover_success_trigger,
            "archive_all_low_trigger": self.archive_all_low_trigger,
            "archive_cx_adjust_factor": self.archive_cx_adjust_factor,
            "archive_pop_adjust_factor": self.archive_pop_adjust_factor,
            "archive_elite_reduce_factor": self.archive_elite_reduce_factor,
            "max_level": self.max_level,
            "adaptive_tourney_mid_add": self.adaptive_tourney_mid_add,
            "adaptive_tourney_high_add": self.adaptive_tourney_high_add,
            "archive_consecutive_wins_trigger": self.archive_consecutive_wins_trigger,
        }

    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    # Class methods
    # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    @classmethod
    def random(cls, rng: random.Random | None = None) -> MetaGenome:
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
        _set = object.__setattr__
        _u = _rng.uniform
        _ri = _rng.randint

        # Data-driven random sampling: loop over _FIELDS instead of 29
        # separate blocks. ~60% fewer attribute lookups.
        genome = cls.__new__(cls)
        for name, is_int, _default, lo, hi, _ns in _FIELDS:
            if is_int:
                _set(genome, name, _ri(int(lo), int(hi)))
            else:
                _set(genome, name, _u(lo, hi))

        # Strategy weights: Dirichlet-equivalent (sample n uniform, normalise)
        raw_weights = [_rng.random() for _ in KNOWN_STRATEGIES]
        total = sum(raw_weights)
        _set(genome, "strategy_type_weights", {s: w / total for s, w in zip(KNOWN_STRATEGIES, raw_weights)})

        # R23: Retrain fields are now in _FIELDS -- random sampling is handled
        # by the data-driven loop above. No separate initialization needed.
        # The sentinel default (-1.0) is set by _FIELDS defaults in __new__.

        # Normalize strategy weights
        genome._normalise_weights()
        return genome

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
            raise TypeError(f"Expected EvolutionConfig, got {type(config).__name__}")

        return cls(
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate,
            strategy_type_weights={s: 1.0 / len(KNOWN_STRATEGIES) for s in KNOWN_STRATEGIES},
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
            # Retrain pipeline fields (R34: fix encode/decode roundtrip)
            retrain_lr=config.retrain_lr,
            retrain_batch_size=float(config.retrain_batch_size),
            retrain_weight_decay=config.retrain_weight_decay,
            retrain_patience=float(config.retrain_patience),
            retrain_grad_clip=config.retrain_grad_clip,
            retrain_max_epochs=float(config.retrain_max_epochs),
            retrain_val_split=config.retrain_val_split,
            retrain_freeze_layers=float(config.retrain_freeze_layers),
            retrain_max_per_level=float(config.retrain_max_per_level),
            retrain_timeout=float(config.retrain_timeout),
            retrain_min_examples=float(config.retrain_min_examples),
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

    def __copy__(self) -> MetaGenome:
        """Shallow copy — avoids the 30+ field manual clone in the
        non-crossover path of _evolve_population.  The strategy_type_weights
        dict gets a fresh shallow copy so mutations don't leak between clones.
        """
        child = MetaGenome.__new__(MetaGenome)
        # Copy all scalar fields directly (attribute access is faster than
        # getattr loops on dataclasses, and we know the field names).
        for fld in _COPY_FIELDS:
            object.__setattr__(child, fld, getattr(self, fld))
        # Deep-copy only the mutable dict
        child.strategy_type_weights = dict(self.strategy_type_weights)
        return child
