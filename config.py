#!/usr/bin/env python3
"""config.py — Unified configuration for the Evolution Engine.

Central Config class that loads and validates configuration from:
  1. Hardcoded safe defaults
  2. JSON config file (e.g., config/engine_config.json)
  3. Environment variable overrides (EE_*, EE_EVOLUTION_*, etc.)
  4. CLI args via argparse

Usage:
    from config import load_config
    cfg = load_config("config/engine_config.json")
    print(cfg.evolution.population_size)

    # Or with CLI overrides:
    cfg = load_config("config/engine_config.json",
                      max_cycles=50, target_fitness=100.0)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from paths import PROJECT_ROOT

# ══════════════════════════════════════════════════════════════════════
# Section dataclasses
# ══════════════════════════════════════════════════════════════════════


@dataclass
class EvolutionSection:
    """Evolution engine parameters."""

    population_size: int = 50
    max_generations: int = 200
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    elite_ratio: float = 0.1
    tournament_size: int = 3
    instruction_budget: int = 10000
    eval_budget: int = 5000
    seed: int = 42
    curriculum_level: int = 0
    auto_curriculum: bool = True
    curriculum_stall_threshold: int = 3
    checkpoint_interval: int = 25
    max_level: int = 13
    max_champion_words: int = 500
    num_eval_seeds: int = 3
    stall_threshold: int = 10
    diversity_threshold: float = 0.3
    diversity_inject_ratio: float = 0.1
    champion_crossover_prob: float = 0.7
    crossover_dedicated_prob: float = 0.05
    max_consecutive_same_strategy: int = 30
    novelty_weight: float = 0.3
    use_novelty_archive: bool = True
    novelty_threshold: float = 0.15
    use_qd_archive: bool = True
    qd_bonus_weight: float = 0.1
    qd_bins: int = 20
    use_gpu: bool = False
    use_strategy_archive: bool = True

    # Mutation rate scheduling
    use_mutation_scheduler: bool = True
    mutation_rate_start: float = 0.5
    mutation_rate_end: float = 0.1
    mutation_schedule_generations: int = 200

    # Temperature annealing
    use_temperature_annealing: bool = True
    temperature_start: float = 2.0
    temperature_end: float = 0.5
    temperature_schedule_generations: int = 200


@dataclass
class RetrainSection:
    """Retrain hyperparameters."""

    retrain_lr: float = 3e-5
    retrain_batch_size: int = 8
    retrain_weight_decay: float = 0.01
    retrain_grad_clip: float = 1.0
    retrain_patience: int = 3
    retrain_freeze_layers: int = 0
    retrain_model_size: str = "xlarge"
    retrain_max_epochs: int = 8
    retrain_val_split: float = 0.15
    retrain_max_per_level: int = 0
    retrain_timeout: int = 3600
    retrain_min_examples: int = 5
    augmentation_k: int = 8


@dataclass
class MetaControllerSection:
    """MetaController tuning bounds and thresholds."""

    lr_min: float = 1e-6
    lr_max: float = 3e-4
    freeze_min: int = 0
    freeze_max: int = 8
    patience_min: int = 2
    patience_max: int = 10
    fitness_plateau_cycles: int = 3
    level_stall_cycles: int = 5
    diversity_collapse_threshold: float = 0.05
    val_loss_regression_ratio: float = 1.5
    checkpoint_acceptance_tolerance: float = 1.05


@dataclass
class AutonomousRunnerSection:
    """Autonomous runner parameters."""

    max_cycles: int | None = 100
    target_fitness: float = 200.0
    generations_per_cycle: int = 100
    health_check_interval: int = 60
    state_file: str = "runner_state.json"
    stop_file: str = "stop"
    max_errors_before_stop: int = 10
    max_stagnation_cycles: int = 15
    max_self_correction_attempts: int = 5


# ══════════════════════════════════════════════════════════════════════
# Central Config
# ══════════════════════════════════════════════════════════════════════


class Config:
    """Central configuration loaded from JSON + env vars + CLI args.

    Sections:
      - evolution: Evolution engine parameters
      - retrain: Retrain hyperparameters
      - meta_controller: MetaController bounds/thresholds
      - autonomous_runner: Runner parameters

    Access via attribute notation:
        cfg.evolution.population_size
        cfg.retrain.retrain_lr
        cfg.autonomous_runner.max_cycles
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.evolution = EvolutionSection()
        self.retrain = RetrainSection()
        self.meta_controller = MetaControllerSection()
        self.autonomous_runner = AutonomousRunnerSection()

        # Load from file
        if config_path is not None:
            self._load_file(config_path)
        else:
            default_path = os.path.join(PROJECT_ROOT, "config", "engine_config.json")
            if os.path.isfile(default_path):
                self._load_file(str(default_path))

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Validate
        self._validate()

    def _load_file(self, config_path: str) -> None:
        """Load configuration from a JSON file."""
        full_path = os.path.join(PROJECT_ROOT, config_path) if not os.path.isabs(config_path) else config_path

        if not os.path.isfile(full_path):
            return

        with open(full_path, encoding="utf-8") as f:
            data = json.load(f)

        # Merge "evolution" section
        evo_data = data.get("evolution", {})
        for key, val in evo_data.items():
            if hasattr(self.evolution, key):
                setattr(self.evolution, key, val)

        # Merge "retrain" section
        retrain_data = data.get("retrain", {})
        for key, val in retrain_data.items():
            if hasattr(self.retrain, key):
                setattr(self.retrain, key, val)

        # Merge "meta_controller" section
        mc_data = data.get("meta_controller", {})
        for key, val in mc_data.items():
            if hasattr(self.meta_controller, key):
                setattr(self.meta_controller, key, val)

        # Merge "autonomous_runner" section
        ar_data = data.get("autonomous_runner", {})
        for key, val in ar_data.items():
            if hasattr(self.autonomous_runner, key):
                setattr(self.autonomous_runner, key, val)

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides.

        Naming convention:
          - EE_<PARAM> for top-level params
          - EE_EVOLUTION_<PARAM> for evolution section
          - EE_RETRAIN_<PARAM> for retrain section
          - EE_MC_<PARAM> for meta_controller section
          - EE_RUNNER_<PARAM> for autonomous_runner section
        """
        # Evolution overrides
        _apply_int_env("EE_POPULATION_SIZE", self.evolution, "population_size")
        _apply_int_env("EE_MAX_GENERATIONS", self.evolution, "max_generations")
        _apply_float_env("EE_MUTATION_RATE", self.evolution, "mutation_rate")
        _apply_float_env("EE_CROSSOVER_RATE", self.evolution, "crossover_rate")
        _apply_float_env("EE_ELITE_RATIO", self.evolution, "elite_ratio")
        _apply_int_env("EE_SEED", self.evolution, "seed")
        _apply_int_env("EE_MAX_LEVEL", self.evolution, "max_level")
        _apply_bool_env("EE_USE_GPU", self.evolution, "use_gpu")
        _apply_float_env("EE_MUTATION_RATE_START", self.evolution, "mutation_rate_start")
        _apply_float_env("EE_MUTATION_RATE_END", self.evolution, "mutation_rate_end")
        _apply_int_env("EE_MUTATION_SCHEDULE_GENERATIONS", self.evolution, "mutation_schedule_generations")
        _apply_float_env("EE_TEMPERATURE_START", self.evolution, "temperature_start")
        _apply_float_env("EE_TEMPERATURE_END", self.evolution, "temperature_end")
        _apply_int_env("EE_TEMPERATURE_SCHEDULE_GENERATIONS", self.evolution, "temperature_schedule_generations")

        # Retrain overrides
        _apply_float_env("EE_RETRAIN_LR", self.retrain, "retrain_lr")
        _apply_int_env("EE_RETRAIN_BATCH_SIZE", self.retrain, "retrain_batch_size")
        _apply_int_env("EE_RETRAIN_MAX_EPOCHS", self.retrain, "retrain_max_epochs")

        # Runner overrides
        _apply_int_env("EE_MAX_CYCLES", self.autonomous_runner, "max_cycles")
        _apply_float_env("EE_TARGET_FITNESS", self.autonomous_runner, "target_fitness")

    def _validate(self) -> None:
        """Validate configuration constraints at startup."""
        e = self.evolution
        r = self.retrain
        ar = self.autonomous_runner

        # Evolution validation
        if e.population_size < 1:
            raise ValueError(f"evolution.population_size must be >= 1, got {e.population_size}")
        if e.max_generations < 1:
            raise ValueError(f"evolution.max_generations must be >= 1, got {e.max_generations}")
        if not 0.0 <= e.mutation_rate <= 1.0:
            raise ValueError(f"evolution.mutation_rate must be in [0,1], got {e.mutation_rate}")
        if not 0.0 <= e.crossover_rate <= 1.0:
            raise ValueError(f"evolution.crossover_rate must be in [0,1], got {e.crossover_rate}")
        if not 0.0 <= e.elite_ratio <= 1.0:
            raise ValueError(f"evolution.elite_ratio must be in [0,1], got {e.elite_ratio}")
        if e.max_level < 1:
            raise ValueError(f"evolution.max_level must be >= 1, got {e.max_level}")
        if not 0.0 <= e.mutation_rate_start <= 1.0:
            raise ValueError(f"evolution.mutation_rate_start must be in [0,1], got {e.mutation_rate_start}")
        if not 0.0 <= e.mutation_rate_end <= 1.0:
            raise ValueError(f"evolution.mutation_rate_end must be in [0,1], got {e.mutation_rate_end}")
        if e.mutation_schedule_generations < 1:
            raise ValueError(
                f"evolution.mutation_schedule_generations must be >= 1, got {e.mutation_schedule_generations}"
            )
        if not 0.0 <= e.temperature_start <= 5.0:
            raise ValueError(f"evolution.temperature_start must be in [0,5], got {e.temperature_start}")
        if not 0.0 <= e.temperature_end <= 5.0:
            raise ValueError(f"evolution.temperature_end must be in [0,5], got {e.temperature_end}")
        if e.temperature_schedule_generations < 1:
            raise ValueError(
                f"evolution.temperature_schedule_generations must be >= 1, got {e.temperature_schedule_generations}"
            )

        # Retrain validation
        if r.retrain_lr <= 0:
            raise ValueError(f"retrain.retrain_lr must be > 0, got {r.retrain_lr}")
        if r.retrain_batch_size < 1:
            raise ValueError(f"retrain.retrain_batch_size must be >= 1, got {r.retrain_batch_size}")
        if r.retrain_max_epochs < 1:
            raise ValueError(f"retrain.retrain_max_epochs must be >= 1, got {r.retrain_max_epochs}")
        if not 0.0 <= r.retrain_val_split <= 1.0:
            raise ValueError(f"retrain.retrain_val_split must be in [0,1], got {r.retrain_val_split}")
        if r.augmentation_k < 1:
            raise ValueError(f"retrain.augmentation_k must be >= 1, got {r.augmentation_k}")

        # Runner validation
        if ar.max_cycles is not None and ar.max_cycles < 1:
            raise ValueError(f"autonomous_runner.max_cycles must be >= 1 or None (no limit), got {ar.max_cycles}")
        if ar.target_fitness < 0:
            raise ValueError(f"autonomous_runner.target_fitness must be >= 0, got {ar.target_fitness}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of all configuration."""
        return {
            "evolution": _dataclass_to_dict(self.evolution),
            "retrain": _dataclass_to_dict(self.retrain),
            "meta_controller": _dataclass_to_dict(self.meta_controller),
            "autonomous_runner": _dataclass_to_dict(self.autonomous_runner),
        }


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return {}


def _apply_int_env(env_var: str, target: Any, attr: str) -> None:
    """Apply an integer env var override to a target object.

    Handles float strings like "3.0" or "5.5" by rounding to the nearest
    integer instead of silently ignoring them.  Only truly non-numeric
    strings (e.g. "abc") are silently ignored.
    """
    val = os.environ.get(env_var)
    if val is not None and val.strip() != "":
        try:
            # First try direct int conversion
            setattr(target, attr, int(val))
        except (ValueError, TypeError):
            try:
                # Try float-then-round for strings like "3.0" or "5.5"
                setattr(target, attr, round(float(val)))
            except (ValueError, TypeError):
                pass


def _apply_float_env(env_var: str, target: Any, attr: str) -> None:
    """Apply a float env var override to a target object."""
    val = os.environ.get(env_var)
    if val is not None:
        try:
            setattr(target, attr, float(val))
        except (ValueError, TypeError):
            pass


def _apply_bool_env(env_var: str, target: Any, attr: str) -> None:
    """Apply a boolean env var override to a target object."""
    val = os.environ.get(env_var)
    if val is not None:
        setattr(target, attr, val.lower() in ("1", "true", "yes"))


def load_config(
    config_path: str | None = None,
    **overrides: Any,
) -> Config:
    """Load configuration with optional overrides.

    Args:
        config_path: Path to JSON config file (relative to PROJECT_ROOT).
        **overrides: Keyword overrides applied after file/env loading.
            Use dotted section names: max_cycles=50 sets
            autonomous_runner.max_cycles.

    Returns:
        A validated Config instance.
    """
    cfg = Config(config_path=config_path)

    # Apply CLI-style overrides
    if "max_cycles" in overrides:
        val = overrides["max_cycles"]
        cfg.autonomous_runner.max_cycles = None if val is None else int(val)
    if "target_fitness" in overrides:
        cfg.autonomous_runner.target_fitness = float(overrides["target_fitness"])
    if "population_size" in overrides:
        cfg.evolution.population_size = int(overrides["population_size"])
    if "generations_per_cycle" in overrides:
        cfg.autonomous_runner.generations_per_cycle = int(overrides["generations_per_cycle"])

    # Re-validate after overrides
    cfg._validate()

    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add standard configuration CLI arguments to an argparse parser.

    Call this on any entry-point script that wants config overrides:
        parser = argparse.ArgumentParser(...)
        add_config_args(parser)
        args = parser.parse_args()
        cfg = load_config(args.config, max_cycles=args.max_cycles)
    """
    parser.add_argument(
        "--config", type=str, default=None, help="Config file path (default: config/engine_config.json)"
    )
    parser.add_argument("--max-cycles", type=int, default=None, help="Override max cycles from config")
    parser.add_argument("--target-fitness", type=float, default=None, help="Override target fitness from config")
    parser.add_argument("--population-size", type=int, default=None, help="Override population size from config")
