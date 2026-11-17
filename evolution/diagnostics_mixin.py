"""DiagnosticsMixin - Logging, diagnostics, summaries, state export.

Extracted from EvolutionEngine in evolution_engine.py.
Uses mixin pattern: all methods access self.* from the host EvolutionEngine.
"""

from __future__ import annotations

import os
import time

from logging_config import get_logger

_logger = get_logger("EvolutionEngine")

from evolution.diversity_helpers import count_backward_branches as _count_backward_branches_fn  # noqa: E402


class DiagnosticsMixin:
    """Mixin providing logging, diagnostics, and state export methods."""

    def _log(self, message: str) -> None:
        """Log a message via structured logger and optional log file.



        Uses lazy file handle caching to avoid opening/closing a file handle

        on every log call. The handle is closed on interpreter shutdown or

        when the log directory changes.  Caches the log file path to avoid

        repeated os.path.join on every call (~hundreds per run).



        Respects _log_handle_closed flag: after _close_log_handle() is called

        (end of run()), file logging is permanently disabled for this run.

        """

        line = f"Gen {self.current_generation:4d} | {message}"

        _logger.info(line)

        # After _close_log_handle() is called, don't reopen the file

        if getattr(self, "_log_handle_closed", False):

            return

        log_dir = self.config.log_dir

        if log_dir:

            # Cache the full log path (only computed once)

            log_path = self._log_path_cache

            if log_path is None:

                log_path = os.path.join(log_dir, "evolution.log")

                self._log_path_cache = log_path

            try:
                if self._log_file is None:
                    self._log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
                self._log_file.write(line + "\n")
                self._log_file.flush()
            except (OSError, ValueError):
                self._log_file = None
                try:
                    self._log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
                    self._log_file.write(line + "\n")
                    self._log_file.flush()
                except (OSError, ValueError):
                    pass  # best-effort: structured logger already captured it

    def _log_generation(self, gen: int, start_time: float) -> None:
        """Log per-generation stats."""

        elapsed = time.time() - start_time

        champion = self.champion

        self._log(
            f"Level={self.champion_level}  "
            f"Fitness={self.champion_fitness:7.1f}  "
            f"W={len(champion) if champion else 0:3d}  "
            f"Stalled={self.generations_stalled:2d}  "
            f"T={elapsed:.2f}s"
        )

    @staticmethod
    def _count_backward_branches(program: list[int]) -> int:
        """Count backward branches (loop structures) in RV32I bytecode."""

        return _count_backward_branches_fn(program)

    def get_champion_program_analysis(self) -> dict:
        """Analyze the current champion program structure."""

        if self.champion is None:

            return {"exists": False}

        program = self.champion

        backward_branches = self._count_backward_branches(program)

        n_ops = len(program)

        # Count distinct opcodes

        opcodes = set()

        for word in program:

            if isinstance(word, int):

                opcodes.add(word & 0x7F)

        return {
            "exists": True,
            "word_count": n_ops,
            "unique_opcodes": len(opcodes),
            "backward_branches": backward_branches,
            "has_ebreak": any(isinstance(w, int) and w == 0x00100073 for w in program),
            "estimated_complexity": "simple" if n_ops <= 5 else "moderate" if n_ops <= 15 else "complex",
        }

    def get_champion_summary(self) -> str:
        """Return a one-line champion summary string."""

        if self.champion is None:

            return "No champion"

        return (
            f"L{self.champion_level} F={self.champion_fitness} "
            f"W={len(self.champion)} verified={getattr(self, '_champion_verified', False)}"
        )

    def get_run_diagnostics(self) -> dict:
        """Return comprehensive diagnostics for the current run state."""

        # Graceful None handling for archives

        novelty_size = 0

        if self.novelty_archive is not None:

            try:

                novelty_size = self.novelty_archive.size

            except (RuntimeError, AttributeError):

                novelty_size = 0

        strategy_records = 0

        if self.strategy_archive is not None:

            try:

                strategy_records = self.strategy_archive.total_records()

            except (RuntimeError, AttributeError):

                strategy_records = 0

        strategy_div = 0.0

        if self.strategy_archive is not None:

            try:

                strategy_div = self.strategy_archive.get_strategy_diversity(self.champion_level)

                if strategy_div is None:

                    strategy_div = 0.0

            except (RuntimeError, AttributeError):

                strategy_div = 0.0

        qd_size = 0

        if self.qd_archive is not None:

            try:

                qd_size = self.qd_archive.size

            except (RuntimeError, AttributeError):

                qd_size = 0

        return {
            # Champion state
            "champion_level": self.champion_level,
            "champion_fitness": self.champion_fitness,
            "champion_words": len(self.champion) if self.champion else 0,
            "champion_verified": getattr(self, "_champion_verified", False),
            "generations_stalled": self.generations_stalled,
            "current_generation": self.current_generation,
            # Archive sizes
            "novelty_archive_size": novelty_size,
            "qd_archive_size": qd_size,
            "strategy_archive_records": strategy_records,
            "hall_of_fame_size": len(self._hall_of_fame._heap),
            # Strategy diversity
            "strategy_diversity": strategy_div,
            # Training data
            "recurrent_examples": self.recurrent_examples_count,
            # Fitness velocity
            "fitness_velocity": getattr(self, "_fitness_velocity", 0.0),
            "fitness_momentum": getattr(self, "_fitness_momentum", 0.0),
            # Mutation stats
            "mutation_rate": self.config.mutation_rate,
            "population_size": self.config.population_size,
            "elite_ratio": self.config.elite_ratio,
        }

    def get_performance_metrics(self) -> dict:
        """Return performance metrics for the current run."""

        return {
            "current_generation": self.current_generation,
            "population_size": self.config.population_size,
            "elite_ratio": self.config.elite_ratio,
            "eval_budget": self.config.eval_budget,
            "model_guided_ratio": self._get_model_guided_ratio(),
            "mutation_rate": self.config.mutation_rate,
            "champion_words": len(self.champion) if self.champion else 0,
            "novelty_weight": self.config.novelty_weight,
            "qd_bonus_weight": self.config.qd_bonus_weight,
            "total_recurrent_examples": self.recurrent_examples_count,
            "hof_size": len(self._hall_of_fame._heap),
        }

    def get_archive_summary(self) -> dict:
        """Aggregate summary of all archive states."""

        novelty = self.novelty_archive

        qd = self.qd_archive

        sa = self.strategy_archive

        return {
            "novelty": {
                "size": novelty.size if novelty else 0,
                "threshold": novelty.novelty_threshold if novelty else 0,
                "diversity": novelty.get_diversity_score() if novelty else 0.0,
            },
            "qd": {
                "coverage": qd.coverage() if qd else 0.0,
                "size": qd.size if qd else 0,
                "cells": qd.num_bins * qd.num_bins if qd else 0,
                "fitness_dist": qd.get_fitness_distribution() if qd else {},
            },
            "strategy": {
                "total_records": sa.total_records() if sa else 0,
                "diversity": sa.get_strategy_diversity(self.champion_level) if sa else 0.0,
            },
            "hall_of_fame": len(self._hall_of_fame._heap),
        }

    def get_training_summary(self) -> dict:
        """Return training-relevant metrics for the retrain pipeline."""

        return {
            "recurrent_examples": self.recurrent_examples_count,
            "champion_fitness": self.champion_fitness,
            "champion_level": self.champion_level,
            "generations_stalled": self.generations_stalled,
            "fitness_velocity": getattr(self, "_fitness_velocity", 0.0),
            "champion_verified": getattr(self, "_champion_verified", False),
            "champion_verify_score": getattr(self, "_champion_verify_score", 0.0),
            "strategy_diversity": (
                self.strategy_archive.get_strategy_diversity(self.champion_level) if self.strategy_archive else 0.0
            ),
            "novelty_diversity": self.novelty_archive.get_diversity_score() if self.novelty_archive else 0.0,
            "qd_coverage": self.qd_archive.coverage() if self.qd_archive else 0.0,
        }

    def get_run_summary(self) -> dict:
        """Single consolidated summary of the entire run state."""

        return {
            "diagnostics": self.get_run_diagnostics(),
            "performance": self.get_performance_metrics(),
            "training": self.get_training_summary(),
            "archives": self.get_archive_summary(),
            "champion_summary": self.get_champion_summary(),
        }

    def get_evolution_state(self) -> dict:
        """Full evolution state for serialization/debugging."""

        return {
            "champion": self.get_champion_program_analysis(),
            "run_summary": self.get_run_summary(),
            "state": {
                "current_generation": self.current_generation,
                "champion_level": self.champion_level,
                "champion_fitness": self.champion_fitness,
                "generations_stalled": self.generations_stalled,
            },
            "config": {
                "max_generations": self.config.max_generations,
                "population_size": self.config.population_size,
                "elite_ratio": self.config.elite_ratio,
                "mutation_rate": self.config.mutation_rate,
                "auto_curriculum": self.config.auto_curriculum,
                "use_gpu": self.config.use_gpu,
            },
        }

    def get_comprehensive_state(self) -> dict:
        """All-in-one state snapshot for dashboards, export, and debugging."""

        return {
            "evolution": self.get_evolution_state(),
            "champion_program": self.get_champion_program_analysis(),
            "run_diagnostics": self.get_run_diagnostics(),
            "performance": self.get_performance_metrics(),
            "training": self.get_training_summary(),
            "archives": self.get_archive_summary(),
            "champion_summary": self.get_champion_summary(),
        }

    def log_comprehensive_diagnostics(self) -> None:
        """Log comprehensive diagnostics for debugging and analysis."""

        state = self.get_comprehensive_state()

        evo = state.get("evolution", {})

        evo_state = evo.get("state", {})

        self._log("=== Comprehensive Diagnostics ===")

        self._log(f"  Champion: {self.get_champion_summary()}")

        self._log(f'  Generation: {evo_state.get("current_generation", 0)}')

        self._log(f'  Fitness velocity: {evo_state.get("fitness_velocity", 0):.4f}')

        archives = state.get("archives", {})

        novelty = archives.get("novelty", {})

        qd = archives.get("qd", {})

        strategy = archives.get("strategy", {})

        self._log(f'  Archives: novelty={novelty.get("size", 0)}, ' f'qd_coverage={qd.get("coverage", 0.0):.2%}')

        self._log(f'  Strategy diversity: {strategy.get("diversity", 0.0):.3f}')

        perf = state.get("performance", {})

        self._log(
            f'  Performance: mut_rate={perf.get("mutation_rate", 0.0):.2f}, ' f'pop={perf.get("population_size", 0)}'
        )

        self._log("=== End Diagnostics ===")

    def log_diagnostics(self) -> None:
        """Log run diagnostics in a readable format."""

        diag = self.get_run_diagnostics()

        lines = [
            "=== Run Diagnostics ===",
            (
                f"  Champion: L{diag['champion_level']} fitness={diag['champion_fitness']:.2f} "
                f"words={diag['champion_words']} verified={diag['champion_verified']}"
            ),
            f"  Generation: {diag['current_generation']} stalled={diag['generations_stalled']}",
            (
                f"  Archives: novelty={diag['novelty_archive_size']} qd={diag['qd_archive_size']} "
                f"strategy={diag['strategy_archive_records']} hof={diag['hall_of_fame_size']}"
            ),
            f"  Strategy diversity: {diag['strategy_diversity']:.3f}",
            f"  Recurrent examples: {diag['recurrent_examples']}",
            f"  Fitness velocity: {diag['fitness_velocity']:.4f} momentum={diag['fitness_momentum']:.4f}",
            f"  Mutation rate: {diag['mutation_rate']:.3f} pop={diag['population_size']} elite={diag['elite_ratio']:.2f}",
        ]

        self._log("\n".join(lines))

    def export_state_for_analysis(self) -> dict:
        """Export full engine state for external analysis.



        Returns a serializable dict with all key metrics, archive contents,

        and configuration for use in Jupyter notebooks or dashboards.

        """

        state = {
            "timestamp": __import__("time").time(),
            "generation": self.current_generation,
            "champion": {
                "level": self.champion_level,
                "fitness": self.champion_fitness,
                "words": len(self.champion) if self.champion else 0,
                "program_hex": [hex(w) for w in (self.champion or [])],
                "verified": getattr(self, "_champion_verified", False),
            },
            "config": {
                "population_size": self.config.population_size,
                "elite_ratio": self.config.elite_ratio,
                "mutation_rate": self.config.mutation_rate,
                "novelty_weight": self.config.novelty_weight,
            },
            "archives": {
                "novelty_size": self.novelty_archive.size if self.novelty_archive else 0,
                "qd_occupied": self.qd_archive.size if self.qd_archive else 0,
                "qd_coverage": self.qd_archive.coverage() if self.qd_archive else 0.0,
                "hof_size": len(self._hall_of_fame._heap),
            },
            "stall": {
                "generations_stalled": self.generations_stalled,
                "fitness_velocity": getattr(self, "_fitness_velocity", 0.0),
            },
            "diagnostics": self.get_run_diagnostics(),
        }

        return state

    def save_state_to_file(self, path: str) -> bool:
        """Save comprehensive state to a JSON file for analysis."""

        try:

            state = self.get_comprehensive_state()

            import json

            with open(path, "w") as f:

                json.dump(state, f, indent=2, default=str)

            return True

        except (RuntimeError, OSError) as e:

            self._log(f"[StateExport] Failed to save to {path}: {e}")

            return False

    def _close_log_handle(self) -> None:
        """Close the lazy log file handle if open.



        Called at the end of run() and on exception to prevent fd exhaustion

        across multi-cycle runs. Safe to call multiple times.

        Sets _log_handle_closed flag and clears path cache so _log won't

        reopen the file after intentional close.

        """

        if self._log_file is not None:

            try:

                self._log_file.close()

            except Exception:  # noqa: BLE001, S110

                pass

            self._log_file = None

        self._log_handle_closed = True

        self._log_path_cache = None  # prevent _log from reopening

    # ------------------------------------------------------------------

    # Load / restore

    # ------------------------------------------------------------------
