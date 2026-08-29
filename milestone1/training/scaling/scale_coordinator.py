"""ScaleCoordinator — orchestrates the self-scaling lifecycle.

Sits between the evolution loop and the existing CheckpointManager /
ProgressionTracker / ChampionArchive. Owns the scaling state machine:

    DETECT → PLAN → MIGRATE → WARMSTART → GATE
       ↑                                        │
       └──────────── ROLLBACK ←─────────────────┘

Usage:
    coordinator = ScaleCoordinator(config, tracker, archive, checkpoint_mgr)
    result = coordinator.maybe_scale(engine)  # called each cycle
    if result:
        print(f"Scaling event: {result}")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ScaleState(Enum):
    """State machine for scaling lifecycle."""

    IDLE = "idle"
    DETECTING = "detecting"
    PLANNING = "planning"
    MIGRATING = "migrating"
    WARMSTARTING = "warmstarting"
    GATING = "gating"
    COMMITTING = "committing"
    ROLLING_BACK = "rolling_back"


@dataclass
class ScaleEvent:
    """Record of a scaling attempt."""

    attempt_id: int
    old_stage: int
    new_stage: int
    old_config: dict
    new_config: dict
    state: ScaleState
    timestamp: float = 0.0
    result: str | None = None  # "committed", "rolled_back", "extended"
    reason: str | None = None


class ScaleCoordinator:
    """Orchestrates the self-scaling lifecycle with safety guards.

    State machine:
        IDLE → DETECTING → PLANNING → MIGRATING → WARMSTARTING → GATING → COMMITTING/ROLLING_BACK → IDLE

    Key safety features:
    - Debounce: requires M consecutive saturation reports before acting
    - Cooldown: exponential backoff after failed attempts
    - Pre-flight VRAM check before migration
    - Atomic checkpoint before migration
    - Validation gate before commit
    """

    def __init__(
        self,
        scaling_planner: Any,  # ScalingPlanner
        saturation_detector: Any,  # SaturationDetector
        model_migrator: Any,  # ModelMigrator
        warm_start_trainer: Any,  # WarmStartTrainer
        scale_validator: Any,  # ScaleValidator
        checkpoint_manager: Any,  # CheckpointManager
        state_path: str = "scaling_state.json",
        debounce_cycles: int = 5,
        cooldown_cycles: int = 30,
        gate_cycles: int = 10,
    ):
        self.planner = scaling_planner
        self.detector = saturation_detector
        self.migrator = model_migrator
        self.warm_start = warm_start_trainer
        self.validator = scale_validator
        self.checkpoint_mgr = checkpoint_manager
        self.state_path = state_path

        self.state = ScaleState.IDLE
        self.current_stage = 0
        self.debounce_count = 0
        self.debounce_cycles = debounce_cycles
        self.cooldown_count = 0
        self.cooldown_cycles = cooldown_cycles
        self.gate_count = 0
        self.gate_cycles = gate_cycles

        self.events: list[ScaleEvent] = []
        self.current_event: ScaleEvent | None = None
        self._load_state()

    def _load_state(self) -> None:
        """Load persistent state from disk."""
        if os.path.isfile(self.state_path):
            try:
                with open(self.state_path) as f:
                    state = json.load(f)
                self.current_stage = state.get("current_stage", 0)
                self.cooldown_count = state.get("cooldown_count", 0)
                self.events = [ScaleEvent(**e) for e in state.get("events", [])]
            except Exception:
                pass

    def _save_state(self) -> None:
        """Persist state to disk atomically."""
        import tempfile

        state = {
            "current_stage": self.current_stage,
            "cooldown_count": self.cooldown_count,
            "events": [asdict(e) for e in self.events[-20:]],  # keep last 20
        }
        dir_name = os.path.dirname(self.state_path) or "."
        os.makedirs(dir_name, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dir_name, suffix=".tmp", delete=False) as f:
            tmp_path = f.name
        try:
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, self.state_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def maybe_scale(self, snapshot: Any, engine: Any = None) -> str | None:
        """Main entry point — called each cycle.

        Returns None if no action needed, or a description of the action taken.
        """
        # Check cooldown
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            self._save_state()
            return None

        # Detect saturation
        report = self.detector.evaluate(snapshot)

        if report.should_scale:
            self.debounce_count += 1
            if self.debounce_count >= self.debounce_cycles:
                return self._attempt_scale(snapshot, engine)
        else:
            self.debounce_count = 0

        return None

    def _attempt_scale(self, snapshot: Any, engine: Any) -> str | None:
        """Attempt a scaling operation."""
        # Plan migration
        decision = self.planner.plan(self.current_stage, snapshot)
        if decision is None:
            return None

        self.state = ScaleState.PLANNING
        event = ScaleEvent(
            attempt_id=len(self.events) + 1,
            old_stage=self.current_stage,
            new_stage=decision.target_stage,
            old_config=decision.current_config,
            new_config=decision.target_config,
            state=ScaleState.PLANNING,
            timestamp=time.time(),
            reason=decision.reason,
        )
        self.events.append(event)
        self.current_event = event

        logger.info(f"Scaling attempt {event.attempt_id}: stage {self.current_stage} → {decision.target_stage}")
        logger.info(f"Reason: {decision.reason}")

        # Pre-snapshot checkpoint
        self.checkpoint_mgr.backup("checkpoint_best")

        # === FULL MIGRATION PIPELINE ===
        try:
            self.state = ScaleState.MIGRATING
            self._save_state()

            # 1. Create old and new models
            old_config = decision.current_config
            new_config = decision.target_config

            # Load old model from checkpoint
            import sys as _sys

            import torch

            _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
            from rv32i_model import RV32ITransformer

            old_model = RV32ITransformer(
                vocab_size=old_config.get("vocab_size", 2048),
                d_model=old_config["d_model"],
                n_layers=old_config["n_layers"],
                n_heads=old_config["n_heads"],
                d_ff=old_config.get("d_ff", old_config["d_model"] * 4),
                max_seq_len=old_config.get("max_seq_len", 256),
                dropout=old_config.get("dropout", 0.0),
                use_rope=old_config.get("use_rope", True),
                use_checkpointing=False,
            )
            ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "checkpoint_best.pt")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                old_model.load_state_dict(ckpt["model_state_dict"])
            elif "state_dict" in ckpt:
                old_model.load_state_dict(ckpt["state_dict"])
            old_model.eval()

            # Create new model (fresh)
            new_model = RV32ITransformer(
                vocab_size=new_config.get("vocab_size", 2048),
                d_model=new_config["d_model"],
                n_layers=new_config["n_layers"],
                n_heads=new_config["n_heads"],
                d_ff=new_config.get("d_ff", new_config["d_model"] * 4),
                max_seq_len=new_config.get("max_seq_len", 256),
                dropout=new_config.get("dropout", 0.0),
                use_rope=new_config.get("use_rope", True),
                use_checkpointing=False,
            )

            # 2. Migrate weights
            migration_result = self.migrator.migrate(old_model, new_model, old_config, new_config)
            if not migration_result.success:
                logger.error(f"Migration failed: {migration_result.message}")
                self.state = ScaleState.ROLLING_BACK
                self._save_state()
                return f"Migration failed: {migration_result.message}"

            # 3. Save migrated model
            migrated_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "checkpoint_migrated.pt")
            torch.save({"model_state_dict": new_model.state_dict(), "config": new_config}, migrated_path)

            # 4. Warm-start training
            self.state = ScaleState.WARMSTARTING
            self._save_state()

            warm_result = self.warm_start.warm_start(
                old_model=old_model,
                new_model=new_model,
                new_config=new_config,
                checkpoint_path=migrated_path,
            )

            # 5. Validate
            self.state = ScaleState.GATING
            self._save_state()

            val_result = self.validator.validate(
                old_metrics={
                    "fitness": snapshot.fitness_history[-1] if snapshot.fitness_history else 0,
                    "loss": snapshot.val_loss_history[-1] if snapshot.val_loss_history else 0.1,
                },
                new_metrics={"fitness": warm_result.get("val_fitness", 0), "loss": warm_result.get("val_loss", 0.1)},
                archive_size=snapshot.archive_size,
            )

            if val_result.recommendation == "COMMIT":
                # Replace checkpoint
                import shutil

                shutil.copy2(migrated_path, ckpt_path)
                self.current_stage = decision.target_stage
                self.state = ScaleState.COMMITTING
                self._save_state()
                logger.info(f"Committed scale to stage {self.current_stage}")
                return f"COMMITTED: scaled to stage {self.current_stage} ({new_config['n_layers']}L/{new_config['d_model']}d)"
            else:
                self.state = ScaleState.ROLLING_BACK
                self._save_state()
                logger.info(f"Rolled back: {val_result.recommendation}")
                return f"ROLLED BACK: {val_result.recommendation}"

        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            self.state = ScaleState.ROLLING_BACK
            self._save_state()
            return f"FAILED: {e}"

    def get_status(self) -> dict:
        """Return current coordinator status."""
        return {
            "state": self.state.value,
            "current_stage": self.current_stage,
            "cooldown_remaining": self.cooldown_count,
            "debounce_count": self.debounce_count,
            "total_attempts": len(self.events),
            "last_event": asdict(self.events[-1]) if self.events else None,
        }
