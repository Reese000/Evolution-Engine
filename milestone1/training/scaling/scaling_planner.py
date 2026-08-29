"""Scaling planner for self-scaling architecture.

Phase 1: Determines when and how to scale the model architecture
based on saturation detection reports.

Includes VRAM pre-flight checks using torch.cuda.mem_get_info() when available.
"""

from typing import Dict, Optional, Tuple

from .types import SaturationReport, ScalingDecision


class ScalingPlanner:
    """Plans architecture scaling decisions based on saturation reports.

    The SCALING_LADDER defines progressive model configurations:
        (stage, embed_dim, ff_dim, heads, params_approx, vram_budget)

    Stages must be scaled sequentially (never skip stages).
    VRAM availability is checked before recommending a target stage.
    """

    # (stage, embed_dim, ff_dim, num_heads, approx_params, vram_budget_gb)
    SCALING_LADDER = [
        (0, 8, 256, 8, "9M", 4.0),
        (1, 14, 512, 8, "35M", 8.5),
        (2, 20, 768, 12, "80M", 11.5),
        (3, 26, 1024, 16, "150M", 12.5),
    ]

    def __init__(self):
        """Initialize the scaling planner."""
        self._ladder = self.SCALING_LADDER

    def plan(self, current_stage: int, report: SaturationReport) -> ScalingDecision | None:
        """Plan a scaling decision based on current stage and saturation report.

        Args:
            current_stage: Current index into the scaling ladder.
            report: Saturation report from SaturationDetector.

        Returns:
            ScalingDecision if scaling should proceed, None otherwise.
        """
        # Only scale if all saturation criteria are met
        if not report.should_scale:
            return None

        # Determine next stage
        next_stage_entry = self.next_stage(current_stage)
        if next_stage_entry is None:
            return None  # Already at max stage

        target_stage, target_embed, target_ff, target_heads, target_params, target_vram = next_stage_entry
        current_entry = self._get_stage_entry(current_stage)

        if current_entry is None:
            return None

        _, current_embed, current_ff, current_heads, current_params, current_vram = current_entry

        # Build config dicts
        current_config = {
            "embed_dim": current_embed,
            "ff_dim": current_ff,
            "num_heads": current_heads,
            "params_approx": current_params,
            "vram_budget_gb": current_vram,
        }

        target_config = {
            "embed_dim": target_embed,
            "ff_dim": target_ff,
            "num_heads": target_heads,
            "params_approx": target_params,
            "vram_budget_gb": target_vram,
        }

        # Build reason
        criteria_met = report.criteria_met()
        reason = (
            f"Saturation detected ({criteria_met}/4 criteria, "
            f"confidence={report.confidence:.1%}). "
            f"Scaling from stage {current_stage} to {target_stage} "
            f"({current_params} -> {target_params} params)."
        )

        return ScalingDecision(
            current_stage=current_stage,
            target_stage=target_stage,
            current_config=current_config,
            target_config=target_config,
            reason=reason,
        )

    def next_stage(self, current_stage: int) -> tuple | None:
        """Get the next stage entry whose VRAM budget fits available GPU memory.

        Args:
            current_stage: Current index into the scaling ladder.

        Returns:
            Tuple of (stage, embed_dim, ff_dim, heads, params, vram) for the
            next valid stage, or None if at max or no suitable stage found.
        """
        if current_stage + 1 >= len(self._ladder):
            return None  # Already at max stage

        # Check available VRAM
        available_vram = self._get_available_vram_gb()

        # Find next stage that fits in available VRAM
        for i in range(current_stage + 1, len(self._ladder)):
            entry = self._ladder[i]
            _stage, _embed, _ff, _heads, _params, vram = entry

            if vram <= available_vram:
                return entry

        return None  # No stage fits in available VRAM

    def _get_stage_entry(self, stage: int) -> tuple | None:
        """Get a specific stage entry from the ladder.

        Args:
            stage: Index into the scaling ladder.

        Returns:
            Tuple of (stage, embed_dim, ff_dim, heads, params, vram) or None.
        """
        if 0 <= stage < len(self._ladder):
            return self._ladder[stage]
        return None

    @staticmethod
    def _get_available_vram_gb() -> float:
        """Get available GPU VRAM in GB.

        Uses torch.cuda.mem_get_info() if CUDA is available.
        Returns a very large number if CUDA is not available (no VRAM limit).

        Returns:
            Available VRAM in GB.
        """
        try:
            import torch

            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                return free / (1024**3)  # Convert bytes to GB
        except (ImportError, RuntimeError):
            pass

        # If no CUDA available, return a large number (no VRAM limit)
        return 1000.0

    def max_stage(self) -> int:
        """Return the maximum stage index in the scaling ladder.

        Returns:
            Index of the last stage in the ladder.
        """
        return len(self._ladder) - 1

    def stage_config(self, stage: int) -> dict | None:
        """Get the configuration dict for a specific stage.

        Args:
            stage: Index into the scaling ladder.

        Returns:
            Dict with embed_dim, ff_dim, num_heads, params_approx, vram_budget_gb
            or None if stage is invalid.
        """
        entry = self._get_stage_entry(stage)
        if entry is None:
            return None

        _, embed, ff, heads, params, vram = entry
        return {
            "embed_dim": embed,
            "ff_dim": ff,
            "num_heads": heads,
            "params_approx": params,
            "vram_budget_gb": vram,
        }
