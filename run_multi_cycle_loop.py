"""Per-cycle evolution/retrain loop for run_multi_cycle."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any

from logging_config import get_logger

_logger_loop = get_logger("RunMultiCycle")


def log(msg: str) -> None:
    _logger_loop.info(msg)


@dataclass
class CycleTuningBundle:
    evo_meta_params: dict[str, Any]
    retrain_meta_params: dict[str, Any]
    archive_params: dict[str, Any]
    gen_count_override: int


@dataclass
class EvolutionPhaseResult:
    evo_result: dict[str, Any]
    evo_elapsed: float
    prev_archive_state: dict[str, Any] | None
    prev_cycle_metrics: dict[str, Any]
    skip_evolution: bool


def _build_merged_retrain_config(retrain_config: Any, retrain_meta_params: dict[str, Any]) -> Any:
    from run_multi_cycle import _HAS_META_CONTROLLER

    if not retrain_meta_params or not _HAS_META_CONTROLLER:
        return retrain_config
    import types

    mc_cfg = types.SimpleNamespace()
    mc_cfg.retrain_lr = retrain_meta_params.get("retrain_lr")
    mc_cfg.retrain_freeze_layers = retrain_meta_params.get("retrain_freeze_layers", 0)
    mc_cfg.retrain_max_per_level = retrain_meta_params.get("retrain_max_per_level")
    mc_cfg.retrain_patience = retrain_meta_params.get("retrain_patience")
    mc_cfg.retrain_max_epochs = retrain_meta_params.get("retrain_max_epochs")
    mc_cfg.retrain_model_size = retrain_meta_params.get("retrain_model_size", "medium")
    mc_cfg.retrain_timeout = 3600
    mc_cfg.retrain_min_examples = 1
    mc_cfg.retrain_val_split = retrain_meta_params.get("retrain_val_split")
    mc_cfg.retrain_grad_clip = None
    mc_cfg.retrain_batch_size = retrain_meta_params.get("retrain_batch_size")
    mc_cfg.retrain_weight_decay = None
    mc_cfg.retrain_adaptive_freeze = False
    if retrain_config is not None:
        if getattr(retrain_config, "retrain_adaptive_freeze", False):
            mc_cfg.retrain_adaptive_freeze = True
        for attr in (
            "retrain_lr",
            "retrain_freeze_layers",
            "retrain_max_per_level",
            "retrain_patience",
            "retrain_max_epochs",
            "retrain_grad_clip",
            "retrain_batch_size",
            "retrain_weight_decay",
            "retrain_adaptive_freeze",
            "retrain_swa_start",
            "retrain_grad_accum_steps",
            "retrain_lr_scheduler",
            "no_amp",
        ):
            cli_val = getattr(retrain_config, attr, None)
            if cli_val is not None:
                setattr(mc_cfg, attr, cli_val)
    return mc_cfg


def _sustained_level_cap(project_root: str, all_results: list[dict[str, Any]]) -> int:
    from evolution.progress_controller import ProgressController
    from run_multi_cycle import RESULTS_FILE, _is_valid_cycle_result

    valid = [r for r in all_results if _is_valid_cycle_result(r)]
    if len(valid) < 10 and os.path.isfile(RESULTS_FILE):
        import json

        tail: list[dict[str, Any]] = []
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _is_valid_cycle_result(row):
                    tail.append(row)
        valid = (tail[-80:] + valid)[-80:]
    return ProgressController.sustained_resume_level(valid)


def _carry_forward_champion_level(
    evo_result: dict[str, Any],
    all_results: list[dict[str, Any]],
    project_root: str,
) -> int:
    from evolution.progress_controller import ProgressController

    final_level = int(evo_result.get("final_level", 0))
    verify = float(evo_result.get("champion_verify_score", 0.0))
    sustained = _sustained_level_cap(project_root, all_results)
    carried = ProgressController.carry_forward_start_level(final_level, verify, sustained)
    if float(evo_result.get("final_fitness", 0)) <= -25.0:
        prev = carried
        carried = max(3, carried - 1)
        if carried < prev:
            log(f"  Timeout champion clamp start L{prev}->L{carried} (fit=-25)")
    if carried < final_level:
        log(f"  Hollow level clamp L{final_level}->L{carried} " f"(verify={verify:.3f}, sustained_cap=L{sustained})")
        evo_result["start_level_clamped"] = True
        evo_result["start_level"] = carried
        is_timeout = float(evo_result.get("final_fitness", 0)) <= -25.0
        if is_timeout:
            evo_result["force_reference_champion"] = True
            try:
                tick_path = os.path.join(project_root, ".agents", "tick_state.json")
                tick: dict[str, Any] = {}
                if os.path.isfile(tick_path):
                    with open(tick_path, encoding="utf-8") as f:
                        tick = json.load(f)
                tick["force_champion_reseed"] = True
                tick["force_reference_champion"] = True
                tick["reseed_level"] = carried
                with open(tick_path, "w", encoding="utf-8") as f:
                    json.dump(tick, f, indent=2)
                os.environ["EE_FORCE_REFERENCE_CHAMPION"] = "1"
                os.environ["EE_REGISTER_REPAIR_BOOST"] = "1"
            except (OSError, json.JSONDecodeError, TypeError):
                pass
    else:
        evo_result["start_level_clamped"] = False
        evo_result["start_level"] = final_level
        os.environ.pop("EE_CLAMPED_START_LEVEL", None)
        os.environ.pop("EE_CLAMPED_MAX_ADVANCE", None)
    if carried < final_level or float(evo_result.get("final_fitness", 0)) <= -25.0:
        os.environ["EE_CLAMPED_START_LEVEL"] = str(carried)
        os.environ.setdefault("EE_CLAMPED_MAX_ADVANCE", "1")
    return carried


def _flag_reference_reseed(project_root: str, level: int) -> None:
    """Request reference champion at *level* for the next cycle."""
    tick_path = os.path.join(project_root, ".agents", "tick_state.json")
    try:
        tick: dict[str, Any] = {}
        if os.path.isfile(tick_path):
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
        tick["force_champion_reseed"] = True
        tick["force_reference_champion"] = True
        tick["reseed_level"] = int(level)
        tick.pop("force_register_repair", None)
        with open(tick_path, "w", encoding="utf-8") as f:
            json.dump(tick, f, indent=2)
        os.environ["EE_FORCE_REFERENCE_CHAMPION"] = "1"
        os.environ.pop("EE_REGISTER_REPAIR_BOOST", None)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass


def _flag_nudge_reference_reseed(project_root: str, nudge_level: int) -> None:
    """Request reference champion at nudged level so next cycle starts clean."""
    _flag_reference_reseed(project_root, nudge_level)


def _flag_nudge_start_level(project_root: str, nudge_level: int) -> None:
    """Bump next-cycle curriculum start without grafting reference champion."""
    tick_path = os.path.join(project_root, ".agents", "tick_state.json")
    try:
        tick: dict[str, Any] = {}
        if os.path.isfile(tick_path):
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
        tick["depth_nudge_level"] = int(nudge_level)
        with open(tick_path, "w", encoding="utf-8") as f:
            json.dump(tick, f, indent=2)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass


def _collapse_reseed_level(project_root: str, level: int, verify: float) -> int:
    """Pick recovery start level — never reference-reseed broken L22+ payloads."""
    from evolution.progress_controller import ProgressController

    lvl = int(level)
    ver = float(verify)
    if ProgressController.passes_certify_verify(lvl, ver):
        return lvl
    if lvl >= 22:
        return 21
    if lvl == 21:
        return 21
    return max(17, lvl - 1)


def _flag_collapse_recovery(project_root: str, reseed_level: int, *, corrupt_cycle: int | None = None) -> None:
    """Recover from collapse without grafting broken high-level reference programs."""
    tick_path = os.path.join(project_root, ".agents", "tick_state.json")
    try:
        tick: dict[str, Any] = {}
        if os.path.isfile(tick_path):
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
        tick.pop("force_reference_champion", None)
        tick.pop("force_champion_reseed", None)
        tick["collapse_reseed"] = True
        tick["reseed_level"] = int(reseed_level)
        tick["depth_nudge_level"] = int(reseed_level)
        tick.pop("force_register_repair", None)
        tick.pop("epi_plateau_boost", None)
        tick.pop("epi_momentum_intervention", None)
        tick.pop("epi_momentum_reseed", None)
        if int(reseed_level) >= 21:
            from evolution.champion_rollback import find_last_clean_champion

            results_file = os.path.join(project_root, "multi_cycle_results.jsonl")
            clean = find_last_clean_champion(
                results_file,
                int(reseed_level),
                before_cycle=corrupt_cycle,
            )
            if clean and clean.get("champion_program"):
                tick["champion_rollback_program"] = clean["champion_program"]
                tick["champion_rollback_cycle"] = int(clean["cycle"])
        with open(tick_path, "w", encoding="utf-8") as f:
            json.dump(tick, f, indent=2)
        os.environ.pop("EE_FORCE_REFERENCE_CHAMPION", None)
        os.environ.pop("EE_REGISTER_REPAIR_BOOST", None)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass


def _prune_stale_tick_reseed_state(project_root: str, active_level: int) -> None:
    """Drop orphan rollback/reseed payloads that would drag depth backward."""
    tick_path = os.path.join(project_root, ".agents", "tick_state.json")
    try:
        if not os.path.isfile(tick_path):
            return
        with open(tick_path, encoding="utf-8") as f:
            tick = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    changed = False
    if not tick.get("collapse_reseed") and not tick.get("apply_champion_rollback"):
        for key in ("champion_rollback_program", "champion_rollback_cycle"):
            if key in tick:
                tick.pop(key, None)
                changed = True
    if tick.get("force_reference_champion") and not tick.get("force_champion_reseed"):
        tick.pop("force_reference_champion", None)
        changed = True
    if (
        tick.get("reseed_level") is not None
        and not tick.get("collapse_reseed")
        and not tick.get("force_champion_reseed")
        and int(tick["reseed_level"]) < int(active_level)
    ):
        tick.pop("reseed_level", None)
        changed = True
    if changed:
        try:
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump(tick, f, indent=2)
        except OSError:
            pass


def _maybe_apply_collapse_recovery(project_root: str, cycle: int, last_champion_level: int) -> int:
    """Drop start level after collapse without reference graft."""
    tick_path = os.path.join(project_root, ".agents", "tick_state.json")
    try:
        if not os.path.isfile(tick_path):
            return last_champion_level
        with open(tick_path, encoding="utf-8") as f:
            tick = json.load(f)
        if not tick.get("collapse_reseed"):
            return last_champion_level
        recovery = int(tick.get("reseed_level", last_champion_level))
        rollback = tick.get("champion_rollback_program")
        if recovery >= last_champion_level:
            if rollback:
                tick.pop("collapse_reseed", None)
                tick.pop("depth_nudge_level", None)
                tick["apply_champion_rollback"] = True
                with open(tick_path, "w", encoding="utf-8") as f:
                    json.dump(tick, f, indent=2)
                log(
                    f"  Cycle {cycle}: Same-level collapse rollback queued "
                    f"(L{recovery}, from C{tick.get('champion_rollback_cycle', '?')})"
                )
                return last_champion_level
            tick.pop("collapse_reseed", None)
            tick.pop("depth_nudge_level", None)
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump(tick, f, indent=2)
            return last_champion_level
        log(f"  Cycle {cycle}: Collapse recovery drop " f"L{last_champion_level}->L{recovery}")
        tick.pop("collapse_reseed", None)
        tick.pop("depth_nudge_level", None)
        with open(tick_path, "w", encoding="utf-8") as f:
            json.dump(tick, f, indent=2)
        state_path = os.path.join(project_root, ".agents", "progress_controller_state.json")
        if os.path.isfile(state_path):
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            state["level"] = recovery
            state["cycles_at_level"] = 0
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        return recovery
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return last_champion_level


def _clear_stale_collapse_tick_flags(project_root: str, evo_result: dict[str, Any]) -> None:
    """Drop rollback/reseed leftovers once L22+ champion is stable."""
    from evolution.curriculum_advance import depth_carry_verify_floor

    level = int(evo_result.get("final_level", 0))
    verify = float(evo_result.get("champion_verify_score", 0.0))
    fitness = float(evo_result.get("final_fitness", 0.0))
    if level < 22 or verify < depth_carry_verify_floor(level):
        return
    if level >= 30 and verify >= 0.99 - 1e-6 and fitness < 150.0:
        return
    tick_path = os.path.join(project_root, ".agents", "tick_state.json")
    try:
        if not os.path.isfile(tick_path):
            return
        with open(tick_path, encoding="utf-8") as f:
            tick = json.load(f)
        if not any(
            tick.get(k)
            for k in (
                "champion_rollback_program",
                "champion_rollback_cycle",
                "collapse_reseed",
                "apply_champion_rollback",
            )
        ):
            return
        tick.pop("champion_rollback_program", None)
        tick.pop("champion_rollback_cycle", None)
        tick.pop("collapse_reseed", None)
        tick.pop("apply_champion_rollback", None)
        if int(tick.get("reseed_level", 0)) < level:
            tick.pop("reseed_level", None)
        with open(tick_path, "w", encoding="utf-8") as f:
            json.dump(tick, f, indent=2)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass


def _auto_depth_escape_start_level(project_root: str, cycle: int, last_champion_level: int) -> int:
    """Bump start level when progress state shows stable depth without tick nudge."""
    from evolution.curriculum_advance import nudge_fitness_passes, nudge_verify_passes

    state_path = os.path.join(project_root, ".agents", "progress_controller_state.json")
    try:
        if not os.path.isfile(state_path):
            return last_champion_level
        with open(state_path, encoding="utf-8") as f:
            st = json.load(f)
        pc_level = int(st.get("level", last_champion_level))
        pc_cycles = int(st.get("cycles_at_level", 0))
        pc_verify = float(st.get("verify_score", 0.0))
        pc_fit = float(st.get("raw_fitness", 0.0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return last_champion_level

    min_depth = int(os.environ.get("EE_DEPTH_ESCAPE_CYCLES", "8"))
    max_level = int(os.environ.get("EE_MAX_LEVEL", "30"))
    if (
        pc_level >= 22
        and pc_level == last_champion_level
        and pc_cycles >= min_depth
        and pc_level + 1 <= max_level
        and nudge_verify_passes(pc_level, pc_verify, pc_cycles)
        and nudge_fitness_passes(pc_level, pc_fit, pc_verify)
    ):
        next_level = pc_level + 1
        log(
            f"  Cycle {cycle}: Auto depth escape L{pc_level}->L{next_level} "
            f"(cycles={pc_cycles} verify={pc_verify:.3f})"
        )
        return next_level
    return last_champion_level


def _maybe_flag_verify_regression_rollback(
    project_root: str,
    evo_result: dict[str, Any],
    cycle: int,
    prev_metrics: dict[str, Any] | None,
) -> None:
    """Auto-rollback when verify drops cycle-over-cycle at L21+."""
    if not prev_metrics:
        return
    level = int(evo_result.get("final_level", 0))
    if level < 21:
        return
    cur_verify = float(evo_result.get("champion_verify_score", 0.0))
    prev_verify = float(prev_metrics.get("champion_verify_score", 0.0))
    prev_prog = prev_metrics.get("champion_program")
    prev_cycle = int(prev_metrics.get("cycle", cycle - 1))
    if not isinstance(prev_prog, list) or not prev_prog:
        return
    from evolution.champion_rollback import flag_verify_regression_rollback

    if flag_verify_regression_rollback(
        project_root,
        level=level,
        prev_verify=prev_verify,
        prev_program=prev_prog,
        prev_cycle=prev_cycle,
        current_verify=cur_verify,
    ):
        log(
            f"  Cycle {cycle}: Verify regression rollback queued "
            f"({prev_verify:.3f}->{cur_verify:.3f}, from C{prev_cycle})"
        )


def _maybe_flag_collapse_reseed(project_root: str, evo_result: dict[str, Any], cycle: int) -> None:
    """Reseed when high-level champion collapses (verify, timeout, or graft bloat)."""
    from evolution.progress_controller import ProgressController

    level = int(evo_result.get("final_level", 0))
    verify = float(evo_result.get("champion_verify_score", 0.0))
    fitness = float(evo_result.get("final_fitness", 0.0))
    words = int(evo_result.get("champion_words", 0))
    if level < 17:
        return

    reason: str | None = None
    if fitness <= -25.0:
        reason = f"fitness={fitness:.1f} timeout"
    elif words > (36 if level >= 21 else 38 if level >= 20 else 40):
        reason = f"bloat={words}w"
    elif level == 21 and verify < 0.72:
        reason = f"verify={verify:.3f} L21 graft band"
    elif level >= 22 and verify < 0.15:
        reason = f"verify={verify:.3f} hollow L{level}"
    elif level == 20 and verify < 0.70:
        reason = f"verify={verify:.3f} graft-corruption band"
    elif verify < 0.55:
        reason = f"verify={verify:.3f}"

    if reason is None:
        return
    reseed_lvl = _collapse_reseed_level(project_root, level, verify)
    _flag_collapse_recovery(project_root, reseed_lvl, corrupt_cycle=cycle)
    log(f"  Cycle {cycle}: Collapse recovery flagged L{reseed_lvl} ({reason})")


def _maybe_flag_clean_depth_escape(project_root: str, evo_result: dict[str, Any], cycle: int) -> int | None:
    """Nudge to next level when L21+ champion is clean — escape graft trap before corruption."""
    from evolution.curriculum_advance import nudge_fitness_passes, nudge_verify_passes
    from evolution.progress_controller import ProgressController

    level = int(evo_result.get("final_level", 0))
    verify = float(evo_result.get("champion_verify_score", 0.0))
    words = int(evo_result.get("champion_words", 0))
    fitness = float(evo_result.get("final_fitness", 0.0))
    if level < 21:
        return None
    from evolution.curriculum_advance import depth_carry_verify_floor

    min_escape_verify = depth_carry_verify_floor(level) if level >= 21 else 0.72
    if verify < min_escape_verify:
        return None
    from evolution.champion_bloat_guard import level_word_cap

    cap = level_word_cap(level) or 36
    if words > cap:
        return None
    state_path = os.path.join(project_root, ".agents", "progress_controller_state.json")
    cycles_at_level = 1
    try:
        if os.path.isfile(state_path):
            with open(state_path, encoding="utf-8") as f:
                cycles_at_level = max(1, int(json.load(f).get("cycles_at_level", 1)))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    if not ProgressController.passes_certify_verify(level, verify) and not nudge_verify_passes(
        level, verify, cycles_at_level
    ):
        return None
    if not nudge_fitness_passes(level, fitness, verify):
        return None
    min_cycles = int(os.environ.get("EE_DEPTH_ESCAPE_CYCLES", "8"))
    if cycles_at_level < min_cycles:
        return None
    max_level = int(os.environ.get("EE_MAX_LEVEL", "30"))
    if level >= max_level:
        return None
    next_level = level + 1
    from evolution.register_repair import reference_passes_nudge_gate

    if reference_passes_nudge_gate(next_level):
        _flag_nudge_reference_reseed(project_root, next_level)
    else:
        _flag_nudge_start_level(project_root, next_level)
    log(f"  Cycle {cycle}: Clean L{level} depth escape nudge -> L{next_level}")
    return next_level


def _maybe_nudge_champion_level(
    cycle: int,
    project_root: str,
    evo_result: dict[str, Any],
    last_champion_level: int,
    progress_controller: Any | None = None,
) -> int:
    if progress_controller is not None:
        from evolution.progress_controller import ProgressAction

        plan = progress_controller.on_cycle_end(
            int(evo_result.get("final_level", last_champion_level)),
            float(evo_result.get("champion_verify_score", 0.0)),
            float(evo_result.get("final_fitness", 0.0)),
        )
        for line in plan.log_lines:
            log(line)
        if ProgressAction.NUDGE_START_LEVEL in plan.actions and plan.nudge_start_level is not None:
            from evolution.register_repair import reference_passes_nudge_gate

            if reference_passes_nudge_gate(plan.nudge_start_level):
                _flag_nudge_reference_reseed(project_root, plan.nudge_start_level)
            else:
                _flag_nudge_start_level(project_root, plan.nudge_start_level)
            log(f"  Cycle {cycle}: Nudge reseed flagged for L{plan.nudge_start_level}")
            return plan.nudge_start_level
        return last_champion_level

    from evolution.level_stall_tracker import update_and_maybe_nudge_start_level

    nudge = update_and_maybe_nudge_start_level(
        project_root,
        int(evo_result.get("final_level", last_champion_level)),
        float(evo_result.get("champion_verify_score", 0.0)),
        float(evo_result.get("final_fitness", 0.0)),
        stall_cycles_before_nudge=3,
    )
    if nudge is not None and nudge > last_champion_level:
        _flag_nudge_reference_reseed(project_root, nudge)
        log(f"  Cycle {cycle}: Cross-cycle anti-plateau nudge L{last_champion_level}->L{nudge}")
        return nudge
    return last_champion_level


def _cap_terminal_archive_state(
    archive_state: dict[str, Any] | None,
    evo_result: dict[str, Any],
) -> dict[str, Any] | None:
    """Bound strategy archive size at certified L30 so cycles stay fast."""
    if not archive_state:
        return archive_state
    from evolution.archive_persistence import cap_strategy_archive_state
    from evolution.terminal_stagnation import terminal_certified_plateau

    level = int(evo_result.get("final_level", 0))
    verify = float(evo_result.get("champion_verify_score", 0.0))
    fitness = float(evo_result.get("final_fitness", 0.0))
    if not terminal_certified_plateau(level, verify, fitness):
        return archive_state
    cap = int(os.environ.get("EE_TERMINAL_STRATEGY_ARCHIVE_CAP", "3200"))
    return cap_strategy_archive_state(archive_state, cap)


def _evaluate_retrain_skip(
    cycle: int,
    args: Any,
    evo_result: dict[str, Any],
    meta_controller: Any,
    checkpoint: str,
    last_champion_fitness: float,
    last_champion_level: int,
    last_champion_words: int,
    cycles_since_last_retrain: int,
    last_retrain_epochs: int,
    retrain_meta_params: dict[str, Any],
) -> tuple[bool, dict[str, Any], int]:
    current_fitness = evo_result.get("final_fitness", 0.0)
    current_level = evo_result.get("final_level", 0)
    current_words = evo_result.get("champion_words", 0)
    champion_verify = float(evo_result.get("champion_verify_score", 0.0))
    from evolution.terminal_stagnation import terminal_certified_plateau

    terminal_plateau = terminal_certified_plateau(int(current_level), champion_verify, float(current_fitness))
    should_retrain = True
    retrain_result: dict[str, Any] = {"retrained": False, "reason": "skipped"}
    force_l17_retrain = False

    if current_level >= 17:
        from evolution.progress_controller import ProgressController

        if not ProgressController.passes_certify_verify(int(current_level), champion_verify):
            from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor

            scan_floor_regression = int(current_level) >= 30 and terminal_efficiency_at_scan_floor(
                max(int(current_words), int(last_champion_words))
            )
            if scan_floor_regression:
                should_retrain = False
                retrain_result = {
                    "retrained": False,
                    "reason": "l30_verify_regression_skip_retrain",
                    "checkpoint": checkpoint,
                }
                log(
                    f"  Cycle {cycle}: Skip retrain — L{current_level} verify "
                    f"{champion_verify:.3f} below certify bar at scan floor "
                    f"(bytecode regression; RV retrain cannot fix)"
                )
            else:
                force_l17_retrain = True
                should_retrain = True
                log(
                    f"  Cycle {cycle}: FORCE RETRAIN — L{current_level} verify "
                    f"{champion_verify:.3f} below certify bar"
                )

    if meta_controller is not None and getattr(meta_controller, "force_retrain", False):
        meta_controller.force_retrain = False
        if terminal_plateau:
            meta_controller._consecutive_none_val_loss = 0
        else:
            log(f"  Cycle {cycle}: FORCE RETRAIN — MetaController detected degenerate loop")

    if args.skip_retrain_if_no_new_data and cycle >= 2:
        from run_multi_cycle import M1_RECURRENT_DATA, PROJECT_ROOT

        _recurrent_path = M1_RECURRENT_DATA
        _last_retrain_mtime_file = os.path.join(PROJECT_ROOT, ".last_retrain_mtime")
        if os.path.isfile(_recurrent_path) and os.path.isfile(_last_retrain_mtime_file):
            _current_mtime = os.path.getmtime(_recurrent_path)
            with open(_last_retrain_mtime_file) as _f:
                _saved_mtime = float(_f.read().strip())
            if _current_mtime <= _saved_mtime:
                should_retrain = False
                retrain_result = {
                    "retrained": False,
                    "reason": "no_new_data_since_last_retrain",
                    "checkpoint": checkpoint,
                }
                log(
                    f"  Cycle {cycle}: Skipping retrain -- recurrent_data.jsonl unchanged "
                    f"(current={_current_mtime:.0f}, last_retrain={_saved_mtime:.0f})"
                )

    min_cycles_between_retrains = 1 if terminal_plateau else 2
    cycles_since_last_retrain += 1
    force_interval = int(getattr(args, "force_retrain_interval", 0) or 0)
    # Default force_retrain_interval=3 must not override certified L30 plateau skip (40-cycle periodic only).
    if not terminal_plateau and force_interval > 0 and cycles_since_last_retrain >= force_interval:
        log(f"  Cycle {cycle}: Force retrain after {cycles_since_last_retrain} " f"cycles (interval={force_interval})")
        should_retrain = True
        cycles_since_last_retrain = 0

    if should_retrain and cycle >= 3 and cycles_since_last_retrain >= min_cycles_between_retrains:
        fitness_threshold = max(abs(last_champion_fitness) * 0.05, 0.01)
        fitness_same = abs(current_fitness - last_champion_fitness) < fitness_threshold
        level_same = current_level == last_champion_level
        words_same = current_words == last_champion_words
        no_new_traces = evo_result.get("champion_traces", 0) == 0
        champion_static = (
            fitness_same
            and words_same
            and (level_same or (terminal_plateau and int(current_level) >= int(os.environ.get("EE_MAX_LEVEL", "30"))))
        )
        # Certified L30: skip retrain even when fitness ticks up. Tiny +0.3 gains
        # were disabling skip and forcing a ~240s retrain every cycle.
        skip_terminal = terminal_plateau
        from evolution.terminal_stagnation import terminal_retrain_interval

        terminal_interval = terminal_retrain_interval()
        periodic_interval = terminal_interval if skip_terminal else force_interval
        if skip_terminal and periodic_interval > 0 and cycles_since_last_retrain >= periodic_interval:
            log(
                f"  Cycle {cycle}: Periodic terminal retrain after "
                f"{cycles_since_last_retrain} cycles (interval={periodic_interval})"
            )
            skip_terminal = False
            should_retrain = True
            retrain_result = {"retrained": False, "reason": "pending", "checkpoint": checkpoint}
        if not force_l17_retrain and (skip_terminal or (champion_static and no_new_traces)):
            reason = "terminal_plateau_unchanged" if skip_terminal else "champion_unchanged"
            log(
                f"  Cycle {cycle}: Champion unchanged (F={current_fitness:.3f} "
                f"L{current_level} W={current_words} traces={evo_result.get('champion_traces', 0)}) "
                f"-- skipping retrain ({reason})"
            )
            should_retrain = False
            retrain_result = {"retrained": False, "reason": reason, "checkpoint": checkpoint}
            if meta_controller is not None and skip_terminal:
                meta_controller._consecutive_none_val_loss = 0

    if last_retrain_epochs > 0 and should_retrain:
        adaptive_max_epochs = min(max(int(last_retrain_epochs * 1.5), 30), 50)
        current_max = retrain_meta_params.get("retrain_max_epochs", 50)
        if adaptive_max_epochs < current_max:
            retrain_meta_params["retrain_max_epochs"] = adaptive_max_epochs
            log(
                f"  Cycle {cycle}: Adaptive retrain budget: "
                f"{current_max} -> {adaptive_max_epochs} epochs "
                f"(last cycle used {last_retrain_epochs})"
            )
    return should_retrain, retrain_result, cycles_since_last_retrain


def _skipped_evolution_result(cycle: int, checkpoint: str) -> dict[str, Any]:
    return {
        "cycle": cycle,
        "generations_completed": 0,
        "final_fitness": 0.0,
        "final_level": 0,
        "champion_words": 0,
        "elapsed_seconds": 0.0,
        "champion_traces": 0,
        "novelty_archive_size": 0,
        "qd_occupied_cells": 0,
        "qd_total_cells": 0,
        "checkpoint": checkpoint,
        "mutation_rate": 0.5,
        "diversity_threshold": 0.3,
        "diversity": 0.3,
    }


def _run_evolution_phase(
    cycle: int,
    checkpoint: str,
    skip_evolution: bool,
    evo_meta_params: dict[str, Any],
    archive_params: dict[str, Any],
    gen_count_override: int,
    meta_controller: Any,
    prev_archive_state: dict[str, Any] | None,
    last_champion_level: int,
) -> EvolutionPhaseResult:
    from run_multi_cycle import PROJECT_ROOT, run_evolution

    _evo_start = time.time()
    prev_ee_max = os.environ.get("EE_MAX_LEVEL")
    clamped_start = os.environ.get("EE_CLAMPED_START_LEVEL")
    try:
        if clamped_start is not None and str(last_champion_level) == clamped_start:
            max_adv = int(os.environ.get("EE_CLAMPED_MAX_ADVANCE", "1"))
            os.environ["EE_MAX_LEVEL"] = str(last_champion_level + max_adv)
            log(
                f"  Cycle {cycle}: Clamped recovery cap EE_MAX_LEVEL="
                f"{os.environ['EE_MAX_LEVEL']} (start L{last_champion_level})"
            )
        if skip_evolution:
            log(f"  Cycle {cycle}: SKIPPING evolution (RESET-AND-RELEARN from previous cycle)")
            evo_result = _skipped_evolution_result(cycle, checkpoint)
            skip_evolution = False
        else:
            if meta_controller is not None and hasattr(meta_controller, "snapshot_before_cycle"):
                meta_controller.snapshot_before_cycle(cycle, checkpoint)
            evo_with_archive = dict(evo_meta_params)
            evo_with_archive.update(archive_params)
            if last_champion_level >= 30 and prev_archive_state:
                from evolution.archive_persistence import cap_strategy_archive_state

                cap = int(os.environ.get("EE_TERMINAL_STRATEGY_ARCHIVE_CAP", "3200"))
                prev_archive_state = cap_strategy_archive_state(prev_archive_state, cap) or prev_archive_state
            evo_result = run_evolution(
                cycle,
                gen_count_override,
                checkpoint,
                meta_evolution_params=evo_with_archive,
                archive_data=prev_archive_state,
                start_level=last_champion_level,
                meta_controller=meta_controller,
            )
    finally:
        if prev_ee_max is None:
            os.environ.pop("EE_MAX_LEVEL", None)
        else:
            os.environ["EE_MAX_LEVEL"] = prev_ee_max
        os.environ.pop("EE_TERMINAL_VERIFY_PROMOTE", None)
    evo_elapsed = time.time() - _evo_start
    if "archive_state" in evo_result:
        prev_archive_state = evo_result["archive_state"]
    elif evo_result.get("archive_state_path"):
        from evolution.archive_persistence import load_archive_from_result

        prev_archive_state = load_archive_from_result(PROJECT_ROOT, evo_result)
    prev_archive_state = _cap_terminal_archive_state(prev_archive_state, evo_result)
    prev_cycle_metrics = evo_result
    current_level = evo_result.get("final_level", 0)
    if (
        meta_controller is not None
        and current_level > meta_controller.best_level
        and hasattr(meta_controller, "snapshot_before_cycle")
    ):
        meta_controller.snapshot_before_cycle(cycle, checkpoint)
        meta_controller._log(f"Best-level snapshot: C{cycle} reached L{current_level}")
    return EvolutionPhaseResult(
        evo_result=evo_result,
        evo_elapsed=evo_elapsed,
        prev_archive_state=prev_archive_state,
        prev_cycle_metrics=prev_cycle_metrics,
        skip_evolution=skip_evolution,
    )


def _maybe_run_meta_evolution(cycle: int, evo_meta_params: dict[str, Any]) -> None:
    log(f"  Cycle {cycle}: *** META-EVOLUTION TRIGGERED ***")
    log(f"  Cycle {cycle}: Running lightweight meta-evolution...")
    try:
        from meta_evolution import MetaEvolutionConfig, MetaEvolutionEngine

        me_config = MetaEvolutionConfig(
            meta_population_size=4,
            meta_generations=2,
            base_generations=20,
            seed=cycle,
        )
        me_engine = MetaEvolutionEngine(meta_config=me_config)
        me_engine.run()
        best_cfg = me_engine.best_config()
        log(
            f"  Cycle {cycle}: Meta-evolution found: pop={best_cfg.population_size} "
            f"mu={best_cfg.mutation_rate:.3f} cx={best_cfg.crossover_rate:.3f}"
        )
        evo_meta_params.update(
            {
                "mutation_rate": best_cfg.mutation_rate,
                "crossover_rate": best_cfg.crossover_rate,
                "elite_ratio": best_cfg.elite_ratio,
                "population_size": best_cfg.population_size,
            }
        )
    except (RuntimeError, ValueError, KeyError) as e:
        log(f"  Cycle {cycle}: Meta-evolution failed: {e}")


def _collect_barrier_mc_tuning(
    cycle: int,
    args: Any,
    barrier_tuner: Any,
    meta_controller: Any,
) -> CycleTuningBundle:
    evo_meta_params: dict[str, Any] = {}
    retrain_meta_params: dict[str, Any] = {}
    archive_params: dict[str, Any] = {}
    gen_count_override = args.gens

    if barrier_tuner is not None:
        evo_meta_params.update(barrier_tuner.tune_evolution_params())
        retrain_meta_params.update(barrier_tuner.tune_retrain_params())
        archive_params.update(barrier_tuner.tune_archive_params())
        gen_count_override = archive_params.get("max_generations", args.gens)
        if gen_count_override != args.gens:
            log(f"  Cycle {cycle}: BarrierTuner overrides gens: {args.gens} -> {gen_count_override}")

    from evolution.terminal_stagnation import terminal_cycles_at_level, terminal_depth

    pc_state: dict[str, Any] = {}
    try:
        from run_multi_cycle import PROJECT_ROOT as _root

        _pc_path = os.path.join(_root, ".agents", "progress_controller_state.json")
        if os.path.isfile(_pc_path):
            with open(_pc_path, encoding="utf-8") as _pcf:
                pc_state = json.load(_pcf)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pc_state = {}
    term_level = int(pc_state.get("level", 0) or 0)
    term_verify = float(pc_state.get("verify_score", 0.0) or 0.0)
    if term_level >= terminal_depth() and term_verify >= 0.99 - 1e-6:
        cap = int(os.environ.get("EE_TERMINAL_MAX_GENS", "30"))
        if gen_count_override > cap:
            log(
                f"  Cycle {cycle}: Terminal gens cap {gen_count_override} -> {cap} "
                f"(stall_cycles={terminal_cycles_at_level()})"
            )
            gen_count_override = cap

    if meta_controller is not None:
        mc_evo = meta_controller.tune_evolution_params(cycle)
        mc_retrain = meta_controller.tune_retrain_params(cycle)
        for k, v in mc_evo.items():
            if k not in evo_meta_params:
                evo_meta_params[k] = v
        for k, v in mc_retrain.items():
            if k not in retrain_meta_params:
                retrain_meta_params[k] = v
        if mc_retrain.get("meta_evolution_requested"):
            _maybe_run_meta_evolution(cycle, evo_meta_params)

    return CycleTuningBundle(evo_meta_params, retrain_meta_params, archive_params, gen_count_override)


def _apply_progression_tuning_policy(
    cycle: int,
    tuning_policy: Any,
    tuning_actuator: Any,
    barrier_tuner: Any,
    evo_meta_params: dict[str, Any],
    retrain_meta_params: dict[str, Any],
) -> None:
    if tuning_policy is None:
        return
    current_cfg_snapshot: dict[str, Any] = {
        "mutation_rate": evo_meta_params.get("mutation_rate", 0.3),
        "crossover_rate": evo_meta_params.get("crossover_rate", 0.2),
        "elite_ratio": evo_meta_params.get("elite_ratio", 0.1),
        "novelty_weight": evo_meta_params.get("novelty_weight", 0.3),
        "qd_bonus_weight": evo_meta_params.get("qd_bonus_weight", 0.1),
        "diversity_threshold": evo_meta_params.get("diversity_threshold", 0.3),
        "diversity_inject_ratio": evo_meta_params.get("diversity_inject_ratio", 0.1),
        "tournament_size": evo_meta_params.get("tournament_size", 3),
        "retrain_max_epochs": retrain_meta_params.get("retrain_max_epochs", 50),
        "retrain_lr": retrain_meta_params.get("retrain_lr", 3e-5),
        "curriculum_stall_threshold": evo_meta_params.get("curriculum_stall_threshold", 2),
    }
    _tracker = None
    if barrier_tuner is not None and hasattr(barrier_tuner, "_tracker"):
        _tracker = barrier_tuner._tracker
    else:
        try:
            from progression_tracker import ProgressionTracker

            _tracker = ProgressionTracker()
        except (ImportError, RuntimeError):
            pass
    if _tracker is None:
        return
    adjustments = tuning_policy.compute_adjustments(_tracker, cycle, current_cfg_snapshot)
    if not adjustments:
        return
    adj = adjustments[0]

    class _Proxy:
        pass

    proxy = _Proxy()
    for k, v in current_cfg_snapshot.items():
        setattr(proxy, k, v)
    if not tuning_actuator.apply(adj, proxy):
        return
    evo_keys = {
        "mutation_rate",
        "crossover_rate",
        "elite_ratio",
        "novelty_weight",
        "qd_bonus_weight",
        "diversity_threshold",
        "diversity_inject_ratio",
        "tournament_size",
    }
    if adj.parameter in evo_meta_params or adj.parameter in evo_keys:
        evo_meta_params[adj.parameter] = adj.value
    else:
        retrain_meta_params[adj.parameter] = adj.value
    tuning_policy.record_applied(adj, cycle)
    log(f"  Cycle {cycle}: TuningPolicy applied: {adj.parameter}={adj.value} ({adj.reason})")


def _apply_terminal_diversity_boost(cycle: int, evo_meta_params: dict[str, Any], project_root: str) -> bool:
    """At max depth with frozen verify, explore via diversity — not reference graft repair.

    Returns True when cross-cycle archives should be skipped (deep terminal stall).
    """
    tick_path = os.path.join(project_root, ".agents", "tick_state.json")
    state_path = os.path.join(project_root, ".agents", "progress_controller_state.json")
    try:
        with open(tick_path, encoding="utf-8") as f:
            tick = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    progress_state: dict[str, Any] = {}
    cycles_at_level = 0
    try:
        if os.path.isfile(state_path):
            with open(state_path, encoding="utf-8") as f:
                progress_state = json.load(f)
                cycles_at_level = int(progress_state.get("cycles_at_level", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    from evolution.epi_momentum_governor import clear_plateau_boost, should_suppress_terminal_diversity_boost
    from evolution.terminal_stagnation import (
        terminal_certified_plateau,
        terminal_diversity_boost_params,
        terminal_diversity_boost_tier,
        terminal_efficiency_at_scan_floor,
    )

    if should_suppress_terminal_diversity_boost(project_root):
        if tick.get("terminal_diversity_boost") or tick.get("epi_plateau_boost"):
            clear_plateau_boost(project_root)
        return False

    if not tick.get("terminal_diversity_boost"):
        level = int(progress_state.get("level", 0))
        verify = float(progress_state.get("verify_score", 0.0))
        fitness = float(progress_state.get("raw_fitness", 0.0))
        if cycles_at_level >= 28 and terminal_certified_plateau(level, verify, fitness):
            tick["terminal_diversity_boost"] = True
            try:
                with open(tick_path, "w", encoding="utf-8") as f:
                    json.dump(tick, f, indent=2)
            except OSError:
                pass
        else:
            return False

    tier = terminal_diversity_boost_tier(cycles_at_level)
    stable_words = int(progress_state.get("terminal_efficiency_stable_words", 0))
    if stable_words > 0 and terminal_efficiency_at_scan_floor(stable_words):
        # Length floor known: tier-7 mutation (0.88) burns GPU on doomed
        # shrink/grow without finding held-out-passing bytecode.
        tier = min(tier, 4)
    floors = terminal_diversity_boost_params(tier)
    mu_floor = float(floors["mutation_rate"])
    evo_meta_params["novelty_weight"] = max(evo_meta_params.get("novelty_weight", 0.3), floors["novelty_weight"])
    evo_meta_params["diversity_threshold"] = max(
        evo_meta_params.get("diversity_threshold", 0.3), floors["diversity_threshold"]
    )
    evo_meta_params["mutation_rate"] = max(evo_meta_params.get("mutation_rate", 0.3), mu_floor)
    # Scheduler + adaptive ceiling otherwise cap exploration below tier floors.
    evo_meta_params["mutation_rate_start"] = max(evo_meta_params.get("mutation_rate_start", 0.0), mu_floor)
    evo_meta_params["mutation_rate_end"] = max(evo_meta_params.get("mutation_rate_end", 0.0), mu_floor * 0.92)
    at_scan_floor = stable_words > 0 and terminal_efficiency_at_scan_floor(stable_words)
    if at_scan_floor:
        evo_meta_params["adaptive_mutation_ceil"] = max(evo_meta_params.get("adaptive_mutation_ceil", 0.0), mu_floor)
    else:
        evo_meta_params["adaptive_mutation_ceil"] = max(
            evo_meta_params.get("adaptive_mutation_ceil", 0.0), mu_floor, 0.95
        )
    evo_meta_params["diversity_inject_ratio"] = max(
        evo_meta_params.get("diversity_inject_ratio", 0.1), floors["diversity_inject_ratio"]
    )
    evo_meta_params["curriculum_stall_threshold"] = min(int(evo_meta_params.get("curriculum_stall_threshold", 2)), 1)
    if tier >= 2:
        evo_meta_params["elite_ratio"] = min(evo_meta_params.get("elite_ratio", 0.1), 0.05)
        evo_meta_params["tournament_size"] = max(int(evo_meta_params.get("tournament_size", 3)), 5)
    if tier >= 5:
        evo_meta_params["crossover_rate"] = max(evo_meta_params.get("crossover_rate", 0.2), 0.45)
        evo_meta_params["champion_crossover_prob"] = max(evo_meta_params.get("champion_crossover_prob", 0.1), 0.35)
    skip_archives = tier >= 3
    if skip_archives:
        tick["terminal_archive_reset"] = True
        try:
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump(tick, f, indent=2)
        except OSError:
            pass
    log(
        f"  Cycle {cycle}: Terminal-depth diversity boost tier {tier} "
        f"(cycles={cycles_at_level}, mu={evo_meta_params['mutation_rate']}, "
        f"inject={evo_meta_params['diversity_inject_ratio']}"
        f"{', archive_reset' if skip_archives else ''})"
    )
    if tier >= 2:
        os.environ["EE_TERMINAL_VERIFY_PROMOTE"] = "1"
    return skip_archives


def cap_evo_meta_params_at_scan_floor(evo_meta_params: dict[str, Any], project_root: str) -> None:
    """Clamp meta-controller / stall tuners at held-out length floor.

    Tier-4 diversity floors already cap boost mutation; compound_stall and
    adaptive_mutation_ceil=0.95 can still push mr>0.85 and inflate cycle time
    without finding shorter held-out-passing bytecode.
    """
    from evolution.terminal_stagnation import terminal_diversity_boost_params, terminal_efficiency_at_scan_floor

    state_path = os.path.join(project_root, ".agents", "progress_controller_state.json")
    try:
        if os.path.isfile(state_path):
            with open(state_path, encoding="utf-8") as f:
                progress_state = json.load(f)
        else:
            return
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return
    stable_words = int(progress_state.get("terminal_efficiency_stable_words", 0))
    if stable_words <= 0 or not terminal_efficiency_at_scan_floor(stable_words):
        return
    mu_cap = float(terminal_diversity_boost_params(4)["mutation_rate"])
    for key in ("mutation_rate", "mutation_rate_start", "adaptive_mutation_ceil"):
        val = evo_meta_params.get(key)
        if val is not None and float(val) > mu_cap:
            evo_meta_params[key] = mu_cap
    end = evo_meta_params.get("mutation_rate_end")
    if end is not None and float(end) > mu_cap * 0.92:
        evo_meta_params["mutation_rate_end"] = mu_cap * 0.92


def _finalize_tuning_before_evolution(
    cycle: int,
    evo_meta_params: dict[str, Any],
    retrain_meta_params: dict[str, Any],
    archive_params: dict[str, Any],
    last_archive_params: dict[str, Any],
    prev_cycle_metrics: dict[str, Any],
    post_retrain_boost_cycles: int,
) -> tuple[dict[str, Any], int]:
    if cycle > 1 and prev_cycle_metrics:
        prev_mu = prev_cycle_metrics.get("mutation_rate")
        prev_dt = prev_cycle_metrics.get("diversity_threshold")
        if prev_mu is not None:
            evo_meta_params.setdefault("mutation_rate", prev_mu)
            log(f"  Cycle {cycle}: Using prev cycle's mutation_rate={prev_mu:.3f}")
        if prev_dt is not None:
            evo_meta_params.setdefault("diversity_threshold", prev_dt)
            log(f"  Cycle {cycle}: Using prev cycle's diversity_threshold={prev_dt:.3f}")
    if post_retrain_boost_cycles > 0:
        boost_novelty = max(evo_meta_params.get("novelty_weight", 0.3), 0.6)
        boost_diversity = max(evo_meta_params.get("diversity_threshold", 0.3), 0.5)
        boost_mutation = max(evo_meta_params.get("mutation_rate", 0.3), 0.7)
        evo_meta_params["novelty_weight"] = boost_novelty
        evo_meta_params["diversity_threshold"] = boost_diversity
        evo_meta_params["mutation_rate"] = boost_mutation
        evo_meta_params["diversity_inject_ratio"] = 0.4
        log(
            f"  Cycle {cycle}: Post-retrain diversity boost ACTIVE "
            f"(novelty={boost_novelty}, div={boost_diversity}, mu={boost_mutation}, "
            f"remaining={post_retrain_boost_cycles})"
        )
        post_retrain_boost_cycles -= 1
    from run_multi_cycle import PROJECT_ROOT

    _apply_terminal_diversity_boost(cycle, evo_meta_params, PROJECT_ROOT)
    cap_evo_meta_params_at_scan_floor(evo_meta_params, PROJECT_ROOT)
    if evo_meta_params:
        log(f"  Cycle {cycle}: Tuned evolution: {evo_meta_params}")
    if retrain_meta_params:
        log(f"  Cycle {cycle}: Tuned retrain: {retrain_meta_params}")
    if archive_params:
        log(f"  Cycle {cycle}: Tuned archive: {archive_params}")
    if not archive_params and last_archive_params:
        archive_params = dict(last_archive_params)
    return archive_params, post_retrain_boost_cycles


def _defrag_gpu_at_cycle_boundary(cycle: int) -> None:
    """Release fragmented CUDA memory at cycle boundaries (v13)."""
    try:
        import gc

        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated() / 1024**2
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated() / 1024**2
            freed = before - after
            if freed > 50:
                log(
                    f"  Cycle {cycle}: GPU memory defrag: freed {freed:.0f}MB "
                    f"(was {before:.0f}MB, now {after:.0f}MB)"
                )
    except (RuntimeError, OSError) as e:
        log(f"  Cycle {cycle}: GPU memory cleanup skipped ({e})")


@dataclass
class CheckpointGatingResult:
    checkpoint: str
    skip_evolution: bool
    post_retrain_boost_cycles: int
    retrain_result: dict[str, Any]


def _resolve_retrain_val_loss(
    cycle: int,
    retrain_result: dict[str, Any],
    project_root: str,
    parse_val_loss_from_retrain_log: Any,
) -> float | None:
    val_loss = retrain_result.get("val_loss")
    if val_loss is None:
        retrain_log = os.path.join(project_root, f"cycle_{cycle}", "retrain.log")
        val_loss = parse_val_loss_from_retrain_log(retrain_log)
    return val_loss


def _strip_best_checkpoint_for_evolution(cycle: int, best_ckpt_path: str, load_checkpoint: Any) -> None:
    try:
        import torch as _torch_strip

        ckpt_full = load_checkpoint(best_ckpt_path)
        sd = ckpt_full.get("model_state_dict", ckpt_full)
        clean_sd = {}
        for k, v in sd.items():
            key = k[11:] if k.startswith("_orig_mod.") else k
            clean_sd[key] = v
        train_cfg = ckpt_full.get("config", {})
        strip_cfg = {
            "vocab_size": train_cfg.get("vocab_size", 2048),
            "d_model": train_cfg.get("d_model", 512),
            "n_layers": train_cfg.get("n_layers", 14),
            "n_heads": train_cfg.get("n_heads", 8),
            "d_ff": train_cfg.get("d_ff", 1365),
            "max_seq_len": train_cfg.get("max_seq_len", 2048),
        }
        if not train_cfg:
            qkv_keys = [k for k in clean_sd if "blocks.0.attn.qkv.weight" in k]
            if qkv_keys:
                strip_cfg["d_model"] = clean_sd[qkv_keys[0]].shape[1]
            blk_keys = [k for k in clean_sd if k.startswith("blocks.") and "ln1.weight" in k]
            if blk_keys:
                strip_cfg["n_layers"] = len(blk_keys)
        _torch_strip.save(
            {
                "model_state_dict": clean_sd,
                "config": strip_cfg,
                "val_loss": ckpt_full.get("val_loss"),
            },
            best_ckpt_path,
        )
        log(f"  Cycle {cycle}: Stripped checkpoint for evolution ({strip_cfg['n_layers']}L/d={strip_cfg['d_model']})")
    except (RuntimeError, ValueError, OSError) as e:
        log(f"  Cycle {cycle}: Failed to strip checkpoint: {e}")


def _promote_cycle_ckpt_to_best(
    cycle: int,
    cycle_ckpt_path: str,
    best_ckpt_path: str,
    load_checkpoint: Any,
) -> None:
    shutil.copy2(cycle_ckpt_path, best_ckpt_path)
    log(f"  Cycle {cycle}: Promoted {os.path.basename(cycle_ckpt_path)} → checkpoint_best.pt")
    _strip_best_checkpoint_for_evolution(cycle, best_ckpt_path, load_checkpoint)


def _copy_ckpt_to_current(cycle: int, new_ck: str, m1_checkpoints: str) -> str | None:
    cycle_ckpt_cur = os.path.join(m1_checkpoints, "checkpoint_current.pt")
    try:
        shutil.copy2(new_ck, cycle_ckpt_cur)
        log(f"  Cycle {cycle}: Next cycle will use: {cycle_ckpt_cur}")
        return cycle_ckpt_cur
    except (OSError, PermissionError) as e:
        log(f"  Cycle {cycle}: Checkpoint copy failed: {e}")
        return None


def _on_checkpoint_accepted(
    cycle: int,
    meta_controller: Any,
    val_loss: float,
    new_ck: str,
    checkpoint: str,
    m1_checkpoints: str,
    load_checkpoint: Any,
) -> tuple[str, int]:
    log(f"  Cycle {cycle}: Checkpoint ACCEPTED (val_loss={val_loss:.4f})")
    if hasattr(meta_controller, "_consecutive_rejections"):
        meta_controller._consecutive_rejections = 0
    post_retrain_boost_cycles = 1
    best_ckpt_path = os.path.join(m1_checkpoints, "checkpoint_best.pt")
    cycle_ckpt_path = os.path.join(m1_checkpoints, f"checkpoint_cycle_{cycle}.pt")
    if os.path.isfile(cycle_ckpt_path):
        try:
            _promote_cycle_ckpt_to_best(cycle, cycle_ckpt_path, best_ckpt_path, load_checkpoint)
        except (OSError, PermissionError) as e:
            log(f"  Cycle {cycle}: Checkpoint promotion failed: {e}")
    if os.path.isfile(new_ck):
        updated = _copy_ckpt_to_current(cycle, new_ck, m1_checkpoints)
        if updated:
            checkpoint = updated
    return checkpoint, post_retrain_boost_cycles


def _on_checkpoint_rejected(
    cycle: int,
    meta_controller: Any,
    val_loss: float,
    checkpoint: str,
    skip_evolution: bool,
    m1_checkpoints: str,
) -> tuple[str, bool]:
    log(
        f"  Cycle {cycle}: Checkpoint REJECTED (val_loss={val_loss:.4f} > "
        f"threshold={meta_controller._best_val_loss * meta_controller.checkpoint_acceptance_tolerance:.4f})"
    )
    cycle_ckpt_rej = os.path.join(m1_checkpoints, f"checkpoint_cycle_{cycle}.pt")
    if os.path.isfile(cycle_ckpt_rej):
        try:
            os.remove(cycle_ckpt_rej)
            log(f"  Cycle {cycle}: Discarded rejected checkpoint {os.path.basename(cycle_ckpt_rej)}")
        except OSError:
            pass
    if not hasattr(meta_controller, "_consecutive_rejections"):
        meta_controller._consecutive_rejections = 0
    meta_controller._consecutive_rejections += 1
    if meta_controller._consecutive_rejections >= 3:
        log(
            f"  Cycle {cycle}: *** RESET-AND-RELEARN TRIGGERED *** "
            f"({meta_controller._consecutive_rejections} consecutive rejections)"
        )
        best_ever_ckpt = os.path.join(m1_checkpoints, "checkpoint_best.pt")
        if os.path.isfile(best_ever_ckpt):
            log(f"  Cycle {cycle}: Reset to best-ever checkpoint: {best_ever_ckpt}")
            checkpoint = best_ever_ckpt
            meta_controller._consecutive_rejections = 0
            if os.environ.get("EE_SKIP_EVOLUTION_ON_REJECTION", "0") == "1":
                log(f"  Cycle {cycle}: Arm B — skipping evolution (pre-R48 behavior)")
                skip_evolution = True
        else:
            log(f"  Cycle {cycle}: Reset failed — best checkpoint not found")
    return checkpoint, skip_evolution


def _log_val_loss_parse_failure(cycle: int, retrain_result: dict[str, Any], project_root: str) -> None:
    retrain_log_path = os.path.join(project_root, f"cycle_{cycle}", "retrain.log")
    retrain_reason = retrain_result.get("reason", "unknown")
    log(f"  Cycle {cycle}: Could not parse val_loss (reason: {retrain_reason})")
    if os.path.isfile(retrain_log_path):
        log(f"  Cycle {cycle}: Retrain log exists ({os.path.getsize(retrain_log_path)} bytes)")
        try:
            with open(retrain_log_path, encoding="utf-8", errors="replace") as rf:
                for rl in rf.readlines()[-5:]:
                    log(f"  Cycle {cycle}: [retrain log] {rl.rstrip()}")
        except OSError:
            pass
        if retrain_result.get("retrained") is False:
            log(f"  Cycle {cycle}: Retrain failed -- not accepting checkpoint")
    else:
        log(f"  Cycle {cycle}: Retrain log NOT FOUND at {retrain_log_path}")


def _retry_retrain_doubled_patience(
    cycle: int,
    checkpoint: str,
    current_level: int,
    retrain_result: dict[str, Any],
    merged_retrain_config: Any,
    meta_controller: Any,
    m1_checkpoints: str,
    retrain_model: Any,
    parse_val_loss_from_retrain_log: Any,
    project_root: str,
) -> tuple[str, dict[str, Any] | None]:
    retrain_reason = retrain_result.get("reason", "unknown")
    if retrain_result.get("retrained") is not False or "train_module" in retrain_reason:
        return checkpoint, None
    log(f"  Cycle {cycle}: Retrying retrain with doubled patience...")
    retry_config = vars(merged_retrain_config) if merged_retrain_config else {}
    retry_config["retrain_patience"] = retry_config.get("retrain_patience", 5) * 2
    retry_config["retrain_max_epochs"] = min(retry_config.get("retrain_max_epochs", 50) * 2, 100)
    from types import SimpleNamespace

    retry_cfg = SimpleNamespace(**retry_config)
    retry_result = retrain_model(cycle, checkpoint, config=retry_cfg, best_level=current_level)
    if not retry_result.get("retrained"):
        log(f"  Cycle {cycle}: Retry failed: {retry_result.get('reason', 'unknown')}")
        return checkpoint, retry_result
    val_loss2 = retry_result.get("val_loss")
    if val_loss2 is None:
        retrain_log2 = os.path.join(project_root, f"cycle_{cycle}", "retrain.log")
        val_loss2 = parse_val_loss_from_retrain_log(retrain_log2)
    if val_loss2 is None:
        log(f"  Cycle {cycle}: Retry also failed to produce val_loss")
        return checkpoint, retry_result
    log(f"  Cycle {cycle}: Retry succeeded! val_loss={val_loss2:.4f}")
    new_ck = retry_result.get("checkpoint")
    if new_ck and os.path.isfile(new_ck):
        meta_controller.accept_checkpoint(val_loss2, new_ck)
        best_ckpt_retry = os.path.join(m1_checkpoints, "checkpoint_best.pt")
        try:
            shutil.copy2(new_ck, best_ckpt_retry)
            log(f"  Cycle {cycle}: Retry promoted {os.path.basename(new_ck)} → checkpoint_best.pt")
        except (OSError, PermissionError) as e:
            log(f"  Cycle {cycle}: Retry promotion failed: {e}")
        cycle_ckpt = os.path.join(m1_checkpoints, "checkpoint_current.pt")
        shutil.copy2(new_ck, cycle_ckpt)
        checkpoint = cycle_ckpt
    return checkpoint, retry_result


def _fallback_accept_unparsable_val_loss(
    cycle: int,
    new_ck: str,
    retrain_result: dict[str, Any],
    checkpoint: str,
    m1_checkpoints: str,
) -> str:
    retry_ckpt = retrain_result.get("checkpoint")
    if not (retry_ckpt and os.path.isfile(retry_ckpt) and retry_ckpt == new_ck):
        log(f"  Cycle {cycle}: Retrain checkpoint skipped (retry or no rollback needed)")
        return checkpoint
    best_ckpt_fb = os.path.join(m1_checkpoints, "checkpoint_best.pt")
    try:
        shutil.copy2(new_ck, best_ckpt_fb)
        log(f"  Cycle {cycle}: Fallback promote {os.path.basename(new_ck)} → checkpoint_best.pt")
    except (OSError, PermissionError) as e:
        log(f"  Cycle {cycle}: Fallback promotion failed: {e}")
    cycle_ckpt = os.path.join(m1_checkpoints, "checkpoint_current.pt")
    try:
        shutil.copy2(new_ck, cycle_ckpt)
        checkpoint = cycle_ckpt
        log(f"  Cycle {cycle}: Fallback accept (val_loss unparsable): {cycle_ckpt}")
    except (OSError, PermissionError) as e:
        log(f"  Cycle {cycle}: Checkpoint copy failed: {e}")
    return checkpoint


def _auto_accept_retrain_without_mc(
    cycle: int,
    new_ck: str,
    m1_checkpoints: str,
    checkpoint: str,
) -> str:
    best_ckpt_no_mc = os.path.join(m1_checkpoints, "checkpoint_best.pt")
    try:
        shutil.copy2(new_ck, best_ckpt_no_mc)
        log(f"  Cycle {cycle}: Promoted {os.path.basename(new_ck)} → checkpoint_best.pt (no MC gate)")
    except (OSError, PermissionError) as e:
        log(f"  Cycle {cycle}: Checkpoint promotion failed: {e}")
    try:
        cycle_ckpt = os.path.join(m1_checkpoints, "checkpoint_current.pt")
        shutil.copy2(new_ck, cycle_ckpt)
        checkpoint = cycle_ckpt
        log(f"  Cycle {cycle}: Next cycle will use: {checkpoint}")
    except (OSError, PermissionError) as e:
        log(f"  Cycle {cycle}: Checkpoint copy failed: {e}")
    return checkpoint


def _apply_checkpoint_gating_with_mc(
    cycle: int,
    meta_controller: Any,
    retrain_result: dict[str, Any],
    checkpoint: str,
    skip_evolution: bool,
    post_retrain_boost_cycles: int,
    merged_retrain_config: Any,
    current_level: int,
    m1_checkpoints: str,
    project_root: str,
    parse_val_loss_from_retrain_log: Any,
    load_checkpoint: Any,
    retrain_model: Any,
) -> CheckpointGatingResult:
    new_ck = retrain_result.get("checkpoint")
    if not new_ck or not os.path.isfile(new_ck):
        if new_ck:
            log(f"  Cycle {cycle}: Retrain checkpoint not found at {new_ck}")
        return CheckpointGatingResult(checkpoint, skip_evolution, post_retrain_boost_cycles, retrain_result)

    val_loss = _resolve_retrain_val_loss(cycle, retrain_result, project_root, parse_val_loss_from_retrain_log)
    if val_loss is not None:
        accepted = meta_controller.accept_checkpoint(val_loss, new_ck)
        retrain_result["val_loss"] = val_loss
        if accepted:
            checkpoint, post_retrain_boost_cycles = _on_checkpoint_accepted(
                cycle, meta_controller, val_loss, new_ck, checkpoint, m1_checkpoints, load_checkpoint
            )
        else:
            checkpoint, skip_evolution = _on_checkpoint_rejected(
                cycle, meta_controller, val_loss, checkpoint, skip_evolution, m1_checkpoints
            )
    else:
        _log_val_loss_parse_failure(cycle, retrain_result, project_root)
        if merged_retrain_config is not None:
            checkpoint, _retry_result = _retry_retrain_doubled_patience(
                cycle,
                checkpoint,
                current_level,
                retrain_result,
                merged_retrain_config,
                meta_controller,
                m1_checkpoints,
                retrain_model,
                parse_val_loss_from_retrain_log,
                project_root,
            )
        if new_ck and os.path.isfile(new_ck):
            checkpoint = _fallback_accept_unparsable_val_loss(cycle, new_ck, retrain_result, checkpoint, m1_checkpoints)

    return CheckpointGatingResult(checkpoint, skip_evolution, post_retrain_boost_cycles, retrain_result)


def _apply_checkpoint_gating(
    cycle: int,
    meta_controller: Any,
    retrain_result: dict[str, Any],
    checkpoint: str,
    skip_evolution: bool,
    post_retrain_boost_cycles: int,
    merged_retrain_config: Any,
    current_level: int,
    m1_checkpoints: str,
    project_root: str,
    parse_val_loss_from_retrain_log: Any,
    load_checkpoint: Any,
    retrain_model: Any,
) -> CheckpointGatingResult:
    if not retrain_result.get("retrained"):
        return CheckpointGatingResult(checkpoint, skip_evolution, post_retrain_boost_cycles, retrain_result)

    if meta_controller is not None:
        return _apply_checkpoint_gating_with_mc(
            cycle,
            meta_controller,
            retrain_result,
            checkpoint,
            skip_evolution,
            post_retrain_boost_cycles,
            merged_retrain_config,
            current_level,
            m1_checkpoints,
            project_root,
            parse_val_loss_from_retrain_log,
            load_checkpoint,
            retrain_model,
        )

    new_ck = retrain_result.get("checkpoint")
    if new_ck and os.path.isfile(new_ck):
        checkpoint = _auto_accept_retrain_without_mc(cycle, new_ck, m1_checkpoints, checkpoint)
    return CheckpointGatingResult(checkpoint, skip_evolution, post_retrain_boost_cycles, retrain_result)


@dataclass
class BarrierTunerPostCycleResult:
    last_archive_params: Any
    last_retrain_epochs: int
    last_champion_fitness: float
    last_champion_level: int
    last_champion_words: int
    should_terminate: bool


def _build_cycle_telemetry_snapshot(
    cycle: int,
    evo_result: dict[str, Any],
    retrain_result: dict[str, Any],
    project_root: str,
    parse_val_loss_from_retrain_log: Any,
    telemetry_snapshot_cls: Any,
) -> Any:
    diversity_proxy = 0.3
    if evo_result.get("novelty_archive_size", 0) > 0:
        qd_total = max(evo_result.get("qd_total_cells", 100), 1)
        diversity_proxy = min(1.0, evo_result.get("qd_occupied_cells", 0) / qd_total)
    snapshot_val_loss = retrain_result.get("val_loss")
    if snapshot_val_loss is None and retrain_result.get("retrained"):
        retrain_log = os.path.join(project_root, f"cycle_{cycle}", "retrain.log")
        snapshot_val_loss = parse_val_loss_from_retrain_log(retrain_log)
        if snapshot_val_loss is None:
            log(
                f"  Cycle {cycle}: [parser debug] val_loss=None, log={retrain_log}, "
                f"exists={os.path.isfile(retrain_log)}"
            )
    return telemetry_snapshot_cls(
        cycle=cycle,
        fitness=evo_result.get("final_fitness", 0.0),
        level=evo_result.get("final_level", 0),
        champion_words=evo_result.get("champion_words", 0),
        val_loss=snapshot_val_loss,
        diversity=diversity_proxy,
        novelty_size=evo_result.get("novelty_archive_size", 0),
        qd_occupied=evo_result.get("qd_occupied_cells", 0),
        qd_total=evo_result.get("qd_total_cells", 100),
        traces_count=evo_result.get("champion_traces", 0),
        retrained=retrain_result.get("retrained", False),
        generations_completed=evo_result.get("generations_completed", 0),
    )


def _update_meta_controller_after_cycle(
    cycle: int,
    meta_controller: Any,
    evo_result: dict[str, Any],
    retrain_result: dict[str, Any],
    checkpoint: str,
    barrier_tuner: Any,
    project_root: str,
    meta_controller_state: str,
    parse_val_loss_from_retrain_log: Any,
    telemetry_snapshot_cls: Any,
) -> tuple[str, dict[str, Any]]:
    snapshot = _build_cycle_telemetry_snapshot(
        cycle, evo_result, retrain_result, project_root, parse_val_loss_from_retrain_log, telemetry_snapshot_cls
    )
    meta_controller.update_cycle(snapshot)
    if retrain_result.get("retrained"):
        log(
            f"  Cycle {cycle}: Retrain diag: "
            f"val_loss={retrain_result.get('val_loss')} "
            f"elapsed={retrain_result.get('elapsed_seconds')}s "
            f"examples={retrain_result.get('example_count')}"
        )
    if meta_controller.should_rollback():
        log(f"  Cycle {cycle}: *** ROLLBACK TRIGGERED ***")
        log(f"  Cycle {cycle}: Rolling back to best checkpoint...")
        use_snap = evo_result.get("final_level", 0) < meta_controller.best_level
        rolled = meta_controller.execute_rollback(checkpoint, use_snapshot=use_snap)
        if rolled is not None:
            checkpoint = rolled
            log(f"  Cycle {cycle}: Rollback complete. Using: {checkpoint}")
            retrain_result["rollback_performed"] = True
            if barrier_tuner is not None:
                barrier_tuner.notify_rollback()
                log(f"  Cycle {cycle}: BarrierTuner: stagnation reset after rollback")
        else:
            log(f"  Cycle {cycle}: Rollback FAILED -- continuing with current checkpoint")
    try:
        meta_controller.save_state(meta_controller_state)
    except (OSError, ValueError) as e:
        log(f"  Cycle {cycle}: Failed to save MetaController state: {e}")
    return checkpoint, retrain_result


def _parse_val_loss_from_retrain_log_file(cycle: int, project_root: str, val_loss_bt_re: Any) -> float | None:
    rl = os.path.join(project_root, f"cycle_{cycle}", "retrain.log")
    try:
        with open(rl, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = val_loss_bt_re.search(line)
                if m:
                    return float(m.group(1))
    except OSError:
        pass
    return None


def _parse_last_retrain_epochs(cycle: int, project_root: str, epoch_re: Any) -> int | None:
    rl = os.path.join(project_root, f"cycle_{cycle}", "retrain.log")
    try:
        last_epoch = None
        with open(rl, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = epoch_re.search(line)
                if m:
                    last_epoch = int(m.group(1))
        if last_epoch is not None:
            return last_epoch + 1
    except OSError:
        pass
    return None


def _update_barrier_tuner_after_cycle(
    cycle: int,
    barrier_tuner: Any,
    evo_result: dict[str, Any],
    retrain_result: dict[str, Any],
    archive_params: dict[str, Any],
    last_archive_params: Any,
    last_retrain_epochs: int,
    last_champion_fitness: float,
    last_champion_level: int,
    last_champion_words: int,
    project_root: str,
    all_results: list[dict[str, Any]],
    val_loss_bt_re: Any,
    epoch_re: Any,
    barrier_type: Any,
    cycle_metrics_cls: Any,
    progress_controller: Any | None = None,
) -> BarrierTunerPostCycleResult:
    val_loss_val = None
    if retrain_result.get("retrained"):
        val_loss_val = _parse_val_loss_from_retrain_log_file(cycle, project_root, val_loss_bt_re)
    bt_metrics = cycle_metrics_cls(
        cycle=cycle,
        fitness=evo_result.get("final_fitness", 0.0),
        level=evo_result.get("final_level", 0),
        champion_words=evo_result.get("champion_words", 0),
        verify_score=float(evo_result.get("champion_verify_score", 0.0) or 0.0),
        diversity=evo_result.get("final_diversity", 0.3),
        qd_occupied=evo_result.get("qd_occupied_cells", 0),
        qd_total=max(evo_result.get("qd_total_cells", 100), 1),
        novelty_size=evo_result.get("novelty_archive_size", 0),
        traces_count=evo_result.get("champion_traces", 0),
        retrained=retrain_result.get("retrained", False),
        val_loss=val_loss_val,
        generations_completed=evo_result.get("generations_completed", 0),
        mutation_rate=evo_result.get("mutation_rate", 0.5),
    )
    bt_metrics.diversity = evo_result.get("diversity", 0.3)
    barrier_info = barrier_tuner.update(bt_metrics)
    if (barrier_info.barrier != barrier_type.NORMAL) if barrier_type else True:
        log(
            f"  Cycle {cycle}: BarrierTuner detected: "
            f"{barrier_info.barrier.value} "
            f"(x{barrier_info.consecutive_count}, "
            f"severity={barrier_info.severity})"
        )
    last_archive_params = archive_params
    if retrain_result.get("retrained"):
        epochs_used = _parse_last_retrain_epochs(cycle, project_root, epoch_re)
        if epochs_used is not None:
            last_retrain_epochs = epochs_used
            log(f"  Cycle {cycle}: Retrain used {last_retrain_epochs} epochs")
    last_champion_fitness = evo_result.get("final_fitness", 0.0)
    if last_champion_fitness == 0.0 and last_champion_level > 3:
        old_level = last_champion_level
        last_champion_level = max(3, last_champion_level - 3)
        log(f"  Cycle {cycle}: F=0.000 → lowering level L{old_level}→L{last_champion_level}")
    else:
        carried = _carry_forward_champion_level(evo_result, all_results, project_root)
        nudged = _maybe_nudge_champion_level(cycle, project_root, evo_result, carried, progress_controller)
        last_champion_level = max(carried, nudged)
    last_champion_words = evo_result.get("champion_words", 0)
    should_terminate = barrier_tuner.should_terminate()
    if should_terminate:
        reason = barrier_tuner.get_termination_reason()
        log(f"  Cycle {cycle}: {reason}")
        log(f"  Cycle {cycle}: Stopping run -- no meaningful progress for extended period")
    return BarrierTunerPostCycleResult(
        last_archive_params=last_archive_params,
        last_retrain_epochs=last_retrain_epochs,
        last_champion_fitness=last_champion_fitness,
        last_champion_level=last_champion_level,
        last_champion_words=last_champion_words,
        should_terminate=should_terminate,
    )


def _build_combined_cycle_results(
    cycle: int,
    evo_result: dict[str, Any],
    retrain_result: dict[str, Any],
    evo_elapsed: float,
    retrain_start: float,
) -> dict[str, Any]:
    if "error" in evo_result:
        log(f"  Cycle {cycle}: Evolution had errors -- see above")
    combined = {**evo_result, **retrain_result}
    retrain_elapsed = time.time() - retrain_start
    total_elapsed = evo_elapsed + retrain_elapsed
    combined["phase_evolution_seconds"] = round(evo_elapsed, 2)
    combined["phase_retrain_seconds"] = round(retrain_elapsed, 2)
    combined["phase_total_seconds"] = round(total_elapsed, 2)
    combined["elapsed_seconds"] = round(total_elapsed, 2)
    fit = float(combined.get("final_fitness", 0))
    if fit <= -25.0:
        start_lvl = int(combined.get("start_level", combined.get("final_level", 0)))
        raw_level = int(combined.get("final_level", start_lvl))
        capped = min(raw_level, start_lvl + 1)
        if capped < raw_level:
            combined["final_level_spike"] = raw_level
            combined["final_level"] = capped
            log(f"  Cycle {cycle}: Timeout level cap L{raw_level}->L{capped} " f"(start L{start_lvl}, fit={fit:.1f})")
    log(
        f"  Cycle {cycle}: phases: evolution={evo_elapsed:.1f}s "
        f"retrain={retrain_elapsed:.1f}s total={total_elapsed:.1f}s"
    )
    return combined


def _run_cycle_retrain_diagnostics(
    cycle: int,
    evo_result: dict[str, Any],
    retrain_result: dict[str, Any],
    checkpoint: str,
    m1_checkpoints: str,
    compute_forgetting_curve: Any,
    compute_weight_delta: Any,
) -> None:
    champion_prog = evo_result.get("champion_program")
    champion_level = evo_result.get("final_level", 0)
    champion_int: list[int] | None = None
    if champion_prog:
        try:
            champion_int = [int(w, 16) for w in champion_prog]
        except (ValueError, TypeError):
            pass
    retrain_ckpt = None
    if checkpoint and os.path.isfile(checkpoint):
        retrain_ckpt = checkpoint
    elif os.path.isfile(os.path.join(m1_checkpoints, "checkpoint_best.pt")):
        retrain_ckpt = os.path.join(m1_checkpoints, "checkpoint_best.pt")
    if not (retrain_result.get("retrained") and champion_int and champion_level > 0):
        return
    try:
        from evolution_engine import VMFarm as _DiagVMFarm

        _diag_vm = _DiagVMFarm(num_workers=1, use_gpu=os.environ.get("EE_USE_GPU", "1") != "0")
        forgetting_scores = compute_forgetting_curve(champion_int, champion_level, _diag_vm)
        fc_perfect = sum(1 for s in forgetting_scores.values() if s >= 140.0)
        fc_total = len(forgetting_scores)
        log(f"  Cycle {cycle}: [ForgettingCurve] {fc_perfect}/{fc_total} levels still perfect")
    except (RuntimeError, ValueError, KeyError) as e:
        log(f"  Cycle {cycle}: Forgetting curve failed: {e}")
    backup_ckpt = retrain_result.get("backup_checkpoint")
    if backup_ckpt and os.path.isfile(backup_ckpt) and retrain_ckpt and os.path.isfile(retrain_ckpt):
        try:
            weight_deltas = compute_weight_delta(backup_ckpt, retrain_ckpt)
            if weight_deltas:
                max_delta = max(weight_deltas.values())
                mean_delta = sum(weight_deltas.values()) / len(weight_deltas)
                log(
                    f"  Cycle {cycle}: [WeightDelta] "
                    f"max_L2={max_delta:.4f} mean_L2={mean_delta:.4f} "
                    f"({len(weight_deltas)} layers)"
                )
        except (RuntimeError, ValueError, OSError) as e:
            log(f"  Cycle {cycle}: Weight delta failed: {e}")


def _cleanup_gpu_memory_end_of_cycle(cycle: int) -> None:
    try:
        import gc

        import torch

        if torch.cuda.is_available():
            for _name in ("model", "optimizer", "scaler", "swa_model"):
                if _name in dir():
                    pass  # can't del local in this scope, but gc handles it
            gc.collect()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated() / 1024**2
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            after = torch.cuda.memory_allocated() / 1024**2
            freed = before - after
            if freed > 10:
                log(
                    f"  Cycle {cycle}: GPU memory cleanup: "
                    f"{before:.0f}MB -> {after:.0f}MB "
                    f"(freed {freed:.0f}MB)"
                )
            peak = torch.cuda.max_memory_allocated() / 1024**2
            log(
                f"  Cycle {cycle}: GPU mem: alloc={after:.0f}MB peak={peak:.0f}MB "
                f"total={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
            )
    except (RuntimeError, OSError) as e:
        log(f"  Cycle {cycle}: GPU cleanup skipped ({e})")


def _notify_progress_trajectory(project_root: str) -> None:
    """Incremental trajectory sync after each cycle (fresh interpreter — never in-process).

    Long-running batch processes keep a stale import of progress_trajectory.py.
    In-process sync_trajectory() wrote pre-bump EPI (~91) and advanced the byte
    offset so the live watcher had nothing left to append.
    """
    script = os.path.join(project_root, "scripts", "progress_trajectory.py")
    if not os.path.isfile(script):
        return
    import subprocess
    import sys

    try:
        subprocess.run(
            [sys.executable, script, "--record"],
            cwd=project_root,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _result_rank(row: dict[str, Any]) -> tuple:
    fit = float(row.get("final_fitness", -1e9))
    healthy = fit > -25.0
    return (
        1 if healthy else 0,
        int(row.get("final_level", 0)),
        float(row.get("champion_verify_score", 0)),
        fit,
    )


def _upsert_cycle_result(results_file: str, slim: dict[str, Any]) -> None:
    """Insert or update one cycle row; never leave duplicate cycle numbers."""
    cycle = int(slim.get("cycle", -1))
    if cycle < 0:
        return
    lock_path = results_file + ".lock"
    acquired = False
    for _ in range(60):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.1)
    if not acquired:
        log(f"  Warning: could not lock {results_file} for cycle {cycle}")
    try:
        by_cycle: dict[int, dict[str, Any]] = {}
        order: list[int] = []
        if os.path.isfile(results_file):
            with open(results_file, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in row:
                        continue
                    c = int(row.get("cycle", -1))
                    if c < 0:
                        continue
                    prev = by_cycle.get(c)
                    if prev is None:
                        order.append(c)
                        by_cycle[c] = row
                    elif _result_rank(row) > _result_rank(prev):
                        by_cycle[c] = row
        prev = by_cycle.get(cycle)
        if prev is None:
            order.append(cycle)
            by_cycle[cycle] = slim
        elif _result_rank(slim) >= _result_rank(prev):
            by_cycle[cycle] = slim
        else:
            return
        with open(results_file, "w", encoding="utf-8") as f:
            for c in sorted(order):
                f.write(json.dumps(by_cycle[c]) + "\n")
    finally:
        if acquired:
            try:
                os.remove(lock_path)
            except OSError:
                pass


def _rewrite_results_cycle(results_file: str, cycle: int, slim: dict[str, Any]) -> None:
    """Replace an existing cycle row in-place (avoids duplicate-cycle metric pollution)."""
    _upsert_cycle_result(results_file, slim)


def _persist_cycle_outputs(
    cycle: int,
    combined: dict[str, Any],
    evo_result: dict[str, Any],
    retrain_result: dict[str, Any],
    cycles_since_last_retrain: int,
    project_root: str,
    results_file: str,
) -> None:
    try:
        from evolution.archive_persistence import slim_result_for_jsonl

        slim = slim_result_for_jsonl(combined)
        try:
            from evolution.fitness_helpers import align_scan_floor_persisted_metrics

            align_scan_floor_persisted_metrics(None, slim)
        except (ImportError, RuntimeError, TypeError, ValueError, AttributeError):
            pass
        _upsert_cycle_result(results_file, slim)
        log(f"  Cycle {cycle}: Results saved to {results_file}")
        _notify_progress_trajectory(project_root)
    except (OSError, PermissionError) as e:
        log(f"  Cycle {cycle}: Failed to write results: {e}")
    try:
        retrain_decision = {
            "cycle": cycle,
            "decision": "attempted" if retrain_result.get("retrained") else "skipped",
            "reason": retrain_result.get("reason", "unknown"),
            "retrained": retrain_result.get("retrained", False),
            "fitness": evo_result.get("final_fitness", 0.0),
            "level": evo_result.get("final_level", 0),
            "words": evo_result.get("champion_words", 0),
            "cycles_since_last_retrain": cycles_since_last_retrain,
            "champion_traces": evo_result.get("champion_traces", 0),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
        retrain_log_path = os.path.join(project_root, "retrain_decisions.jsonl")
        with open(retrain_log_path, "a") as f:
            f.write(json.dumps(retrain_decision) + "\n")
    except OSError as e:
        log(f"  Cycle {cycle}: Failed to write retrain decision log: {e}")


def _finalize_cycle(
    cycle: int,
    evo_result: dict[str, Any],
    retrain_result: dict[str, Any],
    evo_elapsed: float,
    retrain_start: float,
    checkpoint: str,
    m1_checkpoints: str,
    prometheus_server: Any,
    update_metrics: Any,
    compute_forgetting_curve: Any,
    compute_weight_delta: Any,
    project_root: str,
    results_file: str,
    cycles_since_last_retrain: int,
) -> dict[str, Any]:
    combined = _build_combined_cycle_results(cycle, evo_result, retrain_result, evo_elapsed, retrain_start)
    if prometheus_server is not None:
        try:
            update_metrics(combined)
        except (RuntimeError, ValueError, KeyError) as e:
            log(f"  Cycle {cycle}: Prometheus metrics update failed: {e}")
    _run_cycle_retrain_diagnostics(
        cycle, evo_result, retrain_result, checkpoint, m1_checkpoints, compute_forgetting_curve, compute_weight_delta
    )
    _cleanup_gpu_memory_end_of_cycle(cycle)
    _persist_cycle_outputs(
        cycle, combined, evo_result, retrain_result, cycles_since_last_retrain, project_root, results_file
    )
    return combined


def execute_cycles(ctx) -> None:
    """Run all cycles; mutates ctx in place."""
    from run_multi_cycle import (
        _EPOCH_RE,
        _VAL_LOSS_BT_RE,
        M1_CHECKPOINTS,
        M1_RECURRENT_DATA,
        META_CONTROLLER_STATE,
        PROJECT_ROOT,
        RESULTS_FILE,
        BarrierType,
        CycleMetrics,
        TelemetrySnapshot,
        compute_forgetting_curve,
        compute_weight_delta,
        load_checkpoint,
        merge_archive_to_recurrent,
        parse_val_loss_from_retrain_log,
        retrain_model,
        update_metrics,
    )

    ctx.skip_evolution = False  # R6 FIX: initialize BEFORE the loop; consumed inside

    args = ctx.args
    checkpoint = ctx.checkpoint
    retrain_config = ctx.retrain_config
    completed_cycles = ctx.completed_cycles
    all_results = ctx.all_results
    meta_controller = ctx.meta_controller
    barrier_tuner = ctx.barrier_tuner
    prometheus_server = ctx.prometheus_server
    last_archive_params = ctx.last_archive_params
    last_retrain_epochs = ctx.last_retrain_epochs
    last_champion_fitness = ctx.last_champion_fitness
    last_champion_level = ctx.last_champion_level
    last_champion_words = ctx.last_champion_words
    cycles_since_last_retrain = ctx.cycles_since_last_retrain
    post_retrain_boost_cycles = ctx.post_retrain_boost_cycles
    tuning_policy = ctx.tuning_policy
    tuning_actuator = ctx.tuning_actuator
    prev_cycle_metrics = ctx.prev_cycle_metrics
    prev_archive_state = ctx.prev_archive_state
    skip_evolution = ctx.skip_evolution
    cycle_start = getattr(ctx, "cycle_start", 1)
    cycle_end = cycle_start + args.cycles - 1

    if os.environ.get("EE_EPI_MOMENTUM_GOVERNOR", "1") == "1":
        try:
            from evolution.epi_momentum_governor import run_cycle_governor

            epi_report, plateau_report = run_cycle_governor(PROJECT_ROOT)
            if epi_report.needs_intervention:
                log(
                    f"EPI momentum intervention: recent_avg={epi_report.recent_avg:.3f} "
                    f"full_avg={epi_report.avg_momentum:.3f} target>{epi_report.target_avg}"
                )
            elif plateau_report and plateau_report.needs_boost:
                log(
                    f"EPI plateau boost: streak={plateau_report.negative_streak} "
                    f"incoming_avg={plateau_report.incoming_avg:.3f}"
                )
        except Exception as exc:  # noqa: BLE001
            log(f"EPI momentum governor skipped: {exc!r}")

    for cycle in range(cycle_start, cycle_end + 1):
        batch_idx = cycle - cycle_start + 1
        log(f"\n{'-' * 60}")
        log(f"CYCLE {cycle} (batch {batch_idx}/{args.cycles})")
        log(f"{'-' * 60}")

        if args.resume and cycle in completed_cycles:
            log(f"  Cycle {cycle} already completed -- skipping")
            continue

        _defrag_gpu_at_cycle_boundary(cycle)

        _prune_stale_tick_reseed_state(PROJECT_ROOT, last_champion_level)

        last_champion_level = _maybe_apply_collapse_recovery(PROJECT_ROOT, cycle, last_champion_level)

        try:
            from evolution.champion_carry_forward import load_carry_forward_start_level

            sustained = _sustained_level_cap(PROJECT_ROOT, all_results)
            store_level = load_carry_forward_start_level(PROJECT_ROOT, sustained)
            if store_level > last_champion_level:
                log(f"  Cycle {cycle}: Carry-forward start boost " f"L{last_champion_level}->L{store_level}")
                last_champion_level = store_level
                if ctx.progress_controller is not None:
                    ctx.progress_controller.state.level = max(store_level, ctx.progress_controller.state.level)
                    ctx.progress_controller.state.cycles_at_level = max(
                        1, ctx.progress_controller.state.cycles_at_level
                    )
                    ctx.progress_controller.save()
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

        last_champion_level = _auto_depth_escape_start_level(PROJECT_ROOT, cycle, last_champion_level)

        tick_path = os.path.join(PROJECT_ROOT, ".agents", "tick_state.json")
        force_reseed = False
        try:
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
            force_reseed = bool(tick.get("force_reference_champion") or tick.get("force_champion_reseed"))
            if force_reseed:
                os.environ["EE_FORCE_REFERENCE_CHAMPION"] = "1"
            if (tick.get("force_register_repair") or tick.get("force_champion_reseed")) and not tick.get(
                "terminal_diversity_boost"
            ):
                os.environ["EE_REGISTER_REPAIR_BOOST"] = "1"
            if force_reseed and tick.pop("collapse_reseed", None):
                try:
                    with open(tick_path, "w", encoding="utf-8") as f:
                        json.dump(tick, f, indent=2)
                except OSError:
                    pass
            if force_reseed:
                nudge_level = int(tick.get("reseed_level", last_champion_level))
                if nudge_level > last_champion_level:
                    log(f"  Cycle {cycle}: Depth nudge start level L{nudge_level}")
                    last_champion_level = nudge_level
            depth_nudge = tick.pop("depth_nudge_level", None)
            if depth_nudge is not None:
                depth_nudge = int(depth_nudge)
                min_depth = int(os.environ.get("EE_DEPTH_ESCAPE_CYCLES", "8"))
                pc_cycles = 0
                pc_verify = 0.0
                pc_state = os.path.join(PROJECT_ROOT, ".agents", "progress_controller_state.json")
                try:
                    if os.path.isfile(pc_state):
                        with open(pc_state, encoding="utf-8") as f:
                            st = json.load(f)
                        pc_cycles = int(st.get("cycles_at_level", 0))
                        pc_verify = float(st.get("verify_score", 0.0))
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
                if depth_nudge > last_champion_level:
                    from evolution.curriculum_advance import nudge_verify_passes

                    nudge_level = max(last_champion_level, depth_nudge - 1)
                    if pc_cycles >= min_depth and nudge_verify_passes(nudge_level, pc_verify, pc_cycles):
                        log(f"  Cycle {cycle}: Depth nudge start level L{depth_nudge} " f"(keep champion)")
                        last_champion_level = depth_nudge
                    else:
                        log(
                            f"  Cycle {cycle}: Depth nudge L{depth_nudge} deferred "
                            f"(cycles={pc_cycles} verify={pc_verify:.3f})"
                        )
                        tick["depth_nudge_level"] = depth_nudge
                try:
                    with open(tick_path, "w", encoding="utf-8") as f:
                        json.dump(tick, f, indent=2)
                except OSError:
                    pass
        except (OSError, json.JSONDecodeError):
            pass

        archive_for_cycle = None if force_reseed else prev_archive_state
        if force_reseed and prev_archive_state is not None:
            log(f"  Cycle {cycle}: Skipping archive restore for forced champion reseed")
        elif (
            not force_reseed
            and prev_cycle_metrics
            and int(prev_cycle_metrics.get("final_level", 0)) >= 21
            and prev_archive_state is not None
        ):
            from evolution.terminal_stagnation import terminal_certified_plateau

            _prev_lvl = int(prev_cycle_metrics.get("final_level", 0))
            _prev_verify = float(prev_cycle_metrics.get("champion_verify_score", 0.0))
            _prev_fit = float(prev_cycle_metrics.get("final_fitness", 0.0))
            archive_for_cycle = None
            if terminal_certified_plateau(_prev_lvl, _prev_verify, _prev_fit):
                log(f"  Cycle {cycle}: Skipping archive restore " f"(certified L{_prev_lvl} terminal fast path)")
            else:
                log(f"  Cycle {cycle}: Skipping QD archive restore (L21+ depth guard)")
        elif (
            not force_reseed
            and prev_cycle_metrics
            and int(prev_cycle_metrics.get("final_level", 0)) >= 21
            and float(prev_cycle_metrics.get("champion_verify_score", 0.0)) >= 0.72
            and int(prev_cycle_metrics.get("champion_words", 99)) <= 36
            and prev_archive_state is not None
        ):
            archive_for_cycle = None
            log(f"  Cycle {cycle}: Skipping QD archive restore (certified L21 champion)")

        tuning = _collect_barrier_mc_tuning(cycle, args, barrier_tuner, meta_controller)
        evo_meta_params = tuning.evo_meta_params
        retrain_meta_params = tuning.retrain_meta_params
        archive_params = tuning.archive_params
        gen_count_override = tuning.gen_count_override

        _apply_progression_tuning_policy(
            cycle, tuning_policy, tuning_actuator, barrier_tuner, evo_meta_params, retrain_meta_params
        )
        archive_params, post_retrain_boost_cycles = _finalize_tuning_before_evolution(
            cycle,
            evo_meta_params,
            retrain_meta_params,
            archive_params,
            last_archive_params,
            prev_cycle_metrics,
            post_retrain_boost_cycles,
        )
        try:
            with open(tick_path, encoding="utf-8") as f:
                tick_reset = json.load(f)
            if tick_reset.pop("terminal_archive_reset", None):
                archive_for_cycle = None
                log(f"  Cycle {cycle}: Skipping archive restore (terminal-depth stagnation reset)")
                with open(tick_path, "w", encoding="utf-8") as f:
                    json.dump(tick_reset, f, indent=2)
        except (OSError, json.JSONDecodeError):
            pass

        evo_phase = _run_evolution_phase(
            cycle,
            checkpoint,
            skip_evolution,
            evo_meta_params,
            archive_params,
            gen_count_override,
            meta_controller,
            archive_for_cycle,
            last_champion_level,
        )
        evo_result = evo_phase.evo_result
        _evo_elapsed = evo_phase.evo_elapsed
        prev_archive_state = evo_phase.prev_archive_state
        prev_cycle_metrics = evo_phase.prev_cycle_metrics
        skip_evolution = evo_phase.skip_evolution

        # Phase 2: Retrain (unless --no-retrain)
        _retrain_start = time.time()
        retrain_result: dict[str, Any] = {"retrained": False, "reason": "skipped"}
        merged_retrain_config: Any = None
        if not args.no_retrain:
            from evolution.terminal_stagnation import terminal_certified_plateau

            should_retrain, retrain_result, cycles_since_last_retrain = _evaluate_retrain_skip(
                cycle,
                args,
                evo_result,
                meta_controller,
                checkpoint,
                last_champion_fitness,
                last_champion_level,
                last_champion_words,
                cycles_since_last_retrain,
                last_retrain_epochs,
                retrain_meta_params,
            )
            if should_retrain:
                retrain_meta_params.setdefault("retrain_batch_size", 8)
                retrain_meta_params.setdefault("retrain_val_split", 0.15)
                merged_retrain_config = _build_merged_retrain_config(retrain_config, retrain_meta_params)
                merge_archive_to_recurrent(cycle)
                current_level = evo_result.get("final_level", 0)
                retrain_result = retrain_model(
                    cycle, checkpoint, config=merged_retrain_config, best_level=current_level
                )
                cycles_since_last_retrain = 0
                if args.skip_retrain_if_no_new_data and retrain_result.get("retrained"):
                    _recurrent_path = M1_RECURRENT_DATA
                    _last_retrain_mtime_file = os.path.join(PROJECT_ROOT, ".last_retrain_mtime")
                    try:
                        with open(_last_retrain_mtime_file, "w") as _f:
                            _f.write(str(os.path.getmtime(_recurrent_path)))
                        log(f"  Cycle {cycle}: Saved retrain mtime marker for skip-if-no-new-data")
                    except OSError as _e:
                        log(f"  Cycle {cycle}: Warning: could not save retrain mtime: {_e}")

            gating = _apply_checkpoint_gating(
                cycle=cycle,
                meta_controller=meta_controller,
                retrain_result=retrain_result,
                checkpoint=checkpoint,
                skip_evolution=skip_evolution,
                post_retrain_boost_cycles=post_retrain_boost_cycles,
                merged_retrain_config=merged_retrain_config,
                current_level=evo_result.get("final_level", 0),
                m1_checkpoints=M1_CHECKPOINTS,
                project_root=PROJECT_ROOT,
                parse_val_loss_from_retrain_log=parse_val_loss_from_retrain_log,
                load_checkpoint=load_checkpoint,
                retrain_model=retrain_model,
            )
            checkpoint = gating.checkpoint
            skip_evolution = gating.skip_evolution
            post_retrain_boost_cycles = gating.post_retrain_boost_cycles
            retrain_result = gating.retrain_result

        if meta_controller is not None:
            checkpoint, retrain_result = _update_meta_controller_after_cycle(
                cycle,
                meta_controller,
                evo_result,
                retrain_result,
                checkpoint,
                barrier_tuner,
                PROJECT_ROOT,
                META_CONTROLLER_STATE,
                parse_val_loss_from_retrain_log,
                TelemetrySnapshot,
            )

        should_terminate = False
        if barrier_tuner is not None:
            bt_result = _update_barrier_tuner_after_cycle(
                cycle,
                barrier_tuner,
                evo_result,
                retrain_result,
                archive_params,
                last_archive_params,
                last_retrain_epochs,
                last_champion_fitness,
                last_champion_level,
                last_champion_words,
                PROJECT_ROOT,
                all_results,
                _VAL_LOSS_BT_RE,
                _EPOCH_RE,
                BarrierType,
                CycleMetrics,
                ctx.progress_controller,
            )
            last_archive_params = bt_result.last_archive_params
            last_retrain_epochs = bt_result.last_retrain_epochs
            last_champion_fitness = bt_result.last_champion_fitness
            last_champion_level = bt_result.last_champion_level
            last_champion_words = bt_result.last_champion_words
            should_terminate = bt_result.should_terminate

        else:
            carried = _carry_forward_champion_level(evo_result, all_results, PROJECT_ROOT)
            nudged = _maybe_nudge_champion_level(cycle, PROJECT_ROOT, evo_result, carried, ctx.progress_controller)
            last_champion_level = max(carried, nudged)

        if should_terminate:
            break

        combined = _finalize_cycle(
            cycle,
            evo_result,
            retrain_result,
            _evo_elapsed,
            _retrain_start,
            checkpoint,
            M1_CHECKPOINTS,
            prometheus_server,
            update_metrics,
            compute_forgetting_curve,
            compute_weight_delta,
            PROJECT_ROOT,
            RESULTS_FILE,
            cycles_since_last_retrain,
        )
        all_results.append(combined)

        if os.environ.get("EE_FLATLINE_GUARD", "1") == "1":
            try:
                from evolution.flatline_guard import apply_flatline_halt, assess_flatline

                fl_report = assess_flatline(PROJECT_ROOT)
                if fl_report.halted:
                    apply_flatline_halt(PROJECT_ROOT, fl_report)
                    log(f"FLATLINE advisory at cycle {cycle}: " f"{'; '.join(fl_report.reasons)}")
            except Exception as exc:  # noqa: BLE001
                log(f"Flatline guard skipped: {exc!r}")

        if os.environ.get("EE_EPI_MOMENTUM_GOVERNOR", "1") == "1":
            try:
                from evolution.epi_momentum_governor import run_cycle_governor

                run_cycle_governor(PROJECT_ROOT)
            except Exception as exc:  # noqa: BLE001
                log(f"EPI momentum governor skipped: {exc!r}")

        _maybe_flag_collapse_reseed(PROJECT_ROOT, combined, cycle)
        _maybe_flag_verify_regression_rollback(PROJECT_ROOT, combined, cycle, prev_cycle_metrics)
        _clear_stale_collapse_tick_flags(PROJECT_ROOT, combined)
        if ctx.progress_controller is not None:
            fit_for_pc = float(combined.get("final_fitness", 0.0))
            prom_raw = combined.get("promotion_raw_fitness")
            champ_words = int(combined.get("champion_words", 0))
            champ_verify = float(combined.get("champion_verify_score", 0.0))
            if prom_raw is not None and champ_verify >= 0.99 - 1e-6:
                try:
                    from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor

                    if terminal_efficiency_at_scan_floor(champ_words):
                        fit_for_pc = float(prom_raw)
                except ImportError:
                    pass
            ctx.progress_controller.reconcile_from_cycle_result(
                int(combined.get("final_level", 0)),
                champ_verify,
                fit_for_pc,
            )
        escape_level = _maybe_flag_clean_depth_escape(PROJECT_ROOT, combined, cycle)
        if escape_level is not None:
            last_champion_level = escape_level
        try:
            from evolution.champion_carry_forward import persist_carry_forward_champion

            persist_carry_forward_champion(PROJECT_ROOT, combined)
        except (ImportError, TypeError, ValueError, OSError):
            pass

        ctx.checkpoint = checkpoint
        ctx.last_archive_params = last_archive_params
        ctx.last_retrain_epochs = last_retrain_epochs
        ctx.last_champion_fitness = last_champion_fitness
        ctx.last_champion_level = last_champion_level
        ctx.last_champion_words = last_champion_words
        ctx.cycles_since_last_retrain = cycles_since_last_retrain
        ctx.post_retrain_boost_cycles = post_retrain_boost_cycles
        ctx.prev_cycle_metrics = prev_cycle_metrics
        ctx.prev_archive_state = prev_archive_state
        ctx.skip_evolution = skip_evolution
