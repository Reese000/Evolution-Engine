#!/usr/bin/env python3
"""Cron job: Run 10 evolution cycles, check saturation, trigger migration if needed."""
import json
import os
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
from paths import ensure_project_on_path

ensure_project_on_path()

from logging_config import get_logger
from run_multi_cycle import (
    _HAS_META_CONTROLLER,
    M1_CHECKPOINTS,
    META_CONTROLLER_STATE,
    PROJECT_ROOT,
    RESULTS_FILE,
    load_checkpoint,
    merge_archive_to_recurrent,
    retrain_model,
    run_evolution,
)

_logger = get_logger("CronRun10")


def clog(msg):
    _logger.info(msg)
    print(msg, flush=True)


def find_latest_state():
    """Find the latest cycle with archive_state across all result files."""
    best = None
    for fname in ["multi_cycle_results.jsonl", "multi_cycle_results_archive.jsonl"]:
        fpath = os.path.join(PROJECT_ROOT, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    if r.get("archive_state") and (best is None or r["cycle"] > best["cycle"]):
                        best = r
                except json.JSONDecodeError:
                    pass
    return best


def win_to_wsl(path):
    """Convert Windows path to WSL path."""
    if not path:
        return path
    # Already a WSL path
    if path.startswith("/"):
        return path
    # Convert C:\Users\... to /mnt/c/Users/...
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return path


def check_saturation(champion_archive_size, fitness_history, val_loss_history, level_history):
    """Run the saturation detector to check if we should scale."""
    try:
        from milestone1.training.scaling.saturation_detector import SaturationDetector
        from milestone1.training.scaling.types import TrendSnapshot

        detector = SaturationDetector()
        snapshot = TrendSnapshot(
            fitness_history=fitness_history,
            val_loss_history=val_loss_history,
            max_level_history=level_history,
            archive_size=champion_archive_size,
            cycles_observed=len(fitness_history),
        )
        report = detector.evaluate(snapshot)
        return report
    except (RuntimeError, ValueError, KeyError) as e:
        clog(f"  Saturation check failed: {e}")
        return None


def _load_models_from_checkpoint(checkpoint_path):
    """Load old and new model configs from checkpoint."""

    from milestone1.rv32i_model import RV32IConfig, RV32IModel

    ckpt = load_checkpoint(checkpoint_path)
    old_config_dict = {
        "d_model": ckpt.get("config", {}).get("d_model", 256),
        "n_layers": ckpt.get("config", {}).get("n_layers", 8),
        "n_heads": ckpt.get("config", {}).get("n_heads", 8),
        "d_ff": ckpt.get("config", {}).get("d_ff", 1024),
    }
    new_config_dict = {"d_model": 512, "n_layers": 14, "n_heads": 8, "d_ff": 2048}
    old_cfg = RV32IConfig(
        d_model=old_config_dict["d_model"],
        n_layers=old_config_dict["n_layers"],
        n_heads=old_config_dict["n_heads"],
        d_ff=old_config_dict["d_ff"],
    )
    old_model = RV32IModel(old_cfg)
    old_model.load_state_dict(ckpt["model_state_dict"])
    old_model.eval()
    new_cfg = RV32IConfig(
        d_model=new_config_dict["d_model"],
        n_layers=new_config_dict["n_layers"],
        n_heads=new_config_dict["n_heads"],
        d_ff=new_config_dict["d_ff"],
    )
    new_model = RV32IModel(new_cfg)
    new_model.eval()
    return old_model, new_model, old_config_dict, new_config_dict


def _save_migrated_checkpoint(new_model, new_config_dict, old_config_dict, migrate_dir):
    """Save migrated model checkpoint files."""
    import torch

    new_ckpt = {
        "model_state_dict": new_model.state_dict(),
        "config": new_config_dict,
        "n_layers": new_config_dict["n_layers"],
        "d_model": new_config_dict["d_model"],
        "migration_from": old_config_dict,
        "migration_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    new_ckpt_path = os.path.join(migrate_dir, "checkpoint_migrated_14L_512d.pt")
    torch.save(new_ckpt, new_ckpt_path)
    clog(f"  Saved migrated checkpoint: {new_ckpt_path}")
    new_best_path = os.path.join(migrate_dir, "checkpoint_best.pt")
    torch.save(new_ckpt, new_best_path)
    clog("  Updated checkpoint_best.pt with migrated model")


def run_migration(checkpoint_path):
    """Run the full model migration pipeline to 14L/512d."""
    clog("\n" + "=" * 60)
    clog("MIGRATION PIPELINE: Scaling to 14L/512d")
    clog("=" * 60)
    try:
        from milestone1.training.scaling.model_migrator import ModelMigrator

        old_model, new_model, old_config_dict, new_config_dict = _load_models_from_checkpoint(checkpoint_path)
        clog(f"  Old config: {old_config_dict}")
        clog(f"  New config: {new_config_dict}")
        clog(f"  Old model: {sum(p.numel() for p in old_model.parameters()):,} params")
        clog(f"  New model: {sum(p.numel() for p in new_model.parameters()):,} params")

        migrator = ModelMigrator()
        result = migrator.migrate(old_model, new_model, old_config_dict, new_config_dict)
        if not result.success:
            clog(f"  Migration FAILED: {result.message}")
            return False
        clog(f"  Migration: {result.message}")
        migrate_dir = os.path.join(PROJECT_ROOT, "milestone1", "checkpoints")
        _save_migrated_checkpoint(new_model, new_config_dict, old_config_dict, migrate_dir)
        return True
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        clog(f"  Migration error: {e}")
        traceback.print_exc()
        return False


def _find_starting_point():
    """Find the starting point: (start_cycle, prev_archive_state, last_checkpoint)."""
    last_state = find_latest_state()
    if last_state:
        start_cycle = last_state["cycle"] + 1
        prev_archive_state = last_state.get("archive_state")
        last_checkpoint = win_to_wsl(last_state.get("checkpoint", ""))
        clog(f"Resuming from cycle {start_cycle} (prev archive_state from cycle {last_state['cycle']})")
    else:
        start_cycle = 1
        prev_archive_state = None
        candidates = [
            os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt"),
            os.path.join(M1_CHECKPOINTS, "checkpoint_current.pt"),
        ]
        last_checkpoint = ""
        for c in candidates:
            if os.path.isfile(c):
                last_checkpoint = c
                break
        clog("Starting fresh from cycle 1")
    if not last_checkpoint or not os.path.isfile(last_checkpoint):
        for c in [
            os.path.join(M1_CHECKPOINTS, "checkpoint_best.pt"),
            os.path.join(M1_CHECKPOINTS, "checkpoint_current.pt"),
        ]:
            if os.path.isfile(c):
                last_checkpoint = c
                break
    if not last_checkpoint or not os.path.isfile(last_checkpoint):
        clog(f"FATAL: No checkpoint found (checked: {last_checkpoint})")
        sys.exit(1)
    return start_cycle, prev_archive_state, last_checkpoint


def _execute_single_cycle(cycle, num_gens, last_checkpoint, prev_archive_state, meta_controller):
    """Run one evolution cycle. Returns (result, new_checkpoint, evo_metrics) or None on error."""
    evo_metrics = run_evolution(
        cycle=cycle,
        num_generations=num_gens,
        checkpoint_path=last_checkpoint,
        archive_data=prev_archive_state,
    )
    if "error" in evo_metrics:
        clog(f"  Cycle {cycle}: Evolution error: {evo_metrics['error']}")
        return None
    merge_archive_to_recurrent(cycle)
    retrain_result = retrain_model(
        cycle=cycle,
        checkpoint_path=last_checkpoint,
        best_level=evo_metrics.get("final_level", 0),
    )
    val_loss = retrain_result.get("val_loss", None)
    new_checkpoint = retrain_result.get("checkpoint", last_checkpoint)
    retrained = retrain_result.get("retrained", False)
    if meta_controller and val_loss is not None and val_loss > 0:
        accepted = meta_controller.accept_checkpoint(val_loss, new_checkpoint)
        if not accepted:
            clog(f"  Cycle {cycle}: Rejected checkpoint (val_loss={val_loss:.4f})")
            new_checkpoint = last_checkpoint
    result = {
        "cycle": cycle,
        "generations_completed": evo_metrics["generations_completed"],
        "final_fitness": evo_metrics["final_fitness"],
        "final_level": evo_metrics["final_level"],
        "champion_words": evo_metrics["champion_words"],
        "elapsed_seconds": evo_metrics["elapsed_seconds"],
        "champion_traces": evo_metrics["champion_traces"],
        "novelty_archive_size": evo_metrics["novelty_archive_size"],
        "qd_occupied_cells": evo_metrics["qd_occupied_cells"],
        "qd_total_cells": evo_metrics["qd_total_cells"],
        "checkpoint": new_checkpoint,
        "retrained": retrained,
        "archive_state": evo_metrics.get("archive_state"),
    }
    if val_loss is not None:
        result["val_loss"] = val_loss
    if retrain_result.get("exit_code") is not None:
        result["exit_code"] = retrain_result["exit_code"]
    if retrain_result.get("example_count") is not None:
        result["example_count"] = retrain_result["example_count"]
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")
    return result, new_checkpoint, evo_metrics


def _update_meta_controller(meta_controller, cycle, evo_metrics, val_loss, retrained):
    """Update MetaController with telemetry snapshot."""
    from meta_controller import TelemetrySnapshot

    snapshot = TelemetrySnapshot(
        cycle=cycle,
        fitness=evo_metrics["final_fitness"],
        level=evo_metrics["final_level"],
        champion_words=evo_metrics["champion_words"],
        val_loss=val_loss,
        diversity=evo_metrics.get("diversity", 0.5),
        novelty_size=evo_metrics["novelty_archive_size"],
        qd_occupied=evo_metrics["qd_occupied_cells"],
        qd_total=evo_metrics["qd_total_cells"],
        traces_count=evo_metrics["champion_traces"],
        retrained=retrained,
        generations_completed=evo_metrics["generations_completed"],
    )
    meta_controller.update_cycle(snapshot)
    meta_controller.save_state(META_CONTROLLER_STATE)


def _print_final_summary(
    completed_cycles,
    archive_size,
    saturation_triggered,
    last_checkpoint,
    fitness_history,
    level_history,
    val_loss_history,
):
    """Print final summary and save summary JSON."""
    clog(f"\n{'=' * 60}")
    clog("FINAL SUMMARY")
    clog(f"{'=' * 60}")
    clog(f"  Cycles completed this run: {len(completed_cycles)}")
    clog(f"  Champion archive size: {archive_size}")
    if completed_cycles:
        clog(f"  Latest fitness: {completed_cycles[-1]['final_fitness']:.3f}")
        clog(f"  Latest level: {completed_cycles[-1]['final_level']}")
        clog(f"  Fitness history: {[f'{f:.1f}' for f in fitness_history[-10:]]}")
        clog(f"  Level history: {level_history[-10:]}")
    if saturation_triggered:
        clog("\nSaturation detected — triggering model migration to 14L/512d")
        success = run_migration(last_checkpoint)
        if success:
            clog("Migration completed successfully!")
        else:
            clog("Migration failed!")
    else:
        clog(f"\nSaturation not triggered yet (need archive >= 200, currently {archive_size})")
    report = check_saturation(archive_size, fitness_history, val_loss_history, level_history)
    if report:
        clog(f"\nFinal saturation: {report.criteria_met()}/4 criteria, confidence={report.confidence:.2f}")
        clog(f"  should_scale: {report.should_scale}")
    summary = {
        "total_cycles": len(completed_cycles),
        "latest_fitness": completed_cycles[-1]["final_fitness"] if completed_cycles else 0,
        "latest_level": completed_cycles[-1]["final_level"] if completed_cycles else 0,
        "archive_size": archive_size,
        "saturation_triggered": saturation_triggered,
        "fitness_history": fitness_history[-20:],
        "level_history": level_history[-20:],
        "val_loss_history": val_loss_history[-20:],
    }
    with open(os.path.join(PROJECT_ROOT, "multi_cycle_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    clog("\nDone!")


def _init_meta_controller():
    """Initialize MetaController if available."""
    if not _HAS_META_CONTROLLER:
        return None
    from meta_controller import MetaController

    mc = MetaController(
        checkpoint_dir=M1_CHECKPOINTS,
        log_file=os.path.join(PROJECT_ROOT, "meta_controller.log"),
    )
    if os.path.isfile(META_CONTROLLER_STATE):
        mc.load_state(META_CONTROLLER_STATE)
    return mc


def _log_cycle_complete(cycle, evo_metrics, val_loss, result, archive_size):
    """Log cycle completion summary."""
    clog(
        f"  Cycle {cycle} complete: fitness={evo_metrics['final_fitness']:.3f}, "
        f"level={evo_metrics['final_level']}, "
        f"novelty={evo_metrics['novelty_archive_size']}, "
        f"qd={evo_metrics['qd_occupied_cells']}/{evo_metrics['qd_total_cells']}, "
        f"val_loss={val_loss}, retrained={result['retrained']}, "
        f"archive_size={archive_size}"
    )


def _check_and_log_saturation(archive_size, fitness_history, val_loss_history, level_history):
    """Check saturation, log results, return True if should scale."""
    report = check_saturation(archive_size, fitness_history, val_loss_history, level_history)
    if not report:
        return False
    clog(
        f"  Saturation: {report.criteria_met()}/4 criteria met "
        f"(fitness_plateau={report.fitness_plateau}, "
        f"val_loss_plateau={report.val_loss_plateau}, "
        f"curriculum_saturated={report.curriculum_saturated}, "
        f"archive_sufficient={report.archive_sufficient})"
    )
    if report.should_scale:
        clog("  *** SATURATION TRIGGERED — should_scale=True ***")
        return True
    return False


def main():
    NUM_CYCLES = 10
    NUM_GENS = 100

    start_cycle, prev_archive_state, last_checkpoint = _find_starting_point()
    clog(f"Checkpoint: {last_checkpoint}")

    fitness_history = []
    val_loss_history = []
    level_history = []

    archive_path = os.path.join(PROJECT_ROOT, "champions", "archive.jsonl")
    archive_size = 0
    if os.path.isfile(archive_path):
        with open(archive_path) as f:
            archive_size = sum(1 for line in f if line.strip())
    clog(f"Starting archive size: {archive_size}")

    meta_controller = _init_meta_controller()
    completed_cycles = []
    saturation_triggered = False

    for cycle in range(start_cycle, start_cycle + NUM_CYCLES):
        clog(f"\n{'=' * 60}")
        clog(f"CYCLE {cycle}/{start_cycle + NUM_CYCLES - 1}")
        clog(f"{'=' * 60}")
        try:
            result = _execute_single_cycle(cycle, NUM_GENS, last_checkpoint, prev_archive_state, meta_controller)
            if result is None:
                continue
            result, new_checkpoint, evo_metrics = result
            completed_cycles.append(result)
            last_checkpoint = new_checkpoint
            prev_archive_state = evo_metrics.get("archive_state")
            fitness_history.append(evo_metrics["final_fitness"])
            level_history.append(evo_metrics["final_level"])
            val_loss = result.get("val_loss", None)
            if val_loss is not None:
                val_loss_history.append(val_loss)
            if os.path.isfile(archive_path):
                with open(archive_path) as f:
                    archive_size = sum(1 for line in f if line.strip())
            _log_cycle_complete(cycle, evo_metrics, val_loss, result, archive_size)
            if meta_controller and val_loss is not None:
                _update_meta_controller(meta_controller, cycle, evo_metrics, val_loss, result["retrained"])
            if _check_and_log_saturation(archive_size, fitness_history, val_loss_history, level_history):
                saturation_triggered = True
                break
        except (RuntimeError, ValueError, KeyError, OSError) as e:
            clog(f"  Cycle {cycle} CRASHED: {e}")
            traceback.print_exc()
            continue

    _print_final_summary(
        completed_cycles,
        archive_size,
        saturation_triggered,
        last_checkpoint,
        fitness_history,
        level_history,
        val_loss_history,
    )


if __name__ == "__main__":
    main()
