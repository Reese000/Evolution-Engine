#!/usr/bin/env python3
"""Evolution loop runner — called by cron every 15 minutes."""
import os
import subprocess
import sys

PROJECT = r"C:\Users\reese\Projects\Evolution Engine"


def check_loop_running():
    """Check if an evolution loop is already running."""
    result = subprocess.run(
        ["wsl", "bash", "-c", "ps aux | grep PrefixGenerator | grep -v grep | wc -l"], capture_output=True, text=True
    )
    return int(result.stdout.strip()) > 0


def check_saturation():
    """Check if saturation has been triggered."""
    script = """
import sys; sys.path.insert(0,".")
from progression_tracker import ProgressionTracker
tr = ProgressionTracker(path="champions/progression.jsonl")
print(tr.is_saturated())
"""
    result = subprocess.run(
        [
            "wsl",
            "bash",
            "-c",
            f'cd "/mnt/c/Users/reese/Projects/Evolution Engine" && /root/venv/bin/python -c "{script}"',
        ],
        capture_output=True,
        text=True,
    )
    return "True" in result.stdout


def run_loop():
    """Run a 10-cycle evolution loop."""
    script = """
import sys, time
sys.path.insert(0,".")
from collections import Counter
from prefix_generator import PrefixGenerator
from champion_archive import ChampionArchive
from progression_tracker import ProgressionTracker, CycleMetrics

pg = PrefixGenerator("milestone1/checkpoints/checkpoint_best.pt", temperature=1.0, max_new_tokens=32)
arc = ChampionArchive(path="champions/archive.jsonl", max_size=500)
tracker = ProgressionTracker(path="champions/progression.jsonl")
start_cycle = len(tracker.history) + 1
end_cycle = start_cycle + 10

for cycle in range(start_cycle, end_cycle):
    from evolution_engine import EvolutionConfig, EvolutionEngine
    config = EvolutionConfig(); config.max_generations=100; config.population_size=25
    config.champion_dir=None; config.use_gpu=True; config.auto_curriculum=True
    config.model_guided_ratio=0.30
    engine = EvolutionEngine(config); engine._cycle_number=cycle; engine._prefix_generator=pg
    t0=time.time(); result=engine.run(); elapsed=time.time()-t0
    counts=Counter(s for s in engine._last_challenger_strategies if s)
    total=sum(counts.values()); ws=result.get("champion_words",0); fitness=result.get("final_fitness",0)
    level=result.get("final_level",0)
    print(f"C{cycle}: L{level:2d} F={fitness:7.1f} W={ws:2d} T={elapsed:5.1f}s", flush=True)
    tracker.record(CycleMetrics(cycle=cycle, level=level, fitness=fitness, words=ws,
        archive_size=arc.size, val_loss=None, time_s=elapsed,
        mg_count=counts.get("MODEL_GUIDED",0), total_challengers=total,
        champion_improvements=result.get("total_champion_improvements",0)))
    snapshot = tracker.emit_snapshot(current_archive_size=arc.size)
    is_sat = tracker.is_saturated()
    print(f"  Saturation: {is_sat} | Archive: {arc.size}", flush=True)
    if is_sat:
        print("SATURATION DETECTED", flush=True)
        break
    arc.save_recurrent_data("milestone1/recurrent_data.jsonl")
    sys.path.insert(0,"milestone1")
    from training.loop import train
    train(resume_path="milestone1/checkpoints/checkpoint_best.pt", recurrent_data_path="milestone1/recurrent_data.jsonl",
        max_epochs=15, batch_size=8, amp=False, model_size="medium", lr_override=5e-6, max_per_level=200, augmentation_k=10)
print(f"Archive: {arc.size} entries", flush=True)
print(f"Tracker: {tracker.summary()}", flush=True)
print("DONE", flush=True)
"""
    result = subprocess.run(
        [
            "wsl",
            "bash",
            "-c",
            f'cd "/mnt/c/Users/reese/Projects/Evolution Engine" && /root/venv/bin/python -c "{script}"',
        ],
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min timeout
    )
    return result.stdout


def main():
    print("=== Evolution Loop Runner ===")

    if check_loop_running():
        print("Loop already running, skipping.")
        return

    if check_saturation():
        print("SATURATION TRIGGERED — ScaleCoordinator will handle migration.")
        return

    print("Starting 10-cycle evolution loop...")
    output = run_loop()
    print(output)

    # Check saturation after loop
    if check_saturation():
        print("SATURATION TRIGGERED after loop — migration imminent.")


if __name__ == "__main__":
    main()
