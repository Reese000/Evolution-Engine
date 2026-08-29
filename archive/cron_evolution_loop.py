#!/usr/bin/env python3
"""Evolution loop runner - called by cron every 15 minutes.

Checks for stale processes first, then saturation. If either is active,
prints message and exits silently. If clear, runs 10 cycles, checks
saturation again. If saturation triggers mid-loop, stops and exits silently.
"""
import os
import subprocess
import sys
import time

PROJECT = r"C:\Users\reese\Projects\Evolution Engine"


def check_loop_running():
    """Return True if ANY evolution-related Python process is running (excluding this one).

    Checks for processes importing evolution modules or running from the project
    directory. Uses a broader pattern than just PrefixGenerator to catch
    processes that may not import it (e.g., run_multi_cycle.py runners).
    """
    import subprocess as _sp

    for pattern in [
        "evolution_engine",
        "run_multi_cycle",
        "PrefixGenerator",
        "milestone1/training/loop",
        "run_5_cycles",
    ]:
        result = _sp.run(
            ["wsl", "bash", "-c", f"ps aux | grep '{pattern}' | grep -v grep | grep -v cron_evolution | wc -l"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        count = int(result.stdout.strip())
        if count > 0:
            return True
    return False


LOOP_SCRIPT = """
import sys, time
sys.path.insert(0,"{project}")
from collections import Counter
from prefix_generator import PrefixGenerator
from champion_archive import ChampionArchive
from progression_tracker import ProgressionTracker, CycleMetrics

pg = PrefixGenerator("milestone1/checkpoints/checkpoint_best.pt", temperature=1.0, max_new_tokens=32)
arc = ChampionArchive(path="champions/archive.jsonl", max_size=500)
tracker = ProgressionTracker(path="champions/progression.jsonl")

# Check saturation BEFORE starting
snapshot = tracker.emit_snapshot(current_archive_size=arc.size)
if tracker.is_saturated():
    print("SATURATED: Migration ready. Stopping.", flush=True)
    print("DONE", flush=True)
    sys.exit(0)

start_cycle = len(tracker.history) + 1
end_cycle = start_cycle + 10

for cycle in range(start_cycle, end_cycle):
    from evolution_engine import EvolutionConfig, EvolutionEngine
    config = EvolutionConfig()
    config.max_generations = 100
    config.population_size = 25
    config.champion_dir = None
    config.use_gpu = True
    config.auto_curriculum = True
    config.model_guided_ratio = 0.30
    engine = EvolutionEngine(config)
    engine._cycle_number = cycle
    engine._prefix_generator = pg
    t0 = time.time()
    result = engine.run()
    elapsed = time.time() - t0
    counts = Counter(s for s in engine._last_challenger_strategies if s)
    ws = result.get("champion_words", 0)
    fitness = result.get("final_fitness", 0)
    level = result.get("final_level", 0)
    print(f"C{cycle}: L{level:2d} F={fitness:7.1f} W={ws:2d} T={elapsed:5.1f}s", flush=True)
    tracker.record(CycleMetrics(
        cycle=cycle, level=level, fitness=fitness, words=ws,
        archive_size=arc.size, val_loss=None, time_s=elapsed,
        mg_count=counts.get("MODEL_GUIDED", 0),
        total_challengers=sum(counts.values()),
        champion_improvements=result.get("total_champion_improvements", 0)
    ))
    snapshot = tracker.emit_snapshot(current_archive_size=arc.size)
    is_sat = tracker.is_saturated()
    print(f"  Saturation: {is_sat} | Archive: {arc.size}", flush=True)
    if is_sat:
        print("SATURATION TRIGGERED - stopping loop!", flush=True)
        break
    arc.save_recurrent_data("milestone1/recurrent_data.jsonl")
    sys.path.insert(0, "milestone1")
    from training.loop import train
    train(
        resume_path="milestone1/checkpoints/checkpoint_best.pt",
        recurrent_data_path="milestone1/recurrent_data.jsonl",
        max_epochs=15, batch_size=8, amp=False, model_size="medium",
        lr_override=5e-6, max_per_level=200, augmentation_k=10
    )

print(f"Archive: {arc.size} entries", flush=True)
s = tracker.summary()
print(f"Cycles: {s['cycles']}, Best fitness: {s['best_fitness']:.1f}, Trend: {s['fitness_trend']:.2f}", flush=True)
print("DONE", flush=True)
""".replace(
    "{project}", PROJECT
)

# Guard: check for stale processes before starting
if check_loop_running():
    print("Stale evolution process detected. Skipping this tick.", flush=True)
    sys.exit(0)

result = subprocess.run(
    [
        "wsl",
        "bash",
        "-c",
        (f'cd "/mnt/c/Users/reese/Projects/Evolution Engine" && ' f"/root/venv/bin/python -c {LOOP_SCRIPT!r}"),
    ],
    capture_output=True,
    text=True,
    timeout=1200,
)

output = result.stdout + result.stderr
lines = output.strip().split("\n")

# Print last 10 lines of output
for line in lines[-10:]:
    print(line)

# Check if saturated
if any("SATURATED:" in line for line in lines):
    print("System is saturated. Migration ready.")
    sys.exit(0)

if any("SATURATION TRIGGERED" in line for line in lines):
    print("Saturation detected mid-loop. Migration ready.")
    sys.exit(0)

if result.returncode != 0 and result.returncode != 124:
    print(f"Loop exited with code {result.returncode}")
    sys.exit(1)

sys.exit(0)
