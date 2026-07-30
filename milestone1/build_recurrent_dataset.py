#!/usr/bin/env python3
"""
Recurrent Feedback — Champion Trace Dataset Builder

Collects champion programs and execution traces from the evolution engine
and formats them as training data for the seed model fine-tuning pipeline.

Triggered by evolution_engine.py when the curriculum declares STALLED
for 2+ reset cycles.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Add paths
M2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "milestone2"))
M1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "milestone1"))
sys.path.insert(0, M2_DIR)
sys.path.insert(0, M1_DIR)

CHAMPIONS_DIR = os.path.join(M2_DIR, "champions")
# Write directly to milestone1/recurrent_data.jsonl (matches training default)
RECURRENT_FILE = os.path.join(M1_DIR, "recurrent_data.jsonl")
# Legacy directory for backward compat reads
RECURRENT_DIR = os.path.join(os.path.dirname(__file__), "recurrent_data")


def collect_champion_snapshots(engine: Any) -> List[Dict]:
    """Collect champion programs and metadata from the evolution engine.

    Returns a list of snapshots, each containing:
    - program: list of instruction words (ints)
    - fitness: champion fitness score
    - task_score: task score from current level
    - level: current curriculum level
    - generation: generation number
    - code_size: instruction count
    - sha256: integrity hash
    """
    snapshots = []
    if engine.champion is None:
        return snapshots

    champion = engine.champion

    # Extract fitness from engine (various attribute names used across versions)
    fitness = 0.0
    for attr in ("champion_fitness", "best_fitness", "_champion_fitness"):
        if hasattr(engine, attr):
            fitness = getattr(engine, attr, 0.0) or 0.0
            break
    # Fallback: sum of champion scores
    if fitness == 0.0 and hasattr(champion, "scores") and champion.scores:
        fitness = sum(champion.scores.values())

    # Extract task_score from scores dict or engine
    task_score = 0.0
    if hasattr(champion, "scores") and champion.scores:
        current_level = getattr(engine, "current_task_level", 0)
        task_score = champion.scores.get(current_level, 0.0)

    snapshot = {
        "program": list(champion.program),
        "fitness": fitness,
        "task_score": task_score,
        "level": getattr(engine, "current_task_level", 0),
        "generation": champion.generation,
        "code_size": champion.code_size,
        "sha256": champion.sha256,
        "trace_entries": len(engine.champion_trace or []),
        "completed_levels": sorted(list(engine.completed_tasks)),
        "champion_error": engine.champion_error,
    }

    # Include the trace if available (truncate to last 100 entries for size)
    if engine.champion_trace and len(engine.champion_trace) > 0:
        snapshot["trace"] = engine.champion_trace[-100:]

    snapshots.append(snapshot)
    return snapshots


def format_training_examples(snapshots: List[Dict]) -> List[Dict]:
    """Format champion snapshots into training examples.

    Each example has the format expected by data/loading.py:
    {
        'champion_program': [hex(w) for w in program],  # full program as hex strings
        'fitness': float,        # champion fitness
        'task_score': float,     # task score for current level
        'level': int,            # curriculum level
        'generation': int,       # generation number
        'save_reason': str,      # provenance tag
    }
    """
    examples = []
    for snap in snapshots:
        prog = snap.get("program", [])
        if len(prog) < 2:
            continue

        # One example per champion program (full program as hex strings)
        examples.append(
            {
                "champion_program": [hex(w) for w in prog],
                "fitness": snap.get("fitness", 0.0),
                "task_score": snap.get("task_score", 0.0),
                "level": snap.get("level", 0),
                "generation": snap.get("generation", -1),
                "save_reason": "champion_trace",
            }
        )

    return examples


def save_recurrent_data(examples: List[Dict], filename: str = "recurrent_data.jsonl") -> str:
    """Save training examples to JSONL file.

    Writes to milestone1/recurrent_data.jsonl (the canonical path
    that training/loop.py reads from by default).
    """
    # Primary path: milestone1/recurrent_data.jsonl (matches training default)
    path = RECURRENT_FILE
    with open(path, "a") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")

    # Also write to legacy subdirectory for backward compat
    legacy_path = os.path.join(RECURRENT_DIR, filename)
    os.makedirs(RECURRENT_DIR, exist_ok=True)
    try:
        with open(legacy_path, "a") as f:
            for ex in examples:
                f.write(json.dumps(ex, separators=(",", ":")) + "\n")
    except OSError:
        pass  # Legacy path is best-effort

    return path


def get_recurrent_data_path() -> str:
    """Get the path to the accumulated recurrent data file."""
    # Prefer the canonical path, fall back to legacy
    if os.path.isfile(RECURRENT_FILE):
        return RECURRENT_FILE
    legacy = os.path.join(RECURRENT_DIR, "recurrent_data.jsonl")
    if os.path.isfile(legacy):
        return legacy
    return RECURRENT_FILE


def check_ready_for_retrain(min_examples: int = 100) -> bool:
    """Check if enough recurrent data has been accumulated for retraining."""
    path = get_recurrent_data_path()
    if not os.path.exists(path):
        return False
    with open(path) as f:
        count = sum(1 for _ in f)
    return count >= min_examples


def build_and_save(engine: Any) -> Optional[str]:
    """Main entry point: collect champion data, format, save.

    Called by evolution_engine.py when curriculum is STALLED.

    Args:
        engine: The RV32IEvolution instance

    Returns:
        Path to saved data file, or None if nothing collected
    """
    snapshots = collect_champion_snapshots(engine)
    if not snapshots:
        return None

    examples = format_training_examples(snapshots)
    if not examples:
        return None

    path = save_recurrent_data(examples)
    total = 0
    try:
        with open(path) as f:
            total = sum(1 for line in f if line.strip())
    except OSError:
        pass
    print(
        f"[RECURRENT] Collected {len(examples)} training examples "
        f"from champion gen={snapshots[0]['generation']} "
        f"(total in dataset: {total})"
    )
    return path


if __name__ == "__main__":
    # Self-test
    print("Recurrent dataset builder self-test:")
    print(f"  RECURRENT_FILE: {RECURRENT_FILE}")
    print(f"  RECURRENT_DIR (legacy): {RECURRENT_DIR}")
    print(f"  Example data path: {get_recurrent_data_path()}")
    print("  OK")
