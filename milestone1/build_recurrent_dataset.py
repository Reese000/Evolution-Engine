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
from typing import List, Dict, Optional, Any

# Add paths
M2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'milestone2'))
M1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'milestone1'))
sys.path.insert(0, M2_DIR)
sys.path.insert(0, M1_DIR)

CHAMPIONS_DIR = os.path.join(M2_DIR, 'champions')
RECURRENT_DIR = os.path.join(os.path.dirname(__file__), 'recurrent_data')
os.makedirs(RECURRENT_DIR, exist_ok=True)


def collect_champion_snapshots(engine: Any) -> List[Dict]:
    """Collect champion programs and metadata from the evolution engine.
    
    Returns a list of snapshots, each containing:
    - program: list of instruction words
    - scores: per-level scores
    - generation: generation number
    - code_size: instruction count
    - sha256: integrity hash
    """
    snapshots = []
    if engine.champion is None:
        return snapshots
    
    champion = engine.champion
    snapshot = {
        'program': list(champion.program),
        'scores': dict(champion.scores),
        'generation': champion.generation,
        'code_size': champion.code_size,
        'sha256': champion.sha256,
        'trace_entries': len(engine.champion_trace or []),
        'completed_levels': sorted(list(engine.completed_tasks)),
        'current_level': engine.current_task_level,
        'champion_error': engine.champion_error,
    }
    
    # Include the trace if available (truncate to last 100 entries for size)
    if engine.champion_trace and len(engine.champion_trace) > 0:
        snapshot['trace'] = engine.champion_trace[-100:]
    
    snapshots.append(snapshot)
    return snapshots


def format_training_examples(snapshots: List[Dict]) -> List[Dict]:
    """Format champion snapshots into training examples for the seed model.
    
    Each example has the format expected by train_model.py:
    {
        'input_sequence': [...],     # instruction word sequence
        'label': next_instruction,   # next word prediction target
    }
    
    For champion programs, we create examples where each instruction
    (except the last) is paired with the next instruction as a label.
    """
    examples = []
    for snap in snapshots:
        prog = snap.get('program', [])
        if len(prog) < 2:
            continue
        
        # Create next-instruction prediction examples
        for i in range(len(prog) - 1):
            examples.append({
                'input_sequence': prog[:i+1],
                'label': prog[i+1],
                'source': 'champion',
                'generation': snap.get('generation', -1),
            })
        
        # Also add a full-program example (entire sequence)
        examples.append({
            'input_sequence': prog[:-1],
            'label': prog[-1],
            'source': 'champion_full',
            'generation': snap.get('generation', -1),
        })
    
    return examples


def save_recurrent_data(examples: List[Dict], filename: str = 'recurrent_data.jsonl') -> str:
    """Save training examples to JSONL file."""
    path = os.path.join(RECURRENT_DIR, filename)
    with open(path, 'a') as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(',', ':')) + '\n')
    return path


def get_recurrent_data_path() -> str:
    """Get the path to the accumulated recurrent data file."""
    return os.path.join(RECURRENT_DIR, 'recurrent_data.jsonl')


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
    total = sum(1 for _ in open(path) if _.strip())
    print(f"[RECURRENT] Collected {len(examples)} training examples "
          f"from champion gen={snapshots[0]['generation']} "
          f"(total in dataset: {total})")
    return path


if __name__ == '__main__':
    # Self-test
    print("Recurrent dataset builder self-test:")
    print(f"  RECURRENT_DIR: {RECURRENT_DIR}")
    print(f"  Example data path: {get_recurrent_data_path()}")
    print("  OK")
