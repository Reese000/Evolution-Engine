#!/usr/bin/env python3
"""
R51: Isolated augmentation subprocess.

Runs inject_synthetic_data and inject_reference_solutions in a fresh
interpreter with no CUDA context, preventing ACCESS_VIOLATION crashes
when the parent process has corrupted CUDA state.

Called by loop.py via subprocess.run(). Receives config as JSON on
stdin, outputs augmented sequences as JSON on stdout.
"""
from __future__ import annotations

import json
import os
import sys

# This script lives in the project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# milestone1/ is a subdirectory of the project root
M1_DIR = os.path.join(SCRIPT_DIR, "milestone1")

# Ensure both project root and milestone1 are importable
for p in [SCRIPT_DIR, M1_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> None:
    try:
        config = json.load(sys.stdin)
    except Exception as e:
        json.dump({"error": f"Failed to read config: {e}"}, sys.stdout)
        sys.exit(1)

    seed = config.get("seed", 42)
    max_sequences = config.get("max_sequences", 5000)
    data_path = config.get("data_path", "")

    # The caller (milestone1/training/loop.py) parses this process's ENTIRE
    # stdout as JSON. data.augmentation.inject_* print "[Train] ..." progress to
    # stdout, which corrupted that JSON payload and made the caller's json.loads
    # raise -- silently defeating the R51 isolated-subprocess path (loop.py fell
    # back to in-process augmentation every time, re-doing the work and losing
    # the CUDA-crash isolation R51 exists to provide). Redirect stdout to stderr
    # for the duration of the work so only the final JSON lands on real stdout.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        import torch

        # Load existing sequences from data_path to augment
        sequences: list[torch.Tensor] = []
        sample_weights: list[float] = []

        if data_path and os.path.isfile(data_path):
            try:
                with open(data_path) as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        tokens = entry.get("tokens", [])
                        if tokens:
                            sequences.append(torch.tensor(tokens, dtype=torch.long))
                            sample_weights.append(entry.get("weight", 1.0))
            except Exception as e:
                print(f"[AugmentSubprocess] Warning: could not load {data_path}: {e}", file=sys.stderr)

        # 1. Inject synthetic data
        try:
            from data.augmentation import inject_synthetic_data

            inject_synthetic_data(sequences, sample_weights, seed=seed, max_sequences=max_sequences)
        except Exception as e:
            print(f"[AugmentSubprocess] Synthetic data failed: {e}", file=sys.stderr)

        # 2. Inject reference solutions
        try:
            from data.augmentation import inject_reference_solutions
            from rv32i_model import RV32IFieldTokenizer

            tokenizer = RV32IFieldTokenizer()
            inject_reference_solutions(tokenizer, sequences, sample_weights)
        except Exception as e:
            print(f"[AugmentSubprocess] Reference solutions failed: {e}", file=sys.stderr)

        # Build the augmented sequence payload.
        result = []
        for seq, weight in zip(sequences, sample_weights):
            result.append({"tokens": seq.tolist(), "weight": weight})
        payload = {"sequences": result, "count": len(result)}
    finally:
        # Always restore real stdout before emitting the JSON payload.
        sys.stdout = real_stdout

    json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
