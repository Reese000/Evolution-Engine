# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Training pipeline for an RV32I instruction set transformer model. Trains next-token prediction on champion traces from the Evolution Engine to learn instruction patterns.

## Commands

```bash
python build_recurrent_dataset.py   # Build training dataset from traces
python train_model_engine.py        # Standalone training CLI (see --help)
python weight_validator.py          # Validate trained weights
```

Note: `train_model.py` is an import-only facade (re-exports `train`,
`register_rename_augment`, `pack_sequences`) with no CLI — running it directly
does nothing. Retrain normally runs in-process via the engine runners; use
`train_model_engine.py` for standalone training.

## Structure

- `rv32i_model.py` — RV32I transformer model definition
- `train_model.py` — Import-only facade re-exporting the training pipeline API
- `train_model_engine.py` — Standalone training CLI (argparse entry point)
- `build_recurrent_dataset.py` — Dataset builder from EE champion traces
- `weight_validator.py` — Weight validation utility
- `checkpoints/` — Model checkpoints
- `recurrent_data/` — Processed training data
