# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Training pipeline for an RV32I instruction set transformer model. Trains next-token prediction on champion traces from the Evolution Engine to learn instruction patterns.

## Commands

```bash
python build_recurrent_dataset.py   # Build training dataset from traces
python train_model.py               # Run training pipeline
python train_model_engine.py        # Engine-integrated training
python weight_validator.py          # Validate trained weights
```

## Structure

- `rv32i_model.py` — RV32I transformer model definition
- `train_model.py` — Main training pipeline (v51 retrain)
- `build_recurrent_dataset.py` — Dataset builder from EE champion traces
- `weight_validator.py` — Weight validation utility
- `checkpoints/` — Model checkpoints
- `recurrent_data/` — Processed training data
