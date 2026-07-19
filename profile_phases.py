#!/usr/bin/env python3
"""
GPU Utilization Profiler — instruments one full cycle to measure
wall-time + GPU metrics per phase.

Usage:
    python profile_phases.py [--cycle N] [--gens 10] [--batch-size 8]
    
Measures:
    1. Evolution phase (engine.run) — generation + VM-farm evaluation
    2. Data loading (retrain: traces + augmentation + packing)
    3. Retrain training (forward + backward passes)
    4. Validation (inference only)
    5. Checkpoint save
    
Reports wall-time, GPU memory peak, estimated tokens/sec, and phase fraction.
"""

import argparse
import gc
import json
import os
import sys
import time

# ── Ensure project paths ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
M1_DIR = os.path.join(SCRIPT_DIR, "milestone1")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if M1_DIR not in sys.path:
    sys.path.insert(0, M1_DIR)

import torch


class PhaseTimer:
    """Context manager that records wall-time and GPU memory for a phase."""

    def __init__(self, name: str, gpu_sample_interval: float = 0.1):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.gpu_mem_peak = 0
        self.gpu_samples = []
        self.gpu_sample_interval = gpu_sample_interval
        self._sampling = False
        self._sample_thread = None

    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()
        self.gpu_mem_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        self._sampling = True
        self._start_gpu_sampling()
        return self

    def __exit__(self, *args):
        self._sampling = False
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        if torch.cuda.is_available():
            self.gpu_mem_peak = max(self.gpu_mem_peak, torch.cuda.max_memory_allocated())

    def _start_gpu_sampling(self):
        """Sample GPU memory in background."""
        import threading

        def _sample():
            while self._sampling:
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated()
                    self.gpu_samples.append(mem)
                time.sleep(self.gpu_sample_interval)

        self._sample_thread = threading.Thread(target=_sample, daemon=True)
        self._sample_thread.start()

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time

    @property
    def gpu_mem_avg_mb(self) -> float:
        if not self.gpu_samples:
            return 0.0
        return sum(self.gpu_samples) / len(self.gpu_samples) / (1024 * 1024)

    @property
    def gpu_mem_peak_mb(self) -> float:
        return self.gpu_mem_peak / (1024 * 1024)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "wall_seconds": round(self.elapsed, 2),
            "gpu_mem_peak_mb": round(self.gpu_mem_peak_mb, 1),
            "gpu_mem_avg_mb": round(self.gpu_mem_avg_mb, 1),
            "gpu_samples": len(self.gpu_samples),
        }


def profile_retrain_phases(cycle: int, config: dict) -> dict:
    """Profile the retrain subprocess by instrumenting the training loop phases."""
    import json as _json

    from data.augmentation import inject_reference_solutions, inject_synthetic_data, register_rename_augment
    from data.loading import apply_level_cap, load_champion_traces, rebalance_level_weights
    from data.packing import pad_or_pack_sequences
    from rv32i_mask import decode as rv32i_decode
    from rv32i_model import RV32IFieldTokenizer, compute_loss
    from training.checkpoint import find_existing_checkpoint, save_training_checkpoint
    from training.model_factory import load_or_create_model
    from training.schedule import adjust_dropout, build_optimizer_and_schedulers

    phases = {}

    # ── Phase: Data Loading + Augmentation ─────────────────────────
    with PhaseTimer("data_loading") as t:
        tokenizer = RV32IFieldTokenizer()
        data_path = config.get("recurrent_data", os.path.join(SCRIPT_DIR, "recurrent_data.jsonl"))
        sequences, sample_weights, raw_dicts, entry_levels = load_champion_traces(
            data_path, tokenizer, rv32i_decode=rv32i_decode
        )
        apply_level_cap(sequences, sample_weights, raw_dicts, entry_levels)

        # Register renaming
        augmented_seqs, augmented_weights = [], []
        for instr_dicts in raw_dicts:
            variants = register_rename_augment(instr_dicts, k=8)
            for v in variants:
                tokens = tokenizer.encode_trace(v)
                if tokens and len(tokens) >= 2:
                    augmented_seqs.append(torch.tensor(tokens, dtype=torch.long))
                    augmented_weights.append(sample_weights[0] if sample_weights else 1.0)
        sequences.extend(augmented_seqs)
        sample_weights.extend(augmented_weights)
        del raw_dicts, augmented_seqs, augmented_weights

        # Synthetic + reference injection (subprocess)
        aug_script = os.path.join(SCRIPT_DIR, "augmentation_subprocess.py")
        if os.path.isfile(aug_script):
            import subprocess as _sub

            aug_input = _json.dumps({"seed": 42, "max_sequences": 5000, "data_path": data_path})
            result = _sub.run(
                [sys.executable, aug_script],
                input=aug_input,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=SCRIPT_DIR,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Find JSON in stdout (may have stderr mixed in)
                _stdout = result.stdout.strip()
                _json_start = _stdout.find("{")
                if _json_start >= 0:
                    aug_data = _json.loads(_stdout[_json_start:])
                else:
                    aug_data = {}
                for entry in aug_data.get("sequences", []):
                    tokens = entry.get("tokens", [])
                    if tokens:
                        sequences.append(torch.tensor(tokens, dtype=torch.long))
                        sample_weights.append(entry.get("weight", 1.0))

        full_entry_levels = entry_levels + [0] * (len(sequences) - len(entry_levels))
        rebalance_level_weights(sample_weights, full_entry_levels)

        split = pad_or_pack_sequences(
            sequences,
            sample_weights,
            max_seq_len=1024,
            val_split=config.get("val_split", 0.15),
            device=config.get("device", torch.device("cuda")),
        )
    phases["data_loading"] = t.to_dict()
    phases["data_loading"]["n_sequences"] = len(sequences)

    # ── Phase: Model Creation ─────────────────────────────────────
    with PhaseTimer("model_creation") as t:
        ckpt_dir = os.path.join(M1_DIR, "checkpoints")
        ckpt = find_existing_checkpoint(ckpt_dir)
        model, _ = load_or_create_model(
            ckpt,
            model_size=config.get("model_size", "xlarge"),
            device=config.get("device", torch.device("cuda")),
        )
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    phases["model_creation"] = t.to_dict()
    phases["model_creation"]["n_params"] = sum(p.numel() for p in model.parameters())

    # ── Phase: Training ───────────────────────────────────────────
    batch_size = config.get("batch_size", 8)
    grad_accum = config.get("grad_accum_steps", 2)
    max_epochs = config.get("max_epochs", 8)
    use_amp = config.get("amp", True) and config.get("device", torch.device("cuda")).type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_data = split["train_data"]
    train_mask = split["train_loss_mask"]
    train_weights = split["train_weights"]
    n_batches = max(1, len(train_data) // batch_size)

    lr = config.get("lr", 3e-5)
    sched = build_optimizer_and_schedulers(
        model,
        lr,
        config.get("weight_decay", 0.01),
        max_epochs,
        n_batches,
    )
    optimizer = sched["optimizer"]
    scheduler = sched["scheduler"]
    total_tokens = 0

    training_phases = []
    for epoch in range(max_epochs):
        with PhaseTimer(f"train_epoch_{epoch}") as t:
            adjust_dropout(model, epoch, max_epochs)
            teacher_prob = max(0.3, 1.0 - (epoch / max_epochs) * 0.7)
            noise_std = 0.02 * max(0.0, 1.0 - epoch / max_epochs)
            model.train()
            epoch_tokens = 0

            for i in range(0, len(train_data), batch_size):
                batch = train_data[i : i + batch_size]
                b_mask = train_mask[i : i + batch_size] if train_mask is not None else None
                b_weights = train_weights[i : i + batch_size] if train_weights is not None else None

                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = compute_loss(
                        model,
                        batch,
                        boundary_mask=b_mask,
                        sample_weights=b_weights,
                    )
                    loss = loss / grad_accum

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (len(training_phases) * n_batches + i // batch_size + 1) % grad_accum == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                epoch_tokens += (
                    sum(b.shape[0] * b.shape[1] for b in batch)
                    if isinstance(batch, list)
                    else batch.shape[0] * batch.shape[1]
                )

            total_tokens += epoch_tokens
        training_phases.append(t.to_dict())
        training_phases[-1]["epoch"] = epoch + 1
        training_phases[-1]["tokens"] = epoch_tokens

    phases["training"] = {
        "wall_seconds": sum(p["wall_seconds"] for p in training_phases),
        "gpu_mem_peak_mb": max(p["gpu_mem_peak_mb"] for p in training_phases),
        "gpu_mem_avg_mb": sum(p["gpu_mem_avg_mb"] for p in training_phases) / len(training_phases),
        "total_tokens": total_tokens,
        "epochs": len(training_phases),
        "per_epoch": training_phases,
    }

    # ── Phase: Validation ─────────────────────────────────────────
    val_data = split["val_data"]
    val_mask = split["val_loss_mask"]
    val_weights = split["val_weights"]
    with PhaseTimer("validation") as t:
        model.eval()
        vloss = 0.0
        vn = 0
        val_tokens = 0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i : i + batch_size]
                b_mask = val_mask[i : i + batch_size] if val_mask is not None else None
                b_weights = val_weights[i : i + batch_size] if val_weights is not None else None
                vl = compute_loss(model, batch, boundary_mask=b_mask, sample_weights=b_weights)
                vloss += vl.item()
                vn += 1
                val_tokens += (
                    sum(b.shape[0] * b.shape[1] for b in batch)
                    if isinstance(batch, list)
                    else batch.shape[0] * batch.shape[1]
                )
    phases["validation"] = t.to_dict()
    phases["validation"]["val_tokens"] = val_tokens
    phases["validation"]["val_loss"] = vloss / max(1, vn)

    # ── Phase: Checkpoint Save ────────────────────────────────────
    with PhaseTimer("checkpoint_save") as t:
        save_training_checkpoint(
            model,
            vloss / max(1, vn),
            0,
            max_epochs,
            len(sequences),
            ckpt_dir,
            config_name=config.get("model_size", "xlarge"),
        )
    phases["checkpoint_save"] = t.to_dict()

    # ── Cleanup ───────────────────────────────────────────────────
    del model, optimizer, scheduler
    if scaler:
        del scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Compute summary stats ─────────────────────────────────────
    total_wall = sum(v["wall_seconds"] for v in phases.values() if isinstance(v, dict) and "wall_seconds" in v)
    train_wall = phases["training"]["wall_seconds"]
    total_tok = phases["training"]["total_tokens"]
    tokens_per_sec = total_tok / train_wall if train_wall > 0 else 0

    peak_mem = max(
        (v.get("gpu_mem_peak_mb", 0) for v in phases.values() if isinstance(v, dict)),
        default=0,
    )

    # GPU utilization estimate: based on time CUDA is actually doing work
    # If tokens/sec * time_per_token < peak theoretical, we're underutilized
    summary = {
        "total_wall_seconds": round(total_wall, 2),
        "train_wall_seconds": round(train_wall, 2),
        "train_fraction": round(train_wall / total_wall, 2) if total_wall > 0 else 0,
        "total_tokens": total_tok,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "gpu_peak_mb": round(peak_mem, 1),
        "gpu_total_mb": 12282,
        "gpu_utilization_estimate": round(peak_mem / 12282 * 100, 1),
        "n_sequences": phases["data_loading"]["n_sequences"],
        "n_params": phases["model_creation"]["n_params"],
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "val_loss": phases["validation"]["val_loss"],
    }

    return {"phases": phases, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Profile GPU utilization per phase")
    parser.add_argument("--cycle", type=int, default=1, help="Cycle number")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--model-size", type=str, default="xlarge")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Profile] Device: {device}")
    print(
        f"[Profile] Batch={args.batch_size}, GradAccum={args.grad_accum}, "
        f"Effective={args.batch_size * args.grad_accum}, Epochs={args.max_epochs}"
    )
    print(f"[Profile] Model: {args.model_size}")
    if torch.cuda.is_available():
        print(f"[Profile] GPU: {torch.cuda.get_device_name()}")
        print(f"[Profile] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")

    config = {
        "device": device,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "max_epochs": args.max_epochs,
        "model_size": args.model_size,
        "amp": True,
        "lr": 3e-5,
        "weight_decay": 0.01,
        "val_split": 0.15,
        "recurrent_data": os.path.join(SCRIPT_DIR, "recurrent_data.jsonl"),
    }

    print(f"\n[Profile] === Starting profiling cycle {args.cycle} ===\n")
    result = profile_retrain_phases(args.cycle, config)

    # Print phase breakdown
    print("\n[Profile] === PHASE BREAKDOWN ===\n")
    phases = result["phases"]
    total = result["summary"]["total_wall_seconds"]
    for name, data in phases.items():
        if not isinstance(data, dict) or "wall_seconds" not in data:
            continue
        wall = data["wall_seconds"]
        frac = wall / total * 100 if total > 0 else 0
        peak = data.get("gpu_mem_peak_mb", 0)
        print(f"  {name:25s}: {wall:7.1f}s ({frac:5.1f}%)  GPU peak: {peak:7.0f} MB")

    s = result["summary"]
    print("\n[Profile] === SUMMARY ===")
    print(f"  Total wall time:       {s['total_wall_seconds']:.1f}s")
    print(f"  Training wall time:    {s['train_wall_seconds']:.1f}s ({s['train_fraction']*100:.0f}%)")
    print(f"  Total tokens:          {s['total_tokens']:,}")
    print(f"  Tokens/sec:            {s['tokens_per_sec']:.0f}")
    print(
        f"  GPU peak:              {s['gpu_peak_mb']:.0f} / {s['gpu_total_mb']} MB ({s['gpu_utilization_estimate']:.1f}%)"
    )
    print(f"  Sequences:             {s['n_sequences']}")
    print(f"  Model params:          {s['n_params']:,}")
    print(f"  Effective batch:       {s['effective_batch']}")
    print(f"  Val loss:              {s['val_loss']:.4f}")

    # Save to file
    out_path = args.output or os.path.join(SCRIPT_DIR, "profile_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Profile] Results saved to {out_path}")


if __name__ == "__main__":
    main()
