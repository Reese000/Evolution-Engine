#!/usr/bin/env python3
"""
GPU Micro-Benchmark: isolates forward/backward/optim from data loading overhead.
Measures pure GPU compute throughput at various batch sizes.
Also logs power draw via nvidia-smi for honest saturation metric.
"""
import gc
import os
import subprocess
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
M1_DIR = os.path.join(SCRIPT_DIR, "milestone1")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if M1_DIR not in sys.path:
    sys.path.insert(0, M1_DIR)

from rv32i_model import RV32IFieldTokenizer, compute_loss
from training.model_factory import load_or_create_model


def get_power_draw():
    """Get current GPU power draw in watts."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def get_gpu_util():
    """Get GPU utilization %."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(r.stdout.strip())
    except Exception:
        return 0


def benchmark_batch(batch_size, seq_len, n_steps=50, warmup=10, label=""):
    """Benchmark pure forward+backward+optim at given batch size."""
    device = torch.device("cuda")

    # Create synthetic batch: random token sequences
    # Mimic real data: seq_len tokens, vocab ~268
    vocab_size = 268
    batch = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    boundary_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    weights = torch.ones(batch_size, device=device)

    # Create fresh model each run
    model, _ = load_or_create_model(None, model_size="xlarge", device=device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler("cuda")

    # Warmup
    for _ in range(warmup):
        with torch.amp.autocast("cuda", enabled=True):
            loss = compute_loss(model, batch, boundary_mask=boundary_mask, sample_weights=weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Synchronize before timing
    torch.cuda.synchronize()

    # Sample power before
    power_before = get_power_draw()

    # Timed run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    gpu_mems = []
    powers = []
    for i in range(n_steps):
        mem_before = torch.cuda.memory_allocated()

        start_events[i].record()
        with torch.amp.autocast("cuda", enabled=True):
            loss = compute_loss(model, batch, boundary_mask=boundary_mask, sample_weights=weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        end_events[i].record()

        gpu_mems.append(torch.cuda.max_memory_allocated())
        powers.append(get_power_draw())

    torch.cuda.synchronize()
    total_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events))

    avg_ms = total_ms / n_steps
    tokens_per_step = batch_size * seq_len
    tok_per_sec = tokens_per_step / (avg_ms / 1000)
    avg_power = sum(powers) / len(powers) if powers else 0
    peak_power = max(powers) if powers else 0
    peak_mem = max(gpu_mems) / (1024 * 1024) if gpu_mems else 0
    gpu_util = get_gpu_util()

    # Cleanup
    del model, optimizer, scaler, batch, boundary_mask, weights
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "label": label,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "effective_batch": batch_size,
        "avg_ms_per_step": round(avg_ms, 2),
        "tokens_per_step": tokens_per_step,
        "tokens_per_sec": round(tok_per_sec, 0),
        "peak_vram_mb": round(peak_mem, 1),
        "avg_power_w": round(avg_power, 1),
        "peak_power_w": round(peak_power, 1),
        "gpu_util_pct": gpu_util,
        "n_steps": n_steps,
    }


def main():
    print("=" * 70)
    print("GPU MICRO-BENCHMARK — Pure forward/backward/optim throughput")
    print("RTX 4070 Ti | torch 2.4.1+cu121 | bf16 autocast")
    print("=" * 70)

    configs = [
        (4, 256, "b4s256"),
        (8, 256, "b8s256"),
        (16, 256, "b16s256"),
        (32, 256, "b32s256"),
        (8, 512, "b8s512"),
        (16, 512, "b16s512"),
        (32, 512, "b32s512"),
    ]

    results = []
    for bs, sl, label in configs:
        print(f"\n--- Benchmarking: batch={bs}, seq_len={sl} ({label}) ---")
        try:
            r = benchmark_batch(bs, sl, n_steps=50, warmup=10, label=label)
            results.append(r)
            print(
                f"  tok/s: {r['tokens_per_sec']:>8.0f}  "
                f"step: {r['avg_ms_per_step']:>7.1f}ms  "
                f"VRAM: {r['peak_vram_mb']:>7.0f}MB  "
                f"power: {r['avg_power_w']:>5.1f}W  "
                f"gpu: {r['gpu_util_pct']:>3d}%"
            )
        except torch.cuda.OutOfMemoryError:
            print("  OOM — skipping")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"  ERROR: {e}")
            torch.cuda.empty_cache()
            gc.collect()

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Config':>10s} {'tok/s':>10s} {'step ms':>10s} {'VRAM MB':>10s} {'Power W':>10s} {'GPU %':>8s}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['label']:>10s} {r['tokens_per_sec']:>10.0f} {r['avg_ms_per_step']:>10.1f} "
            f"{r['peak_vram_mb']:>10.0f} {r['avg_power_w']:>10.1f} {r['gpu_util_pct']:>8d}"
        )

    # Save
    import json

    out = os.path.join(SCRIPT_DIR, "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
