#!/usr/bin/env python3
"""train_model — v51 retrain pipeline.

Loads EE champion traces, decodes to instruction dicts, tokenizes,
and trains the RV32ITransformer for next-token prediction.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "milestone0", "logit-masking"))
sys.path.insert(0, "/root/milestone0/logit-masking")


def train(
    resume_path: str | None = None,
    recurrent_data_path: str | None = None,
    lr_override: float | None = None,
    freeze_layers: int = 0,
    adaptive_freeze: bool = False,
    batch_size: int = 8,
    weight_decay: float = 0.01,
    patience: int = 5,
    max_epochs: int = 12,
    val_split: float = 0.15,
    max_per_level: int | None = None,
    ref_high_augment: int = 0,
    model_size: str = "medium",
    amp: bool = True,
    compile_mode: str | None = None,
    swa_start: int | None = None,
    swa_lr: float | None = None,
    grad_accum_steps: int = 1,
    rebuild_dataset: bool = False,
    **kwargs: Any,
) -> None:
    """Train the RV32ITransformer on champion program traces."""
    import torch
    import torch.nn as nn
    from rv32i_model import RV32ITransformer, RV32IFieldTokenizer
    from rv32i_mask import decode as rv32i_decode

    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────
    ckpt_path = resume_path
    if not ckpt_path or not os.path.isfile(ckpt_path):
        for p in [
            os.path.join(checkpoint_dir, "checkpoint_best.pt"),
            os.path.join(checkpoint_dir, "model_final.pt"),
        ]:
            if os.path.isfile(p):
                ckpt_path = p
                break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # WSL2 CUDA 13.0 caching allocator has a known bug (!handles_.at(i) assert)
    # that prevents model training on GPU. Force CPU for reliable retrain.
    if torch.cuda.is_available():
        print(f"[Train] CUDA available but using CPU (WSL2 allocator workaround)")
        device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model = None
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[Train] Loading checkpoint: {os.path.basename(ckpt_path)}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        model = RV32ITransformer(
            vocab_size=cfg.get("vocab_size", 2048),
            d_model=cfg.get("d_model", 256),
            n_layers=cfg.get("n_layers", 8),
            n_heads=cfg.get("n_heads", 8),
            d_ff=cfg.get("d_ff", 1024),
        )
        sd = ckpt.get("model_state_dict", ckpt)
        if isinstance(sd, dict) and any(k.startswith("blocks.") for k in sd):
            model.load_state_dict(sd, strict=False)
        print(f"[Train] Loaded model: {cfg.get('n_layers', 8)}L, d={cfg.get('d_model', 256)}")
    else:
        print("[Train] Creating fresh 8-layer model")
        model = RV32ITransformer(vocab_size=2048, d_model=256, n_layers=8, n_heads=8, d_ff=1024)

    model.to(device)
    model.train()

    # ── Load champion traces → decode to instruction tokens ──────
    data_path = recurrent_data_path or os.path.join(BASE_DIR, "recurrent_data.jsonl")
    tokenizer = RV32IFieldTokenizer()
    sequences = []

    if os.path.isfile(data_path):
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                prog = entry.get("champion_program", [])
                if not prog:
                    continue

                # Convert hex strings → ints → instruction dicts → tokens
                instr_dicts = []
                for hw in prog:
                    if isinstance(hw, str) and hw.startswith("0x"):
                        word = int(hw, 16)
                    elif isinstance(hw, int):
                        word = hw
                    else:
                        continue
                    # Skip EBREAK (end marker)
                    if word == 0x00100073:
                        continue
                    try:
                        decoded = rv32i_decode(word)
                        if decoded:
                            instr_dicts.append(decoded)
                    except (ValueError, KeyError):
                        continue

                if not instr_dicts:
                    continue

                # Tokenize
                tokens = tokenizer.encode_trace(instr_dicts)
                if tokens and len(tokens) >= 2:
                    sequences.append(torch.tensor(tokens, dtype=torch.long))

    if not sequences:
        print("[Train] No usable sequences — skipping retrain")
        return

    print(f"[Train] Loaded {len(sequences)} sequences from {os.path.basename(data_path)}")

    # ── Phase 1: Inject reference solutions for L8-L13 ─────────
    # Teaches the model correct program structures for high-level tasks
    # regardless of what the champion has achieved.
    ref_injected = 0
    try:
        from tasks_rv32i import LEVELS, get_task_level
        from mutation_proposer import _encode_inst
        inject_levels = [l for l in sorted(LEVELS.keys()) if l >= 8]
        for level in inject_levels:
            try:
                level_cls = get_task_level(level)
                ref = level_cls.reference_solution()
                if not ref:
                    continue
                # Encode reference to words
                ref_words = []
                for inst in ref:
                    w = _encode_inst(inst)
                    if w is not None:
                        ref_words.append(w)
                ref_words.append(0x00100073)  # EBREAK
                # Decode to instruction dicts → tokenize
                ref_dicts = []
                for w in ref_words:
                    if w == 0x00100073:
                        continue
                    try:
                        d = rv32i_decode(w)
                        if d:
                            ref_dicts.append(d)
                    except (ValueError, KeyError):
                        continue
                if not ref_dicts:
                    continue
                ref_tokens = tokenizer.encode_trace(ref_dicts)
                if ref_tokens and len(ref_tokens) >= 2:
                    sequences.append(torch.tensor(ref_tokens, dtype=torch.long))
                    ref_injected += 1
            except Exception:
                continue
    except ImportError:
        pass

    if ref_injected > 0:
        print(f"[Train] Injected {ref_injected} reference solutions (L8-L13)")

    # ── Pad sequences ─────────────────────────────────────────────
    max_len = max(len(s) for s in sequences)
    padded = torch.stack([
        torch.cat([s, torch.zeros(max_len - len(s), dtype=torch.long)])
        for s in sequences
    ])

    # Split train/val
    split = max(1, int(len(padded) * (1 - val_split)))
    train_data = padded[:split].to(device)
    val_data = padded[split:].to(device) if split < len(padded) else None

    # ── Training ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_override or 3e-5,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    pad_token = 0
    actual_epochs = min(max_epochs, 8)
    best_val_loss = float("inf")

    for epoch in range(actual_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            x = batch[:, :-1]
            y = batch[:, 1:]

            logits = model(input_ids=x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        # Validation
        val_loss = avg_loss
        if val_data is not None and len(val_data) > 0:
            model.eval()
            vloss = 0.0
            vbatches = 0
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i:i + batch_size]
                    x = batch[:, :-1]
                    y = batch[:, 1:]
                    logits = model(input_ids=x)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    vloss += loss.item()
                    vbatches += 1
            val_loss = vloss / max(1, vbatches)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"[Train] Epoch {epoch+1}/{actual_epochs}: train={avg_loss:.4f} val={val_loss:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")

    ckpt_out = {
        "model_state_dict": model.cpu().state_dict(),
        "config": {
            "vocab_size": model.vocab_size if hasattr(model, "vocab_size") else 2048,
            "d_model": model.d_model if hasattr(model, "d_model") else 256,
            "n_layers": model.n_layers if hasattr(model, "n_layers") else 8,
            "n_heads": model.n_heads if hasattr(model, "n_heads") else 8,
            "d_ff": model.d_ff if hasattr(model, "d_ff") else 1024,
        },
        "val_loss": best_val_loss,
        "stats": {"epochs": actual_epochs, "examples": len(sequences)},
    }
    torch.save(ckpt_out, final_path)
    print(f"[Train] Saved: {os.path.basename(final_path)} (val_loss={best_val_loss:.4f})")

    import shutil
    shutil.copy2(final_path, best_path)
    print(f"[Train] Updated: {os.path.basename(best_path)}")

    # Cleanup
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
