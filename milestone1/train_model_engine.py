#!/usr/bin/env python3
"""Train the RV32I micro-model on execution traces for next-instruction prediction.

Usage:
    python train_model.py                          # default 8-layer config
    python train_model.py --layers 2 --epochs 3    # quick test
    python train_model.py --layers 16 --d-model 512 --epochs 20  # production
    python train_model.py --resume checkpoints/epoch_5.pt
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'milestone0', 'logit-masking'))

from rv32i_model import RV32ITransformer, RV32IFieldTokenizer as RV32ITokenizer
from rv32i_mask import encode, decode, get_valid_combinations

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Tokenizer ---
tok = RV32ITokenizer()


def trace_to_tokens(trace_data):
    """Convert a trace entry (program instructions) to token sequence."""
    tokens = []
    for inst in trace_data['instructions']:
        tokens.extend(tok.encode_instruction(inst))
    return tokens


# --- Dataset ---
class RV32ITraceDataset(Dataset):
    def __init__(self, traces, max_len=1024):
        self.inputs = []
        self.targets = []
        for t in traces:
            seq = trace_to_tokens(t)
            if len(seq) >= 14:  # at least 2 instructions
                if len(seq) > max_len:
                    seq = seq[:max_len]
                seq_t = torch.tensor(seq, dtype=torch.long)
                self.inputs.append(seq_t[:-1])
                self.targets.append(seq_t[1:])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_len = max(len(inp) for inp in inputs)
    batch_inp = torch.zeros(len(inputs), max_len, dtype=torch.long)
    batch_tgt = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        batch_inp[i, :len(inp)] = inp
        batch_tgt[i, :len(tgt)] = tgt
    return batch_inp, batch_tgt


def parse_args(argv=None):
    """Parse CLI args. Accept argv list for testing; defaults to sys.argv[1:]."""
    parser = argparse.ArgumentParser(description='Train RV32I seed model')
    parser.add_argument('--layers', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--max-len', type=int, default=1024, help='Max sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training_data.jsonl (default: milestone1/training_data.jsonl)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save path for final model checkpoint (default: checkpoints/model_final.pt)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a checkpoint path')
    parser.add_argument('--no-cuda', action='store_true', help='Force CPU even if CUDA available')
    return parser.parse_args(argv)


def save_checkpoint(model, optimizer, epoch, stats, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats,
        'config': {
            'vocab_size': 2048,
            'd_model': model.d_model if hasattr(model, 'd_model') else 256,
            'n_layers': len(model.layers) if hasattr(model, 'layers') else 8,
            'n_heads': model.n_heads if hasattr(model, 'n_heads') else 8,
            'd_ff': model.d_ff if hasattr(model, 'd_ff') else 1024,
        }
    }, path)
    return path


def main():
    args = parse_args()
    set_seed(args.seed)

    # Paths
    base_dir = os.path.dirname(__file__)
    data_path = args.data or os.path.join(base_dir, 'training_data.jsonl')
    output_path = args.output or os.path.join(base_dir, 'checkpoints', 'model_final.pt')
    os.makedirs(os.path.join(base_dir, 'checkpoints'), exist_ok=True)

    # --- Load training data ---
    print(f"Loading training data from {data_path}...")
    traces = []
    with open(data_path) as f:
        for line in f:
            traces.append(json.loads(line))
    print(f"Loaded {len(traces)} traces")

    # Split
    split = int(len(traces) * 0.8)
    train_traces = traces[:split]
    test_traces = traces[split:]
    print(f"Train: {len(train_traces)}, Test: {len(test_traces)}")

    train_ds = RV32ITraceDataset(train_traces, max_len=args.max_len)
    test_ds = RV32ITraceDataset(test_traces, max_len=args.max_len)
    print(f"Train sequences: {len(train_ds)}, Test sequences: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Model ---
    model = RV32ITransformer(
        vocab_size=2048,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    best_acc = 0.0

    # --- Resume from checkpoint ---
    if args.resume:
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('stats', {}).get('best_acc', 0.0)
        print(f"  Resumed at epoch {start_epoch}, best accuracy {best_acc:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model.to(device)
    print(f"Training on {device}")

    # --- Training ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        start = time.time()

        for batch_inp, batch_tgt in train_loader:
            batch_inp = batch_inp.to(device)
            batch_tgt = batch_tgt.to(device)

            optimizer.zero_grad()
            logits = model(batch_inp)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch_tgt.view(-1),
                ignore_index=0
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * (batch_tgt != 0).sum().item()
            total_tokens += (batch_tgt != 0).sum().item()

        # Evaluate
        model.eval()
        correct = 0
        correct_masked = 0
        total = 0
        test_loss = 0

        with torch.no_grad():
            for batch_inp, batch_tgt in test_loader:
                batch_inp = batch_inp.to(device)
                batch_tgt = batch_tgt.to(device)

                logits = model(batch_inp)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch_tgt.view(-1),
                    ignore_index=0
                )
                test_loss += loss.item() * (batch_tgt != 0).sum().item()

                # Raw accuracy
                preds = logits.argmax(dim=-1)
                mask = batch_tgt != 0
                correct += (preds[mask] == batch_tgt[mask]).sum().item()
                total += mask.sum().item()

                # Masked accuracy
                masked_logits = model.apply_logit_mask(logits, batch_inp)
                masked_preds = masked_logits.argmax(dim=-1)
                correct_masked += (masked_preds[mask] == batch_tgt[mask]).sum().item()

        accuracy = correct / total if total > 0 else 0
        masked_accuracy = correct_masked / total if total > 0 else 0
        best_acc = max(best_acc, accuracy)
        elapsed = time.time() - start

        print(f"  Epoch {epoch+1}: loss={total_loss/total_tokens:.4f}, "
              f"test_loss={test_loss/total:.4f}, "
              f"acc={accuracy:.4f}, masked_acc={masked_accuracy:.4f}, "
              f"{elapsed:.1f}s")

        # Save epoch checkpoint
        epoch_path = os.path.join(base_dir, 'checkpoints', f'epoch_{epoch+1}.pt')
        save_checkpoint(model, optimizer, epoch + 1,
                        {'loss': total_loss / total_tokens, 'acc': accuracy,
                         'masked_acc': masked_accuracy, 'best_acc': best_acc},
                        epoch_path)
        print(f"    Checkpoint saved: {epoch_path}")

    # Save final model
    final_path = save_checkpoint(model, optimizer, args.epochs,
                                  {'loss': total_loss / total_tokens, 'acc': accuracy,
                                   'masked_acc': masked_accuracy, 'best_acc': best_acc},
                                  output_path)
    print(f"\nFinal model saved: {final_path}")

    print(f"\nFinal raw accuracy: {accuracy:.4f}")
    print(f"Final masked accuracy: {masked_accuracy:.4f}")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Model: {total_params:,} params")
    print(f"DONE")


if __name__ == '__main__':
    main()
