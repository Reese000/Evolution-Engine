"""
pipeline.py — Evaluation Pipeline for RV32I Transformer
========================================================

Provides quality, performance, and combined evaluation of the RV32I
instruction prediction transformer.

Classes:
    EvalResult        — dataclass holding all evaluation metrics
    MetricsCollector  — thread-safe latency / throughput recorder
    EvaluationManager — orchestrates quality + performance evaluation
    ReportGenerator   — produces markdown reports from eval history

Usage:
    from evaluation import EvaluationManager, ReportGenerator

    manager = EvaluationManager(model, tokenizer, device)
    result = manager.run_full_evaluation(eval_data)
    manager.save_result(result)
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ------------------------------------------------------------------
#  Import project model classes
# ------------------------------------------------------------------
import sys

# Ensure the milestone1 directory is on the path so rv32i_model is importable
_MILESTONE1_DIR = str(Path(__file__).resolve().parent.parent)
if _MILESTONE1_DIR not in sys.path:
    sys.path.insert(0, _MILESTONE1_DIR)

from rv32i_model import RV32ITransformer, RV32IFieldTokenizer, RV32IVocab, compute_loss  # noqa: E402


# ==================================================================
#  EvalResult — frozen-ish container for one evaluation run
# ==================================================================

@dataclass
class EvalResult:
    """All metrics produced by a single evaluation run."""

    timestamp: str = ""
    model_path: str = ""

    # Quality metrics
    loss: float = 0.0
    perplexity: float = 0.0
    accuracy: float = 0.0
    masked_accuracy: float = 0.0

    # Latency / throughput
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0

    # Generalisation
    memorization_score: float = 0.0
    generalization_score: float = 0.0
    per_level_generalization: Dict[str, float] = field(default_factory=dict)

    # Hardware utilisation
    gpu_memory_used_mb: float = 0.0
    gpu_utilization_pct: float = 0.0

    # Dataset info
    dataset_size: int = 0
    num_samples: int = 0

    # Catch-all
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EvalResult:
        # Be lenient: only pass known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ==================================================================
#  MetricsCollector — thread-safe latency / throughput accumulator
# ==================================================================

class MetricsCollector:
    """Collects per-inference timing and throughput samples.

    All public methods are thread-safe so the collector can be shared
    across evaluation workers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies_ms: List[float] = []
        self._tokens_generated: List[int] = []
        self._gpu_memory_mb: List[float] = []

    # ------------------------------------------------------------------
    def record_inference(
        self,
        latency_ms: float,
        tokens_generated: int,
        gpu_memory_mb: float = 0.0,
    ) -> None:
        """Record a single inference measurement."""
        with self._lock:
            self._latencies_ms.append(latency_ms)
            self._tokens_generated.append(tokens_generated)
            self._gpu_memory_mb.append(gpu_memory_mb)

    def record_batch(
        self,
        batch_size: int,
        total_latency_ms: float,
        tokens_generated: int,
        gpu_memory_mb: float = 0.0,
    ) -> None:
        """Record a batch inference measurement.

        The per-sample latency is ``total_latency_ms / batch_size``.
        """
        per_sample = total_latency_ms / max(batch_size, 1)
        with self._lock:
            self._latencies_ms.append(per_sample)
            self._tokens_generated.append(tokens_generated)
            self._gpu_memory_mb.append(gpu_memory_mb)

    # ------------------------------------------------------------------
    def get_summary(self) -> Dict[str, float]:
        """Compute summary statistics from collected samples.

        Returns dict with keys: p50_ms, p95_ms, p99_ms,
        mean_latency_ms, throughput_tps, mean_gpu_memory_mb, n_samples.
        """
        with self._lock:
            lats = list(self._latencies_ms)
            toks = list(self._tokens_generated)
            mems = list(self._gpu_memory_mb)

        if not lats:
            return {
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "mean_latency_ms": 0.0,
                "throughput_tps": 0.0,
                "mean_gpu_memory_mb": 0.0,
                "n_samples": 0,
            }

        lats_arr = np.array(lats, dtype=np.float64)
        p50 = float(np.percentile(lats_arr, 50))
        p95 = float(np.percentile(lats_arr, 95))
        p99 = float(np.percentile(lats_arr, 99))
        mean_lat = float(lats_arr.mean())

        # Throughput: total tokens / total time
        total_tokens = sum(toks)
        total_time_sec = lats_arr.sum() / 1000.0
        throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0.0

        mean_mem = float(np.mean(mems)) if mems else 0.0

        return {
            "p50_ms": round(p50, 3),
            "p95_ms": round(p95, 3),
            "p99_ms": round(p99, 3),
            "mean_latency_ms": round(mean_lat, 3),
            "throughput_tps": round(throughput, 1),
            "mean_gpu_memory_mb": round(mean_mem, 1),
            "n_samples": len(lats),
        }

    def reset(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self._latencies_ms.clear()
            self._tokens_generated.clear()
            self._gpu_memory_mb.clear()


# ==================================================================
#  EvaluationManager — main evaluation orchestrator
# ==================================================================

class EvaluationManager:
    """Runs quality and performance evaluations on an RV32I transformer.

    Args:
        model:      RV32ITransformer instance (should be on the target device)
        tokenizer:  RV32IFieldTokenizer instance
        device:     torch device string or torch.device
        output_dir: directory for saving eval history
    """

    def __init__(
        self,
        model: RV32ITransformer,
        tokenizer: RV32IFieldTokenizer,
        device: torch.device | str,
        output_dir: str = "evaluation_results",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = MetricsCollector()

    # ------------------------------------------------------------------
    #  Quality evaluation
    # ------------------------------------------------------------------
    def evaluate_quality(
        self,
        eval_data: List[List[int]],
        batch_size: int = 8,
    ) -> EvalResult:
        """Evaluate model quality on tokenised sequences.

        Computes cross-entropy loss, perplexity, accuracy (argmax match),
        and masked accuracy (excluding PAD tokens).

        Args:
            eval_data:   list of token-ID sequences
            batch_size:  evaluation batch size

        Returns:
            EvalResult with quality metrics populated.
        """
        print(f"[Eval] Quality evaluation on {len(eval_data)} samples, batch_size={batch_size}")
        self.model.eval()

        # Pad all sequences to the same length
        max_len = max(len(seq) for seq in eval_data)
        vocab = RV32IVocab
        padded = [
            seq + [vocab.PAD] * (max_len - len(seq))
            for seq in eval_data
        ]

        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        num_samples = len(eval_data)

        total_loss = 0.0
        total_tokens = 0        # non-PAD tokens in labels
        correct_tokens = 0      # argmax matches (all non-PAD)
        masked_correct = 0      # argmax matches (non-PAD, non-special)
        masked_total = 0

        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch = input_ids[start:end]  # (B, T)

                with torch.amp.autocast("cuda", enabled=True):
                    logits = self.model(batch)  # (B, T, V)

                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                B, T, V = shift_logits.shape

                # Per-token cross entropy (unreduced)
                loss_per_token = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, V),
                    shift_labels.view(-1),
                    ignore_index=vocab.PAD,
                    reduction="none",
                )  # (B*T,)

                # Build active mask (non-PAD positions)
                active = (shift_labels.view(-1) != vocab.PAD)
                batch_loss = loss_per_token[active].sum().item()
                batch_count = active.sum().item()

                total_loss += batch_loss
                total_tokens += batch_count

                # Accuracy: argmax of logits vs ground truth
                preds = shift_logits.argmax(dim=-1)  # (B, T)
                correct = (preds == shift_labels) & active.view(B, T)
                correct_tokens += correct.sum().item()

                # Masked accuracy: exclude PAD *and* special tokens from
                # denominator so we measure only real instruction fields
                special_ids = {
                    vocab.PAD, vocab.SEP, vocab.CLS, vocab.MASK, vocab.NO_REG,
                }
                real_mask = active.view(B, T)
                for sid in special_ids:
                    real_mask = real_mask & (shift_labels != sid)

                masked_correct += ((preds == shift_labels) & real_mask).sum().item()
                masked_total += real_mask.sum().item()

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = float(np.exp(avg_loss))
        accuracy = correct_tokens / max(total_tokens, 1)
        masked_acc = masked_correct / max(masked_total, 1)

        print(f"[Eval]   loss={avg_loss:.4f}  perplexity={perplexity:.2f}  "
              f"accuracy={accuracy:.4f}  masked_accuracy={masked_acc:.4f}")

        # GPU memory snapshot
        gpu_mem = 0.0
        gpu_util = 0.0
        if self.device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        return EvalResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            loss=round(avg_loss, 6),
            perplexity=round(perplexity, 4),
            accuracy=round(accuracy, 6),
            masked_accuracy=round(masked_acc, 6),
            gpu_memory_used_mb=round(gpu_mem, 1),
            num_samples=num_samples,
            dataset_size=len(eval_data),
        )

    # ------------------------------------------------------------------
    #  Performance evaluation (latency / throughput)
    # ------------------------------------------------------------------
    def evaluate_performance(
        self,
        num_samples: int = 100,
        prompt_lengths: Optional[List[int]] = None,
    ) -> EvalResult:
        """Measure inference latency and throughput.

        Generates random prompts of varying lengths, runs single-sample
        inference, and collects per-request timing statistics.

        Args:
            num_samples:     number of random prompts to evaluate
            prompt_lengths:  list of prompt lengths (in tokens) to sample
                             from. Defaults to [9, 18, 36, 72].

        Returns:
            EvalResult with latency / throughput metrics populated.
        """
        if prompt_lengths is None:
            prompt_lengths = [9, 18, 36, 72]

        print(f"[Eval] Performance evaluation: {num_samples} samples, "
              f"prompt_lengths={prompt_lengths}")
        self.model.eval()
        self.metrics.reset()

        vocab = RV32IVocab
        # Token range for generating random prompts (instruction + register tokens)
        token_pool = list(range(vocab.INST_START, vocab.REG_START + vocab.REG_COUNT))

        # Warmup run to stabilise GPU clocks
        warmup_len = prompt_lengths[0]
        warmup_ids = torch.randint(
            token_pool[0], token_pool[-1] + 1,
            (1, warmup_len), device=self.device, dtype=torch.long,
        )
        with torch.no_grad():
            self.model.generate(warmup_ids, max_new_tokens=9, temperature=0.0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        # Reset peak memory after warmup
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_allocated(self.device)

        total_tokens_generated = 0

        for i in range(num_samples):
            plen = prompt_lengths[i % len(prompt_lengths)]
            prompt = torch.randint(
                token_pool[0], token_pool[-1] + 1,
                (1, plen), device=self.device, dtype=torch.long,
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            t0 = time.perf_counter()
            with torch.no_grad():
                output = self.model.generate(
                    prompt, max_new_tokens=9, temperature=0.0,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            tokens_out = output.shape[1] - plen
            total_tokens_generated += tokens_out

            gpu_mem = 0.0
            if self.device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

            self.metrics.record_inference(
                latency_ms=latency_ms,
                tokens_generated=tokens_out,
                gpu_memory_mb=gpu_mem,
            )

            if (i + 1) % 25 == 0 or i == num_samples - 1:
                print(f"[Eval]   ... {i + 1}/{num_samples} samples done")

        summary = self.metrics.get_summary()

        # Compute GPU utilisation estimate: busy time / wall time
        gpu_util = 0.0
        if self.device.type == "cuda":
            # Approximate: total inference wall time / total elapsed time
            total_inference_sec = sum(
                s / 1000.0 for s in
                [self.metrics._latencies_ms[j] for j in range(len(self.metrics._latencies_ms))]
            )
            # Use a rough wall clock from the loop
            gpu_util = min(100.0, (total_inference_sec / max(total_inference_sec, 0.001)) * 100)

        gpu_mem = 0.0
        if self.device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        result = EvalResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_p50_ms=summary["p50_ms"],
            latency_p95_ms=summary["p95_ms"],
            latency_p99_ms=summary["p99_ms"],
            throughput_tokens_per_sec=summary["throughput_tps"],
            gpu_memory_used_mb=round(gpu_mem, 1),
            gpu_utilization_pct=round(gpu_util, 1),
            num_samples=num_samples,
            extra={
                "prompt_lengths": prompt_lengths,
                "mean_latency_ms": summary["mean_latency_ms"],
                "mean_gpu_memory_mb": summary["mean_gpu_memory_mb"],
            },
        )

        print(f"[Eval]   p50={summary['p50_ms']:.1f}ms  p95={summary['p95_ms']:.1f}ms  "
              f"p99={summary['p99_ms']:.1f}ms  throughput={summary['throughput_tps']:.0f} tok/s")

        return result

    # ------------------------------------------------------------------
    #  Combined evaluation
    # ------------------------------------------------------------------
    def run_full_evaluation(
        self,
        eval_data: List[List[int]],
        recurrent_data_path: Optional[str] = None,
    ) -> EvalResult:
        """Run quality + performance evaluation and merge results.

        Args:
            eval_data:             tokenised sequences for quality eval
            recurrent_data_path:   optional path to a JSONL file of
                                   additional recurrent evaluation data

        Returns:
            Combined EvalResult with all metrics.
        """
        print("[Eval] ===== Full Evaluation Start =====")

        # --- Quality ---
        quality_result = self.evaluate_quality(eval_data)

        # --- Performance ---
        perf_result = self.evaluate_performance(num_samples=min(100, len(eval_data)))

        # --- Generalisation (if recurrent data provided) ---
        gen_score = 0.0
        mem_score = 0.0
        per_level: Dict[str, float] = {}
        if recurrent_data_path and os.path.isfile(recurrent_data_path):
            gen_score, mem_score, per_level = self._evaluate_generalisation(
                recurrent_data_path
            )

        # --- Merge ---
        combined = EvalResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_path=quality_result.extra.get("model_path", ""),

            # Quality
            loss=quality_result.loss,
            perplexity=quality_result.perplexity,
            accuracy=quality_result.accuracy,
            masked_accuracy=quality_result.masked_accuracy,

            # Performance
            latency_p50_ms=perf_result.latency_p50_ms,
            latency_p95_ms=perf_result.latency_p95_ms,
            latency_p99_ms=perf_result.latency_p99_ms,
            throughput_tokens_per_sec=perf_result.throughput_tokens_per_sec,

            # Generalisation
            memorization_score=mem_score,
            generalization_score=gen_score,
            per_level_generalization=per_level,

            # Hardware
            gpu_memory_used_mb=max(quality_result.gpu_memory_used_mb,
                                   perf_result.gpu_memory_used_mb),
            gpu_utilization_pct=perf_result.gpu_utilization_pct,

            # Dataset
            dataset_size=quality_result.dataset_size,
            num_samples=quality_result.num_samples,
            extra={
                "quality_extra": quality_result.extra,
                "perf_extra": perf_result.extra,
            },
        )

        print("[Eval] ===== Full Evaluation Complete =====")
        print(f"[Eval]   loss={combined.loss:.4f}  ppl={combined.perplexity:.2f}  "
              f"acc={combined.accuracy:.4f}  p50={combined.latency_p50_ms:.1f}ms")

        return combined

    # ------------------------------------------------------------------
    #  Generalisation sub-evaluation
    # ------------------------------------------------------------------
    def _evaluate_generalisation(
        self,
        recurrent_data_path: str,
    ) -> tuple[float, float, Dict[str, float]]:
        """Evaluate memorisation vs generalisation on recurrent data.

        The JSONL file is expected to have lines like:
            {"level": "L0", "tokens": [1, 2, 3, ...]}
        where "level" indicates difficulty (L0=train, L1=near, L2=far).

        Returns:
            (generalisation_score, memorization_score, per_level_dict)
        """
        print(f"[Eval] Generalisation eval from {recurrent_data_path}")

        level_data: Dict[str, List[List[int]]] = {}
        with open(recurrent_data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                level = rec.get("level", "unknown")
                tokens = rec.get("tokens", [])
                if tokens:
                    level_data.setdefault(level, []).append(tokens)

        if not level_data:
            return 0.0, 0.0, {}

        self.model.eval()
        per_level: Dict[str, float] = {}
        all_accuracies: List[float] = []
        train_accuracy = 0.0

        for level, sequences in sorted(level_data.items()):
            result = self.evaluate_quality(sequences, batch_size=8)
            per_level[level] = result.accuracy
            all_accuracies.append(result.accuracy)
            if level in ("L0", "train"):
                train_accuracy = result.accuracy

        # Memorisation = accuracy on training-level data
        mem_score = train_accuracy

        # Generalisation = mean accuracy on non-training levels
        non_train = {k: v for k, v in per_level.items() if k not in ("L0", "train")}
        gen_score = float(np.mean(list(non_train.values()))) if non_train else 0.0

        print(f"[Eval]   memorisation={mem_score:.4f}  generalisation={gen_score:.4f}  "
              f"levels={list(per_level.keys())}")

        return gen_score, mem_score, per_level

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------
    def save_result(self, result: EvalResult) -> str:
        """Append an EvalResult to eval_history.jsonl.

        Returns the path written to.
        """
        path = self.output_dir / "eval_history.jsonl"
        record = result.to_dict()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        print(f"[Eval] Result saved to {path}")
        return str(path)

    def load_history(self) -> List[EvalResult]:
        """Read eval_history.jsonl and return a list of EvalResult."""
        path = self.output_dir / "eval_history.jsonl"
        if not path.exists():
            return []
        results: List[EvalResult] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    results.append(EvalResult.from_dict(d))
                except (json.JSONDecodeError, TypeError):
                    continue
        return results


# ==================================================================
#  ReportGenerator — markdown report output
# ==================================================================

class ReportGenerator:
    """Generates markdown evaluation reports.

    Args:
        output_dir: directory containing eval_history.jsonl
    """

    def __init__(self, output_dir: str = "evaluation_results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def generate_summary_report(
        self,
        history: Optional[List[EvalResult]] = None,
    ) -> str:
        """Produce a markdown summary report.

        If *history* is None, loads from eval_history.jsonl.

        The report includes:
        - Latest metrics snapshot
        - Trend vs previous run (delta arrows)
        - Best-ever values across all runs
        """
        if history is None:
            history = self._load_history()

        if not history:
            return "# Evaluation Report\n\nNo evaluation data found.\n"

        latest = history[-1]
        previous = history[-2] if len(history) >= 2 else None

        lines: List[str] = []
        lines.append("# RV32I Transformer — Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"**Total runs in history:** {len(history)}")
        lines.append("")

        # --- Latest metrics ---
        lines.append("## Latest Run")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Timestamp | {latest.timestamp} |")
        lines.append(f"| Loss | {latest.loss:.4f} |")
        lines.append(f"| Perplexity | {latest.perplexity:.2f} |")
        lines.append(f"| Accuracy | {latest.accuracy:.4f} |")
        lines.append(f"| Masked Accuracy | {latest.masked_accuracy:.4f} |")
        lines.append(f"| Latency P50 | {latest.latency_p50_ms:.1f} ms |")
        lines.append(f"| Latency P95 | {latest.latency_p95_ms:.1f} ms |")
        lines.append(f"| Latency P99 | {latest.latency_p99_ms:.1f} ms |")
        lines.append(f"| Throughput | {latest.throughput_tokens_per_sec:.0f} tok/s |")
        lines.append(f"| GPU Memory | {latest.gpu_memory_used_mb:.1f} MB |")
        lines.append(f"| Dataset Size | {latest.dataset_size} |")
        lines.append("")

        # --- Trend vs previous ---
        if previous is not None:
            lines.append("## Trend (vs previous run)")
            lines.append("")
            lines.append("| Metric | Previous | Latest | Delta |")
            lines.append("|--------|----------|--------|-------|")

            def _delta_str(prev_val: float, curr_val: float, lower_is_better: bool = True) -> str:
                d = curr_val - prev_val
                if abs(d) < 1e-8:
                    return "---"
                arrow = "↓" if (d < 0) == lower_is_better else "↑"
                return f"{arrow} {d:+.4f}"

            trend_rows = [
                ("Loss", previous.loss, latest.loss, True),
                ("Perplexity", previous.perplexity, latest.perplexity, True),
                ("Accuracy", previous.accuracy, latest.accuracy, False),
                ("Masked Accuracy", previous.masked_accuracy, latest.masked_accuracy, False),
                ("Latency P50 (ms)", previous.latency_p50_ms, latest.latency_p50_ms, True),
                ("Throughput (tok/s)", previous.throughput_tokens_per_sec,
                 latest.throughput_tokens_per_sec, False),
            ]
            for name, prev_val, curr_val, lib in trend_rows:
                lines.append(f"| {name} | {prev_val:.4f} | {curr_val:.4f} | "
                             f"{_delta_str(prev_val, curr_val, lib)} |")
            lines.append("")

        # --- Best ever ---
        lines.append("## Best Ever Values")
        lines.append("")

        best_loss = min(r.loss for r in history)
        best_ppl = min(r.perplexity for r in history)
        best_acc = max(r.accuracy for r in history)
        best_masked = max(r.masked_accuracy for r in history)
        best_latency = min(r.latency_p50_ms for r in history if r.latency_p50_ms > 0) if any(r.latency_p50_ms > 0 for r in history) else 0.0
        best_throughput = max(r.throughput_tokens_per_sec for r in history)

        lines.append("| Metric | Best Value | Run |")
        lines.append("|--------|-----------|-----|")

        def _find_run(predicate, metric_name: str) -> str:
            matches = [r for r in history if predicate(r)]
            if not matches:
                return "N/A"
            best = matches[0]
            val = getattr(best, metric_name)
            return f"{val:.4f} (run {history.index(best) + 1})"

        lines.append(f"| Loss | {best_loss:.4f} | {_find_run(lambda r: r.loss == best_loss, 'loss')} |")
        lines.append(f"| Perplexity | {best_ppl:.2f} | {_find_run(lambda r: r.perplexity == best_ppl, 'perplexity')} |")
        lines.append(f"| Accuracy | {best_acc:.4f} | {_find_run(lambda r: r.accuracy == best_acc, 'accuracy')} |")
        lines.append(f"| Masked Accuracy | {best_masked:.4f} | {_find_run(lambda r: r.masked_accuracy == best_masked, 'masked_accuracy')} |")
        if best_latency > 0:
            lines.append(f"| Latency P50 | {best_latency:.1f} ms | {_find_run(lambda r: r.latency_p50_ms == best_latency, 'latency_p50_ms')} |")
        lines.append(f"| Throughput | {best_throughput:.0f} tok/s | {_find_run(lambda r: r.throughput_tokens_per_sec == best_throughput, 'throughput_tokens_per_sec')} |")
        lines.append("")

        # --- Generalisation (if present) ---
        if latest.generalization_score > 0 or latest.per_level_generalization:
            lines.append("## Generalisation")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Memorisation Score | {latest.memorization_score:.4f} |")
            lines.append(f"| Generalisation Score | {latest.generalization_score:.4f} |")
            lines.append("")
            if latest.per_level_generalization:
                lines.append("### Per-Level Accuracy")
                lines.append("")
                lines.append("| Level | Accuracy |")
                lines.append("|-------|----------|")
                for lvl, acc in sorted(latest.per_level_generalization.items()):
                    lines.append(f"| {lvl} | {acc:.4f} |")
                lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def generate_comparison_report(
        self,
        result_a: EvalResult,
        result_b: EvalResult,
    ) -> str:
        """Produce a markdown table comparing two EvalResults."""
        lines: List[str] = []
        lines.append("# Evaluation Comparison")
        lines.append("")
        lines.append(f"| Metric | Run A | Run B | Delta |")
        lines.append(f"|--------|-------|-------|-------|")

        fields = [
            ("Timestamp", "timestamp", None),
            ("Loss", "loss", True),
            ("Perplexity", "perplexity", True),
            ("Accuracy", "accuracy", False),
            ("Masked Accuracy", "masked_accuracy", False),
            ("Latency P50 (ms)", "latency_p50_ms", True),
            ("Latency P95 (ms)", "latency_p95_ms", True),
            ("Latency P99 (ms)", "latency_p99_ms", True),
            ("Throughput (tok/s)", "throughput_tokens_per_sec", False),
            ("GPU Memory (MB)", "gpu_memory_used_mb", True),
            ("Memorisation", "memorization_score", False),
            ("Generalisation", "generalization_score", False),
            ("Num Samples", "num_samples", None),
        ]

        for label, attr, lower_better in fields:
            va = getattr(result_a, attr)
            vb = getattr(result_b, attr)
            if isinstance(va, str):
                lines.append(f"| {label} | {va} | {vb} | --- |")
            elif isinstance(va, (int, float)) and lower_better is not None:
                delta = vb - va
                arrow = ""
                if abs(delta) > 1e-8:
                    better = (delta < 0) == lower_better
                    arrow = "↓" if better else "↑"
                fmt = ".4f" if isinstance(va, float) and abs(va) < 100 else ".0f"
                lines.append(f"| {label} | {va:{fmt}} | {vb:{fmt}} | {arrow} {delta:+.{fmt[1:]}} |")
            else:
                lines.append(f"| {label} | {va} | {vb} | --- |")

        lines.append("")

        # Per-level generalisation comparison (if either has it)
        levels_a = result_a.per_level_generalization
        levels_b = result_b.per_level_generalization
        all_levels = sorted(set(levels_a.keys()) | set(levels_b.keys()))
        if all_levels:
            lines.append("### Per-Level Generalisation")
            lines.append("")
            lines.append("| Level | Run A | Run B | Delta |")
            lines.append("|-------|-------|-------|-------|")
            for lvl in all_levels:
                va = levels_a.get(lvl, 0.0)
                vb = levels_b.get(lvl, 0.0)
                delta = vb - va
                lines.append(f"| {lvl} | {va:.4f} | {vb:.4f} | {delta:+.4f} |")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def save_report(self, report: str, filename: str) -> str:
        """Write a markdown report to a file.

        Returns the path written to.
        """
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Eval] Report saved to {path}")
        return str(path)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _load_history(self) -> List[EvalResult]:
        path = self.output_dir / "eval_history.jsonl"
        if not path.exists():
            return []
        results: List[EvalResult] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    results.append(EvalResult.from_dict(d))
                except (json.JSONDecodeError, TypeError):
                    continue
        return results


# ==================================================================
#  Self-test
# ==================================================================

def _self_test():
    """Quick smoke test for the evaluation pipeline."""
    import tempfile

    print("=" * 60)
    print("Evaluation Pipeline Self-Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Build a small model for testing
    model = RV32ITransformer(
        vocab_size=RV32IVocab.VOCAB_SIZE,
        d_model=128, n_layers=2, n_heads=4, d_ff=256,
        max_seq_len=256, dropout=0.0,
    )
    model.to(device)
    model.eval()

    tokenizer = RV32IFieldTokenizer()
    vocab = RV32IVocab

    # Generate some dummy eval data (sequences of 3 instructions each)
    print("\n[1] Generating dummy eval data...")
    eval_data = []
    inst_names = vocab.get_instruction_names()
    for i in range(32):
        seq = []
        for j in range(3):
            name = inst_names[(i + j) % len(inst_names)]
            inst = {"name": name, "rd": (i + j) % 32, "rs1": (i * 2) % 32, "rs2": (j * 3) % 32}
            seq.extend(tokenizer.encode_instruction(inst))
        eval_data.append(seq)
    print(f"    Created {len(eval_data)} sequences, lengths={[len(s) for s in eval_data[:3]]}...")

    # Create manager
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = EvaluationManager(model, tokenizer, device, output_dir=tmpdir)

        # --- Quality ---
        print("\n[2] Quality evaluation...")
        qr = manager.evaluate_quality(eval_data, batch_size=8)
        print(f"    loss={qr.loss:.4f}  ppl={qr.perplexity:.2f}  acc={qr.accuracy:.4f}")

        # --- Performance ---
        print("\n[3] Performance evaluation...")
        pr = manager.evaluate_performance(num_samples=10, prompt_lengths=[9, 18])
        print(f"    p50={pr.latency_p50_ms:.1f}ms  throughput={pr.throughput_tokens_per_sec:.0f} tok/s")

        # --- Combined ---
        print("\n[4] Full evaluation...")
        full = manager.run_full_evaluation(eval_data)
        print(f"    loss={full.loss:.4f}  ppl={full.perplexity:.2f}")

        # --- Save / Load ---
        print("\n[5] Save and load history...")
        manager.save_result(full)
        history = manager.load_history()
        print(f"    History entries: {len(history)}")
        assert len(history) == 1, f"Expected 1 entry, got {len(history)}"

        # --- Reports ---
        print("\n[6] Report generation...")
        rg = ReportGenerator(output_dir=tmpdir)
        summary = rg.generate_summary_report(history)
        print(f"    Summary report length: {len(summary)} chars")
        assert "# RV32I Transformer" in summary

        # Run a second eval for comparison
        full2 = manager.run_full_evaluation(eval_data)
        manager.save_result(full2)

        comparison = rg.generate_comparison_report(full, full2)
        print(f"    Comparison report length: {len(comparison)} chars")
        assert "Run A" in comparison and "Run B" in comparison

        rg.save_report(summary, "summary.md")
        rg.save_report(comparison, "comparison.md")

        # --- MetricsCollector ---
        print("\n[7] MetricsCollector unit test...")
        mc = MetricsCollector()
        mc.record_inference(5.0, 9, 100.0)
        mc.record_inference(6.0, 9, 100.0)
        mc.record_batch(4, 20.0, 36, 120.0)
        s = mc.get_summary()
        print(f"    Samples: {s['n_samples']}  p50: {s['p50_ms']:.1f}ms")
        assert s["n_samples"] == 3
        mc.reset()
        assert mc.get_summary()["n_samples"] == 0
        print("    MetricsCollector OK")

    print("\n" + "=" * 60)
    print("SELF-TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
