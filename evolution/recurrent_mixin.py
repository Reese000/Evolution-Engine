"""RecurrentMixin - Recurrent feedback and strategy outcome tracking.

Extracted from EvolutionEngine in evolution_engine.py.
Uses mixin pattern: all methods access self.* from the host EvolutionEngine.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import time

from champion_anchor import sha256_program
from evolution.shared import (
    MAX_RECURRENT_ENTRIES,
    MAX_TERMINAL_RECURRENT_PER_CYCLE,
    MIN_FITNESS_FOR_TRACE,
    TERMINAL_RECURRENT_GENERATION_INTERVAL,
)
from paths import MILESTONE1_DIR
from tasks_rv32i import LEVELS


class RecurrentMixin:
    """Mixin providing recurrent feedback and strategy outcome recording."""

    _SKIP_RECURRENT_STRATEGIES_AT_L30 = frozenset(
        {
            "DIVERSITY_INJECT",
            "TERMINAL_VERIFY_PROBE",
            "SEMANTIC_COMPRESS",
            "L30_EASY_FILL",
        }
    )

    def _skip_task_aware_recurrent_at_scan_floor(self) -> bool:
        """Overlay task_aware duplicates TERMINAL_VERIFY_PROBE at held-out floor."""
        if not self._terminal_recurrent_level_fast_path():
            return False
        from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor

        champ = getattr(self, "champion", None)
        return bool(champ) and terminal_efficiency_at_scan_floor(len(champ))

    def _terminal_recurrent_level_fast_path(self) -> bool:
        """At saturated max depth, score traces on champion level only."""
        config = getattr(self, "config", None)
        max_level = int(getattr(config, "max_level", 30) if config is not None else 30)
        return (
            int(getattr(self, "champion_level", 0)) >= max_level
            and float(getattr(self, "_champion_verify_score", 0.0)) >= 0.99 - 1e-6
        )

    def _collect_recurrent_candidates(self, fitnesses, n_fitnesses):
        """Phase 1: Collect eligible candidates for recurrent feedback."""

        _candidates = []

        rejected_count = 0

        _levels = LEVELS

        _verify_levels_sorted = None

        strategies = getattr(self, "_last_challenger_strategies", None)
        skip_terminal_recurrent = int(getattr(self, "champion_level", 0)) >= 30
        champ_sha = (
            sha256_program(self.champion) if skip_terminal_recurrent and getattr(self, "champion", None) else None
        )

        for idx in range(len(self._last_challengers)):

            prog = self._last_challengers[idx]

            if idx >= len(self._last_results):

                continue

            if (
                skip_terminal_recurrent
                and strategies is not None
                and idx < len(strategies)
                and strategies[idx] in self._SKIP_RECURRENT_STRATEGIES_AT_L30
            ):
                continue

            if (
                skip_terminal_recurrent
                and strategies is not None
                and idx < len(strategies)
                and strategies[idx] == "TASK_AWARE"
                and self._skip_task_aware_recurrent_at_scan_floor()
            ):
                continue

            trace = self._last_results[idx]

            if trace.get("status", "") != "ok":

                continue

            sha256 = sha256_program(prog)

            if champ_sha is not None and sha256 == champ_sha:
                continue

            if sha256 in self._saved_sha256s:

                continue

            challenger_fitness = fitnesses[idx] if idx < n_fitnesses else 0.0

            if challenger_fitness <= MIN_FITNESS_FOR_TRACE and self.generations_stalled < 30:

                rejected_count += 1

                continue

            if len(prog) < 3:

                continue

            _trace_instr_count = trace.get("instructions_count", 0)

            if _trace_instr_count <= 0:

                rejected_count += 1

                continue

            if _trace_instr_count > 0 and hasattr(self, "_saved_size_counts"):

                _size_bucket = (_trace_instr_count // 5) * 5

                _total_saved = sum(self._saved_size_counts.values()) if self._saved_size_counts else 0

                if _total_saved > 20:

                    _bucket_count = self._saved_size_counts.get(_size_bucket, 0)

                    if _bucket_count > _total_saved * 0.7:

                        rejected_count += 1

                        continue

                self._saved_size_counts[_size_bucket] = self._saved_size_counts.get(_size_bucket, 0) + 1

            _candidates.append((prog, challenger_fitness, trace, sha256))

        return _candidates, rejected_count, _levels, _verify_levels_sorted

    def _compute_level_weights(self, candidates, levels, verify_levels_sorted):
        """Phase 3.5 + 4: Compute level-diversity and fitness weights."""

        _level_data = []

        if candidates:

            if verify_levels_sorted is None:

                verify_levels_sorted = sorted(levels.keys())

            terminal_fast = self._terminal_recurrent_level_fast_path()

            insts = getattr(self, "_recurrent_verify_instances", None)

            if insts is None or getattr(self, "_recurrent_verify_levels_id", None) is not id(levels):

                if terminal_fast:

                    lvl = int(self.champion_level)

                    insts = {lvl: levels[lvl](seed=lvl)}

                else:

                    insts = {lvl: levels[lvl](seed=lvl) for lvl in verify_levels_sorted}

                self._recurrent_verify_instances = insts

                self._recurrent_verify_levels_id = id(levels)

            if terminal_fast:

                lvl = int(self.champion_level)

                inst = insts[lvl]

                for _prog, _fit, _trace, _sha in candidates:

                    _output = {
                        "registers": _trace.get("final_regs", {}),
                        "memory": _trace.get("final_memory", {}),
                    }

                    try:

                        _task_score = float(inst.verify(_output, self.task_instance))

                    except RuntimeError:

                        _task_score = 0.0

                    _level_data.append((lvl, _task_score))

            else:

                for _prog, _fit, _trace, _sha in candidates:

                    _task_score = 0.0

                    _solved_level = self.champion_level

                    _output = {
                        "registers": _trace.get("final_regs", {}),
                        "memory": _trace.get("final_memory", {}),
                    }

                    try:

                        for lvl in verify_levels_sorted:

                            try:

                                score = insts[lvl].verify(_output, self.task_instance)

                                if score > _task_score:

                                    _task_score = score

                                    _solved_level = lvl

                            except RuntimeError:

                                continue

                    except RuntimeError:

                        pass

                    _level_data.append((_solved_level, _task_score))

        _weights = None

        _n_levels_found = 1

        if candidates:

            _fits = [c[1] for c in candidates]

            _max_f, _min_f = max(_fits), min(_fits)

            _fitness_range = _max_f - _min_f

            _temperature = max(0.5, _fitness_range * 0.2) if _fitness_range > 0 else 1.0

            if _fitness_range > 1e-6:

                _logits = [((f - _min_f) / _fitness_range) * (1.0 / _temperature) for f in _fits]

                _max_logit = max(_logits)

                _exp_logits = [math.exp(logit - _max_logit) for logit in _logits]

                _sum_exp = sum(_exp_logits)

                _fitness_weights = (
                    [e / _sum_exp for e in _exp_logits] if _sum_exp > 0 else [1.0 / len(_logits)] * len(_logits)
                )

            else:

                _fitness_weights = [1.0 / len(_fits)] * len(_fits)

            _level_counter = {}

            for _sd, _ in _level_data:

                _level_counter[_sd] = _level_counter.get(_sd, 0) + 1

            _n_levels_found = len(_level_counter)

            _avg_per_level = len(candidates) / max(1, _n_levels_found)

            _level_div_mult = {lvl: _avg_per_level / max(1, cnt) for lvl, cnt in _level_counter.items()}

            _weights = [_fitness_weights[i] * _level_div_mult[_level_data[i][0]] for i in range(len(candidates))]

            _w_sum = sum(_weights)

            if _w_sum > 0:

                _weights = [w / _w_sum for w in _weights]

        return _level_data, _weights, _n_levels_found

    def _load_saved_sha256_index(self, jsonl_path: str) -> dict:
        """Build dedupe index from recurrent jsonl; tail-only at terminal plateau."""
        index: dict = {}
        if not os.path.isfile(jsonl_path):
            return index
        tail_only = self._terminal_recurrent_level_fast_path()
        try:
            if tail_only:
                from collections import deque

                with open(jsonl_path, encoding="utf-8") as _f:
                    lines = deque(_f, maxlen=MAX_RECURRENT_ENTRIES)
                for _line in lines:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _e = json.loads(_line)
                    except json.JSONDecodeError:
                        continue
                    sha = _e.get("champion_sha256")
                    if sha:
                        index[sha] = None
            else:
                with open(jsonl_path, encoding="utf-8") as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _e = json.loads(_line)
                        except json.JSONDecodeError:
                            continue
                        sha = _e.get("champion_sha256")
                        if sha:
                            index[sha] = None
        except Exception:  # noqa: BLE001, S110
            pass
        return index

    def _should_skip_terminal_recurrent_feedback(self) -> bool:
        """At saturated L30, collect retrain traces every N generations."""
        if not self._terminal_recurrent_level_fast_path():
            return False
        interval = max(1, int(TERMINAL_RECURRENT_GENERATION_INTERVAL))
        if self.champion is not None and int(getattr(self, "champion_level", 0)) >= int(
            getattr(self.config, "max_level", 30)
        ):
            from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor

            if terminal_efficiency_at_scan_floor(len(self.champion)):
                interval = max(interval, 10)
        if interval <= 1:
            return False
        gen = int(getattr(self, "current_generation", 0))
        return gen % interval != 0

    def _cap_terminal_recurrent_candidates(self, candidates: list) -> list:
        """Keep top fitness traces per cycle at saturated L30 (retrain only)."""
        if not self._terminal_recurrent_level_fast_path():
            return candidates
        cap = MAX_TERMINAL_RECURRENT_PER_CYCLE
        if self.champion is not None and int(getattr(self, "champion_level", 0)) >= int(
            getattr(self.config, "max_level", 30)
        ):
            from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor

            if terminal_efficiency_at_scan_floor(len(self.champion)):
                cap = min(cap, 24)
        if len(candidates) <= cap:
            return candidates
        return sorted(candidates, key=lambda item: item[1], reverse=True)[:cap]

    def _apply_novelty_weights(self, candidates, weights, jsonl_path):
        """Phase 6: Apply novelty weighting to training examples."""

        if not candidates:

            return weights, False

        if self._terminal_recurrent_level_fast_path():

            return weights, False

        if not hasattr(self, "_novelty_anchors"):

            self._novelty_anchors, self._novelty_anchors_file_size = [], 0

        try:

            cfs = os.path.getsize(jsonl_path) if os.path.isfile(jsonl_path) else 0

        except OSError:

            cfs = 0

        if cfs > self._novelty_anchors_file_size or not self._novelty_anchors:

            na = []

            try:

                if os.path.isfile(jsonl_path):

                    with open(jsonl_path) as f:

                        for line in f:

                            try:

                                e = json.loads(line)

                                ph = e.get("champion_program", [])

                                if ph:

                                    pw = [int(h, 16) for h in ph]

                                    if len(pw) >= 3:

                                        na.append(pw)

                                    if len(na) >= 200:

                                        break

                            except (json.JSONDecodeError, ValueError):

                                continue

            except Exception:  # noqa: BLE001, S110

                pass

            self._novelty_anchors, self._novelty_anchors_file_size = na, cfs

        if not self._novelty_anchors:

            return weights, False

        ns = []

        for prog, *_ in candidates:

            md = float("inf")

            for a in self._novelty_anchors:

                xb = sum((prog[j] ^ a[j]).bit_count() for j in range(min(len(prog), len(a))))

                tb = (xb + abs(len(prog) - len(a)) * 8) / max(1, len(prog) + len(a))

                md = min(md, tb)

            ns.append(md if md != float("inf") else 0.5)

        mx = max(ns) if ns else 1.0

        nw = [max(0.1, math.sqrt(s / mx)) if mx > 1e-6 else 1.0 for s in ns]

        if weights is not None:

            for i in range(len(weights)):

                weights[i] *= nw[i]

            ws = sum(weights)

            if ws > 0:

                weights = [w / ws for w in weights]

        return weights, True

    def _write_recurrent_examples(self, candidates, weights, level_data, jsonl_path):
        """Phase 2: Write training examples with fitness+diversity+novelty weights."""

        saved_any = False
        lines: list[str] = []

        for i, (prog, challenger_fitness, trace, sha256) in enumerate(candidates):

            solved_level = level_data[i][0] if level_data else self.champion_level

            task_score = level_data[i][1] if level_data else 0.0

            example = {
                "level": solved_level,
                "generation": self.current_generation,
                "fitness": challenger_fitness,
                "champion_sha256": sha256,
                "champion_words": len(prog),
                "champion_program": [hex(w) for w in prog],
                "trace_status": trace.get("status", ""),
                "trace_instr_count": trace.get("instructions_count", 0),
                "stalled_count": self.generations_stalled,
                "task_score": task_score,
                "timestamp": time.time(),
                "trace_quality_score": min(1.0, max(0.0, challenger_fitness / 100.0)),
                "training_weight": weights[i] if weights else 1.0,
            }

            self._saved_sha256s[sha256] = None

            if len(self._saved_sha256s) > 5000:

                keys = list(self._saved_sha256s.keys())

                self._saved_sha256s = {k: None for k in keys[-3000:]}

            lines.append(json.dumps(example) + "\n")

            self.recurrent_examples_count += 1

            saved_any = True

        if lines:
            with open(jsonl_path, "a") as f:
                f.writelines(lines)

        return saved_any

    def _try_recurrent_feedback(self) -> bool:
        """Save model-generated programs that execute successfully to training data."""

        if self._should_skip_terminal_recurrent_feedback():

            return False

        if not self._last_challengers:

            return False

        recur_dir = self.config.recurrent_feedback_dir

        if not recur_dir:

            recur_dir = MILESTONE1_DIR

        if getattr(self, "_recur_dir_made", None) != recur_dir:

            os.makedirs(recur_dir, exist_ok=True)

            self._recur_dir_made = recur_dir

        jsonl_path = os.path.join(recur_dir, "recurrent_data.jsonl")

        if not hasattr(self, "_saved_sha256s"):

            self._saved_sha256s = self._load_saved_sha256_index(jsonl_path)

        if not hasattr(self, "_saved_size_counts"):

            self._saved_size_counts = {}

        fitnesses = self._last_challenger_fitnesses

        n_fitnesses = len(fitnesses)

        _candidates, rejected_count, _levels, _verify_levels_sorted = self._collect_recurrent_candidates(
            fitnesses, n_fitnesses
        )

        _candidates = self._cap_terminal_recurrent_candidates(_candidates)

        _level_data, _weights, _n_levels_found = self._compute_level_weights(
            _candidates, _levels, _verify_levels_sorted
        )

        _weights, _novelty_weight_applied = self._apply_novelty_weights(_candidates, _weights, jsonl_path)

        saved_any = self._write_recurrent_examples(_candidates, _weights, _level_data, jsonl_path)

        if saved_any:

            self._log(
                f"Recurrent feedback saved ({self.recurrent_examples_count} total, {len(self._saved_sha256s)} unique, novelty={_novelty_weight_applied}, levels={_n_levels_found})"
            )

        if rejected_count > 0:

            self._log(f"Rejected {rejected_count} low-quality traces")

        return self.recurrent_examples_count >= 50

    def _finalize_recurrent_buffer(self) -> None:
        """Evict stale recurrent traces once per evolution run (not per generation)."""
        recur_dir = self.config.recurrent_feedback_dir
        if not recur_dir:
            recur_dir = MILESTONE1_DIR
        jsonl_path = os.path.join(recur_dir, "recurrent_data.jsonl")
        self._evict_recurrent_buffer(jsonl_path)

    def _evict_recurrent_buffer(self, jsonl_path: str) -> None:
        """Evict lowest-fitness traces when recurrent buffer exceeds MAX_RECURRENT_ENTRIES.



        Reads all entries, sorts by fitness (ascending), removes the bottom

        entries to bring count below MAX_RECURRENT_ENTRIES, then rewrites.

        This ensures only high-quality traces remain for training.

        """

        if not os.path.isfile(jsonl_path):

            return

        try:

            line_count = 0

            with open(jsonl_path, "rb") as f:

                for chunk in iter(lambda: f.read(1 << 20), b""):

                    line_count += chunk.count(b"\n")

            if line_count <= MAX_RECURRENT_ENTRIES:

                return

            entries = []

            with open(jsonl_path, encoding="utf-8") as f:

                for line in f:

                    line = line.strip()

                    if line:

                        try:

                            entries.append(json.loads(line))

                        except json.JSONDecodeError:

                            continue

            if len(entries) <= MAX_RECURRENT_ENTRIES:

                return

            # Sort by fitness ascending — lowest fitness first for eviction

            entries.sort(key=lambda e: e.get("fitness", 0.0))

            # Keep only the top MAX_RECURRENT_ENTRIES entries

            evicted_count = len(entries) - MAX_RECURRENT_ENTRIES

            entries = entries[evicted_count:]

            # Rewrite file with kept entries (atomic write)

            import tempfile

            dir_name = os.path.dirname(jsonl_path) or "."

            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")

            try:

                with os.fdopen(fd, "w") as f:

                    for entry in entries:

                        f.write(json.dumps(entry) + "\n")

                os.replace(tmp_path, jsonl_path)

                self._log(
                    f"Evicted {evicted_count} low-fitness traces from recurrent buffer "
                    f"(kept {len(entries)} entries)"
                )

            except Exception:

                # Clean up temp file on failure

                try:

                    os.unlink(tmp_path)

                except OSError:

                    pass

                raise

        except (RuntimeError, OSError) as e:

            self._log(f"[RecurrentEvict] Failed: {e}")

    # ------------------------------------------------------------------

    # Strategy archive

    # ------------------------------------------------------------------

    def _strategy_outcome_top_k(self, pair_count: int) -> int:
        """How many top challengers count as strategy successes for this generation."""

        if pair_count <= 0:

            return 0

        if getattr(self, "generations_stalled", 0) >= 15:

            return max(1, pair_count // 5)

        return max(1, pair_count // 10)

    def _record_strategy_outcomes(self) -> None:
        """Record strategy outcomes in the strategy archive.



        Matches each challenger's strategy (from _last_challenger_strategies)

        against its cached stage-1 fitness. Top 10% by fitness are recorded as

        successes, the rest as failures. This gives the archive fine-grained

        signal even when no challenger beats the champion.



        Uses cached stage-1 fitnesses from _select_elite to avoid recomputing

        fitness on every challenger a second time.



        R13: Single-pass threshold computation avoids heapq.nsmallest overhead

        and list slicing. For N=50, this is ~2x faster than the previous approach.

        """

        strategies = self._last_challenger_strategies

        fitnesses = self._last_challenger_fitnesses

        if not strategies or not fitnesses:

            return

        # Guard: truncate to shorter length

        pair_count = min(len(strategies), len(fitnesses))

        if pair_count == 0:

            return

        # Single-pass threshold: find the k-th largest fitness using

        # a partial sort (heapq.nlargest with a small k) without copying

        # the list. For N=50, k=5, heapq.nsmallest is ~3x faster than

        # a full sort.

        k = self._strategy_outcome_top_k(pair_count)

        if k >= pair_count:

            # Edge case: all are "top"

            threshold = min(fitnesses[:pair_count]) if pair_count > 0 else 0.0

        else:

            # Use a heap of size k (avoids full list slice + copy)

            top_k = []

            for i in range(pair_count):

                fi = fitnesses[i]

                if len(top_k) < k:

                    heapq.heappush(top_k, fi)

                elif fi > top_k[0]:

                    heapq.heapreplace(top_k, fi)

            threshold = top_k[0]

        # R15: Batch record all outcomes in one call (single bulk append

        # instead of per-record append + per-record index update).

        sa = self.strategy_archive

        sa.refresh_timestamp()

        gen = self.current_generation

        lvl = self.champion_level

        batch = []

        champ_s1 = getattr(self, "_champion_stage1_fitness", None)

        batch_append = batch.append

        for i in range(pair_count):

            beats_champion = champ_s1 is not None and fitnesses[i] > champ_s1

            batch_append(
                {
                    "strategy": strategies[i],
                    "level": lvl,
                    "success": fitnesses[i] >= threshold or beats_champion,
                    "gen": gen,
                }
            )

        sa.record_batch(batch)

    # ------------------------------------------------------------------

    # Persistence

    # ------------------------------------------------------------------
