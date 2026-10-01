"""SelectionMixin - Elite selection, champion management, Hall of Fame.

Extracted from EvolutionEngine in evolution_engine.py.
Uses mixin pattern: all methods access self.* from the host EvolutionEngine.
"""

from __future__ import annotations

import heapq
import importlib.util
import os

_HAS_QD = importlib.util.find_spec("qd_archive") is not None

from evolution.shared import _CRASH_STATUSES  # noqa: E402
from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS
from mutation_proposer import trim_dead_code  # noqa: E402
from tasks_rv32i import LEVELS  # noqa: E402


def reuse_accepted_trace(champion, accepted_program, existing_trace) -> bool:
    """True when the accepted elite program was not trimmed, so its eval trace is valid."""
    if not isinstance(existing_trace, dict):
        return False
    return list(champion or []) == list(accepted_program or [])


class SelectionMixin:
    """Mixin providing elite selection, champion verification, and HoF management."""

    def _verify_score_from_trace(self, trace: dict) -> float:
        """Partial verify score for a challenger trace at the current level."""
        if trace.get("status") in ("invalid", "error"):
            return 0.0
        try:
            if self.task_instance is None:
                self._refresh_task_instance()
            lvl = self.champion_level
            inst = getattr(self, "_partial_verify_level_inst", None)
            if inst is None or getattr(self, "_partial_verify_level_n", None) != lvl:
                from tasks_rv32i import get_task_level

                inst = get_task_level(lvl)(seed=lvl)
                self._partial_verify_level_inst = inst
                self._partial_verify_level_n = lvl
            output = {
                "registers": trace.get("final_regs", {}),
                "memory": trace.get("final_memory", {}),
            }
            return float(inst.verify(output, self.task_instance))
        except (RuntimeError, TypeError, ValueError, KeyError, AttributeError):
            return 0.0

    def _compute_challenger_fitnesses(self, challengers, results, stage1_budget, raw_only):
        """Compute fitness for all challengers. Returns (scored, fitness_cache)."""

        n_challengers = len(challengers)

        scored = [None] * n_challengers

        fitness_cache = [0.0] * n_challengers

        _compute = self._compute_fitness

        _use_sec = self.config.use_secondary_tasks

        # Per-challenger secondary eval is unused by batch_compute_fitness
        # (it only reads engine._secondary_task_results). Skip it.
        self._secondary_task_results_per_program = None

        try:

            batch_fitnesses, _ = self.batch_compute_fitness(
                challengers, results, eval_budget=stage1_budget, raw_only=raw_only
            )

            for idx in range(n_challengers):

                scored[idx] = (batch_fitnesses[idx], idx, challengers[idx], results[idx])

                fitness_cache[idx] = batch_fitnesses[idx]

        except RuntimeError:

            _status_get = dict.get

            for idx in range(n_challengers):

                trace = results[idx]

                status = _status_get(trace, "status", "")

                if status in _CRASH_STATUSES:

                    fit = -100.0

                elif status == "timeout" or _status_get(trace, "instructions_count", 0) >= stage1_budget:

                    fit = -25.0

                else:

                    if _use_sec:

                        self._secondary_task_results = self._evaluate_secondary_tasks(challengers[idx])

                    else:

                        self._secondary_task_results = None

                    fit = _compute(challengers[idx], trace, eval_budget=stage1_budget, raw_only=raw_only)

                scored[idx] = (fit, idx, challengers[idx], results[idx])

                fitness_cache[idx] = fit

        finally:

            self._secondary_task_results_per_program = None

        return scored, fitness_cache

    def _tournament_select(self, scored, elite_size, n_scored):
        """Fill remaining elite slots via adaptive tournament selection."""

        rng = self.rng

        base_tourney = self.config.tournament_size

        div = self._current_diversity

        if div >= 0.6:

            adaptive_tourney = base_tourney + 2

        elif div >= 0.4:

            adaptive_tourney = base_tourney + 1

        elif div >= 0.2:

            adaptive_tourney = base_tourney

        else:

            adaptive_tourney = max(2, base_tourney - 1)

        tourney_size = min(adaptive_tourney, n_scored)

        _scored_get = scored.__getitem__

        for _ in range(elite_size):

            best_i = rng.randint(0, n_scored - 1)

            best_f = _scored_get(best_i)[0]

            for _t in range(tourney_size - 1):

                ci = rng.randint(0, n_scored - 1)

                cf = _scored_get(ci)[0]

                if cf > best_f:

                    best_f = cf

                    best_i = ci

            yield _scored_get(best_i), best_i

    def _select_elite(self, challengers, results):
        """Select elite challengers for detailed evaluation."""

        import heapq

        stage1_budget = self.config.eval_budget // 2

        len(challengers)

        raw_only = self.champion_level >= self.config.max_level - 1

        scored, fitness_cache = self._compute_challenger_fitnesses(challengers, results, stage1_budget, raw_only)

        self._last_challenger_fitnesses = fitness_cache

        from evolution.terminal_stagnation import terminal_verify_promote_active, terminal_verify_selection_bonus

        if terminal_verify_promote_active() and self.champion_level >= self.config.max_level:
            max_pop_verify = 0.0
            best_pop_fit = float("-inf")
            best_pop_prog = None
            best_pop_trace = None
            from evolution.fitness_helpers import selection_fitness

            weighted: list[tuple[float, int, list, dict]] = []
            for fit, idx, prog, trace in scored:
                cand_verify = self._verify_score_from_trace(trace)
                rank_fit = (
                    float(selection_fitness(self, prog, trace, verify_score=cand_verify))
                    if cand_verify >= 0.99 - 1e-6
                    else float(fit)
                )
                better = False
                if cand_verify > max_pop_verify + 1e-6:
                    better = True
                elif cand_verify + 1e-6 >= max_pop_verify:
                    if rank_fit > best_pop_fit + 1e-6:
                        better = True
                    elif abs(rank_fit - best_pop_fit) <= 1.0 and best_pop_prog is not None:
                        better = len(prog) < len(best_pop_prog)
                if better:
                    max_pop_verify = cand_verify
                    best_pop_fit = rank_fit
                    best_pop_prog = prog
                    best_pop_trace = trace
                weighted.append(
                    (
                        rank_fit if cand_verify >= 0.99 - 1e-6 else fit + terminal_verify_selection_bonus(cand_verify),
                        idx,
                        prog,
                        trace,
                    )
                )
            scored = weighted
            self._last_max_pop_verify = max_pop_verify
            self._last_max_pop_program = best_pop_prog
            self._last_max_pop_trace = best_pop_trace
        else:
            self._last_max_pop_verify = 0.0
            self._last_max_pop_program = None
            self._last_max_pop_trace = None

        if self.champion is not None and self.champion_trace is not None:

            champ_fits, _ = self.batch_compute_fitness(
                [self.champion], [self.champion_trace], eval_budget=stage1_budget, raw_only=raw_only
            )

            self._champion_stage1_fitness = champ_fits[0]

        else:

            self._champion_stage1_fitness = None

        elite_size = max(1, int(self.config.population_size * self.config.elite_ratio))

        from evolution.terminal_stagnation import terminal_elite_heap_key
        from evolution.terminal_stagnation import terminal_verify_promote_active as _tvp

        if _tvp() and self.champion_level >= self.config.max_level:
            top_k = heapq.nsmallest(elite_size, scored, key=lambda x: terminal_elite_heap_key(x[0], x[2]))
        else:
            top_k = heapq.nsmallest(elite_size, scored, key=lambda x: -x[0])

        elites = [prog for _, _, prog, _ in top_k]

        elite_fits = [fit for fit, _, _, _ in top_k]

        elite_indices = [idx for _, idx, _, _ in top_k]

        seen_fps = {(len(p) << 32) ^ (p[0] if p else 0) ^ ((p[-1] << 16) if len(p) > 1 else 0) for p in elites}

        seen_indices = set(elite_indices)

        n_scored = len(scored)

        for (_, _, win_prog, _), best_i in self._tournament_select(scored, elite_size, n_scored):

            fp = (
                (len(win_prog) << 32)
                ^ (win_prog[0] if win_prog else 0)
                ^ ((win_prog[-1] << 16) if len(win_prog) > 1 else 0)
            )

            if fp not in seen_fps and best_i not in seen_indices:

                seen_fps.add(fp)

                seen_indices.add(best_i)

                elites.append(win_prog)

                elite_fits.append(scored[best_i][0])

                elite_indices.append(best_i)

        elites, elite_fits, elite_indices = self._ensure_semantic_compress_elite(elites, elite_fits, elite_indices)
        return elites, elite_fits, elite_indices

    def _ensure_semantic_compress_elite(self, elites, elite_fits, elite_indices):
        """Keep a shorter semantic clone of the champion in the detailed-eval elite."""
        if not self.champion or int(self.champion_level) < int(self.config.max_level):
            return elites, elite_fits, elite_indices
        from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS, semantic_compress

        champ_verify = float(getattr(self, "_champion_verify_score", 0.0))
        champ_len = len(self.champion)
        if champ_verify < 0.99 - 1e-6:
            return elites, elite_fits, elite_indices
        # Compact L30 selection already ignores semantic_compress picks (see
        # saturated_shorter_swap_ok path); prepending duplicates GPU eval.
        if champ_len <= L30_COMPACT_MAX_WORDS:
            compact = None
        else:
            compact = semantic_compress(list(self.champion))
        if compact and len(compact) < champ_len and champ_len > 24:
            key = tuple(compact)
            if not any(tuple(p) == key for p in elites):
                elites = [compact] + list(elites)
                elite_fits = [0.0] + list(elite_fits)
                elite_indices = [-1] + list(elite_indices)
        champ_key = tuple(self.champion)
        if not any(tuple(p) == champ_key for p in elites):
            from evolution.fitness_helpers import selection_fitness

            elites.append(list(self.champion))
            elite_fits.append(
                float(
                    selection_fitness(
                        self,
                        self.champion,
                        self.champion_trace or {},
                        verify_score=champ_verify,
                    )
                )
            )
            elite_indices.append(-2)
        return elites, elite_fits, elite_indices

    def _scan_population_verify_candidates(
        self,
        champ_verify: float,
        verify_best: tuple[float, float, list, dict, float] | None,
    ) -> tuple[float, float, list, dict, float] | None:
        """Scan full population for verify-led promotion at terminal depth."""
        from evolution.terminal_stagnation import (
            effective_verify_margin,
            terminal_verify_promote_active,
            update_verify_best_candidate,
            verify_led_candidate_eligible,
        )

        challengers = getattr(self, "_last_challengers", None)
        results = getattr(self, "_last_results", None)
        fitness_cache = getattr(self, "_last_challenger_fitnesses", None)
        if not challengers or not results:
            return verify_best
        margin = effective_verify_margin()
        promote = terminal_verify_promote_active()
        stored_verify = float(getattr(self, "_champion_verify_score", 0.0))
        for idx, (prog, trace) in enumerate(zip(challengers, results, strict=False)):
            status = trace.get("status", "")
            cand_verify = self._verify_score_from_trace(trace)
            if not verify_led_candidate_eligible(
                champ_verify,
                cand_verify,
                margin,
                champion_certified_verify=stored_verify,
            ):
                continue
            stage1_fit = (
                fitness_cache[idx] if fitness_cache and idx < len(fitness_cache) else self._compute_fitness(prog, trace)
            )
            if promote and (status in _CRASH_STATUSES or status == "timeout" or stage1_fit <= -10.0):
                trace = self._evaluate(prog)
                status = trace.get("status", "")
                cand_verify = self._verify_score_from_trace(trace)
                if not verify_led_candidate_eligible(
                    champ_verify,
                    cand_verify,
                    margin,
                    champion_certified_verify=stored_verify,
                ):
                    continue
            elif status in _CRASH_STATUSES or status == "timeout":
                continue
            from evolution.fitness_helpers import selection_fitness

            fitness = selection_fitness(self, prog, trace, verify_score=cand_verify)
            champ_fit = (
                selection_fitness(self, self.champion, self.champion_trace or trace, verify_score=champ_verify)
                if self.champion is not None
                else self.champion_fitness
            )
            if self.champion is None:
                continue
            verify_best = update_verify_best_candidate(
                self,
                verify_best,
                cand_verify,
                fitness,
                prog,
                trace,
                getattr(self, "_raw_task_score", 0.0),
                champion_verify=champ_verify,
                champion_program=self.champion,
                champion_trace=self.champion_trace or trace,
                champion_fitness=champ_fit,
            )
        return verify_best

    def _find_best_challenger(self, elite, detailed_results, stage1_fitnesses):
        """Find the best challenger that beats the champion. Returns (best_fitness, best_program, best_trace, best_raw_score, improved)."""

        best_program = None

        best_trace = None

        best_raw_score = getattr(self, "_raw_task_score", 0.0)

        best_rank = float("-inf")

        improved = False

        _compute = self._compute_fitness

        _use_sec = self.config.use_secondary_tasks

        _use_rank = self.champion_level >= 12

        from evolution.fitness_helpers import selection_fitness
        from evolution.terminal_stagnation import (
            effective_verify_margin,
            terminal_verify_promote_active,
            update_verify_best_candidate,
        )

        verify_promote = terminal_verify_promote_active()
        verify_best: tuple[float, float, list, dict, float] | None = None
        champ_verify = float(getattr(self, "_champion_verify_score", 0.0))
        if self.champion_trace is not None:
            live = float(self._verify_score_from_trace(self.champion_trace))
            if live > 0.0:
                champ_verify = live
        if verify_promote and self.champion is not None and self.champion_trace is not None:
            best_fitness = float(
                selection_fitness(
                    self,
                    self.champion,
                    self.champion_trace,
                    verify_score=champ_verify,
                )
            )
        else:
            best_fitness = self.champion_fitness
        margin = effective_verify_margin()

        to_score: list[tuple[int, list, dict]] = []
        for idx, (prog, trace) in enumerate(zip(elite, detailed_results, strict=False)):
            if (
                not verify_promote
                and stage1_fitnesses
                and idx < len(stage1_fitnesses)
                and stage1_fitnesses[idx] < best_fitness
            ):
                continue
            to_score.append((idx, prog, trace))
        sec_by_j = None
        if _use_sec and to_score:
            sec_by_j = self._evaluate_secondary_tasks_many([p for _, p, _ in to_score])

        for j, (idx, prog, trace) in enumerate(to_score):

            if _use_sec:

                self._secondary_task_results = sec_by_j[j]

            else:

                self._secondary_task_results = None

            fitness = selection_fitness(self, prog, trace)

            rank = fitness

            if _use_rank:
                from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor

                at_scan_floor = (
                    self.champion is not None
                    and terminal_efficiency_at_scan_floor(len(self.champion))
                    and champ_verify >= 0.99 - 1e-6
                )
                if not at_scan_floor:
                    from evolution.fitness_helpers import challenger_rank_from_trace

                    rank = challenger_rank_from_trace(self, prog, trace, fitness)

            if verify_promote:
                cand_verify = self._verify_score_from_trace(trace)
                champ_fit = (
                    selection_fitness(self, self.champion, self.champion_trace or trace, verify_score=champ_verify)
                    if self.champion is not None
                    else self.champion_fitness
                )
                if self.champion is not None:
                    verify_best = update_verify_best_candidate(
                        self,
                        verify_best,
                        cand_verify,
                        fitness,
                        prog,
                        trace,
                        getattr(self, "_raw_task_score", 0.0),
                        champion_verify=champ_verify,
                        champion_program=self.champion,
                        champion_trace=self.champion_trace or trace,
                        champion_fitness=champ_fit,
                    )

            if rank > best_rank:
                replace = self._should_replace_champion(fitness)
                if replace and verify_promote:
                    cand_verify = self._verify_score_from_trace(trace)
                    if cand_verify < champ_verify - 0.001:
                        replace = False
                    if (
                        self.champion is not None
                        and 12 <= len(self.champion) <= L30_COMPACT_MAX_WORDS
                        and len(prog) < len(self.champion)
                        and cand_verify <= champ_verify + 1e-6
                    ):
                        replace = False
                if replace:
                    best_rank = rank

                    best_fitness = fitness

                    best_program = prog

                    best_trace = trace

                    best_raw_score = getattr(self, "_raw_task_score", 0.0)

                    improved = True

        if verify_promote and champ_verify >= 0.99 - 1e-6 and self.champion is not None:
            from evolution.terminal_stagnation import (
                inherit_verify_if_semantic_clone,
                saturated_shorter_capability_swap_ok,
                semantic_compress,
            )

            champ_len = len(self.champion)
            champ_trace = self.champion_trace
            compact_key = (
                tuple(semantic_compress(list(self.champion))) if champ_len > L30_COMPACT_MAX_WORDS else tuple()
            )
            for prog, trace in zip(elite, detailed_results, strict=False):
                if prog and tuple(prog) == tuple(self.champion):
                    champ_trace = trace
                    break
            pick = None
            for prog, trace in zip(elite, detailed_results, strict=False):
                if not prog or len(prog) >= champ_len:
                    continue
                cv = self._verify_score_from_trace(trace)
                is_compress = tuple(prog) == compact_key and len(prog) >= 12
                if is_compress and champ_len <= L30_COMPACT_MAX_WORDS:
                    continue
                cv = inherit_verify_if_semantic_clone(champ_len, len(prog), champ_trace, trace, cv, champ_verify)
                if is_compress:
                    cv = max(cv, champ_verify)
                if is_compress:
                    swap_ok = True
                else:
                    swap_ok = saturated_shorter_capability_swap_ok(
                        self,
                        list(self.champion),
                        champ_trace or trace,
                        list(prog),
                        trace,
                        champion_verify=champ_verify,
                        candidate_verify=cv,
                    )
                if not swap_ok:
                    continue
                if pick is None or len(prog) < len(pick[0]):
                    pick = (prog, trace, cv)
            if pick is not None:
                prog, trace, cv = pick
                best_program = prog
                best_trace = trace
                best_fitness = selection_fitness(self, prog, trace, verify_score=cv)
                best_raw_score = getattr(self, "_raw_task_score", 0.0)
                improved = True

        if verify_promote:
            verify_best = self._scan_population_verify_candidates(champ_verify, verify_best)

        verify_best_applied = False
        if verify_best is not None:
            from evolution.terminal_stagnation import verify_promotion_beats_champion

            champ_fit = (
                selection_fitness(
                    self,
                    self.champion,
                    self.champion_trace or verify_best[3],
                    verify_score=champ_verify,
                )
                if self.champion is not None
                else float(self.champion_fitness)
            )
            beats_champ = self.champion is not None and verify_promotion_beats_champion(
                self,
                self.champion,
                self.champion_trace or verify_best[3],
                verify_best[2],
                verify_best[3],
                champ_fit,
                verify_best[1],
                champion_verify=champ_verify,
                candidate_verify=verify_best[0],
            )
            apply_verify_best = beats_champ and (not improved or verify_best[1] + 1e-6 >= best_fitness)
            if apply_verify_best:
                best_rank = verify_best[0]
                best_fitness = verify_best[1]
                best_program = verify_best[2]
                best_trace = verify_best[3]
                best_raw_score = verify_best[4]
                improved = True
                verify_best_applied = True

        from evolution.terminal_stagnation import cached_best_compact_l30_fill, terminal_efficiency_at_scan_floor

        want = None
        at_scan_floor = self.champion is not None and terminal_efficiency_at_scan_floor(len(self.champion))
        if self.champion and not verify_best_applied and not at_scan_floor:
            cached = cached_best_compact_l30_fill(tuple(int(x) for x in self.champion))
            want = list(cached) if cached else None
        if want:
            pools = (
                zip(elite, detailed_results, strict=False),
                zip(
                    getattr(self, "_last_challengers", None) or [],
                    getattr(self, "_last_results", None) or [],
                    strict=False,
                ),
            )
            found = False
            for pool in pools:
                for prog, trace in pool:
                    if prog and tuple(prog) == tuple(want):
                        best_program = prog
                        best_trace = trace
                        best_fitness = selection_fitness(self, prog, trace)
                        best_raw_score = getattr(self, "_raw_task_score", 0.0)
                        improved = True
                        found = True
                        break
                if found:
                    break
            if not found:
                dummy = {"status": "success", "registers": {}}
                best_program = list(want)
                best_trace = dummy
                best_fitness = selection_fitness(self, best_program, dummy)
                best_raw_score = 0.0
                improved = True

        if (
            improved
            and self.champion is not None
            and best_program is not None
            and 12 <= len(self.champion) <= L30_COMPACT_MAX_WORDS
            and len(best_program) < len(self.champion)
        ):
            from evolution.terminal_stagnation import heldout_l30_verify_mean

            shorter_heldout = heldout_l30_verify_mean(list(best_program))
            champ_heldout = heldout_l30_verify_mean(list(self.champion))
            cv = float(self._verify_score_from_trace(best_trace) if best_trace else 0.0)
            from evolution.terminal_stagnation import compact_curriculum_beats

            curriculum_loss = compact_curriculum_beats(list(best_program), list(self.champion))
            if (cv <= champ_verify + 1e-6 and shorter_heldout + 1e-6 < champ_heldout) or (
                shorter_heldout + 1e-6 >= champ_heldout and curriculum_loss
            ):
                improved = False
                best_program = None

        return best_fitness, best_program, best_trace, best_raw_score, improved

    def _try_end_of_run_curriculum_push(self) -> bool:
        """End-of-run advance: chain level-ups while verify gates pass (parabolic curriculum)."""

        if self.champion_level >= min(len(LEVELS) - 1, self.config.max_level):

            return False

        saved_stall = self.generations_stalled

        self.generations_stalled = max(saved_stall, self.config.curriculum_stall_threshold)

        advanced_any = False

        max_steps = int(os.environ.get("EE_END_RUN_MAX_ADVANCE", "3"))

        min_fitness = float(os.environ.get("EE_END_RUN_MIN_FITNESS", "80"))

        try:

            for step in range(max_steps):

                self.generations_stalled = max(self.config.curriculum_stall_threshold, self.generations_stalled)

                _pre_verify, _pre_score = self._verify_champion()

                from evolution.progress_controller import ProgressController

                if not ProgressController.passes_certify_verify(self.champion_level, _pre_score):

                    self._log(
                        f"[AntiPlateau] End-of-run chain halted at L{self.champion_level}: "
                        f"verify {_pre_score:.3f} < certify bar "
                        f"{ProgressController.certify_verify_threshold(self.champion_level):.3f}"
                    )

                    break

                advanced = self._try_curriculum_advance() or self._try_forced_curriculum_escape()

                if not advanced:

                    break

                advanced_any = True

                self._log(
                    f"[AntiPlateau] End-of-run advance step {step + 1} -> L{self.champion_level} "
                    f"(F={self.champion_fitness:.1f}, verify={getattr(self, '_champion_verify_score', 0):.2f})"
                )

                if self.champion_fitness < min_fitness:

                    self._log(f"[AntiPlateau] Chain halted: fitness {self.champion_fitness:.1f} < {min_fitness}")

                    break

        finally:

            self.generations_stalled = saved_stall

        return advanced_any

    def _champion_verify_accepts(self, verify_score: float) -> bool:

        pc = getattr(self, "progress_controller", None)

        if pc is not None:

            return pc.champion_verify_accepts(verify_score, self.champion_level)

        if verify_score >= 1.0:

            return True

        threshold = _curriculum_verify_threshold_fn(
            self.champion_level,
            max(self.generations_stalled, self.config.curriculum_stall_threshold),
            self.config.curriculum_stall_threshold,
        )

        return verify_score >= threshold

    def _verify_and_replace_champion(self, best_program, best_trace, best_raw_score, best_fitness):
        """Verify challenger and replace champion if verified. Returns True if replaced."""

        _old_champion = self.champion
        _old_champion_trace = self.champion_trace
        current_verify = (
            float(self._verify_score_from_trace(_old_champion_trace)) if _old_champion_trace is not None else 0.0
        )
        champ_words = len(_old_champion) if _old_champion else 0
        self.champion = list(best_program)
        self.champion_trace = best_trace
        verified, verify_score = self._verify_champion()

        from evolution.terminal_stagnation import inherit_verify_if_semantic_clone, l30_easy_fill_variants
        from evolution.terminal_stagnation import semantic_compress as _semantic_compress
        from evolution.terminal_stagnation import traces_observably_equal

        fills = l30_easy_fill_variants(list(_old_champion)) if _old_champion else []
        fill_keys = {tuple(item) for item in fills}
        semantic_clone = (
            _old_champion is not None
            and len(best_program) < champ_words
            and len(best_program) >= 12
            and tuple(best_program) == tuple(_semantic_compress(list(_old_champion)))
        )
        easy_fill = tuple(best_program) in fill_keys
        if semantic_clone:
            verify_score = max(
                inherit_verify_if_semantic_clone(
                    champ_words,
                    len(best_program),
                    _old_champion_trace,
                    best_trace,
                    float(verify_score),
                    current_verify,
                ),
                current_verify,
            )
            verified = bool(verified) or float(verify_score) >= 0.99 - 1e-6
            self._log(
                f"[Champion] Semantic clone inherit verify={float(verify_score):.3f} "
                f"({champ_words}w -> {len(best_program)}w)"
            )

        escape_timeout = self._champion_is_timeout_penalized() and best_fitness > self.champion_fitness

        def _reject() -> bool:
            self.champion = _old_champion
            self.champion_trace = _old_champion_trace
            return False

        terminal_promote = bool(semantic_clone or easy_fill)
        if self.champion_level >= getattr(self.config, "max_level", 30):
            from evolution.fitness_helpers import selection_fitness
            from evolution.terminal_stagnation import terminal_verify_promote_active, verify_promotion_beats_champion

            champ_fit = (
                float(
                    selection_fitness(
                        self,
                        _old_champion,
                        _old_champion_trace or best_trace,
                        verify_score=current_verify,
                    )
                )
                if _old_champion is not None
                else float(getattr(self, "champion_fitness", 0.0))
            )
            certified_length_swap = (
                current_verify >= 0.99 - 1e-6 and len(best_program) < champ_words and best_fitness > champ_fit + 1e-6
            )
            if terminal_verify_promote_active() and (
                certified_length_swap
                or (
                    _old_champion is not None
                    and verify_promotion_beats_champion(
                        self,
                        _old_champion,
                        _old_champion_trace or best_trace,
                        best_program,
                        best_trace,
                        champ_fit,
                        best_fitness,
                        champion_verify=current_verify,
                        candidate_verify=float(verify_score),
                    )
                )
            ):
                terminal_promote = True
                self._log(f"[Champion] Terminal verify-led swap " f"{current_verify:.3f} -> {float(verify_score):.3f}")

        heldout_gain = False
        if (
            self.champion_level >= getattr(self.config, "max_level", 30)
            and _old_champion is not None
            and 12 <= champ_words <= L30_COMPACT_MAX_WORDS
        ):
            from evolution.terminal_stagnation import heldout_l30_verify_mean

            champ_heldout = heldout_l30_verify_mean(list(_old_champion))
            cand_heldout = heldout_l30_verify_mean(list(best_program))
            if cand_heldout + 1e-6 < champ_heldout:
                self._log(
                    f"[Champion] Held-out L30 protect: challenger " f"{cand_heldout:.3f} < champ {champ_heldout:.3f}"
                )
                return _reject()
            heldout_gain = cand_heldout >= champ_heldout + 0.01
            if not heldout_gain and cand_heldout + 1e-6 >= champ_heldout:
                from evolution.terminal_stagnation import compact_curriculum_beats

                curriculum_gain = compact_curriculum_beats(list(_old_champion), list(best_program))
                if curriculum_gain and len(best_program) <= champ_words:
                    heldout_gain = True
            if not heldout_gain and cand_heldout + 1e-6 >= champ_heldout and len(best_program) == champ_words:
                from evolution.terminal_stagnation import get_scan_floor_gate

                if not get_scan_floor_gate(list(_old_champion)).qualifies(list(best_program)):
                    self._log("[Champion] Compact side-curriculum protect: " "same-length challenger regresses L8-L16")
                    return _reject()

        if (
            not heldout_gain
            and not easy_fill
            and self.champion_level >= getattr(self.config, "max_level", 30)
            and 12 <= champ_words <= L30_COMPACT_MAX_WORDS
            and current_verify >= 0.25
            and float(verify_score) < current_verify - 0.01
        ):
            self._log(
                f"[Champion] Compact held-out protect: challenger "
                f"{float(verify_score):.3f} < live {current_verify:.3f}"
            )
            return _reject()
        from evolution.champion_bloat_guard import certified_champion_pinned, champion_bloat_blocks

        if not heldout_gain and certified_champion_pinned(self.champion_level, champ_words, current_verify):
            if float(verify_score) <= current_verify + 0.005:
                from evolution.fitness_helpers import selection_fitness

                cand_fit = float(
                    selection_fitness(
                        self,
                        best_program,
                        best_trace,
                        verify_score=float(verify_score),
                    )
                )
                champ_fit = (
                    float(
                        selection_fitness(
                            self,
                            _old_champion,
                            _old_champion_trace or best_trace,
                            verify_score=current_verify,
                        )
                    )
                    if _old_champion is not None
                    else float(best_fitness)
                )
                from evolution.terminal_stagnation import scan_floor_min_raw_gain, verify_promotion_beats_champion

                min_gain = scan_floor_min_raw_gain(champ_words) if current_verify >= 0.99 - 1e-6 else 1.0

                length_efficiency_swap = False
                if (
                    current_verify >= 0.99 - 1e-6
                    and _old_champion is not None
                    and verify_promotion_beats_champion(
                        self,
                        _old_champion,
                        _old_champion_trace or best_trace,
                        best_program,
                        best_trace,
                        champ_fit,
                        cand_fit,
                        champion_verify=current_verify,
                        candidate_verify=float(verify_score),
                    )
                ):
                    length_efficiency_swap = True
                if not length_efficiency_swap and cand_fit <= champ_fit + min_gain:
                    self._log(
                        f"[Champion] Pinned certified L{self.champion_level} "
                        f"(verify {current_verify:.3f}, {champ_words}w)"
                    )
                    return _reject()

        if not terminal_promote and current_verify >= 0.99 and float(verify_score) < 0.99 - 1e-6:
            self._log(
                f"[Champion] Certified verify blocked: challenger {float(verify_score):.3f} "
                f"< champion {current_verify:.3f}"
            )
            return _reject()

        if (
            not heldout_gain
            and not easy_fill
            and float(verify_score) >= 0.99 - 1e-6
            and len(best_program) > champ_words + 1
        ):
            from evolution.fitness_helpers import selection_fitness

            cand_fit = float(selection_fitness(self, best_program, best_trace, verify_score=float(verify_score)))
            champ_fit = (
                float(
                    selection_fitness(
                        self,
                        _old_champion,
                        _old_champion_trace or best_trace,
                        verify_score=current_verify,
                    )
                )
                if _old_champion is not None
                else float(best_fitness)
            )
            if not terminal_promote and self.champion_level >= 29 and champ_words >= 60:
                self._log(
                    f"[Champion] Saturated L{self.champion_level} length pin: "
                    f"{len(best_program)}w > {champ_words}w at verify=1.0"
                )
                return _reject()
            if cand_fit < champ_fit * 0.9:
                self._log(
                    f"[Champion] Bloat blocked at verify=1.0: "
                    f"fit {cand_fit:.1f} < {champ_fit:.1f}, words {len(best_program)}>{champ_words}"
                )
                return _reject()

        if not terminal_promote and self.champion_level >= 20 and verify_score < current_verify - 0.005:
            self._log(
                f"[Champion] L{self.champion_level} blocked: challenger verify "
                f"{verify_score:.3f} < champion {current_verify:.3f}"
            )
            return _reject()

        if not terminal_promote and self.champion_level >= 17 and not verified and verify_score < current_verify:
            self._log(
                f"[Champion] L17+ blocked: challenger verify {verify_score:.3f} " f"< champion {current_verify:.3f}"
            )
            return _reject()

        bloat_reason = champion_bloat_blocks(
            self.champion_level,
            champ_words,
            len(best_program),
            float(verify_score),
            current_verify,
        )
        fill_ok = (semantic_clone or easy_fill) and len(best_program) >= 12
        if (
            fill_ok
            and len(best_program) > champ_words
            and current_verify >= 0.99 - 1e-6
            and self.champion_level >= 29
            and champ_words >= 60
        ):
            fill_ok = False
        if bloat_reason is not None and not heldout_gain and not fill_ok and not terminal_promote:
            self._log(
                f"[Champion] L{self.champion_level} bloat blocked: {bloat_reason} "
                f"(verify {verify_score:.3f} vs {current_verify:.3f})"
            )
            return _reject()

        if not verified and not escape_timeout and not terminal_promote:

            if self._champion_verify_accepts(verify_score):

                self._log(
                    f"[Champion] Partial verify: challenger {verify_score:.2f} "
                    f">= threshold {_curriculum_verify_threshold_fn(self.champion_level, max(self.generations_stalled, self.config.curriculum_stall_threshold), self.config.curriculum_stall_threshold):.3f}. Allowing replacement."
                )

                self._champion_partial_verify = verify_score

            else:

                self._log(
                    f"[Champion] BLOCKED (PM Rule #19): challenger verify failed ({verify_score:.2f} < threshold)."
                )

                self._champion_verified = False

                self._champion_verify_score = verify_score

                self._champion_partial_verify = 0.0

                return _reject()

        elif escape_timeout:

            self._log(f"[Champion] Timeout escape: fitness {self.champion_fitness:.1f} -> {best_fitness:.1f}")

        qd = getattr(self, "qd_archive", None)

        if (
            qd is not None
            and _HAS_QD
            and _old_champion is not None
            and _old_champion_trace is not None
            and not self._champion_changed_since_last_qd
        ):

            try:

                from evolution_engine import compute_qd_axes  # lazy to avoid circular import

                old_desc = compute_qd_axes(_old_champion, _old_champion_trace)

                if old_desc is not None:

                    qd.add_solution(
                        descriptor=old_desc,
                        fitness=self.champion_fitness,
                        program=_old_champion,
                        trace=_old_champion_trace,
                    )

            except (TypeError, ValueError, RuntimeError):

                pass

        self.champion = trim_dead_code(list(best_program))

        self.champion_trace = best_trace

        if hasattr(self, "_champion_raw_task_score"):

            self._champion_raw_task_score = best_raw_score

        if terminal_promote and current_verify >= 0.99 - 1e-6:
            verify_score = current_verify
            verified = True
            self._terminal_promoted_fitness = float(best_fitness)
            self._certified_fitness_floor = max(
                float(getattr(self, "_certified_fitness_floor", 0.0)),
                float(best_fitness),
            )

        self.champion_fitness = best_fitness

        self._champion_verified = verified

        self._champion_verify_score = verify_score

        self._champion_partial_verify = float(verify_score)

        try:
            from evolution.fitness_helpers import selection_fitness, terminal_certified_fitness_context

            with terminal_certified_fitness_context(self):
                if reuse_accepted_trace(self.champion, best_program, best_trace):
                    fresh_trace = best_trace
                else:
                    fresh_trace = self._evaluate(self.champion)
                self.champion_trace = fresh_trace
                self.champion_fitness = float(
                    selection_fitness(
                        self,
                        self.champion,
                        fresh_trace,
                        verify_score=float(verify_score),
                    )
                )
                if best_fitness > self.champion_fitness + 1e-6:
                    self.champion_fitness = float(best_fitness)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            pass

        self._champion_changed_since_last_qd = True

        self._champion_changed_since_last_novelty = True

        self._champion_qd_valid = False

        self._cached_champion_qd_desc = None

        if terminal_promote and self.champion_level >= getattr(self.config, "max_level", 30):
            try:
                import os

                from evolution.champion_carry_forward import persist_carry_forward_from_engine

                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                persist_carry_forward_from_engine(project_root, self)
            except (OSError, TypeError, ValueError, AttributeError):
                pass

        return True

    def _update_hall_of_fame(self, elite, stage1_fitnesses):
        """Update Hall of Fame with elite programs."""

        if not stage1_fitnesses or not elite:

            return

        n_f = len(stage1_fitnesses)

        hof_worst = self._hall_of_fame.worst_fitness

        if not isinstance(hof_worst, (int, float)):

            hof_worst = float("-inf")

        if max(stage1_fitnesses) < hof_worst:

            return

        k_top = max(1, n_f // 5)

        threshold_heap = []

        for fi in stage1_fitnesses:

            if len(threshold_heap) < k_top:

                heapq.heappush(threshold_heap, fi)

            elif fi > threshold_heap[0]:

                heapq.heapreplace(threshold_heap, fi)

        quality_threshold = threshold_heap[0] if threshold_heap else 0.0

        level = self.champion_level

        hof = self._hall_of_fame

        gen = getattr(self, "current_generation", 0)

        for idx, prog in enumerate(elite):

            if idx < n_f and stage1_fitnesses[idx] >= quality_threshold:

                hof.add(prog, stage1_fitnesses[idx], level, generation=gen)

    def _try_escape_timeout_champion(self, elite, detailed_results) -> bool:
        """When champion is stuck at -25, inject HOF or best elite that improves fitness."""

        best_prog = None

        best_trace = None

        best_fit = self.champion_fitness

        best_raw = getattr(self, "_champion_raw_task_score", 0.0)

        _compute = self._compute_fitness

        _use_sec = self.config.use_secondary_tasks

        for prog, trace in zip(elite, detailed_results, strict=False):

            if _use_sec:

                self._secondary_task_results = self._evaluate_secondary_tasks(prog)

            else:

                self._secondary_task_results = None

            fit = _compute(prog, trace)

            if fit > best_fit:

                best_fit = fit

                best_prog = prog

                best_trace = trace

                best_raw = getattr(self, "_raw_task_score", 0.0)

        for hof_prog in (
            self._hall_of_fame.get_best(level=self.champion_level),
            self._hall_of_fame.get_best(),
        ):

            if hof_prog is None:

                continue

            trace = self._evaluate(hof_prog)

            if _use_sec:

                self._secondary_task_results = self._evaluate_secondary_tasks(hof_prog)

            else:

                self._secondary_task_results = None

            fit = _compute(hof_prog, trace)

            if fit > best_fit:

                best_fit = fit

                best_prog = hof_prog

                best_trace = trace

                best_raw = getattr(self, "_raw_task_score", 0.0)

                break

        if best_prog is None or best_fit <= self.champion_fitness:

            return False

        self._log(f"[Champion] Timeout recovery candidate fitness={best_fit:.1f}")

        return self._verify_and_replace_champion(best_prog, best_trace, best_raw, best_fit)

    def _try_repair_terminal_champion_verify(self) -> bool:
        """Apply trace-guided register repair when terminal champion verify < 1.0."""
        level = int(self.champion_level)
        max_level = int(getattr(self.config, "max_level", 30))
        if level < max_level or self.champion is None or self.task_instance is None:
            return False
        trace = self.champion_trace
        if trace is None:
            trace = self._evaluate(self.champion)
        verify = float(self._verify_score_from_trace(trace))
        if verify >= 0.99 - 1e-6:
            return False
        from evolution.register_repair import trace_guided_immediate_variants

        variants = trace_guided_immediate_variants(
            self.champion,
            level,
            trace,
            self.task_instance,
            self.rng,
            max_evals=32,
        )
        best_prog = None
        best_trace = None
        best_verify = verify
        best_fit = float(self.champion_fitness)
        from evolution.fitness_helpers import selection_fitness

        for variant in variants:
            cand_trace = self._evaluate(variant)
            cand_verify = float(self._verify_score_from_trace(cand_trace))
            if cand_verify <= best_verify + 1e-6:
                continue
            cand_fit = float(selection_fitness(self, variant, cand_trace, verify_score=cand_verify))
            if cand_verify > best_verify or (cand_verify >= 0.99 - 1e-6 and cand_fit > best_fit):
                best_verify = cand_verify
                best_fit = cand_fit
                best_prog = variant
                best_trace = cand_trace
        if best_prog is None:
            return False
        self._log(f"[Champion] trace-guided repair {verify:.3f} -> {best_verify:.3f}")
        return self._verify_and_replace_champion(best_prog, best_trace, getattr(self, "_raw_task_score", 0.0), best_fit)

    def _maybe_promote_terminal_efficiency_champion(self) -> bool:
        """Promote a shorter fully-verified program at certified L30."""
        max_level = int(getattr(self.config, "max_level", 30))
        if self.champion_level < max_level or self.champion is None:
            self._log(f"[Champion] Terminal efficiency skip: " f"L{self.champion_level}<{max_level} or no champion")
            return False
        from evolution.terminal_stagnation import (
            compact_curriculum_side_score,
            heldout_l30_verify_mean,
            record_terminal_efficiency_promote,
            record_terminal_efficiency_scan_floor,
            record_terminal_efficiency_stable,
            saturated_shorter_swap_ok,
            terminal_efficiency_at_scan_floor,
            terminal_efficiency_drop_span,
            terminal_efficiency_eval_budget,
            terminal_efficiency_last_promote_cycle,
            terminal_efficiency_promote_cooldown_cycles,
            terminal_efficiency_promote_quiescent,
            terminal_efficiency_variants,
            terminal_verify_promote_active,
            verify_promotion_beats_champion,
        )

        if not terminal_verify_promote_active():
            self._log("[Champion] Terminal efficiency skip: verify promote inactive")
            return False
        cycle_no = int(getattr(self, "_cycle_number", 0))
        if getattr(self, "_terminal_efficiency_promote_attempted_cycle", None) == cycle_no:
            self._log("[Champion] Terminal efficiency skip: already attempted this cycle")
            return False
        self._terminal_efficiency_promote_attempted_cycle = cycle_no
        champ_words = len(self.champion)
        cooldown = terminal_efficiency_promote_cooldown_cycles()
        last_promote = max(
            int(getattr(self, "_terminal_efficiency_promote_cycle", 0)),
            terminal_efficiency_last_promote_cycle(),
        )
        if cooldown > 0 and last_promote > 0 and cycle_no > last_promote and (cycle_no - last_promote) < cooldown:
            from evolution.mutation_helpers import scan_floor_capability_growth_allowed

            if not scan_floor_capability_growth_allowed(champ_words):
                self._log(
                    f"[Champion] Terminal efficiency skip: cooldown "
                    f"({cycle_no - last_promote}/{cooldown} cycles since last promote)"
                )
                return False
        record_terminal_efficiency_stable(cycle_no, champ_words)
        compact_promote = 12 <= champ_words <= L30_COMPACT_MAX_WORDS
        from evolution.fitness_helpers import selection_fitness, terminal_certified_fitness_context

        with terminal_certified_fitness_context(self):
            trace = self.champion_trace
            if trace is None:
                trace = self._evaluate(self.champion)
            stored_verify = float(getattr(self, "_champion_verify_score", 0.0))
            champ_verify = max(float(self._verify_score_from_trace(trace)), stored_verify)
            from evolution.terminal_stagnation import terminal_efficiency_promote_skip_quiescent

            if terminal_efficiency_promote_skip_quiescent(cycle_no, champ_words, champ_verify):
                self._log(
                    f"[Champion] Terminal efficiency skip: stable {champ_words}w "
                    f"at verify>={champ_verify:.3f} (periodic rescan)"
                )
                return False
            if champ_verify < 0.99 - 1e-6:
                self._log(f"[Champion] Terminal efficiency skip: verify {champ_verify:.3f} < 0.99")
                return False
            if float(self._verify_score_from_trace(trace)) < 0.99 - 1e-6:
                trace = self._evaluate(self.champion)

            champ_fit = float(selection_fitness(self, self.champion, trace, verify_score=champ_verify))
            certified_floor = float(getattr(self, "champion_fitness", 0.0))
            if certified_floor > champ_fit + 1e-6:
                champ_fit = certified_floor
            from evolution.champion_bloat_guard import level_word_cap

            drop_span = (
                terminal_efficiency_drop_span(champ_words, level_word_cap(int(self.champion_level)))
                if compact_promote
                else 3
            )
            eval_budget = terminal_efficiency_eval_budget()
            variants = terminal_efficiency_variants(
                self.champion,
                self.rng,
                max_evals=eval_budget,
                max_drop_span=drop_span,
            )
            if compact_promote:
                from evolution.mutation_helpers import scan_floor_capability_growth_allowed
                from evolution.terminal_stagnation import (
                    _scan_floor_capability_stalled,
                    ranked_scan_floor_compression_variants,
                )

                if scan_floor_capability_growth_allowed(champ_words):
                    compression = ranked_scan_floor_compression_variants(
                        list(self.champion),
                        cycle_no=cycle_no,
                    )
                    seen_variants = {tuple(variant) for variant in variants}
                    for comp in compression:
                        key = tuple(comp)
                        if key not in seen_variants:
                            variants.insert(0, comp)
                            seen_variants.add(key)
            if compact_promote and variants:
                from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

                # Shorter first; within equal length prefer higher promotion_raw.
                variants = sorted(
                    variants,
                    key=lambda item: (
                        len(item),
                        -float(compact_l30_promotion_raw_fitness(list(item))),
                    ),
                )[:eval_budget]
            best_prog = None
            best_trace = None
            best_fit = champ_fit
            best_raw = -1.0
            champ_side = None
            champ_heldout = None
            if compact_promote:
                champ_key = tuple(self.champion)
                if getattr(self, "_cached_compact_champ_key", None) == champ_key:
                    champ_side = float(self._cached_compact_champ_side)
                    champ_heldout = float(self._cached_compact_champ_heldout)
                else:
                    champ_side = float(compact_curriculum_side_score(list(self.champion)))
                    champ_heldout = float(heldout_l30_verify_mean(list(self.champion), 32))
                    self._cached_compact_champ_key = champ_key
                    self._cached_compact_champ_side = champ_side
                    self._cached_compact_champ_heldout = champ_heldout
            for variant in variants:
                if len(variant) >= champ_words:
                    continue
                if compact_promote:
                    cand_heldout = float(heldout_l30_verify_mean(variant, 32))
                    if cand_heldout + 1e-6 < float(champ_heldout):
                        continue
                    if compact_curriculum_side_score(variant) < float(champ_side) - 1e-6:
                        continue
                cand_trace = self._evaluate(variant)
                cand_verify = float(self._verify_score_from_trace(cand_trace))
                cand_fit = float(
                    selection_fitness(
                        self,
                        variant,
                        cand_trace,
                        verify_score=max(cand_verify, champ_verify),
                    )
                )
                certified_swap = (
                    champ_verify >= 0.99 - 1e-6 and len(variant) < champ_words and cand_fit > champ_fit + 1e-6
                )
                beats = verify_promotion_beats_champion(
                    self,
                    self.champion,
                    trace,
                    variant,
                    cand_trace,
                    champ_fit,
                    cand_fit,
                    champion_verify=champ_verify,
                    candidate_verify=cand_verify,
                )
                qualifies = beats
                if not qualifies:
                    continue
                from evolution.fitness_helpers import promotion_raw_fitness

                cand_raw = float(
                    promotion_raw_fitness(
                        self,
                        variant,
                        cand_trace,
                        verify_score=cand_verify,
                    )
                )
                take = best_prog is None
                if not take and cand_raw > best_raw + 1e-6:
                    take = True
                elif not take and abs(cand_raw - best_raw) <= 1e-6 and len(variant) < len(best_prog):
                    take = True
                elif (
                    not take
                    and abs(cand_raw - best_raw) <= 1e-6
                    and len(variant) == len(best_prog)
                    and cand_fit > best_fit + 1e-6
                ):
                    take = True
                if take:
                    best_fit = cand_fit
                    best_raw = cand_raw
                    best_prog = variant
                    best_trace = cand_trace
            if compact_promote:
                from evolution.mutation_helpers import (
                    scan_floor_capability_growth_allowed,
                    scan_floor_growth_completion_qualifies,
                )

                if scan_floor_capability_growth_allowed(champ_words):
                    from evolution.champion_rollback import collect_growth_material_parents
                    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
                    from evolution.terminal_stagnation import scan_floor_min_raw_gain

                    champ_pr = float(compact_l30_promotion_raw_fitness(list(self.champion)))
                    min_gain = float(scan_floor_min_raw_gain(champ_words))
                    growth_cands: list[list[int]] = []
                    seen_growth: set[tuple[int, ...]] = set()
                    for src in list(variants) + collect_growth_material_parents(
                        list(self.champion),
                        max_parents=max(8, eval_budget),
                    ):
                        key = tuple(src)
                        if key in seen_growth or len(src) <= champ_words:
                            continue
                        seen_growth.add(key)
                        if not scan_floor_growth_completion_qualifies(self.champion, src):
                            continue
                        if float(compact_l30_promotion_raw_fitness(list(src))) <= champ_pr + min_gain - 1e-6:
                            continue
                        growth_cands.append(list(src))
                    growth_cands.sort(
                        key=lambda p: (
                            -float(compact_l30_promotion_raw_fitness(list(p))),
                            len(p),
                        )
                    )
                    for variant in growth_cands[:eval_budget]:
                        cand_trace = self._evaluate(variant)
                        cand_verify = float(self._verify_score_from_trace(cand_trace))
                        cand_fit = float(
                            selection_fitness(
                                self,
                                variant,
                                cand_trace,
                                verify_score=max(cand_verify, champ_verify),
                            )
                        )
                        if not verify_promotion_beats_champion(
                            self,
                            self.champion,
                            trace,
                            variant,
                            cand_trace,
                            champ_fit,
                            cand_fit,
                            champion_verify=champ_verify,
                            candidate_verify=cand_verify,
                        ):
                            continue
                        from evolution.fitness_helpers import promotion_raw_fitness

                        cand_raw = float(
                            promotion_raw_fitness(
                                self,
                                variant,
                                cand_trace,
                                verify_score=cand_verify,
                            )
                        )
                        take = best_prog is None
                        if not take and cand_raw > best_raw + 1e-6:
                            take = True
                        elif not take and abs(cand_raw - best_raw) <= 1e-6 and len(variant) < len(best_prog):
                            take = True
                        elif (
                            not take
                            and abs(cand_raw - best_raw) <= 1e-6
                            and len(variant) == len(best_prog)
                            and cand_fit > best_fit + 1e-6
                        ):
                            take = True
                        if take:
                            best_fit = cand_fit
                            best_raw = cand_raw
                            best_prog = variant
                            best_trace = cand_trace
            if best_prog is None:
                if compact_promote and champ_heldout is not None:
                    record_terminal_efficiency_scan_floor(cycle_no, champ_words, float(champ_heldout))
                self._log(
                    f"[Champion] Terminal efficiency: no qualifying variant "
                    f"(candidates={len(variants)}, champ={champ_words}w, verify={champ_verify:.3f})"
                )
                return False
            self._log(
                f"[Champion] Terminal efficiency {champ_words}w -> {len(best_prog)}w "
                f"raw {best_raw:.3f} fit {champ_fit:.3f} -> {best_fit:.3f}"
            )
            promoted = self._verify_and_replace_champion(
                best_prog, best_trace, getattr(self, "_raw_task_score", 0.0), best_fit
            )
            if promoted:
                self._terminal_efficiency_promote_cycle = cycle_no
                record_terminal_efficiency_promote(cycle_no)
                new_words = len(self.champion)
                new_heldout = float(heldout_l30_verify_mean(list(self.champion), 32))
                record_terminal_efficiency_scan_floor(cycle_no, new_words, new_heldout)
                self._cached_compact_champ_key = None
            return promoted

    def _maybe_promote_max_pop_verify_champion(self) -> bool:
        """When population beats champion on verify, adopt that program."""
        if self.champion_level < int(getattr(self.config, "max_level", 30)):
            return False
        max_pop = float(getattr(self, "_last_max_pop_verify", 0.0))
        if max_pop < 0.99 - 1e-6:
            return False
        champ_verify = float(getattr(self, "_champion_verify_score", 0.0))
        if self.champion_trace is not None:
            champ_verify = float(self._verify_score_from_trace(self.champion_trace))
        if champ_verify >= max_pop - 1e-6:
            return False
        prog = getattr(self, "_last_max_pop_program", None)
        trace = getattr(self, "_last_max_pop_trace", None)
        if prog is None or trace is None:
            return False
        from evolution.fitness_helpers import selection_fitness

        fitness = float(selection_fitness(self, prog, trace, verify_score=max_pop))
        champ_fit = float(
            selection_fitness(
                self,
                self.champion,
                self.champion_trace or trace,
                verify_score=champ_verify,
            )
        )
        if max_pop >= 0.99 - 1e-6 and fitness < champ_fit * 0.85 and len(prog) >= len(self.champion):
            return False
        self._log(f"[Champion] max_pop verify promote {champ_verify:.3f} -> {max_pop:.3f}")
        return self._verify_and_replace_champion(prog, trace, getattr(self, "_raw_task_score", 0.0), fitness)

    def _maybe_restore_best_certified_champion(self) -> bool:
        """Restore the newest fully-verified champion when current regressed."""
        level = int(self.champion_level)
        max_level = int(getattr(self.config, "max_level", 30))
        if level < max_level or self.champion is None:
            return False
        if len(self.champion) >= 12:
            return False
        import os

        from evolution.champion_rollback import find_best_certified_champion
        from evolution.fitness_helpers import selection_fitness, terminal_certified_fitness_context

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results = os.path.join(root, "multi_cycle_results.jsonl")
        clean = find_best_certified_champion(results, level, min_verify=0.99, min_fitness=100.0)
        if not clean:
            return False
        clean_prog = [int(x, 16) if isinstance(x, str) else int(x) for x in clean["champion_program"]]
        with terminal_certified_fitness_context(self):
            cur_trace = self.champion_trace or self._evaluate(self.champion)
            cur_verify = max(
                float(self._verify_score_from_trace(cur_trace)),
                float(getattr(self, "_champion_verify_score", 0.0)),
            )
            cur_fit = float(selection_fitness(self, self.champion, cur_trace, verify_score=cur_verify))
            clean_trace = self._evaluate(clean_prog)
            clean_verify = float(self._verify_score_from_trace(clean_trace))
            logged_verify = float(clean.get("champion_verify_score", 0.0))
            logged_fit = float(clean.get("final_fitness", 0.0))
            if clean_verify < 0.99 - 1e-6 and logged_verify < 0.99 - 1e-6:
                return False
            effective_clean_verify = max(clean_verify, logged_verify)
            clean_fit = float(
                selection_fitness(
                    self,
                    clean_prog,
                    clean_trace,
                    verify_score=effective_clean_verify,
                )
            )
            if logged_fit > clean_fit:
                clean_fit = logged_fit
        stub_replace = len(self.champion) < 12 and len(clean_prog) >= 12
        if not stub_replace:
            if cur_verify >= 0.99 - 1e-6 and len(self.champion) < len(clean_prog) and len(self.champion) >= 12:
                return False
            if cur_verify >= 0.99 - 1e-6 and cur_fit >= clean_fit - 1.0 and len(self.champion) <= len(clean_prog):
                return False
        restore = (
            stub_replace
            or clean_fit > cur_fit + 1.0
            or (level >= max_level - 1 and cur_fit < 150.0 and clean_fit >= 200.0)
            or (
                not stub_replace
                and cur_verify >= 0.99 - 1e-6
                and len(self.champion) > len(clean_prog)
                and clean_fit >= cur_fit * 0.9
            )
        )
        if not restore:
            return False
        self._sync_champion_after_reseed(
            clean_prog,
            f"certified_cycle_{clean['cycle']}",
            (
                f"Restore certified C{clean['cycle']}: "
                f"fit {cur_fit:.1f}->{clean_fit:.1f}, "
                f"words {len(self.champion)}->{len(clean_prog)}"
            ),
            trusted_verify=effective_clean_verify,
            trusted_fitness=clean_fit,
        )
        return True

    def _update_champion(self, elite, detailed_results, stage1_fitnesses=None):
        """Update champion if any elite challenger has higher fitness."""

        best_fitness, best_program, best_trace, best_raw_score, improved = self._find_best_challenger(
            elite, detailed_results, stage1_fitnesses
        )

        if improved and best_program is not None:

            prev_words = len(self.champion) if self.champion is not None else 0
            prev_fit = float(self.champion_fitness)
            replaced = self._verify_and_replace_champion(best_program, best_trace, best_raw_score, best_fitness)
            improved = bool(replaced)

        elif self._champion_is_timeout_penalized():

            improved = self._try_escape_timeout_champion(elite, detailed_results)

        if not improved:
            improved = self._maybe_promote_max_pop_verify_champion()

        self._update_hall_of_fame(elite, stage1_fitnesses)

        self.best_fitness_history.append(self.champion_fitness)

        max_hist = getattr(self, "_max_history_len", 500)

        if len(self.best_fitness_history) > max_hist:

            self.best_fitness_history = self.best_fitness_history[-max_hist:]

        return improved
