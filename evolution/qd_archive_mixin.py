"""QDArchiveMixin - MAP-Elites QD archive and novelty archive management.

Extracted from EvolutionEngine in evolution_engine.py.
Uses mixin pattern: all methods access self.* from the host EvolutionEngine.
"""
from __future__ import annotations

import heapq
from typing import Any

import importlib.util
_HAS_QD = importlib.util.find_spec("qd_archive") is not None

from mutation_proposer import _crossover_mutation, trim_dead_code  # noqa: E402


class QDArchiveMixin:
    """Mixin providing QD archive and novelty archive operations."""

    def _try_qd_crossover(self) -> list[int] | None:

        """Attempt to breed a hybrid program from the QD archive.



        When stalled >= 8 gens and the QD archive has 2+ distinct solutions,

        perform a crossover between the top two fitness-ranked programs.

        Returns the hybrid program (with EBREAK) or None.



        This brings dormant archive diversity back into the active population.



        R33: Uses heapq.nlargest(2) for O(N log 2) top-2 selection instead

        of full O(N log N) sort. For a 100-cell QD grid this avoids sorting

        98 unnecessary elements every call.

        """

        if self.qd_archive is None:

            return None

        if self.generations_stalled < 8:

            return None



        try:

            solutions = self.qd_archive.grid_solutions()

            if len(solutions) < 2:

                return None



            # R39: Use heapq.nlargest(3) to find top-3 by fitness.

            # The old code used solutions[2] as fallback, but grid_solutions()

            # returns cells in grid order, not fitness order — index 2 is

            # NOT guaranteed to be the 3rd-best. This caused the crossover

            # to sometimes use a low-fitness fallback, producing hybrids

            # that were worse than expected.

            top3 = heapq.nlargest(3, solutions, key=lambda s: s["fitness"])

            best = top3[0]

            second = top3[1] if len(top3) > 1 else None



            if best["program"] == (second["program"] if second else []):

                second = top3[2] if len(top3) > 2 else None

                if second is None or best["program"] == second["program"]:

                    return None



            hybrid = _crossover_mutation(

                program=list(best["program"]),

                task_level=self.champion_level,

                rng=self.rng,

                champion=list(second["program"]),

                champion_crossover_prob=self.config.champion_crossover_prob,

            )

            if not hybrid:

                return None

            if hybrid[-1] != 0x0010_0073:

                hybrid.append(0x0010_0073)

            return hybrid

        except RuntimeError as e:

            self._log(f"[QD Cross] Crossover error: {e}")

            return None

    def _clear_scan_floor_growth_crossover_cache(self) -> None:
        """Drop per-cycle growth crossover pool (call at cycle start)."""
        self._scan_floor_growth_crossover_pool = None
        self._scan_floor_growth_crossover_pool_cycle = None

    def _try_scan_floor_growth_crossover(self) -> list[int] | None:
        """Blend champion with a held-out-safe growth fill when at scan floor."""
        if self.champion is None or len(self.champion) < 4:
            return None
        try:
            import os

            from evolution.mutation_helpers import (
                build_scan_floor_growth_crossover_pool,
                collect_scan_floor_growth_crossover_fills,
                scan_floor_anchor_program,
                scan_floor_breeding_active,
                scan_floor_growth_heldout_qualifies,
            )

            verify = float(getattr(self, "_champion_verify_score", 0.0))
            if not scan_floor_breeding_active(
                len(self.champion),
                level=int(self.champion_level),
                verify=verify,
            ):
                return None
            anchor = scan_floor_anchor_program(list(self.champion))
            champ = list(anchor)
            cycle_no = int(getattr(self, "_cycle_number", 0))
            gen_idx = int(getattr(self, "current_generation", 0))
            pool = getattr(self, "_scan_floor_growth_crossover_pool", None)
            pool_cycle = getattr(self, "_scan_floor_growth_crossover_pool_cycle", None)
            if pool is None or pool_cycle != cycle_no:
                fills = collect_scan_floor_growth_crossover_fills(anchor)
                if not fills:
                    return None
                max_attempts = max(
                    3,
                    int(os.environ.get("EE_SCAN_FLOOR_GROWTH_CROSSOVER_ATTEMPTS", "6")),
                )
                pool = build_scan_floor_growth_crossover_pool(
                    champ,
                    fills,
                    self.rng,
                    cycle_no=cycle_no,
                    max_attempts=max_attempts,
                )
                self._scan_floor_growth_crossover_pool = pool
                self._scan_floor_growth_crossover_pool_cycle = cycle_no
                if pool:
                    self._log(
                        f"[GrowthCross] Built {len(pool)} hybrid(s) for cycle {cycle_no}"
                    )
            if not pool:
                return None
            from evolution.mutation_helpers import scan_floor_growth_crossover_max_words
            from evolution.terminal_stagnation import get_scan_floor_gate

            gate = get_scan_floor_gate(champ)
            max_words = scan_floor_growth_crossover_max_words(len(champ))
            usable = [
                p
                for p in pool
                if len(p) <= max_words
                and (
                    len(p) == len(champ)
                    or scan_floor_growth_heldout_qualifies(champ, p)
                )
            ]
            if not usable:
                return None
            equal = [p for p in usable if len(p) == len(champ) and gate.qualifies(p)]
            pick_from = equal if equal else usable
            return list(pick_from[(cycle_no + gen_idx) % len(pick_from)])
        except (ImportError, RuntimeError, TypeError, ValueError):
            return None

    def _distinct_qd_cell_programs(self, *, longer_first: bool = False) -> list[list[int]]:
        """Archive programs that differ from the current champion."""
        qd = getattr(self, "qd_archive", None)
        if qd is None:
            return []
        try:
            solutions = qd.grid_solutions()
        except (AttributeError, RuntimeError, TypeError):
            return []
        champ = tuple(self.champion or [])
        seen = {champ}
        out: list[list[int]] = []
        for sol in solutions:
            prog = sol.get("program") if isinstance(sol, dict) else None
            if not isinstance(prog, list) or not prog:
                continue
            key = tuple(prog)
            if key in seen:
                continue
            seen.add(key)
            out.append(list(prog))
        out.sort(key=len, reverse=longer_first)
        return out

    def _pick_mutation_parent(self):
        """Sometimes mutate a MAP-Elites cell instead of only the champion.

        Same population size and eval coverage. Champion remains the default
        parent (~65%); archive parents are the shorter distinct cells.
        """
        champ = self.champion
        champ_trace = getattr(self, "champion_trace", None)
        champ_traces = [champ_trace] if champ_trace is not None else []
        if champ is None:
            return champ, champ_traces
        from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor

        at_floor = terminal_efficiency_at_scan_floor(len(champ))
        stall_floor = 1 if at_floor else 2
        if int(getattr(self, "generations_stalled", 0) or 0) < stall_floor:
            return champ, champ_traces
        rng = getattr(self, "rng", None)
        if rng is None:
            return champ, champ_traces

        from evolution.mutation_helpers import _curriculum_l10_cached

        champ_l10 = _curriculum_l10_cached(list(champ))
        if at_floor and champ_l10 < 0.85 - 1e-6:
            import os

            lift_prob = float(os.environ.get("EE_L10_LIFT_MUTATION_PARENT_PROB", "0.45"))
            if rng.random() < lift_prob:
                try:
                    from evolution.champion_rollback import (
                        collect_growth_material_compress_parents,
                    )

                    parents = collect_growth_material_compress_parents(
                        list(champ), max_parents=8
                    )
                    l10_parents = [
                        list(p)
                        for p in parents
                        if _curriculum_l10_cached(list(p)) > champ_l10 + 0.01
                    ]
                    if l10_parents:
                        l10_parents.sort(
                            key=lambda p: (
                                -_curriculum_l10_cached(list(p)),
                                len(p),
                            )
                        )
                        pick = min(6, len(l10_parents))
                        return (
                            list(l10_parents[rng.randrange(pick)]),
                            champ_traces,
                        )
                except ImportError:
                    pass

        archive_prob = 0.45 if at_floor else 0.35
        if rng.random() >= archive_prob:
            return champ, champ_traces
        qd = getattr(self, "qd_archive", None)
        pool: list[list[int]] = []
        if qd is not None:
            try:
                sols = [
                    s for s in qd.grid_solutions() if isinstance(s, dict) and s.get("program")
                ]
            except (AttributeError, RuntimeError, TypeError):
                sols = []
            champ_key = tuple(champ)
            others = [s for s in sols if tuple(s.get("program") or ()) != champ_key]
            others.sort(key=lambda s: len(s["program"]))
            pool = [list(s["program"]) for s in others[: max(1, min(8, len(others)))]]
        if not pool and hasattr(self, "_terminal_archive_parents"):
            pool = self._terminal_archive_parents()
        if not pool:
            return champ, champ_traces
        prog = list(pool[rng.randrange(len(pool))])
        return prog, champ_traces

    def _inject_qd_archive_hybrids(self, challengers: list) -> int:
        """Evaluate MAP-Elites diversity in the live population.

        Replaces the tail of ``challengers`` with (1) distinct archive
        programs, then (2) crossovers. Eval budget and fitness are unchanged.
        """
        if not challengers:
            return 0
        n_slots = min(6, max(1, len(challengers) // 5))
        injected = 0
        strategies = getattr(self, "_last_challenger_strategies", None)

        def _place(idx: int, prog: list[int], label: str) -> None:
            challengers[idx] = prog
            if isinstance(strategies, list) and idx < len(strategies):
                strategies[idx] = label

        from evolution.mutation_helpers import scan_floor_breeding_active

        at_floor = bool(
            self.champion
            and scan_floor_breeding_active(
                len(self.champion),
                level=int(self.champion_level),
                verify=float(getattr(self, "_champion_verify_score", 0.0)),
            )
        )
        cell_stall = 1 if at_floor else 4
        cross_stall = 2 if at_floor else 8
        cells = self._distinct_qd_cell_programs(longer_first=at_floor)
        if at_floor and cells and self.champion:
            from evolution.epi_momentum_governor import (
                promotion_raw_capability_stalled,
                scan_floor_length_prom_stalled,
            )
            from evolution.mutation_helpers import (
                _evolution_project_root,
                compress_growth_probe_pool,
                scan_floor_growth_compress_sources,
            )
            from evolution.terminal_stagnation import get_scan_floor_gate

            champ = list(self.champion)
            long_cells = [c for c in cells if len(c) > len(champ)]
            root = _evolution_project_root()
            stalled = (
                promotion_raw_capability_stalled(root)
                or scan_floor_length_prom_stalled(root, len(champ))
            )
            compress_src = scan_floor_growth_compress_sources(
                champ,
                long_cells,
                stalled=stalled,
                max_trials=max(4, n_slots * 2),
            )
            if compress_src:
                merged = compress_growth_probe_pool(
                    champ,
                    compress_src,
                    gate=get_scan_floor_gate(champ),
                )
                eq = [c for c in merged if len(c) == len(champ)]
                if eq:
                    rest = [c for c in cells if tuple(c) not in {tuple(x) for x in eq}]
                    cells = eq + rest
        n_cells = min(
            len(cells),
            max(1, n_slots // 2) if self.generations_stalled >= cell_stall else 0,
        )
        for i in range(n_cells):
            idx = len(challengers) - 1 - injected
            if idx < 0:
                break
            _place(idx, cells[i], "QD_CELL")
            injected += 1
        n_cross = n_slots - injected
        for _ in range(max(0, n_cross)):
            if self.generations_stalled < cross_stall:
                break
            hybrid = None
            label = "QD_CROSSOVER"
            if at_floor:
                hybrid = self._try_scan_floor_growth_crossover()
                if hybrid is not None:
                    label = "GROWTH_CROSSOVER"
            if hybrid is None:
                hybrid = self._try_qd_crossover()
            if not hybrid:
                break
            idx = len(challengers) - 1 - injected
            if idx < 0:
                break
            _place(idx, hybrid, label)
            injected += 1
        if injected:
            self._log(
                f"[QD] Injected {injected} archive program(s) into population"
            )
        return injected

    # ------------------------------------------------------------------

    # MAP-Elites QD archive (v13)

    # ------------------------------------------------------------------





    def _should_skip_qd_archive_update(self) -> bool:

        """Return True when QD archive update should not run this generation."""

        qd = self.qd_archive

        if qd is None or not _HAS_QD:

            return True

        if not self._champion_changed_since_last_qd:

            return True

        if self.champion is None or self.champion_trace is None:

            return True

        return self.config.qd_bonus_weight <= 0.0 and not self.config.use_qd_archive





    def _log_qd_descriptor_diagnostics(self, qd: Any, qd_desc: Any, was_added: bool) -> None:

        """Log QD descriptor range and cell-add events for diagnostics."""

        gen = self.current_generation

        if gen % 25 == 0 or was_added:

            self._log(

                f"[QD] Descriptor range gen={gen}: "

                f"inst_div=[{qd._descriptor_min[0]:+.2f}, {qd._descriptor_max[0]:+.2f}], "

                f"reg_press=[{qd._descriptor_min[1]:+.2f}, {qd._descriptor_max[1]:+.2f}], "

                f"bounds=[{qd._per_dim_lower[0]:+.2f}..{qd._per_dim_upper[0]:+.2f}, "

                f"{qd._per_dim_lower[1]:+.2f}..{qd._per_dim_upper[1]:+.2f}], "

                f"coverage={qd.coverage():.0%}, "

                f"rebins={qd._rebins}"

            )

        if was_added:

            self._log(

                f"[QD] Added champion to grid cell "

                f"(inst_div={qd_desc[0]:+.2f}, "

                f"reg_press={qd_desc[1]:+.2f}, "

                f"coverage={qd.coverage():.0%})"

            )





    def _update_qd_archive(self) -> None:

        """Extract champion descriptor and add to MAP-Elites QD archive.



        Called every generation after the champion has been updated.

        Silently skips if the QD archive is not initialised.



        R21: Skips entirely when champion hasn't changed since the last

        update, avoiding redundant QD axes recomputation every generation.

        """

        if self._should_skip_qd_archive_update():

            return

        champion = self.champion

        trace = self.champion_trace

        qd = self.qd_archive

        try:

            from evolution_engine import compute_qd_axes  # lazy to avoid circular import
            qd_desc = compute_qd_axes(champion, trace)

            # R31: Cache the computed descriptor for _compute_fitness niche penalty

            self._cached_champion_qd_desc = qd_desc

            self._champion_qd_valid = True

            if qd_desc is None:

                return

            was_added = qd.add_solution(

                descriptor=qd_desc,

                fitness=self.champion_fitness,

                program=champion,

                trace=trace,

            )

            self._qd_coverage = qd.coverage()

            self._log_qd_descriptor_diagnostics(qd, qd_desc, was_added)

        except (TypeError, ValueError, RuntimeError) as exc:

            self._log(f"[QD] Skipped: {exc}")





    def _attempt_qd_archive_champion_replace(self, qd_prog: list[int], archive_fitness: float) -> bool:

        """Evaluate, verify (R50), and optionally replace champion from QD archive."""

        qd_trace = self._evaluate(qd_prog)

        qd_fitness = self._compute_fitness(qd_prog, qd_trace)

        qd_discounted = qd_fitness * 0.9



        self._log(

            f"[QD Reset] Found archive candidate: "

            f"archive_fitness={archive_fitness:.4f}, "

            f"recomputed={qd_fitness:.4f}, "

            f"{len(qd_prog)} words (stalled={self.generations_stalled})"

        )



        if qd_discounted <= self.champion_fitness:

            self._log(

                f"[QD Reset] Candidate not better: "

                f"current={self.champion_fitness:.4f}, "

                f"qd={qd_discounted:.4f} -- keeping champion"

            )

            return False



        _old_champion = self.champion

        _old_trace = self.champion_trace

        _old_fitness = self.champion_fitness

        current_verify = float(getattr(self, "_champion_verify_score", 0.0))
        champ_words = len(_old_champion) if _old_champion else 0

        from evolution.champion_bloat_guard import certified_champion_pinned, champion_bloat_blocks

        bloat_reason = champion_bloat_blocks(
            self.champion_level,
            champ_words,
            len(qd_prog),
            0.0,
            current_verify,
        )
        if bloat_reason is not None:
            self._log(
                f"[QD Reset] BLOCKED bloat: {bloat_reason} "
                f"({len(qd_prog)}w vs champion {champ_words}w)"
            )
            return False

        self.champion = qd_prog

        self.champion_trace = qd_trace

        verified, verify_score = self._verify_champion()

        if certified_champion_pinned(self.champion_level, champ_words, current_verify):
            if float(verify_score) <= current_verify + 0.005:
                self.champion = _old_champion
                self.champion_trace = _old_trace
                self.champion_fitness = _old_fitness
                self._log(
                    f"[QD Reset] BLOCKED pinned L{self.champion_level} "
                    f"(verify {verify_score:.3f} vs {current_verify:.3f})"
                )
                return False

        if self.champion_level >= 20 and float(verify_score) < current_verify - 0.005:
            self.champion = _old_champion
            self.champion_trace = _old_trace
            self.champion_fitness = _old_fitness
            self._log(
                f"[QD Reset] BLOCKED verify regression "
                f"{verify_score:.3f} < {current_verify:.3f}"
            )
            return False

        bloat_reason = champion_bloat_blocks(
            self.champion_level,
            champ_words,
            len(qd_prog),
            float(verify_score),
            current_verify,
        )
        if bloat_reason is not None:
            self.champion = _old_champion
            self.champion_trace = _old_trace
            self.champion_fitness = _old_fitness
            self._log(f"[QD Reset] BLOCKED bloat after verify: {bloat_reason}")
            return False

        if not verified and verify_score <= 0.0:

            self.champion = _old_champion

            self.champion_trace = _old_trace

            self.champion_fitness = _old_fitness

            self._log(

                f"[QD Reset] BLOCKED (R50): verify failed "

                f"({verify_score:.2f}), keeping champion"

            )

            return False



        old_fitness = self.champion_fitness

        self.champion = trim_dead_code(qd_prog)

        self.champion_fitness = qd_fitness

        self._champion_verified = verified

        self._champion_verify_score = verify_score

        self.generations_stalled = 0

        self._champion_changed_since_last_qd = True

        self._champion_changed_since_last_novelty = True

        self._log(

            f"[QD Reset] Champion replaced: "

            f"{old_fitness:.4f} -> {qd_fitness:.4f} "

            f"(from QD archive, stall reset, verify={verify_score:.2f})"

        )

        self._champion_qd_valid = False

        self._cached_champion_qd_desc = None

        return True





    def _try_qd_champion_reset(self) -> bool:

        """Replace champion with best QD archive program when deeply stalled.



        When ``generations_stalled >= 50`` and the QD archive has at least

        one solution, finds the highest-fitness program in the archive and

        replaces the champion if the archive program (with a 0.9x task-

        specific discount) has higher discounted fitness.



        Only attempts once per stall episode (``_qd_champion_reset_tried``

        flag). Returns True if champion was replaced.



        This is a fallback for when the QD crossover and other diversity

        mechanisms have not been enough to escape deep stall.

        """

        if self.generations_stalled < 35:

            return False

        if self.qd_archive is None or not _HAS_QD:

            return False

        if self._qd_champion_reset_tried:

            return False



        self._qd_champion_reset_tried = True



        try:

            solutions = self.qd_archive.grid_solutions()

            if not solutions:

                self._log("[QD Reset] QD archive empty -- no champion to inject")

                return False



            best_qd = max(solutions, key=lambda s: s["fitness"])

            return self._attempt_qd_archive_champion_replace(

                list(best_qd["program"]), best_qd["fitness"]

            )

        except RuntimeError as e:

            self._log(f"[QD Reset] Error: {e}")

            return False



    # ------------------------------------------------------------------

    # Novelty archive (v13)

    # ------------------------------------------------------------------





    def _should_skip_novelty_archive_update(self) -> bool:

        """Return True when novelty archive update should not run this generation."""

        if self.novelty_archive is None:

            return True

        if not self._champion_changed_since_last_novelty:

            return True

        if self.champion is None or self.champion_trace is None:

            return True

        return self._extract_behavioral_descriptor is None





    def _build_novelty_archive_metadata(self, champion: list[int], trace: dict[str, Any]) -> dict[str, Any]:

        """Build metadata payload stored alongside a novelty archive descriptor."""

        return {

            "generation": self.current_generation,

            "level": self.champion_level,

            "fitness": self.champion_fitness,

            "champion_size": len(champion),

            "instructions_count": trace.get("instructions_count", 0),

            "status": trace.get("status", ""),

        }





    def _manage_novelty_archive_capacity(self) -> None:

        """Evict low-novelty entries when the archive exceeds capacity."""

        if self.novelty_archive is None:

            return

        try:

            removed = self.novelty_archive.manage_capacity()

            if removed > 0:

                self._log(f"[Novelty] Capacity management: removed {removed} low-novelty entries")

        except Exception:  # noqa: BLE001, S110

            pass





    def _update_novelty_archive(self) -> None:

        """Extract champion descriptor and add to the novelty archive.



        Called every generation after the champion has been updated.

        Silently skips if the novelty archive is not initialised.



        R21: Skips when champion unchanged since last update (avoids

        redundant behavioral descriptor extraction).

        R39: Uses dedicated _champion_changed_since_last_novelty flag

        instead of the QD flag, so novelty and QD archives update

        independently.

        """

        if self._should_skip_novelty_archive_update():

            return

        champion = self.champion

        trace = self.champion_trace

        novelty_archive = self.novelty_archive

        try:

            descriptor = self._extract_behavioral_descriptor(

                trace,

                champion,

                budget=self.config.eval_budget,

            )

            metadata = self._build_novelty_archive_metadata(champion, trace)

            was_added, score = novelty_archive.add_to_archive(

                descriptor,

                metadata=metadata,

                check_threshold=novelty_archive.size > 0,

            )

            if was_added:

                self._log(f"[Novelty] Added descriptor (score={score:.4f}, archive_size={novelty_archive.size})")

            else:

                self._log(f"[Novelty] Below threshold (score={score:.4f})")

        except (TypeError, ValueError, RuntimeError) as exc:

            self._log(f"[Novelty] Skipped: {exc}")

        self._manage_novelty_archive_capacity()

    def _fill_empty_qd_cells_from_evaluated(
        self, programs, traces, strategies: list[str] | None = None
    ) -> int:
        """Insert evaluated programs into empty MAP-Elites cells only.

        Uses existing traces (no extra VM eval). Does not replace occupied
        cells, so champion quality is unchanged while illumination grows
        enough for QD crossover to fire.
        """
        try:
            return self._fill_empty_qd_cells_from_evaluated_inner(
                programs, traces, strategies=strategies
            )
        except Exception as exc:  # noqa: BLE001 — must not abort the generation loop
            try:
                self._log(f"[QD] Population fill skipped: {type(exc).__name__}: {exc}")
            except Exception:  # noqa: BLE001
                pass
            return 0

    _QD_FILL_SKIP_STRATEGIES = frozenset({"DIVERSITY_INJECT", "TERMINAL_VERIFY_PROBE"})

    def _fill_empty_qd_cells_from_evaluated_inner(
        self, programs, traces, strategies: list[str] | None = None
    ) -> int:
        qd = getattr(self, "qd_archive", None)
        if qd is None or not _HAS_QD or not programs or not traces:
            return 0
        use_qd = bool(getattr(self.config, "use_qd_archive", False))
        bonus = float(getattr(self.config, "qd_bonus_weight", 0.0) or 0.0)
        if bonus <= 0.0 and not use_qd:
            return 0
        from evolution_engine import compute_qd_axes

        added = 0
        n = min(len(programs), len(traces))
        step = 1 if n <= 64 else max(1, n // 64)
        bins = max(2, int(getattr(qd, "bins", 20) or 20))
        skip = self._QD_FILL_SKIP_STRATEGIES
        for i in range(0, n, step):
            if strategies is not None and i < len(strategies) and strategies[i] in skip:
                continue
            prog = programs[i]
            trace = traces[i]
            if not isinstance(prog, list) or not isinstance(trace, dict):
                continue
            if trace.get("status") in ("invalid", "error"):
                continue
            champ = getattr(self, "champion", None)
            if champ and int(getattr(self, "champion_level", 0)) >= 30:
                try:
                    from evolution.champion_rollback import (
                        _growth_material_storage_anchor,
                        record_scan_floor_growth_material,
                    )

                    anchor = _growth_material_storage_anchor(list(champ))
                    if len(prog) > len(anchor):
                        record_scan_floor_growth_material(
                            list(champ),
                            list(prog),
                            cycle=int(getattr(self, "_cycle_number", 0)),
                        )
                except (ImportError, TypeError, ValueError, OSError):
                    pass
            desc = compute_qd_axes(prog, trace)
            if desc is None:
                continue
            uniq_words = [w for w in prog if isinstance(w, int)]
            uniq = len(set(uniq_words)) / max(len(prog), 1)
            dither = (uniq - 0.5) * (2.0 / bins)
            desc = [max(-1.0, min(1.0, float(desc[0]) + dither)), float(desc[1])]
            if not qd.cell_is_empty(desc):
                continue
            # Illumination rank only (not curriculum fitness): occupied
            # champion cells keep their real scores; empty cells get a
            # cheap behavioral proxy so crossover parents are not all ties at 0.
            regs = trace.get("final_regs") or {}
            n_regs = 0
            if isinstance(regs, dict):
                n_regs = sum(1 for v in regs.values() if v)
            n_ops = len({w & 0x7F for w in uniq_words})
            fit = float(n_regs) + 0.01 * float(n_ops)
            if qd.add_solution(desc, fit, list(prog), trace):
                added += 1
        if added:
            self._qd_coverage = qd.coverage()
            self._log(
                f"[QD] Filled {added} empty cells from evaluated population "
                f"(coverage={qd.coverage():.0%})"
            )
        return added






