# v51 Evolution Engine — Audit Fixes Applied

## All Fixes (completed 2026-06-22)

### V51 Fixes (from original session)
- Fix A: champion_fitness sentinel -inf → 0.0
- Fix 1: Level-aware graduated bonus (0.95→0.85 threshold)
- Fix B: Curriculum threshold 140→120
- Fix 2: Strategy archive recording (was no-op pass)
- Fix 3: GPU cleanup (gc.collect before empty_cache, end-of-cycle)
- Fix 4: Trace gate (only blocks novelty-inflated positives)

### Crash Bootstrap Fixes
- `_pick_value()`: produces signed i32 values now (kernel rejected unsigned >2^31)
- `_run_one_evaluation()`: `if init_mem:` → `if init_mem is not None:` (empty dict was falsy)
- `_refresh_task_instance()`: `level_cls(seed).generate_instance()` not `get_task_level(seed=...)`
- `reference_solution()`: no kwargs passed (levels don't accept count/load_offset)
- `champion_anchor.save()`: `os.makedirs(dir, exist_ok=True)` before writing
- `_raw_task_score`: reset to 0.0 on crash/timeout returns (prevented stale score from advancing curriculum)

### Audit Fixes (supplemental)
- `_encode_inst`: now checks `"name"` key first, then `"opcode"` (reference solutions use `"name"`)
- `_eval_result_to_trace`: removed hardcoded `"fitness": 0.0`, fixed status: `"ok" if success else "crash"`
- Strategy archive: computes real fitness via `_compute_fitness()` instead of reading 0.0 from trace
- Level 10 reference: `SLTU` → `SLT` for signed max-finding
- Phase 1 Knowledge Injection: reference solutions for L8-L13 added to training data
- Task instance refresh: every 25 gens (already implemented)
- `or True` debug leftover removed from GPU logging
- GPU batch acceleration disabled (dict→EvalResult conversion broken in GPU vm_farm)

### Current Run: v51 48-cycle
- **PID**: 1454358
- **Started**: 2026-06-22 15:56 EDT
- **Config**: --cycles 48 --gens 300 --retrain-max-epochs 12 --retrain-max-per-level 200 --ref-high-augment 20
- **GPU**: Disabled (CPU evaluation, stable)
- **Phase 1**: 20 reference solutions injected per cycle
- **Status**: Running, will notify on completion

---

## 2026-08-28 — Retrain-loop verification + test-suite fixes

Scope: address the two highest-priority open items — (1) prove the
evolve->retrain->evolve loop actually produces a real `val_loss` (null across
all ~1,534 historical cycles per ADVERSARIAL_AUDIT_2026_07_17.md), and (2) fix
the two deterministic unit-test failures found during environment setup.

### Fixes applied
- **meta_controller.py**: `MetaController.__init__` defaulted
  `checkpoint_acceptance_tolerance` to `1.0`, but `config.py` (and the retrain
  gate's intended behaviour) use `1.05`. No production caller
  (`run_multi_cycle.py`, `autonomous_runner.py`, `cron_run_10.py`) passed the
  kwarg, so the documented tolerance was dead — checkpoints within the 5%
  regression band were wrongly rejected. Aligned the constructor default to
  `1.05`. Fixes `test_round9_coverage.py::TestAcceptCheckpointEdgeCases::
  test_val_loss_within_tolerance`.
- **evolution_engine.py**: `run()` opens the lazy log handle during champion
  initialisation (via `self._log`) BEFORE the generation loop's `try/finally`.
  An exception during init (e.g. `_verify_champion`) leaked the handle. Wrapped
  champion-init in a `try/except` that calls `_close_log_handle()` and
  re-raises (init errors still propagate; they are not converted to a partial
  summary). Fixes `test_round6_improvements.py::TestCloseLogHandle::
  test_run_always_closes_log_handle_on_exception`.

### New tooling
- **init_checkpoint.py**: bootstraps a fresh (untrained) RV32I transformer to
  `milestone1/checkpoints/checkpoint_{best,current}.pt` in the format expected
  by both `model_factory.load` (retrain) and `PrefixGenerator` (model-guided
  evolution). The multi-cycle/autonomous entry points abort without a seed
  checkpoint, so a clean clone could not run the loop at all. `medium` preset
  (default) => 8.92M params, 35.7 MB.

### Verification
- Targeted tests: 271 passed (round6/round9/round13/round3), 0 regressions.
- 3-cycle retrain smoke test (`run_multi_cycle.py --cycles 3 --gens 12
  --retrain-max-epochs 1`, CPU): cycle 1 recorded `retrained: true`,
  `val_loss=3.3071` (first non-null val_loss in the project's history).
  Full 3-cycle results recorded in multi_cycle_results.jsonl.
