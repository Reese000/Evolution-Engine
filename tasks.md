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

### Follow-up: retrain epoch cap now bounds the L5+ boost
- **run_multi_cycle.py**: `retrain_model()` hard-set `max_epochs=8` for L5+
  champions, silently overriding a lower `--retrain-max-epochs`/config cap.
  Once the champion reached L5+, every CPU retrain ran 8 epochs regardless,
  making short retrains impossible (cycle 2 above stalled at ~8x cost).
  Extracted `_effective_max_epochs(requested, best_level)`: L5+ still targets a
  floor of 8 when unspecified, but an explicit requested cap is now an upper
  bound that wins. Added `tests/test_retrain_epoch_cap.py` (7 tests; first
  regression coverage for run_multi_cycle logic).
- **Verification**: full suite 1408 passed / 0 failed. A clean full 3-cycle run
  (`--cycles 3 --gens 20 --retrain-max-epochs 1`, CPU, fresh bootstrap
  checkpoint) completed in ~9 min (exit 0) with the cap binding even at L10.
  val_loss decreased every cycle -- compounding self-improvement:
    - C1: L6  F=387.6  val=4.3185  Gate ACCEPT (best=inf)
    - C2: L10 F=407.5  val=4.1193  Gate ACCEPT (best=4.3185, threshold=4.5344)
    - C3: L10 F=412.2  val=3.8390  Gate ACCEPT (best=4.1193, threshold=4.3253)
  The finite-threshold gate decisions (threshold = best * 1.05) confirm the
  checkpoint_acceptance_tolerance fix operating in a live run.

### Follow-up: retrain-decision telemetry reflects the effective budget
- **run_multi_cycle.py**: the existing `retrain_decisions.jsonl` record (audit
  v2) logged the decision/reason but NOT the budget that actually ran or the
  gate outcome, so it could not be used to audit the epoch-cap behaviour above.
  Enriched each record with `requested_max_epochs`, `effective_max_epochs`
  (via `_effective_max_epochs`), `retrain_lr`, `retrain_freeze_layers`,
  `val_loss`, and `gate_accepted`. Values are captured on `retrain_result` at
  the retrain call site / gating step so the record reflects what actually ran.
- Inspection during this change also confirmed the L5+ auto-freeze skip
  (retrain_model, best_level>=5) is intentional/documented behaviour, not a
  bug, and the MetaController epoch line is necessarily a pre-evolution ceiling
  suggestion (level is unknown at tune time); the effective value is logged at
  retrain time and now recorded in the decision log.
- **Verification**: full suite 1408 passed / 0 failed. A live 1-cycle retrain
  produced a decision record with the new fields populated, e.g.:
  `{cycle:1, retrained:true, requested_max_epochs:1, effective_max_epochs:1,
  retrain_lr:0.0003, retrain_freeze_layers:0, val_loss:4.3517,
  gate_accepted:true}`.
