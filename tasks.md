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

### Follow-up: test-integrity coverage + sustained-improvement validation
- Verified audit HIGH-5 (fitness_benchmark "wrong constructor call") is already
  fixed in the current code, and added `tests/test_fitness_benchmark.py`
  (config builder kwarg-validity + full tiny run) so it can't silently regress.
- Added `tests/test_val_loss_parser.py` (6 edge cases for the gate's
  val_loss parser: legacy `Val Loss:` format, min-across-formats,
  integer-no-match, malformed/binary robustness) -- this parser silently
  degrades gating when it fails, the original root cause of null val_loss.
- Added `tests/test_init_checkpoint.py` (3 tests) guarding the new bootstrap
  entry point against model-API drift (structure/arch + state_dict round-trip).
- **Full suite: 1420 passed / 0 failed** (+12 new regression tests).
- **Sustained-improvement validation**: a clean 6-cycle run
  (`--cycles 6 --gens 20 --retrain-max-epochs 1`, CPU) completed cleanly and
  showed val_loss decreasing EVERY cycle -- the compounding self-improvement
  the audit said never occurred in ~1,534 historical cycles:
    C1 L6  val=4.3459 | C2 L10 val=4.0176 | C3 L10 val=3.8121
    C4 L10 val=3.7475 | C5 L10 val=3.6909 | C6 L10 val=3.6397
  Fitness saturates at L10/412.2 by C3 while the model keeps refining
  (val_loss keeps dropping); every checkpoint accepted at threshold=best*1.05.

---

## 2026-08-29 — Autonomous session: test-integrity coverage + long validation

Focus: broaden regression coverage of previously-untested modules (audit
HIGH-4) and validate that the retrain loop's improvement is sustained over a
longer horizon.

### New regression tests (8 files, +40 tests; suite 1420 -> 1460, 0 failing)
- `tests/test_fitness_benchmark.py` -- verified audit HIGH-5 (benchmark
  "wrong constructor call") is already fixed; locks the constructor + a full
  tiny run so it can't silently regress.
- `tests/test_val_loss_parser.py` -- edge cases for the gate's val_loss parser
  (legacy `Val Loss:` format, min-across-formats, integer-no-match, malformed).
- `tests/test_init_checkpoint.py` -- bootstrap structure + state_dict round-trip.
- `tests/test_gpu_vm_sign_extend.py` -- `_sign_extend`/`_sign_extend32` (RV32I
  immediate decoding / signed compares): precomputed + general-path widths,
  masking, 32-bit boundaries.
- `tests/test_dashboard_helpers.py` -- v11_dashboard pure helpers (fmt_time,
  compute_metrics_trend, format_val_loss_trajectory, JSON readers).
- `tests/test_cycle_state.py` -- checksummed resume-state persistence:
  roundtrip, tamper-rejection -> defaults, missing/corrupt/old-version.
- `tests/test_monitoring_alerts.py` -- alert trigger logic + rate limiting
  (network-free).
- (earlier this branch: `tests/test_retrain_epoch_cap.py`.)

### Documentation
- README: rewrote the "val_loss is null across all cycles" caveat (now
  resolved + verified) and refreshed the verification table.

### Long-horizon validation
- 10-cycle CPU run (`--cycles 10 --gens 20 --retrain-max-epochs 1`) completed
  cleanly with val_loss decreasing EVERY cycle -- no plateau/divergence:
    4.4297 -> 4.0586 -> 3.8336 -> 3.7615 -> 3.7015 -> 3.647 -> 3.5959 ->
    3.5706 -> 3.5342 -> 3.4699
  Fitness saturates at L10 (~412.2) by C2 while the model keeps refining.
  (An earlier 15-cycle attempt was stopped: running the pytest suite
  concurrently starved the in-process retrain of CPU; long validations should
  run without a concurrent full-suite run.)

### Bug fix: checkpoint dir escaped the repo when log_dir was unset
- Found by running `benchmarks/verify_v13_features.py` (not in the pytest
  suite): it was 4/5, not the README's 5/5 -- "Full Evolution Cycle" crashed
  with `PermissionError: '/milestone1/checkpoints'`.
- Root cause: `EvolutionEngine._checkpoint()` defaulted `cp_dir` to
  `os.path.join(SCRIPT_DIR, "..", "milestone1", "checkpoints")`, a stale path
  from when the module lived under `milestone2/`. With the module now at the
  repo root, `".."` escaped to `<repo-parent>/milestone1` (e.g. `/milestone1`),
  crashing any engine run without an explicit `log_dir`.
- Fix: use the canonical `MILESTONE1_DIR`. `verify_v13_features.py` is now 5/5;
  `verify_improvements.py` also passes. Added
  `tests/test_checkpoint_dir_default.py`. Swept the codebase for other
  `".."`+milestone path escapes -- none remain.

### Production-path validation: autonomous_runner.py end-to-end
- `autonomous_runner.py` (the README's recommended entry point) previously
  aborted with no seed checkpoint; with `init_checkpoint.py` it now runs.
- 2-cycle CPU smoke (`--max-cycles 2`, gens=10, epochs=1): completed cleanly
  (exit 0, 0 errors) and retrained both cycles with real, decreasing val_loss
  (metrics/cycles.jsonl): C1 val=6.3827, C2 val=5.7041 (retrained=True).
- Note: the enriched `retrain_decisions.jsonl` is written by
  run_multi_cycle's main loop; autonomous_runner calls retrain_model directly
  and instead records val_loss/retrained in metrics/cycles.jsonl +
  progress_report.json. (Wiring the decision log into autonomous_runner is a
  possible follow-up for observability parity.)

### Follow-up: retrain-decision log observability parity
- Extracted `build_retrain_decision` / `write_retrain_decision` in
  run_multi_cycle.py and refactored the inline loop write to use them
  (behavior-preserving). Wired the same helper into
  `autonomous_runner.run_cycle`, and enriched autonomous_runner's
  retrain_result with requested/effective epochs, lr, freeze, and
  `gate_accepted`. The recommended production entry point now emits the same
  `retrain_decisions.jsonl` schema as run_multi_cycle.
- Added `tests/test_retrain_decision_log.py` (3 tests). Suite: 1463 passed / 1
  skipped.
- Verified end-to-end: a 1-cycle autonomous_runner produced a fully-populated
  decision record (requested/effective_max_epochs, retrain_lr,
  retrain_freeze_layers, val_loss, gate_accepted).
