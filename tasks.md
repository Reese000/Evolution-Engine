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
