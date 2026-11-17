"""Apply 8 targeted stale-test repairs to tests/unit/test_archive_scan_floor_probes.py.

Each replacement asserts its old_string occurs EXACTLY once (fails loudly otherwise).
Repairs, all matching production drift observed 2026-09-24 (7 failing tests):

1. F1 dual_slot: production now fires dual+triple anchor injects
   (EE_GROWTH_MG_DUAL_ANCHOR_INJECT/_TRIPLE_ default on) when generation ==
   inject generation -> [False, True, True, True]. Record growth_mode AND
   growth_anchor_rank_offset; assert the full 4-call sequence. Strictly
   stronger than the old [False, True] (first two elements preserved) and
   adds the only coverage for the anchor features (grep: none exists).
2. F3 inject_eq_refine: production passes probe_material kwarg
   (mutation_helpers.py:3671) -> mock lambda gains the kwarg.
3. F4 record_and_load: 4-word anchor below the 20-word persist floor
   (mutation_helpers.py:3193) -> disable the guard via monkeypatch.setenv,
   same proven pattern as reviewed edit #1 in the throughput file.
4. F5 skip_empty_rescan: production passes the champion program as 3rd arg
   to should_rescan_terminal_efficiency_compression (terminal_stagnation.py:3519;
   part of this task's own deploy changes) -> absorb extra arg in mock.
5. F2 growth_inject_rank (order-dependent only): scan_floor_growth_inject_rank
   reads module-level _compact_prom_raw_cycle_cache / _curriculum_l10_cycle_cache
   (mutation_helpers.py:2333-2342) populated by earlier tests -> clear caches at
   test start (file-wide convention). Oracle unchanged.
6. F6 model_inference_gate: bulk stall-collect (prom-scored best-of, the path
   the test is named for) now requires generation == inject generation and
   stall mode (gen-0 gate added for cycle-time); defaults skip it and the raw
   first completion replaces the slot -> pin generation attrs + stall mode.
   Oracle unchanged (still asserts slot word == 42, the best-prom candidate).
7. F7 compact_l30_isolates: data-dependent index 37 exceeds recent compact
   L30 champions (live rows ~27w) -> flip a valid interior word instead.
   Oracle unchanged (mutant must still score strictly below champion).
"""

import pathlib
import sys

PATH = pathlib.Path(__file__).resolve().parent / "tests" / "unit" / "test_archive_scan_floor_probes.py"
text = PATH.read_text(encoding="utf-8")
edits = []


def rep(old: str, new: str, tag: str) -> None:
    n = text.count(old)
    assert n == 1, f"[{tag}] expected exactly 1 occurrence, found {n}: {old[:80]!r}"
    edits.append(tag)


# --- F1: record (mode, offset) tuples -------------------------------------
OLD_F1A = """    calls: list[bool | None] = []

    def _fake_replace(_challengers, **_kwargs):
        calls.append(_kwargs.get("scan_floor_growth_mode"))
        return [len(calls) + 2]
"""
NEW_F1A = """    calls: list[tuple[bool | None, int | None]] = []

    def _fake_replace(_challengers, **_kwargs):
        calls.append(
            (
                _kwargs.get("scan_floor_growth_mode"),
                _kwargs.get("growth_anchor_rank_offset"),
            )
        )
        return [len(calls) + 2]
"""
rep(OLD_F1A, NEW_F1A, "F1a-record")

OLD_F1B = "    assert calls == [False, True]\n"
NEW_F1B = (
    "    # Dual stalled slots (growth False then True) plus the dual/triple\n"
    "    # anchor injects that fire when generation == inject generation\n"
    "    # (EE_GROWTH_MG_DUAL_ANCHOR_INJECT / _TRIPLE_ default on; offsets 1, 2).\n"
    "    assert calls == [(False, None), (True, None), (True, 1), (True, 2)]\n"
)
rep(OLD_F1B, NEW_F1B, "F1b-assert")

# --- F3: probe_material kwarg ---------------------------------------------
OLD_F3 = "        lambda _c, max_seeds=4: [alt],\n"
NEW_F3 = "        lambda _c, max_seeds=4, probe_material=False: [alt],\n"
rep(OLD_F3, NEW_F3, "F3-probe-material")

# --- F4: persist-floor env -------------------------------------------------
OLD_F4 = """    seed_path = str(tmp_path / "eq_seeds.json")
    record_scan_floor_eq_refine_seeds(champ, [alt, champ], cycle=42)
"""
NEW_F4 = """    seed_path = str(tmp_path / "eq_seeds.json")
    # 4-word test anchor sits below the production persist floor (20 words);
    # disable the size guard so this roundtrip test exercises record/load.
    monkeypatch.setenv("EE_EQ_REFINE_PERSIST_MIN_WORDS", "0")
    record_scan_floor_eq_refine_seeds(champ, [alt, champ], cycle=42)
"""
rep(OLD_F4, NEW_F4, "F4-persist-floor")

# --- F5: should_rescan now takes the champion program ----------------------
OLD_F5 = """        "evolution.terminal_stagnation.should_rescan_terminal_efficiency_compression",
        lambda _c, _w: True,
"""
NEW_F5 = """        "evolution.terminal_stagnation.should_rescan_terminal_efficiency_compression",
        lambda _c, _w, *_a: True,
"""
rep(OLD_F5, NEW_F5, "F5-rescan-arity")

# --- F2: clear module caches (order-dependence) ----------------------------
OLD_F2 = """def test_scan_floor_growth_inject_rank_stalled_prefers_growth_phase_longer(monkeypatch):
    from evolution.mutation_helpers import scan_floor_growth_inject_rank

    champ = [1, 2, 3, 0x00100073]
"""
NEW_F2 = """def test_scan_floor_growth_inject_rank_stalled_prefers_growth_phase_longer(monkeypatch):
    from evolution.mutation_helpers import (
        clear_scan_floor_inject_caches,
        scan_floor_growth_inject_rank,
    )

    # Module-level prom/l10 cycle caches retain values from earlier tests in
    # the same session; clear them so this test's fitness mock is authoritative.
    clear_scan_floor_inject_caches()
    champ = [1, 2, 3, 0x00100073]
"""
rep(OLD_F2, NEW_F2, "F2-cache-clear")

# --- F6: pin generation + stall mode for the collect path ------------------
OLD_F6 = """    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = champ
    eng.rng.seed(0)
    eng._last_challengers = [list(champ) for _ in range(6)]
"""
NEW_F6 = """    eng = EvolutionEngine(config=EvolutionConfig(max_level=30))
    eng.champion = champ
    eng.rng.seed(0)
    eng._last_challengers = [list(champ) for _ in range(6)]
    # Bulk stall-collect (prom-scored best-of, the path this test is named for)
    # only runs when generation == inject generation AND stall mode is active;
    # the gen-0 default skips it and the raw first completion replaces the
    # slot instead. Pin both so the best-prom_raw collect path is exercised.
    monkeypatch.setattr(
        "evolution.terminal_stagnation._scan_floor_probe_stall_mode",
        lambda _w: True,
    )
    eng.config.max_generations = 30
    eng._scan_floor_inject_generation = 29
    eng.current_generation = 29
"""
rep(OLD_F6, NEW_F6, "F6-collect-pin")

# --- F7: interior index valid for compact champions ------------------------
OLD_F7 = """    mutant = list(champ)
    mutant[37] ^= 1
"""
NEW_F7 = """    mutant = list(champ)
    # Live L30 champions vary in length (recent jsonl rows are compact, e.g.
    # 27w); flip a valid interior word instead of the fixed index 37.
    mutant[len(champ) // 2] ^= 1
"""
rep(OLD_F7, NEW_F7, "F7-interior-index")

# --- apply ------------------------------------------------------------------
for old, new in (
    (OLD_F1A, NEW_F1A),
    (OLD_F1B, NEW_F1B),
    (OLD_F3, NEW_F3),
    (OLD_F4, NEW_F4),
    (OLD_F5, NEW_F5),
    (OLD_F2, NEW_F2),
    (OLD_F6, NEW_F6),
    (OLD_F7, NEW_F7),
):
    text = text.replace(old, new, 1)

PATH.write_text(text, encoding="utf-8")
print(f"APPLIED {len(edits)}/{len(edits)}: " + ", ".join(edits))
sys.exit(0)
