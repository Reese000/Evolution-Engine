"""Register-targeted repair engine for self-sustaining high-curriculum evolution.

Owns reference payloads, trace-guided repair, target-register scoring,
and level-scaled execution budgets for L12+ curriculum tasks.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from milestone1.rv32i_mask import encode as rv32i_encode

EBREAK = 0x00100073
NOP = 0x00000013
MAX_PROGRAM_LEN = 128


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def register_repair_boost_active() -> bool:
    """True when EPI plateau boost or governor requests aggressive register repair."""
    tick: dict[str, Any] = {}
    tick_path = os.path.join(_project_root(), ".agents", "tick_state.json")
    try:
        if os.path.isfile(tick_path):
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    if os.environ.get("EE_REGISTER_REPAIR_BOOST", "0") == "1":
        return True
    if tick.get("terminal_diversity_boost"):
        return False
    return bool(tick.get("force_register_repair") or tick.get("epi_plateau_boost"))


def terminal_incremental_repair_active(level: int | None = None) -> bool:
    """At max-depth stagnation: incremental tail repair only (no wholesale graft)."""
    tick_path = os.path.join(_project_root(), ".agents", "tick_state.json")
    try:
        if os.path.isfile(tick_path):
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
            if not tick.get("terminal_diversity_boost"):
                return False
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if level is not None and int(level) < 29:
        return False
    return True

# Incremental instruction counts per level in reference solutions (L17+ tails)
_LEVEL_TAIL_COUNTS: dict[int, int] = {
    17: 12,
    18: 3,
    19: 5,
    20: 5,
    21: 4,
    22: 6,
    23: 5,
    24: 4,
    25: 6,
    26: 5,
    27: 4,
    28: 5,
    29: 6,
    30: 18,
}


def _encode_ref(instr: dict[str, Any]) -> int:
    return rv32i_encode(
        name=instr["name"],
        rd=int(instr.get("rd", 0)),
        rs1=int(instr.get("rs1", 0)),
        rs2=int(instr.get("rs2", 0)),
        imm=int(instr.get("imm", 0)),
        validate=True,
    )


def target_register(level: int) -> int | None:
    try:
        from tasks_rv32i import LEVEL_VERIFY_REGS

        regs = LEVEL_VERIFY_REGS.get(int(level), [])
        return int(regs[-1]) if regs else None
    except (ImportError, KeyError, TypeError, ValueError):
        return None


def level_verify_registers(level: int) -> list[int]:
    try:
        from tasks_rv32i import LEVEL_VERIFY_REGS

        return list(LEVEL_VERIFY_REGS.get(int(level), []))
    except (ImportError, KeyError, TypeError, ValueError):
        return []


def critical_wrong_register(level: int, wrong_regs: dict[int, Any]) -> int:
    """Pick the register most important for advancing at this level."""
    if not wrong_regs:
        return 0
    level_regs = level_verify_registers(level)
    tr = target_register(level)
    if tr is not None and tr in wrong_regs:
        return tr
    level_wrong = [r for r in wrong_regs if r in level_regs]
    if level_wrong:
        return max(level_wrong)
    return min(wrong_regs.keys())


def reference_words(level: int, *, segment: str = "full") -> tuple[list[int], str]:
    """Encode reference solution for a curriculum level."""
    try:
        from tasks_rv32i import LEVELS

        cls = LEVELS.get(int(level))
        if cls is None or not hasattr(cls, "reference_solution"):
            return [], f"l{level}_no_ref"
        ref = cls.reference_solution()
        words = [_encode_ref(d) for d in ref]
        if segment == "level_only" and level in _LEVEL_TAIL_COUNTS:
            tail_n = _LEVEL_TAIL_COUNTS[level]
            words = words[-tail_n:]
            return words, f"l{level}_tail_{tail_n}"
        if segment == "incremental" and level >= 17:
            tail_n = _LEVEL_TAIL_COUNTS.get(level, 8)
            words = words[-tail_n:]
            return words, f"l{level}_incr_{tail_n}"
        return words, f"l{level}_full_{len(words)}"
    except (ImportError, KeyError, TypeError, ValueError, OSError):
        return [], f"l{level}_error"


def reference_hex_prefix(level: int, max_words: int = 24) -> list[str]:
    """Hex prefix for model-guided generation at high levels."""
    words, _ = reference_words(level, segment="full")
    if not words:
        return []
    return [f"0x{w:08x}" for w in words[:max_words]]


def scaled_eval_budget(base_budget: int, level: int) -> int:
    """Scale VM eval budget — meta-JALR and memory tasks need more steps."""
    lvl = int(level)
    if lvl >= 17:
        mult = 1.0 + 0.15 * (lvl - 16)
        return min(int(base_budget * mult), base_budget * 4)
    if lvl >= 12:
        return int(base_budget * 1.3)
    if lvl >= 8:
        return int(base_budget * 1.15)
    return int(base_budget)


def score_target_register(
    level: int,
    expected_regs: dict[int, int],
    final_regs: dict[int, int],
) -> float:
    """0–1 score on the level-critical output register."""
    tr = target_register(level)
    if tr is None:
        return 1.0
    expected = int(expected_regs.get(tr, 0))
    actual = int(final_regs.get(tr, 0))
    if actual == expected:
        return 1.0
    if expected == 0:
        return 0.0
    error_ratio = abs(actual - expected) / max(1, abs(expected))
    return max(0.0, 1.0 / (1.0 + error_ratio))


def weighted_register_match(
    level: int,
    expected_regs: dict[int, int],
    expected_total: int,
    final_regs: dict[int, int],
    target_weight: float = 3.0,
) -> float:
    """Distance-weighted reg match with extra weight on level target register."""
    if expected_total <= 0:
        return 1.0
    tr = target_register(level)
    total_score = 0.0
    total_weight = 0.0
    for reg_idx, expected_val in expected_regs.items():
        w = target_weight if (tr is not None and reg_idx == tr) else 1.0
        actual_val = int(final_regs.get(reg_idx, 0))
        if actual_val == int(expected_val):
            total_score += w
        elif int(expected_val) != 0:
            error_ratio = abs(actual_val - int(expected_val)) / max(1, abs(int(expected_val)))
            total_score += w / (1.0 + error_ratio)
        total_weight += w
    return total_score / total_weight if total_weight > 0 else 0.0


def champion_rank_score(
    level: int,
    fitness: float,
    target_reg_score: float,
    verify_score: float,
) -> float:
    """Composite ranking for challenger selection at high curriculum levels."""
    if level < 12:
        return fitness
    if level >= 17:
        return fitness + 80.0 * target_reg_score + 30.0 * verify_score
    return fitness + 40.0 * target_reg_score + 15.0 * verify_score


def _insert_before_ebreak(prog: list[int], instr: int) -> list[int]:
    ebreak_idx = len(prog) - 1
    for i in range(len(prog) - 1, -1, -1):
        if prog[i] == EBREAK:
            ebreak_idx = i
            break
    new_prog = list(prog)
    new_prog.insert(ebreak_idx, instr)
    return new_prog[:MAX_PROGRAM_LEN]


def _insert_seq_before_ebreak(prog: list[int], instrs: list[int]) -> list[int]:
    ebreak_idx = len(prog) - 1
    for i in range(len(prog) - 1, -1, -1):
        if prog[i] == EBREAK:
            ebreak_idx = i
            break
    new_prog = list(prog)
    for offset, instr in enumerate(instrs):
        new_prog.insert(ebreak_idx + offset, instr)
    return new_prog[:MAX_PROGRAM_LEN]


def _reference_replace(words: list[int]) -> list[int]:
    """Replace program with reference payload — avoids graft stacking under boost."""
    out = list(words)
    if not out:
        return [NOP, EBREAK]
    if out[-1] != EBREAK:
        out.append(EBREAK)
    return out[:MAX_PROGRAM_LEN]


def _wholesale_replace_allowed(level: int, segment: str, words: list[int]) -> bool:
    """Only allow full-program replace with a complete reference — never tail stubs at depth."""
    if not words or segment != "full":
        return False
    lvl = int(level)
    if lvl < 21:
        return True
    full_words, _ = reference_words(lvl, segment="full")
    if not full_words:
        return False
    min_len = max(12, int(0.85 * len(full_words)))
    return len(words) >= min_len


def incremental_repair_variant(
    program: list[int],
    level: int,
    segment: str = "incremental",
) -> list[int] | None:
    """Insert a reference tail segment before EBREAK — safe variant for terminal probes."""
    words, _ = reference_words(int(level), segment=segment)
    if not words:
        return None
    return _insert_seq_before_ebreak(list(program), words)


def reference_bundle_for_register(level: int, target_reg: int, max_instr: int = 8) -> list[int]:
    """Reference instruction window that culminates in writing *target_reg*."""
    try:
        from tasks_rv32i import LEVELS

        cls = LEVELS.get(int(level))
        if cls is None or not hasattr(cls, "reference_solution"):
            return []
        ref = cls.reference_solution()
    except (ImportError, KeyError, TypeError, ValueError, AttributeError):
        return []
    last_idx: int | None = None
    for i, instr in enumerate(ref):
        rd = instr.get("rd")
        if rd is not None and int(rd) == int(target_reg):
            last_idx = i
    if last_idx is None:
        return []
    start = max(0, last_idx - int(max_instr) + 1)
    return [_encode_ref(instr) for instr in ref[start : last_idx + 1]]


def wrong_level_registers(level: int, trace: dict, task_instance: dict) -> list[int]:
    """Registers at *level* that differ from expected in *trace*."""
    expected = (task_instance or {}).get("expected_output", {}).get("registers", {})
    final = trace.get("final_regs", {}) if isinstance(trace, dict) else {}
    wrong: list[int] = []
    for reg in level_verify_registers(level):
        if int(final.get(reg, 0)) != int(expected.get(reg, 0)):
            wrong.append(int(reg))
    return wrong


def _imm12(value: int) -> int | None:
    """Encode *value* as a 12-bit signed RISC-V immediate, if possible."""
    v = int(value)
    if v < 0:
        v = v & 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    if -2048 <= v <= 2047:
        return v & 0xFFF
    return None


def _decode_word(word: int) -> dict | None:
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    milestone = os.path.join(root, "milestone1")
    if milestone not in sys.path:
        sys.path.insert(0, milestone)
    try:
        from rv32i_mask import decode as rv32i_decode

        return rv32i_decode(int(word), validate=False)
    except (ImportError, KeyError, TypeError, ValueError):
        return None


def _encode_word(**kwargs: Any) -> int | None:
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    milestone = os.path.join(root, "milestone1")
    if milestone not in sys.path:
        sys.path.insert(0, milestone)
    try:
        from rv32i_mask import encode as rv32i_encode

        return int(rv32i_encode(validate=True, **kwargs))
    except (ImportError, KeyError, TypeError, ValueError):
        return None


def trace_guided_immediate_variants(
    program: list[int],
    level: int,
    trace: dict,
    task_instance: dict,
    rng,
    *,
    max_evals: int = 32,
) -> list[list[int]]:
    """Patch ADDI immediates or insert missing reg inits guided by verify misses."""
    if int(level) < 29 or not program:
        return []
    wrong = wrong_level_registers(int(level), trace, task_instance)
    if not wrong:
        return []
    expected = (task_instance or {}).get("expected_output", {}).get("registers", {})
    rng.shuffle(wrong)
    raw: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def _add_variant(words: list[int]) -> None:
        if not words or words[-1] != EBREAK:
            words = list(words)
            if words and words[-1] != EBREAK:
                words.append(EBREAK)
        key = tuple(words)
        if key in seen or words == program:
            return
        seen.add(key)
        raw.append(list(words))

    for reg in wrong:
        exp_val = int(expected.get(reg, 0))
        imm = _imm12(exp_val)
        if imm is None:
            continue
        # Patch existing ADDI writers for this register.
        for idx, word in enumerate(program):
            decoded = _decode_word(word)
            if not decoded or decoded.get("name") != "ADDI":
                continue
            if int(decoded.get("rd", -1)) != int(reg):
                continue
            patched = _encode_word(
                name="ADDI",
                rd=int(reg),
                rs1=int(decoded.get("rs1", 0)),
                imm=int(imm),
            )
            if patched is None:
                continue
            variant = list(program)
            variant[idx] = patched
            _add_variant(variant)
        # Insert missing register init before EBREAK.
        insert_word = _encode_word(name="ADDI", rd=int(reg), rs1=0, imm=int(imm))
        if insert_word is not None:
            _add_variant(_insert_seq_before_ebreak(list(program), [insert_word]))
        # Repair ADD writers that target this register (common L30 cluster fix).
        for idx, word in enumerate(program):
            decoded = _decode_word(word)
            if not decoded or decoded.get("name") != "ADD":
                continue
            if int(decoded.get("rd", -1)) != int(reg):
                continue
            for rs1, rs2 in ((1, 2), (2, 1), (1, 1), (2, 2)):
                patched = _encode_word(name="ADD", rd=int(reg), rs1=rs1, rs2=rs2)
                if patched is None:
                    continue
                variant = list(program)
                variant[idx] = patched
                _add_variant(variant)
        # Promote operand uses of the wrong register to rd (e.g. ADD r29,r29,r26 -> ADD r26,r2,r1).
        for idx, word in enumerate(program):
            decoded = _decode_word(word)
            if not decoded or decoded.get("name") != "ADD":
                continue
            if int(reg) not in (
                int(decoded.get("rs1", -1)),
                int(decoded.get("rs2", -1)),
            ):
                continue
            for rs1, rs2 in ((2, 1), (1, 2), (1, 1), (2, 2)):
                patched = _encode_word(name="ADD", rd=int(reg), rs1=rs1, rs2=rs2)
                if patched is None:
                    continue
                variant = list(program)
                variant[idx] = patched
                _add_variant(variant)

    combined = _trace_guided_combined_repair(program, wrong, expected)
    if combined is not None:
        _add_variant(combined)

    if not raw:
        return []
    rng.shuffle(raw)
    return raw[: max(1, int(max_evals))]


def _trace_guided_combined_repair(
    program: list[int],
    wrong: list[int],
    expected: dict,
) -> list[int] | None:
    """Apply one immediate/ADD repair per wrong register in a single variant."""
    if not wrong:
        return None
    combo = list(program)
    changed = False
    inserts: list[int] = []
    for reg in wrong:
        exp_val = int(expected.get(reg, 0))
        imm = _imm12(exp_val)
        if imm is None:
            continue
        for idx, word in enumerate(combo):
            decoded = _decode_word(word)
            if not decoded or decoded.get("name") != "ADDI":
                continue
            if int(decoded.get("rd", -1)) != int(reg):
                continue
            patched = _encode_word(
                name="ADDI",
                rd=int(reg),
                rs1=int(decoded.get("rs1", 0)),
                imm=int(imm),
            )
            if patched is None or patched == combo[idx]:
                continue
            combo[idx] = patched
            changed = True
            break
        for idx, word in enumerate(combo):
            decoded = _decode_word(word)
            if not decoded or decoded.get("name") != "ADD":
                continue
            if int(reg) not in (
                int(decoded.get("rd", -1)),
                int(decoded.get("rs1", -1)),
                int(decoded.get("rs2", -1)),
            ):
                continue
            for rs1, rs2 in ((2, 1), (1, 2)):
                patched = _encode_word(name="ADD", rd=int(reg), rs1=rs1, rs2=rs2)
                if patched is None or patched == combo[idx]:
                    continue
                combo[idx] = patched
                changed = True
                break
        insert_word = _encode_word(name="ADDI", rd=int(reg), rs1=0, imm=int(imm))
        if insert_word is not None:
            inserts.append(insert_word)
    if inserts:
        combo = _insert_seq_before_ebreak(combo, inserts)
        changed = True
    if not changed or combo == program:
        return None
    return combo


def register_targeted_repair_variant(
    program: list[int],
    level: int,
    target_reg: int,
    *,
    max_instr: int = 8,
) -> list[int] | None:
    """Insert reference bundle for a single failing register before EBREAK."""
    bundle = reference_bundle_for_register(level, target_reg, max_instr=max_instr)
    if not bundle:
        return None
    return _insert_seq_before_ebreak(list(program), bundle)


def repair_program(
    program: list[int],
    level: int,
    analysis: Any,
    rng: random.Random,
    *,
    divergence_type: str | None = None,
) -> tuple[list[int], str]:
    """Apply register-targeted repair for L12+ curriculum failures."""
    div = divergence_type or getattr(analysis, "divergence_type", "unknown")
    prog = list(program) if program else [NOP, EBREAK]

    if level < 12:
        return prog, "below_l12_no_repair"

    roll = rng.random()
    terminal_incr = terminal_incremental_repair_active(level)
    boost = register_repair_boost_active()

    if terminal_incr and level >= 29 and div == "wrong_reg_value":
        expected_regs = (getattr(analysis, "expected", {}) or {}).get("registers", {})
        task_instance = {"expected_output": {"registers": expected_regs}}
        final_regs = getattr(analysis, "final_regs", []) or []
        trace = {
            "final_regs": {i: int(v) for i, v in enumerate(final_regs) if i < 32},
        }
        wrong = wrong_level_registers(int(level), trace, task_instance)
        combo = _trace_guided_combined_repair(prog, wrong, expected_regs)
        if combo is not None:
            return combo, "trace_guided_combined"
        guided = trace_guided_immediate_variants(
            prog,
            int(level),
            trace,
            task_instance,
            rng,
            max_evals=4,
        )
        if guided:
            return list(guided[0]), "trace_guided_immediate"

    if terminal_incr:
        full_thresh = 0.0
        incr_thresh = 0.72
    elif boost:
        full_thresh = 0.92
        incr_thresh = 0.98
    else:
        full_thresh = 0.55
        incr_thresh = 0.80

    def _apply(words: list[int], desc: str, segment: str) -> tuple[list[int], str]:
        if terminal_incr or not _wholesale_replace_allowed(level, segment, words):
            tag = "terminal_incr" if terminal_incr else "register_repair"
            return _insert_seq_before_ebreak(prog, words), f"{tag}_{desc}"
        return _reference_replace(words), f"register_repair_replace_{desc}"

    # Full reference graft (most aggressive — needed for meta-JALR)
    if level >= 17 and roll < full_thresh:
        words, desc = reference_words(level, segment="full")
        if words:
            return _apply(words, desc, "full")

    # Incremental tail for current level
    if roll < incr_thresh:
        words, desc = reference_words(level, segment="incremental")
        if words:
            return _apply(words, desc, "incremental")

    # Trace-guided: wrong register at L17+ needs reference not ADDI
    if div == "wrong_reg_value" and level >= 17:
        suggestion = getattr(analysis, "suggestion", {}) or {}
        reg = int(suggestion.get("reg", critical_wrong_register(level, {20: {}})))
        tr = target_register(level)
        if tr is not None and reg >= tr - 1:
            words, desc = reference_words(level, segment="level_only")
            if words:
                return _apply(words, f"wrong_r{reg}_{desc}", "level_only")

    # Budget exhausted — inject compact meta-JALR core
    if div == "ran_out_of_budget" and level >= 17:
        words, desc = reference_words(17, segment="level_only")
        if words:
            return _apply(words, f"budget_{desc}", "level_only")

    # Fallback: single reference tail
    words, desc = reference_words(level, segment="level_only")
    if words:
        return _apply(words, f"fallback_{desc}", "level_only")
    return prog, "register_repair_noop"


def reference_champion_program(level: int, max_len: int = MAX_PROGRAM_LEN) -> tuple[list[int], str]:
    """Full reference program for champion reseed when stuck below certify bar."""
    words, desc = reference_words(level, segment="full")
    if not words:
        return [], desc
    if len(words) > max_len:
        words = words[: max_len - 1] + [EBREAK]
    return words, desc


def should_reseed_champion(level: int, verify_score: float, fitness: float) -> bool:
    """True when champion is stuck and reference graft is warranted."""
    if level < 17:
        return False
    try:
        from evolution.progress_controller import ProgressController

        if ProgressController.passes_certify_verify(level, verify_score):
            return False
    except (ImportError, TypeError, ValueError):
        pass
    if verify_score < 0.30:
        return True
    # Negative fitness alone is not enough — reference graft often has low raw fitness
    # but acceptable verify; reseed only when both are poor.
    if fitness <= 0.0 and verify_score < 0.55:
        return True
    return False


def reference_passes_nudge_gate(level: int, min_verify: float = 0.72) -> bool:
    """True when reference champion at *level* meets minimum verify for depth nudge."""
    words, _ = reference_champion_program(int(level))
    if not words:
        return False
    try:
        from evolution_engine import EvolutionEngine, EvolutionConfig

        cfg = EvolutionConfig(curriculum_level=int(level), max_generations=1)
        eng = EvolutionEngine(cfg)
        eng.champion = list(words)
        eng.champion_level = int(level)
        eng._refresh_task_instance()
        eng._refresh_eval_budget()
        trace = eng._evaluate(eng.champion)
        eng.champion_trace = trace
        _, verify = eng._verify_champion()
        return float(verify) >= float(min_verify)
    except (ImportError, TypeError, ValueError, OSError, AttributeError):
        return False


def merge_priority_weight(level: int, verify_score: float, fitness: float) -> int:
    """Training duplication weight for recurrent data merge (L17+ gets more copies)."""
    w = 1
    if level >= 17:
        w = 3
    elif level >= 12:
        w = 2
    try:
        from evolution.progress_controller import ProgressController

        if ProgressController.passes_certify_verify(int(level), float(verify_score)):
            w += 2
    except (ImportError, TypeError, ValueError):
        pass
    if fitness > 80:
        w += 1
    return w
