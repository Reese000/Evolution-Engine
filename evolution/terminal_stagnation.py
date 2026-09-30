"""Helpers for max-depth curriculum stagnation (L30 terminal plateau)."""

from __future__ import annotations

import functools
import json
import os
import random as _random
import struct
import zlib

TERMINAL_VERIFY_MARGIN = float(os.environ.get("EE_TERMINAL_VERIFY_MARGIN", "0.005"))
TERMINAL_VERIFY_MIN_FITNESS_FRAC = float(os.environ.get("EE_TERMINAL_VERIFY_MIN_FIT_FRAC", "0.4"))
TERMINAL_VERIFY_SELECTION_WEIGHT = float(os.environ.get("EE_TERMINAL_VERIFY_SEL_WEIGHT", "120"))
_BRANCH_OPS = frozenset({"BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU", "JAL", "JALR"})
_KNOWN_STRAIGHT_SLICES: dict[int, list[tuple[int, int]]] = {
    30: [(24, 25), (70, 72), (71, 72), (73, 75), (74, 76), (58, 59)],
}

_NOP = 0x00000013
_EBREAK = 0x00100073
_VERIFY_SATURATED = 0.99
L30_COMPACT_MAX_WORDS = 72


def peephole_compress(program: list[int]) -> list[int]:
    """Fold ADDI chains and drop ADDI rd,rd,0. No VM; length only shrinks."""
    if not program:
        return []
    try:
        from rv32i_mask import decode as rv32i_decode
        from rv32i_mask import encode as rv32i_encode
    except ImportError:
        return list(program)

    def _once(prog: list[int]) -> list[int]:
        has_eb = bool(prog) and prog[-1] == _EBREAK
        body = prog[:-1] if has_eb else list(prog)
        out: list[int] = []
        i = 0
        while i < len(body):
            word = body[i]
            try:
                a = rv32i_decode(word, validate=False)
            except (KeyError, TypeError, ValueError):
                a = None
            if not a:
                out.append(word)
                i += 1
                continue
            if a.get("name") == "ADDI" and int(a.get("imm") or 0) == 0 and a.get("rd") == a.get("rs1"):
                i += 1
                continue
            if i + 1 < len(body) and a.get("name") == "ADDI":
                try:
                    b = rv32i_decode(body[i + 1], validate=False)
                except (KeyError, TypeError, ValueError):
                    b = None
                if b and b.get("name") == "ADDI" and a.get("rd") == b.get("rd") and b.get("rs1") == a.get("rd"):
                    imm = int(a.get("imm") or 0) + int(b.get("imm") or 0)
                    if -2048 <= imm <= 2047:
                        try:
                            folded = rv32i_encode(
                                name="ADDI",
                                rd=a["rd"],
                                rs1=a["rs1"],
                                imm=imm,
                                validate=True,
                            )
                            out.append(folded)
                            i += 2
                            continue
                        except (ValueError, KeyError, TypeError):
                            pass
            out.append(word)
            i += 1
        if has_eb:
            out.append(_EBREAK)
        return out

    cur = list(program)
    for _ in range(8):
        nxt = _once(cur)
        if nxt == cur:
            return nxt
        cur = nxt
    return cur


def overwrite_dead_compress(program: list[int]) -> list[int]:
    """Drop writes that are overwritten before any read. Straight-line only."""
    if not program:
        return []
    try:
        from rv32i_mask import decode as rv32i_decode
    except ImportError:
        return list(program)
    has_eb = bool(program) and program[-1] == _EBREAK
    body = program[:-1] if has_eb else list(program)
    decoded = []
    for word in body:
        try:
            decoded.append(rv32i_decode(word, validate=False))
        except (KeyError, TypeError, ValueError):
            decoded.append({"name": "?", "rd": 0, "rs1": 0, "rs2": 0})
    names = [d.get("name") or "" for d in decoded]
    if any(n in _BRANCH_OPS for n in names):
        return list(program)
    stores = {"SW", "SH", "SB"}
    last_writer: dict[int, int] = {}
    dead: set[int] = set()
    for i, d in enumerate(decoded):
        name = names[i]
        word = body[i]
        opcode = word & 0x7F
        rs1 = int(d.get("rs1") or 0)
        rs2 = int(d.get("rs2") or 0)
        if opcode not in (0x37, 0x17, 0x6F) and rs1:
            last_writer.pop(rs1, None)
        if opcode in (0x33, 0x23, 0x63) and rs2:
            last_writer.pop(rs2, None)
        if name in stores:
            continue
        rd = int(d.get("rd") or 0)
        if not rd:
            continue
        prev = last_writer.get(rd)
        if prev is not None:
            dead.add(prev)
        last_writer[rd] = i
    if not dead:
        return list(program)
    out = [w for i, w in enumerate(body) if i not in dead]
    if has_eb:
        out.append(_EBREAK)
    if not out:
        return list(program)
    if len(out) < len(program):
        return overwrite_dead_compress(out)
    return out


def li_mem_fold_compress(program: list[int]) -> list[int]:
    """Fold ADDI rd,x0,imm into a later load/store using rs1=rd."""
    if not program:
        return []
    try:
        from rv32i_mask import decode as rv32i_decode
        from rv32i_mask import encode as rv32i_encode
    except ImportError:
        return list(program)
    has_eb = bool(program) and program[-1] == _EBREAK
    body = program[:-1] if has_eb else list(program)
    loads = {"LB", "LH", "LW", "LBU", "LHU"}
    stores = {"SB", "SH", "SW"}
    decoded = []
    for word in body:
        try:
            decoded.append(rv32i_decode(word, validate=False))
        except (KeyError, TypeError, ValueError):
            decoded.append(None)
    names = [(d or {}).get("name") or "" for d in decoded]
    if any(n in _BRANCH_OPS for n in names):
        return list(program)

    def _uses_reg(d, opcode, reg: int) -> bool:
        if not d or not reg:
            return False
        rs1 = int(d.get("rs1") or 0)
        rs2 = int(d.get("rs2") or 0)
        if opcode not in (0x37, 0x17, 0x6F) and rs1 == reg:
            return True
        if opcode in (0x33, 0x23, 0x63) and rs2 == reg:
            return True
        return False

    out = list(body)
    i = 0
    while i < len(out):
        try:
            a = rv32i_decode(out[i], validate=False)
        except (KeyError, TypeError, ValueError):
            a = None
        if not a:
            i += 1
            continue
        if a.get("name") != "ADDI" or int(a.get("rs1") or 0) != 0:
            i += 1
            continue
        rd = int(a.get("rd") or 0)
        if not rd:
            i += 1
            continue
        imm0 = int(a.get("imm") or 0)
        folded = False
        for j in range(i + 1, len(out)):
            try:
                b = rv32i_decode(out[j], validate=False)
            except (KeyError, TypeError, ValueError):
                b = None
            if not b:
                continue
            opcode = out[j] & 0x7F
            name = b.get("name") or ""
            if _uses_reg(b, opcode, rd):
                if name in loads | stores and int(b.get("rs1") or 0) == rd:
                    if name in stores and int(b.get("rs2") or 0) == rd:
                        break
                    new_imm = imm0 + int(b.get("imm") or 0)
                    if not (-2048 <= new_imm <= 2047):
                        break
                    try:
                        if name in stores:
                            out[j] = rv32i_encode(
                                name=name, rs1=0, rs2=int(b.get("rs2") or 0), imm=new_imm, validate=True
                            )
                        else:
                            out[j] = rv32i_encode(
                                name=name, rd=int(b.get("rd") or 0), rs1=0, imm=new_imm, validate=True
                            )
                    except (ValueError, KeyError, TypeError):
                        break
                    later = False
                    for k in range(j + 1, len(out)):
                        try:
                            c = rv32i_decode(out[k], validate=False)
                        except (KeyError, TypeError, ValueError):
                            continue
                        if _uses_reg(c, out[k] & 0x7F, rd):
                            later = True
                            break
                    if not later:
                        out.pop(i)
                        folded = True
                    else:
                        i += 1
                        folded = True
                    break
                break
        if not folded:
            i += 1
    if has_eb:
        out.append(_EBREAK)
    return out


def zero_identity_compress(program: list[int]) -> list[int]:
    """Drop OR/XOR/ADD/SUB that combine rd with a never-written (zero) source."""
    if not program:
        return []
    try:
        from rv32i_mask import decode as rv32i_decode
    except ImportError:
        return list(program)
    has_eb = bool(program) and program[-1] == _EBREAK
    body = program[:-1] if has_eb else list(program)
    decoded = []
    for word in body:
        try:
            decoded.append(rv32i_decode(word, validate=False))
        except (KeyError, TypeError, ValueError):
            decoded.append(None)
    names = [(d or {}).get("name") or "" for d in decoded]
    if any(n in _BRANCH_OPS for n in names):
        return list(program)
    identity = {"OR", "XOR", "ADD", "SUB"}
    written: set[int] = set()
    keep: list[int] = []
    for i, word in enumerate(body):
        d = decoded[i]
        name = names[i]
        opcode = word & 0x7F
        if d and name in identity and opcode == 0x33:
            rd = int(d.get("rd") or 0)
            rs1 = int(d.get("rs1") or 0)
            rs2 = int(d.get("rs2") or 0)
            rs1_z = rs1 == 0 or rs1 not in written
            rs2_z = rs2 == 0 or rs2 not in written
            if name == "SUB":
                if rd and rd == rs1 and rs2_z:
                    continue
            elif rd and ((rd == rs1 and rs2_z) or (rd == rs2 and rs1_z)):
                continue
        if d:
            wr = int(d.get("rd") or 0)
            if wr and name not in {"SW", "SH", "SB"}:
                written.add(wr)
        keep.append(word)
    if has_eb:
        keep.append(_EBREAK)
    return keep if keep else list(program)


def li_ori_fold_compress(program: list[int]) -> list[int]:
    """ADDI t,x0,imm then OR rd,rd,t with t killed → ORI, and drop the ADDI."""
    if not program:
        return []
    try:
        from rv32i_mask import decode as rv32i_decode
        from rv32i_mask import encode as rv32i_encode
    except ImportError:
        return list(program)
    has_eb = bool(program) and program[-1] == _EBREAK
    body = program[:-1] if has_eb else list(program)
    decoded = []
    for word in body:
        try:
            decoded.append(rv32i_decode(word, validate=False))
        except (KeyError, TypeError, ValueError):
            decoded.append(None)
    names = [(d or {}).get("name") or "" for d in decoded]
    if any(n in _BRANCH_OPS for n in names):
        return list(program)
    logic = {"OR": "ORI", "XOR": "XORI", "AND": "ANDI"}
    out = list(body)
    i = 0
    while i < len(out):
        try:
            a = rv32i_decode(out[i], validate=False)
        except (KeyError, TypeError, ValueError):
            a = None
        if not a:
            i += 1
            continue
        if a.get("name") != "ADDI" or int(a.get("rs1") or 0) != 0:
            i += 1
            continue
        t = int(a.get("rd") or 0)
        imm = int(a.get("imm") or 0)
        if not t or not (-2048 <= imm <= 2047):
            i += 1
            continue
        folded = False
        for j in range(i + 1, len(out)):
            try:
                b = rv32i_decode(out[j], validate=False)
            except (KeyError, TypeError, ValueError):
                b = None
            if not b:
                continue
            opcode = out[j] & 0x7F
            rs1 = int(b.get("rs1") or 0)
            rs2 = int(b.get("rs2") or 0)
            rd = int(b.get("rd") or 0)
            name = b.get("name") or ""
            uses_t = (rs1 == t and opcode not in (0x37, 0x17, 0x6F)) or (rs2 == t and opcode in (0x33, 0x23, 0x63))
            if not uses_t:
                continue
            if name in logic and opcode == 0x33 and rd and (rs1 == t or rs2 == t):
                other = rs2 if rs1 == t else rs1
                later = False
                for k in range(j + 1, len(out)):
                    try:
                        c = rv32i_decode(out[k], validate=False)
                    except (KeyError, TypeError, ValueError):
                        c = None
                    if not c:
                        continue
                    cop = out[k] & 0x7F
                    if int(c.get("rs1") or 0) == t and cop not in (0x37, 0x17, 0x6F):
                        later = True
                        break
                    if int(c.get("rs2") or 0) == t and cop in (0x33, 0x23, 0x63):
                        later = True
                        break
                if later:
                    break
                try:
                    out[j] = rv32i_encode(name=logic[name], rd=rd, rs1=other, imm=imm, validate=True)
                    out.pop(i)
                    folded = True
                except (ValueError, KeyError, TypeError):
                    pass
            break
        if not folded:
            i += 1
    if has_eb:
        out.append(_EBREAK)
    return out


def semantic_compress(program: list[int]) -> list[int]:
    """Iterate peephole + identity + LI folds + overwrite-DCE. Length never grows."""
    cur = list(program)
    for _ in range(8):
        nxt = overwrite_dead_compress(
            li_mem_fold_compress(li_ori_fold_compress(zero_identity_compress(peephole_compress(cur))))
        )
        if nxt == cur:
            return nxt
        cur = nxt
    return cur


def terminal_elite_heap_key(score, program) -> tuple[float, int]:
    """Prefer higher rank, then shorter programs, when filling the elite set."""
    return (-float(score), len(program or ()))


def inherit_verify_if_semantic_clone(
    champ_words: int,
    cand_words: int,
    champ_trace: dict | None,
    cand_trace: dict | None,
    cand_verify: float,
    champ_verify: float,
) -> float:
    """Same observable IO as a shorter clone keeps the champion's verify score."""
    if int(cand_words) >= int(champ_words):
        return float(cand_verify)
    if not traces_observably_equal(cand_trace, champ_trace):
        return float(cand_verify)
    return max(float(cand_verify), float(champ_verify))


def traces_observably_equal(a: dict | None, b: dict | None) -> bool:
    """Same final regs/memory; used to accept a shorter semantic clone."""
    if not a or not b:
        return False
    if a.get("status") in {"invalid", "error"} or b.get("status") in {"invalid", "error"}:
        return False

    def _norm_regs(regs) -> dict[int, int]:
        out = {}
        for k, v in (regs or {}).items():
            try:
                out[int(k)] = int(v) & 0xFFFFFFFF
            except (TypeError, ValueError):
                continue
        return out

    ra = _norm_regs(a.get("final_regs"))
    rb = _norm_regs(b.get("final_regs"))
    keys = set(ra) | set(rb)
    if any(ra.get(k, 0) != rb.get(k, 0) for k in keys):
        return False
    ma, mb = a.get("final_memory") or {}, b.get("final_memory") or {}
    return ma == mb


def saturated_shorter_swap_ok(champ_words: int, cand_words: int, champ_verify: float, cand_verify: float) -> bool:
    """Same-or-better verify at saturation: shorter challenger may replace."""
    return (
        float(champ_verify) >= 0.99 - 1e-6
        and float(cand_verify) + 1e-6 >= float(champ_verify)
        and int(cand_words) < int(champ_words)
    )


def saturated_shorter_capability_swap_ok(
    engine,
    champion_program: list[int],
    champion_trace: dict,
    candidate_program: list[int],
    candidate_trace: dict,
    *,
    champion_verify: float,
    candidate_verify: float,
) -> bool:
    """Shorter saturated swap only when promotion_raw capability does not regress."""
    if not saturated_shorter_swap_ok(
        len(champion_program),
        len(candidate_program),
        champion_verify,
        candidate_verify,
    ):
        return False
    return verify_promotion_beats_champion(
        engine,
        champion_program,
        champion_trace,
        candidate_program,
        candidate_trace,
        0.0,
        0.0,
        champion_verify=champion_verify,
        candidate_verify=candidate_verify,
    )


def terminal_efficiency_variants(program, rng, *, max_evals=16, max_drop_span: int = 3):
    """Probe shorter variants once verify is saturated."""
    if not program:
        return []
    raw = []
    seen = {tuple(program)}

    def _add(words):
        if not words:
            return
        if words[-1] != _EBREAK:
            words = list(words)
            words.append(_EBREAK)
        key = tuple(words)
        if key in seen:
            return
        seen.add(key)
        raw.append(list(words))

    collapsed = []
    for word in program:
        if word == _NOP and collapsed and collapsed[-1] == _NOP:
            continue
        collapsed.append(word)
    _add(collapsed)
    _add([w for w in program if w != _NOP])
    try:
        from mutation_proposer import trim_dead_code

        _add(trim_dead_code(list(program)))
    except ImportError:
        pass
    _add(peephole_compress(list(program)))
    _add(overwrite_dead_compress(list(program)))
    _add(semantic_compress(list(program)))
    # No-NOP champions never shrink from the paths above. Drop interior
    # instructions so length-efficiency promotion can see shorter programs
    # in the official eval batch (no extra VM here).
    body = list(program[:-1]) if program and program[-1] == _EBREAK else list(program)
    drop_at = list(range(len(body)))
    rng.shuffle(drop_at)
    budget = max(1, int(max_evals))
    span = max(1, min(3, int(max_drop_span)))
    for width in range(1, span + 1):
        for i in drop_at:
            if len(raw) >= budget:
                break
            if i + width > len(body):
                continue
            _add(body[:i] + body[i + width :])
    rng.shuffle(raw)
    return raw[:budget]


def maybe_drop_interior_instruction(program, rng, p: float = 0.25, n_drop: int = 1):
    """With probability p, drop n_drop interior instructions (keep EBREAK).

    Used on mutated challengers so length-efficiency is not limited to the
    probe slice. No extra VM; the generation batch scores the result.
    """
    if not program or rng.random() >= float(p):
        return program
    body = list(program[:-1]) if program[-1] == _EBREAK else list(program)
    drop = max(1, int(n_drop))
    if len(body) <= drop:
        return program
    start = rng.randrange(0, len(body) - drop + 1)
    out = body[:start] + body[start + drop :]
    if not out or out[-1] != _EBREAK:
        out.append(_EBREAK)
    return out


def terminal_length_drop_params(n_words: int, cap: int | None) -> tuple[float, int]:
    """Length-drop rate for mutated challengers.

    Above the reference cap: stronger drops. At or below cap-16: keep a
    light drop so TASK_AWARE mutations are not mostly deleted.
    """
    n = int(n_words)
    if not cap:
        return 0.25, 1
    c = int(cap)
    if n > c:
        return 0.40, 2
    if n > max(12, c - 16):
        return 0.25, 1
    # Very short saturated champions: random interior deletes often
    # destroy verify=1.0. TASK_AWARE / probes still search shorter programs.
    if n <= 20:
        return 0.0, 1
    return 0.12, 1


def terminal_efficiency_drop_span(n_words: int, cap: int | None) -> int:
    """Max consecutive instructions to drop in efficiency probes."""
    n = int(n_words)
    if not cap:
        return 3
    c = int(cap)
    if n > c:
        return 3
    if n > max(12, c - 16):
        return 2
    return 1


def _sort_probe_candidates(candidates, rng, *, score_fn=None, fitness_fn=None, min_verify=None):
    saturated = min_verify is not None and float(min_verify) >= _VERIFY_SATURATED
    if score_fn is not None and fitness_fn is not None and saturated:
        candidates.sort(key=lambda prog: (float(score_fn(prog)), float(fitness_fn(prog))), reverse=True)
    elif score_fn is not None:
        candidates.sort(key=lambda prog: float(score_fn(prog)), reverse=True)
    else:
        rng.shuffle(candidates)


def _filter_probe_candidates(
    raw: list[list[int]],
    *,
    score_fn=None,
    min_verify: float | None = None,
    fitness_fn=None,
    min_fitness: float | None = None,
) -> list[list[int]]:
    floor = float(min_verify) if min_verify is not None else None
    fit_floor = float(min_fitness) if min_fitness is not None else None
    candidates: list[list[int]] = []
    for variant in raw:
        if score_fn is not None and floor is not None:
            try:
                if float(score_fn(variant)) < floor - 1e-6:
                    continue
            except (RuntimeError, TypeError, ValueError, AttributeError):
                continue
        if fitness_fn is not None and fit_floor is not None:
            try:
                if float(fitness_fn(variant)) < fit_floor - 1e-6:
                    continue
            except (RuntimeError, TypeError, ValueError, AttributeError):
                continue
        candidates.append(variant)
    return candidates


def terminal_branch_flip_variants(
    program: list[int],
    rng,
    *,
    max_evals: int = 24,
    tail_window: int = 32,
) -> list[list[int]]:
    """Flip branch/ALU opcodes in the champion tail to explore alternate control flow."""
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    milestone = os.path.join(root, "milestone1")
    if milestone not in sys.path:
        sys.path.insert(0, milestone)
    if root not in sys.path:
        sys.path.insert(0, root)
    from rv32i_mask import decode as rv32i_decode
    from rv32i_mask import encode as rv32i_encode

    from mutation_proposer import ALU_SWAP_PAIRS, BRANCH_SWAP_PAIRS, EBREAK

    if not program:
        return []
    tail_start = max(0, len(program) - int(tail_window))
    candidates: list[tuple[int, str]] = []
    for i in range(tail_start, len(program) - 1):
        try:
            decoded = rv32i_decode(program[i], validate=False)
            name = decoded.get("name", "")
        except (KeyError, TypeError, ValueError):
            continue
        if name in BRANCH_SWAP_PAIRS:
            candidates.append((i, "branch"))
        elif name in ALU_SWAP_PAIRS:
            candidates.append((i, "alu"))
    rng.shuffle(candidates)
    raw: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for i, swap_type in candidates[: max(1, int(max_evals))]:
        try:
            decoded = rv32i_decode(program[i], validate=False)
            name = decoded.get("name", "")
        except (KeyError, TypeError, ValueError):
            continue
        if swap_type == "branch" and name in BRANCH_SWAP_PAIRS:
            new_name = BRANCH_SWAP_PAIRS[name]
            new_word = rv32i_encode(
                name=new_name,
                rs1=decoded["rs1"],
                rs2=decoded["rs2"],
                imm=decoded["imm"],
                validate=True,
            )
        elif swap_type == "alu" and name in ALU_SWAP_PAIRS:
            new_name = ALU_SWAP_PAIRS[name]
            new_word = rv32i_encode(
                name=new_name,
                rd=decoded["rd"],
                rs1=decoded["rs1"],
                rs2=decoded["rs2"],
                validate=True,
            )
        else:
            continue
        variant = list(program)
        variant[i] = new_word
        if variant[-1] != EBREAK:
            variant.append(EBREAK)
        key = tuple(variant)
        if key in seen or variant == program:
            continue
        seen.add(key)
        raw.append(variant)
    return raw


def straight_line_reference_variants(
    program: list[int],
    level: int,
    rng,
    *,
    max_evals: int = 40,
) -> list[list[int]]:
    """Branch-free reference slices safe to graft before EBREAK."""
    from evolution.register_repair import _encode_ref, _insert_seq_before_ebreak

    lvl = int(level)
    if lvl < 29 or not program:
        return []
    try:
        from tasks_rv32i import LEVELS

        cls = LEVELS.get(lvl)
        if cls is None or not hasattr(cls, "reference_solution"):
            return []
        ref = cls.reference_solution()
    except (ImportError, KeyError, TypeError, ValueError, AttributeError):
        return []
    windows: list[tuple[int, int]] = list(_KNOWN_STRAIGHT_SLICES.get(lvl, []))
    scanned: list[tuple[int, int]] = []
    for start in range(len(ref)):
        for end in range(start + 1, min(start + 9, len(ref) + 1)):
            chunk = ref[start:end]
            if any(instr.get("name") in _BRANCH_OPS for instr in chunk):
                continue
            scanned.append((start, end))
    rng.shuffle(scanned)
    seen_windows = set(windows)
    for window in scanned:
        if window in seen_windows:
            continue
        seen_windows.add(window)
        windows.append(window)
    raw: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for start, end in windows[: max(1, int(max_evals))]:
        words = [_encode_ref(instr) for instr in ref[start:end]]
        if not words:
            continue
        variant = _insert_seq_before_ebreak(list(program), words)
        key = tuple(variant)
        if key in seen or variant == program:
            continue
        seen.add(key)
        raw.append(variant)
    return raw


def _project_agents_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".agents",
    )


def _vm_score_disk_path() -> str:
    return os.path.join(_project_agents_dir(), "vm_score_disk_cache.json")


def _vm_score_program_key(program: tuple[int, ...]) -> str:
    return ",".join(f"{int(w):08X}" for w in program)


_VM_SCORE_DISK_MEM: dict[str, dict[str, float]] | None = None
_VM_SCORE_DISK_LOADED = False


def _vm_score_disk_reset() -> None:
    """Clear in-process disk cache (tests only)."""
    global _VM_SCORE_DISK_MEM, _VM_SCORE_DISK_LOADED
    _VM_SCORE_DISK_MEM = None
    _VM_SCORE_DISK_LOADED = False


def _vm_score_disk_ensure_loaded() -> dict[str, dict[str, float]]:
    global _VM_SCORE_DISK_MEM, _VM_SCORE_DISK_LOADED
    if _VM_SCORE_DISK_LOADED:
        return _VM_SCORE_DISK_MEM if _VM_SCORE_DISK_MEM is not None else {}
    _VM_SCORE_DISK_LOADED = True
    _VM_SCORE_DISK_MEM = {}
    path = _vm_score_disk_path()
    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            _VM_SCORE_DISK_MEM = {
                str(sec): {str(k): float(v) for k, v in bucket.items()}
                for sec, bucket in existing.items()
                if isinstance(bucket, dict)
            }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return _VM_SCORE_DISK_MEM


def _vm_score_disk_flush() -> None:
    """Persist in-memory VM score cache to disk (one write per probe-pool build)."""
    if not _VM_SCORE_DISK_LOADED or _VM_SCORE_DISK_MEM is None:
        return
    path = _vm_score_disk_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_VM_SCORE_DISK_MEM, f)


def _vm_score_disk_get(section: str, key: str) -> float | None:
    payload = _vm_score_disk_ensure_loaded()
    bucket = payload.get(section)
    if not isinstance(bucket, dict):
        return None
    value = bucket.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _vm_score_disk_put(section: str, key: str, value: float) -> None:
    payload = _vm_score_disk_ensure_loaded()
    bucket = dict(payload.get(section, {}))
    bucket[key] = float(value)
    if len(bucket) > 2048:
        for stale in list(bucket.keys())[:-1536]:
            if stale != key:
                bucket.pop(stale, None)
    payload[section] = bucket


_progress_state_cache: dict | None = None
_progress_state_mtime: float = -1.0


def _invalidate_progress_state_cache() -> None:
    global _progress_state_cache, _progress_state_mtime
    _progress_state_cache = None
    _progress_state_mtime = -1.0


def _read_progress_state() -> dict:
    global _progress_state_cache, _progress_state_mtime
    state_path = os.path.join(_project_agents_dir(), "progress_controller_state.json")
    try:
        if os.path.isfile(state_path):
            mtime = os.path.getmtime(state_path)
            if _progress_state_cache is not None and mtime == _progress_state_mtime:
                return _progress_state_cache
            with open(state_path, encoding="utf-8") as f:
                _progress_state_cache = json.load(f)
            _progress_state_mtime = mtime
            return _progress_state_cache
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    _progress_state_cache = {}
    _progress_state_mtime = -1.0
    return _progress_state_cache


def effective_verify_margin() -> float:
    """Shrink verify margin after long terminal-depth stagnation."""
    base = TERMINAL_VERIFY_MARGIN
    cycles = int(_read_progress_state().get("cycles_at_level", 0))
    if cycles >= 45:
        return min(base, 0.001)
    if cycles >= 35:
        return min(base, 0.002)
    if cycles >= 28:
        return min(base, 0.003)
    return base


def _read_tick() -> dict:
    tick_path = os.path.join(_project_agents_dir(), "tick_state.json")
    try:
        if os.path.isfile(tick_path):
            with open(tick_path, encoding="utf-8") as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return {}


def terminal_cycles_at_level() -> int:
    return int(_read_progress_state().get("cycles_at_level", 0))


def terminal_depth() -> int:
    """Canonical terminal curriculum depth (not EE_MAX_LEVEL clamp recovery caps)."""
    return int(
        os.environ.get(
            "EE_TERMINAL_DEPTH",
            os.environ.get("EE_CURRICULUM_MAX_LEVEL", "30"),
        )
    )


def terminal_certified_plateau(level: int, verify: float, fitness: float) -> bool:
    """L30+ certified champion with saturated verify — fitness exploration phase."""
    depth = terminal_depth()
    min_fitness = float(os.environ.get("EE_TERMINAL_PLATEAU_MIN_FITNESS", "250"))
    min_verify = float(os.environ.get("EE_TERMINAL_PLATEAU_MIN_VERIFY", "0.99"))
    return int(level) >= depth and float(verify) >= min_verify - 1e-6 and float(fitness) >= min_fitness


def terminal_retrain_interval(cycles_at_level: int | None = None) -> int:
    """Shorter retrain cadence as terminal stagnation deepens (tier-scaled)."""
    default = int(os.environ.get("EE_TERMINAL_RETRAIN_INTERVAL", "40"))
    tier = terminal_diversity_boost_tier(cycles_at_level)
    if tier >= 7:
        # Deep terminal plateau: fewer retrains → more fast-path cycles (~210s vs ~450s).
        return min(default, int(os.environ.get("EE_TERMINAL_RETRAIN_INTERVAL_TIER7", "32")))
    if tier >= 5:
        return min(default, 28)
    if tier >= 3:
        return min(default, 34)
    return default


def terminal_probe_escalation_ratio(cycles_at_level: int | None = None) -> float:
    """Widen verify probe slice when max-depth fitness is frozen."""
    base = float(os.environ.get("EE_TERMINAL_VERIFY_PROBE_RATIO", "0.3"))
    cycles = int(cycles_at_level if cycles_at_level is not None else terminal_cycles_at_level())
    if cycles >= 400:
        return max(base, 0.72)
    if cycles >= 300:
        return max(base, 0.65)
    if cycles >= 200:
        return max(base, 0.5)
    if cycles >= 100:
        return max(base, 0.42)
    if cycles >= 50:
        return max(base, 0.36)
    return base


def terminal_efficiency_promote_cooldown_cycles() -> int:
    """Min cycles between successful terminal-efficiency promotions (throughput guard)."""
    return max(0, int(os.environ.get("EE_TERMINAL_EFFICIENCY_PROMOTE_COOLDOWN", "5")))


def _patch_progress_state(updates: dict) -> None:
    """Merge keys into progress_controller_state.json (atomic write)."""
    state_path = os.path.join(_project_agents_dir(), "progress_controller_state.json")
    state: dict = {}
    try:
        if os.path.isfile(state_path):
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        state = {}
    state.update(updates)
    tmp_path = state_path + ".tmp"
    try:
        payload = json.dumps(state, indent=2)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, state_path)
        _invalidate_progress_state_cache()
    except OSError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _infer_terminal_efficiency_promote_cycle() -> int:
    """Most recent L30 cycle where champion word count decreased."""
    results = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "multi_cycle_results.jsonl",
    )
    if not os.path.isfile(results):
        return 0
    last_words: int | None = None
    last_shrink = 0
    try:
        with open(results, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row.get("final_level", 0)) != 30:
                    continue
                words = int(row.get("champion_words", 0))
                cycle = int(row.get("cycle", 0))
                if last_words is not None and words < last_words:
                    last_shrink = cycle
                last_words = words
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0
    return int(last_shrink)


def terminal_efficiency_last_promote_cycle() -> int:
    """Persisted cycle of last terminal-efficiency promotion (survives daemon restart)."""
    stored = int(_read_progress_state().get("terminal_efficiency_promote_cycle", 0))
    if stored > 0:
        return stored
    return _infer_terminal_efficiency_promote_cycle()


def record_terminal_efficiency_promote(cycle: int) -> None:
    _patch_progress_state({"terminal_efficiency_promote_cycle": int(cycle)})


def _terminal_efficiency_words_trackable(champ_words: int) -> bool:
    """Reject ephemeral programs that diverge from the certified compact champion length."""
    words = int(champ_words)
    if not (12 <= words <= L30_COMPACT_MAX_WORDS):
        return False
    state = _read_progress_state()
    certified = int(state.get("terminal_efficiency_certified_words", 0))
    if certified > 0 and abs(words - certified) > 1:
        return False
    return True


def record_terminal_efficiency_stable(cycle: int, champ_words: int) -> None:
    """Track when champion word count last changed (for promotion quiescence)."""
    if not _terminal_efficiency_words_trackable(champ_words):
        return
    state = _read_progress_state()
    prev_stable = int(state.get("terminal_efficiency_stable_words", -1))
    if prev_stable == int(champ_words):
        return
    patch: dict[str, int | float] = {
        "terminal_efficiency_stable_words": int(champ_words),
        "terminal_efficiency_stable_since_cycle": int(cycle),
    }
    # Only drop scan-floor arming when length actually changes, not on first stable record.
    if prev_stable >= 0 and prev_stable != int(champ_words):
        patch["terminal_efficiency_scan_floor_words"] = 0
        patch["terminal_efficiency_scan_floor_heldout"] = 0.0
    _patch_progress_state(patch)


def record_terminal_efficiency_scan_floor(cycle: int, champ_words: int, heldout: float) -> None:
    """Remember a failed shrink scan at saturated held-out (rarer periodic rescans)."""
    if float(heldout) < 0.99 - 1e-6:
        return
    if not _terminal_efficiency_words_trackable(champ_words):
        return
    state = _read_progress_state()
    prev_words = int(state.get("terminal_efficiency_scan_floor_words", 0))
    prev_cycle = int(state.get("terminal_efficiency_scan_floor_cycle", 0))
    patch: dict[str, int | float] = {
        "terminal_efficiency_scan_floor_words": int(champ_words),
        "terminal_efficiency_scan_floor_heldout": float(heldout),
    }
    # Keep floor_cycle stable while length is unchanged so idle advances and
    # should_rescan_terminal_efficiency_compression can honor stall_interval.
    if prev_words != int(champ_words) or prev_cycle <= 0:
        patch["terminal_efficiency_scan_floor_cycle"] = int(cycle)
    _patch_progress_state(patch)


def compact_l30_certified_champion_words(
    metrics: dict,
    project_root: str | None = None,
) -> int:
    """Authoritative compact L30 champion length for terminal-efficiency tracking."""
    level = int(metrics.get("final_level", 0))
    verify = float(metrics.get("champion_verify_score", 0.0))
    words = int(metrics.get("champion_words", 0))
    prom_raw = float(metrics.get("promotion_raw_fitness", 0.0) or 0.0)
    if level >= 30 and verify >= 0.99 - 1e-6 and 12 <= words <= L30_COMPACT_MAX_WORDS and prom_raw > 300.0:
        return words
    if project_root is None:
        from evolution.mutation_helpers import _evolution_project_root

        project_root = _evolution_project_root()
    try:
        from evolution.champion_rollback import find_best_certified_champion

        results = os.path.join(project_root, "multi_cycle_results.jsonl")
        row = find_best_certified_champion(results, 30, min_verify=0.99, min_fitness=100.0)
        if row:
            row_words = int(row.get("champion_words", 0))
            if 12 <= row_words <= L30_COMPACT_MAX_WORDS:
                return row_words
    except (ImportError, OSError, TypeError, ValueError):
        pass
    state = _read_progress_state()
    certified = int(state.get("terminal_efficiency_certified_words", 0))
    if 12 <= certified <= L30_COMPACT_MAX_WORDS:
        return certified
    return 0


def reconcile_terminal_efficiency_compact_state(
    certified_words: int,
    cycle: int,
    *,
    heldout: float | None = None,
    verify_score: float | None = None,
) -> bool:
    """Clamp terminal-efficiency tracking to the certified compact champion length."""
    words = int(certified_words)
    if not (12 <= words <= L30_COMPACT_MAX_WORDS):
        return False
    if verify_score is not None and float(verify_score) < 0.99 - 1e-6:
        return False
    state = _read_progress_state()
    prev_certified = int(state.get("terminal_efficiency_certified_words", 0))
    if prev_certified > 0 and abs(words - prev_certified) > 1:
        if verify_score is None or float(verify_score) < 0.99 - 1e-6:
            words = prev_certified
    floor_heldout = (
        float(heldout)
        if heldout is not None and float(heldout) >= 0.99 - 1e-6
        else float(state.get("verify_score", 1.0))
    )
    patches: dict[str, int | float] = {
        "terminal_efficiency_certified_words": words,
    }
    stable = int(state.get("terminal_efficiency_stable_words", 0))
    floor = int(state.get("terminal_efficiency_scan_floor_words", 0))
    if stable != words and (stable <= 0 or abs(stable - words) > 1 or stable > L30_COMPACT_MAX_WORDS):
        patches["terminal_efficiency_stable_words"] = words
        patches["terminal_efficiency_stable_since_cycle"] = int(cycle)
    if floor <= 0 or abs(floor - words) > 1 or floor > L30_COMPACT_MAX_WORDS:
        patches["terminal_efficiency_scan_floor_words"] = words
        patches["terminal_efficiency_scan_floor_heldout"] = floor_heldout
        patches["terminal_efficiency_scan_floor_cycle"] = int(cycle)
    if len(patches) <= 1:
        _patch_progress_state(patches)
        return False
    _patch_progress_state(patches)
    return True


def maybe_rearm_scan_floor_above_champion(
    cycle_no: int,
    champion_program: list[int],
    *,
    verify_score: float | None = None,
) -> bool:
    """Re-arm compression floor when champion outgrew a shorter armed floor at saturated verify."""
    from evolution.mutation_helpers import scan_floor_plateau_armed

    if not champion_program:
        return False
    champ_words = len(champion_program)
    if not _terminal_efficiency_words_trackable(champ_words):
        return False
    armed, floor_w = scan_floor_plateau_armed()
    if not armed:
        return False
    if champ_words <= int(floor_w):
        return False
    if verify_score is not None and float(verify_score) < 0.99 - 1e-6:
        return False
    heldout = float(heldout_l30_verify_mean(list(champion_program), 32))
    if heldout < 0.99 - 1e-6:
        return False
    record_terminal_efficiency_scan_floor(int(cycle_no), champ_words, heldout)
    return True


def clear_terminal_efficiency_scan_floor() -> None:
    _patch_progress_state(
        {
            "terminal_efficiency_scan_floor_words": 0,
            "terminal_efficiency_scan_floor_heldout": 0.0,
            "terminal_efficiency_scan_floor_cycle": 0,
        }
    )


def terminal_efficiency_at_scan_floor(champ_words: int) -> bool:
    """True when a recent promotion scan failed at saturated held-out for this length."""
    state = _read_progress_state()
    return (
        int(state.get("terminal_efficiency_scan_floor_words", 0)) == int(champ_words)
        and float(state.get("terminal_efficiency_scan_floor_heldout", 0.0)) >= 0.99 - 1e-6
    )


def terminal_efficiency_rescan_interval(champ_words: int) -> int:
    """Cycles between promotion rescans while champion length is stable."""
    base = max(1, int(os.environ.get("EE_TERMINAL_EFFICIENCY_RESCAN_INTERVAL", "10")))
    state = _read_progress_state()
    floor_words = int(state.get("terminal_efficiency_scan_floor_words", 0))
    floor_h = float(state.get("terminal_efficiency_scan_floor_heldout", 0.0))
    if floor_words == int(champ_words) and floor_h >= 0.99 - 1e-6:
        return max(
            base,
            int(os.environ.get("EE_TERMINAL_EFFICIENCY_HELDOUT_FLOOR_RESCAN", "20")),
        )
    return base


def _compression_plateau_exhausted_at_words(champ_words: int) -> bool:
    """True when a prior ranked compression sweep at this length scored zero hits."""
    payload = _load_scan_floor_compression_miss()
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        if int(item.get("hits") or 0) > 0:
            continue
        if int(item.get("words") or 0) == int(champ_words):
            return True
    return False


def _compression_rank_disk_warm(program: list[int]) -> bool:
    """True when a prior ranked compression sweep found hits for this champion."""
    if not program:
        return False
    key = tuple(int(x) for x in program)
    disk = _load_scan_floor_compression_rank_disk(key)
    if not disk:
        return False
    miss = _load_scan_floor_compression_miss()
    entry = miss.get(_compression_rank_disk_key(key))
    if not isinstance(entry, dict):
        return bool(disk)
    return int(entry.get("hits") or 0) > 0


def should_rescan_terminal_efficiency_compression(
    cycle_no: int,
    champ_words: int,
    program: list[int] | None = None,
) -> bool:
    """Whether to run the expensive ranked compression search this cycle."""
    if not terminal_efficiency_at_scan_floor(champ_words):
        return True
    if _compression_plateau_exhausted_at_words(int(champ_words)):
        return False
    if _scan_floor_probe_stall_mode(int(champ_words)):
        state = _read_progress_state()
        floor_cycle = int(state.get("terminal_efficiency_scan_floor_cycle", 0))
        if floor_cycle <= 0:
            return True
        idle = int(cycle_no) - floor_cycle
        if idle <= 0:
            return True
        stall_interval = max(
            1,
            int(os.environ.get("EE_SCAN_FLOOR_STALL_COMPRESSION_RESCAN_INTERVAL", "3")),
        )
        if program and _compression_rank_disk_warm(list(program)):
            stall_interval = max(
                stall_interval,
                int(
                    os.environ.get(
                        "EE_SCAN_FLOOR_STALL_COMPRESSION_RESCAN_WARM_INTERVAL",
                        "12",
                    )
                ),
            )
        return idle % stall_interval == 0
    state = _read_progress_state()
    floor_cycle = int(state.get("terminal_efficiency_scan_floor_cycle", 0))
    if floor_cycle <= 0:
        return True
    idle = int(cycle_no) - floor_cycle
    if idle <= 0:
        return True
    interval = terminal_efficiency_rescan_interval(champ_words)
    return idle >= interval or idle % interval == 0


def terminal_efficiency_promote_quiescent(cycle_no: int, champ_words: int, champ_verify: float) -> bool:
    """Skip routine scans when length is stable at saturated verify (periodic rescan kept)."""
    if float(champ_verify) < 0.99 - 1e-6:
        return False
    state = _read_progress_state()
    stable_words = int(state.get("terminal_efficiency_stable_words", 0))
    stable_since = int(state.get("terminal_efficiency_stable_since_cycle", 0))
    if stable_words != int(champ_words) or stable_since <= 0:
        return False
    idle = int(cycle_no) - stable_since
    if idle < 3:
        return False
    interval = terminal_efficiency_rescan_interval(champ_words)
    return idle % interval != 0


def terminal_efficiency_promote_skip_quiescent(cycle_no: int, champ_words: int, champ_verify: float) -> bool:
    """Whether post-cycle efficiency promote may skip due to quiescent (not when prom_raw stalled)."""
    if not terminal_efficiency_promote_quiescent(cycle_no, champ_words, champ_verify):
        return False
    if _scan_floor_probe_stall_mode(int(champ_words)):
        return False
    from evolution.mutation_helpers import scan_floor_capability_growth_allowed

    if scan_floor_capability_growth_allowed(champ_words):
        return False
    return True


def terminal_efficiency_eval_budget(cycles_at_level: int | None = None) -> int:
    """More efficiency variants when terminal fitness has not moved in many cycles."""
    cycles = int(cycles_at_level if cycles_at_level is not None else terminal_cycles_at_level())
    if cycles >= 400:
        return 56
    if cycles >= 300:
        return 48
    if cycles >= 200:
        return 40
    if cycles >= 100:
        return 32
    if cycles >= 50:
        return 24
    return 20


def terminal_diversity_boost_tier(cycles_at_level: int | None = None) -> int:
    """Escalate exploration pressure when verify-saturated fitness is frozen."""
    cycles = int(cycles_at_level if cycles_at_level is not None else terminal_cycles_at_level())
    if cycles >= 300:
        return 7
    if cycles >= 200:
        return 6
    if cycles >= 100:
        return 5
    if cycles >= 50:
        return 4
    if cycles >= 28:
        return 3
    if cycles >= 20:
        return 2
    return 1


def terminal_diversity_boost_params(tier: int) -> dict[str, float]:
    """Per-tier floors for terminal diversity boost (mutation, inject, novelty)."""
    floors = {
        1: (0.55, 0.30, 0.55),
        2: (0.68, 0.40, 0.65),
        3: (0.75, 0.50, 0.72),
        4: (0.78, 0.55, 0.75),
        5: (0.82, 0.60, 0.78),
        6: (0.85, 0.65, 0.82),
        7: (0.88, 0.70, 0.85),
    }
    mu, inject, novelty = floors.get(int(tier), floors[1])
    diversity_threshold = 0.45 if tier < 3 else (0.60 if tier >= 6 else 0.55)
    return {
        "mutation_rate": mu,
        "diversity_inject_ratio": inject,
        "novelty_weight": novelty,
        "diversity_threshold": diversity_threshold,
    }


def terminal_verify_promote_active() -> bool:
    if os.environ.get("EE_TERMINAL_VERIFY_PROMOTE", "0") == "1":
        return True
    if _read_tick().get("terminal_diversity_boost"):
        return True
    state = _read_progress_state()
    depth = terminal_depth()
    level = int(state.get("level", 0))
    verify = float(state.get("verify_score", 0.0))
    cycles = int(state.get("cycles_at_level", 0))
    fitness = float(state.get("raw_fitness", 0.0))
    if terminal_certified_plateau(level, verify, fitness):
        return True
    return level >= depth and verify >= 0.99 - 1e-6 and cycles >= 8


def terminal_verify_selection_bonus(verify: float) -> float:
    """Extra elite-selection score at terminal depth to favor higher-verify programs."""
    if not terminal_verify_promote_active():
        return 0.0
    return TERMINAL_VERIFY_SELECTION_WEIGHT * max(0.0, verify)


class ScanFloorGateBaseline:
    """Cached champion held-out + side scores for probe qualification."""

    __slots__ = ("_champion", "_heldout", "_qualify_cache", "_side")

    def __init__(self, champion: list[int]) -> None:
        self._champion = tuple(int(x) for x in champion)
        self._heldout = float(heldout_l30_verify_mean(champion, 32))
        self._side = tuple(float(curriculum_verify_mean(champion, level)) for level in _COMPACT_SIDE_LEVELS)
        self._qualify_cache: dict[tuple[int, ...], bool] = {}

    def qualifies(self, variant: list[int]) -> bool:
        if not variant or len(variant) != len(self._champion):
            return False
        key = tuple(int(x) for x in variant)
        cached = self._qualify_cache.get(key)
        if cached is not None:
            return cached
        cand_h = heldout_l30_verify_mean(variant, 32)
        if cand_h + 1e-6 < self._heldout:
            self._qualify_cache[key] = False
            return False
        for idx, level in enumerate(_COMPACT_SIDE_LEVELS):
            if curriculum_verify_mean(variant, level) + 1e-6 < self._side[idx]:
                self._qualify_cache[key] = False
                return False
        self._qualify_cache[key] = True
        return True


def scan_floor_diversity_threshold() -> float:
    """Opcode-category diversity floor while held-out scan floor is active."""
    return float(os.environ.get("EE_SCAN_FLOOR_DIVERSITY_THRESHOLD", "0.99"))


_SCAN_FLOOR_GATE_KEY: tuple[int, ...] | None = None
_SCAN_FLOOR_GATE: ScanFloorGateBaseline | None = None


def get_scan_floor_gate(champion: list[int]) -> ScanFloorGateBaseline:
    """Reuse one gate baseline per champion (avoids re-scoring champion each probe build)."""
    global _SCAN_FLOOR_GATE_KEY, _SCAN_FLOOR_GATE
    key = tuple(int(x) for x in champion)
    if _SCAN_FLOOR_GATE is None or _SCAN_FLOOR_GATE_KEY != key:
        _SCAN_FLOOR_GATE_KEY = key
        _SCAN_FLOOR_GATE = ScanFloorGateBaseline(list(champion))
    return _SCAN_FLOOR_GATE


def scan_floor_probe_qualifies(champion: list[int], variant: list[int]) -> bool:
    """CPU gate: same length, held-out and compact side curriculum do not regress."""
    if not champion or not variant:
        return False
    return get_scan_floor_gate(champion).qualifies(variant)


def terminal_overlay_keep_length_variants(
    program: list[int],
    level: int,
    rng,
    *,
    max_evals: int = 12,
) -> list[list[int]]:
    """Same-length reference overlays (task_aware path) for scan-floor capability search."""
    from evolution.register_repair import reference_words
    from mutation_proposer import DirectedMutator

    lvl = int(level)
    if lvl < 29 or not program:
        return []
    champ_n = len(program)
    raw: list[list[int]] = []
    per_segment = max(2, min(6, int(max_evals) // 3))
    for segment in ("incremental", "level_only", "full"):
        seq, _ = reference_words(lvl, segment=segment)
        if not seq:
            continue
        for _ in range(per_segment):
            variant = DirectedMutator.overlay_seq_keep_length(program, seq, rng)
            if len(variant) == champ_n and variant != program:
                raw.append(variant)
    seen: set[tuple[int, ...]] = {tuple(program)}
    out: list[list[int]] = []
    for variant in raw:
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(variant))
    rng.shuffle(out)
    return out[: max(1, int(max_evals))]


_SCAN_FLOOR_WIDE_IMM_DELTAS: tuple[int, ...] = (
    -64,
    -32,
    -16,
    -8,
    -4,
    -2,
    -1,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
)


def terminal_imm_perturb_probe_variants(
    program: list[int],
    rng,
    *,
    max_evals: int = 24,
    max_attempts: int | None = None,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Same-length immediate nudges that pass held-out and side curriculum gates."""
    from mutation_proposer import DirectedMutator

    if not program:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    attempt_cap = int(max_attempts) if max_attempts is not None else max(24, int(max_evals) * 3)
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    attempts = 0
    while len(out) < int(max_evals) and attempts < attempt_cap:
        attempts += 1
        variant, _ = DirectedMutator.immediate_perturbation(list(program), rng)
        if len(variant) != champ_n:
            continue
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(list(variant))
    rng.shuffle(out)
    return out


def _wide_imm_probe_indices(program: list[int]) -> tuple[list[int], list[int]]:
    """Branch slots first, then other immediates — shared wide-imm indexing."""
    from milestone1.rv32i_mask import decode as rv32i_decode
    from mutation_proposer import _is_ebreak

    branch_candidates: list[int] = []
    other_candidates: list[int] = []
    for i, word in enumerate(program):
        if _is_ebreak(word):
            continue
        try:
            decoded = rv32i_decode(word, validate=False)
            name = str(decoded.get("name", "")) if decoded else ""
        except (ValueError, TypeError, KeyError, AttributeError):
            name = ""
        if name in _BRANCH_OPS:
            branch_candidates.append(i)
        else:
            other_candidates.append(i)
    return branch_candidates, other_candidates


def scan_floor_compact_tail_min_position(champ_n: int) -> int:
    """First index eligible for tail-focused capability probes at compact scan floor."""
    tail_words = max(
        6,
        int(os.environ.get("EE_SCAN_FLOOR_TAIL_FOCUS_WORDS", "12")),
    )
    return max(0, int(champ_n) - tail_words)


_PROBE_GATE_REPAIR_BUDGET: int | None = None


def _reset_probe_gate_repair_budget() -> None:
    """Cap expensive gate repairs during one scan-floor probe pool build."""
    global _PROBE_GATE_REPAIR_BUDGET
    if not _scan_floor_capability_stalled():
        _PROBE_GATE_REPAIR_BUDGET = 0
        return
    _PROBE_GATE_REPAIR_BUDGET = max(
        0,
        int(os.environ.get("EE_SCAN_FLOOR_PROBE_GATE_REPAIR_BUDGET", "8")),
    )


def _scan_floor_gate_repair_variant(
    program: list[int],
    variant: list[int],
    gate: ScanFloorGateBaseline,
    *,
    max_evals: int = 12,
) -> list[int] | None:
    """Return a gate-qualified equal-length variant, repairing near-misses when needed."""
    champ_t = tuple(program)
    if len(variant) != len(program) or tuple(variant) == champ_t:
        return None
    if gate.qualifies(variant):
        return list(variant)
    if not _scan_floor_capability_stalled():
        return None
    global _PROBE_GATE_REPAIR_BUDGET
    if _PROBE_GATE_REPAIR_BUDGET is not None and _PROBE_GATE_REPAIR_BUDGET <= 0:
        return None
    if _PROBE_GATE_REPAIR_BUDGET is not None:
        _PROBE_GATE_REPAIR_BUDGET -= 1
    from evolution.mutation_helpers import repair_model_guided_for_gate

    probe_evals = max(
        4,
        int(os.environ.get("EE_SCAN_FLOOR_PROBE_GATE_REPAIR_EVALS", "8")),
    )
    if _scan_floor_capability_stalled():
        probe_evals = max(
            probe_evals,
            int(os.environ.get("EE_SCAN_FLOOR_GATE_REPAIR_EVALS", "48")),
        )
    repaired = repair_model_guided_for_gate(
        list(variant),
        gate,
        max_evals=probe_evals,
        champion=list(program),
        prefix_count=int(scan_floor_compact_tail_min_position(len(program))),
        eval_cap=probe_evals,
    )
    if repaired and gate.qualifies(repaired) and tuple(repaired) != champ_t:
        return list(repaired)
    return None


def terminal_dual_imm_probe_variants(
    program: list[int],
    *,
    min_position: int = 0,
    max_evals: int = 32,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Two-slot wide-imm perturbations for stalled equal-length capability search."""
    from milestone1.rv32i_mask import decode as rv32i_decode
    from milestone1.rv32i_mask import encode as rv32i_encode

    if not program or len(program) < 4:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    branch_candidates, other_candidates = _wide_imm_probe_indices(program)
    min_pos = max(0, int(min_position))
    slots = [idx for idx in branch_candidates + other_candidates if int(idx) >= min_pos]
    if len(slots) < 2:
        return []
    deltas = _SCAN_FLOOR_WIDE_IMM_DELTAS
    combos: list[tuple[int, int, int, int]] = []
    for left in range(len(slots)):
        for right in range(left + 1, len(slots)):
            for d1 in deltas:
                for d2 in deltas:
                    combos.append((slots[left], int(d1), slots[right], int(d2)))
    if not combos:
        return []
    offset = int(start_offset) % len(combos)
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    for step in range(len(combos)):
        if len(out) >= int(max_evals):
            break
        idx1, d1, idx2, d2 = combos[(offset + step) % len(combos)]
        variant = list(program)
        for idx, delta in ((idx1, d1), (idx2, d2)):
            try:
                decoded = rv32i_decode(variant[idx], validate=False)
                if "imm" not in decoded:
                    raise ValueError("no imm")
                variant[idx] = int(
                    rv32i_encode(
                        name=str(decoded["name"]),
                        rd=int(decoded.get("rd", 0)),
                        rs1=int(decoded.get("rs1", 0)),
                        rs2=int(decoded.get("rs2", 0)),
                        imm=int(decoded.get("imm", 0)) + int(delta),
                        validate=True,
                    )
                )
            except (ValueError, TypeError, KeyError):
                break
        else:
            key = tuple(variant)
            if key in seen:
                continue
            seen.add(key)
            if gate.qualifies(variant):
                out.append(variant)
    return out


def terminal_tail_wide_imm_probe_variants(
    program: list[int],
    *,
    min_position: int = 0,
    max_evals: int = 24,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Wide-imm sweep restricted to tail slots (compact scan-floor capability hunt)."""
    from milestone1.rv32i_mask import decode as rv32i_decode
    from milestone1.rv32i_mask import encode as rv32i_encode

    if not program:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    branch_candidates, other_candidates = _wide_imm_probe_indices(program)
    min_pos = max(0, int(min_position))
    candidates = [idx for idx in branch_candidates + other_candidates if int(idx) >= min_pos]
    if not candidates:
        return []
    deltas = _SCAN_FLOOR_WIDE_IMM_DELTAS
    total_pairs = len(candidates) * len(deltas)
    if total_pairs <= 0:
        return []
    offset = int(start_offset) % total_pairs
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    for step in range(total_pairs):
        if len(out) >= int(max_evals):
            break
        pair_idx = (offset + step) % total_pairs
        idx = candidates[pair_idx // len(deltas)]
        delta = deltas[pair_idx % len(deltas)]
        try:
            decoded = rv32i_decode(program[idx], validate=False)
            if "imm" not in decoded:
                continue
            new_word = rv32i_encode(
                name=str(decoded["name"]),
                rd=int(decoded.get("rd", 0)),
                rs1=int(decoded.get("rs1", 0)),
                rs2=int(decoded.get("rs2", 0)),
                imm=int(decoded.get("imm", 0)) + int(delta),
                validate=True,
            )
        except (ValueError, TypeError, KeyError):
            continue
        variant = list(program)
        variant[idx] = int(new_word)
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(variant)
    return out


def terminal_wide_imm_sweep_probe_variants(
    program: list[int],
    *,
    max_evals: int = 24,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Deterministic (slot, delta) sweep so every branch displacement is tried over time."""
    from milestone1.rv32i_mask import decode as rv32i_decode
    from milestone1.rv32i_mask import encode as rv32i_encode

    if not program:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    branch_candidates, other_candidates = _wide_imm_probe_indices(program)
    candidates = branch_candidates + other_candidates
    if not candidates:
        return []
    deltas = _SCAN_FLOOR_WIDE_IMM_DELTAS
    total_pairs = len(candidates) * len(deltas)
    if total_pairs <= 0:
        return []
    offset = int(start_offset) % total_pairs
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    for step in range(total_pairs):
        if len(out) >= int(max_evals):
            break
        pair_idx = (offset + step) % total_pairs
        idx = candidates[pair_idx // len(deltas)]
        delta = deltas[pair_idx % len(deltas)]
        try:
            decoded = rv32i_decode(program[idx], validate=False)
            if "imm" not in decoded:
                continue
            new_word = rv32i_encode(
                name=str(decoded["name"]),
                rd=int(decoded.get("rd", 0)),
                rs1=int(decoded.get("rs1", 0)),
                rs2=int(decoded.get("rs2", 0)),
                imm=int(decoded.get("imm", 0)) + int(delta),
                validate=True,
            )
        except (ValueError, TypeError, KeyError):
            continue
        variant = list(program)
        variant[idx] = int(new_word)
        if len(variant) != champ_n:
            continue
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(variant)
    return out


_locus_probe_cache: dict[tuple[tuple[int, ...], int, int], tuple[tuple[int, ...], ...]] = {}


def clear_locus_probe_cache() -> None:
    """Reset per-cycle growth-material locus probe memoization."""
    _locus_probe_cache.clear()


def terminal_growth_material_locus_probe_variants(
    program: list[int],
    *,
    max_evals: int = 16,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Grafts and imm sweeps at growth-material diff loci (pre-tail index 26, etc.)."""
    from evolution.champion_rollback import _load_growth_material_entries
    from milestone1.rv32i_mask import decode as rv32i_decode
    from milestone1.rv32i_mask import encode as rv32i_encode

    if not program:
        return []
    cache_key = (
        tuple(int(x) for x in program),
        int(max_evals),
        int(start_offset),
    )
    cached = _locus_probe_cache.get(cache_key)
    if cached is not None:
        return [list(v) for v in cached]
    champ_n = len(program)
    champ_t = tuple(int(x) for x in program)
    gate = gate or get_scan_floor_gate(program)
    _anchor, entries = _load_growth_material_entries(list(program))
    if not entries:
        return []
    champ = list(program)
    material_progs = [list(entry.get("program")) for entry in entries if isinstance(entry.get("program"), list)]
    try:
        from evolution.mutation_helpers import _expand_growth_material_for_grafts

        material_progs = _expand_growth_material_for_grafts(
            champ,
            material_progs,
            gate=gate,
            cycle_no=int(start_offset),
        )
    except ImportError:
        pass
    index_bytes: dict[int, set[int]] = {}
    for prog in material_progs:
        if not prog or len(prog) < champ_n:
            continue
        for i in range(champ_n):
            alt = int(prog[i])
            if alt != int(champ[i]):
                index_bytes.setdefault(int(i), set()).add(alt)
    if not index_bytes:
        return []
    ranked_indices = sorted(
        index_bytes.keys(),
        key=lambda i: (-len(index_bytes[i]), int(i)),
    )[: max(1, int(os.environ.get("EE_SCAN_FLOOR_GROWTH_LOCUS_INDICES", "6")))]
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    for idx in ranked_indices:
        for alt in sorted(index_bytes[idx]):
            if len(out) >= int(max_evals):
                break
            variant = list(champ)
            variant[idx] = int(alt)
            key = tuple(variant)
            if key in seen:
                continue
            seen.add(key)
            if gate.qualifies(variant):
                out.append(variant)
    pair_budget = max(
        0,
        int(os.environ.get("EE_SCAN_FLOOR_GROWTH_LOCUS_PAIR_BUDGET", "6")),
    )
    if pair_budget > 0 and len(ranked_indices) >= 2 and len(out) < int(max_evals):
        import itertools

        pair_trials = 0
        for i, j in itertools.combinations(ranked_indices[:4], 2):
            if pair_trials >= pair_budget or len(out) >= int(max_evals):
                break
            alts_i = sorted(index_bytes[i])[:4]
            alts_j = sorted(index_bytes[j])[:4]
            for ai in alts_i:
                for aj in alts_j:
                    if pair_trials >= pair_budget or len(out) >= int(max_evals):
                        break
                    pair_trials += 1
                    variant = list(champ)
                    variant[i] = int(ai)
                    variant[j] = int(aj)
                    key = tuple(variant)
                    if key in seen:
                        continue
                    seen.add(key)
                    if gate.qualifies(variant):
                        out.append(variant)
    triple_budget = max(
        0,
        int(os.environ.get("EE_SCAN_FLOOR_GROWTH_LOCUS_TRIPLE_BUDGET", "4")),
    )
    if triple_budget > 0 and len(ranked_indices) >= 3 and len(out) < int(max_evals):
        import itertools

        triple_trials = 0
        for i, j, k in itertools.combinations(ranked_indices[:5], 3):
            if triple_trials >= triple_budget or len(out) >= int(max_evals):
                break
            for ai in sorted(index_bytes[i])[:2]:
                for aj in sorted(index_bytes[j])[:2]:
                    for ak in sorted(index_bytes[k])[:2]:
                        if triple_trials >= triple_budget or len(out) >= int(max_evals):
                            break
                        triple_trials += 1
                        variant = list(champ)
                        variant[i] = int(ai)
                        variant[j] = int(aj)
                        variant[k] = int(ak)
                        key = tuple(variant)
                        if key in seen:
                            continue
                        seen.add(key)
                        if gate.qualifies(variant):
                            out.append(variant)
    deltas = _SCAN_FLOOR_WIDE_IMM_DELTAS
    pairs = [(idx, delta) for idx in ranked_indices for delta in deltas]
    if pairs and len(out) < int(max_evals):
        offset = int(start_offset) % len(pairs)
        for step in range(len(pairs)):
            if len(out) >= int(max_evals):
                break
            idx, delta = pairs[(offset + step) % len(pairs)]
            try:
                decoded = rv32i_decode(champ[idx], validate=False)
                if "imm" not in decoded:
                    continue
                new_word = rv32i_encode(
                    name=str(decoded["name"]),
                    rd=int(decoded.get("rd", 0)),
                    rs1=int(decoded.get("rs1", 0)),
                    rs2=int(decoded.get("rs2", 0)),
                    imm=int(decoded.get("imm", 0)) + int(delta),
                    validate=True,
                )
            except (ValueError, TypeError, KeyError):
                continue
            variant = list(champ)
            variant[idx] = int(new_word)
            key = tuple(variant)
            if key in seen:
                continue
            seen.add(key)
            if gate.qualifies(variant):
                out.append(variant)
    if out:
        from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

        out.sort(
            key=lambda variant: float(compact_l30_promotion_raw_fitness(list(variant))),
            reverse=True,
        )
    _locus_probe_cache[cache_key] = tuple(tuple(int(x) for x in v) for v in out)
    if len(_locus_probe_cache) > 48:
        _locus_probe_cache.pop(next(iter(_locus_probe_cache)))
    return out


def terminal_wide_imm_perturb_probe_variants(
    program: list[int],
    rng,
    *,
    max_evals: int = 24,
    max_attempts: int | None = None,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Displacement-scale immediate nudges for scan-floor capability search.

    Fine-step ``immediate_perturbation`` uses +/-8 at most; saturated L30
    champions often need larger branch displacements to improve raw fitness
    while held-out and side curriculum stay at 1.0.
    """
    from milestone1.rv32i_mask import decode as rv32i_decode
    from milestone1.rv32i_mask import encode as rv32i_encode

    if not program:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    branch_candidates, other_candidates = _wide_imm_probe_indices(program)
    candidates = branch_candidates + other_candidates
    if not candidates:
        return []
    attempt_cap = int(max_attempts) if max_attempts is not None else max(48, int(max_evals) * 4)
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    attempts = 0
    cand_idx = 0
    while len(out) < int(max_evals) and attempts < attempt_cap:
        attempts += 1
        idx = candidates[cand_idx % len(candidates)]
        cand_idx += 1
        try:
            decoded = rv32i_decode(program[idx], validate=False)
            if "imm" not in decoded:
                continue
            delta = rng.choice(_SCAN_FLOOR_WIDE_IMM_DELTAS)
            new_word = rv32i_encode(
                name=str(decoded["name"]),
                rd=int(decoded.get("rd", 0)),
                rs1=int(decoded.get("rs1", 0)),
                rs2=int(decoded.get("rs2", 0)),
                imm=int(decoded.get("imm", 0)) + int(delta),
                validate=True,
            )
        except (ValueError, TypeError, KeyError):
            continue
        variant = list(program)
        variant[idx] = int(new_word)
        if len(variant) != champ_n:
            continue
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(variant)
    return out


def terminal_block_reorder_probe_variants(
    program: list[int],
    rng,
    *,
    max_evals: int = 12,
    max_attempts: int | None = None,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Rotate contiguous instruction blocks; gate-qualified reorder probes."""
    from mutation_proposer import DirectedMutator

    if not program or len(program) < 6:
        return []
    champ_n = len(program)
    gate = gate or get_scan_floor_gate(program)
    attempt_cap = int(max_attempts) if max_attempts is not None else max(64, int(max_evals) * 4)
    seen: set[tuple[int, ...]] = {tuple(program)}
    out: list[list[int]] = []
    attempts = 0
    while len(out) < int(max_evals) and attempts < attempt_cap:
        attempts += 1
        variant, _ = DirectedMutator._reorder_block(list(program), rng)
        if len(variant) != champ_n:
            continue
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(variant)
    rng.shuffle(out)
    return out


def terminal_deterministic_instr_swap_probe_variants(
    program: list[int],
    *,
    max_evals: int = 12,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Exhaustive pair-swaps (cycle-rotated) for stalled scan-floor capability search."""
    from mutation_proposer import _is_ebreak

    if not program or len(program) < 4:
        return []
    champ_n = len(program)
    gate = gate or get_scan_floor_gate(program)
    candidates = [i for i in range(champ_n) if not _is_ebreak(program[i])]
    pairs: list[tuple[int, int]] = []
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            pairs.append((candidates[left], candidates[right]))
    if not pairs:
        return []
    seen: set[tuple[int, ...]] = {tuple(program)}
    out: list[list[int]] = []
    offset = int(start_offset)
    for step in range(len(pairs)):
        if len(out) >= int(max_evals):
            break
        i, j = pairs[(offset + step) % len(pairs)]
        variant = list(program)
        variant[i], variant[j] = variant[j], variant[i]
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(variant)
    return out


def terminal_instr_swap_probe_variants(
    program: list[int],
    rng,
    *,
    max_evals: int = 12,
    max_attempts: int | None = None,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Swap two non-EBREAK slots; gate-qualified same-length reorder probes."""
    from mutation_proposer import _is_ebreak

    if not program or len(program) < 4:
        return []
    champ_n = len(program)
    gate = gate or get_scan_floor_gate(program)
    attempt_cap = int(max_attempts) if max_attempts is not None else max(32, int(max_evals) * 4)
    candidates = [i for i in range(champ_n) if not _is_ebreak(program[i])]
    if len(candidates) < 2:
        return []
    seen: set[tuple[int, ...]] = {tuple(program)}
    out: list[list[int]] = []
    attempts = 0
    while len(out) < int(max_evals) and attempts < attempt_cap:
        attempts += 1
        i, j = rng.sample(candidates, 2)
        variant = list(program)
        variant[i], variant[j] = variant[j], variant[i]
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(variant)
    rng.shuffle(out)
    return out


def _scan_floor_uniform_crossover(
    parent_a: list[int],
    parent_b: list[int],
    rng,
) -> list[int]:
    """Per-slot uniform crossover; EBREAK slots are never mixed."""
    from mutation_proposer import _is_ebreak

    child = list(parent_a)
    for idx, (a_word, b_word) in enumerate(zip(parent_a, parent_b, strict=False)):
        if _is_ebreak(a_word) or _is_ebreak(b_word):
            continue
        if rng.random() < 0.5:
            child[idx] = b_word
    return child


def _deterministic_bit_crossover(
    parent_a: list[int],
    parent_b: list[int],
    mask_bits: int,
) -> list[int]:
    """Per-slot crossover from a rotating bit mask; EBREAK slots are never mixed."""
    from mutation_proposer import _is_ebreak

    child = list(parent_a)
    mask = int(mask_bits)
    for idx, (a_word, b_word) in enumerate(zip(parent_a, parent_b, strict=False)):
        if _is_ebreak(a_word) or _is_ebreak(b_word):
            continue
        if (mask >> (idx % 31)) & 1:
            child[idx] = b_word
    return child


def terminal_archive_word_diff_probe_variants(
    program: list[int],
    archive_parents: list[list[int]],
    *,
    max_evals: int = 8,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Single-word grafts at indices where an archive parent differs from champion."""
    from mutation_proposer import _is_ebreak

    if not program or not archive_parents:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    pool: list[list[int]] = []
    seen_pool: set[tuple[int, ...]] = set()
    for parent in archive_parents:
        if not parent or len(parent) != champ_n:
            continue
        key = tuple(parent)
        if key == champ_t or key in seen_pool:
            continue
        seen_pool.add(key)
        pool.append(list(parent))
    if not pool:
        return []
    pool.sort(key=_scan_floor_probe_rank_key, reverse=True)
    candidates: list[tuple[int, int, int]] = []
    for pi, parent in enumerate(pool):
        for idx in range(1, champ_n - 1):
            if _is_ebreak(program[idx]) or _is_ebreak(parent[idx]):
                continue
            if parent[idx] == program[idx]:
                continue
            candidates.append((pi, idx, parent[idx]))
    if not candidates:
        return []
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    offset = int(start_offset)
    for step in range(len(candidates)):
        if len(out) >= int(max_evals):
            break
        pi, idx, word = candidates[(offset + step) % len(candidates)]
        variant = list(program)
        variant[idx] = int(word)
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(variant)
    return out


def terminal_segment_archive_crossover_probe_variants(
    program: list[int],
    archive_parents: list[list[int]],
    *,
    max_evals: int = 8,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
    min_span: int = 1,
    max_span: int = 10,
) -> list[list[int]]:
    """Contiguous window swaps from archive parents (between bit-cross and full graft)."""
    from mutation_proposer import _is_ebreak

    if not program or not archive_parents:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    pool: list[list[int]] = []
    seen_pool: set[tuple[int, ...]] = set()
    for parent in archive_parents:
        if not parent or len(parent) != champ_n:
            continue
        key = tuple(parent)
        if key == champ_t or key in seen_pool:
            continue
        seen_pool.add(key)
        pool.append(list(parent))
    if not pool:
        return []
    pool.sort(key=_scan_floor_probe_rank_key, reverse=True)
    body_end = champ_n - 1
    windows: list[tuple[int, int, int]] = []
    for pi in range(len(pool)):
        for span in range(int(min_span), int(max_span) + 1):
            if span >= body_end:
                continue
            for start in range(1, body_end - span):
                if any(_is_ebreak(program[start + k]) for k in range(span)):
                    continue
                windows.append((pi, start, span))
    if not windows:
        return []
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    offset = int(start_offset)
    attempt_cap = max(int(max_evals) * 4, int(max_evals))
    for step in range(min(len(windows), attempt_cap)):
        if len(out) >= int(max_evals):
            break
        pi, start, span = windows[(offset + step) % len(windows)]
        parent = pool[pi]
        variant = list(program)
        variant[start : start + span] = parent[start : start + span]
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        repaired = _scan_floor_gate_repair_variant(program, variant, gate)
        if repaired is not None:
            out.append(repaired)
    return out


def terminal_archive_consensus_graft_probe_variants(
    program: list[int],
    archive_parents: list[list[int]],
    *,
    max_evals: int = 8,
    start_offset: int = 0,
    min_agreement: int = 2,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Single-position grafts where multiple archive parents agree on an instruction."""
    if not program or not archive_parents:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    pool: list[list[int]] = []
    seen_pool: set[tuple[int, ...]] = {champ_t}
    for parent in archive_parents:
        if not parent or len(parent) != champ_n:
            continue
        key = tuple(parent)
        if key in seen_pool:
            continue
        seen_pool.add(key)
        pool.append(list(parent))
    if not pool:
        return []
    pool.sort(key=_scan_floor_probe_rank_key, reverse=True)
    from collections import Counter

    candidates: list[tuple[int, int, float]] = []
    for pos in range(champ_n):
        counts = Counter(parent[pos] for parent in pool)
        counts.pop(int(program[pos]), None)
        if not counts:
            continue
        word, agree = counts.most_common(1)[0]
        if agree < int(min_agreement):
            continue
        parent_rank = max(_scan_floor_probe_rank_key(parent)[0] for parent in pool if parent[pos] == word)
        candidates.append((pos, int(word), parent_rank))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[2], reverse=True)
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    for step in range(len(candidates)):
        if len(out) >= int(max_evals):
            break
        pos, word, _ = candidates[(int(start_offset) + step) % len(candidates)]
        variant = list(program)
        variant[pos] = word
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        repaired = _scan_floor_gate_repair_variant(program, variant, gate)
        if repaired is not None:
            out.append(repaired)
    return out


def terminal_dual_word_diff_probe_variants(
    program: list[int],
    archive_parents: list[list[int]],
    *,
    max_evals: int = 8,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Two-position grafts across archive parents (cross-parent word_diff pairs)."""
    from mutation_proposer import _is_ebreak

    if not program or not archive_parents:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    pool: list[list[int]] = []
    seen_pool: set[tuple[int, ...]] = set()
    for parent in archive_parents:
        if not parent or len(parent) != champ_n:
            continue
        key = tuple(parent)
        if key == champ_t or key in seen_pool:
            continue
        seen_pool.add(key)
        pool.append(list(parent))
    if not pool:
        return []
    pool.sort(key=_scan_floor_probe_rank_key, reverse=True)
    diff_entries: list[tuple[int, int]] = []
    for parent in pool:
        for idx in range(1, champ_n - 1):
            if _is_ebreak(program[idx]) or _is_ebreak(parent[idx]):
                continue
            if parent[idx] != program[idx]:
                diff_entries.append((int(idx), int(parent[idx])))
    if len(diff_entries) < 2:
        return []
    pairs: list[tuple[int, int]] = []
    for left in range(len(diff_entries)):
        i_idx, _i_word = diff_entries[left]
        for right in range(left + 1, len(diff_entries)):
            j_idx, _j_word = diff_entries[right]
            if i_idx == j_idx:
                continue
            pairs.append((left, right))
    if not pairs:
        return []
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    offset = int(start_offset)
    for step in range(len(pairs)):
        if len(out) >= int(max_evals):
            break
        li, ri = pairs[(offset + step) % len(pairs)]
        i_idx, i_word = diff_entries[li]
        j_idx, j_word = diff_entries[ri]
        variant = list(program)
        variant[i_idx] = i_word
        variant[j_idx] = j_word
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        repaired = _scan_floor_gate_repair_variant(program, variant, gate)
        if repaired is not None:
            out.append(repaired)
    return out


def terminal_tri_word_diff_probe_variants(
    program: list[int],
    archive_parents: list[list[int]],
    *,
    max_evals: int = 12,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Three-position cross-parent grafts with gate repair (capability-stall search)."""
    import itertools

    from mutation_proposer import _is_ebreak

    if not program or not archive_parents:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    pool: list[list[int]] = []
    seen_pool: set[tuple[int, ...]] = set()
    for parent in archive_parents:
        if not parent or len(parent) != champ_n:
            continue
        key = tuple(parent)
        if key == champ_t or key in seen_pool:
            continue
        seen_pool.add(key)
        pool.append(list(parent))
    if not pool:
        return []
    pool.sort(key=_scan_floor_probe_rank_key, reverse=True)
    index_words: dict[int, set[int]] = {}
    for parent in pool:
        for idx in range(1, champ_n - 1):
            if _is_ebreak(program[idx]) or _is_ebreak(parent[idx]):
                continue
            if parent[idx] != program[idx]:
                index_words.setdefault(int(idx), set()).add(int(parent[idx]))
    ranked_indices = sorted(
        index_words.keys(),
        key=lambda i: (-len(index_words[i]), int(i)),
    )[: max(3, int(os.environ.get("EE_SCAN_FLOOR_TRI_GRAFT_INDICES", "8")))]
    if len(ranked_indices) < 3:
        return []
    triples: list[tuple[int, int, int]] = []
    for combo in itertools.combinations(ranked_indices, 3):
        triples.append(combo)
    if not triples:
        return []
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    offset = int(start_offset)
    pair_cap = max(
        1,
        int(os.environ.get("EE_SCAN_FLOOR_TRI_GRAFT_PAIR_BUDGET", "24")),
    )
    trials = 0
    for step in range(len(triples)):
        if len(out) >= int(max_evals) or trials >= pair_cap:
            break
        i, j, k = triples[(offset + step) % len(triples)]
        words_i = sorted(index_words[i])[:3]
        words_j = sorted(index_words[j])[:3]
        words_k = sorted(index_words[k])[:3]
        for wi in words_i:
            for wj in words_j:
                for wk in words_k:
                    if trials >= pair_cap or len(out) >= int(max_evals):
                        break
                    trials += 1
                    variant = list(program)
                    variant[i] = int(wi)
                    variant[j] = int(wj)
                    variant[k] = int(wk)
                    key = tuple(variant)
                    if key in seen:
                        continue
                    seen.add(key)
                    repaired = _scan_floor_gate_repair_variant(program, variant, gate)
                    if repaired is not None:
                        out.append(repaired)
    return out


def terminal_long_archive_compress_probe_variants(
    program: list[int],
    long_parents: list[list[int]],
    *,
    max_evals: int = 12,
    cycle_no: int | None = None,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Compress longer archive parents to champion length; rank by promotion_raw."""
    from evolution.mutation_helpers import equal_length_from_longer_growth

    if not program or not long_parents:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    cycle_key = int(cycle_no) if cycle_no is not None else int(terminal_cycles_at_level())
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    parents = sorted(
        [list(p) for p in long_parents if p and len(p) > champ_n],
        key=len,
    )
    for parent in parents:
        if len(out) >= int(max_evals):
            break
        drop_trials = max(24, len(parent) - champ_n + 8)
        compressed = equal_length_from_longer_growth(
            list(program),
            list(parent),
            gate=gate,
            max_drop_trials=drop_trials,
            cycle_no=cycle_key,
        )
        key = tuple(compressed)
        if key in seen or len(compressed) != champ_n or not gate.qualifies(compressed):
            continue
        seen.add(key)
        out.append(list(compressed))
    out.sort(
        key=lambda variant: _scan_floor_probe_capability_score(program, variant),
        reverse=True,
    )
    return out[: int(max_evals)]


def terminal_growth_prefix_tail_splice_probe_variants(
    program: list[int],
    long_parents: list[list[int]],
    *,
    max_evals: int = 12,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Equal-length splice: champion prefix + growth-material tail at true LCP."""
    from evolution.mutation_helpers import growth_anchor_lcp_prefix
    from mutation_proposer import _is_ebreak

    if not program or not long_parents:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    candidates: list[tuple[int, int, int]] = []
    for pi, parent in enumerate(long_parents):
        if not parent or len(parent) < champ_n:
            continue
        lcp = growth_anchor_lcp_prefix(list(program), list(parent))
        for delta in (-2, -1, 0, 1, 2):
            c_start = max(1, min(champ_n - 2, lcp + int(delta)))
            need = champ_n - c_start
            if c_start + need > len(parent):
                continue
            if c_start > 0 and _is_ebreak(program[c_start - 1]):
                continue
            candidates.append((pi, c_start, abs(int(delta))))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[2], item[1]))
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    offset = int(start_offset)
    for step in range(len(candidates)):
        if len(out) >= int(max_evals):
            break
        pi, c_start, _prio = candidates[(offset + step) % len(candidates)]
        parent = long_parents[pi]
        variant = list(program[:c_start]) + list(parent[c_start : c_start + (champ_n - c_start)])
        if len(variant) != champ_n:
            continue
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        repaired = _scan_floor_gate_repair_variant(program, variant, gate)
        if repaired is not None:
            out.append(repaired)
        elif gate.qualifies(variant):
            out.append(variant)
    return out


def terminal_long_parent_window_graft_probe_variants(
    program: list[int],
    long_parents: list[list[int]],
    *,
    max_evals: int = 8,
    start_offset: int = 0,
    min_span: int = 3,
    max_span: int = 8,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Sliding-window grafts from longer archive parents into the champion."""
    from evolution.mutation_helpers import growth_anchor_lcp_prefix
    from mutation_proposer import _is_ebreak

    if not program or not long_parents:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    candidates: list[tuple[int, int, int, int, int]] = []
    for pi, parent in enumerate(long_parents):
        if not parent or len(parent) <= champ_n:
            continue
        lcp = growth_anchor_lcp_prefix(list(program), list(parent))
        for span in range(int(min_span), int(max_span) + 1):
            if span >= champ_n - 2:
                continue
            for p_start in range(max(0, lcp - 2), len(parent) - span + 1):
                for c_start in range(1, champ_n - span - 1):
                    if _is_ebreak(program[c_start]) or _is_ebreak(program[c_start + span - 1]):
                        continue
                    if program[c_start : c_start + span] == parent[p_start : p_start + span]:
                        continue
                    candidates.append((pi, p_start, c_start, span, len(parent)))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[4], -item[3], item[1]), reverse=True)
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    offset = int(start_offset)
    for step in range(len(candidates)):
        if len(out) >= int(max_evals):
            break
        pi, p_start, c_start, span, _ = candidates[(offset + step) % len(candidates)]
        parent = long_parents[pi]
        variant = list(program)
        variant[c_start : c_start + span] = parent[p_start : p_start + span]
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        repaired = _scan_floor_gate_repair_variant(program, variant, gate)
        if repaired is not None:
            out.append(repaired)
    return out


def terminal_archive_intercross_probe_variants(
    program: list[int],
    archive_parents: list[list[int]],
    *,
    max_evals: int = 8,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Single-cut crossover between archive parents (rotates across cycles)."""
    if not program or len(archive_parents) < 2:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    pool: list[list[int]] = []
    seen_pool: set[tuple[int, ...]] = {champ_t}
    for parent in archive_parents:
        if not parent or len(parent) != champ_n:
            continue
        key = tuple(parent)
        if key in seen_pool:
            continue
        seen_pool.add(key)
        pool.append(list(parent))
    if len(pool) < 2:
        return []
    pool.sort(key=_scan_floor_probe_rank_key, reverse=True)
    pairs: list[tuple[int, int, int]] = []
    for left in range(len(pool)):
        for right in range(left + 1, len(pool)):
            for cut in range(8, champ_n - 8):
                pairs.append((left, right, cut))
    if not pairs:
        return []
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    offset = int(start_offset)
    attempt_cap = max(int(max_evals) * 4, int(max_evals))
    for step in range(min(len(pairs), attempt_cap)):
        if len(out) >= int(max_evals):
            break
        left, right, cut = pairs[(offset + step) % len(pairs)]
        variant = pool[left][:cut] + pool[right][cut:]
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(list(variant))
    return out


def terminal_archive_program_probe_variants(
    program: list[int],
    archive_parents: list[list[int]],
    *,
    max_evals: int = 8,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Inject gate-qualified archive programs as same-length scan-floor probes."""
    if not program or not archive_parents:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    pool: list[list[int]] = []
    seen_pool: set[tuple[int, ...]] = set()
    for parent in archive_parents:
        if not parent or len(parent) != champ_n:
            continue
        key = tuple(parent)
        if key == champ_t or key in seen_pool:
            continue
        if not gate.qualifies(list(parent)):
            continue
        seen_pool.add(key)
        pool.append(list(parent))
    if not pool:
        return []
    pool.sort(key=_scan_floor_probe_rank_key, reverse=True)
    offset = int(start_offset) % len(pool)
    out: list[list[int]] = []
    for step in range(min(int(max_evals), len(pool))):
        out.append(pool[(offset + step) % len(pool)])
    return out


def terminal_archive_mutate_probe_variants(
    program: list[int],
    archive_parents: list[list[int]],
    *,
    max_evals: int = 8,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Wide-imm sweep on historical archive parents (not only the live champion)."""
    if not program or not archive_parents:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    parents: list[list[int]] = []
    seen_parent: set[tuple[int, ...]] = set()
    for parent in archive_parents:
        if not parent or len(parent) != champ_n:
            continue
        key = tuple(parent)
        if key == champ_t or key in seen_parent:
            continue
        seen_parent.add(key)
        parents.append(list(parent))
    if not parents:
        return []
    parents.sort(key=_scan_floor_probe_rank_key, reverse=True)
    per_parent = max(2, int(max_evals) // max(1, len(parents)))
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    for parent_idx, parent in enumerate(parents):
        sweep = terminal_wide_imm_sweep_probe_variants(
            parent,
            max_evals=per_parent,
            start_offset=int(start_offset) + parent_idx * per_parent,
            gate=gate,
        )
        for variant in sweep:
            key = tuple(variant)
            if key in seen:
                continue
            seen.add(key)
            out.append(list(variant))
            if len(out) >= int(max_evals):
                return out
    return out


def terminal_deterministic_archive_crossover_probe_variants(
    program: list[int],
    archive_parents: list[list[int]],
    *,
    max_evals: int = 8,
    start_offset: int = 0,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Rotate deterministic champion×archive crossovers across cycles."""
    if not program or not archive_parents:
        return []
    champ_n = len(program)
    champ_t = tuple(program)
    gate = gate or get_scan_floor_gate(program)
    pool: list[list[int]] = []
    seen_pool: set[tuple[int, ...]] = set()
    for parent in archive_parents:
        if not parent or len(parent) != champ_n:
            continue
        key = tuple(parent)
        if key == champ_t or key in seen_pool:
            continue
        seen_pool.add(key)
        pool.append(list(parent))
    if not pool:
        return []
    pool.sort(key=_scan_floor_probe_rank_key, reverse=True)
    attempt_cap = max(int(max_evals) * len(pool), int(max_evals))
    seen: set[tuple[int, ...]] = {champ_t}
    out: list[list[int]] = []
    offset = int(start_offset)
    for step in range(attempt_cap):
        if len(out) >= int(max_evals):
            break
        parent = pool[(offset + step) % len(pool)]
        mask = (offset + step) * 2654435761
        variant = _deterministic_bit_crossover(program, parent, mask)
        if len(variant) != champ_n:
            continue
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(list(variant))
    return out


def terminal_crossover_probe_variants(
    program: list[int],
    parents: list[list[int]],
    rng,
    *,
    max_evals: int = 8,
    max_attempts: int | None = None,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Recombine gate-qualified parents into new same-length probe variants."""
    if not program or len(parents) < 2:
        return []
    champ_n = len(program)
    gate = gate or get_scan_floor_gate(program)
    pool = [list(p) for p in parents if p and len(p) == champ_n and tuple(p) != tuple(program)]
    if len(pool) < 2:
        return []
    attempt_cap = int(max_attempts) if max_attempts is not None else max(24, int(max_evals) * 4)
    seen: set[tuple[int, ...]] = {tuple(program)}
    out: list[list[int]] = []
    attempts = 0
    while len(out) < int(max_evals) and attempts < attempt_cap:
        attempts += 1
        parent_a, parent_b = rng.sample(pool, 2)
        variant = _scan_floor_uniform_crossover(parent_a, parent_b, rng)
        if len(variant) != champ_n:
            continue
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(list(variant))
    rng.shuffle(out)
    return out


def _probe_rank_heldout_seeds() -> int:
    """Held-out seed count for probe ordering only (not promotion gates)."""
    return max(4, int(os.environ.get("EE_SCAN_FLOOR_PROBE_HELDOUT_SEEDS", "8")))


def scan_floor_compression_hot(
    program: list[int],
    rng=None,
    *,
    max_evals: int | None = None,
) -> bool:
    """True when held-out-qualified shorter compression candidates exist."""
    if not program or not terminal_efficiency_at_scan_floor(len(program)):
        return False
    import random

    from evolution.champion_bloat_guard import level_word_cap

    rng = rng if rng is not None else random.Random(0)
    word_cap = level_word_cap(30)
    span = terminal_efficiency_drop_span(len(program), word_cap)
    budget = (
        int(max_evals)
        if max_evals is not None
        else max(12, int(os.environ.get("EE_SCAN_FLOOR_COMPRESS_PROBE_BUDGET", "16")))
    )
    compress_variants = terminal_efficiency_variants(
        program,
        rng,
        max_evals=budget,
        max_drop_span=min(2, span),
    )
    champ_n = len(program)
    return any(
        len(variant) < champ_n and scan_floor_compression_probe_qualifies(program, variant)
        for variant in compress_variants
    )


def scan_floor_compression_probe_qualifies(champion: list[int], variant: list[int]) -> bool:
    """Held-out-safe shorter probe candidates (length-efficiency promotion path)."""
    if not champion or not variant or len(variant) >= len(champion):
        return False
    seeds = _probe_rank_heldout_seeds()
    champ_h = float(heldout_l30_verify_mean(list(champion), seeds))
    cand_h = float(heldout_l30_verify_mean(list(variant), seeds))
    if cand_h + 1e-6 < champ_h:
        return False
    champ_side = float(compact_curriculum_side_score(list(champion)))
    return compact_curriculum_side_score(list(variant)) + 1e-6 >= champ_side


def scan_floor_compression_promote_qualifies(champion: list[int], variant: list[int]) -> bool:
    """Shorter candidates that pass the same held-out floor as post-cycle efficiency promote."""
    if not champion or not variant or len(variant) >= len(champion):
        return False
    champ_h = float(heldout_l30_verify_mean(list(champion), 32))
    cand_h = float(heldout_l30_verify_mean(list(variant), 32))
    if cand_h + 1e-6 < champ_h:
        return False
    champ_side = float(compact_curriculum_side_score(list(champion)))
    return compact_curriculum_side_score(list(variant)) + 1e-6 >= champ_side


_PROBE_RANK_CHAMPION: list[int] | None = None


def _set_scan_floor_probe_rank_champion(champion: list[int]) -> None:
    global _PROBE_RANK_CHAMPION
    _PROBE_RANK_CHAMPION = [int(x) for x in champion]


def _clear_scan_floor_probe_rank_champion() -> None:
    global _PROBE_RANK_CHAMPION
    _PROBE_RANK_CHAMPION = None


def _champion_l10_weak_for_probe_rank() -> bool:
    champ = _PROBE_RANK_CHAMPION
    if not champ:
        return False
    return float(curriculum_verify_mean(list(champ), 10)) < 0.85 - 1e-6


def _scan_floor_probe_rank_key(variant: list[int]) -> tuple[float, ...]:
    """Prefer higher held-out and compact side margins when picking probes."""
    held = float(heldout_l30_verify_mean(variant, _probe_rank_heldout_seeds()))
    side = float(compact_curriculum_side_score(variant))
    if _champion_l10_weak_for_probe_rank():
        return (held, float(curriculum_verify_mean(variant, 10)), side)
    return (held, side)


def _scan_floor_capability_stalled() -> bool:
    """True when primary promotion_raw has plateaued (probe pool prioritization)."""
    try:
        from evolution.epi_momentum_governor import promotion_raw_capability_stalled
        from evolution.mutation_helpers import _evolution_project_root

        return promotion_raw_capability_stalled(_evolution_project_root())
    except ImportError:
        return False


def _scan_floor_probe_stall_mode(champ_words: int | None = None) -> bool:
    """True when scan-floor probes should use stalled head-inject (no rotation)."""
    if _scan_floor_capability_stalled():
        return True
    if champ_words is None or int(champ_words) <= 0:
        return False
    try:
        from evolution.epi_momentum_governor import scan_floor_length_prom_stalled
        from evolution.mutation_helpers import _evolution_project_root

        return scan_floor_length_prom_stalled(_evolution_project_root(), int(champ_words))
    except ImportError:
        return False


def _scan_floor_capability_rank_budget() -> int:
    """CPU promotion_raw rank budget for scan-floor probe ordering."""
    base = max(0, int(os.environ.get("EE_SCAN_FLOOR_CAPABILITY_RANK_BUDGET", "12")))
    if _scan_floor_capability_stalled():
        base = max(
            base,
            int(os.environ.get("EE_SCAN_FLOOR_CAPABILITY_RANK_BUDGET_STALLED", "24")),
        )
    return base


_CAPABILITY_SCORE_CACHE: dict[tuple[int, ...], tuple[tuple[int, ...], float]] = {}


def _scan_floor_probe_capability_score(champion: list[int], variant: list[int]) -> float:
    """Promotion fitness for scan-floor probe ordering (once-per-cycle budget)."""
    champ_key = tuple(int(x) for x in champion)
    var_key = tuple(int(x) for x in variant)
    cached = _CAPABILITY_SCORE_CACHE.get(var_key)
    if cached is not None and cached[0] == champ_key:
        return cached[1]
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    score = float(compact_l30_promotion_raw_fitness(list(variant)))
    if len(_CAPABILITY_SCORE_CACHE) > 256:
        _CAPABILITY_SCORE_CACHE.clear()
    _CAPABILITY_SCORE_CACHE[var_key] = (champ_key, score)
    return score


def _rank_scan_floor_picked_probes(
    champion: list[int],
    picked: list[list[int]],
    *,
    budget: int | None = None,
) -> list[list[int]]:
    """Re-order picked probes by promotion_raw_fitness on a capped CPU budget."""
    if not picked:
        return picked
    cap = int(budget if budget is not None else _scan_floor_capability_rank_budget())
    if cap <= 0:
        return picked
    pre = sorted(picked, key=_scan_floor_probe_rank_key, reverse=True)
    head = pre[:cap]
    l10_weak = float(curriculum_verify_mean(list(champion), 10)) < 0.85 - 1e-6
    scored: list[tuple[float, ...]] = []
    for variant in head:
        try:
            prom = float(_scan_floor_probe_capability_score(champion, variant))
        except (RuntimeError, ValueError, TypeError, OSError):
            prom = float(_scan_floor_probe_rank_key(variant)[0])
        if l10_weak:
            scored.append((prom, float(curriculum_verify_mean(list(variant), 10)), list(variant)))
        else:
            scored.append((prom, list(variant)))
    if l10_weak:
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    else:
        scored.sort(key=lambda item: item[0], reverse=True)
    ranked_head = [list(item[-1]) for item in scored]
    seen = {tuple(v) for v in ranked_head}
    tail = [list(v) for v in pre if tuple(v) not in seen]
    return ranked_head + tail


def terminal_chained_reorder_imm_probe_variants(
    program: list[int],
    rng,
    *,
    max_evals: int = 12,
    max_attempts: int | None = None,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Block reorder then immediate nudge; gate-qualified same-length probes."""
    from mutation_proposer import DirectedMutator

    if not program or len(program) < 6:
        return []
    champ_n = len(program)
    gate = gate or get_scan_floor_gate(program)
    attempt_cap = int(max_attempts) if max_attempts is not None else max(64, int(max_evals) * 4)
    seen: set[tuple[int, ...]] = {tuple(program)}
    out: list[list[int]] = []
    attempts = 0
    while len(out) < int(max_evals) and attempts < attempt_cap:
        attempts += 1
        variant, _ = DirectedMutator._reorder_block(list(program), rng)
        if len(variant) != champ_n:
            continue
        variant, _ = DirectedMutator.immediate_perturbation(variant, rng)
        if len(variant) != champ_n:
            continue
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(list(variant))
    rng.shuffle(out)
    return out


def terminal_chained_imm_perturb_probe_variants(
    program: list[int],
    rng,
    *,
    chain_steps: int = 2,
    max_evals: int = 12,
    max_attempts: int | None = None,
    gate: ScanFloorGateBaseline | None = None,
) -> list[list[int]]:
    """Multi-step immediate nudges that pass held-out and side curriculum gates."""
    from mutation_proposer import DirectedMutator

    if not program or int(chain_steps) < 2:
        return []
    champ_n = len(program)
    gate = gate or get_scan_floor_gate(program)
    attempt_cap = int(max_attempts) if max_attempts is not None else max(32, int(max_evals) * 4)
    seen: set[tuple[int, ...]] = {tuple(program)}
    out: list[list[int]] = []
    attempts = 0
    steps = int(chain_steps)
    while len(out) < int(max_evals) and attempts < attempt_cap:
        attempts += 1
        variant = list(program)
        for _ in range(steps):
            variant, _ = DirectedMutator.immediate_perturbation(variant, rng)
        if len(variant) != champ_n:
            continue
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if gate.qualifies(variant):
            out.append(list(variant))
    rng.shuffle(out)
    return out


def terminal_equal_length_refine_variants(
    program: list[int],
    level: int,
    rng,
    *,
    max_evals: int = 24,
) -> list[list[int]]:
    """Same-length probes when shrink is exhausted at saturated held-out."""
    from evolution.register_repair import incremental_repair_variant

    lvl = int(level)
    if lvl < 29 or not program:
        return []
    champ_n = len(program)
    raw: list[list[int]] = []
    gated = _l11_fib_gate_imm(program) == 13
    side_saturated = compact_curriculum_side_score(program) >= 0.99 - 1e-6
    at_floor = terminal_efficiency_at_scan_floor(len(program))
    capability_stalled = _scan_floor_probe_stall_mode(champ_n) if at_floor else False
    if not gated and not side_saturated:
        raw.extend(terminal_branch_flip_variants(program, rng, max_evals=max_evals))
    if not gated and not side_saturated:
        raw.extend(terminal_overlay_keep_length_variants(program, lvl, rng, max_evals=max(8, int(max_evals) // 2)))
    if not gated:
        for segment in ("incremental", "level_only"):
            variant = incremental_repair_variant(program, lvl, segment)
            if variant and len(variant) == champ_n and variant != program:
                raw.append(variant)
    if side_saturated or at_floor:
        gate = get_scan_floor_gate(program)
        budget = max(12, int(max_evals))
        if capability_stalled:
            ref_budget = max(8, budget // 6)
            for variant in terminal_branch_flip_variants(program, rng, max_evals=ref_budget):
                if len(variant) == champ_n and gate.qualifies(variant):
                    raw.append(variant)
            for variant in terminal_overlay_keep_length_variants(program, lvl, rng, max_evals=ref_budget):
                if gate.qualifies(variant):
                    raw.append(variant)
            for segment in ("incremental", "level_only"):
                variant = incremental_repair_variant(program, lvl, segment)
                if variant and len(variant) == champ_n and variant != program and gate.qualifies(variant):
                    raw.append(variant)
        eighth = max(2, budget // 8)
        imm_variants = terminal_imm_perturb_probe_variants(
            program,
            rng,
            max_evals=max(6, budget - 7 * eighth),
            gate=gate,
        )
        # Deterministic wide-imm at scan floor lives in select_terminal_probes_for_batch;
        # skip the rng copy here to avoid generating the same axis twice per batch.
        wide_imm_variants: list[list[int]] = []
        if not at_floor:
            wide_imm_variants = terminal_wide_imm_perturb_probe_variants(
                program,
                rng,
                max_evals=eighth,
                gate=gate,
            )
        swap_variants = terminal_instr_swap_probe_variants(
            program,
            rng,
            max_evals=eighth,
            gate=gate,
        )
        det_swap_variants: list[list[int]] = []
        tail_wide_variants: list[list[int]] = []
        dual_imm_variants: list[list[int]] = []
        locus_variants: list[list[int]] = []
        if capability_stalled:
            det_swap_variants = terminal_deterministic_instr_swap_probe_variants(
                program,
                max_evals=max(eighth, 12),
                start_offset=int(terminal_cycles_at_level()),
                gate=gate,
            )
            locus_variants = terminal_growth_material_locus_probe_variants(
                program,
                max_evals=max(
                    8,
                    int(os.environ.get("EE_SCAN_FLOOR_GROWTH_LOCUS_PROBE_BUDGET", "16")),
                ),
                start_offset=int(terminal_cycles_at_level()) + 3,
                gate=gate,
            )
            tail_min = scan_floor_compact_tail_min_position(champ_n)
            tail_wide_variants = terminal_tail_wide_imm_probe_variants(
                program,
                min_position=tail_min,
                max_evals=max(16, budget // 4),
                start_offset=int(terminal_cycles_at_level()) + 1,
                gate=gate,
            )
            dual_imm_variants = terminal_dual_imm_probe_variants(
                program,
                min_position=tail_min,
                max_evals=max(24, budget // 3),
                start_offset=int(terminal_cycles_at_level()) + 2,
                gate=gate,
            )
        chained_variants = terminal_chained_imm_perturb_probe_variants(
            program,
            rng,
            chain_steps=2,
            max_evals=eighth,
            gate=gate,
        )
        triple_chain_variants = terminal_chained_imm_perturb_probe_variants(
            program,
            rng,
            chain_steps=3,
            max_evals=eighth,
            gate=gate,
        )
        quad_chain_variants = terminal_chained_imm_perturb_probe_variants(
            program,
            rng,
            chain_steps=4,
            max_evals=eighth,
            gate=gate,
        )
        penta_chain_variants: list[list[int]] = []
        hexa_chain_variants: list[list[int]] = []
        if capability_stalled:
            deep = max(eighth, int(max_evals) // 6)
            penta_chain_variants = terminal_chained_imm_perturb_probe_variants(
                program,
                rng,
                chain_steps=5,
                max_evals=deep,
                gate=gate,
            )
            hexa_chain_variants = terminal_chained_imm_perturb_probe_variants(
                program,
                rng,
                chain_steps=6,
                max_evals=deep,
                gate=gate,
            )
        reorder_variants = terminal_block_reorder_probe_variants(
            program,
            rng,
            max_evals=eighth,
            gate=gate,
        )
        reorder_imm_variants = terminal_chained_reorder_imm_probe_variants(
            program,
            rng,
            max_evals=eighth,
            gate=gate,
        )
        cross_budget = max(eighth, int(max_evals) // 4) if capability_stalled else eighth
        parent_pool = (
            imm_variants
            + wide_imm_variants
            + swap_variants
            + chained_variants
            + triple_chain_variants
            + quad_chain_variants
            + penta_chain_variants
            + hexa_chain_variants
            + reorder_variants
            + reorder_imm_variants
        )
        raw.extend(imm_variants)
        raw.extend(wide_imm_variants)
        raw.extend(swap_variants)
        raw.extend(det_swap_variants)
        raw.extend(locus_variants)
        raw.extend(tail_wide_variants)
        raw.extend(dual_imm_variants)
        raw.extend(chained_variants)
        raw.extend(triple_chain_variants)
        raw.extend(quad_chain_variants)
        raw.extend(penta_chain_variants)
        raw.extend(hexa_chain_variants)
        raw.extend(reorder_variants)
        raw.extend(reorder_imm_variants)
        raw.extend(
            terminal_crossover_probe_variants(
                program,
                parent_pool,
                rng,
                max_evals=cross_budget,
                gate=gate,
            )
        )
    seen: set[tuple[int, ...]] = {tuple(program)}
    out: list[list[int]] = []
    for variant in raw:
        if len(variant) != champ_n:
            continue
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(variant))
    if capability_stalled:
        out = _rank_scan_floor_picked_probes(
            program,
            out,
            budget=max(int(max_evals), _scan_floor_capability_rank_budget()),
        )
    elif side_saturated or at_floor:
        out.sort(key=_scan_floor_probe_rank_key, reverse=True)
    else:
        rng.shuffle(out)
    return out[: max(1, int(max_evals))]


def terminal_verify_probe_variants(
    program: list[int],
    level: int,
    rng,
    count: int,
    *,
    score_fn=None,
    min_verify: float | None = None,
    fitness_fn=None,
    min_fitness: float | None = None,
) -> list[list[int]]:
    """Build champion+reference challengers to escape max-depth verify plateaus."""
    from evolution.register_repair import incremental_repair_variant

    lvl = int(level)
    if lvl < 29 or count <= 0 or not program:
        return []
    at_floor = terminal_efficiency_at_scan_floor(len(program))
    raw: list[list[int]] = []
    if at_floor:
        raw.extend(terminal_equal_length_refine_variants(program, lvl, rng, max_evals=32))
    else:
        raw.extend(straight_line_reference_variants(program, lvl, rng, max_evals=40))
    gated = _l11_fib_gate_imm(program) == 13
    if not gated and not at_floor:
        # Inverting L9–L11 skips on a complete champion creates leftover
        # spinners; mutation still performs branch swaps in other slots.
        raw.extend(terminal_branch_flip_variants(program, rng, max_evals=24))
    if (
        min_verify is not None
        and float(min_verify) >= _VERIFY_SATURATED
        and not terminal_efficiency_at_scan_floor(len(program))
    ):
        raw.extend(
            terminal_efficiency_variants(
                program,
                rng,
                max_evals=terminal_efficiency_eval_budget(),
            )
        )
    if not gated and not at_floor:
        for segment in ("incremental", "level_only"):
            variant = incremental_repair_variant(program, lvl, segment)
            if variant and variant != program:
                raw.append(variant)
    candidates = _filter_probe_candidates(
        raw,
        score_fn=score_fn,
        min_verify=min_verify,
        fitness_fn=fitness_fn,
        min_fitness=min_fitness,
    )
    if not candidates:
        return []
    _sort_probe_candidates(candidates, rng, score_fn=score_fn, fitness_fn=fitness_fn, min_verify=min_verify)
    out: list[list[int]] = []
    while len(out) < count:
        out.append(list(candidates[len(out) % len(candidates)]))
    return out


def l30_easy_fill_variants(program: list[int]) -> list[list[int]]:
    """Insert cheap L0-L6 / L30-sum writes the 13w champion is missing."""
    if not program or len(program) < 4:
        return []
    cached = _l30_easy_fill_tuple_cached(tuple(int(x) for x in program))
    return [list(item) for item in cached]


def ranked_scan_floor_growth_fills(program: list[int]) -> list[list[int]]:
    """Held-out-safe growth easy_fills for scan-floor injection (disk + in-proc cache)."""
    if not program:
        return []
    key = tuple(int(x) for x in program)
    disk = _load_scan_floor_growth_rank_disk(key)
    if disk is not None:
        return [list(item) for item in disk]
    ranked = _ranked_scan_floor_growth_fills_cached(key)
    _save_scan_floor_growth_rank_disk(key, ranked)
    return [list(item) for item in ranked]


def ranked_scan_floor_compression_variants(
    program: list[int],
    *,
    cycle_no: int | None = None,
) -> list[list[int]]:
    """Held-out-safe shorter compression candidates for scan-floor injection."""
    if not program:
        return []
    key = tuple(int(x) for x in program)
    champ_words = len(program)
    at_floor = terminal_efficiency_at_scan_floor(champ_words)
    if cycle_no is not None and at_floor:
        if not should_rescan_terminal_efficiency_compression(int(cycle_no), champ_words, list(program)):
            disk = _load_scan_floor_compression_rank_disk(key)
            if disk is not None:
                return [list(item) for item in disk]
            return []
        if _skip_empty_compression_rescan(key, champ_words):
            return []
        ranked = _ranked_scan_floor_compression_variants_cached(key)
        _record_scan_floor_compression_rescan(key, cycle=int(cycle_no), hits=len(ranked))
        if ranked:
            _save_scan_floor_compression_rank_disk(key, ranked)
        return [list(item) for item in ranked]
    disk = _load_scan_floor_compression_rank_disk(key)
    if disk is not None:
        return [list(item) for item in disk]
    ranked = _ranked_scan_floor_compression_variants_cached(key)
    if ranked:
        _save_scan_floor_compression_rank_disk(key, ranked)
    return [list(item) for item in ranked]


def _scan_floor_growth_rank_disk_path() -> str:
    return os.path.join(_project_agents_dir(), "scan_floor_growth_rank_v2.json")


def _scan_floor_compression_rank_disk_path() -> str:
    return os.path.join(_project_agents_dir(), "scan_floor_compression_rank.json")


def _scan_floor_compression_miss_path() -> str:
    return os.path.join(_project_agents_dir(), "scan_floor_compression_miss.json")


def _compression_rank_disk_key(program: tuple[int, ...]) -> str:
    return ",".join(f"{w:08X}" for w in program)


def _load_scan_floor_compression_miss() -> dict[str, Any]:
    path = _scan_floor_compression_miss_path()
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _record_scan_floor_compression_rescan(
    program: tuple[int, ...],
    *,
    cycle: int,
    hits: int,
) -> None:
    path = _scan_floor_compression_miss_path()
    key = _compression_rank_disk_key(program)
    payload = _load_scan_floor_compression_miss()
    payload[key] = {
        "cycle": int(cycle),
        "hits": int(hits),
        "words": len(program),
    }
    if len(payload) > 24:
        for stale in list(payload.keys())[:-20]:
            if stale != key:
                payload.pop(stale, None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _skip_empty_compression_rescan(
    program: tuple[int, ...],
    champ_words: int,
) -> bool:
    """Skip redundant exhaustive sweeps when this floor already scored zero hits.

    Eq-refine ties rotate bytes at the same prom_raw; a zero-hit sweep at 40w
    applies to all same-length plateau champions until a promote moves the bar.
    """
    if not _scan_floor_probe_stall_mode(int(champ_words)):
        return False
    if not terminal_efficiency_at_scan_floor(int(champ_words)):
        return False
    payload = _load_scan_floor_compression_miss()
    key = _compression_rank_disk_key(program)
    entry = payload.get(key)
    if isinstance(entry, dict):
        if int(entry.get("hits") or 0) > 0:
            return False
        if int(entry.get("words") or 0) == len(program):
            return True
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        if int(item.get("hits") or 0) > 0:
            continue
        if int(item.get("words") or 0) == len(program):
            return True
    return False


def _load_scan_floor_growth_rank_disk(
    program: tuple[int, ...],
) -> tuple[tuple[int, ...], ...] | None:
    path = _scan_floor_growth_rank_disk_path()
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    key = ",".join(f"{w:08X}" for w in program)
    entry = payload.get(key)
    if not isinstance(entry, list) or not entry:
        return None
    out: list[tuple[int, ...]] = []
    for item in entry:
        if not isinstance(item, list):
            return None
        out.append(tuple(int(x) for x in item))
    return tuple(out)


def _save_scan_floor_growth_rank_disk(
    program: tuple[int, ...],
    ranked: tuple[tuple[int, ...], ...],
) -> None:
    if not ranked:
        return
    path = _scan_floor_growth_rank_disk_path()
    key = ",".join(f"{w:08X}" for w in program)
    payload: dict[str, list[list[int]]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            payload = {k: v for k, v in existing.items() if isinstance(v, list)}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    payload[key] = [list(item) for item in ranked]
    if len(payload) > 16:
        for stale in list(payload.keys())[:-15]:
            if stale != key:
                payload.pop(stale, None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _load_scan_floor_compression_rank_disk(
    program: tuple[int, ...],
) -> tuple[tuple[int, ...], ...] | None:
    path = _scan_floor_compression_rank_disk_path()
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    key = ",".join(f"{w:08X}" for w in program)
    entry = payload.get(key)
    if not isinstance(entry, list):
        return None
    if not entry:
        return None
    out: list[tuple[int, ...]] = []
    for item in entry:
        if not isinstance(item, list):
            return None
        out.append(tuple(int(x) for x in item))
    return tuple(out)


def _save_scan_floor_compression_rank_disk(
    program: tuple[int, ...],
    ranked: tuple[tuple[int, ...], ...],
) -> None:
    if not ranked:
        return
    path = _scan_floor_compression_rank_disk_path()
    key = ",".join(f"{w:08X}" for w in program)
    payload: dict[str, list[list[int]]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            payload = {k: v for k, v in existing.items() if isinstance(v, list)}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    payload[key] = [list(item) for item in ranked]
    if len(payload) > 16:
        for stale in list(payload.keys())[:-15]:
            if stale != key:
                payload.pop(stale, None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


@functools.lru_cache(maxsize=32)
def _ranked_scan_floor_compression_variants_cached(
    program: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    import random

    from evolution.champion_bloat_guard import level_word_cap
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.mutation_helpers import scan_floor_plateau_armed

    prog = list(program)
    champ_n = len(prog)
    word_cap = level_word_cap(30)
    span = min(3, terminal_efficiency_drop_span(champ_n, word_cap))
    armed, floor_w = scan_floor_plateau_armed()
    if armed and champ_n < int(floor_w):
        below = int(floor_w) - champ_n
        span = max(span, min(below, 4))
        budget = max(
            128,
            int(os.environ.get("EE_SCAN_FLOOR_COMPRESS_BELOW_FLOOR_BUDGET", "384")),
        )
        seed_sweeps = max(
            32,
            int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_BELOW_FLOOR_SWEEPS", "48")),
        )
    elif armed and champ_n == int(floor_w):
        # At-floor cascade must allow multi-word drops (e.g. 59w -> 56w).
        span = max(span, 3)
        budget = max(
            256,
            int(os.environ.get("EE_SCAN_FLOOR_COMPRESS_AT_FLOOR_BUDGET", "512")),
        )
        seed_sweeps = max(
            48,
            int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_AT_FLOOR_SWEEPS", "64")),
        )
    else:
        budget = max(
            64,
            int(os.environ.get("EE_SCAN_FLOOR_COMPRESS_INJECT_BUDGET", "256")),
        )
        seed_sweeps = max(
            1,
            int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_SEED_SWEEPS", "32")),
        )
    prog_key = struct.pack("<" + "I" * len(prog), *prog)
    base_seed = int(zlib.adler32(prog_key + b"compression-rank-v9")) & 0x7FFFFFFF
    per_seed = max(8, budget // seed_sweeps)
    variants: list[list[int]] = []
    seen_keys: set[tuple[int, ...]] = set()
    anchor_evals = max(64, int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_ANCHOR_EVALS", "128")))
    for anchor in (42, 7, 99, 0):
        anchor_rng = random.Random(int(anchor))
        for variant in terminal_efficiency_variants(prog, anchor_rng, max_evals=anchor_evals, max_drop_span=span):
            key = tuple(int(word) for word in variant)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            variants.append(list(variant))
    for sweep in range(seed_sweeps):
        sweep_seed = (base_seed + sweep * 10007) & 0x7FFFFFFF
        rng = random.Random(sweep_seed)
        for variant in terminal_efficiency_variants(prog, rng, max_evals=per_seed, max_drop_span=span):
            key = tuple(int(word) for word in variant)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            variants.append(list(variant))
    champ_h = float(heldout_l30_verify_mean(prog, 32))
    champ_side = float(compact_curriculum_side_score(prog))
    champ_raw = float(compact_l30_promotion_raw_fitness(prog))
    near_miss: list[list[int]] = []
    for variant in variants:
        if len(variant) >= champ_n:
            continue
        if float(heldout_l30_verify_mean(list(variant), 32)) + 1e-6 < champ_h:
            continue
        if float(compact_l30_promotion_raw_fitness(list(variant))) + 1e-6 < champ_raw:
            continue
        if float(compact_curriculum_side_score(list(variant))) + 1e-6 >= champ_side:
            continue
        near_miss.append(list(variant))
    side_match: list[list[int]] = [
        list(variant)
        for variant in variants
        if len(variant) < champ_n
        and float(compact_l30_promotion_raw_fitness(list(variant))) + 1e-6 >= champ_raw
        and float(compact_curriculum_side_score(list(variant))) + 1e-6 >= champ_side
    ]
    if side_match:
        side_match.sort(
            key=lambda item: (
                len(item),
                -float(compact_l30_promotion_raw_fitness(list(item))),
            ),
        )
    repair_bases: list[list[int]] = []
    if near_miss:
        min_gain = float(scan_floor_min_raw_gain(champ_n))
        prom_beating = [
            list(item)
            for item in near_miss
            if float(compact_l30_promotion_raw_fitness(list(item))) > champ_raw + min_gain - 1e-6
        ]
        prom_beating.sort(
            key=lambda item: float(compact_l30_promotion_raw_fitness(list(item))),
            reverse=True,
        )
        near_miss.sort(
            key=lambda item: (
                float(compact_l30_promotion_raw_fitness(list(item))),
                float(compact_curriculum_side_score(list(item))),
            ),
            reverse=True,
        )
        repair_cap = max(4, int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_REPAIR_BASES", "8")))
        repair_bases.extend(prom_beating[:repair_cap])
        for item in near_miss:
            if len(repair_bases) >= repair_cap:
                break
            if tuple(item) in {tuple(base) for base in repair_bases}:
                continue
            repair_bases.append(list(item))
    repair_base_cap = max(8, int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_REPAIR_BASES", "8")))
    for base in side_match[:repair_base_cap]:
        if tuple(base) not in {tuple(item) for item in repair_bases}:
            repair_bases.append(base)
    if repair_bases:
        repair_budget = max(
            32,
            int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_SIDE_REPAIR", "96")),
        )
        if _scan_floor_probe_stall_mode(champ_n):
            repair_budget = max(
                repair_budget,
                int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_SIDE_REPAIR_STALLED", "192")),
            )
        repair_offsets = (
            (0, 32, 64, 96, 128, 160)
            if armed and champ_n <= int(floor_w) and _scan_floor_probe_stall_mode(champ_n)
            else ((0, 32, 64, 96) if armed and champ_n <= int(floor_w) else (0,))
        )
        for repair_offset in repair_offsets:
            for repaired in terminal_compression_side_repair_variants(
                prog,
                repair_bases[:repair_base_cap],
                max_evals=repair_budget,
                start_offset=repair_offset,
            ):
                key = tuple(int(word) for word in repaired)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                variants.append(list(repaired))
    qualified = [list(variant) for variant in variants if scan_floor_compression_promote_qualifies(prog, variant)]
    if qualified:
        cascade_bases = sorted(
            qualified,
            key=lambda item: (
                len(item),
                -float(compact_l30_promotion_raw_fitness(list(item))),
            ),
        )[: max(2, int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_CASCADE_BASES", "4")))]
        cascade_budget = max(
            24,
            int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_CASCADE_BUDGET", "64")),
        )
        for base_idx, base in enumerate(cascade_bases):
            if len(base) >= champ_n:
                continue
            cascade_seed = (base_seed + (base_idx + 1) * 20011) & 0x7FFFFFFF
            cascade_rng = random.Random(cascade_seed)
            for variant in terminal_efficiency_variants(
                list(base),
                cascade_rng,
                max_evals=cascade_budget,
                max_drop_span=span,
            ):
                key = tuple(int(word) for word in variant)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                variants.append(list(variant))
        qualified = [list(variant) for variant in variants if scan_floor_compression_promote_qualifies(prog, variant)]
    if not qualified:
        return ()
    rank_budget = max(
        0,
        int(os.environ.get("EE_SCAN_FLOOR_COMPRESSION_RANK_BUDGET", "8")),
    )
    scored: list[tuple[float, int, list[int]]] = []
    for variant in qualified[: max(len(qualified), rank_budget)]:
        prom_raw = float(compact_l30_promotion_raw_fitness(list(variant)))
        if prom_raw <= 0.0:
            continue
        scored.append((prom_raw, -len(variant), variant))
    if not scored:
        return tuple(tuple(item) for item in qualified)
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return tuple(tuple(item[2]) for item in scored)


def terminal_compression_side_repair_variants(
    champion: list[int],
    bases: list[list[int]],
    *,
    max_evals: int = 64,
    start_offset: int = 0,
) -> list[list[int]]:
    """Wide-immediate repair on shorter high-raw programs that miss side-curriculum floor."""
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from milestone1.rv32i_mask import decode as rv32i_decode
    from milestone1.rv32i_mask import encode as rv32i_encode

    if not champion or not bases:
        return []
    champ_h = float(heldout_l30_verify_mean(list(champion), 32))
    champ_side = float(compact_curriculum_side_score(list(champion)))
    champ_raw = float(compact_l30_promotion_raw_fitness(list(champion)))
    deltas = _SCAN_FLOOR_WIDE_IMM_DELTAS
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = {tuple(int(word) for word in champion)}
    offset = int(start_offset)
    for base in bases:
        if not base:
            continue
        branch_candidates, other_candidates = _wide_imm_probe_indices(list(base))
        slots = branch_candidates + other_candidates
        if not slots:
            continue
        total_pairs = len(slots) * len(deltas)
        for step in range(total_pairs):
            if len(out) >= int(max_evals):
                return out
            pair_idx = (offset + step) % total_pairs
            idx = slots[pair_idx // len(deltas)]
            delta = deltas[pair_idx % len(deltas)]
            try:
                decoded = rv32i_decode(int(base[idx]), validate=False)
                if "imm" not in decoded:
                    continue
                new_word = rv32i_encode(
                    name=str(decoded["name"]),
                    rd=int(decoded.get("rd", 0)),
                    rs1=int(decoded.get("rs1", 0)),
                    rs2=int(decoded.get("rs2", 0)),
                    imm=int(decoded.get("imm", 0)) + int(delta),
                    validate=True,
                )
            except (ValueError, TypeError, KeyError):
                continue
            variant = list(base)
            variant[idx] = int(new_word)
            key = tuple(variant)
            if key in seen:
                continue
            seen.add(key)
            if float(heldout_l30_verify_mean(variant, 32)) + 1e-6 < champ_h:
                continue
            if float(compact_l30_promotion_raw_fitness(variant)) + 1e-6 < champ_raw:
                continue
            if float(compact_curriculum_side_score(variant)) + 1e-6 < champ_side:
                continue
            out.append(variant)
        offset += total_pairs
    return out


def terminal_growth_tail_perturb_variants(
    champion: list[int],
    growth_anchor: list[int],
    rng,
    *,
    max_evals: int = 8,
    start_offset: int = 0,
) -> list[list[int]]:
    """Wide-immediate sweeps on growth near-miss tail (indices after LCP with champion)."""
    from evolution.mutation_helpers import growth_anchor_lcp_prefix, scan_floor_growth_heldout_qualifies
    from milestone1.rv32i_mask import decode as rv32i_decode
    from milestone1.rv32i_mask import encode as rv32i_encode

    if not champion or not growth_anchor or len(growth_anchor) <= len(champion):
        return []
    base = list(growth_anchor)
    lcp = growth_anchor_lcp_prefix(list(champion), base)
    branch_candidates, other_candidates = _wide_imm_probe_indices(base)
    tail_slots = [i for i in branch_candidates + other_candidates if i >= lcp]
    if not tail_slots:
        tail_slots = [i for i in range(lcp, len(base) - 1)]
    if not tail_slots:
        return []
    deltas = _SCAN_FLOOR_WIDE_IMM_DELTAS
    total_pairs = len(tail_slots) * len(deltas)
    offset = int(start_offset) % max(1, total_pairs)
    seen: set[tuple[int, ...]] = {tuple(champion), tuple(base)}
    out: list[list[int]] = []
    for step in range(total_pairs):
        if len(out) >= int(max_evals):
            break
        pair_idx = (offset + step) % total_pairs
        idx = tail_slots[pair_idx // len(deltas)]
        delta = deltas[pair_idx % len(deltas)]
        try:
            decoded = rv32i_decode(base[idx], validate=False)
            if "imm" not in decoded:
                continue
            new_word = rv32i_encode(
                name=str(decoded["name"]),
                rd=int(decoded.get("rd", 0)),
                rs1=int(decoded.get("rs1", 0)),
                rs2=int(decoded.get("rs2", 0)),
                imm=int(decoded.get("imm", 0)) + int(delta),
                validate=True,
            )
        except (ValueError, TypeError, KeyError):
            continue
        variant = list(base)
        variant[idx] = int(new_word)
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        if scan_floor_growth_heldout_qualifies(list(champion), variant):
            out.append(variant)
    rng.shuffle(out)
    return out


@functools.lru_cache(maxsize=32)
def _ranked_scan_floor_growth_fills_cached(
    program: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    import os

    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
    from evolution.mutation_helpers import scan_floor_growth_completion_qualifies

    champ = list(program)
    champ_n = len(champ)
    gate = get_scan_floor_gate(champ)
    champ_l10_weak = float(curriculum_verify_mean(champ, 10)) < 0.85 - 1e-6
    heldout_seeds = max(4, int(os.environ.get("EE_SCAN_FLOOR_GROWTH_HELDOUT_SEEDS", "8")))
    champ_heldout = float(heldout_l30_verify_mean(champ, heldout_seeds))
    candidates: list[list[int]] = []
    for fill in l30_easy_fill_variants(champ):
        if len(fill) <= champ_n:
            continue
        if heldout_l30_verify_mean(fill, heldout_seeds) + 0.01 < champ_heldout:
            continue
        if not scan_floor_growth_completion_qualifies(champ, list(fill), gate=gate):
            continue
        candidates.append(list(fill))
    try:
        from evolution.champion_rollback import collect_jsonl_long_archive_parents
        from evolution.mutation_helpers import _evolution_project_root

        results = os.path.join(_evolution_project_root(), "multi_cycle_results.jsonl")
        rank_seed = max(4, int(os.environ.get("EE_SCAN_FLOOR_GROWTH_RANK_BUDGET", "8")))
        long_parents = collect_jsonl_long_archive_parents(
            results,
            list(program),
            30,
            max_parents=max(8, rank_seed),
            word_tolerance=max(
                6,
                int(os.environ.get("EE_SCAN_FLOOR_ARCHIVE_WORD_TOL_EXHAUSTED", "12")),
            ),
        )
        seen_cand = {tuple(c) for c in candidates}
        for parent in long_parents:
            pt = tuple(int(x) for x in parent)
            if pt in seen_cand or len(parent) <= champ_n:
                continue
            if heldout_l30_verify_mean(list(parent), heldout_seeds) + 0.01 < champ_heldout:
                continue
            if not scan_floor_growth_completion_qualifies(champ, list(parent), gate=gate):
                continue
            candidates.append(list(parent))
            seen_cand.add(pt)
    except (ImportError, OSError, TypeError, ValueError):
        pass
    if not candidates:
        return ()
    candidates.sort(key=len, reverse=True)
    rank_budget = max(0, int(os.environ.get("EE_SCAN_FLOOR_GROWTH_RANK_BUDGET", "8")))
    if rank_budget <= 0:
        return tuple(tuple(item) for item in candidates)
    scored: list[tuple[float, ...]] = []
    for fill in candidates[:rank_budget]:
        prom_raw = float(compact_l30_promotion_raw_fitness(list(fill)))
        if prom_raw <= 0.0:
            continue
        if champ_l10_weak:
            scored.append(
                (
                    prom_raw,
                    float(curriculum_verify_mean(list(fill), 10)),
                    fill,
                )
            )
        else:
            scored.append((prom_raw, fill))
    if scored:
        if champ_l10_weak:
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        else:
            scored.sort(key=lambda item: item[0], reverse=True)
        return tuple(tuple(item[-1]) for item in scored)
    return tuple(tuple(item) for item in candidates)


@functools.lru_cache(maxsize=64)
def _l30_easy_fill_tuple_cached(program: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(item) for item in _l30_easy_fill_variants_impl(list(program)))


def _l30_easy_fill_variants_impl(program: list[int]) -> list[list[int]]:
    if not program or len(program) < 4:
        return []
    from mutation_proposer import DirectedMutator, make_rv32i

    fills = [
        make_rv32i("ADD", rd=4, rs1=1, rs2=1),
        make_rv32i("ADDI", rd=5, rs1=2, imm=0),
        make_rv32i("ADD", rd=6, rs1=2, rs2=2),
        make_rv32i("ADD", rd=7, rs1=4, rs2=6),
        make_rv32i("ADD", rd=20, rs1=1, rs2=2),
        make_rv32i("SLLI", rd=28, rs1=9, imm=1),
        make_rv32i("ADD", rd=30, rs1=1, rs2=2),
        make_rv32i("ADDI", rd=31, rs1=23, imm=0),
    ]
    out: list[list[int]] = []
    bundled = DirectedMutator.insert_seq_before_ebreak(list(program), fills)
    if 12 <= len(bundled) <= L30_COMPACT_MAX_WORDS and bundled != list(program):
        out.append(bundled)
    extras = [
        make_rv32i("ADDI", rd=21, rs1=23, imm=0),
        make_rv32i("ADD", rd=26, rs1=1, rs2=2),
    ]
    extra_bundled = DirectedMutator.insert_seq_before_ebreak(list(program), extras)
    if 12 <= len(extra_bundled) <= L30_COMPACT_MAX_WORDS and extra_bundled != list(program):
        out.append(extra_bundled)
    try:
        base = semantic_compress(list(program)) or list(program)
    except (AttributeError, TypeError, ValueError, KeyError):
        base = list(program)
    gaps = [
        make_rv32i("ADD", rd=22, rs1=1, rs2=2),
        make_rv32i("SUB", rd=29, rs1=2, rs2=1),
        make_rv32i("SUB", rd=25, rs1=1, rs2=2),
        make_rv32i("SUB", rd=27, rs1=1, rs2=2),
    ]
    for take in (4, 3, 2, 1):
        gap_bundled = DirectedMutator.insert_seq_before_ebreak(list(base), gaps[:take])
        if 12 <= len(gap_bundled) <= L30_COMPACT_MAX_WORDS and gap_bundled != list(program) and gap_bundled not in out:
            out.append(gap_bundled)
            break
    x24 = [
        make_rv32i("SLLI", rd=24, rs1=23, imm=2),
        make_rv32i("XOR", rd=24, rs1=24, rs2=1),
    ]
    x24_bundled = DirectedMutator.insert_seq_before_ebreak(list(base), x24)
    if 12 <= len(x24_bundled) <= L30_COMPACT_MAX_WORDS and x24_bundled != list(program) and x24_bundled not in out:
        out.append(x24_bundled)
    full = DirectedMutator.insert_seq_before_ebreak(list(base), gaps + x24)
    if 12 <= len(full) <= L30_COMPACT_MAX_WORDS and full != list(program) and full not in out:
        out.append(full)
    ordered = l30_order_sensitive_variant(program)
    if ordered and 12 <= len(ordered) <= L30_COMPACT_MAX_WORDS and ordered != list(program) and ordered not in out:
        out.append(ordered)
    stripped = l30_strip_dead_l30_stores(ordered or program)
    if stripped and 12 <= len(stripped) <= L30_COMPACT_MAX_WORDS and stripped != list(program) and stripped not in out:
        out.append(stripped)
    l10 = l30_l10_prefix_load_variant(stripped or ordered or program)
    if l10 and 12 <= len(l10) <= L30_COMPACT_MAX_WORDS and l10 != list(program) and l10 not in out:
        out.append(l10)
    l8 = l30_l8_mem_add_variant(l10 or stripped or ordered or program)
    if l8 and 12 <= len(l8) <= L30_COMPACT_MAX_WORDS and l8 != list(program) and l8 not in out:
        out.append(l8)
    for base in (l8, l10, stripped, ordered, program):
        if not base:
            continue
        bounded = l30_l10_bounded_sum_variant(base)
        if bounded and 12 <= len(bounded) <= L30_COMPACT_MAX_WORDS and bounded != list(program) and bounded not in out:
            out.append(bounded)
        maxed = l30_l10_max_in_sum_loop_variant(bounded or base)
        if maxed and 12 <= len(maxed) <= L30_COMPACT_MAX_WORDS and maxed != list(program) and maxed not in out:
            out.append(maxed)
        preloop = l30_l10_sum_max_preloop_insert_variant(
            maxed or bounded or l8 or l10 or stripped or ordered or program
        )
        if preloop and 12 <= len(preloop) <= L30_COMPACT_MAX_WORDS and preloop != list(program) and preloop not in out:
            out.append(preloop)
        fib = l30_l11_gated_fib_variant(preloop or maxed or bounded or base)
        if fib and 12 <= len(fib) <= L30_COMPACT_MAX_WORDS and fib != list(program) and fib not in out:
            out.append(fib)
    fib_live = l30_l11_gated_fib_variant(program)
    if fib_live and 12 <= len(fib_live) <= L30_COMPACT_MAX_WORDS and fib_live != list(program) and fib_live not in out:
        out.append(fib_live)
    tightened = l30_l11_fib_gate_tighten_variant(program)
    if (
        tightened
        and 12 <= len(tightened) <= L30_COMPACT_MAX_WORDS
        and tightened != list(program)
        and tightened not in out
    ):
        out.append(tightened)
    mid = l30_l12_l16_postfix_variant(fib_live or program)
    if mid and 12 <= len(mid) <= L30_COMPACT_MAX_WORDS and mid != list(program) and mid not in out:
        out.append(mid)
    short15 = l30_l15_short_variant(program)
    if short15 and 12 <= len(short15) <= L30_COMPACT_MAX_WORDS and short15 != list(program) and short15 not in out:
        out.append(short15)
    folded = l30_l12_l14_x9_fold_variant(program)
    if folded and 12 <= len(folded) <= L30_COMPACT_MAX_WORDS and folded != list(program) and folded not in out:
        out.append(folded)
    dropped = l30_drop_loop_zero_inits(program)
    if dropped and 12 <= len(dropped) <= L30_COMPACT_MAX_WORDS and dropped != list(program) and dropped not in out:
        out.append(dropped)
    return out


def best_compact_l30_fill(program: list[int]) -> list[int] | None:
    """Highest CPU L30, then L8-L10 side score, among compact fills that do not drop L30."""
    if not program:
        return None
    fills = [fill for fill in l30_easy_fill_variants(list(program)) if 12 <= len(fill) <= L30_COMPACT_MAX_WORDS]
    if not fills:
        return None
    champ_h = heldout_l30_verify_mean(list(program))
    champ_s = compact_curriculum_side_score(list(program))
    best: list[int] | None = None
    best_key: tuple[float, float, int] | None = None
    for fill in fills:
        held = heldout_l30_verify_mean(fill)
        if held + 1e-6 < champ_h:
            continue
        side = compact_curriculum_side_score(fill)
        gate = _l11_fib_gate_imm(fill)
        key = (held, side, -len(fill), 0 if gate is None else -int(gate))
        if best_key is None or key > best_key:
            best = fill
            best_key = key
    if best is None or best_key is None:
        return None
    held, side, _neg_len, _gate = best_key
    fib_tighter = (
        _l11_fib_gate_imm(best) == 13
        and _l11_fib_gate_imm(program) == 16
        and held + 1e-6 >= champ_h
        and side + 1e-6 >= champ_s
    )
    if held >= champ_h + 0.01 or compact_curriculum_beats(program, best) or fib_tighter:
        return best
    return None


@functools.lru_cache(maxsize=64)
def cached_best_compact_l30_fill(program: tuple[int, ...]) -> tuple[int, ...] | None:
    """Memoized best_compact_l30_fill for a frozen program."""
    fill = best_compact_l30_fill(list(program))
    return tuple(fill) if fill else None


def l30_strip_dead_l30_stores(program: list[int]) -> list[int] | None:
    """Drop SW and the unused LUI-style ADDI x24,x0 that L30 verify ignores."""
    if not program or len(program) < 12:
        return None
    from mutation_proposer import rv32i_decode

    out: list[int] = []
    for word in program:
        try:
            decoded = rv32i_decode(word)
        except (TypeError, ValueError, KeyError):
            out.append(word)
            continue
        if not decoded:
            out.append(word)
            continue
        name = decoded.get("name")
        if name == "SW":
            continue
        if name == "ADDI" and int(decoded.get("rd", -1)) == 24 and int(decoded.get("rs1", -1)) == 0:
            continue
        out.append(word)
    if not (12 <= len(out) < len(program) <= L30_COMPACT_MAX_WORDS):
        return None
    return out


def l30_l10_prefix_load_variant(program: list[int]) -> list[int] | None:
    """LW x10, 0(x1) — L10's first mem cell. Does not change L30's register list."""
    if not program or len(program) < 12:
        return None
    from mutation_proposer import DirectedMutator, make_rv32i

    load = make_rv32i("LW", rd=10, rs1=1, imm=0)
    if load in program:
        return None
    bundled = DirectedMutator.insert_seq_before_ebreak(list(program), [load])
    if not (12 <= len(bundled) <= L30_COMPACT_MAX_WORDS) or bundled == list(program):
        return None
    return bundled


def l30_l8_mem_add_variant(program: list[int]) -> list[int] | None:
    """LW x15,0(x2); ADD x11,x10,x15 for L8 x11=mem[x1]+mem[x2]. L30 ignores x11/x15."""
    if not program or len(program) < 12:
        return None
    from mutation_proposer import DirectedMutator, make_rv32i

    seq = [
        make_rv32i("LW", rd=15, rs1=2, imm=0),
        make_rv32i("ADD", rd=11, rs1=10, rs2=15),
    ]
    if seq[1] in program:
        return None
    bundled = DirectedMutator.insert_seq_before_ebreak(list(program), seq)
    if not (12 <= len(bundled) <= L30_COMPACT_MAX_WORDS) or bundled == list(program):
        return None
    return bundled


def _l30_bounded_sum_loop_seq() -> list[int]:
    from mutation_proposer import make_rv32i

    return [
        make_rv32i("ADDI", rd=12, rs1=0, imm=0),
        make_rv32i("ADDI", rd=16, rs1=0, imm=0),
        make_rv32i("SLTIU", rd=19, rs1=2, imm=16),
        make_rv32i("BEQ", rs1=19, rs2=0, imm=24),
        make_rv32i("SLLI", rd=17, rs1=16, imm=2),
        make_rv32i("ADD", rd=17, rs1=1, rs2=17),
        make_rv32i("LW", rd=18, rs1=17, imm=0),
        make_rv32i("ADD", rd=12, rs1=12, rs2=18),
        make_rv32i("ADDI", rd=16, rs1=16, imm=1),
        make_rv32i("BLT", rs1=16, rs2=2, imm=-24),
    ]


def _l30_bounded_sum_max_loop_body() -> list[int]:
    """L9 sum + L10 max. x12/x13/x16 must be 0 on entry (fresh regs or prior zeros)."""
    from mutation_proposer import make_rv32i

    return [
        make_rv32i("SLTIU", rd=19, rs1=2, imm=16),
        make_rv32i("BEQ", rs1=19, rs2=0, imm=40),
        make_rv32i("SLLI", rd=17, rs1=16, imm=2),
        make_rv32i("ADD", rd=17, rs1=1, rs2=17),
        make_rv32i("LW", rd=18, rs1=17, imm=0),
        make_rv32i("ADD", rd=12, rs1=12, rs2=18),
        make_rv32i("SLTU", rd=19, rs1=13, rs2=18),
        make_rv32i("BEQ", rs1=19, rs2=0, imm=8),
        make_rv32i("ADDI", rd=13, rs1=18, imm=0),
        make_rv32i("ADDI", rd=16, rs1=16, imm=1),
        make_rv32i("BLT", rs1=16, rs2=2, imm=-32),
    ]


def _l30_bounded_sum_max_loop_seq() -> list[int]:
    """Bounded sum+max. Skip lands after BLT (imm=40). No zero inits; x12/x13/x16 start at 0."""
    return _l30_bounded_sum_max_loop_body()


def _l30_max_loop_with_zeros() -> list[int]:
    from mutation_proposer import make_rv32i

    return [
        make_rv32i("ADDI", rd=12, rs1=0, imm=0),
        make_rv32i("ADDI", rd=13, rs1=0, imm=0),
        make_rv32i("ADDI", rd=16, rs1=0, imm=0),
        *_l30_bounded_sum_max_loop_body(),
    ]


def _l30_max_loop_x12_x13_zeros() -> list[int]:
    from mutation_proposer import make_rv32i

    return [
        make_rv32i("ADDI", rd=12, rs1=0, imm=0),
        make_rv32i("ADDI", rd=13, rs1=0, imm=0),
        *_l30_bounded_sum_max_loop_body(),
    ]


def l30_drop_loop_zero_inits(program: list[int]) -> list[int] | None:
    """Remove ADDI x12/x13/x16,x0,0 before the bounded max loop when they are identity."""
    if not program or len(program) < 12:
        return None
    new = _l30_bounded_sum_max_loop_body()
    for old in (_l30_max_loop_with_zeros(), _l30_max_loop_x12_x13_zeros()):
        n = len(old)
        for i in range(len(program) - n + 1):
            if program[i : i + n] == old:
                bundled = program[:i] + new + program[i + n :]
                if not (12 <= len(bundled) <= L30_COMPACT_MAX_WORDS) or bundled == list(program):
                    return None
                return bundled
    return None


def l30_l10_bounded_sum_variant(program: list[int]) -> list[int] | None:
    """Sum mem[x1+i] into x12 while x2<16 so L8 addresses do not spin the loop."""
    if not program or len(program) < 12:
        return None
    from mutation_proposer import DirectedMutator

    seq = _l30_bounded_sum_loop_seq()
    if seq[0] in program and seq[-1] in program:
        return None
    bundled = DirectedMutator.insert_seq_before_ebreak(list(program), seq)
    if not (12 <= len(bundled) <= L30_COMPACT_MAX_WORDS) or bundled == list(program):
        return None
    return bundled


def l30_l10_max_in_sum_loop_variant(program: list[int]) -> list[int] | None:
    """Track unsigned max in x13 inside the bounded L9 sum loop. Does not write L30 regs."""
    if not program or len(program) < 12:
        return None
    from mutation_proposer import DirectedMutator

    old = _l30_bounded_sum_loop_seq()
    new = _l30_bounded_sum_max_loop_seq()
    n = len(old)
    for i in range(len(program) - n + 1):
        if program[i : i + n] == old:
            bundled = program[:i] + new + program[i + n :]
            if not (12 <= len(bundled) <= L30_COMPACT_MAX_WORDS) or bundled == list(program):
                return None
            return bundled
    if new[0] in program and new[-1] in program:
        return None
    bundled = DirectedMutator.insert_seq_before_ebreak(list(program), new)
    if not (12 <= len(bundled) <= L30_COMPACT_MAX_WORDS) or bundled == list(program):
        return None
    return bundled


def _program_has_bounded_sum_max_loop(program: list[int]) -> bool:
    body = _l30_bounded_sum_max_loop_body()
    n = len(body)
    if not program or len(program) < n:
        return False
    return any(program[i : i + n] == body for i in range(len(program) - n + 1))


def l10_lift_loop_preserving_semantic_compress(program: list[int]) -> list[int] | None:
    """Semantic-compress only the prefix before an embedded bounded sum+max loop."""
    body = _l30_bounded_sum_max_loop_body()
    n = len(body)
    if not program or len(program) < n + 1:
        return None
    loop_i = None
    for i in range(len(program) - n + 1):
        if list(program[i : i + n]) == body:
            loop_i = i
            break
    if loop_i is None:
        return None
    prefix = list(program[:loop_i])
    if not prefix:
        return None
    suffix = list(program[loop_i:])
    compressed_prefix = semantic_compress(prefix)
    if not compressed_prefix or len(compressed_prefix) >= len(prefix):
        return None
    trial = list(compressed_prefix) + suffix
    if not _program_has_bounded_sum_max_loop(trial):
        return None
    return trial


def l30_l10_lift_loop_prefix_compress_variant(program: list[int]) -> list[int] | None:
    """Iteratively shrink the loop prefix while preserving L10 sum+max semantics."""
    if not _program_has_bounded_sum_max_loop(program):
        return None
    cur = list(program)
    best: list[int] | None = None
    for _ in range(6):
        nxt = l10_lift_loop_preserving_semantic_compress(cur)
        if not nxt or len(nxt) >= len(cur):
            break
        cur = list(nxt)
        best = cur
    if best is None or len(best) >= len(program):
        return None
    if not (12 <= len(best) <= L30_COMPACT_MAX_WORDS):
        return None
    return best


def _normalize_single_trailing_ebreak(program: list[int]) -> list[int]:
    from mutation_proposer import EBREAK

    trial = list(program)
    while trial.count(EBREAK) > 1:
        idx = trial.index(EBREAK)
        if idx < len(trial) - 1:
            trial = trial[:idx] + trial[idx + 1 :]
        else:
            break
    if not trial or trial[-1] != EBREAK:
        trial.append(EBREAK)
    return trial


def l30_l10_sum_max_preloop_insert_variant(program: list[int]) -> list[int] | None:
    """Insert bounded sum+max before the active loop tail (not only the final EBREAK).

    Compact champions often carry a dead mid-body EBREAK before BLT tails. Inserting
    only before the last EBREAK misses the slot that lifts L10 to 1.0 on live champs.
    """
    if not program or len(program) < 12:
        return None
    from mutation_proposer import EBREAK

    if _program_has_bounded_sum_max_loop(program):
        return None
    seq = _l30_bounded_sum_max_loop_seq()
    base_l10 = float(curriculum_verify_mean(list(program), 10))
    if base_l10 >= 0.99 - 1e-6:
        return None

    anchors: list[int] = []
    for i, word in enumerate(program):
        if word == EBREAK and i < len(program) - 1:
            anchors.append(i)
    tail_start = max(0, (len(program) * 2) // 3)
    for pos in range(tail_start, len(program) + 1):
        if pos not in anchors:
            anchors.append(pos)
    if not anchors:
        anchors = [len(program)]

    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    best: list[int] | None = None
    best_rank: tuple[float, int, float, float] | None = None
    for pos in anchors:
        trials = [program[:pos] + seq + program[pos:]]
        if pos < len(program) and int(program[pos]) == EBREAK:
            # Drop dead mid-body EBREAK + duplicate BLT tail after sum+max.
            trials.append(program[:pos] + seq + [EBREAK])
        for raw in trials:
            trial = _normalize_single_trailing_ebreak(raw)
            if not (12 <= len(trial) <= L30_COMPACT_MAX_WORDS) or trial == list(program):
                continue
            l10 = float(curriculum_verify_mean(trial, 10))
            if l10 <= base_l10 + 1e-6:
                continue
            rank = (
                l10,
                -len(trial),
                float(compact_curriculum_side_score(trial)),
                float(compact_l30_promotion_raw_fitness(trial)),
            )
            if best_rank is None or rank > best_rank:
                best = trial
                best_rank = rank
    return best


def _l11_fib_gate_imm(program: list[int]) -> int | None:
    """Immediate on SLTIU x19,x1 used as the L11 fib skip gate, else None."""
    from mutation_proposer import rv32i_decode

    for i, word in enumerate(program[:-1]):
        try:
            decoded = rv32i_decode(word)
            nxt = rv32i_decode(program[i + 1])
        except (TypeError, ValueError, KeyError):
            continue
        if not decoded or not nxt:
            continue
        if (
            decoded.get("name") == "SLTIU"
            and int(decoded.get("rd", -1)) == 19
            and int(decoded.get("rs1", -1)) == 1
            and nxt.get("name") == "BEQ"
            and int(nxt.get("rs1", -1)) == 19
            and int(nxt.get("imm") or 0) == 44
        ):
            return int(decoded.get("imm") or 0)
    return None


def l30_l11_fib_gate_tighten_variant(program: list[int]) -> list[int] | None:
    """Replace SLTIU x19,x1,16 with imm=13 so L30 x1 in {13,14,15} skips F(n)."""
    if _l11_fib_gate_imm(program) != 16:
        return None
    from mutation_proposer import make_rv32i, rv32i_decode

    gate13 = make_rv32i("SLTIU", rd=19, rs1=1, imm=13)
    out: list[int] = []
    replaced = False
    i = 0
    while i < len(program):
        word = program[i]
        try:
            decoded = rv32i_decode(word)
            nxt = rv32i_decode(program[i + 1]) if i + 1 < len(program) else None
        except (TypeError, ValueError, KeyError):
            decoded = None
            nxt = None
        if (
            not replaced
            and decoded
            and nxt
            and decoded.get("name") == "SLTIU"
            and int(decoded.get("rd", -1)) == 19
            and int(decoded.get("rs1", -1)) == 1
            and int(decoded.get("imm") or 0) == 16
            and nxt.get("name") == "BEQ"
            and int(nxt.get("rs1", -1)) == 19
            and int(nxt.get("imm") or 0) == 44
        ):
            out.append(gate13)
            replaced = True
        else:
            out.append(word)
        i += 1
    if not replaced or out == list(program):
        return None
    if not (12 <= len(out) <= L30_COMPACT_MAX_WORDS):
        return None
    return out


def l30_l11_gated_fib_variant(program: list[int]) -> list[int] | None:
    """x14=F(x1) only when x1<13 matching L11 n in 0..12. Temps x15-x19."""
    if not program or len(program) < 12:
        return None
    from mutation_proposer import DirectedMutator, make_rv32i

    seq = [
        make_rv32i("SLTIU", rd=19, rs1=1, imm=13),
        make_rv32i("BEQ", rs1=19, rs2=0, imm=44),
        make_rv32i("ADDI", rd=15, rs1=0, imm=0),
        make_rv32i("ADDI", rd=16, rs1=0, imm=1),
        make_rv32i("ADDI", rd=17, rs1=0, imm=0),
        make_rv32i("BEQ", rs1=1, rs2=0, imm=24),
        make_rv32i("ADD", rd=18, rs1=15, rs2=16),
        make_rv32i("ADDI", rd=15, rs1=16, imm=0),
        make_rv32i("ADDI", rd=16, rs1=18, imm=0),
        make_rv32i("ADDI", rd=17, rs1=17, imm=1),
        make_rv32i("BLT", rs1=17, rs2=1, imm=-16),
        make_rv32i("ADDI", rd=14, rs1=15, imm=0),
    ]
    if seq[-1] in program:
        return None
    bundled = DirectedMutator.insert_seq_before_ebreak(list(program), seq)
    if not (12 <= len(bundled) <= L30_COMPACT_MAX_WORDS) or bundled == list(program):
        return None
    return bundled


def _l15_wide_pred_seq() -> list[int]:
    from mutation_proposer import make_rv32i

    return [
        make_rv32i("SLT", rd=18, rs1=1, rs2=2),
        make_rv32i("BEQ", rs1=18, rs2=0, imm=12),
        make_rv32i("ADDI", rd=18, rs1=0, imm=0),
        make_rv32i("BEQ", rs1=0, rs2=0, imm=8),
        make_rv32i("ADD", rd=18, rs1=1, rs2=2),
    ]


def _l15_short_pred_seq() -> list[int]:
    """x18 = x1+x2 if x2<=x1 else 0. Uses x19 then L16 overwrites it."""
    from mutation_proposer import make_rv32i

    return [
        make_rv32i("ADD", rd=18, rs1=1, rs2=2),
        make_rv32i("SLT", rd=19, rs1=1, rs2=2),
        make_rv32i("BEQ", rs1=19, rs2=0, imm=8),
        make_rv32i("ADDI", rd=18, rs1=0, imm=0),
    ]


def l30_l15_short_variant(program: list[int]) -> list[int] | None:
    """Replace the 5-insn L15 predicate with the 4-insn form. L16 still last."""
    if not program or len(program) < 12:
        return None
    old = _l15_wide_pred_seq()
    new = _l15_short_pred_seq()
    n = len(old)
    for i in range(len(program) - n + 1):
        if program[i : i + n] == old:
            bundled = program[:i] + new + program[i + n :]
            if not (12 <= len(bundled) <= L30_COMPACT_MAX_WORDS) or bundled == list(program):
                return None
            return bundled
    return None


def _l12_l14_wide_seq() -> list[int]:
    from mutation_proposer import make_rv32i

    return [
        make_rv32i("ADD", rd=15, rs1=1, rs2=2),
        make_rv32i("ADDI", rd=16, rs1=1, imm=100),
        make_rv32i("ADD", rd=16, rs1=16, rs2=2),
        make_rv32i("SLLI", rd=17, rs1=1, imm=1),
        make_rv32i("ADD", rd=17, rs1=17, rs2=2),
        make_rv32i("ADDI", rd=17, rs1=17, imm=100),
    ]


def _l12_l14_x9_seq() -> list[int]:
    """L12-L14 via x9=x1+x2: x15=x9, x16=x9+100, x17=x9+x1+100."""
    from mutation_proposer import make_rv32i

    return [
        make_rv32i("ADDI", rd=15, rs1=9, imm=0),
        make_rv32i("ADDI", rd=16, rs1=9, imm=100),
        make_rv32i("ADD", rd=17, rs1=9, rs2=1),
        make_rv32i("ADDI", rd=17, rs1=17, imm=100),
    ]


def l30_l12_l14_x9_fold_variant(program: list[int]) -> list[int] | None:
    """Fold L12-L14 through live x9. Saves two insns; does not write L30 regs."""
    if not program or len(program) < 12:
        return None
    from mutation_proposer import make_rv32i

    if make_rv32i("ADD", rd=9, rs1=1, rs2=2) not in program:
        return None
    old = _l12_l14_wide_seq()
    new = _l12_l14_x9_seq()
    n = len(old)
    for i in range(len(program) - n + 1):
        if program[i : i + n] == old:
            bundled = program[:i] + new + program[i + n :]
            if not (12 <= len(bundled) <= L30_COMPACT_MAX_WORDS) or bundled == list(program):
                return None
            return bundled
    return None


def l30_l12_l16_postfix_variant(program: list[int]) -> list[int] | None:
    """Register oracles L12-L16 (x15-x19). Does not write L30's x20-x31."""
    if not program or len(program) < 12:
        return None
    from mutation_proposer import DirectedMutator, make_rv32i

    has_x9 = make_rv32i("ADD", rd=9, rs1=1, rs2=2) in program
    seq = [
        *(_l12_l14_x9_seq() if has_x9 else _l12_l14_wide_seq()),
        *_l15_short_pred_seq(),
        make_rv32i("SLLI", rd=19, rs1=1, imm=2),
        make_rv32i("SLLI", rd=1, rs1=2, imm=3),
        make_rv32i("ADD", rd=19, rs1=19, rs2=1),
    ]
    if seq[0] in program and seq[-1] in program:
        return None
    bundled = DirectedMutator.insert_seq_before_ebreak(list(program), seq)
    if not (12 <= len(bundled) <= L30_COMPACT_MAX_WORDS) or bundled == list(program):
        return None
    return bundled


def curriculum_verify_mean(program: list[int], level: int, n_seeds: int = 8) -> float:
    """CPU verify mean for one curriculum level. Passes instance memory."""
    if not program:
        return 0.0
    return _curriculum_verify_mean_cached(tuple(int(x) for x in program), int(level), int(n_seeds))


@functools.lru_cache(maxsize=1024)
def _curriculum_verify_mean_cached(program: tuple[int, ...], level: int, n_seeds: int) -> float:
    from gpu_vm_farm import _execute_vm
    from tasks_rv32i import LEVELS

    task = LEVELS[int(level)]
    scores: list[float] = []
    for seed in range(int(n_seeds)):
        inst = task(seed=seed).generate_instance()
        inputs = inst["inputs"]["registers"]
        mem = inst["inputs"].get("memory") or {}
        result = _execute_vm(
            list(program),
            {1: inputs.get(1, 0), 2: inputs.get(2, 0)},
            dict(mem),
            10000,
        )
        scores.append(
            float(
                task(seed=seed).verify(
                    {
                        "registers": result.get("registers") or {},
                        "memory": result.get("memory") or {},
                    },
                    inst,
                )
            )
        )
    return sum(scores) / len(scores) if scores else 0.0


_COMPACT_SIDE_LEVELS = (8, 9, 10, 11, 12, 13, 14, 15, 16)


def compact_curriculum_side_score(program: list[int]) -> float:
    """Mean CPU verify on compact-reachable levels L8-L16."""
    if not program:
        return 0.0
    return sum(curriculum_verify_mean(program, lv) for lv in _COMPACT_SIDE_LEVELS) / float(len(_COMPACT_SIDE_LEVELS))


def compact_curriculum_beats(champion: list[int], candidate: list[int]) -> bool:
    """True if candidate lifts any compact side level by 0.05, or the mean by 0.03."""
    if not champion or not candidate:
        return False
    for level in _COMPACT_SIDE_LEVELS:
        if curriculum_verify_mean(candidate, level) >= curriculum_verify_mean(champion, level) + 0.05:
            return True
    return compact_curriculum_side_score(candidate) >= compact_curriculum_side_score(champion) + 0.03


def l30_order_sensitive_variant(program: list[int]) -> list[int] | None:
    """Replace one-sided SUBs with SLT + BGE so x8/x25/x27/x29 hold off seed 0-7.

    Seeds 0-7 often have x1<=x2 so missing SLT and SUB x1-x2 still score.
    Extra seeds with x1>x2 need SLT x8 and the v1>v2 ADD/abs path.
    """
    if not program or len(program) < 8:
        return None
    from mutation_proposer import EBREAK, find_ebreak_index, make_rv32i

    sub29 = make_rv32i("SUB", rd=29, rs1=2, rs2=1)
    sub25 = make_rv32i("SUB", rd=25, rs1=1, rs2=2)
    sub27 = make_rv32i("SUB", rd=27, rs1=1, rs2=2)
    drop = {sub29, sub25, sub27}
    stripped: list[int] = []
    removed = 0
    for word in program:
        if word in drop:
            removed += 1
            continue
        stripped.append(word)
    if removed < 1:
        return None
    body = [
        make_rv32i("SLT", rd=8, rs1=6, rs2=4),
        make_rv32i("ADD", rd=25, rs1=1, rs2=2),
        make_rv32i("ADD", rd=27, rs1=1, rs2=2),
        make_rv32i("SUB", rd=29, rs1=1, rs2=2),
        make_rv32i("BGE", rs1=1, rs2=2, imm=16),
        make_rv32i("SUB", rd=25, rs1=1, rs2=2),
        make_rv32i("SUB", rd=27, rs1=1, rs2=2),
        make_rv32i("SUB", rd=29, rs1=2, rs2=1),
    ]
    idx = find_ebreak_index(stripped)
    if idx is None:
        stripped.extend(body)
        if stripped[-1] != EBREAK:
            stripped.append(EBREAK)
    else:
        stripped = stripped[:idx] + body + stripped[idx:]
    if not (12 <= len(stripped) <= L30_COMPACT_MAX_WORDS):
        return None
    return stripped


def heldout_l30_verify_mean(program: list[int], n_seeds: int = 32) -> float:
    """CPU L30 verify mean on seeds 0..n-1. Does not use GPU batches."""
    if not program:
        return 0.0
    return _heldout_l30_verify_mean_cached(tuple(int(x) for x in program), int(n_seeds))


@functools.lru_cache(maxsize=1024)
def _heldout_l30_verify_mean_cached(program: tuple[int, ...], n_seeds: int) -> float:
    disk_key = f"{_vm_score_program_key(program)}|{int(n_seeds)}"
    cached = _vm_score_disk_get("heldout_l30", disk_key)
    if cached is not None:
        return cached
    from gpu_vm_farm import _execute_vm
    from tasks_rv32i import LEVELS

    level = LEVELS[30]
    scores: list[float] = []
    for seed in range(int(n_seeds)):
        inst = level(seed=seed).generate_instance()
        inputs = inst["inputs"]["registers"]
        result = _execute_vm(
            list(program),
            {1: inputs.get(1, 0), 2: inputs.get(2, 0)},
            {},
            10000,
        )
        regs = result.get("registers") or {}
        scores.append(float(level(seed=seed).verify({"registers": regs, "memory": {}}, inst)))
    score = sum(scores) / len(scores) if scores else 0.0
    _vm_score_disk_put("heldout_l30", disk_key, score)
    return score


_eq_refine_alt_probe_cache: dict[tuple, tuple[tuple[int, ...], ...]] = {}


def _eq_refine_alt_probe_cache_key(
    program: list[int],
    n_seeds: int,
    refine_budget: int | None = None,
) -> tuple[float, int, int]:
    from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

    pr = float(compact_l30_promotion_raw_fitness(list(program)))
    # Omit refine_budget: hot/cold compression paths oscillate budget (24 vs 96)
    # but produce the same seed programs; reuse avoids 500s rebuild spikes.
    return (round(pr, 4), len(program), int(n_seeds))


def select_terminal_probes_for_batch(
    program: list[int],
    level: int,
    rng,
    n_probe: int,
    *,
    trace: dict | None = None,
    task_instance: dict | None = None,
    pop_size: int | None = None,
    archive_parents: list[list[int]] | None = None,
) -> list[list[int]]:
    """Pick terminal probe programs with no extra VM eval.

    The generation batch already evaluates every challenger. Pre-scoring
    probes with ``_evaluate`` duplicated GPU work and inflated cycle time.
    Prefer shorter unique programs (pinned L30 length-efficiency path).
    Cap slots at 1/4 of the population so mutation stays the majority.
    """
    if n_probe <= 0 or not program:
        return []
    capability_stalled = False
    at_floor = terminal_efficiency_at_scan_floor(len(program))
    if at_floor:
        _reset_probe_gate_repair_budget()
        _set_scan_floor_probe_rank_champion(list(program))
    n_take = int(n_probe)
    pool_limit = n_take
    if at_floor:
        pool_limit = max(
            n_take,
            int(os.environ.get("EE_SCAN_FLOOR_PROBE_POOL", max(n_take * 4, 32))),
        )
    if pop_size:
        probe_cap = max(2, int(pop_size) // 4)
        if at_floor:
            probe_cap = max(
                probe_cap,
                max(2, int(int(pop_size) * terminal_probe_escalation_ratio())),
            )
        n_take = min(n_take, probe_cap)
    variants: list[list[int]] = []
    from evolution.champion_bloat_guard import level_word_cap

    word_cap = level_word_cap(int(level))
    if not at_floor:
        span = terminal_efficiency_drop_span(len(program), word_cap)
        variants.extend(
            terminal_efficiency_variants(
                program,
                rng,
                max_evals=terminal_efficiency_eval_budget(),
                max_drop_span=span,
            )
        )
    if int(level) >= 29 and _l11_fib_gate_imm(program) != 13 and not at_floor:
        # Already-gated L30 champions keep L8–L16; skip grow fills that
        # re-eval the same postfix every generation.
        variants.extend(l30_easy_fill_variants(program))
    compression_hot = False
    if at_floor:
        champ_n_floor = len(program)
        capability_stalled = _scan_floor_probe_stall_mode(champ_n_floor)
        refine_budget = max(
            64,
            int(n_take) * 2,
            terminal_efficiency_eval_budget(),
        )
        if capability_stalled:
            compression_hot = scan_floor_compression_hot(program, rng)
            if _skip_empty_compression_rescan(tuple(int(x) for x in program), champ_n_floor):
                # Recorded zero-hit compression rescan: avoid redundant exhaustive
                # wide-imm / archive sweeps on the same champion bytes.
                compression_hot = True
            if compression_hot:
                refine_budget = max(int(n_take) * 2, 24)
                span = terminal_efficiency_drop_span(champ_n_floor, word_cap)
                variants.extend(
                    terminal_efficiency_variants(
                        program,
                        rng,
                        max_evals=max(
                            12,
                            int(os.environ.get("EE_SCAN_FLOOR_COMPRESS_PROBE_BUDGET", "16")),
                        ),
                        max_drop_span=min(2, span),
                    )
                )
            else:
                refine_budget = max(
                    96,
                    int(n_take) * 4,
                    terminal_efficiency_eval_budget() + 16,
                )
        variants.extend(terminal_equal_length_refine_variants(program, int(level), rng, max_evals=refine_budget))
        if capability_stalled:
            try:
                from evolution.mutation_helpers import load_scan_floor_eq_refine_seeds

                alt_seeds = load_scan_floor_eq_refine_seeds(
                    list(program),
                    max_seeds=max(
                        2,
                        int(os.environ.get("EE_SCAN_FLOOR_EQ_REFINE_ALT_SEEDS", "3")),
                    ),
                )
                if alt_seeds:
                    per_seed = max(
                        16,
                        refine_budget // max(1, len(alt_seeds)),
                    )
                    alt_key = _eq_refine_alt_probe_cache_key(list(program), len(alt_seeds), int(refine_budget))
                    alt_cached = _eq_refine_alt_probe_cache.get(alt_key)
                    if alt_cached is not None:
                        variants.extend([list(v) for v in alt_cached])
                    else:
                        alt_variants: list[list[int]] = []
                        for seed_prog in alt_seeds:
                            if tuple(seed_prog) == tuple(program):
                                continue
                            alt_variants.extend(
                                terminal_equal_length_refine_variants(
                                    list(seed_prog),
                                    int(level),
                                    rng,
                                    max_evals=per_seed,
                                )
                            )
                        if alt_variants:
                            _eq_refine_alt_probe_cache[alt_key] = tuple(
                                tuple(int(w) for w in prog) for prog in alt_variants
                            )
                            if len(_eq_refine_alt_probe_cache) > 32:
                                _eq_refine_alt_probe_cache.pop(next(iter(_eq_refine_alt_probe_cache)))
                        variants.extend(alt_variants)
            except ImportError:
                pass
    else:
        variants.extend(terminal_verify_probe_variants(program, int(level), rng, n_take))
    if trace is not None and task_instance is not None and _l11_fib_gate_imm(program) != 13 and not at_floor:
        variants.extend(terminal_register_targeted_variants(program, int(level), trace, task_instance, rng, n_take))
    champ = tuple(program)
    seen: set[tuple[int, ...]] = {champ}
    uniq: list[list[int]] = []
    for variant in variants:
        if not variant:
            continue
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(list(variant))
    champ_n = len(program)
    if at_floor:
        gate = get_scan_floor_gate(program)
        same_len = [u for u in uniq if len(u) == champ_n]
        qualified = [u for u in same_len if gate.qualifies(u)]
        qualified.sort(key=_scan_floor_probe_rank_key, reverse=True)
        sweep_budget = max(int(n_take), 64)
        if capability_stalled and compression_hot:
            sweep_budget = max(int(n_take), 24)
        elif capability_stalled and not compression_hot:
            sweep_budget = max(96, int(n_take) * 3)
        sweep_offset = int(terminal_cycles_at_level()) * sweep_budget
        wide_imm = terminal_wide_imm_sweep_probe_variants(
            program,
            max_evals=sweep_budget,
            start_offset=sweep_offset,
            gate=gate,
        )
        if len(wide_imm) < sweep_budget:
            prog_key = struct.pack("<" + "I" * len(program), *program)
            wide_seed = int(zlib.adler32(prog_key)) & 0x7FFFFFFF
            wide_imm.extend(
                terminal_wide_imm_perturb_probe_variants(
                    program,
                    _random.Random(wide_seed),
                    max_evals=sweep_budget - len(wide_imm),
                    max_attempts=2000,
                    gate=gate,
                )
            )
        archive_budget = max(6, int(n_take) // 3) if archive_parents else 0
        if capability_stalled and not compression_hot and archive_parents:
            archive_budget = max(16, int(n_take) * 2)
        archive_offset = int(terminal_cycles_at_level()) * archive_budget
        archive_probes: list[list[int]] = []
        archive_cross: list[list[int]] = []
        archive_segment: list[list[int]] = []
        archive_intercross: list[list[int]] = []
        archive_consensus: list[list[int]] = []
        archive_word_diff: list[list[int]] = []
        archive_dual_diff: list[list[int]] = []
        archive_tri_diff: list[list[int]] = []
        archive_long_window: list[list[int]] = []
        archive_long_compress: list[list[int]] = []
        material_compress: list[list[int]] = []
        material_window: list[list[int]] = []
        material_splice: list[list[int]] = []
        archive_mutate: list[list[int]] = []
        long_parents: list[list[int]] = []
        material_stalled = capability_stalled
        if not material_stalled:
            try:
                from evolution.epi_momentum_governor import scan_floor_length_prom_stalled
                from evolution.mutation_helpers import _evolution_project_root

                material_stalled = scan_floor_length_prom_stalled(
                    _evolution_project_root(),
                    len(program),
                )
            except (ImportError, TypeError, ValueError, OSError):
                material_stalled = False
        if material_stalled:
            try:
                from evolution.champion_rollback import collect_growth_material_compress_parents

                material_parents = collect_growth_material_compress_parents(
                    list(program),
                    max_parents=max(8, int(n_take) // 2),
                )
                if material_parents:
                    mat_budget = max(
                        4,
                        int(os.environ.get("EE_SCAN_FLOOR_MATERIAL_COMPRESS_BUDGET", "6")),
                    )
                    mat_budget = max(
                        mat_budget,
                        int(os.environ.get("EE_SCAN_FLOOR_MATERIAL_COMPRESS_BUDGET_STALLED", "10")),
                    )
                    material_compress = terminal_long_archive_compress_probe_variants(
                        program,
                        material_parents,
                        max_evals=mat_budget,
                        gate=gate,
                    )
                    material_window = terminal_long_parent_window_graft_probe_variants(
                        program,
                        material_parents,
                        max_evals=mat_budget,
                        start_offset=int(terminal_cycles_at_level()) * mat_budget,
                        gate=gate,
                    )
                    material_splice = terminal_growth_prefix_tail_splice_probe_variants(
                        program,
                        material_parents,
                        max_evals=mat_budget,
                        start_offset=int(terminal_cycles_at_level()) * mat_budget + 1,
                        gate=gate,
                    )
            except (ImportError, OSError, TypeError, ValueError):
                pass
        if archive_budget and archive_parents:
            archive_probes = terminal_archive_program_probe_variants(
                program,
                archive_parents,
                max_evals=archive_budget,
                start_offset=archive_offset,
                gate=gate,
            )
            cross_parents = list(archive_parents)
            if qualified:
                cross_parents.extend(qualified[: max(4, archive_budget)])
            archive_cross = terminal_deterministic_archive_crossover_probe_variants(
                program,
                cross_parents,
                max_evals=archive_budget,
                start_offset=archive_offset,
                gate=gate,
            )
            archive_segment = terminal_segment_archive_crossover_probe_variants(
                program,
                archive_parents,
                max_evals=archive_budget,
                start_offset=archive_offset + 1,
                gate=gate,
            )
            archive_intercross = terminal_archive_intercross_probe_variants(
                program,
                archive_parents,
                max_evals=archive_budget,
                start_offset=archive_offset + 3,
                gate=gate,
            )
            archive_consensus = terminal_archive_consensus_graft_probe_variants(
                program,
                archive_parents,
                max_evals=archive_budget,
                start_offset=archive_offset + 4,
                gate=gate,
            )
            archive_word_diff = terminal_archive_word_diff_probe_variants(
                program,
                archive_parents,
                max_evals=archive_budget,
                start_offset=archive_offset + 2,
                gate=gate,
            )
            archive_dual_diff: list[list[int]] = []
            archive_tri_diff: list[list[int]] = []
            if capability_stalled:
                archive_dual_diff = terminal_dual_word_diff_probe_variants(
                    program,
                    archive_parents,
                    max_evals=archive_budget,
                    start_offset=archive_offset + 5,
                    gate=gate,
                )
                archive_tri_diff = terminal_tri_word_diff_probe_variants(
                    program,
                    archive_parents,
                    max_evals=max(archive_budget, 12),
                    start_offset=archive_offset + 7,
                    gate=gate,
                )
                try:
                    from evolution.champion_rollback import collect_jsonl_long_archive_parents
                    from evolution.mutation_helpers import _evolution_project_root

                    results = os.path.join(_evolution_project_root(), "multi_cycle_results.jsonl")
                    long_parents = collect_jsonl_long_archive_parents(
                        results,
                        list(program),
                        int(level),
                        max_parents=max(12, archive_budget),
                        word_tolerance=max(
                            6,
                            int(os.environ.get("EE_SCAN_FLOOR_ARCHIVE_WORD_TOL_EXHAUSTED", "12")),
                        ),
                    )
                    if long_parents:
                        archive_long_window = terminal_long_parent_window_graft_probe_variants(
                            program,
                            long_parents,
                            max_evals=archive_budget,
                            start_offset=archive_offset + 6,
                            gate=gate,
                        )
                        archive_long_compress = terminal_long_archive_compress_probe_variants(
                            program,
                            long_parents,
                            max_evals=max(8, archive_budget),
                            gate=gate,
                        )
                except (ImportError, OSError, TypeError, ValueError):
                    pass
            archive_mutate = terminal_archive_mutate_probe_variants(
                program,
                archive_parents,
                max_evals=archive_budget,
                start_offset=archive_offset,
                gate=gate,
            )
        if capability_stalled and compression_hot:
            wide_cap = max(12, min(len(wide_imm), int(n_take) * 2))
            wide_imm = wide_imm[:wide_cap]
        compression_probes: list[list[int]] = []
        if capability_stalled:
            for variant in uniq:
                if len(variant) >= champ_n:
                    continue
                if scan_floor_compression_probe_qualifies(program, variant):
                    compression_probes.append(list(variant))
            compression_probes.sort(key=len)
        picked: list[list[int]] = []
        seen_pick: set[tuple[int, ...]] = {champ}
        cap = pool_limit
        archive_first = (
            material_compress
            + material_splice
            + material_window
            + archive_long_compress
            + archive_long_window
            + archive_consensus
            + archive_dual_diff
            + archive_tri_diff
            + archive_intercross
            + archive_segment
            + archive_cross
            + archive_word_diff
            + archive_probes
            + archive_mutate
        )
        capability_first: list[list[int]] = []
        if capability_stalled:
            cap_reserve = max(
                8,
                int(os.environ.get("EE_SCAN_FLOOR_CAPABILITY_PROBE_RESERVE", "16")),
            )
            champ_t = tuple(program)
            prom_candidates = [list(variant) for variant in qualified if tuple(variant) != champ_t]
            capability_first = _rank_scan_floor_picked_probes(
                program,
                prom_candidates,
                budget=cap_reserve,
            )[:cap_reserve]
        if capability_stalled and compression_hot:
            probe_streams = (
                compression_probes,
                capability_first,
                archive_first,
                wide_imm,
            )
        elif capability_stalled:
            probe_streams = (capability_first, archive_first, wide_imm)
        elif compression_hot:
            probe_streams = (compression_probes, archive_first, wide_imm)
        else:
            probe_streams = (wide_imm, archive_first)
        for stream in probe_streams:
            for variant in stream:
                if len(picked) >= cap:
                    break
                key = tuple(variant)
                if key in seen_pick:
                    continue
                seen_pick.add(key)
                picked.append(list(variant))
        if not capability_stalled:
            for variant in qualified:
                if len(picked) >= cap:
                    break
                key = tuple(variant)
                if key in seen_pick:
                    continue
                seen_pick.add(key)
                picked.append(list(variant))
        picked = _rank_scan_floor_picked_probes(program, picked)
        _vm_score_disk_flush()
        _clear_scan_floor_probe_rank_champion()
        return picked[:pool_limit]
    grows = [u for u in uniq if champ_n < len(u) <= L30_COMPACT_MAX_WORDS]
    shorts = [u for u in uniq if 12 <= len(u) < champ_n]
    other = [u for u in uniq if u not in grows and u not in shorts]
    shorts.sort(key=len)
    n_grow = min(len(grows), max(1, n_take // 3)) if grows else 0
    ordered = grows[:n_grow] + shorts + other
    return ordered[:n_take]


def terminal_register_targeted_variants(
    program: list[int],
    level: int,
    trace: dict,
    task_instance: dict,
    rng,
    count: int,
    *,
    score_fn=None,
    min_verify: float | None = None,
    fitness_fn=None,
    min_fitness: float | None = None,
) -> list[list[int]]:
    """Probe variants that graft reference bundles for failing verify registers."""
    from evolution.register_repair import (
        register_targeted_repair_variant,
        trace_guided_immediate_variants,
        wrong_level_registers,
    )

    lvl = int(level)
    if lvl < 29 or count <= 0 or not program:
        return []
    wrong = wrong_level_registers(lvl, trace, task_instance)
    if not wrong:
        return []
    rng.shuffle(wrong)
    raw: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for reg in wrong:
        for max_instr in (10, 8, 6, 12):
            variant = register_targeted_repair_variant(program, lvl, reg, max_instr=max_instr)
            if not variant:
                continue
            key = tuple(variant)
            if key in seen:
                continue
            seen.add(key)
            raw.append(variant)
    for variant in trace_guided_immediate_variants(
        program,
        lvl,
        trace,
        task_instance,
        rng,
        max_evals=max(32, count * 8),
    ):
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        raw.append(variant)
    floor = float(min_verify) if min_verify is not None else None
    candidates = _filter_probe_candidates(
        raw,
        score_fn=score_fn,
        min_verify=floor,
        fitness_fn=fitness_fn,
        min_fitness=min_fitness,
    )
    if not candidates:
        return []
    _sort_probe_candidates(candidates, rng, score_fn=score_fn, fitness_fn=fitness_fn, min_verify=min_verify)
    out: list[list[int]] = []
    while len(out) < count:
        out.append(list(candidates[len(out) % len(candidates)]))
    return out


def verify_led_candidate_eligible(
    champion_verify: float,
    candidate_verify: float,
    margin: float | None = None,
    *,
    champion_certified_verify: float | None = None,
) -> bool:
    """Whether a candidate should be scored for verify-led promotion."""
    m = float(margin if margin is not None else effective_verify_margin())
    if candidate_verify > champion_verify + m:
        return True
    if (
        champion_verify >= _VERIFY_SATURATED - 1e-6
        and candidate_verify >= _VERIFY_SATURATED - 1e-6
        and candidate_verify + 1e-6 >= champion_verify
    ):
        return True
    if (
        champion_certified_verify is not None
        and float(champion_certified_verify) >= _VERIFY_SATURATED - 1e-6
        and candidate_verify + 1e-6 >= champion_verify - m
    ):
        return True
    return False


def candidate_beats_champion_on_verify(
    champion_verify: float,
    candidate_verify: float,
    champion_fitness: float,
    candidate_fitness: float,
    *,
    champion_raw_fitness: float | None = None,
    candidate_raw_fitness: float | None = None,
    champion_words: int | None = None,
    candidate_words: int | None = None,
) -> bool:
    """Allow verify-led champion swap at max depth when fitness is not catastrophic."""
    if not terminal_verify_promote_active():
        return False
    margin = effective_verify_margin()
    if champion_verify >= _VERIFY_SATURATED - 1e-6 and candidate_verify >= _VERIFY_SATURATED - 1e-6:
        if candidate_verify + 1e-6 < champion_verify:
            return False
        if candidate_fitness > champion_fitness + 1e-6:
            return True
        if candidate_fitness + 1e-6 >= champion_fitness:
            if champion_words is not None and candidate_words is not None and candidate_words < champion_words:
                return True
            if (
                champion_raw_fitness is not None
                and candidate_raw_fitness is not None
                and candidate_raw_fitness > champion_raw_fitness + 1e-6
            ):
                return True
        return False
    if candidate_verify <= champion_verify + margin:
        return False
    if champion_fitness <= 0:
        # Zero/negative plateau: verify margin is the gate; block only worse fitness.
        return candidate_fitness >= champion_fitness
    return candidate_fitness >= champion_fitness * TERMINAL_VERIFY_MIN_FITNESS_FRAC


def update_verify_best_candidate(
    engine,
    verify_best: tuple[float, float, list, dict, float] | None,
    candidate_verify: float,
    candidate_fitness: float,
    candidate_program: list[int],
    candidate_trace: dict,
    candidate_raw_score: float,
    *,
    champion_verify: float,
    champion_program: list[int],
    champion_trace: dict,
    champion_fitness: float,
) -> tuple[float, float, list, dict, float] | None:
    """Pick the strongest verify-led challenger, including raw-fitness ties."""
    if not verify_promotion_beats_champion(
        engine,
        champion_program,
        champion_trace,
        candidate_program,
        candidate_trace,
        champion_fitness,
        candidate_fitness,
        champion_verify=champion_verify,
        candidate_verify=candidate_verify,
    ):
        return verify_best
    candidate = (
        float(candidate_verify),
        float(candidate_fitness),
        list(candidate_program),
        candidate_trace,
        float(candidate_raw_score),
    )
    if verify_best is None:
        return candidate
    from evolution.fitness_helpers import promotion_raw_fitness

    if candidate[0] > verify_best[0] + 1e-6:
        return candidate
    if candidate[0] + 1e-6 < verify_best[0]:
        return verify_best
    if candidate[1] > verify_best[1] + 1e-6:
        return candidate
    if candidate[1] + 1e-6 < verify_best[1]:
        return verify_best
    cand_raw = promotion_raw_fitness(engine, candidate_program, candidate_trace, verify_score=candidate[0])
    best_raw = promotion_raw_fitness(engine, verify_best[2], verify_best[3], verify_score=verify_best[0])
    if cand_raw > best_raw + 1e-6:
        return candidate
    if abs(cand_raw - best_raw) <= 1e-6 and len(candidate_program) < len(verify_best[2]):
        return candidate
    return verify_best


def verify_probe_preference_key(
    engine,
    program: list[int],
    trace: dict,
    verify: float,
    fitness: float,
) -> tuple[float, float, float, int]:
    """Rank verify-led probe candidates: verify, fitness, raw capability, shorter."""
    from evolution.fitness_helpers import promotion_raw_fitness

    raw = promotion_raw_fitness(engine, program, trace, verify_score=float(verify))
    return (float(verify), float(fitness), float(raw), -len(program))


def scan_floor_min_raw_gain(champ_words: int) -> float:
    """Minimum promotion_raw delta for equal-length swaps at scan floor."""
    if not terminal_efficiency_at_scan_floor(int(champ_words)):
        return 0.25
    return max(1e-3, float(os.environ.get("EE_SCAN_FLOOR_MIN_RAW_GAIN", "0.01")))


def scan_floor_promotion_capability_qualifies(
    champion_program: list[int],
    candidate_program: list[int],
) -> bool:
    """Held-out + side parity when promoting on promotion_raw at scan floor."""
    if not champion_program or not candidate_program:
        return False
    if not terminal_efficiency_at_scan_floor(len(champion_program)):
        return True
    from evolution.mutation_helpers import scan_floor_growth_completion_qualifies

    champ_n = len(champion_program)
    cand_n = len(candidate_program)
    if cand_n < champ_n:
        return scan_floor_compression_promote_qualifies(list(champion_program), list(candidate_program))
    if cand_n > champ_n:
        return scan_floor_growth_completion_qualifies(list(champion_program), list(candidate_program))
    return get_scan_floor_gate(list(champion_program)).qualifies(list(candidate_program))


def verify_promotion_beats_champion(
    engine,
    champion_program: list[int],
    champion_trace: dict,
    candidate_program: list[int],
    candidate_trace: dict,
    champion_fitness: float,
    candidate_fitness: float,
    *,
    champion_verify: float | None = None,
    candidate_verify: float | None = None,
) -> bool:
    """Promotion compare with raw-fitness and length tie-breaks at saturated verify."""
    from evolution.fitness_helpers import promotion_raw_fitness

    verify_fn = getattr(engine, "_verify_score_from_trace", None)
    stored_verify = float(getattr(engine, "_champion_verify_score", 0.0))
    champ_trace_v = float(
        champion_verify
        if champion_verify is not None
        else (verify_fn(champion_trace) if callable(verify_fn) else stored_verify)
    )
    cand_v = float(
        candidate_verify
        if candidate_verify is not None
        else (verify_fn(candidate_trace) if callable(verify_fn) else 0.0)
    )
    champ_raw = promotion_raw_fitness(engine, champion_program, champion_trace, verify_score=champ_trace_v)
    cand_raw = promotion_raw_fitness(engine, candidate_program, candidate_trace, verify_score=cand_v)
    margin = effective_verify_margin()

    # Certified champions keep saturated _champion_verify_score while the active
    # training-instance trace verify stays partial. Capability gains must still
    # be allowed when trace verify does not regress.
    if (
        stored_verify >= _VERIFY_SATURATED - 1e-6
        and terminal_verify_promote_active()
        and cand_v + 1e-6 >= champ_trace_v - margin
    ):
        if stored_verify >= _VERIFY_SATURATED - 1e-6:
            min_raw_gain = scan_floor_min_raw_gain(len(champion_program))
        else:
            min_raw_gain = 1.0
        if cand_raw + 1e-6 < champ_raw:
            return False
        if cand_raw > champ_raw + min_raw_gain - 1e-6:
            return scan_floor_promotion_capability_qualifies(list(champion_program), list(candidate_program))
        if len(candidate_program) < len(champion_program) and cand_raw + 1e-6 >= champ_raw:
            return scan_floor_promotion_capability_qualifies(list(champion_program), list(candidate_program))
        if candidate_fitness > champion_fitness + min_raw_gain - 1e-6 and cand_raw + 1e-6 >= champ_raw:
            if len(candidate_program) > len(champion_program) and cand_raw <= champ_raw + min_raw_gain - 1e-6:
                return False
            return scan_floor_promotion_capability_qualifies(list(champion_program), list(candidate_program))
        return False

    return candidate_beats_champion_on_verify(
        champ_trace_v,
        cand_v,
        champion_fitness,
        candidate_fitness,
        champion_raw_fitness=champ_raw,
        candidate_raw_fitness=cand_raw,
        champion_words=len(champion_program),
        candidate_words=len(candidate_program),
    )
