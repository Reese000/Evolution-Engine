"""Diversity, QD archive, and novelty helpers extracted from EvolutionEngine.

Functions for population diversity measurement, QD archive management,
novelty tracking, circuit breaker logic, and program analysis.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Pure / static helpers (no engine state needed)
# ------------------------------------------------------------------


def count_backward_branches(program: list[int]) -> int:
    """Count backward branches (loop structures) in RV32I bytecode.

    Decodes B-type instructions (BEQ, BNE, BLT, BGE, BLTU, BGEU)
    by extracting the 13-bit signed immediate and checking for
    negative offsets (which indicate backward jumps / loops).
    """
    if not program or len(program) < 2:
        return 0
    count = 0
    for word in program:
        if not isinstance(word, int):
            continue
        opcode = word & 0x7F
        # B-type instructions: opcode = 0x63 (0b1100011)
        if opcode == 0x63:
            # B-type immediate encoding (bits 31,7,30:25,11:8 -> 12-bit signed)
            imm = (
                ((word >> 31) & 0x1) << 12
                | ((word >> 7) & 0x1) << 11
                | ((word >> 25) & 0x3F) << 5
                | ((word >> 8) & 0xF) << 1
            )
            # Sign-extend 13-bit immediate
            if imm & 0x1000:
                imm -= 0x2000
            if imm < 0:
                count += 1
    return count


def compute_population_diversity(
    challengers: list[list[int]],
    opcode_cat_array: list[int],
    num_categories: int,
) -> float:
    """Compute opcode diversity of the challenger population.

    Returns a value in [0.0, 1.0] representing the fraction of
    distinct RV32I opcode categories used across all challengers.
    0.0 = all programs use the same opcode; 1.0 = all categories represented.

    Uses a 128-element list for O(1) opcode-to-category lookup
    without hash computation per word. ~30% faster than dict.get()
    for the hot inner loop.

    R37: Early termination -- once all categories are discovered
    (seen_bits == ALL_CATEGORIES_MASK), stops scanning immediately.
    """
    if not challengers:
        return 0.0

    cat_arr = opcode_cat_array
    seen_bits = 0  # bitset: bit i = category i seen
    all_mask = (1 << num_categories) - 1
    for prog in challengers:
        for word in prog:
            if not isinstance(word, int):
                continue
            cat = cat_arr[word & 0x7F]
            if cat >= 0:
                seen_bits |= 1 << cat
        # R37: Early termination -- all categories found
        if seen_bits == all_mask:
            return 1.0

    # Count set bits via C-implemented int.bit_count()
    return seen_bits.bit_count() / num_categories


def population_category_bits(
    challengers: list[list[int]],
    opcode_cat_array: list[int],
) -> int:
    """Bitset of opcode categories present across the challenger population."""
    seen_bits = 0
    for prog in challengers:
        for word in prog:
            cat = opcode_cat_array[word & 0x7F]
            if cat >= 0:
                seen_bits |= 1 << cat
    return seen_bits


# One representative mnemonic per opcode category (index matches category id).
_CATEGORY_SEED_NAMES: tuple[str, ...] = (
    "LUI",
    "AUIPC",
    "JAL",
    "JALR",
    "BEQ",
    "LW",
    "SW",
    "ADDI",
    "ADD",
    "FENCE",
    "ECALL",
)


def generate_category_seed(category: int, rng) -> list[int]:
    """Minimal valid program that introduces one missing opcode category."""
    from mutation_proposer import EBREAK, make_rv32i

    name = _CATEGORY_SEED_NAMES[category]
    word = make_rv32i(
        name,
        rd=rng.randint(1, 31),
        rs1=rng.randint(0, 31),
        rs2=rng.randint(0, 31),
    )
    if name == "ECALL":
        return [word]
    return [word, EBREAK]


def inject_diversity(
    challengers: list[list[int]],
    n_inject: int,
    rng,
    strategies: list[str],
    opcode_cat_array: list[int],
    num_categories: int,
    generate_seed_fn,
    preserve_indices: set[int] | None = None,
) -> None:
    """Replace the least-diverse challengers with fresh random programs.

    Uses a fast O(N * M) heuristic: for each challenger, count how
    many of its opcodes are shared by the majority of the population.
    Programs whose opcodes are all "common" (appear in > 60% of
    population) are replaced.

    When categories are missing from the population union, injected slots
    target those gaps first so opcode diversity can recover at scan floor.

    Injected programs are generated from scratch (not mutated) to
    maximize opcode diversity. They receive the DIVERSITY_INJECT
    strategy tag for archive tracking.
    """
    if not challengers or n_inject <= 0:
        return

    pop_size = len(challengers)
    n_inject = min(n_inject, pop_size)

    cat_arr = opcode_cat_array

    # Compute opcode frequency across population (O(N * M))
    opcode_count: list[int] = [0] * num_categories
    prog_opcodes_bits: list[int] = []
    for prog in challengers:
        bits = 0
        for word in prog:
            cat = cat_arr[word & 0x7F]
            if cat >= 0:
                bits |= 1 << cat
                opcode_count[cat] += 1
        prog_opcodes_bits.append(bits)

    # R21: Precompute a bitmask of which categories are "common" (above threshold).
    common_threshold = pop_size * 0.6
    common_mask = 0
    for cat in range(num_categories):
        if opcode_count[cat] > common_threshold:
            common_mask |= 1 << cat

    # R37: Merge scoring + top-k selection into a single pass.
    import heapq as _heapq

    protected = set(int(i) for i in (preserve_indices or set()))
    top_heap: list[tuple[float, int]] = []  # min-heap of (score, idx)
    for i, bits in enumerate(prog_opcodes_bits):
        if i in protected:
            continue
        if bits == 0:
            score = 1.0
        else:
            common_count = (bits & common_mask).bit_count()
            total_bits = bits.bit_count()
            score = common_count / total_bits if total_bits > 0 else 1.0
        if len(top_heap) < n_inject:
            _heapq.heappush(top_heap, (score, i))
        elif score > top_heap[0][0]:
            _heapq.heapreplace(top_heap, (score, i))
    # Sort descending by score to inject most-redundant first.
    top_heap.sort(reverse=True)
    top_n = top_heap

    all_mask = (1 << num_categories) - 1
    seen_bits = population_category_bits(challengers, cat_arr)
    missing = [cat for cat in range(num_categories) if not (seen_bits & (1 << cat))]
    rng.shuffle(missing)

    for inject_i, (_, idx) in enumerate(top_n):
        if missing:
            cat = missing[inject_i % len(missing)]
            challengers[idx] = generate_category_seed(cat, rng)
            seen_bits |= population_category_bits([challengers[idx]], cat_arr)
            missing = [c for c in missing if not (seen_bits & (1 << c))]
        else:
            challengers[idx] = generate_seed_fn(rng=rng)
        strategies[idx] = "DIVERSITY_INJECT"


# ------------------------------------------------------------------
# Engine-dependent helpers (take ``engine`` as first parameter)
# ------------------------------------------------------------------


def record_level_circuit_outcome(
    level_circuit_stats: dict,
    level: int,
    had_success: bool,
) -> None:
    """Record a cycle outcome for the given level.

    Args:
        level_circuit_stats: dict mapping level -> {"cycles": int, "successes": int}
        level: Curriculum level being evaluated.
        had_success: True if at least one program passed verify() for this level.
    """
    if level not in level_circuit_stats:
        level_circuit_stats[level] = {"cycles": 0, "successes": 0}
    stats = level_circuit_stats[level]
    stats["cycles"] += 1
    if had_success:
        stats["successes"] += 1


def check_level_circuit_breaker(
    level_circuit_stats: dict,
    disabled_levels: set,
    level: int,
    threshold: int,
    enabled: bool,
    log_fn=None,
) -> bool:
    """Check if a level should be disabled by the circuit breaker.

    Returns True if the level is disabled (0% success after threshold cycles).
    """
    if not enabled:
        return False
    if level in disabled_levels:
        return True
    stats = level_circuit_stats.get(level)
    if stats is None:
        return False
    if stats["cycles"] >= threshold and stats["successes"] == 0:
        disabled_levels.add(level)
        if log_fn:
            log_fn(
                f"[CircuitBreaker] Level {level} DISABLED: "
                f"0% success over {stats['cycles']} cycles "
                f"(threshold={threshold})"
            )
        return True
    return False


def is_level_disabled(disabled_levels: set, level: int) -> bool:
    """Check if a level has been auto-disabled by the circuit breaker."""
    return level in disabled_levels


def record_cycle_level_outcome(engine) -> None:
    """Record the current cycle's outcome for the circuit breaker.

    Called after each generation's evaluation to track per-level success.
    Checks if any challenger (or the champion) passed verify() for the
    current level.
    """
    level = engine.champion_level
    # Check champion verification
    had_success = False
    if engine.champion_trace is not None:
        try:
            from tasks_rv32i import get_task_level

            level_cls = get_task_level(level)
            output = {
                "registers": engine.champion_trace.get("final_regs", {}),
                "memory": engine.champion_trace.get("final_memory", {}),
            }
            score = level_cls(seed=level).verify(output, engine.task_instance)
            if score >= 1.0:
                had_success = True
        except (KeyError, TypeError, ValueError, RuntimeError):
            pass
    # Also check last challengers
    if not had_success and engine._last_results:
        try:
            from tasks_rv32i import get_task_level

            level_cls = get_task_level(level)
            for trace in engine._last_results:
                if not isinstance(trace, dict):
                    continue
                if trace.get("status") != "ok":
                    continue
                output = {
                    "registers": trace.get("final_regs", {}),
                    "memory": trace.get("final_memory", {}),
                }
                score = level_cls(seed=level).verify(output, engine.task_instance)
                if score >= 1.0:
                    had_success = True
                    break
        except (KeyError, TypeError, ValueError, RuntimeError):
            pass
    record_level_circuit_outcome(engine._level_circuit_stats, level, had_success)
    check_level_circuit_breaker(
        engine._level_circuit_stats,
        engine._disabled_levels,
        level,
        engine.config.level_circuit_breaker_threshold,
        engine.config.level_circuit_breaker_enabled,
        engine._log,
    )
