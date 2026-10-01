"""Category-targeted diversity inject recovers missing opcode categories."""

from evolution.diversity_helpers import inject_diversity, population_category_bits
from evolution_engine import _OPCODE_CAT_ARRAY, NUM_OPCODE_CATEGORIES
from mutation_proposer import generate_seed_program


def test_inject_targets_missing_categories_from_champion_monoculture():
    # Champion-like RV32I program with 7/11 categories (same as live 0.636 floor).
    champion = [
        0x00008193,
        0x002084B3,
        0x0020CBB3,
        0x00108233,
        0x00010293,
        0x00210333,
        0x006203B3,
        0x00208A33,
        0x00149E13,
        0x00208F33,
        0x000B8F93,
        0x00208B33,
        0x000B8A93,
        0x00208D33,
        0x002B9C13,
        0x001C4C33,
        0x00432433,
        0x00208CB3,
        0x00208DB3,
        0x40208EB3,
        0x0020D863,
        0x40208CB3,
        0x40208DB3,
        0x40110EB3,
        0x0000A503,
        0x00012783,
        0x00F505B3,
        0x01013993,
        0x02098463,
        0x00281893,
        0x011088B3,
        0x0008A903,
        0x01260633,
        0x0126B9B3,
        0x00098463,
        0x00090693,
        0x00180813,
        0xFE2840E3,
        0x00100813,
        0x00008C63,
        0x01078933,
        0x00080793,
        0x00090813,
        0x00188893,
        0xFE18C8E3,
        0x00078713,
        0x002087B3,
        0x06408813,
        0x00280833,
        0x00109893,
        0x002888B3,
        0x06488893,
        0x00208933,
        0x0020A9B3,
        0x00098463,
        0x00000913,
        0x00209993,
        0x00311093,
        0x001989B3,
        0x00100073,
    ]
    pop = [list(champion) for _ in range(50)]
    strategies = ["DIRECTED"] * 50
    import random

    rng = random.Random(42)
    before = population_category_bits(pop, _OPCODE_CAT_ARRAY)
    assert before.bit_count() < 10

    inject_diversity(
        pop,
        11,
        rng,
        strategies,
        _OPCODE_CAT_ARRAY,
        NUM_OPCODE_CATEGORIES,
        generate_seed_program,
    )
    after = population_category_bits(pop, _OPCODE_CAT_ARRAY)
    assert after.bit_count() >= 10
    assert after.bit_count() > before.bit_count()
    assert strategies.count("DIVERSITY_INJECT") == 11


def test_inject_recovers_all_categories_from_eight_of_eleven():
    import random

    # Eight categories present; three missing (matches live 0.727 plateau).
    base = [0x00008193, 0x002084B3, 0x00210333, 0x0000A503, 0x00100073]
    pop = [list(base) for _ in range(40)]
    strategies = ["PROBE"] * 40
    rng = random.Random(1)
    before = population_category_bits(pop, _OPCODE_CAT_ARRAY)
    assert before.bit_count() == 4
    inject_diversity(
        pop,
        15,
        rng,
        strategies,
        _OPCODE_CAT_ARRAY,
        NUM_OPCODE_CATEGORIES,
        generate_seed_program,
    )
    after = population_category_bits(pop, _OPCODE_CAT_ARRAY)
    assert after.bit_count() == NUM_OPCODE_CATEGORIES


def test_inject_diversity_preserves_protected_slots():
    import random

    champion = [0x00008193, 0x002084B3, 0x00210333, 0x0000A503, 0x00100073]
    protected = champion + [0x11111111, 0x22222222, 0x33333333]
    pop = [list(protected)] + [list(champion) for _ in range(9)]
    strategies = ["GROWTH"] + ["PROBE"] * 9
    rng = random.Random(0)
    inject_diversity(
        pop,
        5,
        rng,
        strategies,
        _OPCODE_CAT_ARRAY,
        NUM_OPCODE_CATEGORIES,
        generate_seed_program,
        preserve_indices={0},
    )
    assert pop[0] == protected
