"""Near-length jsonl archive parent alignment for scan-floor crossover."""

from evolution.champion_rollback import align_archive_parent_to_champion, collect_jsonl_archive_parents


def test_align_shorter_parent_extends_champion_tail():
    ebreak = 0x00100073
    champion = tuple(range(1, 61)) + (ebreak,)
    parent = tuple(range(1, 60)) + (ebreak,)
    aligned = align_archive_parent_to_champion(parent, champion)
    assert aligned == champion


def test_align_longer_parent_truncates_to_champion():
    ebreak = 0x00100073
    champion = tuple(range(1, 61)) + (ebreak,)
    parent = tuple(range(1, 63)) + (ebreak,)
    aligned = align_archive_parent_to_champion(parent, champion)
    assert aligned is not None
    assert len(aligned) == len(champion)
    assert aligned[-1] == ebreak


def test_align_rejects_mismatched_prefix():
    ebreak = 0x00100073
    champion = tuple(range(1, 61)) + (ebreak,)
    parent = (99,) + tuple(range(2, 60)) + (ebreak,)
    assert align_archive_parent_to_champion(parent, champion) is None


def test_collect_jsonl_archive_parents_word_tolerance(tmp_path):
    import json

    ebreak = 0x00100073
    champion = list(range(1, 61)) + [ebreak]
    near = list(range(1, 60)) + [0x00A28293, ebreak]
    far = list(range(100, 158)) + [ebreak]
    results = tmp_path / "multi_cycle_results.jsonl"
    with open(results, "w", encoding="utf-8") as f:
        for cycle, prog, words in (
            (1, champion, 61),
            (2, near, 60),
            (3, far, 58),
        ):
            f.write(
                json.dumps(
                    {
                        "cycle": cycle,
                        "final_level": 30,
                        "champion_words": words,
                        "champion_program": [f"0x{w:08X}" for w in prog],
                    }
                )
                + "\n"
            )
    same = collect_jsonl_archive_parents(str(results), champion, 30, max_parents=8, word_tolerance=0)
    near_len = collect_jsonl_archive_parents(str(results), champion, 30, max_parents=8, word_tolerance=2)
    assert same == []
    assert len(near_len) == 1
    assert len(near_len[0]) == len(champion)


def test_collect_jsonl_archive_parents_rank_by_promotion_raw(tmp_path, monkeypatch):
    import json

    ebreak = 0x00100073
    champion = list(range(1, 49)) + [ebreak]
    low = list(range(1, 48)) + [0x00A28293, ebreak]
    high = list(range(1, 48)) + [0x00B28293, ebreak]
    prom_by_prog = {tuple(low): 300.0, tuple(high): 340.0}

    def _fake_prom(prog):
        return float(prom_by_prog.get(tuple(prog), 0.0))

    monkeypatch.setattr(
        "evolution.fitness_helpers.compact_l30_promotion_raw_fitness",
        _fake_prom,
    )
    results = tmp_path / "multi_cycle_results.jsonl"
    with open(results, "w", encoding="utf-8") as f:
        for cycle, prog, prom in ((10, low, 300.0), (5, high, 340.0)):
            f.write(
                json.dumps(
                    {
                        "cycle": cycle,
                        "final_level": 30,
                        "champion_words": len(prog),
                        "promotion_raw_fitness": prom,
                        "champion_program": [f"0x{w:08X}" for w in prog],
                    }
                )
                + "\n"
            )
    recent = collect_jsonl_archive_parents(str(results), champion, 30, max_parents=8, word_tolerance=0)
    ranked = collect_jsonl_archive_parents(
        str(results),
        champion,
        30,
        max_parents=8,
        word_tolerance=0,
        rank_by_promotion_raw=True,
    )
    assert len(recent) == 2
    assert tuple(ranked[0]) == tuple(high)
    assert tuple(recent[0]) == tuple(low)
