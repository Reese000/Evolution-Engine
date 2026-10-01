"""Empty novelty archive must not extract descriptors on the fitness hot path."""

from types import SimpleNamespace

from evolution.fitness_helpers import apply_novelty_qd_blending


class _EmptyNovelty:
    size = 0

    def decayed_novelty(self, *_a, **_k):
        raise AssertionError("should skip empty archive")

    def novelty_score(self, *_a, **_k):
        raise AssertionError("should skip empty archive")


def test_empty_novelty_archive_skips_descriptor_work():
    eng = SimpleNamespace(
        _novelty_weight_cache=0.8,
        _current_diversity=0.82,
        novelty_archive=_EmptyNovelty(),
        _extract_behavioral_descriptor=lambda *a, **k: (_ for _ in ()).throw(AssertionError("extract")),
        config=SimpleNamespace(qd_bonus_weight=0.0, eval_budget=1000),
        qd_archive=None,
        generations_stalled=0,
        champion=None,
        champion_trace=None,
        _cached_champion_qd_desc=None,
        _champion_qd_valid=False,
        _qd_coverage=0.0,
    )
    out = apply_novelty_qd_blending(eng, 100.0, [1, 2, 3], {"status": "ok"}, 50.0, None, None)
    assert out == 100.0 * (1.0 - 0.8)
