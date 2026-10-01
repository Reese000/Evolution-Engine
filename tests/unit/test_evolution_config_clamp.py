"""EvolutionConfig clamp must allow starting at max_level (e.g. L30 when max_level=30)."""

from evolution_engine import EvolutionConfig


def test_clamp_allows_max_curriculum_level():
    cfg = EvolutionConfig(curriculum_level=30, max_level=30)
    cfg._clamp_config()
    assert cfg.curriculum_level == 30


def test_clamp_reduces_above_max_curriculum_level():
    cfg = EvolutionConfig(curriculum_level=35, max_level=30)
    cfg._clamp_config()
    assert cfg.curriculum_level == 30
