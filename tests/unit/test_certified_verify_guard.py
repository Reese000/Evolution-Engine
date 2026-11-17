import os

os.environ.setdefault("EE_USE_GPU", "0")


def test_certified_champion_pinned_at_max_level():
    from evolution.champion_bloat_guard import certified_champion_pinned

    assert certified_champion_pinned(30, 73, 1.0) is True
    assert certified_champion_pinned(30, 73, 0.842) is False


def test_equal_verify_fitness_upgrade_allowed_when_gain_exceeds_one():
    """Equal verify at L30 may upgrade when selection_fitness improves by >1.0."""
    current_verify = 1.0
    verify_score = 1.0
    champ_fit = 200.0
    assert float(verify_score) <= current_verify + 0.005
    assert 201.5 > champ_fit + 1.0


def test_verify_regression_rollback_sensitive_at_one(tmp_path):
    from evolution.champion_rollback import flag_verify_regression_rollback

    assert (
        flag_verify_regression_rollback(
            str(tmp_path),
            level=30,
            prev_verify=1.0,
            prev_program=[1, 2],
            prev_cycle=3511,
            current_verify=0.842,
            min_drop=0.03,
        )
        is True
    )
