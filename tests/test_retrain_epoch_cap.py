#!/usr/bin/env python3
"""Tests for retrain epoch-budget capping in run_multi_cycle._effective_max_epochs.

Regression coverage for the bug where an L5+ champion caused the retrain loop
to hard-set 8 epochs, silently ignoring a lower explicit --retrain-max-epochs
cap (making short/fast retrains impossible once the champion reached L5+).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_multi_cycle import _effective_max_epochs


class TestEffectiveMaxEpochsBelowL5:
    """Below L5, the requested cap is used verbatim (no boost)."""

    def test_none_stays_none(self):
        assert _effective_max_epochs(None, best_level=0) is None

    def test_explicit_value_passthrough(self):
        assert _effective_max_epochs(1, best_level=0) == 1
        assert _effective_max_epochs(3, best_level=4) == 3
        assert _effective_max_epochs(50, best_level=4) == 50


class TestEffectiveMaxEpochsAtL5Plus:
    """At L5+, 8 is a floor when unspecified but an explicit cap is an upper bound."""

    def test_none_boosts_to_floor(self):
        assert _effective_max_epochs(None, best_level=5) == 8
        assert _effective_max_epochs(None, best_level=11) == 8

    def test_lower_cap_wins(self):
        # The core fix: a lower explicit cap must not be overridden by the boost.
        assert _effective_max_epochs(1, best_level=5) == 1
        assert _effective_max_epochs(2, best_level=7) == 2

    def test_higher_request_capped_to_floor(self):
        # MetaController/config default (e.g. 50) is capped down to 8 at L5+.
        assert _effective_max_epochs(50, best_level=7) == 8
        assert _effective_max_epochs(12, best_level=5) == 8

    def test_equal_to_floor(self):
        assert _effective_max_epochs(8, best_level=6) == 8

    def test_l5_boundary_is_inclusive(self):
        # L4 -> passthrough, L5 -> boost/cap behavior.
        assert _effective_max_epochs(50, best_level=4) == 50
        assert _effective_max_epochs(50, best_level=5) == 8
