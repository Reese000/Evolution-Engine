#!/usr/bin/env python3
"""Edge-case tests for parse_val_loss_from_retrain_log.

This parser is the linchpin of the retrain checkpoint gate: if it fails to
extract val_loss, gating silently degrades (the root cause of val_loss being
null for ~1,534 cycles). The happy path is covered in test_val_loss_propagation;
these cover the formats/edge cases the parser's docstring claims to support.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_controller import parse_val_loss_from_retrain_log


def _write_log(text: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(text)
        return f.name


def test_legacy_val_loss_colon_format():
    """The 'Val Loss: X.XXXX' legacy format (per docstring) is parsed."""
    path = _write_log("Epoch 1 complete\nVal Loss: 2.5000\nEpoch 2\nVal Loss: 2.3100\n")
    try:
        assert parse_val_loss_from_retrain_log(path) == 2.31
    finally:
        os.unlink(path)


def test_returns_minimum_across_mixed_formats():
    """Best (minimum) val_loss is returned even across different line formats."""
    path = _write_log(
        "[Train] Epoch 1/8: train=0.77 val=3.5000 (best=3.5000)\n"
        "Val Loss: 2.9000\n"
        "[Train] Saved: checkpoint_cycle_1.pt (val_loss=3.1000, epoch=2)\n"
    )
    try:
        assert parse_val_loss_from_retrain_log(path) == 2.9
    finally:
        os.unlink(path)


def test_integer_only_does_not_match():
    """Patterns require a decimal, so integer-only 'val=2' must not false-match."""
    path = _write_log("val=2 something\nloss is high\n")
    try:
        assert parse_val_loss_from_retrain_log(path) is None
    finally:
        os.unlink(path)


def test_no_val_loss_lines_returns_none():
    """A log with training output but no val_loss returns None."""
    path = _write_log("[Train] Epoch 1/8: train=0.7735\n[Train] starting...\n")
    try:
        assert parse_val_loss_from_retrain_log(path) is None
    finally:
        os.unlink(path)


def test_malformed_lines_do_not_crash():
    """Garbage/binary-ish content is handled gracefully (returns None, no raise)."""
    path = _write_log("\x00\x01 not a log\nval=abc\nVal Loss: xyz\n")
    try:
        assert parse_val_loss_from_retrain_log(path) is None
    finally:
        os.unlink(path)


def test_val_loss_equals_format_not_shadowed_by_val_equals():
    """'val_loss=X' is captured (the 'val=' pattern must not swallow it wrongly)."""
    path = _write_log("[Train] Saved: checkpoint.pt (val_loss=0.1463, epoch=3)\n")
    try:
        assert parse_val_loss_from_retrain_log(path) == 0.1463
    finally:
        os.unlink(path)
