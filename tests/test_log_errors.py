"""Tests for the observable error-suppression helper (Phase 2 / H1)."""

from __future__ import annotations

import pytest
from loguru import logger

from mmm_framework.utils import logged_suppress


def test_suppresses_and_logs():
    captured: list[str] = []
    sink_id = logger.add(captured.append, level="DEBUG")
    try:
        with logged_suppress("computing the thing"):
            raise ValueError("boom")
    finally:
        logger.remove(sink_id)
    # The failure is suppressed (no raise) but observable in the log.
    assert any("computing the thing" in m and "boom" in m for m in captured)


def test_clean_block_passes_through():
    with logged_suppress("ok"):
        value = 2 + 2
    assert value == 4


def test_unlisted_exception_propagates():
    # When specific exceptions are given, others are NOT suppressed.
    with pytest.raises(KeyError):
        with logged_suppress("only value errors", ValueError):
            raise KeyError("not suppressed")


def test_keyboardinterrupt_not_suppressed():
    with pytest.raises(KeyboardInterrupt):
        with logged_suppress("default suppresses Exception, not BaseException"):
            raise KeyboardInterrupt()
