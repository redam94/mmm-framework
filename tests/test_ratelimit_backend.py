"""Rate-limiter window logic with an injectable clock (Phase 3 / S2)."""

from __future__ import annotations

from mmm_framework.auth.ratelimit import _FixedWindow


def _fw(clock):
    return _FixedWindow(time_func=lambda: clock[0])


def test_allows_up_to_limit():
    clock = [1000.0]
    fw = _fw(clock)
    for _ in range(3):
        allowed, retry = fw.hit("k", limit=3, window=60)
        assert allowed and retry == 0


def test_blocks_over_limit_with_retry_after():
    clock = [1000.0]
    fw = _fw(clock)
    for _ in range(3):
        fw.hit("k", 3, 60)
    allowed, retry = fw.hit("k", 3, 60)
    assert not allowed
    assert retry >= 1


def test_window_resets_after_elapsed_time():
    clock = [1000.0]
    fw = _fw(clock)
    for _ in range(3):
        fw.hit("k", 3, 60)
    assert fw.hit("k", 3, 60)[0] is False
    clock[0] += 61  # advance past the window
    assert fw.hit("k", 3, 60)[0] is True


def test_keys_are_independent():
    clock = [1000.0]
    fw = _fw(clock)
    for _ in range(3):
        fw.hit("org-a", 3, 60)
    # A different key has its own budget.
    assert fw.hit("org-b", 3, 60)[0] is True
