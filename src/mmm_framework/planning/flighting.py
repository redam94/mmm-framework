"""Forward flighting calendar: spread allocated channel budgets across future
periods.

This is the *temporal* half of planning. Where :mod:`mmm_framework.planning.budget`
answers "how much per channel", flighting answers "when, week by week". It is
pure numpy/pandas and takes only the per-channel budgets plus a period count, so
a planner can lay out a forward calendar **before** (or independent of) a fit —
the model contributes the budgets, not the schedule shape.

Patterns:

- ``even`` — flat spend every period.
- ``front_loaded`` / ``back_loaded`` — a linear ramp down/up (``front_load``
  sets the steepness, 0.5 = nearly flat → 1.0 = steep).
- ``pulsed`` — ``pulse_on`` periods on, ``pulse_off`` off, repeating (flighting
  in the classic on/off sense).
- ``seasonal`` — weights proportional to a supplied seasonal index (e.g. the
  model's seasonality), tiled/trimmed to the horizon.
- ``custom`` — an explicit per-period weight vector.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FLIGHTING_PATTERNS = (
    "even",
    "front_loaded",
    "back_loaded",
    "pulsed",
    "seasonal",
    "custom",
)


def _pattern_weights(
    pattern: str,
    n_periods: int,
    *,
    front_load: float = 0.65,
    pulse_on: int = 1,
    pulse_off: int = 1,
    seasonal: list[float] | None = None,
    custom: list[float] | None = None,
) -> np.ndarray:
    """Normalized (sum-to-1) period weights for a flighting pattern."""
    n = int(n_periods)
    if n <= 0:
        raise ValueError("n_periods must be >= 1.")

    if pattern == "even":
        w = np.ones(n)
    elif pattern in ("front_loaded", "back_loaded"):
        hi = float(min(max(front_load, 0.5), 1.0))
        lo = max(1.0 - hi, 0.05)
        ramp = np.linspace(hi, lo, n)
        w = ramp if pattern == "front_loaded" else ramp[::-1]
    elif pattern == "pulsed":
        on = max(int(pulse_on), 1)
        off = max(int(pulse_off), 0)
        cycle = on + off
        w = np.array([1.0 if (i % cycle) < on else 0.0 for i in range(n)])
        if w.sum() == 0:
            w = np.ones(n)
    elif pattern == "seasonal":
        if not seasonal:
            w = np.ones(n)
        else:
            s = np.asarray(seasonal, dtype=float)
            w = np.resize(s, n)
            # shift to keep all weights strictly positive (a season can dip
            # below the mean but never gets a negative dollar amount)
            w = w - w.min() + (abs(w).mean() + 1e-6)
    elif pattern == "custom":
        if not custom or len(custom) != n:
            raise ValueError("custom pattern needs exactly one weight per period.")
        w = np.clip(np.asarray(custom, dtype=float), 0.0, None)
        if w.sum() == 0:
            w = np.ones(n)
    else:
        raise ValueError(
            f"Unknown flighting pattern {pattern!r}; choose from {FLIGHTING_PATTERNS}."
        )

    w = np.clip(w, 0.0, None)
    total = w.sum()
    return w / total if total > 0 else np.full(n, 1.0 / n)


def build_flighting_schedule(
    channel_budgets: dict[str, float],
    n_periods: int,
    *,
    pattern: str = "even",
    period_labels: list[str] | None = None,
    front_load: float = 0.65,
    pulse_on: int = 1,
    pulse_off: int = 1,
    seasonal: list[float] | None = None,
    per_channel_pattern: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Distribute each channel's total budget across ``n_periods`` future periods.

    Returns a JSON-safe dict::

        {
          "pattern", "n_periods", "total_budget",
          "periods": [labels...],
          "channels": [names...],
          "schedule": [{"period": label, "<channel>": spend, ..., "total": tot}],
          "by_channel": {channel: [spend per period]},
        }

    ``per_channel_pattern`` overrides the global ``pattern`` for named channels
    (e.g. an always-on retainer line vs a pulsed promo line).
    """
    channels = list(channel_budgets)
    n = int(n_periods)
    if period_labels and len(period_labels) >= n:
        labels = [str(x) for x in period_labels[:n]]
    else:
        labels = [f"P{i + 1}" for i in range(n)]

    by_channel: dict[str, np.ndarray] = {}
    for ch in channels:
        pat = (per_channel_pattern or {}).get(ch, pattern)
        w = _pattern_weights(
            pat,
            n,
            front_load=front_load,
            pulse_on=pulse_on,
            pulse_off=pulse_off,
            seasonal=seasonal,
        )
        by_channel[ch] = float(channel_budgets[ch]) * w

    schedule: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {"period": labels[i]}
        total = 0.0
        for ch in channels:
            v = float(by_channel[ch][i])
            row[ch] = v
            total += v
        row["total"] = total
        schedule.append(row)

    return {
        "pattern": pattern,
        "n_periods": n,
        "total_budget": float(sum(float(v) for v in channel_budgets.values())),
        "periods": labels,
        "channels": channels,
        "schedule": schedule,
        "by_channel": {ch: [float(x) for x in by_channel[ch]] for ch in channels},
    }
