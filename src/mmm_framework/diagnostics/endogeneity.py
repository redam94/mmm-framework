"""Endogeneity-of-spend diagnostic (issue #110, review finding S3).

The load-bearing MMM assumption is **no unobserved confounders**: that media spend
is (conditionally) exogenous to the outcome. In practice spend often responds to
*expected demand* — planners raise budgets ahead of a strong season — which opens
the back-door ``spend ← demand → sales`` and makes the model over-credit
demand-chasing channels. That assumption is usually implicit; this surfaces and
diagnoses it *before* fitting.

The diagnostic is a Granger-style **lead/lag asymmetry** on differenced series
(differencing removes the shared trend/seasonality that would spuriously correlate
everything): for each channel it compares

* **demand → spend** — does a *past* change in the KPI predict a *current* change
  in spend? (spend chasing demand → endogeneity risk), against
* **spend → demand** — does a *past* change in spend predict a *later* change in
  the KPI? (the intended causal direction).

When demand-leads-spend dominates, the channel is flagged: the plain-language
statement is that spend appears to respond to demand, so its effect can't be
cleanly separated from demand without an experiment. Pure numpy; works pre-fit on
the built (unfitted) model's raw media + KPI arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["endogeneity_diagnostic", "DEFAULT_MAX_LAG", "DEFAULT_CORR_THRESHOLD"]

#: Lead/lag horizon (periods) scanned for the cross-correlation asymmetry.
DEFAULT_MAX_LAG = 8
#: |correlation| a demand→spend lead must exceed (and beat spend→demand) to flag.
DEFAULT_CORR_THRESHOLD = 0.30


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return 0.0
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _max_lead_corr(
    driver: np.ndarray, target: np.ndarray, max_lag: int
) -> tuple[float, int]:
    """Largest correlation where ``driver`` LEADS ``target`` by k∈[1,max_lag]:
    ``corr(driver[:-k], target[k:])``. Returns (best_corr, best_lag)."""
    best, best_k = 0.0, 0
    n = min(driver.size, target.size)
    for k in range(1, max_lag + 1):
        if n - k < 3:
            break
        c = _corr(driver[: n - k], target[k:n])
        if abs(c) > abs(best):
            best, best_k = c, k
    return best, best_k


def endogeneity_diagnostic(
    model: Any,
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
) -> dict[str, Any]:
    """Per-channel spend-endogeneity flags from the (pre-fit) model's raw arrays.

    Uses ``model.X_media_raw`` ``(T, C)``, ``model.y_raw`` ``(T,)`` and
    ``model.channel_names``. For a geo panel the arrays are pooled across geos (a
    conservative approximation — flagged in ``notes``). Returns per-channel rows +
    an overall verdict and the plain-language confounding statement.
    """
    X = np.asarray(model.X_media_raw, dtype=float)
    y = np.asarray(model.y_raw, dtype=float)
    channels = [str(c) for c in getattr(model, "channel_names", [])]
    notes: list[str] = []
    is_geo = (
        bool(getattr(model, "has_geo", False)) and int(getattr(model, "n_geos", 1)) > 1
    )
    if is_geo:
        notes.append(
            "Geo panel — series are pooled across geographies, so this is a "
            "conservative screen rather than a per-geo test."
        )

    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[1] != len(channels):
        return {
            "available": False,
            "reason": "shape_mismatch",
            "channels": [],
            "notes": notes,
        }

    dy = np.diff(y)  # KPI change (demand movement)
    rows: list[dict[str, Any]] = []
    flagged: list[str] = []
    for c, ch in enumerate(channels):
        ds = np.diff(X[:, c])  # spend change
        demand_leads, dl_lag = _max_lead_corr(dy, ds, max_lag)  # demand → spend
        spend_leads, sl_lag = _max_lead_corr(ds, dy, max_lag)  # spend → demand
        endogenous = abs(demand_leads) >= corr_threshold and abs(demand_leads) > abs(
            spend_leads
        )
        if endogenous:
            flagged.append(ch)
        rows.append(
            {
                "channel": ch,
                "demand_leads_spend": demand_leads,
                "demand_lead_lag": dl_lag,
                "spend_leads_demand": spend_leads,
                "spend_lead_lag": sl_lag,
                "endogenous": endogenous,
            }
        )

    if flagged:
        assumption = (
            f"Spend on {', '.join(flagged)} appears to respond to demand "
            "(past KPI movements lead spend), so the model may over-credit it: its "
            "effect can't be cleanly separated from demand without an experiment. "
            "The MMM assumes no unobserved confounders — confirm these channels with "
            "a randomized lift test before trusting their point estimates."
        )
    else:
        assumption = (
            "No channel shows spend clearly chasing demand in this screen. The MMM "
            "still ASSUMES no unobserved confounders — a randomized experiment is "
            "the only way to confirm it; this test only rules out the obvious "
            "demand-chasing pattern."
        )

    return {
        "available": True,
        "channels": rows,
        "flagged": flagged,
        "max_lag": max_lag,
        "corr_threshold": corr_threshold,
        "assumption": assumption,
        "notes": notes,
    }
