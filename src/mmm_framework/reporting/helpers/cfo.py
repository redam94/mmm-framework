"""CFO one-pager facts — P&L rollup + spend-cut revenue/profit-at-risk (issue #108).

The CMO carries the MMM number into rooms that speak P&L, not posteriors. This
rolls the model up into two defensible business statements:

* **contribution rollup** — total *incremental* marketing contribution vs the base
  (non-marketing) outcome, with a credible interval, honest about how much
  marketing actually moves;
* **spend-cut sensitivity** — "cut marketing X% → this much revenue (and, with a
  margin, profit) at risk", with credible intervals, for a few cut levels.

Both are read straight off the fitted response curves with paired posterior draws
(so the at-risk *delta* carries genuine uncertainty), model-free of any external
assumptions beyond the margin. Pure numpy over ``sample_channel_contributions``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["cfo_facts", "DEFAULT_CUT_LEVELS"]

#: Spend-cut levels a board typically asks about.
DEFAULT_CUT_LEVELS: tuple[float, ...] = (0.10, 0.25, 0.50)


def _eti(draws: np.ndarray, hdi_prob: float) -> tuple[float, float]:
    """Equal-tailed interval (percentile-based, matching compute_hdi_bounds)."""
    lo = float(np.percentile(draws, 100.0 * (1.0 - hdi_prob) / 2.0))
    hi = float(np.percentile(draws, 100.0 * (1.0 + hdi_prob) / 2.0))
    return lo, hi


def cfo_facts(
    model: Any,
    *,
    margin: float | None = None,
    cut_levels: tuple[float, ...] = DEFAULT_CUT_LEVELS,
    max_draws: int = 300,
    hdi_prob: float = 0.90,
    random_seed: int = 0,
) -> dict[str, Any]:
    """The CFO one-pager facts for a fitted MMM.

    Parameters
    ----------
    margin:
        Gross margin in ``[0, 1]`` — when given, each spend-cut entry also carries
        profit-at-risk (= revenue-at-risk × margin). ``None`` → revenue only.
    cut_levels:
        Marketing spend-cut fractions to evaluate (e.g. ``0.10`` = −10%).
    max_draws, hdi_prob, random_seed:
        Posterior thinning, credible-interval mass, and the (shared) seed that
        pairs the baseline and cut draws so the at-risk delta is a true contrast.

    Returns
    -------
    dict with ``kpi_total``, ``marketing_contribution`` (mean/lower/upper),
    ``base_contribution``, ``marketing_pct``, ``margin``, ``hdi_prob`` and
    ``spend_cuts`` (a list of ``{cut_pct, revenue_at_risk, revenue_lower,
    revenue_upper, pct_of_kpi, [profit_*]}``).
    """
    X = np.asarray(model.X_media_raw, dtype=float)
    y_total = float(np.nansum(np.asarray(model.y_raw, dtype=float)))

    # Total incremental marketing contribution (original KPI scale), per draw.
    base = model.sample_channel_contributions(
        X_media=X, max_draws=max_draws, random_seed=random_seed
    )  # (D, obs, C)
    marketing_draws = np.asarray(base).sum(axis=(1, 2))  # (D,)
    m_mean = float(np.mean(marketing_draws))
    m_lo, m_hi = _eti(marketing_draws, hdi_prob)

    base_contribution = y_total - m_mean  # the non-marketing (base) outcome
    marketing_pct = m_mean / y_total if abs(y_total) > 1e-9 else None

    spend_cuts: list[dict[str, Any]] = []
    for cut in cut_levels:
        cut = float(cut)
        scaled = model.sample_channel_contributions(
            X_media=X * (1.0 - cut), max_draws=max_draws, random_seed=random_seed
        )
        scaled_draws = np.asarray(scaled).sum(axis=(1, 2))  # (D,)
        # Revenue lost by cutting spend — paired with the baseline draw.
        at_risk = marketing_draws - scaled_draws  # (D,)
        r_mean = float(np.mean(at_risk))
        r_lo, r_hi = _eti(at_risk, hdi_prob)
        entry: dict[str, Any] = {
            "cut_pct": cut,
            "revenue_at_risk": r_mean,
            "revenue_lower": r_lo,
            "revenue_upper": r_hi,
            "pct_of_kpi": (r_mean / y_total if abs(y_total) > 1e-9 else None),
        }
        if margin is not None:
            entry["profit_at_risk"] = r_mean * margin
            entry["profit_lower"] = r_lo * margin
            entry["profit_upper"] = r_hi * margin
        spend_cuts.append(entry)

    return {
        "kpi_total": y_total,
        "marketing_contribution": {"mean": m_mean, "lower": m_lo, "upper": m_hi},
        "base_contribution": base_contribution,
        "marketing_pct": marketing_pct,
        "margin": margin,
        "hdi_prob": hdi_prob,
        "spend_cuts": spend_cuts,
    }
