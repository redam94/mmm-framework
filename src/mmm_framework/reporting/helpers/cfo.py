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


#: Component totals that sum to the model's fitted outcome. Kept explicit (not
#: a wildcard over the dataclass) so a new component added upstream shows up as
#: a widening residual rather than silently changing what "fitted" means.
_FITTED_COMPONENT_FIELDS = (
    "total_intercept",
    "total_trend",
    "total_seasonality",
    "total_media",
    "total_controls",
    "total_geo",
    "total_product",
    "total_events",
    "total_interactions",
    "total_levers",
)


def _fitted_total(model: Any) -> float | None:
    """The model's own fitted total, or ``None`` if it cannot be established.

    The baseline must be the MODELLED non-media outcome. Deriving it as
    ``observed − modelled media`` instead folds the model's residual into a
    number labelled "base demand", which is the defect this exists to avoid.
    """
    try:
        decomp = model.compute_component_decomposition()
    except Exception:  # noqa: BLE001 — fall through to the predictive mean
        decomp = None
    if decomp is not None:
        total = 0.0
        seen = False
        for field in _FITTED_COMPONENT_FIELDS:
            v = getattr(decomp, field, None)
            if v is None:
                continue
            fv = float(np.sum(np.asarray(v, dtype=float)))
            if np.isfinite(fv):
                total += fv
                seen = True
        if seen and np.isfinite(total):
            return float(total)
    try:
        pred = model.predict(return_original_scale=True, random_seed=0)
        mean = np.asarray(pred.y_pred_mean, dtype=float)
        tot = float(np.nansum(mean))
        return tot if np.isfinite(tot) else None
    except Exception:  # noqa: BLE001
        return None


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
    dict with ``kpi_total`` (observed), ``marketing_contribution``
    (mean/lower/upper), ``base_contribution``, ``fitted_total``,
    ``unexplained``, ``baseline_basis``, ``marketing_pct``, ``margin``,
    ``hdi_prob`` and ``spend_cuts`` (a list of ``{cut_pct, revenue_at_risk,
    revenue_lower, revenue_upper, pct_of_kpi, [profit_*]}``).

    ``base_contribution`` is the model's **fitted** non-marketing outcome, not
    "observed minus modelled media" — the latter hides the model's residual
    inside a number a CFO reads as base demand. The residual is reported
    separately as ``unexplained``, so the rollup still reconciles::

        base_contribution + marketing_contribution + unexplained == kpi_total

    ``baseline_basis`` is ``"fitted"`` normally, or ``"observed_minus_media"``
    when no fitted total could be established — in which case ``unexplained``
    is ``None`` and the baseline does absorb the residual, which every renderer
    must say rather than imply otherwise.
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

    # The baseline is the model's FITTED non-marketing outcome, and it is
    # labelled as such. Computing it as `observed - modelled media` would fold
    # every discrepancy between what the model fits and what happened into a
    # number a CFO reads as base demand.
    #
    # The gap that remains is a real quantity with a name and is reported
    # separately, so the rollup still reconciles to the observed KPI:
    #     base_contribution + marketing_contribution + unexplained == kpi_total
    fitted_total = _fitted_total(model)
    if fitted_total is not None:
        base_contribution = fitted_total - m_mean
        unexplained = y_total - fitted_total
        baseline_basis = "fitted"
    else:
        # No fitted total available: fall back to the historical definition, but
        # SAY that the baseline absorbs the residual rather than implying the
        # model accounted for everything.
        base_contribution = y_total - m_mean
        unexplained = None
        baseline_basis = "observed_minus_media"
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
        "fitted_total": fitted_total,
        "unexplained": unexplained,
        "baseline_basis": baseline_basis,
        "marketing_pct": marketing_pct,
        "margin": margin,
        "hdi_prob": hdi_prob,
        "spend_cuts": spend_cuts,
    }
