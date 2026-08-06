"""Financial discounting, computed one way (issue #224).

Two sites discounted future value before this module existed, and they did not
agree. :func:`~mmm_framework.planning.experiment_value.compute_experiment_net_value`
built per-week weights ``(1+r)^(-w/52)`` and took their mean;
``examples/garden_models/bayesian_clv.py`` applied a single **mid-horizon**
factor ``(1+weekly)^(-H/2)``. Same annual-compounding convention, different
arithmetic, no shared code — the second inconsistency of that kind this repo
shipped (half-life was the first, issue #218). Both now delegate here.

**The default rate is 0.0, and that is a statement, not an omission.** At the
horizons this codebase actually reports (13–52 weeks) a typical corporate
discount rate moves the answer by very little — measured across the repo's own
surfaces, between 0.33% and 2.4% of the undiscounted value at r=10%/yr over
26–52 weeks. A payback horizon or an experiment's net value is dominated by the
carryover kernel and the effect size, not by the time value of money. Passing a
rate is supported and disclosed; assuming one silently is not.

Weeks are the unit throughout because every horizon in ``planning`` is in
weeks; the annual→weekly conversion is exact geometric compounding
``(1+r)^(w/52)``, not a nominal rate divided by 52.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_DISCOUNT_RATE_ANNUAL",
    "discount_weights",
    "mid_horizon_discount_factor",
]

#: The default annual discount rate. Zero — see the module docstring: at the
#: horizons reported here the correction is 0.33%–2.4%, and an assumed nonzero
#: rate would be a silent input to every payback and net-value number.
DEFAULT_DISCOUNT_RATE_ANNUAL = 0.0


def discount_weights(
    horizon_weeks: int,
    *,
    rate_annual: float = DEFAULT_DISCOUNT_RATE_ANNUAL,
    half_life_weeks: float | None = None,
) -> np.ndarray:
    """Per-week present-value (× optional retention) weights over a horizon.

    ``weights[w] = retention(w) · (1+rate_annual)^(-w/52)`` for weeks
    ``w = 0..horizon_weeks-1``, with ``retention(w) = 0.5^(w/half_life_weeks)``
    when a half-life is given and ``1`` otherwise. Week 0 always has weight
    ``retention(0) = 1`` — value landing now is not discounted.

    The retention term exists for the experiment-value use ("how much of what
    the test taught is still true in week w"); a pure financial discount passes
    ``half_life_weeks=None``. They multiply because they answer independent
    questions.

    Parameters
    ----------
    horizon_weeks : int
        Number of weeks; floored at 1.
    rate_annual : float
        Annual discount rate as a fraction (0.10 = 10%/yr), compounded
        geometrically: week ``w`` is worth ``(1+r)^(-w/52)``.
    half_life_weeks : float, optional
        Information half-life in weeks; ``None`` (default) applies no
        retention decay.

    Returns
    -------
    np.ndarray
        ``(horizon_weeks,)`` weights in ``(0, 1]``, week 0 first.
    """
    w = np.arange(max(int(horizon_weeks), 1), dtype=float)
    eps = 1e-9
    ret = (
        np.power(0.5, w / max(float(half_life_weeks), eps))
        if half_life_weeks
        else np.ones_like(w)
    )
    disc = (
        np.power(1.0 + float(rate_annual), -w / 52.0)
        if rate_annual
        else np.ones_like(w)
    )
    return ret * disc


def mid_horizon_discount_factor(
    horizon_weeks: float,
    rate_annual: float = DEFAULT_DISCOUNT_RATE_ANNUAL,
) -> float:
    """The single-point discount at half the horizon: ``(1+r)^(-H/104)``.

    The MVP approximation ``bayesian_clv.py`` uses — a flow spread evenly over
    ``H`` weeks is discounted as if it all arrived at week ``H/2``. Exact for a
    linear discount curve, slightly generous for a convex one; kept as a named
    function so the approximation is visible at the call site rather than
    inlined arithmetic that has to be re-derived to be reviewed.
    """
    if not rate_annual:
        return 1.0
    weekly = (1.0 + float(rate_annual)) ** (1.0 / 52.0) - 1.0
    return float((1.0 + weekly) ** (-float(horizon_weeks) / 2.0))
